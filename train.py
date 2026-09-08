#!/usr/bin/env python3
"""
train.py
========
Training script for Disentangled CycleGAN fMRI motion artefact correction.
"""
import argparse
import contextlib
import csv
import os
import random
import re
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import SequentialLR, ConstantLR, LinearLR
from tqdm import tqdm



from dataset import build_dataloaders, psc_denormalise, TARGET_SPATIAL  # dataset.py
from losses  import (generator_loss, discriminator_loss,        # losses.py
                     LossWeights, ModelOutputs,
                     temporal_consistency_loss, fc_loss,
                     temporal_discriminator_loss, temporal_generator_loss,
                     roi_cycle_consistency_loss)
from atlas_fc import (SchaeferAtlas, load_age_atlases,           # atlas_fc.py
                      age_group_for_subject, DEFAULT_ATLAS_PATHS,
                      SchaeferAtlasCropped, load_age_atlases_cropped, PADDED_SPATIAL)
from models.model import DisentangledCycleGAN
from models.st_model import SpatioTemporalCycleGAN
from models.roi_discriminator import MultiScaleROITemporalDiscriminator
from grade_dataset import (FMRIUnpairedGradeDataset, worker_init_fn,  # grade_dataset.py
                           GRADE_A, GRADES_B_ALL, denormalize_chunk,
                           DEFAULT_CHUNK_METADATA_CSV as GRADE_DEFAULT_CHUNK_CSV,
                           DEFAULT_RUN_STATS_CSV as GRADE_DEFAULT_STATS_CSV)


# fMRI quality metrics
def compute_dvars(x: torch.Tensor) -> float:
    """
    DVARS — RMS of the temporal derivative of the global signal.
    For each timepoint t > 0:
        dvars(t) = sqrt( mean( (x[:,t,...] - x[:,t-1,...])^2 ) )
    Returns the mean DVARS across all timepoints and batch items.

    Lower = less frame-to-frame signal change = fewer motion spikes.

    Args:
        x : (B, T, X, Y, Z)  PSC-normalised chunk, on CPU
    Returns:
        float  mean DVARS
    """
    # Difference between consecutive timepoints: (B, T-1, X, Y, Z)
    diff = x[:, 1:, ...] - x[:, :-1, ...]
    # RMS over spatial dims for each frame: (B, T-1)
    dvars_per_frame = diff.pow(2).mean(dim=(-3, -2, -1)).sqrt()
    return dvars_per_frame.mean().item()


def compute_tsnr(x: torch.Tensor) -> float:
    """
    Temporal SNR — mean signal divided by temporal std, averaged over brain.

    tSNR(voxel) = mean(x, dim=T) / std(x, dim=T)
    Returns mean tSNR over all brain voxels (non-zero mean) and batch items.

    Higher = cleaner signal relative to noise.

    Args:
        x : (B, T, X, Y, Z)  PSC-normalised chunk, on CPU
    Returns:
        float  mean tSNR
    """
    mean = x.mean(dim=1)                        # (B, X, Y, Z)
    std  = x.std(dim=1)                         # (B, X, Y, Z)
    valid = std > 1e-3                          # exclude constant voxels (PSC-zeroed boundaries)
    if valid.sum() == 0:
        return 0.0
    tsnr = mean[valid].abs() / std[valid]
    return tsnr.mean().item()


def compute_global_signal_std(x: torch.Tensor) -> float:
    """
    Global signal stability — std of the mean brain signal over time.

    Global signal(t) = mean over all brain voxels at timepoint t.
    Returns std of that timeseries, averaged over the batch.

    Lower = more stable global signal = less motion-driven fluctuation.

    Args:
        x : (B, T, X, Y, Z)  PSC-normalised chunk, on CPU
    Returns:
        float  mean global signal std
    """
    # Mean over spatial dims at each timepoint: (B, T)
    gs = x.mean(dim=(-3, -2, -1))
    return gs.std(dim=1).mean().item()


def compute_spatial_smoothness(x: torch.Tensor) -> float:
    """
    Spatial smoothness — mean absolute gradient across spatial dims.

    Approximates FWHM by measuring how quickly signal changes between
    adjacent voxels.  Higher value = more spatial variation = less smooth.
    If this increases after correction the model is over-smoothing.

    Computed as mean of |∇x| across X, Y, Z dimensions.

    Args:
        x : (B, T, X, Y, Z)  PSC-normalised chunk, on CPU
    Returns:
        float  mean spatial gradient magnitude
    """
    # Finite differences along each spatial axis
    grad_x = (x[:, :, 1:, :, :] - x[:, :, :-1, :, :]).abs()
    grad_y = (x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).abs()
    grad_z = (x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).abs()
    smoothness = (grad_x.mean() + grad_y.mean() + grad_z.mean()) / 3.0
    return smoothness.item()


def r1_gradient_penalty(
    discriminator: nn.Module,
    real_samples: torch.Tensor,
) -> torch.Tensor:
    real_samples = real_samples.detach().requires_grad_(True)
    d_real = discriminator(real_samples)
    # multi-scale discriminator returns a list — sum all scale outputs
    output = sum(s.sum() for s in d_real) if isinstance(d_real, list) else d_real.sum()
    grad_real = torch.autograd.grad(
        outputs=output,
        inputs=real_samples,
        create_graph=True,
    )[0]
    penalty = grad_real.pow(2).reshape(grad_real.size(0), -1).sum(dim=1).mean()
    return penalty


def compute_fmri_metrics(
    x_input:    torch.Tensor,
    x_corrected: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute all fMRI quality metrics on input and corrected chunks.

    Args:
        x_input     : (B, T, X, Y, Z)  original corrupted chunk (PSC)
        x_corrected : (B, T, X, Y, Z)  model output (PSC)

    Returns:
        Dict with keys:
            dvars_input, dvars_corrected, dvars_improvement
            tsnr_input,  tsnr_corrected,  tsnr_improvement
            gs_std_input, gs_std_corrected, gs_std_improvement
            smoothness_input, smoothness_corrected, smoothness_ratio
    """
    xi = x_input.detach().cpu()
    xc = x_corrected.detach().cpu()

    dvars_in  = compute_dvars(xi)
    dvars_out = compute_dvars(xc)

    tsnr_in   = compute_tsnr(xi)
    tsnr_out  = compute_tsnr(xc)

    gs_in     = compute_global_signal_std(xi)
    gs_out    = compute_global_signal_std(xc)

    sm_in     = compute_spatial_smoothness(xi)
    sm_out    = compute_spatial_smoothness(xc)

    return {
        "dvars_input":        dvars_in,
        "dvars_corrected":    dvars_out,
        "dvars_improvement":  dvars_in - dvars_out,       # positive = better
        "tsnr_input":         tsnr_in,
        "tsnr_corrected":     tsnr_out,
        "tsnr_improvement":   tsnr_out - tsnr_in,         # positive = better
        "gs_std_input":       gs_in,
        "gs_std_corrected":   gs_out,
        "gs_std_improvement": gs_in - gs_out,             # positive = better
        "smoothness_input":   sm_in,
        "smoothness_corrected": sm_out,
        "smoothness_ratio":   sm_out / (sm_in + 1e-8),   # <1 fine, >1.1 = over-smooth
    }

# Composite validation score for best-model selection
# Higher = better.  Penalises over-smoothing.
# Each term is normalised by the expected ideal improvement (from QC stats)
# so all three contribute roughly equally (~1.0 each for perfect correction).
def compute_val_score(metrics: Dict[str, float]) -> float:
    score = (metrics["tsnr_improvement"] / 100.0
             + metrics["dvars_improvement"] / 3.0
             + metrics["gs_std_improvement"] / 0.3)

    if metrics["smoothness_ratio"] > 1.1:
        score -= (metrics["smoothness_ratio"] - 1.1) * 2.0

    return score

# Scheduler builder
# Warmup 5 epochs → constant → linear decay to 0 over second half
def build_scheduler(optimiser: Adam,
                    n_epochs:   int,
                    warmup:     int = 5) -> SequentialLR:
    """
    Linear warmup → constant LR → linear decay to zero.

    Phase 1  (epochs 1 – warmup)         : LR ramps from 0 to base_lr
    Phase 2  (epochs warmup – n_epochs/2): constant LR
    Phase 3  (epochs n_epochs/2 – end)   : linear decay to 0

    Args:
        optimiser : Adam optimiser
        n_epochs  : total training epochs
        warmup    : number of warmup epochs

    Returns:
        SequentialLR scheduler (call .step() once per epoch)
    """
    half        = n_epochs // 2
    decay_steps = n_epochs - half - warmup

    s_warmup    = LinearLR(optimiser,
                           start_factor = 0.01,   # fix3: 1e-6 → 0.01 so epoch 1 has usable LR
                           end_factor   = 1.0,
                           total_iters  = warmup)

    s_constant  = ConstantLR(optimiser,
                             factor      = 1.0,
                             total_iters = half)

    s_decay     = LinearLR(optimiser,
                           start_factor = 1.0,
                           end_factor   = 1e-6,
                           total_iters  = max(decay_steps, 1))

    return SequentialLR(optimiser,
                        schedulers  = [s_warmup, s_constant, s_decay],
                        milestones  = [warmup, warmup + half])

# Checkpoint helpers
def save_checkpoint(
    path:       Path,
    epoch:      int,
    model:      nn.Module,
    opt_G:      Adam,
    opt_D:      Adam,
    sched_G,
    sched_D,
    best_score: float,
    args:       argparse.Namespace,
    roi_disc:   Optional[MultiScaleROITemporalDiscriminator] = None,
    opt_D_roi:  Optional[Adam] = None,
) -> None:
    """Save full training state to path."""
    ckpt = {
        "epoch":        epoch,
        "model":        model.state_dict(),
        "opt_G":        opt_G.state_dict(),
        "opt_D":        opt_D.state_dict(),
        "sched_G":      sched_G.state_dict(),
        "sched_D":      sched_D.state_dict(),
        "best_score":   best_score,
        "args":         vars(args),
        "rng_torch":    torch.get_rng_state(),
        "rng_numpy":    np.random.get_state(),
        "rng_python":   random.getstate(),
        "rng_cuda":     torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    if roi_disc is not None:
        ckpt["roi_disc"]  = roi_disc.state_dict()
        ckpt["opt_D_roi"] = opt_D_roi.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(
    path:   Path,
    model:  nn.Module,
    opt_G:  Adam,
    opt_D:  Adam,
    sched_G,
    sched_D,
    device: torch.device,
    verbose: bool = True,
    roi_disc:  Optional[MultiScaleROITemporalDiscriminator] = None,
    opt_D_roi: Optional[Adam] = None,
) -> Tuple[int, float]:
    """
    Load training state from checkpoint.

    Returns:
        (start_epoch, best_score)
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    opt_G.load_state_dict(ckpt["opt_G"])
    opt_D.load_state_dict(ckpt["opt_D"])
    sched_G.load_state_dict(ckpt["sched_G"])
    sched_D.load_state_dict(ckpt["sched_D"])

    if roi_disc is not None:
        if "roi_disc" in ckpt:
            roi_disc.load_state_dict(ckpt["roi_disc"])
            opt_D_roi.load_state_dict(ckpt["opt_D_roi"])
        elif verbose:
            print("  WARNING: --use_roi_discriminator is on but this checkpoint has no "
                  "roi_disc state (saved before it was enabled) -- roi_disc starts from "
                  "random init instead of resuming.")
    elif "roi_disc" in ckpt and verbose:
        print("  WARNING: checkpoint has roi_disc state but --use_roi_discriminator is "
              "off this run -- that state is being discarded.")

    torch.set_rng_state(ckpt["rng_torch"].cpu().byte())
    np.random.set_state(ckpt["rng_numpy"])
    random.setstate(ckpt["rng_python"])
    if ckpt["rng_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(ckpt["rng_cuda"].cpu().byte())

    if verbose:
        print(f"  Resumed from epoch {ckpt['epoch']}  best_score={ckpt['best_score']:.4f}")
    return ckpt["epoch"] + 1, ckpt["best_score"]

# CSV logger
class CSVLogger:
    """Appends one row per call to a CSV file. Creates file + header on init."""

    def __init__(self, path: Path, fieldnames: list):
        self.path       = path
        self.fieldnames = fieldnames
        self._exists    = path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: dict) -> None:
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self._exists:
                writer.writeheader()
                self._exists = True
            writer.writerow({k: row.get(k, "") for k in self.fieldnames})


# get epoch weights
def get_epoch_weights(
    base_weights: LossWeights,
    epoch:        int,
    warmup_epochs: int,
) -> LossWeights:
    """Linearly ramp cycle/identity weights from 1.0 to full value."""
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return base_weights
    t = epoch / warmup_epochs
    return LossWeights(
        adv      = base_weights.adv,
        cyc      = 1.0 + (base_weights.cyc - 1.0) * t,
        idt      = 1.0 + (base_weights.idt - 1.0) * t,
        temporal = base_weights.temporal,
        fc       = base_weights.fc,
    )

class ReplayBuffer:
    """
    Stores past generator fakes for discriminator training stability.
    On each push_and_pop call, each sample in the batch has a 50% chance
    of being swapped with a historical fake from the buffer.
    Stored on CPU to save GPU memory.
    """
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.data: list = []

    def push_and_pop(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for i in range(x.size(0)):
            item = x[i].unsqueeze(0).detach().cpu()
            if len(self.data) < self.max_size:
                self.data.append(item.clone())
                out.append(item)
            elif random.random() > 0.5:
                idx = random.randint(0, self.max_size - 1)
                old = self.data[idx].clone()
                self.data[idx] = item.clone()
                out.append(old)
            else:
                out.append(item)
        return torch.cat(out, dim=0)


# ROI-timeseries discriminator (models/roi_discriminator.py) support.
# Operates on raw per-chunk data (T=chunk_size, e.g. 5) -- no sequence
# stitching needed, unlike temporal_consistency_loss / fc_loss above.
def _subject_id_from_path(path: str) -> str:
    """Parse 'ICC103A' out of a filename like 'sub-ICC103A_ses-1_...'."""
    m = re.search(r"sub-([A-Za-z0-9]+)_", os.path.basename(path))
    if not m:
        raise ValueError(f"Could not parse subject_id from path: {path}")
    return m.group(1)


def _extract_roi_ts_batch(
    volumes: torch.Tensor,
    paths,
    atlases: Dict[str, SchaeferAtlas],
) -> torch.Tensor:
    """
    Per-sample ROI-timeseries extraction for a flat (non-sequence) batch.

    Different samples in the same batch can come from different subjects
    (and therefore different age-appropriate Schaefer atlases, unlike
    sequence mode where one manifest row == one subject == one atlas), so
    this resolves the atlas per sample rather than once for the batch.

    Args:
        volumes: (B, T, X, Y, Z) at the atlases' target_spatial resolution
        paths:   length-B sequence of source file paths (e.g.
                 batch["path_A"] / batch["path_B"] from the flat loader)
        atlases: {"2mo": SchaeferAtlas, "9mo": SchaeferAtlas}

    Returns:
        (B, n_rois, T) -- ROI axis as channels, ready for
        MultiScaleROITemporalDiscriminator.
    """
    roi_seqs = []
    for b, path in enumerate(paths):
        subject_id = _subject_id_from_path(path)
        atlas = atlases[age_group_for_subject(subject_id)]
        roi_seqs.append(atlas.extract_roi_timeseries(volumes[b]))  # (T, n_rois)
    roi_batch = torch.stack(roi_seqs, dim=0)   # (B, T, n_rois)
    return roi_batch.permute(0, 2, 1)          # (B, n_rois, T)


def train_one_epoch(
    model:    DisentangledCycleGAN,
    loader,
    opt_G:    Adam,
    opt_D:    Adam,
    weights:  LossWeights,
    device:   torch.device,
    epoch:    int,
    max_grad_norm:     float = 10.0,
    d_update_every:    int   = 1,
    label_smooth_real: float = 1.0,
    label_smooth_fake: float = 0.0,
    replay_buffer_a:   Optional['ReplayBuffer'] = None,
    replay_buffer_b:   Optional['ReplayBuffer'] = None,
    r1_weight:         float = 0.0,
    r1_every:          int   = 8,
    use_sequences:     bool  = False,
    atlases:           Optional[Dict[str, SchaeferAtlas]] = None,
    max_batches:       Optional[int] = None,
    fc_mask_strategy:  str   = "threshold",
    fc_threshold:      float = 0.3,
    fc_top_k:          Optional[int] = None,
    fc_percentile:     Optional[float] = None,
    show_progress:     bool  = True,
    is_ddp:            bool  = False,  # fix1: manual D grad all-reduce when multi-GPU
    world_size:        int   = 1,
    roi_disc:          Optional[MultiScaleROITemporalDiscriminator] = None,
    opt_D_roi:         Optional[Adam] = None,
    lambda_roi:        float = 0.5,
    w_roi_adv:         float = 1.0,
    w_roi_cycle:       float = 0.0,
) -> Dict[str, float]:
    """
    Run one full training epoch.

    Returns dict of mean losses over the epoch.
    """
    model.train()
    raw = model.module if isinstance(model, DDP) else model  # unwrap for param/submodule access

    # Accumulators
    acc = {k: 0.0 for k in
           ["G_adv", "G_cyc", "G_idt", "G_total",
            "G_temporal", "G_fc", "G_fc_n_retained", "G_fc_retained_frac",
            "D_A", "D_B", "D_total", "D_r1", "grad_norm_G",
            "score_real_a", "score_fake_a",
            "score_real_b", "score_fake_b",
            "G_roi_adv", "D_roi_total", "G_roi_cycle"]}
    n_batches = 0
    n_d_updates = 0
    n_r1_updates = 0
    n_fc_updates = 0
    n_roi_updates = 0

    # Advance A queue at the start of each epoch
    # FMRIUnpairedGradeDataset (grade_dataset.py) uses set_epoch(epoch) (must be seeded by the
    # real epoch number for reproducible resumes); the flat/sequence datasets in dataset.py use
    # the no-arg on_epoch_start().
    if hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(epoch)
    else:
        loader.dataset.on_epoch_start()

    pbar = tqdm(loader,
                desc=f"Epoch {epoch:03d} [train]",
                leave=False,
                dynamic_ncols=True,
                disable=not show_progress)

    last_d_a = 0.0
    last_d_b = 0.0

    for i, batch in enumerate(pbar):
        if max_batches is not None and i >= max_batches:
            break
        # grade_dataset.py's FMRIUnpairedGradeDataset uses "A_paths"/"B_paths"; the flat
        # dataset.py loader uses "path_A"/"path_B" -- same content, different key names.
        path_a_key = "A_paths" if "A_paths" in batch else "path_A"
        path_b_key = "B_paths" if "B_paths" in batch else "path_B"
        if use_sequences:
            assert batch["A"].shape[0] == 1, (  # fix5: squeeze(0) assumes batch_size=1
                f"Sequence mode requires DataLoader batch_size=1, got {batch['A'].shape[0]}. "
                "Set --batch_size 1 or let build_dataloaders enforce it."
            )
            x_a = batch["A"].squeeze(0).to(device)   # (S, T, X, Y, Z)
            x_b = batch["B"].squeeze(0).to(device)   # (S, T, X, Y, Z)
            atlas = atlases[age_group_for_subject(batch["subject_id"][0])] \
                if atlases is not None else None
        else:
            x_a = batch["A"].to(device)   # (B, T, X, Y, Z)  corrupted
            x_b = batch["B"].to(device)   # (B, T, X, Y, Z)  motion-free

        # Step 1 — Update generators + encoders
        for p in raw.discriminator_parameters():
            p.requires_grad_(False)
        for p in raw.generator_parameters():
            p.requires_grad_(True)
        if roi_disc is not None:
            for p in roi_disc.parameters():
                p.requires_grad_(False)  # same discipline as D_A/D_B above

        out = model(x_a, x_b, detach_fakes_for_D=False)
        g_losses = generator_loss(out, x_a, x_b, weights)

        # Sequence losses: temporal consistency + FC preservation
        total_loss = g_losses["total"]
        if use_sequences:
            if weights.temporal > 0:
                L_temp = temporal_consistency_loss(x_a, out.x_hat_b)
                total_loss = total_loss + weights.temporal * L_temp
                g_losses["temporal"] = L_temp
            if weights.fc > 0 and atlas is not None:
                S, T = x_a.shape[:2]
                x_a_concat = x_a.reshape(S * T, *x_a.shape[2:])
                y_concat   = out.x_hat_b.reshape(S * T, *out.x_hat_b.shape[2:])
                input_roi_ts     = atlas.extract_roi_timeseries(x_a_concat)
                corrected_roi_ts = atlas.extract_roi_timeseries(y_concat)
                fc_stats = {}
                L_fc = fc_loss(
                    input_roi_ts, corrected_roi_ts,
                    mask_strategy=fc_mask_strategy,
                    fc_threshold=fc_threshold,
                    top_k=fc_top_k,
                    percentile=fc_percentile,
                    validate=True,
                    stats=fc_stats,
                )
                total_loss = total_loss + weights.fc * L_fc
                g_losses["fc"] = L_fc
                g_losses["fc_stats"] = fc_stats

        # ROI-timeseries adversarial loss (chunk-level, no stitching --
        # not compatible with sequence mode's (S, T, ...) batch shape)
        if roi_disc is not None and not use_sequences:
            fake_roi = _extract_roi_ts_batch(out.x_hat_b, batch[path_a_key], atlases)  # not detached -- grad must reach G
            g_roi_out = roi_disc(fake_roi)
            g_roi_losses = temporal_generator_loss(g_roi_out, lambda_roi=lambda_roi)
            total_loss = total_loss + w_roi_adv * g_roi_losses["total"]
            g_losses["roi_adv"] = g_roi_losses["total"]

        # ROI-timeseries cycle-consistency loss (chunk-level, no stitching;
        # independent of roi_disc -- no discriminator needed, just L1
        # between the input's and the cyclic reconstruction's ROI dynamics)
        if w_roi_cycle > 0 and atlases is not None and not use_sequences:
            input_roi_ts = _extract_roi_ts_batch(x_a, batch[path_a_key], atlases)
            cycle_roi_ts = _extract_roi_ts_batch(out.x_cycle_a, batch[path_a_key], atlases)
            L_roi_cycle = roi_cycle_consistency_loss(input_roi_ts, cycle_roi_ts)
            total_loss = total_loss + w_roi_cycle * L_roi_cycle
            g_losses["roi_cycle"] = L_roi_cycle

        opt_G.zero_grad()
        total_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            raw.generator_parameters(), max_norm=max_grad_norm
        ).item()

        opt_G.step()

        # Step 2 — Update discriminators (every d_update_every steps)
        if i % d_update_every == 0:
            for p in raw.discriminator_parameters():
                p.requires_grad_(True)
            if roi_disc is not None:
                for p in roi_disc.parameters():
                    p.requires_grad_(True)

            # Query replay buffer — mix current fakes with historical ones
            if replay_buffer_a is not None and replay_buffer_b is not None:
                fake_a = replay_buffer_a.push_and_pop(out.x_hat_a).to(device)
                fake_b = replay_buffer_b.push_and_pop(out.x_hat_b).to(device)
            else:
                fake_a = out.x_hat_a.detach()
                fake_b = out.x_hat_b.detach()

            # Run discriminators on reals + buffered fakes (no full model forward)
            score_real_a = raw.D_A(x_a)
            score_fake_a = raw.D_A(fake_a)
            score_real_b = raw.D_B(x_b)
            score_fake_b = raw.D_B(fake_b)

            # LSGAN loss — handles single tensor or multi-scale list
            def _lsgan_d(scores_real, scores_fake, real_tgt, fake_tgt):
                if isinstance(scores_real, list):
                    n = len(scores_real)
                    return sum(
                        0.5 * (F.mse_loss(sr, torch.full_like(sr, real_tgt)) +
                               F.mse_loss(sf, torch.full_like(sf, fake_tgt)))
                        for sr, sf in zip(scores_real, scores_fake)
                    ) / n
                return 0.5 * (F.mse_loss(scores_real, torch.full_like(scores_real, real_tgt)) +
                              F.mse_loss(scores_fake, torch.full_like(scores_fake, fake_tgt)))

            L_D_A = _lsgan_d(score_real_a, score_fake_a, label_smooth_real, label_smooth_fake)
            L_D_B = _lsgan_d(score_real_b, score_fake_b, label_smooth_real, label_smooth_fake)

            opt_D.zero_grad()
            # no_sync suppresses DDP's AccumulateGrad hooks so async NCCL ops
            # from DDP don't conflict with our explicit manual all_reduce below
            _nosync = model.no_sync() if is_ddp else contextlib.nullcontext()
            with _nosync:
                (L_D_A + L_D_B).backward()
                if r1_weight > 0 and n_d_updates % r1_every == 0:
                    r1_a = r1_gradient_penalty(raw.D_A, x_a)
                    r1_b = r1_gradient_penalty(raw.D_B, x_b)
                    r1_loss = (r1_weight / 2.0) * (r1_a + r1_b) * r1_every
                    r1_loss.backward()
                    acc["D_r1"] += (r1_a + r1_b).item()
                    n_r1_updates += 1

            if is_ddp:  # fix1: explicit D grad sync after hooks suppressed
                for p in raw.discriminator_parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad /= world_size
            opt_D.step()

            last_d_a = L_D_A.item()
            last_d_b = L_D_B.item()
            acc["D_A"]     += last_d_a
            acc["D_B"]     += last_d_b
            acc["D_total"] += (L_D_A + L_D_B).item()
            n_d_updates += 1

            # ROI-timeseries temporal discriminator (chunk-level, no stitching)
            if roi_disc is not None and not use_sequences:
                real_roi = _extract_roi_ts_batch(x_b, batch[path_b_key], atlases)
                fake_roi = _extract_roi_ts_batch(out.x_hat_b.detach(), batch[path_a_key], atlases)

                opt_D_roi.zero_grad()
                real_output = roi_disc(real_roi)
                fake_output = roi_disc(fake_roi)  # already detached above -- stop grad into G
                losses_D_roi = temporal_discriminator_loss(
                    real_output, fake_output, lambda_roi=lambda_roi,
                )
                losses_D_roi["total"].backward()
                if is_ddp:
                    # roi_disc isn't DDP-wrapped, so this backward has no automatic hooks and
                    # no no_sync() to suppress (unlike D_A/D_B, which ARE part of the wrapped
                    # model) -- just average each rank's local gradient directly.
                    for p in roi_disc.parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                            p.grad /= world_size
                opt_D_roi.step()

                acc["D_roi_total"] += losses_D_roi["total"].item()
                n_roi_updates += 1

        # Accumulate
        acc["G_adv"]     += g_losses["adv"].item()
        acc["G_cyc"]     += g_losses["cyc"].item()
        acc["G_idt"]     += g_losses["idt"].item()
        acc["G_total"]   += total_loss.item()
        acc["grad_norm_G"] += grad_norm
        if "temporal" in g_losses:
            acc["G_temporal"] += g_losses["temporal"].item()
        if "fc" in g_losses:
            acc["G_fc"] += g_losses["fc"].item()
            acc["G_fc_n_retained"]    += g_losses["fc_stats"]["n_retained"]
            acc["G_fc_retained_frac"] += g_losses["fc_stats"]["retained_fraction"]
            n_fc_updates += 1
        if "roi_adv" in g_losses:
            acc["G_roi_adv"] += g_losses["roi_adv"].item()
        if "roi_cycle" in g_losses:
            acc["G_roi_cycle"] += g_losses["roi_cycle"].item()
        def _score_mean(s):
            return (s[0] if isinstance(s, list) else s).mean().item()
        acc["score_real_a"] += _score_mean(out.score_real_a)
        acc["score_fake_a"] += _score_mean(out.score_fake_a)
        acc["score_real_b"] += _score_mean(out.score_real_b)
        acc["score_fake_b"] += _score_mean(out.score_fake_b)
        n_batches += 1

        postfix = {
            "G":    f"{total_loss.item():.3f}",
            "cyc":  f"{g_losses['cyc'].item():.3f}",
            "idt":  f"{g_losses['idt'].item():.3f}",
            "D_A":  f"{last_d_a:.3f}",
            "D_B":  f"{last_d_b:.3f}",
            "∇G":   f"{grad_norm:.2f}",
        }
        if "temporal" in g_losses:
            postfix["tc"] = f"{g_losses['temporal'].item():.4f}"
        if "fc" in g_losses:
            postfix["fc"] = f"{g_losses['fc'].item():.4f}"
            postfix["fc_n"] = str(g_losses["fc_stats"]["n_retained"])
        if "roi_adv" in g_losses:
            postfix["roi_adv"] = f"{g_losses['roi_adv'].item():.4f}"
        if "roi_cycle" in g_losses:
            postfix["roi_cyc"] = f"{g_losses['roi_cycle'].item():.4f}"
        pbar.set_postfix(postfix)

    result = {k: v / max(n_batches, 1) for k, v in acc.items()}
    if n_d_updates > 0:
        for k in ["D_A", "D_B", "D_total"]:
            result[k] = acc[k] / n_d_updates
    if n_r1_updates > 0:
        result["D_r1"] = acc["D_r1"] / n_r1_updates
    if n_fc_updates > 0:
        result["G_fc_n_retained"]    = acc["G_fc_n_retained"] / n_fc_updates
        result["G_fc_retained_frac"] = acc["G_fc_retained_frac"] / n_fc_updates
    if n_roi_updates > 0:
        result["D_roi_total"] = acc["D_roi_total"] / n_roi_updates
    return result


# Validation epoch
@torch.no_grad()
def validate(
    model:   DisentangledCycleGAN,
    loader,
    weights: LossWeights,
    device:  torch.device,
    epoch:   int,
    use_sequences: bool = False,
    atlases: Optional[Dict[str, SchaeferAtlas]] = None,
    max_batches: Optional[int] = None,
    fc_mask_strategy: str = "threshold",
    fc_threshold: float = 0.3,
    fc_top_k: Optional[int] = None,
    fc_percentile: Optional[float] = None,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    Run full validation pass.

    Computes:
        - Generator losses (cyc, idt) — no adversarial
        - fMRI quality metrics on x_a vs x_hat_b (corrected output)
        - Temporal consistency and FC metrics (sequence mode only)

    Returns dict of mean metrics over the full val set.
    """
    model.eval()

    loss_acc = {k: 0.0 for k in ["cyc", "idt",
                                  "temporal", "fc",
                                  "fc_n_retained", "fc_retained_frac"]}
    metric_acc = {k: 0.0 for k in [
        "dvars_input", "dvars_corrected", "dvars_improvement",
        "tsnr_input",  "tsnr_corrected",  "tsnr_improvement",
        "gs_std_input", "gs_std_corrected", "gs_std_improvement",
        "smoothness_input", "smoothness_corrected", "smoothness_ratio",
    ]}
    n_batches = 0
    n_fc_updates = 0

    # Grade-dataset-only metrics, ported from pytorch-CycleGAN-and-pix2pix's validate.py:
    # residual magnitude by grade, the Grade-1 entry of that same breakdown doubling as the
    # "identity trend" check, and D_B's score distribution over real-clean/raw-corrupted/
    # corrected. All computed in the same normalized units the network operates in (that repo's
    # residual_by_grade/grade1_identity_trend do the same -- no denormalization step there either).
    residual_l1_by_grade = defaultdict(list)
    disc_scores = {"real_clean": [], "raw_corrupted": [], "corrected": []}

    def _score_mean(s):
        return (s[0] if isinstance(s, list) else s).mean().item()

    pbar = tqdm(loader,
                desc=f"Epoch {epoch:03d} [val]  ",
                leave=False,
                dynamic_ncols=True,
                disable=not show_progress)

    for i, batch in enumerate(pbar):
        if max_batches is not None and i >= max_batches:
            break
        grade_meta_a = None
        if use_sequences:
            assert batch["A"].shape[0] == 1, (  # fix5: mirrors train_one_epoch guard
                f"Sequence mode requires DataLoader batch_size=1, got {batch['A'].shape[0]}."
            )
            x_a = batch["A"].squeeze(0).to(device)
            x_b = batch["B"].squeeze(0).to(device)
            mean_vol_a = batch["mean_vol_A"].squeeze(0).to(device)
            atlas = atlases[age_group_for_subject(batch["subject_id"][0])] \
                if atlases is not None else None
        else:
            x_a = batch["A"].to(device)
            x_b = batch["B"].to(device)
            # Two mutually exclusive denormalization schemes, depending on which dataset
            # produced this batch (see grade_dataset.py vs dataset.py's _psc_normalise):
            #   - dataset.py's flat loader: PSC, per-voxel mean_vol_A
            #   - grade_dataset.py: robust_p5p95, per-sample (median, scale) in A_meta
            mean_vol_a = batch.get("mean_vol_A")
            if mean_vol_a is not None:
                mean_vol_a = mean_vol_a.to(device)
            grade_meta_a = batch.get("A_meta")

        out = model(x_a, x_b)

        # Losses (no adversarial at val time)
        g = generator_loss(out, x_a, x_b, weights)
        loss_acc["cyc"] += g["cyc"].item()
        loss_acc["idt"] += g["idt"].item()

        if grade_meta_a is not None:
            grade_meta_b = batch["B_meta"]

            # Residual magnitude by grade: out.x_hat_b is already G_B(x_a) -- reuse it rather
            # than a second forward pass. Also run the SAME correction pipeline on x_b (Grade 1,
            # already clean) via model.correct(), which G_B never sees during normal training --
            # this is the "what if you feed it already-clean data" probe.
            l1_a = (out.x_hat_b - x_a).abs().mean(dim=(1, 2, 3, 4))
            for grade, l1 in zip(grade_meta_a["grade"], l1_a.tolist()):
                residual_l1_by_grade[grade].append(l1)

            gb_on_b = model.correct(x_b)
            l1_b = (gb_on_b - x_b).abs().mean(dim=(1, 2, 3, 4))
            for grade, l1 in zip(grade_meta_b["grade"], l1_b.tolist()):
                residual_l1_by_grade[grade].append(l1)

            # D_B score distribution: real clean (x_b) vs raw uncorrected corrupted (x_a) vs
            # corrected (out.x_hat_b, already scored as out.score_fake_b -- reused, not recomputed)
            disc_scores["real_clean"].append(_score_mean(out.score_real_b))
            disc_scores["raw_corrupted"].append(_score_mean(model.D_B(x_a)))
            disc_scores["corrected"].append(_score_mean(out.score_fake_b))

        # Sequence-specific losses
        if use_sequences:
            if weights.temporal > 0:
                loss_acc["temporal"] += temporal_consistency_loss(
                    x_a, out.x_hat_b
                ).item()
            if weights.fc > 0 and atlas is not None:
                S, T = x_a.shape[:2]
                x_a_concat = x_a.reshape(S * T, *x_a.shape[2:])
                y_concat   = out.x_hat_b.reshape(S * T, *out.x_hat_b.shape[2:])
                input_roi_ts     = atlas.extract_roi_timeseries(x_a_concat)
                corrected_roi_ts = atlas.extract_roi_timeseries(y_concat)
                fc_stats = {}
                loss_acc["fc"] += fc_loss(
                    input_roi_ts, corrected_roi_ts,
                    mask_strategy=fc_mask_strategy,
                    fc_threshold=fc_threshold,
                    top_k=fc_top_k,
                    percentile=fc_percentile,
                    validate=True,
                    stats=fc_stats,
                ).item()
                loss_acc["fc_n_retained"]    += fc_stats["n_retained"]
                loss_acc["fc_retained_frac"] += fc_stats["retained_fraction"]
                n_fc_updates += 1

        # fMRI metrics: denormalise to raw BOLD space so metrics are meaningful.
        metrics = {}
        if mean_vol_a is not None:
            bold_input     = psc_denormalise(x_a, mean_vol_a)
            bold_corrected = psc_denormalise(out.x_hat_b, mean_vol_a)
            metrics = compute_fmri_metrics(bold_input, bold_corrected)
        elif grade_meta_a is not None:
            median_a = grade_meta_a["median"].to(device)
            scale_a  = grade_meta_a["scale"].to(device)
            # Ground-truth mask from the real input, NOT the model's own output -- x_hat_b has
            # no constraint keeping background at 0, so deriving the mask from x_hat_b itself
            # would treat an untrained/imperfect model's background noise as real tissue.
            brain_mask = x_a != 0
            bold_input     = denormalize_chunk(x_a, median_a, scale_a, brain_mask)
            bold_corrected = denormalize_chunk(out.x_hat_b, median_a, scale_a, brain_mask)
            metrics = compute_fmri_metrics(bold_input, bold_corrected)
        if metrics:
            for k, v in metrics.items():
                metric_acc[k] += v

        n_batches += 1

        postfix = {
            "cyc":       f"{g['cyc'].item():.3f}",
            "idt":       f"{g['idt'].item():.3f}",
        }
        if metrics:
            postfix["tSNR↑"]  = f"{metrics['tsnr_improvement']:+.3f}"
            postfix["DVARS↓"] = f"{metrics['dvars_improvement']:+.3f}"
        if use_sequences and weights.temporal > 0:
            postfix["tc"] = f"{loss_acc['temporal']/n_batches:.4f}"
        pbar.set_postfix(postfix)

    results = {}
    for k, v in loss_acc.items():
        if k in ("fc_n_retained", "fc_retained_frac"):
            results[f"val_{k}"] = v / max(n_fc_updates, 1)
        else:
            results[f"val_{k}"] = v / max(n_batches, 1)
    for k, v in metric_acc.items():
        results[f"val_{k}"] = v / max(n_batches, 1)

    if residual_l1_by_grade:
        for grade, vals in residual_l1_by_grade.items():
            results[f"val_residual_l1_{grade.replace(' ', '')}"] = float(np.mean(vals))
        # Same number as the "Grade1" entry above, re-exposed under its own name for
        # parity with validate.py's grade1_identity_trend -- not recomputed.
        if GRADE_A in residual_l1_by_grade:
            results["val_grade1_identity_l1"] = float(np.mean(residual_l1_by_grade[GRADE_A]))
    if disc_scores["real_clean"]:
        results["val_disc_score_real_clean"]    = float(np.mean(disc_scores["real_clean"]))
        results["val_disc_score_raw_corrupted"] = float(np.mean(disc_scores["raw_corrupted"]))
        results["val_disc_score_corrected"]     = float(np.mean(disc_scores["corrected"]))

    return results


# Argument parser
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Disentangled CycleGAN for fMRI motion correction"
    )
    # Paths
    p.add_argument("--data_root", type=str,
        default="/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
                "faizan_motion_correction_dataset/cyclegans_dataset",
        help="Path to cyclegans_dataset/ root")
    p.add_argument("--ckpt_root", type=str,
        default="/lustre/disk/home/users/mfaizan/motion_correction/prototyping/motion-artefacts-correction",
        help="Root directory for checkpoints")
    p.add_argument("--run_name", type=str, default='baseline',
        help="Run name (default: run_<timestamp>)")
    p.add_argument("--resume", type=str, default=None,
        help="Path to checkpoint to resume from (e.g. checkpoints/run_01/latest.pt)")
    p.add_argument("--finetune", type=str, default=None,
        help="Path to checkpoint to load model weights from for fine-tuning. "
             "Unlike --resume, optimizer/scheduler/epoch are NOT restored — "
             "training starts fresh from epoch 1 with new LR schedule.")

    # Grade dataset (motion_grades_chunk_5_dataset_hfiltered: unpaired Grade1 vs
    # pooled Grade2-6, see grade_dataset.py). Mutually exclusive with the
    # cyclegans_dataset A_corrupted/B_motion_free loader above.
    p.add_argument("--use_grade_dataset", action="store_true",
        help="Train on motion_grades_chunk_5_dataset_hfiltered (Grade1 vs pooled "
             "Grade2-6) instead of the flat A_corrupted/B_motion_free dataset. "
             "Spatial dims are fixed at (64,72,56) for this dataset (see "
             "atlas_fc.PADDED_SPATIAL) -- overrides --in_timepoints' spatial_dims.")
    p.add_argument("--grade_chunk_metadata_csv", type=str, default=GRADE_DEFAULT_CHUNK_CSV,
        help="chunk_metadata.csv for the grade dataset")
    p.add_argument("--grade_run_stats_csv", type=str, default=GRADE_DEFAULT_STATS_CSV,
        help="run_normalization_stats.csv for the grade dataset -- must be computed "
             "against the SAME (hfiltered vs unfiltered) source as "
             "--grade_chunk_metadata_csv, or normalization will use the wrong stats")

    # Training configs 
    p.add_argument("--epochs",      type=int,   default=300)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--num_workers", type=int,   default=8)
    p.add_argument("--val_every",   type=int,   default=5,
        help="Run validation every N epochs")
    p.add_argument("--save_every",  type=int,   default=10,
        help="Save numbered checkpoint every N epochs")
    p.add_argument("--warmup",      type=int,   default=5,
        help="LR warmup epochs")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--deterministic", action="store_true",
        help="Force cudnn deterministic mode (slower; default uses cudnn.benchmark for speed)")

    # Optimiser
    p.add_argument("--lr_G",   type=float, default=2e-4, help="Generator LR")
    p.add_argument("--lr_D",   type=float, default=5e-5, help="Discriminator LR")
    p.add_argument("--beta1",  type=float, default=0.5)
    p.add_argument("--beta2",  type=float, default=0.999)

    # Loss weights
    p.add_argument("--w_adv",     type=float, default=1.0)
    p.add_argument("--w_cyc",     type=float, default=10.0)
    p.add_argument("--w_idt",     type=float, default=5.0)

    # Training stabilisation of the discriminator
    p.add_argument("--max_grad_norm",     type=float, default=10.0,
        help="Generator gradient clip max norm")
    p.add_argument("--d_update_every",    type=int,   default=2,
        help="Update discriminator every N batches")
    p.add_argument("--label_smooth_real", type=float, default=0.9,
        help="Label smoothing target for real samples")
    p.add_argument("--label_smooth_fake", type=float, default=0.1,
        help="Label smoothing target for fake samples")
    p.add_argument("--loss_warmup_epochs", type=int,  default=20,
        help="Epochs to linearly ramp cycle/identity weights from 1.0 to full")
    p.add_argument("--r1_weight", type=float, default=0.0,
        help="R1 gradient penalty weight (gamma). 0=disabled. Recommended: 10.0")
    p.add_argument("--r1_every", type=int, default=8,
        help="Apply R1 penalty every N discriminator updates (lazy regularization)")

    # Model configurations 
    p.add_argument("--in_timepoints",    type=int, default=20)
    p.add_argument("--content_ch",       type=int, default=384)
    p.add_argument("--content_base_ch",  type=int, default=64)
    p.add_argument("--content_n_res",    type=int, default=5)
    p.add_argument("--artefact_base_ch", type=int, default=64)
    p.add_argument("--global_code_dim",  type=int, default=64)
    p.add_argument("--spatial_code_ch",  type=int, default=32)
    p.add_argument("--disc_base_ch",     type=int, default=64)
    p.add_argument("--num_disc_scales",  type=int, default=2,
        help="Number of scales in the multi-scale PatchGAN discriminator")
    p.add_argument("--no_disc_temporal_diffs", action="store_true",
        help="Disable temporal difference channels in discriminator input (now off by default)")
    p.add_argument("--residual", action="store_true",
        help="Residual learning: decoders predict delta, added to input via skip connection")
    p.add_argument("--use_st_model", action="store_true",
        help="Use SpatioTemporalCycleGAN (factorized R(3+1)D) instead of DisentangledCycleGAN")

    # Sequence training (temporal consistency + FC losses)
    p.add_argument("--use_sequences", action="store_true",
        help="Sequence-aware training with temporal and FC losses")
    p.add_argument("--manifest_csv", type=str, default=None,
        help="Path to video_sequence_manifest.csv")
    p.add_argument("--chunk_metadata_csv", type=str, default=None,
        help="Path to video_chunk_metadata_with_paths.csv")
    p.add_argument("--w_temporal", type=float, default=1.0,
        help="Temporal consistency loss weight (sequence mode)")
    p.add_argument("--w_fc", type=float, default=0.0,
        help="FC preservation loss weight (sequence mode)")
    p.add_argument("--fc_mask_strategy", type=str, default="threshold",
        choices=["threshold", "topk", "percentile"],
        help="Strategy for selecting which ROI-pair FC connections to "
             "regularise — only retained pairs contribute to the FC loss")
    p.add_argument("--fc_threshold", type=float, default=0.3,
        help="|r| cutoff for fc_mask_strategy=threshold: only input ROI "
             "pairs with correlation strength above this are preserved")
    p.add_argument("--fc_top_k", type=int, default=None,
        help="Number of strongest ROI pairs to keep for fc_mask_strategy=topk")
    p.add_argument("--fc_percentile", type=float, default=None,
        help="Percentile (0-100) cutoff for fc_mask_strategy=percentile")
    p.add_argument("--atlas_2mo_path", type=str,
        default=DEFAULT_ATLAS_PATHS["2mo"],
        help="Schaefer-400 atlas (nihpd-02-05) for 2mo subjects, "
             "already aligned to the BOLD common-space grid")
    p.add_argument("--atlas_9mo_path", type=str,
        default=DEFAULT_ATLAS_PATHS["9mo"],
        help="Schaefer-400 atlas (nihpd-08-11) for 9mo subjects "
             "(subject_id ends with 'A')")

    # ROI-timeseries temporal discriminator (chunk-level, no sequence
    # stitching needed -- see models/roi_discriminator.py)
    p.add_argument("--use_roi_discriminator", action="store_true",
        help="Adversarially match ROI-timeseries realism via "
             "MultiScaleROITemporalDiscriminator, operating directly on "
             "raw per-chunk ROI timeseries. Not compatible with "
             "--use_sequences (chunk-level only).")
    p.add_argument("--lambda_roi", type=float, default=0.5,
        help="Weight on the per-ROI term vs. the whole-brain global term "
             "inside temporal_discriminator_loss / temporal_generator_loss")
    p.add_argument("--w_roi_adv", type=float, default=1.0,
        help="Generator-side weight on the ROI-timeseries adversarial loss")
    p.add_argument("--lr_D_roi", type=float, default=None,
        help="LR for the ROI discriminator's optimizer (default: --lr_D)")
    p.add_argument("--w_roi_cycle", type=float, default=0.0,
        help="Weight on the ROI-timeseries cycle-consistency loss "
             "(x_a vs x_cycle_a, in ROI space) -- chunk-level, no "
             "--use_sequences needed, and independent of "
             "--use_roi_discriminator (no discriminator required). "
             "0 = disabled (default).")

    # Smoke testing
    p.add_argument("--max_train_batches", type=int, default=None,
        help="Cap train batches per epoch (smoke testing)")
    p.add_argument("--max_val_batches", type=int, default=None,
        help="Cap val batches per epoch (smoke testing)")

    # WandB logging
    p.add_argument("--wandb_project", type=str, default="fmri-motion-correction")
    p.add_argument("--wandb_entity",  type=str, default=None,
        help="WandB username or team (leave None to use default)")
    p.add_argument("--no_wandb", action="store_true",
        help="Disable WandB logging entirely")

    args = p.parse_args()

    if args.fc_mask_strategy == "topk" and args.fc_top_k is None:
        p.error("--fc_mask_strategy=topk requires --fc_top_k to be set")
    if args.fc_mask_strategy == "percentile" and args.fc_percentile is None:
        p.error("--fc_mask_strategy=percentile requires --fc_percentile to be set")
    if args.use_roi_discriminator and args.use_sequences:
        p.error("--use_roi_discriminator is chunk-level only (operates on the "
                 "flat (B, T, ...) batch shape) and isn't wired up for "
                 "--use_sequences's (S, T, ...) batch shape")
    if args.w_roi_cycle > 0 and args.use_sequences:
        p.error("--w_roi_cycle is chunk-level only, same as "
                 "--use_roi_discriminator -- not wired up for --use_sequences")
    if args.use_grade_dataset and args.use_sequences:
        p.error("--use_grade_dataset is a flat (unpaired, non-sequence) dataset, "
                 "same as the default A_corrupted/B_motion_free loader -- not "
                 "compatible with --use_sequences")

    return args


# Reproducibility
def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    else:
        torch.backends.cudnn.benchmark     = True  # fix4: faster when --deterministic not set

def main() -> None:
    args = parse_args()

    # ── DDP initialisation ────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_ddp     = local_rank >= 0

    if is_ddp:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=480))  # non-main ranks wait here during rank-0 validation
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank       = 0
        world_size = 1
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = (rank == 0)
    # ─────────────────────────────────────────────────────────────────────────

    set_seed(args.seed + rank, deterministic=args.deterministic)  # different seed per rank to diversify augmentation

    # Run directory
    if args.run_name is None:
        args.run_name = f"run_{int(time.time())}"

    run_dir = Path(args.ckpt_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if is_main:
        print(f"\nRun directory : {run_dir}")
        print(f"Device        : {device}")
        if is_ddp:
            print(f"DDP           : {world_size} GPU(s)")
        if torch.cuda.is_available():
            print(f"GPU           : {torch.cuda.get_device_name(local_rank if is_ddp else 0)}")
            print(f"VRAM          : {torch.cuda.get_device_properties(local_rank if is_ddp else 0).total_memory / 1e9:.1f} GB")

    # WandB, offline mode for compute nodes without internet
    os.environ.setdefault("WANDB_MODE", "offline")

    if is_main and not args.no_wandb:
        wandb.init(
            project = args.wandb_project,
            entity  = args.wandb_entity,
            name    = args.run_name,
            dir     = str(run_dir),
            config  = vars(args),
            resume  = "allow",
        )
        print(f"WandB         : offline  (sync with: wandb sync {run_dir}/wandb/)")

    # CSV loggers (rank 0 only)
    train_fields = ["epoch", "lr_G", "lr_D",
                     "G_adv", "G_cyc", "G_idt", "G_total",
                     "D_A", "D_B", "D_total", "D_r1",
                     "grad_norm_G",
                     "score_real_a", "score_fake_a",
                     "score_real_b", "score_fake_b"]
    val_fields   = ["epoch",
                    "val_cyc", "val_idt",
                    "val_dvars_input",     "val_dvars_corrected",     "val_dvars_improvement",
                    "val_tsnr_input",      "val_tsnr_corrected",      "val_tsnr_improvement",
                    "val_gs_std_input",    "val_gs_std_corrected",    "val_gs_std_improvement",
                    "val_smoothness_input","val_smoothness_corrected","val_smoothness_ratio",
                    "val_score"]
    if args.use_sequences:
        train_fields += ["G_temporal", "G_fc", "G_fc_n_retained", "G_fc_retained_frac"]
        val_fields   += ["val_temporal", "val_fc", "val_fc_n_retained", "val_fc_retained_frac"]
    if args.use_roi_discriminator:
        train_fields += ["G_roi_adv", "D_roi_total"]
        # not wired into validate() yet -- train-only metric for now
    if args.w_roi_cycle > 0:
        train_fields += ["G_roi_cycle"]
    if args.use_grade_dataset:
        # Ported from pytorch-CycleGAN-and-pix2pix's validate.py: residual magnitude by grade,
        # grade1 identity (same number as val_residual_l1_Grade1, exposed separately for parity
        # with that script's naming), and D_B's score distribution.
        val_fields += [f"val_residual_l1_{g.replace(' ', '')}" for g in ["Grade 1"] + list(GRADES_B_ALL)]
        val_fields += ["val_grade1_identity_l1",
                       "val_disc_score_real_clean", "val_disc_score_raw_corrupted", "val_disc_score_corrected"]

    if is_main:
        train_csv = CSVLogger(path=run_dir / "train_losses.csv", fieldnames=train_fields)
        val_csv   = CSVLogger(path=run_dir / "val_metrics.csv",  fieldnames=val_fields)

    # Dataloaders
    if is_main:
        print("\nBuilding dataloaders ...")
    if args.use_sequences:
        if args.manifest_csv is None or args.chunk_metadata_csv is None:
            raise ValueError(
                "--manifest_csv and --chunk_metadata_csv required "
                "with --use_sequences"
            )
        if is_main:
            print(f"  Sequence mode enabled")
            print(f"  Manifest       : {args.manifest_csv}")
            print(f"  Chunk metadata : {args.chunk_metadata_csv}")
    # Train loader: sharded across GPUs with DistributedSampler
    # Val loader: full set on every rank (validation only runs on rank 0)
    if args.use_grade_dataset:
        if is_main:
            print(f"  Grade dataset enabled: Grade1 vs pooled {GRADES_B_ALL}")
            print(f"  Chunk metadata : {args.grade_chunk_metadata_csv}")
            print(f"  Run stats      : {args.grade_run_stats_csv}")
        train_ds = FMRIUnpairedGradeDataset(
            split="train", chunk_metadata_csv=args.grade_chunk_metadata_csv,
            run_stats_csv=args.grade_run_stats_csv, full_coverage=False,
        )
        val_ds = FMRIUnpairedGradeDataset(
            split="val", chunk_metadata_csv=args.grade_chunk_metadata_csv,
            run_stats_csv=args.grade_run_stats_csv, full_coverage=True,
        )
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
        loaders = {
            "train": DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
                sampler=train_sampler, num_workers=args.num_workers, pin_memory=True,
                worker_init_fn=worker_init_fn, drop_last=True,
            ),
            "val": DataLoader(
                val_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn,
            ),
        }
        if is_main:
            print(f"  train: A={train_ds.A_size} B={train_ds.B_size} epoch_len={len(train_ds)}")
            print(f"  val  : A={val_ds.A_size} B={val_ds.B_size} epoch_len={len(val_ds)} (full_coverage)")
    else:
        _loader_kwargs = dict(
            dataset_root       = args.data_root,
            batch_size         = args.batch_size,
            num_workers        = args.num_workers,
            pin_memory         = True,
            augment_train      = True,
            sequence_mode      = args.use_sequences,
            manifest_csv       = args.manifest_csv,
            chunk_metadata_csv = args.chunk_metadata_csv,
        )
        loaders = build_dataloaders(
            splits=["train"], distributed=is_ddp, world_size=world_size, rank=rank,
            **_loader_kwargs,
        )
        loaders.update(build_dataloaders(
            splits=["val"], distributed=False,
            **_loader_kwargs,
        ))
    
    # Model
    if is_main:
        print("Building model ...")
    model_spatial_dims = PADDED_SPATIAL if args.use_grade_dataset else (80, 96, 72)
    if is_main:
        print(f"  spatial_dims: {model_spatial_dims}" + (" (grade dataset)" if args.use_grade_dataset else ""))
    if args.use_st_model:
        model = SpatioTemporalCycleGAN(
            in_timepoints    = args.in_timepoints,
            spatial_dims     = model_spatial_dims,
            content_base_ch  = args.content_base_ch,
            content_n_res    = args.content_n_res,
            artefact_base_ch = args.artefact_base_ch,
            global_code_dim  = args.global_code_dim,
            spatial_code_ch  = args.spatial_code_ch,
            disc_base_ch     = args.disc_base_ch,
            num_disc_scales  = args.num_disc_scales,
            residual         = args.residual,
        ).to(device)
    else:
        model = DisentangledCycleGAN(
            in_timepoints    = args.in_timepoints,
            spatial_dims     = model_spatial_dims,
            content_ch       = args.content_ch,
            content_base_ch  = args.content_base_ch,
            content_n_res    = args.content_n_res,
            artefact_base_ch = args.artefact_base_ch,
            global_code_dim  = args.global_code_dim,
            spatial_code_ch  = args.spatial_code_ch,
            disc_base_ch        = args.disc_base_ch,
            num_disc_scales     = args.num_disc_scales,
            disc_temporal_diffs = not args.no_disc_temporal_diffs,
            residual            = args.residual,
        ).to(device)

    # Wrap with DDP if multi-GPU
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    raw_model = model.module if is_ddp else model  # access submodules without DDP wrapper

    # Parameter count summary (rank 0 only)
    if is_main:
        param_counts = raw_model.count_parameters()
        print("\nParameter counts:")
        for name, count in param_counts.items():
            print(f"  {name:<40} {count:>12,}")
        if not args.no_wandb:
            wandb.config.update({"param_counts": param_counts})

    # Loss weights
    weights = LossWeights(
        adv      = args.w_adv,
        cyc      = args.w_cyc,
        idt      = args.w_idt,
        temporal = args.w_temporal if args.use_sequences else 0.0,
        fc       = args.w_fc       if args.use_sequences else 0.0,
    )

    # Schaefer-400 atlases for FC loss (age-appropriate: 2mo vs 9mo) and/or
    # the ROI-timeseries temporal discriminator / cycle-consistency loss
    atlases = None
    if (args.use_sequences and weights.fc > 0) or args.use_roi_discriminator or args.w_roi_cycle > 0:
        if is_main:
            print("\nLoading Schaefer-400 atlases ...")
        if args.use_grade_dataset:
            # Crop-based (not resize-based) -- matches motion_grades_chunk_5_dataset_hfiltered's
            # own voxel grid exactly, see atlas_fc.SchaeferAtlasCropped.
            atlases = load_age_atlases_cropped(
                atlas_paths={"2mo": args.atlas_2mo_path, "9mo": args.atlas_9mo_path},
            )
        else:
            atlases = load_age_atlases(
                target_spatial=TARGET_SPATIAL,
                atlas_paths={"2mo": args.atlas_2mo_path, "9mo": args.atlas_9mo_path},
            )

    # Optimisers
    betas = (args.beta1, args.beta2)

    opt_G = Adam(raw_model.generator_parameters(),     lr=args.lr_G, betas=betas)
    opt_D = Adam(raw_model.discriminator_parameters(), lr=args.lr_D, betas=betas)

    # ROI-timeseries temporal discriminator (chunk-level, no stitching)
    roi_disc  = None
    opt_D_roi = None
    if args.use_roi_discriminator:
        atlas_2mo, atlas_9mo = atlases["2mo"], atlases["9mo"]
        if atlas_2mo.active_labels != atlas_9mo.active_labels:
            raise ValueError(
                "--use_roi_discriminator requires both age-group atlases to "
                "retain the exact same set of ROIs after downsampling (a "
                "single MultiScaleROITemporalDiscriminator is shared across "
                f"both) -- got {atlas_2mo.n_rois} active ROIs for 2mo vs "
                f"{atlas_9mo.n_rois} for 9mo, or a mismatched ROI set at "
                "equal counts. Use a target_spatial where both variants "
                "keep identical ROI coverage."
            )
        if is_main:
            print(f"ROI discriminator: n_rois={atlas_2mo.n_rois}  "
                  f"lambda_roi={args.lambda_roi}  w_roi_adv={args.w_roi_adv}")
        roi_disc  = MultiScaleROITemporalDiscriminator(n_rois=atlas_2mo.n_rois).to(device)
        if is_ddp:
            # roi_disc is never wrapped in DDP() (see train_one_epoch's manual all_reduce for
            # why), so it misses DDP's automatic broadcast-from-rank-0-at-construction-time --
            # every rank built its own random init (set_seed uses a different seed per rank),
            # so without this every rank starts training a DIFFERENT roi_disc.
            for p in roi_disc.parameters():
                dist.broadcast(p.data, src=0)
        opt_D_roi = Adam(roi_disc.parameters(), lr=args.lr_D_roi or args.lr_D, betas=betas)

  
    # Schedulers
    sched_G = build_scheduler(opt_G, args.epochs, warmup=args.warmup)
    sched_D = build_scheduler(opt_D, args.epochs, warmup=args.warmup)

    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_score  = float("-inf")

    if args.finetune is not None:
        finetune_path = Path(args.finetune)
        if finetune_path.exists():
            if is_main:
                print(f"\nFine-tuning from {finetune_path} (model weights only) ...")
            ckpt = torch.load(finetune_path, map_location=device, weights_only=False)
            raw_model.load_state_dict(ckpt["model"])
            if is_main:
                print(f"  Loaded model weights from epoch {ckpt['epoch']}  "
                      f"(optimizer/scheduler reset to epoch-1 state)")
        else:
            if is_main:
                print(f"  Warning: finetune path {finetune_path} not found — starting fresh")

    elif args.resume is not None:
        resume_path = Path(args.resume)
        if resume_path.exists():
            if is_main:
                print(f"\nResuming from {resume_path} ...")
            start_epoch, best_score = load_checkpoint(
                resume_path, raw_model, opt_G, opt_D, sched_G, sched_D, device,
                verbose=is_main, roi_disc=roi_disc, opt_D_roi=opt_D_roi,
            )
        else:
            if is_main:
                print(f"  Warning: resume path {resume_path} not found — starting fresh")

    # Also check for latest.pt in run_dir automatically
    elif (run_dir / "latest.pt").exists():
        if is_main:
            print(f"\nFound latest.pt in {run_dir} — resuming automatically ...")
        start_epoch, best_score = load_checkpoint(
            run_dir / "latest.pt",
            raw_model, opt_G, opt_D, sched_G, sched_D, device,
            verbose=is_main, roi_disc=roi_disc, opt_D_roi=opt_D_roi,
        )

    # Replay buffers for discriminator training stability
    replay_buf_a = ReplayBuffer(max_size=50)
    replay_buf_b = ReplayBuffer(max_size=50)

    # Training loop
    if is_main:
        print(f"\nStarting training: epochs {start_epoch} → {args.epochs}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # Tell DistributedSampler which epoch this is (shuffles differently each epoch)
        if is_ddp and hasattr(loaders["train"].sampler, "set_epoch"):
            loaders["train"].sampler.set_epoch(epoch)

        #  Train
        epoch_weights = get_epoch_weights(weights, epoch, args.loss_warmup_epochs)
        train_metrics = train_one_epoch(
            model, loaders["train"], opt_G, opt_D, epoch_weights, device, epoch,
            max_grad_norm=args.max_grad_norm,
            d_update_every=args.d_update_every,
            label_smooth_real=args.label_smooth_real,
            label_smooth_fake=args.label_smooth_fake,
            replay_buffer_a=replay_buf_a,
            replay_buffer_b=replay_buf_b,
            r1_weight=args.r1_weight,
            r1_every=args.r1_every,
            use_sequences=args.use_sequences,
            atlases=atlases,
            max_batches=args.max_train_batches,
            fc_mask_strategy=args.fc_mask_strategy,
            fc_threshold=args.fc_threshold,
            fc_top_k=args.fc_top_k,
            fc_percentile=args.fc_percentile,
            show_progress=is_main,
            is_ddp=is_ddp,
            world_size=world_size,
            roi_disc=roi_disc,
            opt_D_roi=opt_D_roi,
            lambda_roi=args.lambda_roi,
            w_roi_adv=args.w_roi_adv,
            w_roi_cycle=args.w_roi_cycle,
        )

        # Barrier: wait for all ranks to finish training before rank 0 does
        # validation/checkpointing. Without this, non-main ranks advance to the
        # next epoch and enter DDP forward while rank 0 is still validating → deadlock.
        if is_ddp:
            dist.barrier()

        # Step schedulers
        sched_G.step()
        sched_D.step()

        lr_G = sched_G.get_last_lr()[0]
        lr_D = sched_D.get_last_lr()[0]
        epoch_time = time.time() - epoch_start

        if is_main:
            # Console summary
            summary = (
                f"Epoch {epoch:03d}/{args.epochs}  "
                f"({epoch_time:.0f}s)  "
                f"G={train_metrics['G_total']:.4f}  "
                f"cyc={train_metrics['G_cyc']:.4f}  "
                f"idt={train_metrics['G_idt']:.4f}  "
                f"D_A={train_metrics['D_A']:.4f}  "
                f"D_B={train_metrics['D_B']:.4f}  "
                f"r1={train_metrics.get('D_r1', 0):.4f}  "
                f"∇G={train_metrics['grad_norm_G']:.3f}  "
                f"lr_G={lr_G:.2e}"
            )
            if args.use_sequences:
                summary += (
                    f"  tc={train_metrics['G_temporal']:.4f}  "
                    f"fc={train_metrics['G_fc']:.4f}  "
                    f"fc_pairs={train_metrics.get('G_fc_n_retained', 0):.0f}"
                    f"({train_metrics.get('G_fc_retained_frac', 0):.1%})"
                )
            print(summary)

            # CSV
            train_csv.write({
                "epoch": epoch,
                "lr_G":  lr_G,
                "lr_D":  lr_D,
                **{k: f"{v:.6f}" for k, v in train_metrics.items()},
            })
            # WandB train
            if not args.no_wandb:
                wandb.log({
                    "epoch": epoch,
                    "lr_G":  lr_G,
                    "lr_D":  lr_D,
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                }, step=epoch)

        #  Validation (rank 0 only)
        if is_main and epoch % args.val_every == 0:
            torch.cuda.empty_cache()  # release training memory before val pass
            val_metrics = validate(
                raw_model, loaders["val"], epoch_weights, device, epoch,  # raw_model: no DDP overhead during val
                use_sequences=args.use_sequences,
                atlases=atlases,
                max_batches=args.max_val_batches,
                fc_mask_strategy=args.fc_mask_strategy,
                fc_threshold=args.fc_threshold,
                fc_top_k=args.fc_top_k,
                fc_percentile=args.fc_percentile,
            )
            val_score = compute_val_score({
                k.replace("val_", ""): v
                for k, v in val_metrics.items()
                if k.startswith("val_")
            })
            val_metrics["val_score"] = val_score

            # Console
            val_summary = (
                f"  [VAL]  "
                f"cyc={val_metrics['val_cyc']:.4f}  "
                f"idt={val_metrics['val_idt']:.4f}  "
                f"tSNR↑={val_metrics['val_tsnr_improvement']:+.4f}  "
                f"DVARS↓={val_metrics['val_dvars_improvement']:+.4f}  "
                f"GS_std↓={val_metrics['val_gs_std_improvement']:+.4f}  "
                f"smooth={val_metrics['val_smoothness_ratio']:.3f}  "
                f"score={val_score:.4f}"
            )
            if args.use_sequences:
                val_summary += (
                    f"  tc={val_metrics['val_temporal']:.4f}  "
                    f"fc={val_metrics['val_fc']:.4f}  "
                    f"fc_pairs={val_metrics.get('val_fc_n_retained', 0):.0f}"
                    f"({val_metrics.get('val_fc_retained_frac', 0):.1%})"
                )
            print(val_summary)

            # CSV
            val_csv.write({"epoch": epoch,
                           **{k: f"{v:.6f}" for k, v in val_metrics.items()}})

            # WandB
            if not args.no_wandb:
                wandb.log({
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                }, step=epoch)

            # Best model
            if val_score > best_score:
                best_score = val_score
                best_path  = run_dir / "best_model.pt"
                save_checkpoint(best_path, epoch, raw_model,
                                opt_G, opt_D, sched_G, sched_D,
                                best_score, args, roi_disc=roi_disc, opt_D_roi=opt_D_roi)
                print(f"  [VAL]  ✓ New best score={best_score:.4f}  saved → {best_path}")

        # Numbered checkpoint every save_every epochs (rank 0 only)
        if is_main and epoch % args.save_every == 0:
            numbered = run_dir / f"epoch_{epoch:03d}.pt"
            save_checkpoint(numbered, epoch, raw_model,
                            opt_G, opt_D, sched_G, sched_D,
                            best_score, args, roi_disc=roi_disc, opt_D_roi=opt_D_roi)

        #  Latest checkpoint every epoch (crash recovery, rank 0 only)
        if is_main:
            save_checkpoint(run_dir / "latest.pt", epoch, raw_model,
                            opt_G, opt_D, sched_G, sched_D,
                            best_score, args, roi_disc=roi_disc, opt_D_roi=opt_D_roi)

        # Barrier at end of epoch: non-main ranks wait here until rank 0 finishes
        # saving checkpoints before advancing to the next epoch.
        if is_ddp:
            dist.barrier()

    # End of training
    if is_main:
        print(f"\nTraining complete.  Best val score: {best_score:.4f}")
        print(f"Checkpoints saved to: {run_dir}")
        if not args.no_wandb:
            wandb.finish()
            print(f"Sync WandB with: wandb sync {run_dir}/wandb/")

    if is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()