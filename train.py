#!/usr/bin/env python3
"""
train.py
========
Training script for Disentangled CycleGAN fMRI motion artefact correction.
"""

import argparse
import csv
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import SequentialLR, ConstantLR, LinearLR
from tqdm import tqdm



from dataset import build_dataloaders, psc_denormalise          # dataset.py
from losses  import (generator_loss, discriminator_loss,        # losses.py
                     LossWeights, ModelOutputs)
from models.model import DisentangledCycleGAN                 



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
    mean = x.mean(dim=1)                        #(B, X, Y, Z)
    std  = x.std(dim=1).clamp(min=1e-6)        # (B, X, Y, Z)  avoid /0
    tsnr = mean.abs() / std                     # (B, X, Y, Z)
    # Mask: only brain voxels (non-zero mean across T)
    brain = mean.abs() > 0
    if brain.sum() == 0:
        return 0.0
    return tsnr[brain].mean().item()


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
def compute_val_score(metrics: Dict[str, float]) -> float:
    """
    Composite score for best-model checkpoint selection.

    Rewards:
        tSNR improvement    — primary signal quality metric
        DVARS improvement   — artefact reduction
        GS std improvement  — global signal stability

    Penalises:
        smoothness_ratio > 1.1  — over-smoothing destroys spatial structure
    """
    score = (metrics["tsnr_improvement"]
             + metrics["dvars_improvement"]
             + metrics["gs_std_improvement"])

    if metrics["smoothness_ratio"] > 1.1:
        score -= (metrics["smoothness_ratio"] - 1.1) * 10.0

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
                           start_factor = 1e-6,
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
) -> None:
    """Save full training state to path."""
    torch.save({
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
    }, path)


def load_checkpoint(
    path:   Path,
    model:  nn.Module,
    opt_G:  Adam,
    opt_D:  Adam,
    sched_G,
    sched_D,
    device: torch.device,
) -> Tuple[int, float]:
    """
    Load training state from checkpoint.

    Returns:
        (start_epoch, best_score)
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    opt_G.load_state_dict(ckpt["opt_G"])
    opt_D.load_state_dict(ckpt["opt_D"])
    sched_G.load_state_dict(ckpt["sched_G"])
    sched_D.load_state_dict(ckpt["sched_D"])

    torch.set_rng_state(ckpt["rng_torch"])
    np.random.set_state(ckpt["rng_numpy"])
    random.setstate(ckpt["rng_python"])
    if ckpt["rng_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(ckpt["rng_cuda"])

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



# Single training epoch
def train_one_epoch(
    model:    DisentangledCycleGAN,
    loader,
    opt_G:    Adam,
    opt_D:    Adam,
    weights:  LossWeights,
    device:   torch.device,
    epoch:    int,
) -> Dict[str, float]:
    """
    Run one full training epoch.

    Returns dict of mean losses over the epoch.
    """
    model.train()

    # Accumulators
    acc = {k: 0.0 for k in
           ["G_adv", "G_cyc", "G_idt", "G_content", "G_art", "G_total",
            "D_A", "D_B", "D_total", "grad_norm_G",
            "score_real_a", "score_fake_a",
            "score_real_b", "score_fake_b"]}
    n_batches = 0

    # Advance A queue at the start of each epoch
    loader.dataset.on_epoch_start()

    pbar = tqdm(loader,
                desc=f"Epoch {epoch:03d} [train]",
                leave=False,
                dynamic_ncols=True)

    for batch in pbar:
        x_a = batch["A"].to(device)   # (B, T, X, Y, Z)  corrupted
        x_b = batch["B"].to(device)   # (B, T, X, Y, Z)  motion-free

    
        # Step 1 — Update discriminators
        for p in model.discriminator_parameters():
            p.requires_grad_(True)
        for p in model.generator_parameters():
            p.requires_grad_(False)

        with torch.no_grad():
            out_for_disc = model(x_a, x_b)   # scores needed, no gen gradients

        d_losses = discriminator_loss(out_for_disc)
        opt_D.zero_grad()
        d_losses["total"].backward()
        opt_D.step()

        # Step 2 — Update generators + encoders
        for p in model.discriminator_parameters():
            p.requires_grad_(False)
        for p in model.generator_parameters():
            p.requires_grad_(True)

        out = model(x_a, x_b)
        g_losses = generator_loss(out, x_a, x_b, weights)
        opt_G.zero_grad()
        g_losses["total"].backward()

        # Gradient clipping — generators only, norm=1.0
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.generator_parameters(), max_norm=1.0
        ).item()

        opt_G.step()

        
        # Accumulate
        acc["G_adv"]     += g_losses["adv"].item()
        acc["G_cyc"]     += g_losses["cyc"].item()
        acc["G_idt"]     += g_losses["idt"].item()
        acc["G_content"] += g_losses["content"].item()
        acc["G_art"]     += g_losses["art"].item()
        acc["G_total"]   += g_losses["total"].item()
        acc["D_A"]       += d_losses["D_A"].item()
        acc["D_B"]       += d_losses["D_B"].item()
        acc["D_total"]   += d_losses["total"].item()
        acc["grad_norm_G"] += grad_norm
        acc["score_real_a"] += out.score_real_a.mean().item()
        acc["score_fake_a"] += out.score_fake_a.mean().item()
        acc["score_real_b"] += out.score_real_b.mean().item()
        acc["score_fake_b"] += out.score_fake_b.mean().item()
        n_batches += 1

        # tqdm postfix — show key losses live
        pbar.set_postfix({
            "G":    f"{g_losses['total'].item():.3f}",
            "cyc":  f"{g_losses['cyc'].item():.3f}",
            "idt":  f"{g_losses['idt'].item():.3f}",
            "D_A":  f"{d_losses['D_A'].item():.3f}",
            "D_B":  f"{d_losses['D_B'].item():.3f}",
            "∇G":   f"{grad_norm:.2f}",
        })

    return {k: v / n_batches for k, v in acc.items()}



# Validation epoch
@torch.no_grad()
def validate(
    model:   DisentangledCycleGAN,
    loader,
    weights: LossWeights,
    device:  torch.device,
    epoch:   int,
) -> Dict[str, float]:
    """
    Run full validation pass.

    Computes:
        - Generator losses (cyc, idt, content, art) — no adversarial
        - fMRI quality metrics on x_a vs x_hat_b (corrected output)

    Returns dict of mean metrics over the full val set.
    """
    model.eval()

    loss_acc = {k: 0.0 for k in ["cyc", "idt", "content", "art"]}
    metric_acc = {k: 0.0 for k in [
        "dvars_input", "dvars_corrected", "dvars_improvement",
        "tsnr_input",  "tsnr_corrected",  "tsnr_improvement",
        "gs_std_input", "gs_std_corrected", "gs_std_improvement",
        "smoothness_input", "smoothness_corrected", "smoothness_ratio",
    ]}
    n_batches = 0

    pbar = tqdm(loader,
                desc=f"Epoch {epoch:03d} [val]  ",
                leave=False,
                dynamic_ncols=True)

    for batch in pbar:
        x_a = batch["A"].to(device)
        x_b = batch["B"].to(device)

        out = model(x_a, x_b)

        # Losses (no adversarial at val time)
        g = generator_loss(out, x_a, x_b, weights)
        loss_acc["cyc"]     += g["cyc"].item()
        loss_acc["idt"]     += g["idt"].item()
        loss_acc["content"] += g["content"].item()
        loss_acc["art"]     += g["art"].item()

        # fMRI metrics: compare corrupted input vs corrected output
        metrics = compute_fmri_metrics(x_a, out.x_hat_b)
        for k, v in metrics.items():
            metric_acc[k] += v

        n_batches += 1

        pbar.set_postfix({
            "cyc":       f"{g['cyc'].item():.3f}",
            "idt":       f"{g['idt'].item():.3f}",
            "tSNR↑":     f"{metrics['tsnr_improvement']:+.3f}",
            "DVARS↓":    f"{metrics['dvars_improvement']:+.3f}",
        })

    results = {}
    for k, v in loss_acc.items():
        results[f"val_{k}"] = v / n_batches
    for k, v in metric_acc.items():
        results[f"val_{k}"] = v / n_batches

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
        default="/lustre/disk/home/users/mfaizan/motion_correction/checkpoints",
        help="Root directory for checkpoints")
    p.add_argument("--run_name", type=str, default=None,
        help="Run name (default: run_<timestamp>)")
    p.add_argument("--resume", type=str, default=None,
        help="Path to checkpoint to resume from (e.g. checkpoints/run_01/latest.pt)")

    # Training
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

    # Optimiser
    p.add_argument("--lr_G",   type=float, default=2e-4, help="Generator LR")
    p.add_argument("--lr_D",   type=float, default=1e-4, help="Discriminator LR")
    p.add_argument("--beta1",  type=float, default=0.5)
    p.add_argument("--beta2",  type=float, default=0.999)

    # Loss weights
    p.add_argument("--w_adv",     type=float, default=1.0)
    p.add_argument("--w_cyc",     type=float, default=20.0)
    p.add_argument("--w_idt",     type=float, default=20.0)
    p.add_argument("--w_content", type=float, default=0.5)
    p.add_argument("--w_art",     type=float, default=0.1)

    # Model
    p.add_argument("--in_timepoints",    type=int, default=20)
    p.add_argument("--content_ch",       type=int, default=384)
    p.add_argument("--content_base_ch",  type=int, default=64)
    p.add_argument("--content_n_res",    type=int, default=5)
    p.add_argument("--artefact_base_ch", type=int, default=64)
    p.add_argument("--global_code_dim",  type=int, default=64)
    p.add_argument("--spatial_code_ch",  type=int, default=32)
    p.add_argument("--disc_base_ch",     type=int, default=64)

    # WandB
    p.add_argument("--wandb_project", type=str, default="fmri-motion-correction")
    p.add_argument("--wandb_entity",  type=str, default=None,
        help="WandB username or team (leave None to use default)")
    p.add_argument("--no_wandb", action="store_true",
        help="Disable WandB logging entirely")

    return p.parse_args()



# Reproducibility
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

  
    # Run directory
    if args.run_name is None:
        args.run_name = f"run_{int(time.time())}"

    run_dir = Path(args.ckpt_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRun directory : {run_dir}")
    print(f"Device        : {device}")
    if torch.cuda.is_available():
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
        print(f"VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    
    # WandB — offline mode for compute nodes without internet
    os.environ.setdefault("WANDB_MODE", "offline")

    if not args.no_wandb:
        wandb.init(
            project = args.wandb_project,
            entity  = args.wandb_entity,
            name    = args.run_name,
            dir     = str(run_dir),
            config  = vars(args),
            resume  = "allow",
        )
        print(f"WandB         : offline  (sync with: wandb sync {run_dir}/wandb/)")

    # CSV loggers
    train_csv = CSVLogger(
        path       = run_dir / "train_losses.csv",
        fieldnames = ["epoch", "lr_G", "lr_D",
                      "G_adv", "G_cyc", "G_idt", "G_content", "G_art", "G_total",
                      "D_A", "D_B", "D_total",
                      "grad_norm_G",
                      "score_real_a", "score_fake_a",
                      "score_real_b", "score_fake_b"],
    )

    val_csv = CSVLogger(
        path       = run_dir / "val_metrics.csv",
        fieldnames = ["epoch",
                      "val_cyc", "val_idt", "val_content", "val_art",
                      "val_dvars_input",     "val_dvars_corrected",     "val_dvars_improvement",
                      "val_tsnr_input",      "val_tsnr_corrected",      "val_tsnr_improvement",
                      "val_gs_std_input",    "val_gs_std_corrected",    "val_gs_std_improvement",
                      "val_smoothness_input","val_smoothness_corrected","val_smoothness_ratio",
                      "val_score"],
    )

    # Dataloaders
    print("\nBuilding dataloaders ...")
    loaders = build_dataloaders(
        dataset_root  = args.data_root,
        splits        = ["train", "val"],
        batch_size    = args.batch_size,
        num_workers   = args.num_workers,
        pin_memory    = True,
        augment_train = True,
    )

    # Model
    print("Building model ...")
    model = DisentangledCycleGAN(
        in_timepoints    = args.in_timepoints,
        spatial_dims     = (80, 96, 72),
        content_ch       = args.content_ch,
        content_base_ch  = args.content_base_ch,
        content_n_res    = args.content_n_res,
        artefact_base_ch = args.artefact_base_ch,
        global_code_dim  = args.global_code_dim,
        spatial_code_ch  = args.spatial_code_ch,
        disc_base_ch     = args.disc_base_ch,
    ).to(device)

    # Parameter count summary
    param_counts = model.count_parameters()
    print("\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name:<40} {count:>12,}")

    if not args.no_wandb:
        wandb.config.update({"param_counts": param_counts})

    # Loss weights
    weights = LossWeights(
        adv     = args.w_adv,
        cyc     = args.w_cyc,
        idt     = args.w_idt,
        content = args.w_content,
        art     = args.w_art,
    )

    # Optimisers
    betas = (args.beta1, args.beta2)

    opt_G = Adam(model.generator_parameters(),     lr=args.lr_G, betas=betas)
    opt_D = Adam(model.discriminator_parameters(), lr=args.lr_D, betas=betas)

  
    # Schedulers
    sched_G = build_scheduler(opt_G, args.epochs, warmup=args.warmup)
    sched_D = build_scheduler(opt_D, args.epochs, warmup=args.warmup)

    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_score  = float("-inf")

    if args.resume is not None:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"\nResuming from {resume_path} ...")
            start_epoch, best_score = load_checkpoint(
                resume_path, model, opt_G, opt_D, sched_G, sched_D, device
            )
        else:
            print(f"  Warning: resume path {resume_path} not found — starting fresh")

    # Also check for latest.pt in run_dir automatically
    elif (run_dir / "latest.pt").exists():
        print(f"\nFound latest.pt in {run_dir} — resuming automatically ...")
        start_epoch, best_score = load_checkpoint(
            run_dir / "latest.pt",
            model, opt_G, opt_D, sched_G, sched_D, device
        )

    # Training loop
    print(f"\nStarting training: epochs {start_epoch} → {args.epochs}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # ---- Train ----
        train_metrics = train_one_epoch(
            model, loaders["train"], opt_G, opt_D, weights, device, epoch
        )

        # Step schedulers
        sched_G.step()
        sched_D.step()

        lr_G = sched_G.get_last_lr()[0]
        lr_D = sched_D.get_last_lr()[0]
        epoch_time = time.time() - epoch_start

        # ---- Console summary ----
        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"({epoch_time:.0f}s)  "
            f"G={train_metrics['G_total']:.4f}  "
            f"cyc={train_metrics['G_cyc']:.4f}  "
            f"idt={train_metrics['G_idt']:.4f}  "
            f"art={train_metrics['G_art']:.4f}  "
            f"D_A={train_metrics['D_A']:.4f}  "
            f"D_B={train_metrics['D_B']:.4f}  "
            f"∇G={train_metrics['grad_norm_G']:.3f}  "
            f"lr_G={lr_G:.2e}"
        )

        # ---- CSV ----
        train_csv.write({
            "epoch": epoch,
            "lr_G":  lr_G,
            "lr_D":  lr_D,
            **{k: f"{v:.6f}" for k, v in train_metrics.items()},
        })

        # ---- WandB train ----
        if not args.no_wandb:
            wandb.log({
                "epoch": epoch,
                "lr_G":  lr_G,
                "lr_D":  lr_D,
                **{f"train/{k}": v for k, v in train_metrics.items()},
            }, step=epoch)

        # ---- Validation ----
        if epoch % args.val_every == 0:
            val_metrics = validate(
                model, loaders["val"], weights, device, epoch
            )
            val_score = compute_val_score({
                k.replace("val_", ""): v
                for k, v in val_metrics.items()
                if k.startswith("val_")
            })
            val_metrics["val_score"] = val_score

            # Console
            print(
                f"  [VAL]  "
                f"cyc={val_metrics['val_cyc']:.4f}  "
                f"idt={val_metrics['val_idt']:.4f}  "
                f"tSNR↑={val_metrics['val_tsnr_improvement']:+.4f}  "
                f"DVARS↓={val_metrics['val_dvars_improvement']:+.4f}  "
                f"GS_std↓={val_metrics['val_gs_std_improvement']:+.4f}  "
                f"smooth={val_metrics['val_smoothness_ratio']:.3f}  "
                f"score={val_score:.4f}"
            )

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
                save_checkpoint(best_path, epoch, model,
                                opt_G, opt_D, sched_G, sched_D,
                                best_score, args)
                print(f"  [VAL]  ✓ New best score={best_score:.4f}  saved → {best_path}")

        # ---- Numbered checkpoint every save_every epochs ----
        if epoch % args.save_every == 0:
            numbered = run_dir / f"epoch_{epoch:03d}.pt"
            save_checkpoint(numbered, epoch, model,
                            opt_G, opt_D, sched_G, sched_D,
                            best_score, args)

        # ---- Latest checkpoint every epoch (crash recovery) ----
        save_checkpoint(run_dir / "latest.pt", epoch, model,
                        opt_G, opt_D, sched_G, sched_D,
                        best_score, args)

   
    # End of training
    print(f"\nTraining complete.  Best val score: {best_score:.4f}")
    print(f"Checkpoints saved to: {run_dir}")
    if not args.no_wandb:
        wandb.finish()
        print(f"Sync WandB with: wandb sync {run_dir}/wandb/")


if __name__ == "__main__":
    main()