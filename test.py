#!/usr/bin/env python3
"""
test.py
=======
Inference and evaluation script for Disentangled CycleGAN fMRI motion
artefact correction.

Usage:
    python test.py --checkpoint /path/to/best_model.pt

    Override output location:
    python test.py --checkpoint /path/to/best_model.pt --data_root /path/to/cyclegans_dataset
"""
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import (
    build_dataloaders,
    psc_denormalise,
    upsample_to_original,
    _load_nifti,
    _psc_normalise,
    TARGET_SPATIAL,
)
from models.model import DisentangledCycleGAN


# fMRI quality metrics
# Identical to train.py — kept here so test.py is self-contained
def compute_dvars(x: torch.Tensor) -> float:
    """RMS of frame-to-frame signal change. Lower = fewer motion spikes."""
    diff = x[:, 1:, ...] - x[:, :-1, ...]
    return diff.pow(2).mean(dim=(-3, -2, -1)).sqrt().mean().item()


def compute_tsnr(x: torch.Tensor) -> float:
    """Mean / std over time, averaged over brain voxels. Higher = cleaner."""
    mean  = x.mean(dim=1)
    std   = x.std(dim=1).clamp(min=1e-6)
    tsnr  = mean.abs() / std
    brain = mean.abs() > 0
    return tsnr[brain].mean().item() if brain.sum() > 0 else 0.0


def compute_global_signal_std(x: torch.Tensor) -> float:
    """Std of whole-brain mean timeseries. Lower = more stable global signal."""
    return x.mean(dim=(-3, -2, -1)).std(dim=1).mean().item()


def compute_spatial_smoothness(x: torch.Tensor) -> float:
    """Mean absolute spatial gradient. Higher = more variation = less smooth."""
    gx = (x[:, :, 1:] - x[:, :, :-1]).abs()
    gy = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs()
    gz = (x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).abs()
    return ((gx.mean() + gy.mean() + gz.mean()) / 3.0).item()


def compute_all_metrics(
    x_input:     torch.Tensor,
    x_corrected: torch.Tensor,
    x_reference: torch.Tensor = None,
) -> Dict[str, float]:
    """
    Compute quality metrics for input, corrected, and optional reference.

    Args:
        x_input     : (B, T, X, Y, Z)  corrupted chunk in PSC space
        x_corrected : (B, T, X, Y, Z)  model corrected output in PSC space
        x_reference : (B, T, X, Y, Z)  motion-free reference (upper bound)

    Returns:
        Dict of scalar metrics with _input, _corrected, _improvement suffixes.
        If x_reference provided, also includes _reference entries.
    """
    xi = x_input.detach().cpu()
    xc = x_corrected.detach().cpu()

    metrics = {}

    for tag, x in [("input", xi), ("corrected", xc)]:
        metrics[f"dvars_{tag}"]      = compute_dvars(x)
        metrics[f"tsnr_{tag}"]       = compute_tsnr(x)
        metrics[f"gs_std_{tag}"]     = compute_global_signal_std(x)
        metrics[f"smoothness_{tag}"] = compute_spatial_smoothness(x)

    if x_reference is not None:
        xr = x_reference.detach().cpu()
        for tag, x in [("reference", xr)]:
            metrics[f"dvars_{tag}"]      = compute_dvars(x)
            metrics[f"tsnr_{tag}"]       = compute_tsnr(x)
            metrics[f"gs_std_{tag}"]     = compute_global_signal_std(x)
            metrics[f"smoothness_{tag}"] = compute_spatial_smoothness(x)

    metrics["dvars_improvement"]      = metrics["dvars_input"]      - metrics["dvars_corrected"]
    metrics["tsnr_improvement"]       = metrics["tsnr_corrected"]   - metrics["tsnr_input"]
    metrics["gs_std_improvement"]     = metrics["gs_std_input"]     - metrics["gs_std_corrected"]
    metrics["smoothness_ratio"]       = (metrics["smoothness_corrected"]
                                         / (metrics["smoothness_input"] + 1e-8))
    return metrics


def summary_stats(values: List[float]) -> Dict[str, float]:
    """Mean, std, median, 25th and 75th percentile of a list of values."""
    arr = np.array(values)
    return {
        "mean":   float(arr.mean()),
        "std":    float(arr.std()),
        "median": float(np.median(arr)),
        "p25":    float(np.percentile(arr, 25)),
        "p75":    float(np.percentile(arr, 75)),
    }


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[DisentangledCycleGAN, argparse.Namespace]:
    """
    Load model from checkpoint. Reconstructs architecture from saved args.

    Returns:
        (model, args)  where args is the Namespace saved during training
    """
    ckpt       = torch.load(checkpoint_path, map_location=device)
    train_args = argparse.Namespace(**ckpt["args"])

    model = DisentangledCycleGAN(
        in_timepoints    = train_args.in_timepoints,
        spatial_dims     = (80, 96, 72),
        content_ch       = train_args.content_ch,
        content_base_ch  = train_args.content_base_ch,
        content_n_res    = train_args.content_n_res,
        artefact_base_ch = train_args.artefact_base_ch,
        global_code_dim  = train_args.global_code_dim,
        spatial_code_ch  = train_args.spatial_code_ch,
        disc_base_ch     = train_args.disc_base_ch,
        disc_temporal_diffs = not getattr(train_args, 'no_disc_temporal_diffs', False),
        residual         = getattr(train_args, 'residual', False),
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Loaded checkpoint from epoch {ckpt['epoch']}  "
          f"(best val score: {ckpt['best_score']:.4f})")
    return model, train_args


def save_nifti(
    data:       torch.Tensor,
    orig_shape: Tuple[int, int, int],
    source_path: str,
    out_path:   Path,
) -> None:
    """
    Upsample corrected tensor to original resolution and save as .nii.

    Args:
        data        : (T, X, Y, Z)  corrected chunk in BOLD units
        orig_shape  : (X, Y, Z)  original spatial dims before downsampling
        source_path : path to original NIfTI (used to copy affine/header)
        out_path    : where to save the output .nii file
    """
    upsampled = upsample_to_original(data, orig_shape)   # (T, X, Y, Z)

    # Transpose back to (X, Y, Z, T) for NIfTI convention
    arr = upsampled.cpu().numpy().transpose(1, 2, 3, 0)  # (X, Y, Z, T)

    # Copy affine and header from source so spatial metadata is preserved
    ref    = nib.load(source_path)
    img    = nib.Nifti1Image(arr, affine=ref.affine, header=ref.header)
    nib.save(img, str(out_path))


def select_representative_chunks(
    per_chunk_metrics: List[Dict],
    n_each: int = 3,
) -> Dict[str, List[int]]:
    """
    Select representative chunk indices for figure generation.

    Selects n_each worst, n_each median, n_each best by DVARS improvement.
    This covers the full correction quality spectrum without cherry-picking.

    Args:
        per_chunk_metrics : list of metric dicts, one per chunk
        n_each            : how many from each group

    Returns:
        Dict with keys "worst", "median", "best" containing chunk indices
    """
    improvements = [m["dvars_improvement"] for m in per_chunk_metrics]
    order        = np.argsort(improvements)   # ascending

    n     = len(order)
    mid   = n // 2

    worst  = list(order[:n_each])
    median = list(order[mid - n_each // 2 : mid - n_each // 2 + n_each])
    best   = list(order[-n_each:])

    return {"worst": worst, "median": median, "best": best}


# Figure generation
def plot_violin_comparison(
    per_chunk_metrics: List[Dict],
    out_path: Path,
) -> None:
    """
    Violin plots of input vs corrected distributions for all four metrics.
    One panel per metric, input (red) vs corrected (blue) vs reference (green).
    """
    metrics_to_plot = [
        ("dvars",      "DVARS",                "Lower = better"),
        ("tsnr",       "tSNR",                 "Higher = better"),
        ("gs_std",     "Global Signal Std",    "Lower = better"),
        ("smoothness", "Spatial Smoothness",   "Lower = better"),
    ]

    has_ref = "dvars_reference" in per_chunk_metrics[0]
    n_cols  = len(metrics_to_plot)

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 6))
    fig.suptitle("Test Set Quality Metrics: Input vs Corrected", fontsize=14, y=1.02)

    for ax, (key, label, note) in zip(axes, metrics_to_plot):
        data_in  = [m[f"{key}_input"]     for m in per_chunk_metrics]
        data_cor = [m[f"{key}_corrected"] for m in per_chunk_metrics]

        parts = ax.violinplot([data_in, data_cor],
                              positions=[1, 2],
                              showmedians=True,
                              showextrema=True)

        colors = ["#E07070", "#7090E0"]
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        labels    = ["Input", "Corrected"]
        positions = [1, 2]

        if has_ref:
            data_ref = [m[f"{key}_reference"] for m in per_chunk_metrics]
            ref_parts = ax.violinplot([data_ref], positions=[3], showmedians=True)
            for pc in ref_parts["bodies"]:
                pc.set_facecolor("#70C070")
                pc.set_alpha(0.7)
            labels.append("Reference")
            positions.append(3)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_title(f"{label}\n({note})", fontsize=11)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_scatter_improvement(
    per_chunk_metrics: List[Dict],
    out_path: Path,
) -> None:
    """
    Scatter plots: input metric vs corrected metric, one per metric.
    Points above the diagonal = improvement. Colour = improvement magnitude.
    """
    metrics_to_plot = [
        ("dvars",      "DVARS",             True),    # True = lower is better
        ("tsnr",       "tSNR",              False),   # False = higher is better
        ("gs_std",     "Global Signal Std", True),
        ("smoothness", "Spatial Smoothness",True),
    ]

    fig, axes = plt.subplots(1, len(metrics_to_plot),
                             figsize=(5 * len(metrics_to_plot), 5))
    fig.suptitle("Per-Chunk: Input vs Corrected (above diagonal = improvement)",
                 fontsize=13, y=1.02)

    for ax, (key, label, lower_better) in zip(axes, metrics_to_plot):
        x = np.array([m[f"{key}_input"]     for m in per_chunk_metrics])
        y = np.array([m[f"{key}_corrected"] for m in per_chunk_metrics])

        improvement = x - y if lower_better else y - x
        sc = ax.scatter(x, y, c=improvement, cmap="RdYlGn",
                        alpha=0.6, s=15, vmin=-np.abs(improvement).max(),
                        vmax=np.abs(improvement).max())

        lim = [min(x.min(), y.min()) * 0.95, max(x.max(), y.max()) * 1.05]
        ax.plot(lim, lim, "k--", linewidth=1, alpha=0.5, label="No change")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"Input {label}")
        ax.set_ylabel(f"Corrected {label}")
        ax.set_title(label)
        ax.legend(fontsize=8)
        plt.colorbar(sc, ax=ax, label="Improvement")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_timeseries_comparison(
    chunk_data: Dict,
    out_path:   Path,
    title:      str,
) -> None:
    """
    Global signal timeseries for one chunk: input, corrected, reference.

    Args:
        chunk_data : dict with keys "input", "corrected", optionally "reference"
                     each value is a (T, X, Y, Z) tensor in PSC space
        out_path   : save path
        title      : plot title (e.g. "Best improvement — sub-ICC105 chunk-314")
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    for tag, color, label in [
        ("input",     "#E07070", "Input (corrupted)"),
        ("corrected", "#7090E0", "Corrected"),
        ("reference", "#70C070", "Reference (motion-free)"),
    ]:
        if tag not in chunk_data:
            continue
        x  = chunk_data[tag].cpu()              # (T, X, Y, Z)
        gs = x.mean(dim=(-3, -2, -1)).numpy()   # (T,)
        ax.plot(gs, color=color, label=label, linewidth=1.2)

    ax.set_xlabel("Timepoint")
    ax.set_ylabel("Global Signal (PSC)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_carpet(
    x:        torch.Tensor,
    title:    str,
    out_path: Path,
) -> None:
    """
    Carpet plot (voxels × time) for one chunk.
    Voxels ordered by mean signal intensity (brightest first, approximates GM).

    Args:
        x        : (T, X, Y, Z)  PSC-space tensor
        title    : plot title
        out_path : save path
    """
    data  = x.cpu().numpy()                     # (T, X, Y, Z)
    T     = data.shape[0]
    flat  = data.reshape(T, -1).T               # (N_voxels, T)

    mean_signal = flat.mean(axis=1)
    brain_mask  = mean_signal > (mean_signal.max() * 0.01)
    flat_brain  = flat[brain_mask]               # (N_brain, T)

    order      = np.argsort(flat_brain.mean(axis=1))[::-1]
    flat_brain = flat_brain[order]

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(flat_brain, aspect="auto", interpolation="none",
                   cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_xlabel("Timepoint")
    ax.set_ylabel("Brain voxels (sorted by mean signal)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="PSC", shrink=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_carpet_comparison(
    chunk_data: Dict,
    chunk_name: str,
    fig_dir:    Path,
    group:      str,
) -> None:
    """
    Save carpet plots for input, corrected, and reference for one chunk.

    Args:
        chunk_data : dict with keys "input", "corrected", "reference"
        chunk_name : filename stem used for output naming
        fig_dir    : figures output directory
        group      : "best", "median", or "worst" for subdirectory organisation
    """
    subdir = fig_dir / "carpet_plots" / group
    subdir.mkdir(parents=True, exist_ok=True)

    for tag, label in [
        ("input",     "Input (corrupted)"),
        ("corrected", "Corrected"),
        ("reference", "Reference (motion-free)"),
    ]:
        if tag not in chunk_data:
            continue
        plot_carpet(
            x        = chunk_data[tag],
            title    = f"{label}  |  {chunk_name}",
            out_path = subdir / f"{chunk_name}_{tag}_carpet.png",
        )

def plot_mean_std_maps(
    chunk_data: Dict,
    chunk_name: str,
    fig_dir:    Path,
    group:      str,
) -> None:
    """
    Mean and std volume slices for input vs corrected for one chunk.
    Shows central axial, coronal, sagittal slices.

    Args:
        chunk_data : dict with "input" and "corrected" (T, X, Y, Z) tensors
        chunk_name : used for output filename
        fig_dir    : figures directory
        group      : "best", "median", or "worst"
    """
    subdir = fig_dir / "spatial_maps" / group
    subdir.mkdir(parents=True, exist_ok=True)

    tags = [t for t in ["input", "corrected", "reference"] if t in chunk_data]

    for stat, fn in [("mean", lambda x: x.mean(dim=0)), ("std", lambda x: x.std(dim=0))]:
        fig, axes = plt.subplots(len(tags), 3,
                                 figsize=(12, 4 * len(tags)))
        if len(tags) == 1:
            axes = axes[np.newaxis, :]

        for row, tag in enumerate(tags):
            vol = fn(chunk_data[tag].cpu())   # (X, Y, Z)
            X, Y, Z = vol.shape
            slices  = [
                vol[X // 2, :, :].numpy(),
                vol[:, Y // 2, :].numpy(),
                vol[:, :, Z // 2].numpy(),
            ]
            titles = ["Axial", "Coronal", "Sagittal"]
            vmin, vmax = vol.min().item(), vol.max().item()

            for col, (sl, title) in enumerate(zip(slices, titles)):
                ax = axes[row, col]
                im = ax.imshow(sl.T, origin="lower", cmap="gray",
                               vmin=vmin, vmax=vmax, aspect="auto")
                ax.set_title(f"{tag.capitalize()} {stat}  |  {title}")
                ax.axis("off")
                plt.colorbar(im, ax=ax, shrink=0.7)

        plt.suptitle(f"{stat.capitalize()} volume — {chunk_name}", fontsize=12)
        plt.tight_layout()
        plt.savefig(subdir / f"{chunk_name}_{stat}.png",
                    dpi=300, bbox_inches="tight")
        plt.close()




def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test Disentangled CycleGAN for fMRI motion correction"
    )

    p.add_argument("--checkpoint", type=str,
        default="/lustre/disk/home/users/mfaizan/motion_correction/prototyping/"
                "motion-artefacts-correction/baseline/best_model.pt",
        help="Path to trained checkpoint (best_model.pt or epoch_NNN.pt)")

    p.add_argument("--data_root", type=str,
        default="/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
                "faizan_motion_correction_dataset/cyclegans_dataset",
        help="Path to cyclegans_dataset/ root")

    p.add_argument("--batch_size",  type=int, default=1,
        help="Inference batch size (1 recommended for NIfTI saving)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--n_rep",       type=int, default=3,
        help="Number of chunks per group (worst/median/best) for figures")
    p.add_argument("--no_save_nifti", action="store_true",
        help="Skip saving corrected NIfTI files (useful for quick metric runs)")

    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint)
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"

    # Output directories
    out_root = Path(args.data_root) / "test" / "corrected"
    nifti_dir = out_root / "niftis"
    fig_dir   = out_root / "figures"
    for d in [nifti_dir, fig_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\nCheckpoint  : {checkpoint_path}")
    print(f"Output root : {out_root}")
    print(f"Device      : {device}\n")

    # Load model
    model, _ = load_model(checkpoint_path, device)

    # Build test dataloader
    # Test set has both A_corrupted and B_motion_free so we use val-style loader
    # pointing at the test directory
    test_dir = Path(args.data_root) / "test"

    from dataset import ValFMRIDataset
    test_dataset = ValFMRIDataset(root_dir=str(test_dir))
    test_loader  = torch.utils.data.DataLoader(
        test_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
    )

    print(f"Test chunks : {len(test_dataset)}\n")

    # CSV setup
    metric_keys = [
        "dvars_input",     "dvars_corrected",     "dvars_improvement",
        "tsnr_input",      "tsnr_corrected",       "tsnr_improvement",
        "gs_std_input",    "gs_std_corrected",     "gs_std_improvement",
        "smoothness_input","smoothness_corrected", "smoothness_ratio",
        "dvars_reference", "tsnr_reference",
        "gs_std_reference","smoothness_reference",
    ]
    per_chunk_csv_path = out_root / "metrics_per_chunk.csv"
    per_chunk_rows     = []

    # Storage for representative chunk selection
    all_metrics:    List[Dict]  = []
    all_chunk_data: List[Dict]  = []   # stores PSC tensors for figure chunks
    all_paths:      List[str]   = []

    # Inference loop
    pbar = tqdm(test_loader, desc="Inference", dynamic_ncols=True)

    for batch in pbar:
        x_a      = batch["A"].to(device)          # (B, T, 80, 96, 72)  corrupted
        x_b      = batch["B"].to(device)          # (B, T, 80, 96, 72)  motion-free ref
        mean_A   = batch["mean_vol_A"].to(device)
        orig_A   = batch["orig_shape_A"]
        path_A   = batch["path_A"]

        # Inference — E_c + G_B only
        corrected_psc  = model.correct(x_a)                          # (B, T, 80, 96, 72)
        corrected_bold = psc_denormalise(corrected_psc, mean_A)      # original BOLD units

        # Process each item in batch individually for NIfTI saving
        for i in range(x_a.shape[0]):
            src_path   = path_A[i] if isinstance(path_A, (list, tuple)) else path_A
            chunk_name = Path(src_path).name.replace(".nii.gz", "").replace(".nii", "")
            out_name   = f"{chunk_name}_corrected.nii"

            if isinstance(orig_A[0], torch.Tensor):
                orig_shape = tuple(int(orig_A[j][i].item()) for j in range(3))
            else:
                orig_shape = tuple(int(o) for o in orig_A)

            if not args.no_save_nifti:
                save_nifti(
                    data        = corrected_bold[i],
                    orig_shape  = orig_shape,
                    source_path = src_path,
                    out_path    = nifti_dir / out_name,
                )

            # Compute metrics (PSC space, unbatched)
            metrics = compute_all_metrics(
                x_input     = x_a[i:i+1],
                x_corrected = corrected_psc[i:i+1],
                x_reference = x_b[i:i+1],
            )
            all_metrics.append(metrics)
            all_paths.append(src_path)

            # Store PSC tensors for representative chunk figures (CPU)
            all_chunk_data.append({
                "input":     x_a[i].cpu(),
                "corrected": corrected_psc[i].cpu(),
                "reference": x_b[i].cpu(),
            })

            per_chunk_rows.append({
                "chunk": chunk_name,
                **{k: f"{metrics.get(k, ''):.6f}"
                   if isinstance(metrics.get(k, ""), float) else ""
                   for k in metric_keys},
            })

            pbar.set_postfix({
                "tSNR↑": f"{metrics['tsnr_improvement']:+.3f}",
                "DVARS↓": f"{metrics['dvars_improvement']:+.3f}",
            })

    # Save per-chunk CSV
    with open(per_chunk_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["chunk"] + metric_keys)
        writer.writeheader()
        writer.writerows(per_chunk_rows)
    print(f"\nPer-chunk metrics saved → {per_chunk_csv_path}")

    # Summary statistics CSV
    summary_path = out_root / "metrics_summary.csv"
    with open(summary_path, "w", newline="") as f:
        stat_keys   = ["mean", "std", "median", "p25", "p75"]
        fieldnames  = ["metric"] + stat_keys
        writer      = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in metric_keys:
            vals = [m[key] for m in all_metrics if key in m]
            if not vals:
                continue
            stats = summary_stats(vals)
            writer.writerow({"metric": key, **{k: f"{stats[k]:.6f}" for k in stat_keys}})

        # Print summary to console
        print("\nSummary statistics:")
        print(f"  {'Metric':<30} {'Mean':>10} {'Std':>10} {'Median':>10}")
        for key in ["dvars_improvement", "tsnr_improvement",
                    "gs_std_improvement", "smoothness_ratio"]:
            vals = [m[key] for m in all_metrics if key in m]
            if not vals:
                continue
            stats = summary_stats(vals)
            print(f"  {key:<30} {stats['mean']:>10.4f} "
                  f"{stats['std']:>10.4f} {stats['median']:>10.4f}")

    print(f"Summary metrics saved → {summary_path}")

    # Representative chunk selection
    groups = select_representative_chunks(all_metrics, n_each=args.n_rep)

    print(f"\nGenerating figures ...")

    # Violin plots — all chunks
    plot_violin_comparison(
        per_chunk_metrics = all_metrics,
        out_path          = fig_dir / "violin_comparison.png",
    )
    print(f"  Violin plots saved")

    # Scatter plots — all chunks
    plot_scatter_improvement(
        per_chunk_metrics = all_metrics,
        out_path          = fig_dir / "scatter_improvement.png",
    )
    print(f"  Scatter plots saved")

    # Per-chunk figures for representative chunks
    for group_name, indices in groups.items():
        for idx in indices:
            chunk_name = Path(all_paths[idx]).name.replace(".nii.gz","").replace(".nii","")
            chunk_data = all_chunk_data[idx]
            metrics    = all_metrics[idx]

            # Timeseries comparison
            plot_timeseries_comparison(
                chunk_data = chunk_data,
                out_path   = fig_dir / f"timeseries_{group_name}_{chunk_name}.png",
                title      = (f"{group_name.capitalize()} DVARS improvement "
                              f"({metrics['dvars_improvement']:+.3f})  |  {chunk_name}"),
            )

            # Carpet plots
            plot_carpet_comparison(
                chunk_data = chunk_data,
                chunk_name = chunk_name,
                fig_dir    = fig_dir,
                group      = group_name,
            )

            # Spatial mean/std maps
            plot_mean_std_maps(
                chunk_data = chunk_data,
                chunk_name = chunk_name,
                fig_dir    = fig_dir,
                group      = group_name,
            )

    print(f"  Representative chunk figures saved → {fig_dir}")
    print(f"\nTest complete. All outputs in: {out_root}")


if __name__ == "__main__":
    main()