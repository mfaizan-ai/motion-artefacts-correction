#!/usr/bin/env python3
"""
visualize_dataloader_qc.py
==========================
Visual QC of the data loading pipeline.

For a few random training chunks, shows side-by-side:
    Row 1: Raw BOLD (after spatial resample, before PSC)
    Row 2: PSC-normalised
    Row 3: Temporal mean volume
    Row 4: Brain mask used for PSC

Each column is a different timepoint from the chunk.
Three orthogonal slices (axial, coronal, sagittal) through the volume center.

Produces one figure per chunk, for both corrupted (A) and clean (B) domains.
"""

import argparse
import os
import random
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

TARGET_SPATIAL = (80, 96, 72)
_PSC_EPS = 1e-6


def spatial_resample(x, target):
    x = x.unsqueeze(0)
    x = F.interpolate(x, size=target, mode="trilinear", align_corners=False)
    return x.squeeze(0)


def load_nifti_raw(path):
    img = nib.load(path, mmap=True)
    data = np.asarray(img.dataobj, dtype=np.float32)
    data = data.transpose(3, 0, 1, 2)
    return torch.from_numpy(data)


def psc_normalise(x):
    mean_vol = x.mean(dim=0, keepdim=True)
    brain_mask = mean_vol > (mean_vol.max() * 0.01)
    psc = (x - mean_vol) / (mean_vol.abs() + _PSC_EPS)
    psc = psc * brain_mask
    return psc, mean_vol, brain_mask


def plot_chunk(raw, resampled, psc, mean_vol, brain_mask, title, out_path,
               n_timepoints=3):
    T = resampled.shape[0]
    vol_indices = sorted(random.sample(range(T), min(n_timepoints, T)))

    mid_x = resampled.shape[1] // 2
    mid_y = resampled.shape[2] // 2
    mid_z = resampled.shape[3] // 2

    n_cols = len(vol_indices)
    n_rows = 5  # raw, resampled, PSC, mean_vol, brain_mask
    fig, axes = plt.subplots(n_rows, n_cols * 3, figsize=(4 * n_cols * 3, 4 * n_rows))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)

    row_labels = [
        "Raw BOLD\n(original resolution)",
        "Resampled BOLD\n(80×96×72)",
        "PSC Normalised",
        "Temporal Mean",
        "Brain Mask",
    ]

    slice_labels = ["Sagittal", "Coronal", "Axial"]

    for col_i, t in enumerate(vol_indices):
        raw_vol = raw[t].numpy()
        res_vol = resampled[t].numpy()
        psc_vol = psc[t].numpy()
        mean_v = mean_vol[0].numpy()
        mask_v = brain_mask[0].float().numpy()

        raw_mid_x = raw_vol.shape[0] // 2
        raw_mid_y = raw_vol.shape[1] // 2
        raw_mid_z = raw_vol.shape[2] // 2
        raw_slices = [
            raw_vol[raw_mid_x, :, :],
            raw_vol[:, raw_mid_y, :],
            raw_vol[:, :, raw_mid_z],
        ]

        res_slices = [res_vol[mid_x, :, :], res_vol[:, mid_y, :], res_vol[:, :, mid_z]]
        psc_slices = [psc_vol[mid_x, :, :], psc_vol[:, mid_y, :], psc_vol[:, :, mid_z]]
        mean_slices = [mean_v[mid_x, :, :], mean_v[:, mid_y, :], mean_v[:, :, mid_z]]
        mask_slices = [mask_v[mid_x, :, :], mask_v[:, mid_y, :], mask_v[:, :, mid_z]]

        all_slices = [raw_slices, res_slices, psc_slices, mean_slices, mask_slices]
        cmaps = ["gray", "gray", "RdBu_r", "gray", "Reds"]

        for row_i, (slices, cmap) in enumerate(zip(all_slices, cmaps)):
            for s_i, sl in enumerate(slices):
                ax_col = col_i * 3 + s_i
                ax = axes[row_i, ax_col]

                if row_i == 2:  # PSC: symmetric colormap
                    vmax = max(abs(np.nanpercentile(sl, 1)),
                               abs(np.nanpercentile(sl, 99)), 0.01)
                    im = ax.imshow(sl.T, origin="lower", cmap=cmap,
                                   vmin=-vmax, vmax=vmax, aspect="auto")
                elif row_i == 4:  # mask: binary
                    im = ax.imshow(sl.T, origin="lower", cmap=cmap,
                                   vmin=0, vmax=1, aspect="auto")
                else:
                    im = ax.imshow(sl.T, origin="lower", cmap=cmap, aspect="auto")

                div = make_axes_locatable(ax)
                cax = div.append_axes("right", size="5%", pad=0.02)
                plt.colorbar(im, cax=cax, format="%.1f")

                ax.set_xticks([])
                ax.set_yticks([])

                if row_i == 0:
                    ax.set_title(f"t={t}  {slice_labels[s_i]}", fontsize=9)

                if s_i == 0 and col_i == 0:
                    ax.set_ylabel(row_labels[row_i], fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_intensity_histograms(raw, resampled, psc, title, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{title} — Intensity Distributions", fontsize=13,
                 fontweight="bold")

    raw_vals = raw.numpy().flatten()
    raw_vals = raw_vals[raw_vals > 0]
    axes[0].hist(raw_vals, bins=100, color="steelblue", alpha=0.8, edgecolor="none")
    axes[0].set_title(f"Raw BOLD (original)\nrange: [{raw_vals.min():.1f}, "
                      f"{raw_vals.max():.1f}]")
    axes[0].set_xlabel("Intensity")
    axes[0].set_ylabel("Count")
    axes[0].axvline(raw_vals.mean(), color="red", ls="--", label=f"mean={raw_vals.mean():.1f}")
    axes[0].legend(fontsize=8)

    res_vals = resampled.numpy().flatten()
    res_vals = res_vals[res_vals > 0]
    axes[1].hist(res_vals, bins=100, color="darkorange", alpha=0.8, edgecolor="none")
    axes[1].set_title(f"Resampled (80×96×72)\nrange: [{res_vals.min():.1f}, "
                      f"{res_vals.max():.1f}]")
    axes[1].set_xlabel("Intensity")
    axes[1].axvline(res_vals.mean(), color="red", ls="--", label=f"mean={res_vals.mean():.1f}")
    axes[1].legend(fontsize=8)

    psc_vals = psc.numpy().flatten()
    psc_vals = psc_vals[psc_vals != 0]
    axes[2].hist(psc_vals, bins=100, color="seagreen", alpha=0.8, edgecolor="none")
    axes[2].set_title(f"PSC Normalised\nrange: [{psc_vals.min():.4f}, "
                      f"{psc_vals.max():.4f}]")
    axes[2].set_xlabel("PSC value")
    axes[2].axvline(psc_vals.mean(), color="red", ls="--", label=f"mean={psc_vals.mean():.4f}")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_temporal_profile(raw, resampled, psc, mean_vol, brain_mask, title, out_path):
    mask = brain_mask[0].bool()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"{title} — Temporal Profiles", fontsize=13, fontweight="bold")

    T = resampled.shape[0]
    t_axis = np.arange(T)

    gs_raw = resampled[:, mask].mean(dim=1).numpy()
    axes[0, 0].plot(t_axis, gs_raw, "o-", color="steelblue", markersize=4)
    axes[0, 0].set_title("Global Signal (resampled BOLD)")
    axes[0, 0].set_ylabel("Mean intensity")
    axes[0, 0].grid(alpha=0.3)

    gs_psc = psc[:, mask].mean(dim=1).numpy()
    axes[0, 1].plot(t_axis, gs_psc, "o-", color="seagreen", markersize=4)
    axes[0, 1].set_title("Global Signal (PSC)")
    axes[0, 1].set_ylabel("Mean PSC")
    axes[0, 1].axhline(0, color="gray", ls="--", alpha=0.5)
    axes[0, 1].grid(alpha=0.3)

    if T > 1:
        diff = torch.diff(resampled, dim=0)
        dvars = diff[:, mask].pow(2).mean(dim=1).sqrt().numpy()
        axes[1, 0].plot(np.arange(1, T), dvars, "o-", color="darkorange", markersize=4)
        axes[1, 0].set_title("DVARS (frame-to-frame change)")
        axes[1, 0].set_ylabel("DVARS")
        axes[1, 0].set_xlabel("Timepoint")
        axes[1, 0].grid(alpha=0.3)

    mid_x = resampled.shape[1] // 2
    mid_y = resampled.shape[2] // 2
    mid_z = resampled.shape[3] // 2
    voxel_ts = resampled[:, mid_x, mid_y, mid_z].numpy()
    voxel_psc = psc[:, mid_x, mid_y, mid_z].numpy()
    axes[1, 1].plot(t_axis, voxel_ts, "o-", color="steelblue", markersize=4,
                    label="BOLD")
    ax2 = axes[1, 1].twinx()
    ax2.plot(t_axis, voxel_psc, "s--", color="seagreen", markersize=4,
             label="PSC", alpha=0.8)
    axes[1, 1].set_title(f"Center voxel ({mid_x},{mid_y},{mid_z})")
    axes[1, 1].set_xlabel("Timepoint")
    axes[1, 1].set_ylabel("BOLD intensity", color="steelblue")
    ax2.set_ylabel("PSC", color="seagreen")
    axes[1, 1].grid(alpha=0.3)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlabel("Timepoint")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def process_one_chunk(nifti_path, label, out_dir, chunk_idx):
    print(f"\n--- {label} chunk {chunk_idx}: {Path(nifti_path).name} ---")

    raw = load_nifti_raw(nifti_path)
    orig_shape = (raw.shape[1], raw.shape[2], raw.shape[3])
    print(f"  Original shape: {raw.shape} (T, X, Y, Z)")

    if orig_shape != TARGET_SPATIAL:
        resampled = spatial_resample(raw, TARGET_SPATIAL)
    else:
        resampled = raw.clone()
    print(f"  Resampled shape: {resampled.shape}")

    psc, mean_vol, brain_mask = psc_normalise(resampled)
    brain_pct = brain_mask.float().mean().item() * 100
    print(f"  Brain mask coverage: {brain_pct:.1f}%")
    print(f"  Raw intensity range: [{raw.min():.1f}, {raw.max():.1f}]")
    print(f"  PSC range: [{psc.min():.4f}, {psc.max():.4f}]")

    prefix = f"{label}_chunk{chunk_idx}"

    plot_chunk(raw, resampled, psc, mean_vol, brain_mask,
              f"{label} — {Path(nifti_path).name}\n"
              f"orig={tuple(raw.shape)} → resampled={tuple(resampled.shape)}",
              out_dir / f"{prefix}_slices.png")

    plot_intensity_histograms(raw, resampled, psc,
                              f"{label} — {Path(nifti_path).name}",
                              out_dir / f"{prefix}_histograms.png")

    plot_temporal_profile(raw, resampled, psc, mean_vol, brain_mask,
                          f"{label} — {Path(nifti_path).name}",
                          out_dir / f"{prefix}_temporal.png")


def main():
    parser = argparse.ArgumentParser(
        description="Visual QC of data loading pipeline")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to cyclegans dataset root")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--n_chunks", type=int, default=3,
                        help="Number of random chunks to visualize per domain")
    parser.add_argument("--output_dir", type=str, default="dataloader_visual_qc")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    a_dir = Path(args.data_root) / args.split / "A_corrupted"
    b_dir = Path(args.data_root) / args.split / "B_motion_free"

    a_files = sorted([str(f) for f in a_dir.glob("*.nii.gz")])
    b_files = sorted([str(f) for f in b_dir.glob("*.nii.gz")])

    print(f"Dataset: {args.data_root}")
    print(f"Split: {args.split}")
    print(f"A (corrupted): {len(a_files)} files")
    print(f"B (motion-free): {len(b_files)} files")
    print(f"Output: {out_dir}")

    a_sample = random.sample(a_files, min(args.n_chunks, len(a_files)))
    b_sample = random.sample(b_files, min(args.n_chunks, len(b_files)))

    for i, path in enumerate(a_sample):
        process_one_chunk(path, "corrupted", out_dir, i)

    for i, path in enumerate(b_sample):
        process_one_chunk(path, "motion_free", out_dir, i)

    print(f"\nDone — all plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
