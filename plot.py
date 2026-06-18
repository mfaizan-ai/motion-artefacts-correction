#!/usr/bin/env python3
"""
plot.py
============
Plot training and validation logs from CycleGAN fMRI motion correction.

Usage:
    python plot.py --mode train --train_csv path/to/train_losses.csv
    python plot.py --mode val   --val_csv   path/to/val_metrics.csv
    python plot.py --mode both  --train_csv path/to/train_losses.csv \
                                     --val_csv   path/to/val_metrics.csv
"""

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns


# Global style
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = {
    "input":     "#D95F5F",
    "corrected": "#4C72B0",
    "reference": "#55A868",
    "gen":       "#4C72B0",
    "disc_a":    "#D95F5F",
    "disc_b":    "#E07A3A",
    "real":      "#2E8B57",
    "fake":      "#C0392B",
    "score":     "#8E44AD",
    "lr":        "#2980B9",
    "grad":      "#E67E22",
    "neutral":   "#555555",
}
SMOOTH_WINDOW = 10   # rolling mean window for noisy training curves


def smooth(values: pd.Series, window: int = SMOOTH_WINDOW) -> pd.Series:
    """Rolling mean with min_periods=1 so edges are not NaN."""
    return values.rolling(window=window, min_periods=1, center=True).mean()


def save(fig: plt.Figure, path: Path, name: str) -> None:
    """Save figure as PNG at 300 Dpi and close it."""
    out = path / name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.name}")


def annotate_best(ax: plt.Axes, x: np.ndarray, y: np.ndarray,
                  higher_better: bool = True) -> None:
    """Mark the best epoch with a gold star annotation."""
    idx  = int(np.argmax(y) if higher_better else np.argmin(y))
    best = y[idx]
    ax.plot(x[idx], best, marker="*", color="gold",
            markersize=14, zorder=5, label=f"Best ({best:.4f})")


def add_improvement_arrow(ax: plt.Axes, higher_better: bool) -> None:
    """Add a small text arrow in the top corner showing improvement direction."""
    text = "↑ better" if higher_better else "↓ better"
    ax.annotate(text, xy=(0.98, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=9,
                color=PALETTE["neutral"],
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6))


def plot_generator_losses(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Six-panel figure showing each generator loss term plus total.
    Raw values plotted faintly, smoothed curve on top.
    """
    terms = [
        ("G_adv",     "Adversarial",       False),
        ("G_cyc",     "Cycle Consistency", False),
        ("G_idt",     "Identity",          False),
        ("G_content", "Content",           False),
        ("G_art",     "Artefact Suppression", False),
        ("G_total",   "Total Generator",   False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes      = axes.flatten()
    fig.suptitle("Generator Losses", fontsize=15, fontweight="bold", y=1.01)

    for ax, (col, label, higher) in zip(axes, terms):
        if col not in df.columns:
            ax.set_visible(False)
            continue

        epochs = df["epoch"].values
        raw    = df[col].values
        s      = smooth(df[col])

        color = PALETTE["gen"] if col != "G_total" else PALETTE["neutral"]

        ax.plot(epochs, raw, color=color, alpha=0.2, linewidth=0.8)
        ax.plot(epochs, s,   color=color, linewidth=2.0, label="Smoothed")
        ax.set_title(label, fontweight="semibold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
        add_improvement_arrow(ax, higher_better=False)

    plt.tight_layout()
    save(fig, out_dir, "01_generator_losses.png")


def plot_discriminator(df: pd.DataFrame, out_dir: Path) -> None:
    """
    2x2 figure: D_A and D_B losses, then real vs fake scores per domain.
    GAN equilibrium is visible when real and fake scores converge to 0.5.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Discriminator Losses and Scores", fontsize=15,
                 fontweight="bold", y=1.01)

    epochs = df["epoch"].values

    # D_A loss
    ax = axes[0, 0]
    ax.plot(epochs, df["D_A"], color=PALETTE["disc_a"], alpha=0.2, linewidth=0.8)
    ax.plot(epochs, smooth(df["D_A"]), color=PALETTE["disc_a"],
            linewidth=2, label="D_A")
    ax.set_title("Discriminator A Loss (Corrupted Domain)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    add_improvement_arrow(ax, higher_better=False)

    # D_B loss
    ax = axes[0, 1]
    ax.plot(epochs, df["D_B"], color=PALETTE["disc_b"], alpha=0.2, linewidth=0.8)
    ax.plot(epochs, smooth(df["D_B"]), color=PALETTE["disc_b"],
            linewidth=2, label="D_B")
    ax.set_title("Discriminator B Loss (Motion-free Domain)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    add_improvement_arrow(ax, higher_better=False)

    # Domain A scores: real vs fake
    ax = axes[1, 0]
    ax.plot(epochs, df["score_real_a"], color=PALETTE["real"],
            alpha=0.2, linewidth=0.8)
    ax.plot(epochs, smooth(df["score_real_a"]), color=PALETTE["real"],
            linewidth=2, label="Real A")
    ax.plot(epochs, df["score_fake_a"], color=PALETTE["fake"],
            alpha=0.2, linewidth=0.8)
    ax.plot(epochs, smooth(df["score_fake_a"]), color=PALETTE["fake"],
            linewidth=2, label="Fake A", linestyle="--")
    ax.axhline(0.5, color=PALETTE["neutral"], linewidth=1,
               linestyle=":", label="Equilibrium (0.5)")
    ax.set_title("D_A Scores  (Corrupted Domain)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Discriminator Output")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)

    # Domain B scores: real vs fake
    ax = axes[1, 1]
    ax.plot(epochs, df["score_real_b"], color=PALETTE["real"],
            alpha=0.2, linewidth=0.8)
    ax.plot(epochs, smooth(df["score_real_b"]), color=PALETTE["real"],
            linewidth=2, label="Real B")
    ax.plot(epochs, df["score_fake_b"], color=PALETTE["fake"],
            alpha=0.2, linewidth=0.8)
    ax.plot(epochs, smooth(df["score_fake_b"]), color=PALETTE["fake"],
            linewidth=2, label="Fake B", linestyle="--")
    ax.axhline(0.5, color=PALETTE["neutral"], linewidth=1,
               linestyle=":", label="Equilibrium (0.5)")
    ax.set_title("D_B Scores  (Motion-free Domain)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Discriminator Output")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)

    for ax in axes.flatten():
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))

    plt.tight_layout()
    save(fig, out_dir, "02_discriminator.png")


def plot_grad_lr(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Two-panel figure: gradient norm trajectory and learning rate schedule.
    Gradient norm above clip threshold (1.0) highlighted in red.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Gradient Norm and Learning Rate", fontsize=15,
                 fontweight="bold")

    epochs = df["epoch"].values

    # Gradient norm
    ax = axes[0]
    raw = df["grad_norm_G"].values
    s   = smooth(df["grad_norm_G"])

    ax.plot(epochs, raw, color=PALETTE["grad"], alpha=0.2, linewidth=0.8)
    ax.plot(epochs, s,   color=PALETTE["grad"], linewidth=2, label="∇G (smoothed)")
    ax.axhline(1.0, color=PALETTE["fake"], linewidth=1.2,
               linestyle="--", label="Clip threshold (1.0)")
    ax.fill_between(epochs, 1.0, np.maximum(raw, 1.0),
                    color=PALETTE["fake"], alpha=0.15, label="Clipped region")
    ax.set_title("Generator Gradient Norm")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 Norm (pre-clip)")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    # Learning rate schedule
    ax = axes[1]
    ax.plot(epochs, df["lr_G"], color=PALETTE["lr"],
            linewidth=2, label="lr_G")
    if "lr_D" in df.columns:
        ax.plot(epochs, df["lr_D"], color=PALETTE["disc_a"],
                linewidth=2, linestyle="--", label="lr_D")
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.legend(fontsize=8)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    for ax in axes:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))

    plt.tight_layout()
    save(fig, out_dir, "03_grad_lr.png")


def plot_val_losses(df_val: pd.DataFrame,
                    df_train: Optional[pd.DataFrame],
                    out_dir: Path) -> None:
    """
    Validation losses, optionally overlaid with training equivalents.
    Useful for spotting overfitting — val and train diverging is the signal.
    """
    terms = ["cyc", "idt", "content", "art"]
    labels = {
        "cyc":     "Cycle Consistency",
        "idt":     "Identity",
        "content": "Content",
        "art":     "Artefact Suppression",
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Validation Losses" +
                 (" vs Training Losses" if df_train is not None else ""),
                 fontsize=15, fontweight="bold")

    for ax, term in zip(axes, terms):
        val_col = f"val_{term}"
        if val_col not in df_val.columns:
            ax.set_visible(False)
            continue

        ax.plot(df_val["epoch"], df_val[val_col],
                color=PALETTE["corrected"], linewidth=2,
                marker="o", markersize=4, label="Val")

        if df_train is not None and term in df_train.columns:
            ax.plot(df_train["epoch"], smooth(df_train[term]),
                    color=PALETTE["input"], linewidth=1.5,
                    linestyle="--", alpha=0.8, label="Train (smoothed)")

        ax.set_title(labels[term], fontweight="semibold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        add_improvement_arrow(ax, higher_better=False)

    plt.tight_layout()
    save(fig, out_dir, "04_val_losses.png")


def plot_fmri_metrics(df: pd.DataFrame, out_dir: Path) -> None:
    """
    2x2 figure: one panel per fMRI metric (DVARS, tSNR, GS std, smoothness).
    Input and corrected lines per panel with shaded region between them.
    """
    metrics = [
        ("dvars",      "DVARS",              True,  "Lower = fewer motion spikes"),
        ("tsnr",       "tSNR",               False, "Higher = cleaner signal"),
        ("gs_std",     "Global Signal Std",  True,  "Lower = more stable"),
        ("smoothness", "Spatial Smoothness", True,  "Ratio: corrected / input"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes      = axes.flatten()
    fig.suptitle("Validation fMRI Quality Metrics", fontsize=15,
                 fontweight="bold", y=1.01)

    epochs = df["epoch"].values

    for ax, (key, label, lower_better, note) in zip(axes, metrics):
        if key == "smoothness":
            col_in  = f"val_{key}_input"
            col_cor = f"val_{key}_corrected"
        else:
            col_in  = f"val_{key}_input"
            col_cor = f"val_{key}_corrected"

        if col_in not in df.columns or col_cor not in df.columns:
            ax.set_visible(False)
            continue

        y_in  = df[col_in].values
        y_cor = df[col_cor].values

        ax.plot(epochs, y_in,  color=PALETTE["input"],
                linewidth=2, marker="o", markersize=4, label="Input (corrupted)")
        ax.plot(epochs, y_cor, color=PALETTE["corrected"],
                linewidth=2, marker="o", markersize=4, label="Corrected")

        # Shade region between input and corrected
        better_mask = y_in > y_cor if lower_better else y_cor > y_in
        ax.fill_between(epochs, y_in, y_cor,
                        where=better_mask,
                        color=PALETTE["reference"], alpha=0.2,
                        label="Improvement region")
        ax.fill_between(epochs, y_in, y_cor,
                        where=~better_mask,
                        color=PALETTE["fake"], alpha=0.15,
                        label="Regression region")

        ref_col = f"val_{key}_reference"
        if ref_col in df.columns:
            ax.plot(epochs, df[ref_col].values,
                    color=PALETTE["reference"], linewidth=1.5,
                    linestyle=":", label="Reference (motion-free)")

        ax.set_title(f"{label}\n{note}", fontweight="semibold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
        add_improvement_arrow(ax, higher_better=not lower_better)

    plt.tight_layout()
    save(fig, out_dir, "05_val_fmri_metrics.png")


def plot_improvement_trajectory(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Single figure showing all improvement metrics over val epochs.
    Primary y-axis: DVARS, tSNR, GS std improvements.
    Secondary y-axis: smoothness ratio with warning threshold at 1.1.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.suptitle("Validation Improvement Trajectory", fontsize=15,
                 fontweight="bold")

    epochs = df["epoch"].values

    improvements = [
        ("val_dvars_improvement",   "DVARS improvement",    PALETTE["disc_a"]),
        ("val_tsnr_improvement",    "tSNR improvement",     PALETTE["corrected"]),
        ("val_gs_std_improvement",  "GS std improvement",   PALETTE["reference"]),
    ]

    for col, label, color in improvements:
        if col not in df.columns:
            continue
        ax1.plot(epochs, df[col], color=color, linewidth=2,
                 marker="o", markersize=4, label=label)

    ax1.axhline(0, color=PALETTE["neutral"], linewidth=1,
                linestyle="--", alpha=0.6, label="No change (0)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Improvement (positive = better)")
    ax1.legend(loc="upper left", fontsize=9)

    # Smoothness ratio on secondary axis
    if "val_smoothness_ratio" in df.columns:
        ax2 = ax1.twinx()
        ax2.plot(epochs, df["val_smoothness_ratio"],
                 color=PALETTE["score"], linewidth=2,
                 marker="s", markersize=4, linestyle="-.",
                 label="Smoothness ratio")
        ax2.axhline(1.0, color=PALETTE["score"], linewidth=0.8,
                    linestyle=":", alpha=0.5)
        ax2.axhline(1.1, color=PALETTE["fake"], linewidth=1.2,
                    linestyle="--", label="Over-smooth threshold (1.1)")
        ax2.fill_between(epochs, 1.1,
                         np.maximum(df["val_smoothness_ratio"].values, 1.1),
                         color=PALETTE["fake"], alpha=0.1)
        ax2.set_ylabel("Smoothness Ratio (corrected / input)",
                       color=PALETTE["score"])
        ax2.tick_params(axis="y", labelcolor=PALETTE["score"])
        ax2.legend(loc="upper right", fontsize=9)

    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))
    plt.tight_layout()
    save(fig, out_dir, "06_improvement_trajectory.png")


def plot_val_score(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Composite validation score over val epochs with best epoch marked.
    This is the single number used for best checkpoint selection.
    """
    if "val_score" not in df.columns:
        print("  val_score column not found, skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Composite Validation Score", fontsize=15, fontweight="bold")

    epochs = df["epoch"].values
    scores = df["val_score"].values

    ax.plot(epochs, scores, color=PALETTE["score"], linewidth=2,
            marker="o", markersize=5, label="Val score")
    ax.axhline(0, color=PALETTE["neutral"], linewidth=1,
               linestyle="--", alpha=0.5)

    annotate_best(ax, epochs, scores, higher_better=True)

    best_idx   = int(np.argmax(scores))
    best_epoch = epochs[best_idx]
    ax.axvline(best_epoch, color="gold", linewidth=1.5,
               linestyle="--", alpha=0.7)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score  (higher = better)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))

    plt.tight_layout()
    save(fig, out_dir, "07_val_score.png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot training and validation logs for CycleGAN fMRI correction"
    )
    p.add_argument("--mode", type=str, choices=["train", "val", "both"],
        default="both",
        help="Which logs to plot: train, val, or both")
    p.add_argument("--train_csv", type=str, default=None,
        help="Path to train_losses.csv")
    p.add_argument("--val_csv", type=str, default=None,
        help="Path to val_metrics.csv")
    p.add_argument("--out_dir", type=str, default=None,
        help="Output directory for plots (default: same directory as CSV)")
    p.add_argument("--smooth_window", type=int, default=SMOOTH_WINDOW,
        help="Rolling mean window for training curve smoothing")
    return p.parse_args()


def resolve_out_dir(args: argparse.Namespace) -> Path:
    """
    Determine output directory. Uses --out_dir if given, otherwise
    creates a plots/ subdirectory next to whichever CSV was provided.
    """
    if args.out_dir is not None:
        out = Path(args.out_dir)
    elif args.train_csv is not None:
        out = Path(args.train_csv).parent / "plots"
    elif args.val_csv is not None:
        out = Path(args.val_csv).parent / "plots"
    else:
        raise ValueError("Provide at least one of --train_csv or --val_csv")
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    args = parse_args()

    global SMOOTH_WINDOW
    SMOOTH_WINDOW = args.smooth_window

    out_dir  = resolve_out_dir(args)
    df_train = None
    df_val   = None

    if args.mode in ("train", "both"):
        if args.train_csv is None:
            raise ValueError("--train_csv required for mode train or both")
        df_train = pd.read_csv(args.train_csv)
        print(f"Loaded train CSV: {args.train_csv}  ({len(df_train)} epochs)")

    if args.mode in ("val", "both"):
        if args.val_csv is None:
            raise ValueError("--val_csv required for mode val or both")
        df_val = pd.read_csv(args.val_csv)
        print(f"Loaded val CSV  : {args.val_csv}  ({len(df_val)} val checkpoints)")

    print(f"Output directory: {out_dir}\n")

    if df_train is not None:
        print("Plotting training figures ...")
        plot_generator_losses(df_train, out_dir)
        plot_discriminator(df_train, out_dir)
        plot_grad_lr(df_train, out_dir)

    if df_val is not None:
        print("Plotting validation figures ...")
        plot_val_losses(df_val, df_train, out_dir)
        plot_fmri_metrics(df_val, out_dir)
        plot_improvement_trajectory(df_val, out_dir)
        plot_val_score(df_val, out_dir)

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()