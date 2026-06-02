#!/usr/bin/env python3
"""
plot_fd_dvars_relationship.py
==============================
Plot the relationship between chunk-level mean FD and mean DVARS
for high-motion (corrupted) and low-motion (motion-free) chunks.

Reads the chunk_level_metrics.csv files produced by qc_chunk_analysis.py
and produces a publication-quality scatter plot showing that FD-based
selection corresponds to actual image instability (DVARS).

Usage
-----
    python plot_fd_dvars_relationship.py \\
        --corrupted_csv   qc_results/train_qc/analysis_output/corrupted/train_chunk_level_metrics.csv \\
        --motfree_csv     qc_results/train_qc/analysis_output/motion_free/train_chunk_level_metrics.csv \\
        --output_dir      qc_results/train_qc/plots \\
        --split           train

    # Or point directly at the qc_results root and let the script find the files
    python plot_fd_dvars_relationship.py \\
        --qc_dir    qc_results/train_qc \\
        --split     train
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =============================================================================
# LOGGING
# =============================================================================
def setup_logging() -> logging.Logger:
    log = logging.getLogger("fd_dvars_plot")
    log.setLevel(logging.INFO)
    if not log.handlers:
        fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                                datefmt="%H:%M:%S")
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        log.addHandler(ch)
    return log

log = setup_logging()

# =============================================================================
# CONSTANTS
# =============================================================================
COLOR_CORRUPTED  = "#E05C5C"
COLOR_MOTFREE    = "#5C8AE0"
ALPHA_SCATTER    = 0.35
POINT_SIZE       = 18
POINT_SIZE_LARGE = 22


# =============================================================================
# ARGS
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot FD vs DVARS relationship for QC validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Convenience: point at qc_results root
    parser.add_argument("--qc_dir", type=Path, default=None,
                        help="Root QC output dir (e.g. qc_results/train_qc). "
                             "If set, CSV paths are inferred automatically.")
    # Or specify CSVs explicitly
    parser.add_argument("--corrupted_csv", type=Path, default=None,
                        help="Path to corrupted chunk_level_metrics.csv")
    parser.add_argument("--motfree_csv", type=Path, default=None,
                        help="Path to motion_free chunk_level_metrics.csv")

    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Where to save plots. Defaults to <qc_dir>/plots.")
    parser.add_argument("--split", type=str, default="train",
                        help="Split name (used in plot titles and filenames).")
    parser.add_argument("--fd_col", type=str, default="par_fd_mean",
                        choices=["par_fd_mean", "csv_mean_fd"],
                        help="Which FD column to use.")
    parser.add_argument("--max_fd_clip", type=float, default=None,
                        help="Clip FD axis at this value for readability "
                             "(e.g. 5.0). Default: no clipping.")
    parser.add_argument("--max_dvars_clip", type=float, default=None,
                        help="Clip DVARS axis at this value. Default: no clipping.")
    return parser.parse_args()


# =============================================================================
# LOAD DATA
# =============================================================================
def resolve_paths(args) -> tuple[Path, Path]:
    if args.qc_dir is not None:
        base = args.qc_dir / "analysis_output"
        c_path  = base / "corrupted"   / f"{args.split}_chunk_level_metrics.csv"
        mf_path = base / "motion_free" / f"{args.split}_chunk_level_metrics.csv"
    elif args.corrupted_csv and args.motfree_csv:
        c_path  = args.corrupted_csv
        mf_path = args.motfree_csv
    else:
        log.error("Provide either --qc_dir OR both --corrupted_csv and --motfree_csv")
        sys.exit(1)

    for p in [c_path, mf_path]:
        if not p.exists():
            log.error(f"CSV not found: {p}")
            sys.exit(1)

    return c_path, mf_path


def load_and_clean(path: Path, domain: str,
                   fd_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["domain"] = domain

    required = [fd_col, "dvars_mean"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        log.error(f"[{domain}] Missing columns: {missing}")
        sys.exit(1)

    before = len(df)
    df = df.dropna(subset=required)
    df = df[np.isfinite(df[fd_col]) & np.isfinite(df["dvars_mean"])]
    dropped = before - len(df)
    if dropped:
        log.warning(f"  [{domain}] Dropped {dropped} rows with NaN/Inf in "
                    f"{fd_col} or dvars_mean")

    log.info(f"  [{domain}] {len(df)} valid chunks  "
             f"FD: {df[fd_col].mean():.3f} ± {df[fd_col].std():.3f}  "
             f"DVARS: {df['dvars_mean'].mean():.3f} ± {df['dvars_mean'].std():.3f}")
    return df


# =============================================================================
# REGRESSION LINE + STATS
# =============================================================================
def regression_stats(x: np.ndarray,
                     y: np.ndarray) -> tuple[float, float, float, float, float]:
    """Returns slope, intercept, r, p, r²."""
    slope, intercept, r, p, _ = stats.linregress(x, y)
    return slope, intercept, r, p, r ** 2


def regression_line(ax, x: np.ndarray, y: np.ndarray,
                    color: str, label: str = ""):
    slope, intercept, r, p, r2 = regression_stats(x, y)
    x_range = np.linspace(x.min(), x.max(), 200)
    y_fit   = slope * x_range + intercept

    ax.plot(x_range, y_fit, color=color, linewidth=2.0,
            linestyle="--", alpha=0.9, zorder=5)

    p_str = f"p<0.001" if p < 0.001 else f"p={p:.3f}"
    return f"r²={r2:.3f}, {p_str}"


# =============================================================================
# MAIN SCATTER — combined
# =============================================================================
def plot_combined_scatter(df_c: pd.DataFrame,
                          df_m: pd.DataFrame,
                          fd_col: str,
                          split: str,
                          output_dir: Path,
                          max_fd: float | None,
                          max_dvars: float | None):
    """
    Main figure: single scatter with both domains overlaid,
    regression lines, marginal histograms, and stat annotations.
    """
    fig = plt.figure(figsize=(10, 8))
    gs  = gridspec.GridSpec(
        2, 2,
        width_ratios=[4, 1], height_ratios=[1, 4],
        hspace=0.05, wspace=0.05,
    )
    ax_main  = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)

    x_c  = df_c[fd_col].values
    y_c  = df_c["dvars_mean"].values
    x_m  = df_m[fd_col].values
    y_m  = df_m["dvars_mean"].values

    # ── Clip if requested ─────────────────────────────────────────────────
    if max_fd is not None:
        mask_c = x_c <= max_fd;  mask_m = x_m <= max_fd
        x_c, y_c = x_c[mask_c], y_c[mask_c]
        x_m, y_m = x_m[mask_m], y_m[mask_m]
        ax_main.set_xlim(right=max_fd)
    if max_dvars is not None:
        mask_c = y_c <= max_dvars;  mask_m = y_m <= max_dvars
        x_c, y_c = x_c[mask_c], y_c[mask_c]
        x_m, y_m = x_m[mask_m], y_m[mask_m]
        ax_main.set_ylim(top=max_dvars)

    # ── Scatter ───────────────────────────────────────────────────────────
    ax_main.scatter(x_c, y_c, c=COLOR_CORRUPTED,  alpha=ALPHA_SCATTER,
                    s=POINT_SIZE, linewidths=0, label="High-Motion (Corrupted)",
                    zorder=3)
    ax_main.scatter(x_m, y_m, c=COLOR_MOTFREE,    alpha=ALPHA_SCATTER,
                    s=POINT_SIZE, linewidths=0, label="Low-Motion (Motion-free)",
                    zorder=3)

    # ── Regression lines ──────────────────────────────────────────────────
    stats_c = regression_line(ax_main, x_c, y_c, COLOR_CORRUPTED)
    stats_m = regression_line(ax_main, x_m, y_m, COLOR_MOTFREE)

    # ── Axes labels ───────────────────────────────────────────────────────
    fd_label = "Mean FD — from .par file (mm)"
    ax_main.set_xlabel(fd_label, fontsize=12)
    ax_main.set_ylabel("Mean DVARS", fontsize=12)
    ax_main.grid(alpha=0.25, zorder=0)

    # ── Stat annotation box ───────────────────────────────────────────────
    txt = (
        f"High-Motion  {stats_c}\n"
        f"Low-Motion   {stats_m}"
    )
    ax_main.text(0.98, 0.97, txt,
                 transform=ax_main.transAxes,
                 ha="right", va="top", fontsize=8.5,
                 bbox=dict(boxstyle="round,pad=0.4",
                           facecolor="white", alpha=0.85, edgecolor="#cccccc"))

    # ── Marginal histograms ───────────────────────────────────────────────
    bins_fd   = np.linspace(0, (max_fd   or max(x_c.max(), x_m.max())), 50)
    bins_dv   = np.linspace(0, (max_dvars or max(y_c.max(), y_m.max())), 50)

    ax_histx.hist(x_c, bins=bins_fd, color=COLOR_CORRUPTED,
                  alpha=0.6, density=True, label="Corrupted")
    ax_histx.hist(x_m, bins=bins_fd, color=COLOR_MOTFREE,
                  alpha=0.6, density=True, label="Motion-free")
    ax_histx.set_ylabel("Density", fontsize=8)
    ax_histx.tick_params(labelbottom=False)
    ax_histx.grid(alpha=0.2)
    ax_histx.set_title(
        f"[{split.upper()}]  FD vs DVARS — "
        f"High-Motion (n={len(x_c)}) vs Low-Motion (n={len(x_m)})",
        fontsize=12, fontweight="bold", pad=8,
    )

    ax_histy.hist(y_c, bins=bins_dv, color=COLOR_CORRUPTED,
                  alpha=0.6, density=True, orientation="horizontal")
    ax_histy.hist(y_m, bins=bins_dv, color=COLOR_MOTFREE,
                  alpha=0.6, density=True, orientation="horizontal")
    ax_histy.set_xlabel("Density", fontsize=8)
    ax_histy.tick_params(labelleft=False)
    ax_histy.grid(alpha=0.2)

    # ── Legend ────────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COLOR_CORRUPTED, markersize=8,
               label=f"High-Motion (Corrupted)  n={len(x_c)}"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COLOR_MOTFREE, markersize=8,
               label=f"Low-Motion (Motion-free)  n={len(x_m)}"),
        Line2D([0], [0], color=COLOR_CORRUPTED, linewidth=2,
               linestyle="--", label="Regression (Corrupted)"),
        Line2D([0], [0], color=COLOR_MOTFREE, linewidth=2,
               linestyle="--", label="Regression (Motion-free)"),
    ]
    ax_main.legend(handles=legend_elements, fontsize=8.5,
                   loc="lower right", framealpha=0.9)

    out_path = output_dir / f"{split}_fd_vs_dvars_scatter.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved → {out_path.name}")


# =============================================================================
# SECONDARY FIGURE — side-by-side per domain with density colouring
# =============================================================================
def plot_per_domain_scatter(df_c: pd.DataFrame,
                             df_m: pd.DataFrame,
                             fd_col: str,
                             split: str,
                             output_dir: Path,
                             max_fd: float | None,
                             max_dvars: float | None):
    """
    Two-panel figure: left = corrupted, right = motion-free.
    Points coloured by local density so overplotting is visible.
    """
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                              sharey=False, sharex=False)
    fig.suptitle(
        f"[{split.upper()}]  FD vs DVARS — Per-Domain Density Scatter",
        fontsize=13, fontweight="bold",
    )

    for ax, df, domain, color, title in [
        (axes[0], df_c, "corrupted",   COLOR_CORRUPTED,
         f"High-Motion (Corrupted)\nn={len(df_c)}"),
        (axes[1], df_m, "motion_free", COLOR_MOTFREE,
         f"Low-Motion (Motion-free)\nn={len(df_m)}"),
    ]:
        x = df[fd_col].values
        y = df["dvars_mean"].values

        if max_fd    is not None: mask = x <= max_fd;    x, y = x[mask], y[mask]
        if max_dvars is not None: mask = y <= max_dvars; x, y = x[mask], y[mask]

        # Kernel density for point colouring
        if len(x) > 10:
            try:
                kde    = gaussian_kde(np.vstack([x, y]))
                colors = kde(np.vstack([x, y]))
                colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-9)
            except Exception:
                colors = color
        else:
            colors = color

        sc = ax.scatter(x, y, c=colors,
                        cmap="RdYlBu_r" if domain == "corrupted" else "Blues",
                        alpha=0.5, s=POINT_SIZE_LARGE,
                        linewidths=0, zorder=3)

        # Regression
        if len(x) > 2:
            slope, intercept, r, p, r2 = regression_stats(x, y)
            x_fit = np.linspace(x.min(), x.max(), 200)
            ax.plot(x_fit, slope * x_fit + intercept,
                    color="black", linewidth=2, linestyle="--",
                    alpha=0.85, zorder=5,
                    label=f"r²={r2:.3f}, {'p<0.001' if p<0.001 else f'p={p:.3f}'}")
            ax.legend(fontsize=9, loc="upper left", framealpha=0.85)

        ax.set_xlabel("Mean FD (mm)", fontsize=11)
        ax.set_ylabel("Mean DVARS",   fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.25)

        # Colourbar
        cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        cbar.set_label("Local density", fontsize=8)

    plt.tight_layout()
    out_path = output_dir / f"{split}_fd_vs_dvars_per_domain.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved → {out_path.name}")


# =============================================================================
# TERTIARY FIGURE — FD bins showing DVARS distribution per bin
# =============================================================================
def plot_fd_bins_dvars(df_c: pd.DataFrame,
                       df_m: pd.DataFrame,
                       fd_col: str,
                       split: str,
                       output_dir: Path):
    """
    Bin chunks by FD range and show DVARS box-per-bin.
    Makes the monotonic FD→DVARS relationship very clear.
    """
    combined = pd.concat([df_c, df_m], ignore_index=True)
    combined = combined.dropna(subset=[fd_col, "dvars_mean"])

    # Define FD bins spanning the full range
    fd_max  = combined[fd_col].quantile(0.99)   # ignore extreme outliers for bins
    n_bins  = 8
    bin_edges = np.linspace(0, fd_max, n_bins + 1)
    bin_labels = [f"{bin_edges[i]:.2f}–{bin_edges[i+1]:.2f}"
                  for i in range(n_bins)]

    combined["fd_bin"] = pd.cut(combined[fd_col],
                                bins=bin_edges, labels=bin_labels,
                                include_lowest=True)

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle(
        f"[{split.upper()}]  DVARS Distribution by FD Bin\n"
        f"(shows monotonic FD → DVARS relationship across both domains)",
        fontsize=12, fontweight="bold",
    )

    positions_c = np.arange(n_bins) * 2.2
    positions_m = positions_c + 0.8

    for i, label in enumerate(bin_labels):
        subset = combined[combined["fd_bin"] == label]
        dc = subset[subset["domain"] == "corrupted"]["dvars_mean"].dropna().values
        dm = subset[subset["domain"] == "motion_free"]["dvars_mean"].dropna().values

        for pos, d, color in [(positions_c[i], dc, COLOR_CORRUPTED),
                               (positions_m[i], dm, COLOR_MOTFREE)]:
            if len(d) == 0:
                continue
            bp = ax.boxplot(d, positions=[pos], widths=0.6,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color="black", linewidth=1.5),
                            boxprops=dict(facecolor=color, alpha=0.7),
                            whiskerprops=dict(color=color),
                            capprops=dict(color=color))

    ax.set_xticks(positions_c + 0.4)
    ax.set_xticklabels(bin_labels, rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("Mean FD bin (mm)", fontsize=11)
    ax.set_ylabel("Mean DVARS",       fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    legend_elements = [
        Patch(facecolor=COLOR_CORRUPTED, alpha=0.7, label="High-Motion (Corrupted)"),
        Patch(facecolor=COLOR_MOTFREE,   alpha=0.7, label="Low-Motion (Motion-free)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper left")

    plt.tight_layout()
    out_path = output_dir / f"{split}_fd_bins_dvars_boxplot.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved → {out_path.name}")


# =============================================================================
# PRINT SUMMARY STATS
# =============================================================================
def print_summary(df_c, df_m, fd_col):
    log.info("\n" + "=" * 65)
    log.info("  FD vs DVARS SUMMARY")
    log.info("=" * 65)
    for label, df in [("High-Motion (Corrupted)", df_c),
                      ("Low-Motion  (Motion-free)", df_m)]:
        x = df[fd_col].dropna().values
        y = df["dvars_mean"].dropna().values
        if len(x) > 2:
            _, _, r, p, r2 = regression_stats(x, y)
            p_str = "p<0.001" if p < 0.001 else f"p={p:.4f}"
        else:
            r2, p_str = float("nan"), "n/a"
        log.info(f"  {label}")
        log.info(f"    n chunks    : {len(df)}")
        log.info(f"    Mean FD     : {x.mean():.4f} ± {x.std():.4f}")
        log.info(f"    Mean DVARS  : {y.mean():.4f} ± {y.std():.4f}")
        log.info(f"    r² (FD~DVARS): {r2:.4f}  {p_str}")
    log.info("=" * 65)


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()

    c_path, mf_path = resolve_paths(args)

    output_dir = (args.output_dir
                  if args.output_dir
                  else (args.qc_dir / "plots" if args.qc_dir
                        else Path("qc_results/plots")))
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 65)
    log.info("  FD vs DVARS RELATIONSHIP PLOT")
    log.info("=" * 65)
    log.info(f"  Corrupted CSV  : {c_path}")
    log.info(f"  Motion-free CSV: {mf_path}")
    log.info(f"  FD column      : {args.fd_col}")
    log.info(f"  Split          : {args.split}")
    log.info(f"  Output dir     : {output_dir}")
    log.info("=" * 65)

    log.info("\nLoading data...")
    df_c = load_and_clean(c_path,  "corrupted",   args.fd_col)
    df_m = load_and_clean(mf_path, "motion_free", args.fd_col)

    print_summary(df_c, df_m, args.fd_col)

    log.info("\nGenerating plots...")

    # 1. Main combined scatter with marginals
    plot_combined_scatter(df_c, df_m, args.fd_col, args.split,
                          output_dir, args.max_fd_clip, args.max_dvars_clip)

    # 2. Per-domain density scatter
    plot_per_domain_scatter(df_c, df_m, args.fd_col, args.split,
                             output_dir, args.max_fd_clip, args.max_dvars_clip)

    # 3. FD bins → DVARS boxplot
    plot_fd_bins_dvars(df_c, df_m, args.fd_col, args.split, output_dir)

    log.info("\n" + "=" * 65)
    log.info("  COMPLETE")
    log.info(f"  Saved 3 plots → {output_dir.resolve()}")
    log.info("=" * 65)


if __name__ == "__main__":
    main()