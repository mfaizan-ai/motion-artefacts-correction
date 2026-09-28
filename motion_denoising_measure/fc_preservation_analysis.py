"""
fc_preservation_analysis.py
=============================
Does denoising preserve genuine FC signal while removing motion
contamination? Extends the QC-FC analysis (denoising_evaluation.py) with an
edge-wise raw-vs-denoised comparison:

For every unique FC edge, across the same 128 subjects used everywhere else
in this pipeline:
  1. One-sample t-test of Fisher-z FC against zero -- separately for raw and
     denoised -- asks "does this edge show a real, consistent connection
     across the population?"
  2. BH-FDR correction, applied separately to the raw and denoised p-values.
  3. Edges are classified as preserved / lost / gained / neither based on
     those two significance masks.
  4. Cross-referenced against the existing QC-FC FDR masks (already saved by
     denoising_evaluation.py) to see how many preserved edges also had their
     motion contamination removed.
  5. Effect-size comparisons (mean Fisher-z FC, |QC-FC r|, sign agreement)
     on top of the significance-only picture.

Usage:
    python fc_preservation_analysis.py \\
        --corrected_roi_timeseries_root <...> \\
        --raw_qcfc_dir <raw pipeline output_dir> \\
        --corrected_qcfc_dir <corrected pipeline output_dir>
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from statsmodels.stats.multitest import multipletests

from denoising_evaluation import (
    EvalConfig, PIPELINE_EVAL_ROOT, corrected_roi_ts_path, fisher_z,
    roi_ts_path, select_first_runs,
)
from losses import _pearson_corr_matrix

CATEGORIES = ["preserved", "lost", "gained", "neither"]


def compute_subject_fc(config: EvalConfig) -> tuple:
    """Fisher-z FC edges for every subject, raw and denoised: (n_subjects,
    n_edges) each, plus the flat edge indices (iu) they were extracted with."""
    runs = select_first_runs(config)
    if config.max_subjects is not None:
        runs = runs.head(config.max_subjects)

    raw_rows, corrected_rows = [], []
    n_rois, iu = None, None
    for i, row in enumerate(runs.itertuples(index=False), 1):
        raw_ts = np.load(roi_ts_path(row.source_volume_path, config))
        corrected_ts = np.load(corrected_roi_ts_path(row.source_volume_path, config))

        raw_fc = _pearson_corr_matrix(torch.from_numpy(raw_ts).float()).numpy()
        corrected_fc = _pearson_corr_matrix(torch.from_numpy(corrected_ts).float()).numpy()

        if iu is None:
            n_rois = raw_fc.shape[0]
            iu = np.triu_indices(n_rois, k=1)

        raw_rows.append(fisher_z(raw_fc[iu]))
        corrected_rows.append(fisher_z(corrected_fc[iu]))

        if i % 32 == 0 or i == len(runs):
            print(f"  FC: {i}/{len(runs)}")

    return np.stack(raw_rows), np.stack(corrected_rows), n_rois, iu


def one_sample_fdr_test(z_all: np.ndarray, alpha: float) -> dict:
    """Per-edge one-sample t-test of Fisher-z FC against zero, BH-FDR
    corrected across all edges. z_all: (n_subjects, n_edges)."""
    t, p = stats.ttest_1samp(z_all, popmean=0.0, axis=0)
    significant = np.zeros_like(p, dtype=bool)
    valid = np.isfinite(p)
    significant[valid], q, _, _ = multipletests(p[valid], alpha=alpha, method="fdr_bh")
    q_full = np.full_like(p, np.nan)
    q_full[valid] = q
    return {"t": t, "p": p, "q": q_full, "significant": significant, "mean_z": z_all.mean(axis=0)}


def classify_edges(raw_significant: np.ndarray, corrected_significant: np.ndarray) -> dict:
    return {
        "preserved": raw_significant & corrected_significant,
        "lost": raw_significant & ~corrected_significant,
        "gained": ~raw_significant & corrected_significant,
        "neither": ~raw_significant & ~corrected_significant,
    }


def load_qcfc_masks(raw_qcfc_dir: str, corrected_qcfc_dir: str, iu: tuple) -> dict:
    """Edge-wise (flat, same `iu` ordering) QC-FC r and FDR-significance,
    reusing the matrices denoising_evaluation.py already saved."""
    return {
        "raw_r": np.load(os.path.join(raw_qcfc_dir, "qc_fc_r.npy"))[iu],
        "raw_significant": np.load(os.path.join(raw_qcfc_dir, "qc_fc_fdr_significant.npy"))[iu],
        "corrected_r": np.load(os.path.join(corrected_qcfc_dir, "qc_fc_r.npy"))[iu],
        "corrected_significant": np.load(os.path.join(corrected_qcfc_dir, "qc_fc_fdr_significant.npy"))[iu],
    }


def build_edge_table(fc_raw: dict, fc_corrected: dict, category: dict, qcfc: dict) -> pd.DataFrame:
    category_label = np.full(len(fc_raw["mean_z"]), "neither", dtype=object)
    for name, mask in category.items():
        category_label[mask] = name

    return pd.DataFrame({
        "category": category_label,
        "raw_fc_p": fc_raw["p"], "raw_fc_q": fc_raw["q"], "raw_fc_significant": fc_raw["significant"],
        "corrected_fc_p": fc_corrected["p"], "corrected_fc_q": fc_corrected["q"],
        "corrected_fc_significant": fc_corrected["significant"],
        "raw_mean_z": fc_raw["mean_z"], "corrected_mean_z": fc_corrected["mean_z"],
        "raw_qcfc_r": qcfc["raw_r"], "raw_qcfc_significant": qcfc["raw_significant"],
        "corrected_qcfc_r": qcfc["corrected_r"], "corrected_qcfc_significant": qcfc["corrected_significant"],
    })


def summarize(edges: pd.DataFrame) -> dict:
    n = len(edges)
    counts = edges["category"].value_counts().reindex(CATEGORIES, fill_value=0)

    preserved = edges[edges["category"] == "preserved"]
    same_sign = np.sign(preserved["raw_mean_z"]) == np.sign(preserved["corrected_mean_z"])
    qcfc_removed = preserved["raw_qcfc_significant"] & ~preserved["corrected_qcfc_significant"]

    z_delta = edges["corrected_mean_z"].abs() - edges["raw_mean_z"].abs()
    qcfc_delta = edges["corrected_qcfc_r"].abs() - edges["raw_qcfc_r"].abs()

    summary = {"n_edges": n}
    for name in CATEGORIES:
        summary[f"n_{name}"] = int(counts[name])
        summary[f"pct_{name}"] = 100 * counts[name] / n
    summary.update({
        "n_preserved_same_sign": int(same_sign.sum()),
        "pct_preserved_same_sign": 100 * same_sign.mean() if len(preserved) else float("nan"),
        "n_preserved_qcfc_removed": int(qcfc_removed.sum()),
        "pct_preserved_qcfc_removed": 100 * qcfc_removed.mean() if len(preserved) else float("nan"),
        "n_qcfc_significant_raw": int(edges["raw_qcfc_significant"].sum()),
        "n_qcfc_significant_corrected": int(edges["corrected_qcfc_significant"].sum()),
        "median_abs_z_change": float(z_delta.median()), "mean_abs_z_change": float(z_delta.mean()),
        "median_abs_qcfc_change": float(qcfc_delta.median()), "mean_abs_qcfc_change": float(qcfc_delta.mean()),
    })
    return summary


def print_summary(summary: dict) -> None:
    print(f"\n{summary['n_edges']:,} unique FC edges")
    for name in CATEGORIES:
        print(f"  {name}: {summary[f'n_{name}']:,} ({summary[f'pct_{name}']:.2f}%)")
    print(f"Preserved edges keeping the same sign: {summary['n_preserved_same_sign']:,} "
          f"({summary['pct_preserved_same_sign']:.2f}%)")
    print(f"Preserved edges that lost QC-FC significance: {summary['n_preserved_qcfc_removed']:,} "
          f"({summary['pct_preserved_qcfc_removed']:.2f}%)")
    print(f"QC-FC significant edges: raw={summary['n_qcfc_significant_raw']:,}, "
          f"corrected={summary['n_qcfc_significant_corrected']:,}")
    print(f"Median |change| in mean Fisher-z FC: {summary['median_abs_z_change']:.4f}")
    print(f"Median |change| in |QC-FC r|: {summary['median_abs_qcfc_change']:.4f}")


# ---------------------------------------------------------------- plots ----

def plot_category_counts(fig_dir: str, summary: dict) -> None:
    counts = [summary[f"n_{name}"] for name in CATEGORIES]
    pcts = [summary[f"pct_{name}"] for name in CATEGORIES]
    colors = ["#4C78A8", "#E45756", "#54A24B", "#B0B0B0"]

    # At their true share (0.06% / 0.15%), gained/neither are sub-degree
    # wedges -- invisible at any figure size, no matter the explode. Floor
    # each wedge's *displayed* size to a minimum angle so its color still
    # shows as a thin sliver, then renormalize so the pie still sums to
    # 360 deg. All printed numbers (on-wedge label, legend) stay tied to
    # the true counts/percentages, not this display-only adjustment.
    total = sum(counts)
    min_display_frac = 0.02
    display = [max(c / total, min_display_frac) for c in counts]
    display = [d / sum(display) for d in display]

    true_pct_iter = iter(pcts)

    def autopct(_):
        pct = next(true_pct_iter)
        return f"{pct:.1f}%" if pct >= 1 else ""

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, _, _ = ax.pie(
        display, autopct=autopct, pctdistance=0.75, colors=colors, startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}, textprops={"fontsize": 10},
    )
    legend_labels = [f"{name} ({count:,}, {pct:.2f}%)" for name, count, pct in zip(CATEGORIES, counts, pcts)]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    ax.set_title("FC edge preservation after denoising", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fc_preservation_categories.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_qcfc_significant_counts(fig_dir: str, summary: dict) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 5))
    counts = [summary["n_qcfc_significant_raw"], summary["n_qcfc_significant_corrected"]]
    bars = ax.bar(["Raw", "Denoised"], counts, color=["#E45756", "#4C78A8"])
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{count:,}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("QC-FC FDR-significant edges")
    ax.set_title("QC-FC significant edges: before vs after", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "qcfc_significant_before_after.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_preserved_qcfc_transition(fig_dir: str, edges: pd.DataFrame) -> None:
    preserved = edges[edges["category"] == "preserved"]
    # "not sig." first on both axes: the two concordant cells (not-sig/not-sig
    # and sig/sig, the common outcomes) then sit on the diagonal, while the
    # rare discordant cells (QC-FC significance flipping) land off-diagonal.
    table = pd.crosstab(
        preserved["raw_qcfc_significant"].map({True: "QC-FC sig.", False: "not sig."}),
        preserved["corrected_qcfc_significant"].map({True: "QC-FC sig.", False: "not sig."}),
    ).reindex(index=["not sig.", "QC-FC sig."], columns=["not sig.", "QC-FC sig."], fill_value=0)

    # Single-hue sequential colormap: the diagonal's large concordant counts
    # render as strong blue, the off-diagonal's small discordant counts as
    # pale blue -- same hue throughout, intensity tracks magnitude.
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(table.values, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(table.columns)
    ax.set_yticks([0, 1]); ax.set_yticklabels(table.index)
    ax.set_xlabel("After denoising")
    ax.set_ylabel("Before denoising")
    ax.set_title("QC-FC for preserved FC edges", loc="left", fontweight="bold")

    # Dark divider between the 2x2 cells, plus a matching border, since the
    # pale off-diagonal cells otherwise blend into the white figure background.
    ax.axhline(0.5, color="black", linewidth=1.5)
    ax.axvline(0.5, color="black", linewidth=1.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.5)

    for i in range(2):
        for j in range(2):
            # Blues is light-to-dark (opposite of viridis): high counts render
            # as dark cells needing white text, low counts as pale cells needing black.
            ax.text(j, i, f"{table.values[i, j]:,}", ha="center", va="center",
                     fontsize=13, color="white" if table.values[i, j] > table.values.max() / 2 else "black")

    # make_axes_locatable derives the colorbar axis from ax's own box, so it
    # always matches the matrix's rendered height exactly.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, label="Preserved edges")

    fig.savefig(os.path.join(fig_dir, "preserved_edges_qcfc_transition.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_magnitude_scatter(fig_dir: str, edges: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    ax.scatter(edges["raw_mean_z"], edges["corrected_mean_z"], s=3, alpha=0.2, color="#4C78A8")
    lim = np.nanmax(np.abs(edges[["raw_mean_z", "corrected_mean_z"]].to_numpy()))
    ax.plot([-lim, lim], [-lim, lim], color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Raw mean Fisher-z FC")
    ax.set_ylabel("Denoised mean Fisher-z FC")
    ax.set_title("FC magnitude", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.scatter(edges["raw_qcfc_r"].abs(), edges["corrected_qcfc_r"].abs(), s=3, alpha=0.2, color="#E45756")
    lim = np.nanmax(np.abs(edges[["raw_qcfc_r", "corrected_qcfc_r"]].to_numpy()))
    ax.plot([0, lim], [0, lim], color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Raw |QC-FC r|")
    ax.set_ylabel("Denoised |QC-FC r|")
    ax.set_title("QC-FC magnitude", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fc_qcfc_magnitude_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_plots(output_dir: str, edges: pd.DataFrame, summary: dict) -> None:
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_category_counts(fig_dir, summary)
    plot_qcfc_significant_counts(fig_dir, summary)
    plot_preserved_qcfc_transition(fig_dir, edges)
    plot_magnitude_scatter(fig_dir, edges)
    print(f"saved plots -> {fig_dir}")


def run(config: EvalConfig, raw_qcfc_dir: str, corrected_qcfc_dir: str, edge_alpha: float) -> None:
    raw_z_all, corrected_z_all, n_rois, iu = compute_subject_fc(config)
    print(f"{raw_z_all.shape[0]} subjects, {raw_z_all.shape[1]:,} unique edges ({n_rois} ROIs)")

    fc_raw = one_sample_fdr_test(raw_z_all, edge_alpha)
    fc_corrected = one_sample_fdr_test(corrected_z_all, edge_alpha)
    category = classify_edges(fc_raw["significant"], fc_corrected["significant"])
    qcfc = load_qcfc_masks(raw_qcfc_dir, corrected_qcfc_dir, iu)

    edges = build_edge_table(fc_raw, fc_corrected, category, qcfc)
    summary = summarize(edges)
    print_summary(summary)

    os.makedirs(config.output_dir, exist_ok=True)
    edges.to_csv(os.path.join(config.output_dir, "fc_preservation_edges.csv"), index=False)
    with open(os.path.join(config.output_dir, "fc_preservation_summary.txt"), "w") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value:.4f}\n" if isinstance(value, float) else f"{key}: {value}\n")
    print(f"saved results -> {config.output_dir}")

    make_plots(config.output_dir, edges, summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw-vs-denoised FC signal preservation analysis")
    parser.add_argument(
        "--chunk_metadata_csv", default=(
            "/lustre/disk/home/users/mfaizan/motion_correction/cycleGANS_on_2d_images/"
            "pytorch-CycleGAN-and-pix2pix/data_preprocessing/motion_grades_chunk_5_dataset_hfiltered/"
            "chunk_metadata.csv"
        ),
    )
    parser.add_argument(
        "--source_root", default=(
            "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
            "faizan_motion_correction_dataset/brain_masked_cropped_hfiltered_normalized_to_common_space"
        ),
    )
    parser.add_argument(
        "--roi_timeseries_root", default=(
            "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
            "faizan_motion_correction_dataset/roi_timeseries_hfiltered_videos"
        ),
    )
    parser.add_argument(
        "--corrected_roi_timeseries_root", default=(
            "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
            "faizan_motion_correction_dataset/roi_timeseries_st_v4_corrected"
        ),
    )
    parser.add_argument(
        "--raw_qcfc_dir",
        default=os.path.join(PIPELINE_EVAL_ROOT, "raw_data_each_sub_first_run"),
        help="Pipeline output_dir written by denoising_evaluation.py for the raw data",
    )
    parser.add_argument(
        "--corrected_qcfc_dir",
        default=os.path.join(PIPELINE_EVAL_ROOT, "st_v4_corrected_each_sub_first_run"),
        help="Pipeline output_dir written by denoising_evaluation.py for the corrected data",
    )
    parser.add_argument("--pipeline_name", default="fc_signal_preservation")
    parser.add_argument("--edge_alpha", type=float, default=0.05)
    parser.add_argument("--max_subjects", type=int, default=None)
    args = parser.parse_args()

    config = EvalConfig(
        chunk_metadata_csv=args.chunk_metadata_csv,
        source_root=args.source_root,
        roi_timeseries_root=args.roi_timeseries_root,
        corrected_roi_timeseries_root=args.corrected_roi_timeseries_root,
        atlas_path="",
        output_dir=os.path.join(PIPELINE_EVAL_ROOT, args.pipeline_name),
        edge_alpha=args.edge_alpha,
        max_subjects=args.max_subjects,
    )
    return config, args.raw_qcfc_dir, args.corrected_qcfc_dir, args.edge_alpha


if __name__ == "__main__":
    run(*parse_args())
