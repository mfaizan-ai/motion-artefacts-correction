"""
plot_isc_comparison_masked.py
================================
4x2 grid for one order: the 4 network-ISFC panels (raw unprocessed, raw
preprocessed, denoised st_v4, delta) shown twice -- once masking cells that
fail the parametric one-sample-t-test FDR correction, once masking cells
that fail the proper subject-level bootstrap FDR correction (see
bootstrap_isc_significance.py). Masked cells are shown blank (light gray, no
value printed) instead of starred, so the two significance criteria's
disagreement is visually obvious at a glance.

Run plot_isc_comparison.py and bootstrap_isc_significance.py first (same
--order) -- this script only reads their saved outputs.
"""
import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PANEL_TITLES = {
    "raw_unprocessed": "Raw, no preprocessing",
    "raw_preprocessed": "Raw, preprocessed",
    "denoised_st_v4": "Denoised (st_v4)",
    "delta": "Delta: denoised - raw preprocessed",
}
PANELS = ["raw_unprocessed", "raw_preprocessed", "denoised_st_v4", "delta"]


def comparison_dir(order):
    """Figures only."""
    return os.path.join(REPO_ROOT, "motion_denoising_pipeline_level_evaluation", "ISC", f"comparison_order_{order}")


def comparison_data_dir(order):
    """Significance CSVs + bootstrap distributions."""
    return os.path.join(REPO_ROOT, "motion_denoising_pipeline_level_evaluation", "ISC", "comparison_data", f"order_{order}")


def group_csvs(order):
    return {
        "raw_unprocessed": os.path.join(
            REPO_ROOT, "ISC_analysis", "results", "raw_data_network_level", f"order_{order}_network_isfc_group.csv"
        ),
        "raw_preprocessed": os.path.join(
            REPO_ROOT, "motion_denoising_pipeline_level_evaluation", "ISC", "train_data_raw_preprocessed",
            f"order_{order}", "figures", f"order_{order}_train_data_raw_preprocessed_network_isfc_group.csv",
        ),
        "denoised_st_v4": os.path.join(
            REPO_ROOT, "motion_denoising_pipeline_level_evaluation", "ISC", "denoised_st_v4",
            f"order_{order}", "figures", f"order_{order}_denoised_st_v4_network_isfc_group.csv",
        ),
    }


def load_matrix(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    return df.values, list(df.columns)


def load_significance_matrix(path, names):
    n = len(names)
    idx = {name: i for i, name in enumerate(names)}
    sig = np.zeros((n, n), dtype=bool)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            i, j = idx[r["network_i"]], idx[r["network_j"]]
            is_sig = r["significant"] in ("True", "true", "1")
            sig[i, j] = sig[j, i] = is_sig
    return sig


def plot_masked_heatmap(ax, matrix, sig_matrix, names, title, cmap, vlim, colorbar_label):
    masked = np.ma.masked_where(~sig_matrix, matrix)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#e8e8e8")
    im = ax.imshow(masked, cmap=cmap_obj, vmin=-vlim, vmax=vlim)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, ha="center")
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            if not sig_matrix[i, j]:
                continue
            value = matrix[i, j]
            color = "white" if abs(value) > vlim * 0.6 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, label=colorbar_label, fraction=0.046, pad=0.04)


def main(order):
    out_dir = comparison_dir(order)
    data_dir = comparison_data_dir(order)
    group, names = {}, None
    for tag, csv_path in group_csvs(order).items():
        group[tag], names = load_matrix(csv_path)
    group["delta"] = group["denoised_st_v4"] - group["raw_preprocessed"]

    isc_vlim = max(np.abs(group[t]).max() for t in ("raw_unprocessed", "raw_preprocessed", "denoised_st_v4"))
    delta_vlim = np.abs(group["delta"]).max()

    parametric_sig = {
        tag: load_significance_matrix(os.path.join(data_dir, f"order_{order}_{tag}_isfc_significance.csv"), names)
        for tag in PANELS
    }
    bootstrap_sig = {
        tag: load_significance_matrix(
            os.path.join(data_dir, f"order_{order}_{tag}_isfc_bootstrap_significance.csv"), names
        )
        for tag in PANELS
    }

    fig, axes = plt.subplots(2, 4, figsize=(22, 12))
    for col, tag in enumerate(PANELS):
        cmap = "RdBu_r" if tag != "delta" else "PuOr_r"
        vlim = isc_vlim if tag != "delta" else delta_vlim
        label = "ISC (r)" if tag != "delta" else "Delta ISC (r)"
        plot_masked_heatmap(
            axes[0, col], group[tag], parametric_sig[tag], names,
            f"{PANEL_TITLES[tag]}\n(parametric t-test)", cmap, vlim, label,
        )
        plot_masked_heatmap(
            axes[1, col], group[tag], bootstrap_sig[tag], names,
            f"{PANEL_TITLES[tag]}\n(subject bootstrap, n=10000)", cmap, vlim, label,
        )

    fig.suptitle(
        f"Order {order} network ISC (7x7 ISFC): parametric vs. bootstrap FDR masking (q<0.05, gray = not significant)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = os.path.join(out_dir, "isc_network_comparison_masked_parametric_vs_bootstrap.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", default="A")
    args = ap.parse_args()
    main(args.order)
