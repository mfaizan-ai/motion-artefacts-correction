"""
plot_isc_comparison.py
=========================
2x2 network-ISFC comparison for one order: the original unprocessed data (no
cropping/masking/high-pass filter), the actual training-data raw (cropped,
brain-masked, high-pass filtered), its st_v4-denoised counterpart, and a
difference map (denoised - raw preprocessed).

Each panel's cells are FDR-significance tested (two-sided, BH-FDR across the
7 diagonal + 21 unique off-diagonal cells) and marked with "*" -- the three
ISC panels via a one-sample t-test of each cell's per-subject Fisher-z value
against 0, the delta panel via a paired t-test (denoised - raw, per subject)
against 0, since both use the same subjects. Per-panel significance tables
are saved alongside the figure.
"""
import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "ISC_analysis"))
from loo_isc import fisher_z  # noqa: E402


def dataset_paths(order):
    raw_unprocessed_dir = os.path.join(REPO_ROOT, "ISC_analysis", "results", "raw_data_network_level")
    raw_preprocessed_dir = os.path.join(
        REPO_ROOT, "motion_denoising_pipeline_level_evaluation", "ISC", "train_data_raw_preprocessed",
        f"order_{order}", "figures",
    )
    denoised_dir = os.path.join(
        REPO_ROOT, "motion_denoising_pipeline_level_evaluation", "ISC", "denoised_st_v4", f"order_{order}", "figures"
    )
    return {
        "raw_unprocessed": (
            raw_unprocessed_dir, f"order_{order}_network_isfc_group.csv", f"order_{order}_network_isfc_persubject.npy",
        ),
        "raw_preprocessed": (
            raw_preprocessed_dir, f"order_{order}_train_data_raw_preprocessed_network_isfc_group.csv",
            f"order_{order}_train_data_raw_preprocessed_network_isfc_persubject.npy",
        ),
        "denoised_st_v4": (
            denoised_dir, f"order_{order}_denoised_st_v4_network_isfc_group.csv",
            f"order_{order}_denoised_st_v4_network_isfc_persubject.npy",
        ),
    }


def output_dir_for(order):
    """Figures only -- see data_dir_for for the CSVs/npy this order's data lives in."""
    return os.path.join(REPO_ROOT, "motion_denoising_pipeline_level_evaluation", "ISC", f"comparison_order_{order}")


def data_dir_for(order):
    return os.path.join(REPO_ROOT, "motion_denoising_pipeline_level_evaluation", "ISC", "comparison_data", f"order_{order}")


def load_group_matrix(directory, group_csv):
    df = pd.read_csv(os.path.join(directory, group_csv), index_col=0)
    return df.values, list(df.columns)


def load_persubject(directory, persubject_npy):
    """Per-subject ISFC isn't symmetric (isfc[i,j] = corr(target's net i, others' net
    j) != isfc[j,i] in general for one subject) -- symmetrize per subject the same way
    the group summary does, so significance testing matches exactly what's plotted."""
    isfc = np.load(os.path.join(directory, persubject_npy))
    return (isfc + isfc.transpose(0, 2, 1)) / 2


def upper_triangle_indices(n):
    return [(i, j) for i in range(n) for j in range(i, n)]


def cellwise_significance(persubject_z, names, alpha=0.05):
    """One-sample two-sided t-test per unique cell (diagonal + upper triangle) against
    0, BH-FDR corrected across those unique cells only -- avoids wasting the correction
    budget on the symmetric duplicate of each off-diagonal cell."""
    n = len(names)
    pairs = upper_triangle_indices(n)
    t_vals, p_vals = [], []
    for i, j in pairs:
        t, p = stats.ttest_1samp(persubject_z[:, i, j], popmean=0.0)
        t_vals.append(t)
        p_vals.append(p)
    significant, q_vals, _, _ = multipletests(p_vals, alpha=alpha, method="fdr_bh")

    sig_matrix = np.zeros((n, n), dtype=bool)
    rows = []
    for (i, j), t, p, q, sig in zip(pairs, t_vals, p_vals, q_vals, significant):
        sig_matrix[i, j] = sig_matrix[j, i] = sig
        rows.append({"network_i": names[i], "network_j": names[j], "t": t, "p": p, "q": q, "significant": sig})
    return sig_matrix, rows


def save_significance(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmap(ax, matrix, sig_matrix, names, title, cmap, vlim, colorbar_label):
    im = ax.imshow(matrix, cmap=cmap, vmin=-vlim, vmax=vlim)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, ha="center")
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            value = matrix[i, j]
            star = "*" if sig_matrix[i, j] else ""
            color = "white" if abs(value) > vlim * 0.6 else "black"
            ax.text(j, i, f"{value:.2f}{star}", ha="center", va="center", fontsize=7, color=color)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, label=colorbar_label, fraction=0.046, pad=0.04)


def save_single_panel(output_path, matrix, sig_matrix, names, title, cmap, vlim, colorbar_label):
    fig, ax = plt.subplots(figsize=(7, 6.5))
    plot_heatmap(ax, matrix, sig_matrix, names, title, cmap, vlim, colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {output_path}")


def main(order):
    datasets = dataset_paths(order)
    output_dir = output_dir_for(order)
    data_dir = data_dir_for(order)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    group, persubject_z, sig = {}, {}, {}
    names = None
    for tag, (directory, group_csv, persubject_npy) in datasets.items():
        group[tag], names = load_group_matrix(directory, group_csv)
        persubject_z[tag] = fisher_z(load_persubject(directory, persubject_npy))
        sig_matrix, rows = cellwise_significance(persubject_z[tag], names)
        sig[tag] = sig_matrix
        save_significance(os.path.join(data_dir, f"order_{order}_{tag}_isfc_significance.csv"), rows)
        print(f"order {order} {tag}: {sum(r['significant'] for r in rows)}/{len(rows)} unique cells FDR-significant")

    delta = group["denoised_st_v4"] - group["raw_preprocessed"]
    delta_z = persubject_z["denoised_st_v4"] - persubject_z["raw_preprocessed"]
    delta_sig, delta_rows = cellwise_significance(delta_z, names)
    save_significance(os.path.join(data_dir, f"order_{order}_delta_isfc_significance.csv"), delta_rows)
    print(f"order {order} delta (paired): {sum(r['significant'] for r in delta_rows)}/{len(delta_rows)} unique cells FDR-significant")

    isc_vlim = max(np.abs(group[t]).max() for t in datasets)
    delta_vlim = np.abs(delta).max()

    panel_specs = [
        ("raw_unprocessed", group["raw_unprocessed"], sig["raw_unprocessed"],
         "Raw, no preprocessing\n(no crop/brain-mask/high-pass filter)", "RdBu_r", isc_vlim, "ISC (r)"),
        ("raw_preprocessed", group["raw_preprocessed"], sig["raw_preprocessed"],
         "Raw, preprocessed\n(cropped, brain-masked, high-pass filtered)", "RdBu_r", isc_vlim, "ISC (r)"),
        ("denoised_st_v4", group["denoised_st_v4"], sig["denoised_st_v4"],
         "Denoised (st_v4)", "RdBu_r", isc_vlim, "ISC (r)"),
        ("delta", delta, delta_sig,
         "Delta: denoised - raw preprocessed\n(paired t-test)", "PuOr_r", delta_vlim, "Delta ISC (r)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    for ax, (_, matrix, sig_matrix, title, cmap, vlim, label) in zip(axes.flat, panel_specs):
        plot_heatmap(ax, matrix, sig_matrix, names, title, cmap, vlim, label)
    fig.suptitle(
        f"Order {order} network ISC (7x7 ISFC) comparison  (* = FDR-significant, q<0.05)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = os.path.join(output_dir, "isc_network_comparison.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {output_path}")

    for tag, matrix, sig_matrix, title, cmap, vlim, label in panel_specs:
        save_single_panel(
            os.path.join(output_dir, f"isc_network_comparison_{tag}.png"),
            matrix, sig_matrix, names, f"Order {order}: {title}", cmap, vlim, label,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", default="A")
    args = ap.parse_args()
    main(args.order)
