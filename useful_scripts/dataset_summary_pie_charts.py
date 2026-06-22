#!/usr/bin/env python3
"""
dataset_summary_pie_charts.py
=============================
Pie charts summarising the chunk-5 dataset before and after train/val/test split.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PREPROC_DIR = (
    "/lustre/disk/home/users/mfaizan/motion_correction/prototyping/"
    "motion-artefacts-correction/dataset/chunk-5/preprocessed"
)
META_DIR = (
    "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
    "faizan_motion_correction_dataset/cyclegans_chunk5_dataset/metadata"
)
OUT_DIR = (
    "/lustre/disk/home/users/mfaizan/motion_correction/prototyping/"
    "motion-artefacts-correction/dataset_quality_check/chunk5"
)

COLORS = {
    "corrupted": "#E05C5C",
    "rest_clean": "#5C8AE0",
    "video_clean": "#5CB8E0",
    "motion_free": "#5C8AE0",
    "train": "#4CAF50",
    "val": "#FF9800",
    "test": "#9C27B0",
}


def load_counts():
    n_corr = len(pd.read_csv(f"{PREPROC_DIR}/motion_corrupted_chunk_dataset_with_preprocessed.csv"))
    n_rest = len(pd.read_csv(f"{PREPROC_DIR}/resting_state_clean_chunk_dataset_with_preprocessed.csv"))
    n_video = len(pd.read_csv(f"{PREPROC_DIR}/video_state_clean_chunk_dataset_with_preprocessed.csv"))

    splits = {}
    for split in ["train", "val", "test"]:
        c_file = f"{split}_corrupted_all.csv" if split == "train" else f"{split}_corrupted_balanced.csv"
        n_c = len(pd.read_csv(f"{META_DIR}/{c_file}"))
        n_mf = len(pd.read_csv(f"{META_DIR}/{split}_motion_free.csv"))
        splits[split] = {"corrupted": n_c, "motion_free": n_mf}

    return n_corr, n_rest, n_video, splits


def main():
    n_corr, n_rest, n_video, splits = load_counts()
    n_mf_total = n_rest + n_video

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Chunk-5 Dataset Summary", fontsize=16, fontweight="bold", y=0.98)

    # --- Row 1: Before split ---

    # 1a: Corrupted vs Motion Free
    sizes = [n_corr, n_mf_total]
    labels = [f"Corrupted\n{n_corr:,}", f"Motion Free\n{n_mf_total:,}"]
    colors = [COLORS["corrupted"], COLORS["motion_free"]]
    axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
                   startangle=90, textprops={"fontsize": 10},
                   wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[0, 0].set_title(f"Before Split — Domain Balance\n(Total: {n_corr + n_mf_total:,})",
                         fontsize=11, fontweight="bold")

    # 1b: Motion Free breakdown (resting vs video)
    sizes = [n_rest, n_video]
    labels = [f"Resting State\n{n_rest:,}", f"Video Task\n{n_video:,}"]
    colors = [COLORS["rest_clean"], COLORS["video_clean"]]
    axes[0, 1].pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
                   startangle=90, textprops={"fontsize": 10},
                   wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[0, 1].set_title(f"Motion Free Breakdown\n(Total: {n_mf_total:,})",
                         fontsize=11, fontweight="bold")

    # 1c: All three categories
    sizes = [n_corr, n_rest, n_video]
    labels = [f"Corrupted\n{n_corr:,}", f"Clean Resting\n{n_rest:,}",
              f"Clean Video\n{n_video:,}"]
    colors = [COLORS["corrupted"], COLORS["rest_clean"], COLORS["video_clean"]]
    axes[0, 2].pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
                   startangle=90, textprops={"fontsize": 10},
                   wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[0, 2].set_title(f"All Categories\n(Total: {n_corr + n_mf_total:,})",
                         fontsize=11, fontweight="bold")

    # --- Row 2: After split ---

    # 2a: Corrupted split
    sizes = [splits[s]["corrupted"] for s in ["train", "val", "test"]]
    total_c = sum(sizes)
    labels = [f"Train\n{sizes[0]:,}", f"Val\n{sizes[1]:,}", f"Test\n{sizes[2]:,}"]
    colors = [COLORS["train"], COLORS["val"], COLORS["test"]]
    axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
                   startangle=90, textprops={"fontsize": 10},
                   wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1, 0].set_title(f"Corrupted — Train/Val/Test\n(Total: {total_c:,})",
                         fontsize=11, fontweight="bold")

    # 2b: Motion Free split
    sizes = [splits[s]["motion_free"] for s in ["train", "val", "test"]]
    total_mf = sum(sizes)
    labels = [f"Train\n{sizes[0]:,}", f"Val\n{sizes[1]:,}", f"Test\n{sizes[2]:,}"]
    axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
                   startangle=90, textprops={"fontsize": 10},
                   wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1, 1].set_title(f"Motion Free — Train/Val/Test\n(Total: {total_mf:,})",
                         fontsize=11, fontweight="bold")

    # 2c: Combined per-split breakdown (stacked style as nested pie)
    outer_sizes = []
    outer_colors = []
    outer_labels = []
    inner_sizes = []
    inner_colors = []
    inner_labels = []

    corr_alpha = {"train": "#E05C5C", "val": "#E8A0A0", "test": "#F0C8C8"}
    mf_alpha = {"train": "#5C8AE0", "val": "#A0C0E8", "test": "#C8D8F0"}

    for s in ["train", "val", "test"]:
        c = splits[s]["corrupted"]
        m = splits[s]["motion_free"]
        outer_sizes.append(c + m)
        outer_colors.append(COLORS[s])
        outer_labels.append(f"{s.capitalize()}\n{c + m:,}")
        inner_sizes.extend([c, m])
        inner_colors.extend([corr_alpha[s], mf_alpha[s]])
        inner_labels.extend([f"C:{c:,}", f"MF:{m:,}"])

    axes[1, 2].pie(outer_sizes, labels=outer_labels, colors=outer_colors,
                   autopct="%1.1f%%", startangle=90, pctdistance=0.82,
                   textprops={"fontsize": 10}, radius=1.0,
                   wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1, 2].pie(inner_sizes, labels=inner_labels, colors=inner_colors,
                   startangle=90, textprops={"fontsize": 7}, radius=0.65,
                   wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    axes[1, 2].set_title(f"All Splits — Corrupted vs Motion Free\n"
                         f"(Total: {total_c + total_mf:,})",
                         fontsize=11, fontweight="bold")

    plt.tight_layout()
    out_path = f"{OUT_DIR}/dataset_summary_pie_charts.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
