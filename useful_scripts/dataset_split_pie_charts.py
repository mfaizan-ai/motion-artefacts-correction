#!/usr/bin/env python3
"""
dataset_split_pie_charts.py
===========================
Two clean pie charts: train/val/test split for corrupted and motion free.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

META_DIR = (
    "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
    "faizan_motion_correction_dataset/cyclegans_chunk5_dataset/metadata"
)
OUT_DIR = (
    "/lustre/disk/home/users/mfaizan/motion_correction/prototyping/"
    "motion-artefacts-correction/dataset_quality_check/chunk5"
)

COLORS = ["#4CAF50", "#FF9800", "#9C27B0"]


def load_counts():
    splits = {}
    for split in ["train", "val", "test"]:
        c_file = f"{split}_corrupted_all.csv" if split == "train" else f"{split}_corrupted_balanced.csv"
        splits[split] = {
            "corrupted": len(pd.read_csv(f"{META_DIR}/{c_file}")),
            "motion_free": len(pd.read_csv(f"{META_DIR}/{split}_motion_free.csv")),
        }
    return splits


def make_pie(ax, sizes, title):
    total = sum(sizes)
    labels = [
        f"Train\n{sizes[0]:,} ({sizes[0]/total*100:.1f}%)",
        f"Val\n{sizes[1]:,} ({sizes[1]/total*100:.1f}%)",
        f"Test\n{sizes[2]:,} ({sizes[2]/total*100:.1f}%)",
    ]
    wedges, texts = ax.pie(
        sizes, labels=labels, colors=COLORS,
        startangle=90, textprops={"fontsize": 12},
        wedgeprops={"edgecolor": "white", "linewidth": 2.5},
    )
    ax.set_title(f"{title}\n(n = {total:,})", fontsize=14, fontweight="bold", pad=15)


def main():
    splits = load_counts()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    make_pie(ax1,
             [splits[s]["corrupted"] for s in ["train", "val", "test"]],
             "Motion Corrupted")

    make_pie(ax2,
             [splits[s]["motion_free"] for s in ["train", "val", "test"]],
             "Motion Free")

    plt.tight_layout(pad=3)
    out_path = f"{OUT_DIR}/dataset_split_pie_charts.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
