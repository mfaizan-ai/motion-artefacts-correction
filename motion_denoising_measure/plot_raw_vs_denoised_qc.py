"""
plot_raw_vs_denoised_qc.py
=============================
Qualitative raw-vs-denoised comparison: sagittal/coronal/axial mid-slices,
raw on top and denoised below, an axial difference map (denoised - raw) in
the rightmost column, and the mean signed/absolute correction applied.

The difference map's color range is symmetric around 0 and shared across
every plot in one run (the largest |difference| seen across all picks), so
different plots' correction magnitudes are visually comparable rather than
each auto-scaling to its own max.

Two selection modes:
  --grade1     pick verified Grade-1 (very-low-motion, safe-zone-checked)
               chunks straight from chunk_metadata.csv -- raw and denoised
               should look essentially identical here, since there's no
               motion to correct.
  (default)    random subjects/runs, a random timepoint.
"""
import argparse
import os
import random

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

CHUNK_METADATA_CSV = (
    "/lustre/disk/home/users/mfaizan/motion_correction/cycleGANS_on_2d_images/"
    "pytorch-CycleGAN-and-pix2pix/data_preprocessing/motion_grades_chunk_5_dataset_hfiltered/"
    "chunk_metadata.csv"
)
RAW_ROOT = (
    "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/faizan_motion_correction_dataset/"
    "brain_masked_cropped_hfiltered_normalized_to_common_space"
)
DENOISED_ROOT = (
    "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/faizan_motion_correction_dataset/"
    "motion_corrected_st_v4_all_video_runs"
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "motion_denoising_pipeline_level_evaluation", "raw_vs_denoised_qc",
)

FD_WINDOW = 5  # volumes immediately before the plotted timepoint (random mode)
# (name, mean_fd_lo, mean_fd_hi, max_fd_cap) -- same grading scheme as build_chunk_dataset.py
GRADES = [
    ("Grade 1", 0.0, 0.2, 0.25),
    ("Grade 2", 0.2, 0.5, 0.5),
    ("Grade 3", 0.5, 1.0, 1.0),
    ("Grade 4", 1.0, 2.0, 2.0),
    ("Grade 5", 2.0, 4.0, 4.0),
    ("Grade 6", 4.0, 10.0, 10.0),
]


def classify_grade(mean_fd, max_fd):
    """Buckets by mean FD alone (QC labeling, not the stricter mean+max-cap
    rule build_chunk_dataset.py uses to decide what's usable for training)."""
    if max_fd > 10.0:
        return "catastrophic"
    for name, lo, hi, _ in GRADES:
        if lo <= mean_fd < hi:
            return name
    return "Grade 6"


def denoised_path(raw_path):
    path = raw_path.replace(RAW_ROOT, DENOISED_ROOT)
    return path.replace(".nii.gz", "_corrected.nii.gz")


def select_runs(chunk_metadata_csv):
    meta = pd.read_csv(chunk_metadata_csv, dtype={"session_id": str, "run_id": str})
    v = meta[(meta["task"] == "videos") & (meta["age_group"] == "2mo")]
    v = v[~v["subject_id"].str.endswith("A")]
    cols = ["subject_id", "session_id", "run_id", "source_volume_path", "fd_path"]
    return v[cols].drop_duplicates().sort_values(["subject_id", "session_id", "run_id"])


def select_chunks_by_grade(chunk_metadata_csv, grade):
    """Already safe-zone-checked (no >=0.5mm spike in the 15 volumes before the
    chunk) by build_chunk_dataset.py -- no need to re-derive that here."""
    meta = pd.read_csv(chunk_metadata_csv, dtype={"session_id": str, "run_id": str})
    v = meta[(meta["task"] == "videos") & (meta["age_group"] == "2mo") & (meta["grade"] == grade)]
    v = v[~v["subject_id"].str.endswith("A")]
    cols = ["subject_id", "session_id", "run_id", "source_volume_path", "chunk_start", "chunk_end",
            "chunk_mean_fd", "chunk_max_fd"]
    return v[cols].drop_duplicates().reset_index(drop=True)


def motion_at_timepoint(fd_path, t, window=FD_WINDOW):
    fd = pd.read_csv(fd_path)["FramewiseDisplacement"].to_numpy()
    fd_window = fd[max(0, t - window):t]
    mean_fd, max_fd = fd_window.mean(), fd_window.max()
    return mean_fd, max_fd


def load_pick(raw_path, den_path, t, subject_id, session_id, run_id, mean_fd, max_fd):
    raw_vol = np.asarray(nib.load(raw_path).dataobj[..., t], dtype=np.float32)
    den_vol = np.asarray(nib.load(den_path).dataobj[..., t], dtype=np.float32)
    mask = raw_vol != 0
    diff = den_vol - raw_vol
    return {
        "subject_id": subject_id, "session_id": session_id, "run_id": run_id, "t": t,
        "mean_fd": mean_fd, "max_fd": max_fd, "raw_vol": raw_vol, "den_vol": den_vol,
        "mean_signed_correction": float(diff[mask].mean()),
        "mean_abs_correction": float(np.abs(diff[mask]).mean()),
    }


def plot_comparison(pick, output_dir, tag):
    raw_vol, den_vol = pick["raw_vol"], pick["den_vol"]
    grade = classify_grade(pick["mean_fd"], pick["max_fd"])
    x, y, z = (s // 2 for s in raw_vol.shape)

    fig, axes = plt.subplots(2, 4, figsize=(13.5, 7.5))
    views = [("sagittal", raw_vol[x, :, :], den_vol[x, :, :]),
             ("coronal", raw_vol[:, y, :], den_vol[:, y, :]),
             ("axial", raw_vol[:, :, z], den_vol[:, :, z])]
    for col, (name, raw_slice, den_slice) in enumerate(views):
        vmax = max(raw_slice.max(), den_slice.max())
        axes[0, col].imshow(raw_slice.T, cmap="gray", origin="lower", vmin=0, vmax=vmax)
        axes[0, col].set_title(name)
        axes[1, col].imshow(den_slice.T, cmap="gray", origin="lower", vmin=0, vmax=vmax)
        for ax in (axes[0, col], axes[1, col]):
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"sub-{pick['subject_id']} ses-{pick['session_id']} run-{pick['run_id']}, t={pick['t']}  |  "
        f"mean FD={pick['mean_fd']:.3f}, max FD={pick['max_fd']:.3f} ({grade}, raw)\n"
        f"mean correction: signed={pick['mean_signed_correction']:+.3f}, "
        f"absolute={pick['mean_abs_correction']:.3f}",
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.subplots_adjust(wspace=0.15)

    # Per-plot color scale (not shared across the batch): a shared scale washes low-motion
    # diffs out to near-white, since their range is tiny next to a high-motion plot's. The
    # printed mean absolute correction + each plot's own colorbar range are what make
    # cross-plot comparison possible instead.
    axial_diff = den_vol[:, :, z] - raw_vol[:, :, z]
    coronal_diff = den_vol[:, y, :] - raw_vol[:, y, :]
    dmax = max(np.abs(axial_diff).max(), np.abs(coronal_diff).max())

    im = axes[0, 3].imshow(axial_diff.T, cmap="RdBu_r", origin="lower", vmin=-dmax, vmax=dmax)
    axes[0, 3].set_title("axial diff\n(denoised - raw)")
    axes[1, 3].imshow(coronal_diff.T, cmap="RdBu_r", origin="lower", vmin=-dmax, vmax=dmax)
    axes[1, 3].set_title("coronal diff\n(denoised - raw)")
    for ax in (axes[0, 3], axes[1, 3]):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=[axes[0, 3], axes[1, 3]], label="signal difference", fraction=0.08, pad=0.03)

    row_x = axes[0, 0].get_position().x0 - 0.02
    fig.text(row_x, axes[0, 0].get_position().y0 + axes[0, 0].get_position().height / 2,
              "raw", va="center", ha="right", rotation=90, fontweight="bold")
    fig.text(row_x, axes[1, 0].get_position().y0 + axes[1, 0].get_position().height / 2,
              "denoised", va="center", ha="right", rotation=90, fontweight="bold")
    out_path = os.path.join(
        output_dir, f"sub-{pick['subject_id']}_ses-{pick['session_id']}_run-{pick['run_id']}_{tag}.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


def run_batch(picks, output_dir, tag):
    for pick in picks:
        plot_comparison(pick, output_dir, tag)


def run_random(args):
    runs = select_runs(args.chunk_metadata_csv)
    rng = random.Random(args.seed)
    candidates = runs.iloc[rng.sample(range(len(runs)), args.n_subjects)]

    picks = []
    for row in candidates.itertuples(index=False):
        den_path = denoised_path(row.source_volume_path)
        if not os.path.exists(den_path):
            print(f"skipping {row.subject_id}: no denoised volume")
            continue
        n_frames = nib.load(row.source_volume_path).shape[-1]
        t = rng.randrange(n_frames)
        mean_fd, max_fd = motion_at_timepoint(row.fd_path, t)
        picks.append(load_pick(row.source_volume_path, den_path, t, row.subject_id, row.session_id, row.run_id,
                                mean_fd, max_fd))
    run_batch(picks, args.output_dir, "raw_vs_denoised")


def run_grade_chunks(chunks, n_subjects, seed, output_dir, tag):
    rng = random.Random(seed)
    candidates = chunks.iloc[rng.sample(range(len(chunks)), n_subjects)]

    picks = []
    for row in candidates.itertuples(index=False):
        den_path = denoised_path(row.source_volume_path)
        if not os.path.exists(den_path):
            print(f"skipping {row.subject_id}: no denoised volume")
            continue
        t = (row.chunk_start + row.chunk_end) // 2
        picks.append(load_pick(row.source_volume_path, den_path, t, row.subject_id, row.session_id, row.run_id,
                                row.chunk_mean_fd, row.chunk_max_fd))
    run_batch(picks, output_dir, tag)


def run_grade1(args):
    chunks = select_chunks_by_grade(args.chunk_metadata_csv, "Grade 1")
    run_grade_chunks(chunks, args.n_subjects, args.seed, args.output_dir, "grade1_raw_vs_denoised")


def run_grade_wise(args):
    output_dir = os.path.join(args.output_dir, "grade_wise")
    os.makedirs(output_dir, exist_ok=True)
    for grade in args.grades:
        chunks = select_chunks_by_grade(args.chunk_metadata_csv, grade)
        tag = grade.lower().replace(" ", "") + "_raw_vs_denoised"
        run_grade_chunks(chunks, 1, args.seed, output_dir, tag)


def main():
    ap = argparse.ArgumentParser(description="Qualitative raw-vs-denoised slice comparison")
    ap.add_argument("--n_subjects", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grade1", action="store_true", help="Sample verified Grade-1 (safe-zone-checked) chunks instead of random timepoints")
    ap.add_argument("--grade_wise", action="store_true", help="One verified chunk per grade in --grades, saved under output_dir/grade_wise/")
    ap.add_argument("--grades", nargs="+", default=["Grade 2", "Grade 3", "Grade 4", "Grade 6"])
    ap.add_argument("--chunk_metadata_csv", default=CHUNK_METADATA_CSV)
    ap.add_argument("--output_dir", default=OUTPUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.grade_wise:
        run_grade_wise(args)
    elif args.grade1:
        run_grade1(args)
    else:
        run_random(args)


if __name__ == "__main__":
    main()
