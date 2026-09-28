"""
video_low_motion_reference.py
================================
Low-motion FC reference built from the video task itself, rather than
resting-state (see run_low_motion_reference_analysis in denoising_evaluation.py
for the rest10 version).

Video-task runs rarely hold still for a full run (mean FD ~1.09mm vs ~0.21mm
for rest10 -- see the rest-vs-video FD scan earlier this session), so instead
of requiring a whole low-motion RUN, this scans each run for individual
low-motion 20-volume CHUNKS using the exact grid/safe-zone convention already
established for this project's grade1 chunk dataset (see
data_preprocessing/motion_grades_chunk_5_dataset_hfiltered/build_chunk_dataset.py):
non-overlapping chunks of CHUNK_SIZE volumes, stepped by CHUNK_SIZE from the
start of the run; a chunk qualifies if its CHUNK_SIZE-1 internal FD values
(FD[start:start+CHUNK_SIZE-1], since FD is a frame-to-frame difference so has
one fewer value than volumes) have mean <= MEAN_FD_THRESH and max <=
MAX_FD_THRESH, AND no FD spike > SPIKE_THRESHOLD in the BUFFER_VOLUMES before
the chunk (a chunk immediately after a big head movement can still carry spin-
history/T1-relaxation artifacts even once FD itself has settled).

ROI timeseries for the "videos" task are already precomputed per full run
(create_roi_timeseries.py), so each qualifying chunk's FC is just a slice of
that array -- no volume loading needed.

Hierarchical averaging (Fisher-z throughout, only averaged back to r for
reporting): chunk -> run (mean over a run's qualifying chunks) -> session
(mean over a subject's sessions that have >=1 qualifying run) -> subject
(mean over sessions) -> group (mean over subjects, plus the one-sample FDR
test). This gives every qualifying subject equal weight in the final result,
regardless of how many low-motion chunks they happened to have.

Output schema matches run_low_motion_reference_analysis's (fc_mean_z.npy,
fc_p.npy, fc_q.npy, fc_significant.npy, fc_significance_summary.txt), so the
existing plot_denoising_qc.py plotting functions work on it unchanged.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from losses import _pearson_corr_matrix
from denoising_evaluation import EvalConfig, roi_ts_path, fisher_z, PIPELINE_EVAL_ROOT

CHUNK_SIZE = 20
SPIKE_THRESHOLD = 0.3
BUFFER_VOLUMES = 15
MEAN_FD_THRESH = 0.2
MAX_FD_THRESH = 0.25
EDGE_ALPHA = 0.05

CHUNK_METADATA_CSV = (
    "/lustre/disk/home/users/mfaizan/motion_correction/cycleGANS_on_2d_images/"
    "pytorch-CycleGAN-and-pix2pix/data_preprocessing/motion_grades_chunk_5_dataset_hfiltered/"
    "chunk_metadata.csv"
)
SOURCE_ROOT = (
    "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
    "faizan_motion_correction_dataset/brain_masked_cropped_hfiltered_normalized_to_common_space"
)
ROI_TIMESERIES_ROOT = (
    "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
    "faizan_motion_correction_dataset/roi_timeseries_hfiltered_videos"
)
OUTPUT_DIR = os.path.join(PIPELINE_EVAL_ROOT, "video_low_motion_fc")


def select_video_runs() -> pd.DataFrame:
    """All video-task runs (2mo, non-A), one row per run -- unlike
    select_first_runs() this keeps every run/session, since a subject's
    low-motion chunks can come from any of them."""
    meta = pd.read_csv(CHUNK_METADATA_CSV)
    v = meta[(meta["task"] == "videos") & (meta["age_group"] == "2mo")]
    v = v[~v["subject_id"].str.endswith("A")]
    cols = ["subject_id", "session_id", "run_id", "source_volume_path", "fd_path"]
    return v[cols].drop_duplicates().sort_values(["subject_id", "session_id", "run_id"])


def has_preceding_spike(fd: np.ndarray, start: int) -> bool:
    lo = max(0, start - BUFFER_VOLUMES)
    return bool((fd[lo:start] > SPIKE_THRESHOLD).any())


def find_qualifying_chunks(fd: np.ndarray) -> list:
    """Non-overlapping CHUNK_SIZE-volume grid, same convention as
    build_chunk_dataset.py's grade1 chunking. Returns [(start, end_inclusive), ...]."""
    T = len(fd) + 1
    chunks = []
    for start in range(0, T - CHUNK_SIZE + 1, CHUNK_SIZE):
        end = start + CHUNK_SIZE - 1
        fd_chunk = fd[start:start + CHUNK_SIZE - 1]
        if len(fd_chunk) < CHUNK_SIZE - 1:
            continue
        if fd_chunk.mean() <= MEAN_FD_THRESH and fd_chunk.max() <= MAX_FD_THRESH \
                and not has_preceding_spike(fd, start):
            chunks.append((start, end))
    return chunks


def run_level_fc(roi_ts: np.ndarray, chunks: list, iu) -> np.ndarray:
    """Mean Fisher-z FC (edge vector) across a run's qualifying chunks."""
    zs = []
    for start, end in chunks:
        window = roi_ts[start:end + 1, :]  # (CHUNK_SIZE, n_rois)
        fc = _pearson_corr_matrix(torch.from_numpy(window).float()).numpy()
        zs.append(fisher_z(fc[iu]))
    return np.mean(zs, axis=0)


def main():
    config = EvalConfig(
        chunk_metadata_csv=CHUNK_METADATA_CSV, source_root=SOURCE_ROOT,
        roi_timeseries_root=ROI_TIMESERIES_ROOT, atlas_path="", output_dir=OUTPUT_DIR,
    )
    runs = select_video_runs()
    print(f"{len(runs)} video runs to scan (2mo, non-A)")

    n_rois, iu = None, None
    # subject_id -> session_id -> list of run-level Fisher-z edge vectors
    subject_sessions: dict = {}
    chunk_scan_rows = []

    for row in runs.itertuples(index=False):
        fd = pd.read_csv(row.fd_path)["FramewiseDisplacement"].to_numpy()
        chunks = find_qualifying_chunks(fd)
        chunk_scan_rows.append({
            "subject_id": row.subject_id, "session_id": row.session_id, "run_id": row.run_id,
            "n_chunks": len(chunks),
        })
        if not chunks:
            continue

        ts_path = roi_ts_path(row.source_volume_path, config)
        if not os.path.exists(ts_path):
            print(f"  WARNING: no ROI timeseries for {row.subject_id}/{row.session_id}/{row.run_id}, skipping")
            continue
        roi_ts = np.load(ts_path)
        if iu is None:
            n_rois = roi_ts.shape[1]
            iu = np.triu_indices(n_rois, k=1)

        run_z = run_level_fc(roi_ts, chunks, iu)
        subject_sessions.setdefault(row.subject_id, {}).setdefault(row.session_id, []).append(run_z)

    # Session-level: mean of its runs. Subject-level: mean of its sessions.
    subject_z = {}
    for subject_id, sessions in subject_sessions.items():
        session_means = [np.mean(run_zs, axis=0) for run_zs in sessions.values()]
        subject_z[subject_id] = np.mean(session_means, axis=0)

    n = len(subject_z)
    print(f"{n} subjects with >=1 qualifying low-motion video chunk "
          f"(chunk_size={CHUNK_SIZE}, spike>{SPIKE_THRESHOLD}mm buffer={BUFFER_VOLUMES}vol, "
          f"mean_fd<={MEAN_FD_THRESH}, max_fd<={MAX_FD_THRESH})")

    edges = np.stack(list(subject_z.values()))  # (n_subjects, n_edges)
    _, p = stats.ttest_1samp(edges, popmean=0.0, axis=0)
    mean_z = edges.mean(axis=0)

    significant = np.zeros_like(p, dtype=bool)
    q = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    significant[valid], q[valid], _, _ = multipletests(p[valid], alpha=EDGE_ALPHA, method="fdr_bh")

    n_sig = int(significant.sum())
    n_edges = len(p)
    median_abs_fc_r = float(np.median(np.abs(np.tanh(mean_z))))
    print(f"FC one-sample test: {n_sig:,}/{n_edges:,} FDR-significant ({100 * n_sig / n_edges:.2f}%), "
          f"median |r| = {median_abs_fc_r:.4f}")

    mean_z_mat = np.full((n_rois, n_rois), np.nan)
    p_mat = np.full((n_rois, n_rois), np.nan)
    q_mat = np.full((n_rois, n_rois), np.nan)
    sig_mat = np.zeros((n_rois, n_rois), dtype=bool)
    for mat, vec in ((mean_z_mat, mean_z), (p_mat, p), (q_mat, q)):
        mat[iu] = vec
        mat.T[iu] = vec
    sig_mat[iu] = significant
    sig_mat.T[iu] = significant

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "fc_mean_z.npy"), mean_z_mat)
    np.save(os.path.join(OUTPUT_DIR, "fc_p.npy"), p_mat)
    np.save(os.path.join(OUTPUT_DIR, "fc_q.npy"), q_mat)
    np.save(os.path.join(OUTPUT_DIR, "fc_significant.npy"), sig_mat)
    pd.DataFrame(chunk_scan_rows).to_csv(os.path.join(OUTPUT_DIR, "chunk_scan.csv"), index=False)
    pd.DataFrame({"subject_id": list(subject_z.keys())}).to_csv(
        os.path.join(OUTPUT_DIR, "qualifying_subjects.csv"), index=False
    )

    with open(os.path.join(OUTPUT_DIR, "fc_significance_summary.txt"), "w") as f:
        f.write(f"n_subjects: {n}\n")
        f.write(f"chunk_size: {CHUNK_SIZE}\n")
        f.write(f"spike_threshold: {SPIKE_THRESHOLD}\n")
        f.write(f"buffer_volumes: {BUFFER_VOLUMES}\n")
        f.write(f"mean_fd_thresh: {MEAN_FD_THRESH}\n")
        f.write(f"max_fd_thresh: {MAX_FD_THRESH}\n")
        f.write(f"n_edges: {n_edges}\n")
        f.write(f"n_fc_significant: {n_sig}\n")
        f.write(f"pct_fc_significant: {100 * n_sig / n_edges:.4f}\n")
        f.write(f"mean_z_min: {np.nanmin(mean_z):.4f}\n")
        f.write(f"mean_z_max: {np.nanmax(mean_z):.4f}\n")
        f.write(f"median_abs_fc_r: {median_abs_fc_r:.4f}\n")
    print(f"saved results -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
