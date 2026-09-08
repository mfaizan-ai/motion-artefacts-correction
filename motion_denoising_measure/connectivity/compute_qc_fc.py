"""
compute_qc_fc.py
=================
QC-FC: first video run (first session, first run) per 2mo subject (excluding
subject_id ending in "A"). Correlates each ROI-pair edge's FC across subjects
against subject mean FD. Saves the QC-FC r/p matrices and the run manifest used.
"""
import argparse
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
import torch
from scipy import stats
from statsmodels.stats.multitest import multipletests
from losses import _pearson_corr_matrix


@dataclass
class QCFCConfig:
    chunk_metadata_csv: str
    source_root: str
    roi_timeseries_root: str
    output_dirname: str = "qc_fc_2mo_firstrun"

    @property
    def output_dir(self) -> str:
        return os.path.join(self.roi_timeseries_root, self.output_dirname)


def select_first_runs(config: QCFCConfig) -> pd.DataFrame:
    meta = pd.read_csv(config.chunk_metadata_csv)
    v = meta[(meta["task"] == "videos") & (meta["age_group"] == "2mo")]
    v = v[~v["subject_id"].str.endswith("A")]
    cols = ["subject_id", "session_id", "run_id", "source_volume_path", "fd_path"]
    runs = v[cols].drop_duplicates().sort_values(["subject_id", "session_id", "run_id"])
    return runs.groupby("subject_id", as_index=False).first()


def roi_ts_path(source_volume_path: str, config: QCFCConfig) -> str:
    rel_dir = os.path.relpath(os.path.dirname(source_volume_path), config.source_root)
    return os.path.join(config.roi_timeseries_root, rel_dir, "roi_timeseries.npy")


def mean_fd(fd_path: str) -> float:
    return pd.read_csv(fd_path)["FramewiseDisplacement"].mean()


def fisher_z(r: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(r, -1 + 1e-7, 1 - 1e-7))


def apply_fdr(p: np.ndarray, alpha: float = 0.05) -> tuple:
    valid = np.isfinite(p)

    fdr_significant = np.zeros_like(p, dtype=bool)
    fdr_p = np.full_like(p, np.nan, dtype=float)

    fdr_significant[valid], fdr_p[valid], _, _ = multipletests(
        p[valid], alpha=alpha, method="fdr_bh"
    )

    print(f"FDR-significant edges: {fdr_significant.sum()} / {valid.sum()}")
    print(
        f"Percentage significant: "
        f"{100 * fdr_significant.sum() / valid.sum():.2f}%"
    )
    return fdr_significant, fdr_p



def build_qc_fc(config: QCFCConfig) -> None:
    runs = select_first_runs(config)
    print(f"{len(runs)} subjects")

    fds, edge_rows, n_rois = [], [], None
    for row in runs.itertuples(index=False):
        roi_ts = np.load(roi_ts_path(row.source_volume_path, config))
        fc = _pearson_corr_matrix(torch.from_numpy(roi_ts).float()).numpy()
        n_rois = fc.shape[0]
        iu = np.triu_indices(n_rois, k=1)
        edge_rows.append(fisher_z(fc[iu]))
        fds.append(mean_fd(row.fd_path))

    edges = np.stack(edge_rows)                 #(N, E)
    fd = np.array(fds)                          #(N,)
    n = len(fd)

    if n < 3:
        raise ValueError("At least 3 independent subjects are required")
    if not np.all(np.isfinite(fd)):
        raise ValueError("Mean FD contains NaN or infinite values")
    if not np.all(np.isfinite(edges)):
        raise ValueError("FC edges contain NaN or infinite values")
    if np.std(fd) == 0:
        raise ValueError("Mean FD has zero variance across subjects")

    fd_c = fd - fd.mean()
    edges_c = edges - edges.mean(axis=0, keepdims=True)
    r = (edges_c.T @ fd_c) / np.sqrt((edges_c ** 2).sum(axis=0) * (fd_c ** 2).sum())

    r_safe = np.clip(r, -1 + 1e-12, 1 - 1e-12)
    t = r_safe * np.sqrt((n - 2) / (1 - r_safe ** 2))
    p = 2 * stats.t.sf(np.abs(t), df=n - 2)

    fdr_significant, fdr_p = apply_fdr(p)
    median_abs_qcfc = np.median(np.abs(r))
    print(f"Median |QC-FC|: {median_abs_qcfc:.4f}")

    iu = np.triu_indices(n_rois, k=1)
    r_mat = np.full((n_rois, n_rois), np.nan)
    p_mat = np.full((n_rois, n_rois), np.nan)
    fdr_significant_mat = np.zeros((n_rois, n_rois), dtype=bool)
    fdr_p_mat = np.full((n_rois, n_rois), np.nan)
    r_mat[iu] = r
    r_mat.T[iu] = r
    p_mat[iu] = p
    p_mat.T[iu] = p
    fdr_significant_mat[iu] = fdr_significant
    fdr_significant_mat.T[iu] = fdr_significant
    fdr_p_mat[iu] = fdr_p
    fdr_p_mat.T[iu] = fdr_p

    os.makedirs(config.output_dir, exist_ok=True)
    np.save(os.path.join(config.output_dir, "qc_fc_r.npy"), r_mat)
    np.save(os.path.join(config.output_dir, "qc_fc_p.npy"), p_mat)
    np.save(os.path.join(config.output_dir, "qc_fc_fdr_significant.npy"), fdr_significant_mat)
    np.save(os.path.join(config.output_dir, "qc_fc_fdr_p.npy"), fdr_p_mat)
    runs.assign(mean_fd=fds).to_csv(os.path.join(config.output_dir, "manifest.csv"), index=False)
    with open(os.path.join(config.output_dir, "summary.txt"), "w") as f:
        f.write(f"n_subjects: {n}\n")
        f.write(f"fdr_significant_edges: {fdr_significant.sum()} / {len(r)}\n")
        f.write(f"pct_fdr_significant: {100 * fdr_significant.sum() / len(r):.2f}\n")
        f.write(f"median_abs_qcfc: {median_abs_qcfc:.4f}\n")

    print(f"saved qc_fc_r.npy, qc_fc_p.npy, qc_fc_fdr_significant.npy, qc_fc_fdr_p.npy, "
          f"manifest.csv, summary.txt -> {config.output_dir}")


def parse_args() -> QCFCConfig:
    parser = argparse.ArgumentParser(description="QC-FC: first-run FC vs mean FD across 2mo subjects")
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
    parser.add_argument("--output_dirname", default="qc_fc_2mo_firstrun")
    args = parser.parse_args()
    return QCFCConfig(**vars(args))


if __name__ == "__main__":
    build_qc_fc(parse_args())
