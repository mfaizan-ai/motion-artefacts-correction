"""
denoising_evaluation.py
=========================
Consolidated denoising-quality evaluation across subjects: QC-FC (with FDR
and median |QC-FC|), QC-FC distance-dependence, network modularity Q
(correlated with mean FD), and nipype-convention DVARS/tSNR. One pass over
subjects computes all of these and writes the results (manifest.csv,
qc_fc_*.npy, distance_matrix.npy, summary.txt) to output_dir.

This script only computes and saves data -- see plot_denoising_qc.py for
generating figures from these saved results.
"""
import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from netneurotools.modularity import consensus_modularity
from nilearn import plotting
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from losses import _pearson_corr_matrix

PIPELINE_EVAL_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "motion_denoising_pipeline_level_evaluation"
)

@dataclass
class EvalConfig:
    chunk_metadata_csv: str
    source_root: str
    roi_timeseries_root: str
    atlas_path: str
    output_dir: str
    edge_alpha: float = 0.05
    repeats: int = 100
    gamma: float = 1.0
    seed: int = 12345
    max_subjects: Optional[int] = None
    log_path: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "connectivity", "results", "run_log.csv"
    )
    run_label: str = "raw"
    # Set to a denoise_runs.py output_root (e.g. motion_corrected_st_v4) to
    # evaluate a corrected pipeline's DVARS/tSNR/global-signal-std instead of
    # the raw volumes -- source_root stays pointed at the RAW tree either way.
    corrected_root: Optional[str] = None

def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def append_run_log(config: EvalConfig, n: int, metrics: dict) -> None:
    """
    One row per invocation, appended (never overwritten) -- so every run's
    parameters, code version, and headline results stay queryable even after
    output_dir contents get replaced by a later run.
    """
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_label": config.run_label,
        "git_commit": get_git_commit(),
        "output_dir": config.output_dir,
        "n_subjects": n,
        "repeats": config.repeats,
        "gamma": config.gamma,
        "edge_alpha": config.edge_alpha,
        **metrics,
    }
    os.makedirs(os.path.dirname(config.log_path), exist_ok=True)
    header = not os.path.exists(config.log_path)
    pd.DataFrame([row]).to_csv(config.log_path, mode="a", header=header, index=False)
    print(f"logged run -> {config.log_path}")

def select_first_runs(config: EvalConfig) -> pd.DataFrame:
    meta = pd.read_csv(config.chunk_metadata_csv)
    v = meta[(meta["task"] == "videos") & (meta["age_group"] == "2mo")]
    v = v[~v["subject_id"].str.endswith("A")]
    cols = ["subject_id", "session_id", "run_id", "source_volume_path", "fd_path"]
    runs = v[cols].drop_duplicates().sort_values(["subject_id", "session_id", "run_id"])
    return runs.groupby("subject_id", as_index=False).first()

def roi_ts_path(source_volume_path: str, config: EvalConfig) -> str:
    rel_dir = os.path.relpath(os.path.dirname(source_volume_path), config.source_root)
    return os.path.join(config.roi_timeseries_root, rel_dir, "roi_timeseries.npy")

def mean_fd(fd_path: str) -> float:
    return pd.read_csv(fd_path)["FramewiseDisplacement"].mean()


def fisher_z(r: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(r, -1 + 1e-7, 1 - 1e-7))

def fisher_mean(r: np.ndarray) -> float:
    """Mean correlation via the Fisher-z transform, NaN-safe (a constant-
    variance run's undefined correlation is skipped, not propagated)."""
    return float(np.tanh(np.nanmean(fisher_z(r))))

def signed_asymmetric_q(W, communities, gamma=1.0):
    W = np.asarray(W, dtype=float).copy()
    ci = np.asarray(communities).reshape(-1)
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0)

    W_pos = np.maximum(W, 0)
    W_neg = np.maximum(-W, 0)
    s_pos = W_pos.sum()
    s_neg = W_neg.sum()

    k_pos = W_pos.sum(axis=1)
    B_pos = W_pos - gamma * np.outer(k_pos, k_pos) / s_pos

    if s_neg > 0:
        k_neg = W_neg.sum(axis=1)
        B_neg = W_neg - gamma * np.outer(k_neg, k_neg) / s_neg
    else:
        B_neg = np.zeros_like(W)

    B_signed = B_pos / s_pos - B_neg / (s_pos + s_neg)
    same_module = ci[:, None] == ci[None, :]
    return float(B_signed[same_module].sum())


def consensus_q_from_fc(fc: np.ndarray, config: EvalConfig) -> float:
    fc = (fc + fc.T) / 2
    fc = fc.copy()
    np.fill_diagonal(fc, 0)
    ci, _, _ = consensus_modularity(
        adjacency=fc, gamma=config.gamma, B="negative_asym",
        repeats=config.repeats, seed=config.seed,
    )
    return signed_asymmetric_q(fc, ci, gamma=config.gamma)


def build_distance_matrix(atlas_path: str) -> np.ndarray:
    atlas_img = nib.load(atlas_path)
    coords, _ = plotting.find_parcellation_cut_coords(atlas_img, return_label_names=True)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def load_run_volume_and_mask(source_volume_path: str) -> tuple:
    """(X, Y, Z, T) run volume + a brain mask derived from its own nonzero
    voxels at t=0 -- these runs are already brain-masked, so no separate
    mask file exists per run and none needs to be written."""
    func = np.float32(nib.load(source_volume_path).dataobj)
    mask = func[..., 0] != 0
    return func, mask


def corrected_volume_path(source_volume_path: str, source_root: str, corrected_root: str) -> str:
    """Raw source_volume_path -> its corrected counterpart, same convention
    as denoise_runs.py's output_path() (same relative dir, "_corrected"
    suffix)."""
    rel_dir = os.path.relpath(os.path.dirname(source_volume_path), source_root)
    fname = os.path.basename(source_volume_path).replace(".nii.gz", "_corrected.nii.gz")
    return os.path.join(corrected_root, rel_dir, fname)


def nipype_dvars_timeseries(
    func: np.ndarray,
    mask: np.ndarray,
    remove_zerovariance: bool = True,
    intensity_normalization: float = 1000.0,
    variance_tol: float = 1e-7,
) -> np.ndarray:
    """
    Per-frame non-standardized DVARS (T-1 values), nipype's exact formula
    (nipype.algorithms.confounds.compute_dvars) reimplemented on in-memory
    arrays instead of file paths -- verified to match nipype's own function
    to floating-point precision on real data.
    """
    mfunc = func[mask]
    if intensity_normalization != 0:
        mfunc = (mfunc / np.median(mfunc)) * intensity_normalization

    # Robust SD per voxel (IQR / 1.349), "lower" interpolation to match FSL
    func_sd = (
        np.percentile(mfunc, 75, axis=1, method="lower")
        - np.percentile(mfunc, 25, axis=1, method="lower")
    ) / 1.349

    if remove_zerovariance:
        keep = func_sd > variance_tol
        mfunc = mfunc[keep, :]

    func_diff = np.diff(mfunc, axis=1)
    return np.sqrt(np.square(func_diff).mean(axis=0))


def nipype_dvars(func: np.ndarray, mask: np.ndarray, **kwargs) -> float:
    """Mean non-standardized DVARS -- see nipype_dvars_timeseries."""
    return float(nipype_dvars_timeseries(func, mask, **kwargs).mean())


def compute_fd_dvars_correlation(fd, dvars):
    """
    Calculate the within-run correlation between FD and DVARS.

    Parameters
    ----------
    fd : array-like
        Framewise-displacement time series.

        It may contain:
        - T values, where the first value is usually zero, or
        - T-1 values already aligned with DVARS.

    dvars : array-like
        DVARS time series, normally containing T-1 values.

    Returns
    -------
    dict
        Pearson and Spearman correlations, p-values and the
        number of valid frame transitions.
    """

    fd = np.asarray(fd, dtype=float).squeeze()
    dvars = np.asarray(dvars, dtype=float).squeeze()

    if fd.ndim != 1 or dvars.ndim != 1:
        raise ValueError("FD and DVARS must be one-dimensional arrays.")

    # DVARS contains T-1 differences.
    # If FD contains T values, discard its first undefined/zero value.
    if len(fd) == len(dvars) + 1:
        fd = fd[1:]

    if len(fd) != len(dvars):
        raise ValueError(
            f"FD and DVARS are not aligned: "
            f"FD has {len(fd)} values and "
            f"DVARS has {len(dvars)} values."
        )

    # Remove pairs where either value is missing or infinite
    valid = np.isfinite(fd) & np.isfinite(dvars)

    fd_valid = fd[valid]
    dvars_valid = dvars[valid]

    if len(fd_valid) < 3:
        raise ValueError(
            "At least three valid FD-DVARS pairs are required."
        )

    # Correlation is undefined if either series is constant
    if np.std(fd_valid) == 0 or np.std(dvars_valid) == 0:
        return {
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "n_transitions": len(fd_valid)
        }

    pearson_r, pearson_p = stats.pearsonr(
        fd_valid,
        dvars_valid
    )

    spearman_rho, spearman_p = stats.spearmanr(
        fd_valid,
        dvars_valid
    )

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "n_transitions": len(fd_valid)
    }


def nipype_tsnr(func: np.ndarray, mask: np.ndarray) -> float:
    """
    Mean tSNR over brain voxels, nipype's exact formula
    (nipype.algorithms.confounds.TSNR, a file-writing-only Node with no bare
    function equivalent) reimplemented in memory -- verified to match
    nipype's own node output to floating-point precision on real data.
    """
    mean_img = func.mean(axis=-1)
    std_img = func.std(axis=-1)
    tsnr_img = np.zeros_like(mean_img)
    valid = std_img > 1e-3
    tsnr_img[valid] = mean_img[valid] / std_img[valid]
    return float(tsnr_img[mask].mean())


def global_signal_std(func: np.ndarray, mask: np.ndarray) -> float:
    """Std over time of the whole-brain mean signal (global signal), same
    definition as train.py's compute_global_signal_std."""
    global_signal = func[mask].mean(axis=0)
    return float(global_signal.std())


def run_evaluation(config: EvalConfig) -> dict:
    runs = select_first_runs(config)
    if config.max_subjects is not None:
        runs = runs.head(config.max_subjects)
    n = len(runs)
    print(f"{n} subjects")

    n_rois, edge_rows, fds, q_values, subject_ids = None, [], [], [], []
    dvars_values, tsnr_values, gs_std_values = [], [], []
    fd_dvars_pearson_r, fd_dvars_pearson_p = [], []
    fd_dvars_spearman_rho, fd_dvars_spearman_p, fd_dvars_n = [], [], []
    for row in runs.itertuples(index=False):
        roi_ts = np.load(roi_ts_path(row.source_volume_path, config))
        fc = _pearson_corr_matrix(torch.from_numpy(roi_ts).float()).numpy()
        n_rois = fc.shape[0]
        # this function returns the upper-triangular indices of a square matrix, excluding the diagonal.
        iu = np.triu_indices(n_rois, k=1)

        edge_rows.append(fisher_z(fc[iu]))
        fds.append(mean_fd(row.fd_path))
        q_values.append(consensus_q_from_fc(fc, config))
        subject_ids.append(row.subject_id)

        # DVARS/tSNR/global-signal-std need the run volume itself, not the
        # ROI timeseries used above. --corrected_root set -> load the
        # corrected volume instead of raw.
        volume_path = row.source_volume_path
        if config.corrected_root:
            volume_path = corrected_volume_path(volume_path, config.source_root, config.corrected_root)
        func, mask = load_run_volume_and_mask(volume_path)
        dvars_ts = nipype_dvars_timeseries(func, mask)
        dvars_values.append(float(dvars_ts.mean()))
        tsnr_values.append(nipype_tsnr(func, mask))
        gs_std_values.append(global_signal_std(func, mask))

        fd_ts = pd.read_csv(row.fd_path)["FramewiseDisplacement"].to_numpy()
        fd_dvars = compute_fd_dvars_correlation(fd_ts, dvars_ts)
        fd_dvars_pearson_r.append(fd_dvars["pearson_r"])
        fd_dvars_pearson_p.append(fd_dvars["pearson_p"])
        fd_dvars_spearman_rho.append(fd_dvars["spearman_rho"])
        fd_dvars_spearman_p.append(fd_dvars["spearman_p"])
        fd_dvars_n.append(fd_dvars["n_transitions"])

    edges = np.stack(edge_rows)
    fd = np.array(fds)
    q_values = np.array(q_values)
    dvars_values = np.array(dvars_values)
    tsnr_values = np.array(tsnr_values)
    gs_std_values = np.array(gs_std_values)
    fd_dvars_pearson_r = np.array(fd_dvars_pearson_r)
    fd_dvars_pearson_p = np.array(fd_dvars_pearson_p)
    fd_dvars_spearman_rho = np.array(fd_dvars_spearman_rho)
    fd_dvars_spearman_p = np.array(fd_dvars_spearman_p)
    fd_dvars_n = np.array(fd_dvars_n)

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
    qcfc_r = (edges_c.T @ fd_c) / np.sqrt((edges_c ** 2).sum(axis=0) * (fd_c ** 2).sum())

    r_safe = np.clip(qcfc_r, -1 + 1e-12, 1 - 1e-12)
    t = r_safe * np.sqrt((n - 2) / (1 - r_safe ** 2))
    qcfc_p = 2 * stats.t.sf(np.abs(t), df=n - 2)

    valid = np.isfinite(qcfc_p)
    fdr_significant = np.zeros_like(qcfc_p, dtype=bool)
    fdr_p = np.full_like(qcfc_p, np.nan)
    fdr_significant[valid], fdr_p[valid], _, _ = multipletests(
        qcfc_p[valid], alpha=config.edge_alpha, method="fdr_bh"
    )
    median_abs_qcfc = np.nanmedian(np.abs(qcfc_r))
    print(f"QC-FC: {fdr_significant.sum()}/{len(qcfc_r)} FDR-significant "
          f"({100 * fdr_significant.sum() / len(qcfc_r):.2f}%), median |r| = {median_abs_qcfc:.4f}")
    print(f"valid edges: {valid.sum()}, invalid/skipped edges: {(~valid).sum()}")

    iu = np.triu_indices(n_rois, k=1)
    dist = build_distance_matrix(config.atlas_path)
    assert dist.shape == (n_rois, n_rois), "distance matrix ROI count mismatch with edges"
    dist_edges = dist[iu]
    dd_r, dd_p = stats.pearsonr(dist_edges[valid], qcfc_r[valid])
    n_valid = valid.sum()
    print(f"QC-FC-DD: r={dd_r:.4f}, p={dd_p:.4e}, n_valid={n_valid}")

    q_fd_r, q_fd_p = stats.pearsonr(q_values, fd)
    print(f"Modularity Q: mean={q_values.mean():.4f}, sd={q_values.std(ddof=1):.4f}")
    print(f"Q vs mean FD: r={q_fd_r:.4f}, p={q_fd_p:.4e}")

    print(f"DVARS (nipype convention): mean={dvars_values.mean():.4f}, sd={dvars_values.std(ddof=1):.4f}")
    print(f"tSNR (nipype convention): mean={tsnr_values.mean():.4f}, sd={tsnr_values.std(ddof=1):.4f}")
    print(f"Global signal std: mean={gs_std_values.mean():.4f}, sd={gs_std_values.std(ddof=1):.4f}")
    print(f"FD-DVARS within-run Pearson r: median={np.nanmedian(fd_dvars_pearson_r):.4f}, "
          f"Fisher-z mean={fisher_mean(fd_dvars_pearson_r):.4f}")

    os.makedirs(config.output_dir, exist_ok=True)
    r_mat = np.full((n_rois, n_rois), np.nan)
    p_mat = np.full((n_rois, n_rois), np.nan)
    fdr_mat = np.zeros((n_rois, n_rois), dtype=bool)
    r_mat[iu] = qcfc_r
    r_mat.T[iu] = qcfc_r
    p_mat[iu] = qcfc_p
    p_mat.T[iu] = qcfc_p
    fdr_mat[iu] = fdr_significant
    fdr_mat.T[iu] = fdr_significant

    np.save(os.path.join(config.output_dir, "qc_fc_r.npy"), r_mat)
    np.save(os.path.join(config.output_dir, "qc_fc_p.npy"), p_mat)
    np.save(os.path.join(config.output_dir, "qc_fc_fdr_significant.npy"), fdr_mat)
    np.save(os.path.join(config.output_dir, "distance_matrix.npy"), dist)
    pd.DataFrame({
        "subject_id": subject_ids, "mean_fd": fd, "Q": q_values,
        "dvars": dvars_values, "tsnr": tsnr_values, "gs_std": gs_std_values,
        "fd_dvars_r": fd_dvars_pearson_r, "fd_dvars_p": fd_dvars_pearson_p,
        "fd_dvars_spearman_rho": fd_dvars_spearman_rho, "fd_dvars_spearman_p": fd_dvars_spearman_p,
        "fd_dvars_n_transitions": fd_dvars_n,
    }).to_csv(os.path.join(config.output_dir, "manifest.csv"), index=False)
    with open(os.path.join(config.output_dir, "summary.txt"), "w") as f:
        f.write(f"n_subjects: {n}\n")
        f.write(f"fdr_significant_edges: {fdr_significant.sum()} / {len(qcfc_r)}\n")
        f.write(f"median_abs_qcfc: {median_abs_qcfc:.4f}\n")
        f.write(f"qcfc_dd_r: {dd_r:.4f}\n")
        f.write(f"qcfc_dd_p: {dd_p:.4e}\n")
        f.write(f"modularity_Q_mean: {q_values.mean():.4f}\n")
        f.write(f"modularity_Q_sd: {q_values.std(ddof=1):.4f}\n")
        f.write(f"Q_vs_FD_r: {q_fd_r:.4f}\n")
        f.write(f"Q_vs_FD_p: {q_fd_p:.4e}\n")
        f.write(f"dvars_mean: {dvars_values.mean():.4f}\n")
        f.write(f"dvars_sd: {dvars_values.std(ddof=1):.4f}\n")
        f.write(f"tsnr_mean: {tsnr_values.mean():.4f}\n")
        f.write(f"tsnr_sd: {tsnr_values.std(ddof=1):.4f}\n")
        f.write(f"gs_std_mean: {gs_std_values.mean():.4f}\n")
        f.write(f"gs_std_sd: {gs_std_values.std(ddof=1):.4f}\n")
        f.write(f"fd_dvars_pearson_r_median: {np.nanmedian(fd_dvars_pearson_r):.4f}\n")
        f.write(f"fd_dvars_pearson_r_fisher_mean: {fisher_mean(fd_dvars_pearson_r):.4f}\n")
        f.write(f"fd_dvars_spearman_rho_median: {np.nanmedian(fd_dvars_spearman_rho):.4f}\n")
    print(f"saved results -> {config.output_dir}")
    append_run_log(config, n, {
        "fdr_significant_edges": int(fdr_significant.sum()),
        "total_edges": len(qcfc_r),
        "median_abs_qcfc": median_abs_qcfc,
        "qcfc_dd_r": dd_r,
        "qcfc_dd_p": dd_p,
        "modularity_Q_mean": q_values.mean(),
        "modularity_Q_sd": q_values.std(ddof=1),
        "Q_vs_FD_r": q_fd_r,
        "Q_vs_FD_p": q_fd_p,
        "dvars_mean": dvars_values.mean(),
        "dvars_sd": dvars_values.std(ddof=1),
        "tsnr_mean": tsnr_values.mean(),
        "tsnr_sd": tsnr_values.std(ddof=1),
        "gs_std_mean": gs_std_values.mean(),
        "gs_std_sd": gs_std_values.std(ddof=1),
        "fd_dvars_pearson_r_median": np.nanmedian(fd_dvars_pearson_r),
        "fd_dvars_pearson_r_fisher_mean": fisher_mean(fd_dvars_pearson_r),
        "fd_dvars_spearman_rho_median": np.nanmedian(fd_dvars_spearman_rho),
    })

    return {
        "qcfc_r": r_mat, "qcfc_fdr_significant": fdr_mat, "median_abs_qcfc": median_abs_qcfc,
        "dd_r": dd_r, "dd_p": dd_p, "Q": q_values, "mean_fd": fd,
        "q_fd_r": q_fd_r, "q_fd_p": q_fd_p,
        "dvars": dvars_values, "tsnr": tsnr_values, "gs_std": gs_std_values,
        "fd_dvars_r": fd_dvars_pearson_r, "fd_dvars_p": fd_dvars_pearson_p,
        "fd_dvars_spearman_rho": fd_dvars_spearman_rho, "fd_dvars_spearman_p": fd_dvars_spearman_p,
    }

def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Denoising evaluation: QC-FC, QC-FC-DD, network modularity")
    parser.add_argument(
        "--chunk_metadata_csv", default=(
            "/lustre/disk/home/users/mfaizan/motion_correction/cycleGANS_on_2d_images/"
            "pytorch-CycleGAN-and-pix2pix/data_preprocessing/motion_grades_chunk_5_dataset_hfiltered/"
            "chunk_metadata.csv"
        ),
    )
    parser.add_argument(
        # Keep pointed at the RAW tree even for a corrected-pipeline run --
        # only used to compute each run's relative subdir (see --corrected_root).
        "--source_root", default=(
            "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
            "faizan_motion_correction_dataset/brain_masked_cropped_hfiltered_normalized_to_common_space"
        ),
    )
    parser.add_argument(
        # For a corrected pipeline, point this at the ROI timeseries
        # extracted from the corrected volumes (create_roi_timeseries.py
        # --corrected_root), not the raw ones.
        "--roi_timeseries_root", default=(
            "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
            "faizan_motion_correction_dataset/roi_timeseries_hfiltered_videos"
        ),
    )
    parser.add_argument(
        # e.g. .../motion_corrected_st_v4 -- a denoise_runs.py output_root.
        "--corrected_root", default=None,
        help="Set to evaluate DVARS/tSNR/global-signal-std on a corrected pipeline instead of raw",
    )
    parser.add_argument(
        "--atlas_path", default=(
            "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
            "templates/rois/Schaefer2018_400Parcels_7Networks_order_space-nihpd-02-05_2mm.nii.gz"
        ),
    )
    parser.add_argument(
        "--pipeline_name", default="raw_data_each_sub_first_run",
        help=(
            "Unique subfolder name for this pipeline's results, always created under "
            f"{PIPELINE_EVAL_ROOT} (e.g. 'raw_data_each_sub_first_run', 'st_v4_corrected_each_sub_first_run')."
        ),
    )
    parser.add_argument("--edge_alpha", type=float, default=0.05)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max_subjects", type=int, default=None,
                         help="Limit to the first N selected subjects (for quick testing)")
    parser.add_argument("--debug", action="store_true",
                         help="Fast debug run: max_subjects=4, repeats=5 (overrides those two)")
    parser.add_argument("--run_label", default="raw",
                         help="Tag for this run in run_log.csv, e.g. 'raw' or 'st_v4_corrected'")
    parser.add_argument("--log_path", default=EvalConfig.__dataclass_fields__["log_path"].default,
                         help="CSV to append this run's params + headline results to")
    args = parser.parse_args()
    if args.debug:
        args.max_subjects = 4
        args.repeats = 5
        print("DEBUG mode: max_subjects=4, repeats=5")
    del args.debug

    kwargs = vars(args)
    kwargs["output_dir"] = os.path.join(PIPELINE_EVAL_ROOT, kwargs.pop("pipeline_name"))
    return EvalConfig(**kwargs)


if __name__ == "__main__":
    run_evaluation(parse_args())
