"""
denoising_evaluation.py
=========================
Consolidated denoising-quality evaluation across subjects: QC-FC (with FDR
and median |QC-FC|), QC-FC distance-dependence, network modularity Q
(correlated with mean FD), and nipype-convention DVARS/tSNR. One pass over
subjects computes all of these; --plot additionally saves the QC-FC matrix,
QC-FC-DD scatter+fit, modularity Q violin, and Q-vs-FD scatter.

DVARS/tSNR reuse nipype's exact formulas (verified to match nipype's own
file-writing ComputeDVARS/TSNR nodes to floating-point precision) but take
already-loaded arrays and write nothing to disk -- nipype's bare functions
either don't exist (TSNR) or require file paths (compute_dvars), neither of
which fits a per-subject loop over 100+ runs.
"""
import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from netneurotools.modularity import consensus_modularity
from nilearn import plotting
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from losses import _pearson_corr_matrix

# All evaluation runs (raw baseline, each corrected pipeline, ...) land here,
# one uniquely-named subfolder per pipeline -- so results never get scattered
# across the dataset root or silently overwritten by the next run.
PIPELINE_EVAL_ROOT = (
    "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
    "faizan_motion_correction_dataset/motion_denoising_pipeline_level_evaluation"
)

@dataclass
class EvalConfig:
    chunk_metadata_csv: str
    source_root: str
    roi_timeseries_root: str
    atlas_path: str
    connectome_atlas_path: str
    output_dir: str
    edge_alpha: float = 0.05
    edge_percentile: str = "98%"
    repeats: int = 100
    gamma: float = 1.0
    seed: int = 12345
    plot: bool = False
    max_subjects: Optional[int] = None
    log_path: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "connectivity", "results", "run_log.csv"
    )
    run_label: str = "raw"

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
        "edge_percentile": config.edge_percentile,
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


def nipype_dvars(
    func: np.ndarray,
    mask: np.ndarray,
    remove_zerovariance: bool = True,
    intensity_normalization: float = 1000.0,
    variance_tol: float = 1e-7,
) -> float:
    """
    Mean non-standardized DVARS (Power et al. 2012), nipype's exact formula
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
    dvars_nstd = np.sqrt(np.square(func_diff).mean(axis=0))
    return float(dvars_nstd.mean())


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


def run_evaluation(config: EvalConfig) -> dict:
    runs = select_first_runs(config)
    if config.max_subjects is not None:
        runs = runs.head(config.max_subjects)
    n = len(runs)
    print(f"{n} subjects")

    n_rois, edge_rows, fds, q_values, subject_ids = None, [], [], [], []
    dvars_values, tsnr_values = [], []
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

        # DVARS/tSNR need the raw run volume, not the ROI timeseries used above
        func, mask = load_run_volume_and_mask(row.source_volume_path)
        dvars_values.append(nipype_dvars(func, mask))
        tsnr_values.append(nipype_tsnr(func, mask))

    edges = np.stack(edge_rows)
    fd = np.array(fds)
    q_values = np.array(q_values)
    dvars_values = np.array(dvars_values)
    tsnr_values = np.array(tsnr_values)

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

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "figures"), exist_ok=True)
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
        "dvars": dvars_values, "tsnr": tsnr_values,
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
    })

    if config.plot:
        make_plots(config, r_mat, fdr_mat, dist_edges, qcfc_r, q_values, fd, dvars_values, tsnr_values)

    return {
        "qcfc_r": r_mat, "qcfc_fdr_significant": fdr_mat, "median_abs_qcfc": median_abs_qcfc,
        "dd_r": dd_r, "dd_p": dd_p, "Q": q_values, "mean_fd": fd,
        "q_fd_r": q_fd_r, "q_fd_p": q_fd_p,
        "dvars": dvars_values, "tsnr": tsnr_values,
    }

def make_plots(config: EvalConfig, r_mat, fdr_mat, dist_edges, qcfc_r, q_values, fd,
               dvars_values, tsnr_values) -> None:
    fig_dir = os.path.join(config.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(r_mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("QC-FC matrix")
    ax.set_xlabel("ROI")
    ax.set_ylabel("ROI")
    fig.colorbar(im, ax=ax, label="QC-FC (r)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "qc_fc_matrix.png"), dpi=150)
    plt.close(fig)

    smoothed = lowess(qcfc_r, dist_edges, frac=0.3)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(dist_edges, qcfc_r, s=2, alpha=0.15, color="gray")
    ax.plot(smoothed[:, 0], smoothed[:, 1], color="crimson", linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("ROI-pair distance (mm)")
    ax.set_ylabel("QC-FC (r)")
    ax.set_title("QC-FC distance dependence")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "qc_fc_distance_dependence.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 6))
    sns.violinplot(y=q_values, ax=ax, width=0.7, inner="quartile", color="#54A24B", alpha=0.6)
    sns.stripplot(y=q_values, ax=ax, color="black", size=3, jitter=0.15, alpha=0.4)
    ax.set_xlabel(f"Subjects (n={len(q_values)})")
    ax.set_ylabel("Modularity Q")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Modularity Q")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "modularity_q_violin.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(fd, q_values, s=15, alpha=0.7)
    slope, intercept, *_ = stats.linregress(fd, q_values)
    xs = np.linspace(fd.min(), fd.max(), 100)
    ax.plot(xs, slope * xs + intercept, color="crimson")
    ax.set_xlabel("Mean FD")
    ax.set_ylabel("Modularity Q")
    ax.set_title("Modularity Q vs mean FD")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "modularity_q_vs_fd.png"), dpi=150)
    plt.close(fig)

    sim_df = pd.DataFrame({"Mean_DVARS": dvars_values, "Mean_tSNR": tsnr_values})
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    metrics = [
        ("Mean_DVARS", "Mean DVARS", "#4C78A8"),
        ("Mean_tSNR", "Mean tSNR", "#F58518"),
    ]
    for ax, (column, ylabel, colour) in zip(axes, metrics):
        sns.violinplot(
            data=sim_df, y=column, ax=ax, width=0.7,
            inner="quartile", color=colour, alpha=0.6,
        )
        sns.stripplot(
            data=sim_df, y=column, ax=ax,
            color="black", size=3, jitter=0.15, alpha=0.4,
        )
        ax.set_xlabel(f"Subjects (n={len(sim_df)})")
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle(f"DVARS and tSNR (nipype convention, n={len(sim_df)} subjects)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "dvars_tsnr_violin.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    if not fdr_mat.any():
        print("no FDR-significant edges -- skipping connectome plot")
    else:
        # Nilearn-style connectome: default schematic glass brain (no real
        # anatomy), node coords from the standard adult FSLMNI152 Schaefer
        # atlas -- a different template than config.atlas_path (which stays
        # the correct infant-space atlas for the distance matrix/QC-FC-DD
        # above). ROI identity (label order) still matches; only the display
        # coordinates come from the adult template, for a familiar/legible
        # schematic rather than an anatomically-exact infant background.
        coords, _ = plotting.find_parcellation_cut_coords(
            config.connectome_atlas_path, return_label_names=True
        )
        adjacency = np.where(fdr_mat, r_mat, 0)
        n_significant = int(fdr_mat.sum() / 2)
        top_pct = 100 - int(config.edge_percentile.rstrip("%"))

        fig = plt.figure(figsize=(9, 7))
        plotting.plot_connectome(
            adjacency_matrix=adjacency,
            node_coords=coords,
            display_mode="ortho",
            node_color="#555555",
            node_size=8,
            edge_cmap="RdBu_r",
            edge_threshold=config.edge_percentile,
            edge_kwargs={"linewidth": 1.0, "alpha": 0.85},
            title=f"QC-FC -- top {top_pct}% strongest of {n_significant:,} FDR-significant edges",
            annotate=False,
            colorbar=True,
            figure=fig,
        )
        fig.savefig(os.path.join(fig_dir, "connectome.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"saved plots -> {config.output_dir}")

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
        "--atlas_path", default=(
            "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
            "templates/rois/Schaefer2018_400Parcels_7Networks_order_space-nihpd-02-05_2mm.nii.gz"
        ),
    )
    parser.add_argument(
        "--connectome_atlas_path", default=(
            "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
            "templates/rois/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
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
    parser.add_argument("--edge_percentile", default="98%")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--plot", action="store_true")
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
