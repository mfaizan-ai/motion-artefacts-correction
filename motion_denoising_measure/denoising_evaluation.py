"""
denoising_evaluation.py
=========================
Consolidated denoising-quality evaluation across subjects: QC-FC (with FDR
and median |QC-FC|), QC-FC distance-dependence, and network modularity Q
(correlated with mean FD). One pass over subjects computes all three;
--plot additionally saves the QC-FC matrix, QC-FC-DD scatter+fit, modularity
Q violin, and Q-vs-FD scatter.
"""
import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional


import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from netneurotools.modularity import consensus_modularity
from nilearn import plotting
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from losses import _pearson_corr_matrix


@dataclass
class EvalConfig:
    chunk_metadata_csv: str
    source_root: str
    roi_timeseries_root: str
    atlas_path: str
    t1_path: str
    output_dir: str
    edge_alpha: float = 0.05
    edge_percentile: str = "98%"
    repeats: int = 100
    gamma: float = 1.0
    seed: int = 12345
    plot: bool = False
    max_subjects: Optional[int] = None


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


def run_evaluation(config: EvalConfig) -> dict:
    runs = select_first_runs(config)
    if config.max_subjects is not None:
        runs = runs.head(config.max_subjects)
    n = len(runs)
    print(f"{n} subjects")

    n_rois, edge_rows, fds, q_values, subject_ids = None, [], [], [], []
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

    edges = np.stack(edge_rows)
    fd = np.array(fds)
    q_values = np.array(q_values)

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
    pd.DataFrame({"subject_id": subject_ids, "mean_fd": fd, "Q": q_values}).to_csv(
        os.path.join(config.output_dir, "manifest.csv"), index=False
    )
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
    print(f"saved results -> {config.output_dir}")

    if config.plot:
        make_plots(config, r_mat, fdr_mat, dist_edges, qcfc_r, q_values, fd)

    return {
        "qcfc_r": r_mat, "qcfc_fdr_significant": fdr_mat, "median_abs_qcfc": median_abs_qcfc,
        "dd_r": dd_r, "dd_p": dd_p, "Q": q_values, "mean_fd": fd,
        "q_fd_r": q_fd_r, "q_fd_p": q_fd_p,
    }


def make_plots(config: EvalConfig, r_mat, fdr_mat, dist_edges, qcfc_r, q_values, fd) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(r_mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("QC-FC matrix")
    ax.set_xlabel("ROI")
    ax.set_ylabel("ROI")
    fig.colorbar(im, ax=ax, label="QC-FC (r)")
    fig.tight_layout()
    fig.savefig(os.path.join(config.output_dir, "qc_fc_matrix.png"), dpi=150)
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
    fig.savefig(os.path.join(config.output_dir, "qc_fc_distance_dependence.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 6))
    ax.violinplot(q_values, showmedians=True)
    ax.set_ylabel("Modularity Q")
    ax.set_xticks([])
    ax.set_title(f"Modularity Q (n={len(q_values)})")
    fig.tight_layout()
    fig.savefig(os.path.join(config.output_dir, "modularity_q_violin.png"), dpi=150)
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
    fig.savefig(os.path.join(config.output_dir, "modularity_q_vs_fd.png"), dpi=150)
    plt.close(fig)

    if not fdr_mat.any():
        print("no FDR-significant edges -- skipping connectome plot")
    else:
        atlas_img = nib.load(config.atlas_path)
        coords, _ = plotting.find_parcellation_cut_coords(atlas_img, return_label_names=True)
        adjacency = np.where(fdr_mat, r_mat, 0)
        degree = fdr_mat.sum(axis=1)
        node_size = 2 + 8 * (degree / degree.max())

        top_pct = 100 - int(config.edge_percentile.rstrip("%"))
        t1_img = nib.load(config.t1_path)
        display = plotting.plot_glass_brain(
            t1_img, display_mode="ortho", black_bg=False, colorbar=False,
            title=f"QC-FC -- top {top_pct}% strongest FDR-significant edges",
            alpha=0.3, cmap="gray_r",
        )
        display.add_graph(
            adjacency, coords,
            node_color="dimgray", node_size=node_size,
            edge_cmap="RdBu_r", edge_threshold=config.edge_percentile,
            edge_kwargs={"linewidth": 1.0, "alpha": 0.9},
            colorbar=True,
        )
        display.savefig(os.path.join(config.output_dir, "connectome.png"), dpi=150)
        display.close()

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
        "--t1_path", default=(
            "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
            "templates/mask/nihpd_asym_02-05_t1w_masked_2mm.nii.gz"
        ),
    )
    parser.add_argument(
        "--output_dir", default=(
            "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
            "faizan_motion_correction_dataset/roi_timeseries_hfiltered_videos/denoising_eval_2mo_firstrun"
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
    args = parser.parse_args()

    if args.debug:
        args.max_subjects = 4
        args.repeats = 5
        print("DEBUG mode: max_subjects=4, repeats=5")
    del args.debug

    return EvalConfig(**vars(args))


if __name__ == "__main__":
    run_evaluation(parse_args())
