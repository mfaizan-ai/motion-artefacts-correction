import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from netneurotools.modularity import consensus_modularity


def signed_asymmetric_q(W, communities, gamma=1.0):
    """
    Evaluate signed asymmetric modularity Q for a fixed partition.

    Matches the negative_asym objective implemented by
    bctpy.community_louvain.
    """
    W = np.asarray(W, dtype=float).copy()
    ci = np.asarray(communities).reshape(-1)

    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("FC matrix must be square.")

    if len(ci) != W.shape[0]:
        raise ValueError("One community label is required per ROI.")

    if not np.all(np.isfinite(W)):
        raise ValueError("FC matrix contains NaN or Inf.")

    W = (W + W.T) / 2
    np.fill_diagonal(W, 0)

    W_pos = np.maximum(W, 0)
    W_neg = np.maximum(-W, 0)

    s_pos = W_pos.sum()
    s_neg = W_neg.sum()

    if s_pos == 0:
        raise ValueError("No positive weights in FC matrix.")

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


def subject_consensus_modularity(roi_timeseries, repeats=100, gamma=1.0, seed=12345):
    """
    Parameters
    ----------
    roi_timeseries : array, shape (timepoints, ROIs)

    Returns
    -------
    dict with Q, communities, FC, Q_runs, mean/sd optimization Q, zrand
    """
    ts = np.asarray(roi_timeseries, dtype=float)

    if ts.ndim != 2:
        raise ValueError("Time series must have shape (timepoints, ROIs).")

    fc = np.corrcoef(ts, rowvar=False)

    if not np.all(np.isfinite(fc)):
        raise ValueError("FC contains NaN/Inf. Check for constant ROI time series.")

    fc = (fc + fc.T) / 2
    np.fill_diagonal(fc, 0)

    consensus_ci, q_runs, zrand = consensus_modularity(
        adjacency=fc, gamma=gamma, B="negative_asym", repeats=repeats, seed=seed,
    )

    consensus_q = signed_asymmetric_q(fc, consensus_ci, gamma=gamma)

    return {
        "Q": consensus_q,
        "communities": consensus_ci,
        "FC": fc,
        "Q_runs": q_runs,
        "mean_optimization_Q": float(np.mean(q_runs)),
        "sd_optimization_Q": float(np.std(q_runs, ddof=1)),
        "zrand": zrand,
    }


@dataclass
class ModularityConfig:
    qc_fc_dir: str
    source_root: str
    roi_timeseries_root: str
    output_dirname: str = "modularity"
    repeats: int = 100
    gamma: float = 1.0
    seed: int = 12345

    @property
    def output_dir(self) -> str:
        return os.path.join(self.qc_fc_dir, self.output_dirname)


def roi_ts_path(source_volume_path: str, config: ModularityConfig) -> str:
    rel_dir = os.path.relpath(os.path.dirname(source_volume_path), config.source_root)
    return os.path.join(config.roi_timeseries_root, rel_dir, "roi_timeseries.npy")


def build_all_modularity(config: ModularityConfig) -> None:
    manifest = pd.read_csv(os.path.join(config.qc_fc_dir, "manifest.csv"))
    print(f"{len(manifest)} subjects")

    os.makedirs(config.output_dir, exist_ok=True)
    rows, failed = [], []
    for row in manifest.itertuples(index=False):
        try:
            roi_ts = np.load(roi_ts_path(row.source_volume_path, config))
            result = subject_consensus_modularity(
                roi_ts, repeats=config.repeats, gamma=config.gamma, seed=config.seed,
            )
            np.savez(
                os.path.join(config.output_dir, f"{row.subject_id}.npz"),
                Q=result["Q"], communities=result["communities"], Q_runs=result["Q_runs"],
            )
            rows.append({
                "subject_id": row.subject_id,
                "Q": result["Q"],
                "n_communities": np.unique(result["communities"]).size,
                "mean_optimization_Q": result["mean_optimization_Q"],
                "sd_optimization_Q": result["sd_optimization_Q"],
            })
        except Exception as e:
            failed.append((row.subject_id, str(e)))

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(config.output_dir, "modularity_summary.csv"), index=False)

    print(f"succeeded: {len(rows)} / {len(manifest)}")
    print(f"mean Q: {summary['Q'].mean():.4f}  mean n_communities: {summary['n_communities'].mean():.2f}")
    for subject_id, err in failed:
        print(f"  FAILED {subject_id}: {err}")


def parse_args() -> ModularityConfig:
    parser = argparse.ArgumentParser(description="Consensus network modularity per subject")
    parser.add_argument(
        "--qc_fc_dir", default=(
            "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
            "faizan_motion_correction_dataset/roi_timeseries_hfiltered_videos/qc_fc_2mo_firstrun"
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
    parser.add_argument("--output_dirname", default="modularity")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    return ModularityConfig(**vars(args))


if __name__ == "__main__":
    build_all_modularity(parse_args())
