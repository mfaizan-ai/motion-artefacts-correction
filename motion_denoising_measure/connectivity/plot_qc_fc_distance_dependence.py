import argparse
import os
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

sys.path.insert(0, os.path.dirname(__file__))
from compute_distance_qcfc import DistanceQCFCConfig, build_distance_matrix


@dataclass
class QCFCDistanceDependenceConfig:
    qc_fc_dir: str
    atlas_path: str
    output_dirname: str = "qc-fc-dd"

    @property
    def output_dir(self) -> str:
        return os.path.join(self.qc_fc_dir, self.output_dirname)

    @property
    def distance_matrix_path(self) -> str:
        return os.path.join(self.qc_fc_dir, "distance_matrix.npy")


def get_distance_matrix(config: QCFCDistanceDependenceConfig) -> np.ndarray:
    if os.path.exists(config.distance_matrix_path):
        return np.load(config.distance_matrix_path)
    dist = build_distance_matrix(DistanceQCFCConfig(config.qc_fc_dir, config.atlas_path))
    np.save(config.distance_matrix_path, dist)
    return dist


def plot_distance_dependence(config: QCFCDistanceDependenceConfig) -> None:
    r = np.load(os.path.join(config.qc_fc_dir, "qc_fc_r.npy"))
    dist = get_distance_matrix(config)

    n_rois = r.shape[0]
    iu = np.triu_indices(n_rois, k=1)
    x = dist[iu]
    y = r[iu]

    corr, p = stats.pearsonr(x, y)
    print(f"distance-QC-FC correlation: r={corr:.4f}, p={p:.4e}")

    smoothed = lowess(y, x, frac=0.3)

    os.makedirs(config.output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, s=2, alpha=0.15, color="gray")
    ax.plot(smoothed[:, 0], smoothed[:, 1], color="crimson", linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("ROI-pair distance (mm)")
    ax.set_ylabel("QC-FC (r)")
    ax.set_title(f"QC-FC distance dependence\nr={corr:.3f}, p={p:.1e}")
    fig.tight_layout()

    out_path = os.path.join(config.output_dir, "qc_fc_distance_dependence.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")

def parse_args() -> QCFCDistanceDependenceConfig:
    parser = argparse.ArgumentParser(description="QC-FC distance-dependence scatter + LOWESS fit")
    parser.add_argument(
        "--qc_fc_dir", default=(
            "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
            "faizan_motion_correction_dataset/roi_timeseries_hfiltered_videos/qc_fc_2mo_firstrun"
        ),
    )
    parser.add_argument(
        "--atlas_path", default=(
            "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
            "templates/rois/Schaefer2018_400Parcels_7Networks_order_space-nihpd-02-05_2mm.nii.gz"
        ),
    )
    parser.add_argument("--output_dirname", default="qc-fc-dd")
    args = parser.parse_args()
    return QCFCDistanceDependenceConfig(**vars(args))


if __name__ == "__main__":
    plot_distance_dependence(parse_args())
