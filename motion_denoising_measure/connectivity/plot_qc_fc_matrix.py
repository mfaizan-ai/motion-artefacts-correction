"""
plot_qc_fc_matrix.py
=====================
Heatmap of the QC-FC (400x400) correlation matrix, plus a printed summary of
FDR-significant edge count and median |QC-FC|. Saved next to the connectome plot.
"""
import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images_fdr_edges_significance_check")


@dataclass
class QCFCMatrixPlotConfig:
    qc_fc_dir: str
    output_dir: str = IMAGES_DIR
    output_filename: str = "qc_fc_matrix.png"

    @property
    def output_path(self) -> str:
        return os.path.join(self.output_dir, self.output_filename)


def plot_qc_fc_matrix(config: QCFCMatrixPlotConfig) -> None:
    r = np.load(os.path.join(config.qc_fc_dir, "qc_fc_r.npy"))
    sig = np.load(os.path.join(config.qc_fc_dir, "qc_fc_fdr_significant.npy"))

    n_rois = r.shape[0]
    iu = np.triu_indices(n_rois, k=1)
    n_significant = int(sig[iu].sum())
    n_edges = len(iu[0])
    median_abs_qcfc = np.median(np.abs(r[iu]))

    print(f"FDR-significant edges: {n_significant} / {n_edges}")
    print(f"Median |QC-FC|: {median_abs_qcfc:.4f}")

    os.makedirs(config.output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(r, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(
        f"QC-FC matrix (2mo, first run)\n"
        f"{n_significant}/{n_edges} FDR-significant, median |r| = {median_abs_qcfc:.3f}"
    )
    ax.set_xlabel("ROI")
    ax.set_ylabel("ROI")
    fig.colorbar(im, ax=ax, label="QC-FC (r)")
    fig.tight_layout()
    fig.savefig(config.output_path, dpi=150)
    plt.close(fig)
    print(f"saved {config.output_path}")


def parse_args() -> QCFCMatrixPlotConfig:
    parser = argparse.ArgumentParser(description="Plot the QC-FC matrix heatmap")
    parser.add_argument(
        "--qc_fc_dir", default=(
            "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
            "faizan_motion_correction_dataset/roi_timeseries_hfiltered_videos/qc_fc_2mo_firstrun"
        ),
    )
    parser.add_argument("--output_dir", default=IMAGES_DIR)
    parser.add_argument("--output_filename", default="qc_fc_matrix.png")
    args = parser.parse_args()
    return QCFCMatrixPlotConfig(**vars(args))


if __name__ == "__main__":
    plot_qc_fc_matrix(parse_args())
