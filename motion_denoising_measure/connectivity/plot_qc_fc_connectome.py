"""
plot_qc_fc_connectome.py
=========================
Glass-brain connectome plot of the strongest FDR-significant QC-FC edges,
on the real NIHPD (2mo) infant T1 anatomy. Node coordinates come directly
from the 2mo Schaefer-400 atlas file (same space/label order as the QC-FC
ROI axes), not a generic adult-MNI atlas -- verified against roi_labels_2mo.npy.
"""
import argparse
import os
from dataclasses import dataclass

import nibabel as nib
import numpy as np
from nilearn import plotting


@dataclass
class ConnectomePlotConfig:
    qc_fc_dir: str
    atlas_path: str
    t1_path: str
    edge_percentile: str = "98%"
    output_filename: str = "connectome.png"

    @property
    def output_path(self) -> str:
        return os.path.join(self.qc_fc_dir, self.output_filename)


def build_node_coords(config: ConnectomePlotConfig) -> np.ndarray:
    atlas_img = nib.load(config.atlas_path)
    coords, _ = plotting.find_parcellation_cut_coords(atlas_img, return_label_names=True)
    return coords


def build_connectome_plot(config: ConnectomePlotConfig) -> None:
    coords = build_node_coords(config)

    r = np.load(os.path.join(config.qc_fc_dir, "qc_fc_r.npy"))
    sig = np.load(os.path.join(config.qc_fc_dir, "qc_fc_fdr_significant.npy"))
    adjacency = np.where(sig, r, 0)

    degree = sig.sum(axis=1)
    node_size = 2 + 8 * (degree / degree.max())

    top_pct = 100 - int(config.edge_percentile.rstrip("%"))
    t1_img = nib.load(config.t1_path)
    display = plotting.plot_glass_brain(
        t1_img, display_mode="ortho", black_bg=False, colorbar=False,
        title=f"QC-FC (2mo, first run) -- top {top_pct}% strongest FDR-significant edges",
        alpha=0.3, cmap="gray_r",
    )
    display.add_graph(
        adjacency, coords,
        node_color="dimgray", node_size=node_size,
        edge_cmap="RdBu_r", edge_threshold=config.edge_percentile,
        edge_kwargs={"linewidth": 1.0, "alpha": 0.9},
        colorbar=True,
    )
    display.savefig(config.output_path, dpi=150)
    display.close()
    print(f"saved {config.output_path}")


def parse_args() -> ConnectomePlotConfig:
    parser = argparse.ArgumentParser(description="Plot QC-FC connectome on real infant anatomy")
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
    parser.add_argument(
        "--t1_path", default=(
            "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
            "templates/mask/nihpd_asym_02-05_t1w_masked_2mm.nii.gz"
        ),
    )
    parser.add_argument("--edge_percentile", default="98%")
    parser.add_argument("--output_filename", default="connectome.png")
    args = parser.parse_args()
    return ConnectomePlotConfig(**vars(args))


if __name__ == "__main__":
    build_connectome_plot(parse_args())
