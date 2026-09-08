import argparse
import os
from dataclasses import dataclass

import nibabel as nib
import numpy as np
from nilearn import plotting
from scipy import stats


@dataclass
class DistanceQCFCConfig:
    qc_fc_dir: str
    atlas_path: str


def build_distance_matrix(config: DistanceQCFCConfig) -> np.ndarray:
    atlas_img = nib.load(config.atlas_path)
    coords, _ = plotting.find_parcellation_cut_coords(atlas_img, return_label_names=True)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def correlate_distance_qcfc(config: DistanceQCFCConfig) -> None:
    dist = build_distance_matrix(config)
    r = np.load(os.path.join(config.qc_fc_dir, "qc_fc_r.npy"))

    n_rois = r.shape[0]
    iu = np.triu_indices(n_rois, k=1)
    dist_edges = dist[iu]
    qcfc_edges = r[iu]

    corr, p = stats.pearsonr(dist_edges, qcfc_edges)
    print(f"distance-QC-FC correlation: r={corr:.4f}, p={p:.4e}")

    np.save(os.path.join(config.qc_fc_dir, "distance_matrix.npy"), dist)
    print(f"saved {os.path.join(config.qc_fc_dir, 'distance_matrix.npy')}")


def parse_args() -> DistanceQCFCConfig:
    parser = argparse.ArgumentParser(description="Correlate ROI-pair distance with QC-FC")
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
    args = parser.parse_args()
    return DistanceQCFCConfig(**vars(args))


if __name__ == "__main__":
    correlate_distance_qcfc(parse_args())
