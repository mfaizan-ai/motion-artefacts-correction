"""
create_roi_timeseries.py
=========================
Extract Schaefer-400 ROI timeseries for every video-state run (2mo + 9mo) listed
in a chunk-metadata CSV, apply a cosine high-pass filter (nilearn.signal.clean),
and save mirroring each run's source directory structure under output_root. 
"""
import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nilearn.signal import clean

from atlas_fc import CROPPED_SHAPE, PADDED_SPATIAL, load_age_atlases_cropped

PAD = PADDED_SPATIAL[0] - CROPPED_SHAPE[0]  # zero-pad axis 0: 60 -> 64


@dataclass
class ROITimeseriesConfig:
    chunk_metadata_csv: str
    source_root: str
    output_root: str
    high_pass_hz: float = 0.01
    # Set to a denoise_runs.py output_root (e.g. motion_corrected_st_v4) to
    # extract ROI timeseries from a corrected pipeline instead of the raw
    # volumes -- source_root/output_root stay pointed at the RAW tree either
    # way (see corrected_volume_path).
    corrected_root: Optional[str] = None


def load_run_volume(path: str) -> torch.Tensor:
    """(60, 72, 56, T) on disk -> zero-padded (T, 64, 72, 56) tensor."""
    data = nib.load(path).get_fdata()
    data = np.moveaxis(data, -1, 0)
    lo, hi = PAD // 2, PAD - PAD // 2
    data = np.pad(data, [(0, 0), (lo, hi), (0, 0), (0, 0)])
    return torch.from_numpy(data).float()


def corrected_volume_path(source_volume_path: str, source_root: str, corrected_root: str) -> str:
    """Raw source_volume_path -> its corrected counterpart, same convention
    as denoise_runs.py's output_path() (same relative dir, "_corrected"
    suffix)."""
    rel_dir = os.path.relpath(os.path.dirname(source_volume_path), source_root)
    fname = os.path.basename(source_volume_path).replace(".nii.gz", "_corrected.nii.gz")
    return os.path.join(corrected_root, rel_dir, fname)


def output_path(source_volume_path: str, config: ROITimeseriesConfig) -> str:
    # Always keyed off the RAW source_volume_path/source_root, even when
    # extracting from corrected volumes -- keeps output_root's directory
    # structure identical between raw and corrected runs.
    rel_dir = os.path.relpath(os.path.dirname(source_volume_path), config.source_root)
    return os.path.join(config.output_root, rel_dir, "roi_timeseries.npy")


def extract_and_filter(row, atlases: dict, config: ROITimeseriesConfig) -> np.ndarray:
    atlas = atlases[row.age_group]
    # --corrected_root set -> load the corrected volume instead of raw.
    volume_path = row.source_volume_path
    if config.corrected_root:
        volume_path = corrected_volume_path(volume_path, config.source_root, config.corrected_root)
    volume_seq = load_run_volume(volume_path)
    roi_ts = atlas.extract_roi_timeseries(volume_seq).numpy()  # (T, n_rois)
    return clean(
        roi_ts, detrend=False, standardize=None,
        filter="cosine", high_pass=config.high_pass_hz, t_r=row.tr_seconds,
    )


def build_run_table(config: ROITimeseriesConfig) -> pd.DataFrame:
    meta = pd.read_csv(config.chunk_metadata_csv)
    return (
        meta[meta["task"] == "videos"]
        [["subject_id", "session_id", "run_id", "age_group", "tr_seconds", "source_volume_path"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def build_all_roi_timeseries(config: ROITimeseriesConfig) -> None:
    runs = build_run_table(config)
    print(f"{len(runs)} video runs "
          f"({(runs.age_group == '2mo').sum()} 2mo, {(runs.age_group == '9mo').sum()} 9mo)")

    atlases = load_age_atlases_cropped()
    os.makedirs(config.output_root, exist_ok=True)
    for group, atlas in atlases.items():
        np.save(os.path.join(config.output_root, f"roi_labels_{group}.npy"), np.array(atlas.active_labels))

    failed = []
    for i, row in enumerate(runs.itertuples(index=False), 1):
        try:
            roi_ts = extract_and_filter(row, atlases, config)
            out_path = output_path(row.source_volume_path, config)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.save(out_path, roi_ts)
        except Exception as e:
            failed.append((row.source_volume_path, str(e)))
        if i % 25 == 0 or i == len(runs):
            print(f"  {i}/{len(runs)} done")

    print(f"finished: {len(runs) - len(failed)}/{len(runs)} succeeded")
    for path, err in failed:
        print(f"  FAILED {path}: {err}")


def parse_args() -> ROITimeseriesConfig:
    parser = argparse.ArgumentParser(description="Extract ROI timeseries for every video-state run")
    parser.add_argument(
        "--chunk_metadata_csv", default=(
            "/lustre/disk/home/users/mfaizan/motion_correction/cycleGANS_on_2d_images/"
            "pytorch-CycleGAN-and-pix2pix/data_preprocessing/motion_grades_chunk_5_dataset_hfiltered/"
            "chunk_metadata.csv"
        ),
        help="Chunk-level metadata CSV (only used for its per-run columns)",
    )
    parser.add_argument(
        # Keep this pointed at the RAW tree even for a corrected-pipeline
        # run -- it's only used to compute each run's relative subdir, not
        # which file gets loaded (see --corrected_root).
        "--source_root", default=(
            "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
            "faizan_motion_correction_dataset/brain_masked_cropped_hfiltered_normalized_to_common_space"
        ),
        help="Root dir of RAW whole-run source volumes (do not repoint for corrected runs)",
    )
    parser.add_argument(
        # Change this to a new dir for a corrected-pipeline run so raw and
        # corrected ROI timeseries never overwrite each other.
        "--output_root", default=(
            "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
            "faizan_motion_correction_dataset/roi_timeseries_hfiltered_videos"
        ),
        help="Where to save ROI timeseries, mirroring source_root's structure",
    )
    parser.add_argument("--high_pass_hz", type=float, default=0.01,
                         help="Cosine high-pass filter cutoff frequency")
    parser.add_argument(
        # e.g. .../motion_corrected_st_v4 -- a denoise_runs.py output_root.
        "--corrected_root", default=None,
        help="Set to extract from a corrected pipeline's volumes instead of raw",
    )
    args = parser.parse_args()
    return ROITimeseriesConfig(
        chunk_metadata_csv=args.chunk_metadata_csv,
        source_root=args.source_root,
        output_root=args.output_root,
        high_pass_hz=args.high_pass_hz,
        corrected_root=args.corrected_root,
    )


if __name__ == "__main__":
    build_all_roi_timeseries(parse_args())
