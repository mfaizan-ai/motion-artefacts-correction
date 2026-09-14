"""
backfill_fd_dvars_r.py
=======================
One-off: add the fd_dvars_r column to manifest.csv files that were written
before denoising_evaluation.py computed it, without rerunning the whole
(expensive, modularity-dominated) evaluation.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd

from denoising_evaluation import (
    EvalConfig, corrected_volume_path, load_run_volume_and_mask, nipype_dvars_timeseries,
    compute_fd_dvars_correlation, select_first_runs,
)

# Same defaults as denoising_evaluation.py's parse_args() -- duplicated
# directly rather than calling parse_args() itself, since that reads
# sys.argv and would collide with this script's own --result_dir flag.
CHUNK_METADATA_CSV = (
    "/lustre/disk/home/users/mfaizan/motion_correction/cycleGANS_on_2d_images/"
    "pytorch-CycleGAN-and-pix2pix/data_preprocessing/motion_grades_chunk_5_dataset_hfiltered/"
    "chunk_metadata.csv"
)
SOURCE_ROOT = (
    "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
    "faizan_motion_correction_dataset/brain_masked_cropped_hfiltered_normalized_to_common_space"
)


def backfill(result_dir: str, corrected_root: str = None) -> None:
    manifest_path = os.path.join(result_dir, "manifest.csv")
    manifest = pd.read_csv(manifest_path)
    if "fd_dvars_r" in manifest.columns:
        print(f"{result_dir}: fd_dvars_r already present, skipping")
        return

    config = EvalConfig(
        chunk_metadata_csv=CHUNK_METADATA_CSV, source_root=SOURCE_ROOT,
        roi_timeseries_root="", atlas_path="", output_dir="",
    )
    runs = select_first_runs(config).set_index("subject_id")

    pearson_r, pearson_p, spearman_rho, spearman_p, n_transitions = [], [], [], [], []
    for i, subject_id in enumerate(manifest["subject_id"], 1):
        row = runs.loc[subject_id]
        volume_path = row.source_volume_path
        if corrected_root:
            volume_path = corrected_volume_path(volume_path, config.source_root, corrected_root)
        func, mask = load_run_volume_and_mask(volume_path)
        dvars_ts = nipype_dvars_timeseries(func, mask)
        fd_ts = pd.read_csv(row.fd_path)["FramewiseDisplacement"].to_numpy()
        result = compute_fd_dvars_correlation(fd_ts, dvars_ts)
        pearson_r.append(result["pearson_r"])
        pearson_p.append(result["pearson_p"])
        spearman_rho.append(result["spearman_rho"])
        spearman_p.append(result["spearman_p"])
        n_transitions.append(result["n_transitions"])
        if i % 32 == 0:
            print(f"  {i}/{len(manifest)}")

    manifest["fd_dvars_r"] = pearson_r
    manifest["fd_dvars_p"] = pearson_p
    manifest["fd_dvars_spearman_rho"] = spearman_rho
    manifest["fd_dvars_spearman_p"] = spearman_p
    manifest["fd_dvars_n_transitions"] = n_transitions
    manifest.to_csv(manifest_path, index=False)
    print(f"{result_dir}: fd_dvars_r added (median={pd.Series(pearson_r).median():.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--corrected_root", default=None)
    args = parser.parse_args()
    backfill(args.result_dir, args.corrected_root)
