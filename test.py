"""
test.py
=======
Run model.correct() on the held-out test-split chunks and compute the same
fMRI quality metrics used to validate the model during training (train.py's
compute_fmri_metrics: DVARS, tSNR, global signal std, spatial smoothness),
grouped by grade. Reuses grade_dataset.py's exact normalization/padding and
train.py's exact metric functions -- same pipeline as training validation.
"""
import argparse
import os
from dataclasses import dataclass

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from grade_dataset import _load_run_stats, _run_key, _load_chunk_rows, _pad_h, denormalize_chunk
from models.st_model import SpatioTemporalCycleGAN
from train import compute_fmri_metrics


@dataclass
class TestConfig:
    checkpoint_path: str
    chunk_metadata_csv: str
    run_stats_csv: str
    output_dir: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(config: TestConfig) -> SpatioTemporalCycleGAN:
    ckpt = torch.load(config.checkpoint_path, map_location=config.device, weights_only=False)
    train_args = argparse.Namespace(**ckpt["args"])

    from atlas_fc import PADDED_SPATIAL
    model = SpatioTemporalCycleGAN(
        in_timepoints=train_args.in_timepoints,
        spatial_dims=PADDED_SPATIAL,
        content_base_ch=train_args.content_base_ch,
        content_n_res=train_args.content_n_res,
        artefact_base_ch=train_args.artefact_base_ch,
        global_code_dim=train_args.global_code_dim,
        spatial_code_ch=train_args.spatial_code_ch,
        disc_base_ch=train_args.disc_base_ch,
        num_disc_scales=train_args.num_disc_scales,
        residual=train_args.residual,
    ).to(config.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (best val score: {ckpt['best_score']:.4f})")
    return model


def load_chunk(row, run_stats, device) -> torch.Tensor:
    """Same normalization + H-padding as grade_dataset.py's _load_chunk_tensor."""
    median, scale = run_stats[_run_key(row)]
    data = np.asarray(nib.load(row["chunk_path"]).dataobj, dtype=np.float32)  # (60,72,56,5)
    mask = data[..., 0] != 0

    normalized = np.zeros_like(data, dtype=np.float32)
    normalized[mask] = (data[mask] - median) / scale

    tensor = torch.from_numpy(normalized).permute(3, 0, 1, 2)  # (T, 60, 72, 56)
    tensor = _pad_h(tensor)                                     # (T, 64, 72, 56)
    return tensor.unsqueeze(0).to(device), median, scale


def select_test_rows(config: TestConfig) -> dict:
    rows_by_grade = _load_chunk_rows(config.chunk_metadata_csv, split="test", task="videos")
    return {
        grade: [r for r in rows if r["age_group"] == "2mo" and not r["subject_id"].endswith("A")]
        for grade, rows in rows_by_grade.items()
    }

def run_test(config: TestConfig) -> None:
    model = load_model(config)
    run_stats = _load_run_stats(config.run_stats_csv)
    rows_by_grade = select_test_rows(config)

    records = []
    for grade, rows in rows_by_grade.items():
        for i, row in enumerate(rows, 1):
            x_in, median, scale = load_chunk(row, run_stats, config.device)
            with torch.no_grad():
                x_corrected = model.correct(x_in)

            median_t = torch.tensor([median], device=config.device)
            scale_t = torch.tensor([scale], device=config.device)
            mask = x_in != 0
            bold_input = denormalize_chunk(x_in, median_t, scale_t, mask)
            bold_corrected = denormalize_chunk(x_corrected, median_t, scale_t, mask)

            metrics = compute_fmri_metrics(bold_input, bold_corrected)
            metrics.update(grade=grade, subject_id=row["subject_id"], chunk_path=row["chunk_path"])
            records.append(metrics)

            if i % 200 == 0:
                print(f"  {grade}: {i}/{len(rows)}")
        print(f"{grade}: {len(rows)} chunks done")

    df = pd.DataFrame(records)
    os.makedirs(config.output_dir, exist_ok=True)
    df.to_csv(os.path.join(config.output_dir, "test_metrics_per_chunk.csv"), index=False)

    summary = df.groupby("grade")[
        ["dvars_improvement", "tsnr_improvement", "gs_std_improvement", "smoothness_ratio"]
    ].agg(["mean", "std"])
    summary.to_csv(os.path.join(config.output_dir, "test_metrics_summary_by_grade.csv"))
    print(summary)
    print(f"saved -> {config.output_dir}")

def parse_args() -> TestConfig:
    parser = argparse.ArgumentParser(description="Test the model on held-out test-split chunks")
    parser.add_argument(
        "--checkpoint_path", default=(
            "/lustre/disk/home/users/mfaizan/motion_correction/prototyping/"
            "motion-artefacts-correction/runs/st_v4_ddp_disc_temporal_roi/best_model.pt"
        ),
    )
    parser.add_argument(
        "--chunk_metadata_csv", default=(
            "/lustre/disk/home/users/mfaizan/motion_correction/cycleGANS_on_2d_images/"
            "pytorch-CycleGAN-and-pix2pix/data_preprocessing/motion_grades_chunk_5_dataset_hfiltered/"
            "chunk_metadata.csv"
        ),
    )
    parser.add_argument(
        "--run_stats_csv", default=(
            "/lustre/disk/home/users/mfaizan/motion_correction/prototyping/"
            "motion-artefacts-correction/run_normalization_stats_hfiltered.csv"
        ),
    )
    parser.add_argument(
        "--output_dir", default=(
            "/lustre/disk/home/users/mfaizan/motion_correction/prototyping/"
            "motion-artefacts-correction/runs/st_v4_ddp_disc_temporal_roi/test"
        ),
    )
    args = parser.parse_args()
    return TestConfig(**vars(args))

if __name__ == "__main__":
    run_test(parse_args())
