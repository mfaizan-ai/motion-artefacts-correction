import argparse
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from atlas_fc import CROPPED_SHAPE, PADDED_SPATIAL
from grade_dataset import _load_run_stats
from test import load_model, TestConfig

PAD = PADDED_SPATIAL[0] - CROPPED_SHAPE[0]  # 4, zero-pad axis 0: 60 -> 64
CHUNK_T = 5


@dataclass
class DenoiseConfig:
    checkpoint_path: str
    chunk_metadata_csv: str
    run_stats_csv: str
    source_root: str
    output_root: str
    log_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def append_log(config: DenoiseConfig, row: dict) -> None:
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": get_git_commit(),
        "checkpoint_path": config.checkpoint_path,
        **row,
    }
    os.makedirs(os.path.dirname(config.log_path), exist_ok=True)
    header = not os.path.exists(config.log_path)
    pd.DataFrame([row]).to_csv(config.log_path, mode="a", header=header, index=False)


def select_runs(config: DenoiseConfig) -> pd.DataFrame:
    """Same selection as denoising_evaluation.py's select_first_runs (2mo,
    non-A, first session/run) -- kept in sync manually, small enough that
    duplicating avoids a cross-directory import into motion_denoising_measure/."""
    meta = pd.read_csv(config.chunk_metadata_csv, dtype={"session_id": str, "run_id": str})
    v = meta[(meta["task"] == "videos") & (meta["age_group"] == "2mo")]
    v = v[~v["subject_id"].str.endswith("A")]
    cols = ["subject_id", "session_id", "run_id", "source_volume_path"]
    runs = v[cols].drop_duplicates().sort_values(["subject_id", "session_id", "run_id"])
    return runs.groupby("subject_id", as_index=False).first()


def pad_spatial(chunk: torch.Tensor) -> torch.Tensor:
    lo, hi = PAD // 2, PAD - PAD // 2
    return F.pad(chunk, (0, 0, 0, 0, lo, hi))


def crop_spatial(chunk: torch.Tensor) -> torch.Tensor:
    lo = PAD // 2
    return chunk[:, :, lo:lo + CROPPED_SHAPE[0], :, :]


@torch.no_grad()
def correct_run(volume: np.ndarray, median: float, scale: float, model, device: str) -> np.ndarray:
    """volume: (60,72,56,T) raw BOLD -> (60,72,56,T) corrected BOLD, same T."""
    X, Y, Z, T = volume.shape
    mask = volume[..., 0] != 0

    normalized = np.zeros_like(volume, dtype=np.float32)
    normalized[mask] = (volume[mask] - median) / scale
    vol = torch.from_numpy(normalized).permute(3, 0, 1, 2)  # (T, X, Y, Z)

    corrected_chunks = []
    for start in range(0, T, CHUNK_T):
        end = min(start + CHUNK_T, T)
        chunk = vol[start:end]
        n_real = chunk.shape[0]
        if n_real < CHUNK_T:
            pad_frames = chunk[-1:].repeat(CHUNK_T - n_real, 1, 1, 1)
            chunk = torch.cat([chunk, pad_frames], dim=0)

        chunk = chunk.unsqueeze(0).to(device)  # (1, 5, X, Y, Z)
        chunk = pad_spatial(chunk)             # (1, 5, 64, Y, Z)
        corrected = model.correct(chunk)       # (1, 5, 64, Y, Z)
        corrected = crop_spatial(corrected)    # (1, 5, X, Y, Z)
        corrected_chunks.append(corrected[0, :n_real].cpu().numpy())

    corrected_norm = np.concatenate(corrected_chunks, axis=0)  # (T, X, Y, Z)
    corrected_bold = np.zeros_like(corrected_norm)
    mask_t = np.broadcast_to(mask, corrected_norm.shape)
    corrected_bold[mask_t] = corrected_norm[mask_t] * scale + median
    return corrected_bold.transpose(1, 2, 3, 0)  # (X, Y, Z, T)


def output_path(source_volume_path: str, config: DenoiseConfig) -> str:
    rel_dir = os.path.relpath(os.path.dirname(source_volume_path), config.source_root)
    fname = os.path.basename(source_volume_path).replace(".nii.gz", "_corrected.nii.gz")
    return os.path.join(config.output_root, rel_dir, fname)


def run_denoise(config: DenoiseConfig) -> None:
    model = load_model(TestConfig(
        checkpoint_path=config.checkpoint_path, chunk_metadata_csv=config.chunk_metadata_csv,
        run_stats_csv=config.run_stats_csv, output_dir=config.output_root, device=config.device,
    ))
    run_stats = _load_run_stats(config.run_stats_csv)
    runs = select_runs(config)
    print(f"{len(runs)} subjects")

    for i, row in enumerate(runs.itertuples(index=False), 1):
        key = (row.subject_id, str(row.session_id), str(row.run_id), "videos")
        if key not in run_stats:
            print(f"  [{i}/{len(runs)}] {row.subject_id}: SKIPPED, no entry in run_stats_csv")
            append_log(config, {"subject_id": row.subject_id, "status": "skipped_no_run_stats"})
            continue
        median, scale = run_stats[key]

        img = nib.load(row.source_volume_path)
        volume = np.asarray(img.dataobj).astype(np.float32)
        corrected = correct_run(volume, median, scale, model, config.device)

        out_path = output_path(row.source_volume_path, config)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        nib.save(nib.Nifti1Image(corrected, img.affine, img.header), out_path)

        print(f"  [{i}/{len(runs)}] {row.subject_id}: saved {out_path}")
        append_log(config, {
            "subject_id": row.subject_id, "session_id": row.session_id, "run_id": row.run_id,
            "source_volume_path": row.source_volume_path, "output_path": out_path,
            "status": "ok",
        })

    print(f"done -> {config.output_root}")
    print(f"log -> {config.log_path}")


def parse_args() -> DenoiseConfig:
    parser = argparse.ArgumentParser(description="Whole-run model correction on the raw-baseline population")
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
        "--source_root", default=(
            "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
            "faizan_motion_correction_dataset/brain_masked_cropped_hfiltered_normalized_to_common_space"
        ),
    )
    parser.add_argument(
        "--output_root", default=(
            "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
            "faizan_motion_correction_dataset/motion_corrected_st_v4"
        ),
    )
    parser.add_argument(
        "--log_path", default=(
            "/lustre/disk/home/users/mfaizan/motion_correction/prototyping/"
            "motion-artefacts-correction/runs/st_v4_ddp_disc_temporal_roi/denoise_runs_log.csv"
        ),
    )
    args = parser.parse_args()
    return DenoiseConfig(**vars(args))


if __name__ == "__main__":
    run_denoise(parse_args())
