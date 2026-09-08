import argparse
import csv
import os
import sys

import nibabel as nib
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from dataset import TARGET_SPATIAL, _psc_normalise, _spatial_resample, psc_denormalise, upsample_to_original
from models.model import DisentangledCycleGAN
from models.st_model import SpatioTemporalCycleGAN

DEST_ROOT = "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/isc_segmenting/isc_comparison_data_cyclegans"
DEFAULT_CHECKPOINT = os.path.join(REPO_ROOT, "runs", "fix_v3", "best_model.pt")


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    # Build the architecture from the checkpoint's own training args (mirrors train.py's
    # construction) rather than assuming defaults -- checkpoints differ in in_timepoints
    # (chunk size), residual mode, and architecture family (SpatioTemporalCycleGAN vs
    # DisentangledCycleGAN), and getting any of these wrong either crashes on
    # load_state_dict (shape mismatch) or, worse for `residual`, silently loads fine but
    # skips the x_a + out addition in correct(), returning a residual instead of the
    # corrected volume.
    if a.get("use_st_model", False):
        model = SpatioTemporalCycleGAN(
            in_timepoints    = a.get("in_timepoints", 5),
            spatial_dims     = (80, 96, 72),
            content_base_ch  = a.get("content_base_ch", 64),
            content_n_res    = a.get("content_n_res", 5),
            artefact_base_ch = a.get("artefact_base_ch", 64),
            global_code_dim  = a.get("global_code_dim", 64),
            spatial_code_ch  = a.get("spatial_code_ch", 32),
            disc_base_ch     = a.get("disc_base_ch", 64),
            num_disc_scales  = a.get("num_disc_scales", 2),
            residual         = a.get("residual", False),
        )
    else:
        model = DisentangledCycleGAN(
            in_timepoints       = a.get("in_timepoints", 20),
            spatial_dims        = (80, 96, 72),
            content_ch          = a.get("content_ch", 384),
            content_base_ch     = a.get("content_base_ch", 64),
            content_n_res       = a.get("content_n_res", 5),
            artefact_base_ch    = a.get("artefact_base_ch", 64),
            global_code_dim     = a.get("global_code_dim", 64),
            spatial_code_ch     = a.get("spatial_code_ch", 32),
            disc_base_ch        = a.get("disc_base_ch", 64),
            num_disc_scales     = a.get("num_disc_scales", 2),
            disc_temporal_diffs = not a.get("no_disc_temporal_diffs", False),
            residual             = a.get("residual", False),
        )
    result = model.load_state_dict(ckpt["model"], strict=False)

    for key in list(result.missing_keys) + list(result.unexpected_keys):
        if key.split(".")[0] in ("E_c", "G_B"):
            raise RuntimeError(f"Inference path mismatch in checkpoint: {key}")

    model.to(device)
    model.eval()
    return model


def load_chunk_manifest(order, exclude_suffix="A"):
    manifest_path = os.path.join(DEST_ROOT, order, f"order_{order}_manifest.csv")
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if not r["subject"].endswith(exclude_suffix)]


def build_windows(n_frames, window_size):
    windows = []
    start = 0
    while start + window_size <= n_frames:
        windows.append((start, start + window_size))
        start += window_size
    if not windows or windows[-1][1] != n_frames:
        windows.append((n_frames - window_size, n_frames))
    return windows


def stitch_windows(corrected_windows, windows, n_frames):
    out = torch.empty((n_frames,) + corrected_windows[0].shape[1:], dtype=corrected_windows[0].dtype)
    covered = 0
    for (start, end), corrected in zip(windows, corrected_windows):
        write_start = max(start, covered)
        offset = write_start - start
        out[write_start:end] = corrected[offset:]
        covered = max(covered, end)
    return out

def correct_chunk(chunk_path, model, device, window_size):
    img = nib.load(chunk_path)
    affine = img.affine
    data = np.asarray(img.dataobj, dtype=np.float32).transpose(3, 0, 1, 2)  # (T, X, Y, Z)
    n_frames = data.shape[0]
    orig_shape = data.shape[1:]

    resampled = _spatial_resample(torch.from_numpy(data), TARGET_SPATIAL)  # (T, 80, 96, 72)

    windows = build_windows(n_frames, window_size)
    corrected_windows = []
    for start, end in windows:
        psc, mean_vol = _psc_normalise(resampled[start:end])
        psc = psc.unsqueeze(0).to(device)
        mean_vol = mean_vol.unsqueeze(0).to(device)

        with torch.no_grad():
            corrected_psc = model.correct(psc)
        corrected_bold = psc_denormalise(corrected_psc, mean_vol)
        corrected_windows.append(corrected_bold.squeeze(0).cpu())

    stitched = stitch_windows(corrected_windows, windows, n_frames)  # (T, 80, 96, 72)
    upsampled = upsample_to_original(stitched, orig_shape)  # (T, X, Y, Z)

    out_data = upsampled.numpy().transpose(1, 2, 3, 0).astype(np.float32)  # (X, Y, Z, T)
    return nib.Nifti1Image(out_data, affine)


def build_output_path(row, order, out_root):
    run_dir = os.path.join(out_root, order, f"sub-{row['subject']}", f"ses-{row['session']}", f"run-{int(row['run']):03d}")
    os.makedirs(run_dir, exist_ok=True)
    return os.path.join(run_dir, os.path.basename(row["chunk_path"]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", default="A")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--window_size", type=int, default=20)
    ap.add_argument("--out_dir_name", default="motion_corrected",
                     help="Subdirectory under DEST_ROOT to write corrected chunks to, "
                          "e.g. 'motion_correction_with_residual' for a different checkpoint/chunk-size run.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    model = load_model(args.checkpoint, device)
    rows = load_chunk_manifest(args.order)
    print(f"Chunks to correct for order {args.order}: {len(rows)}", flush=True)

    out_root = os.path.join(DEST_ROOT, args.out_dir_name)

    for i, row in enumerate(rows, 1):
        out_img = correct_chunk(row["chunk_path"], model, device, args.window_size)
        out_path = build_output_path(row, args.order, out_root)
        nib.save(out_img, out_path)
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} done ({out_path})", flush=True)

    print(f"\nWrote {len(rows)} corrected chunks under {out_root}/{args.order}", flush=True)


if __name__ == "__main__":
    main()
