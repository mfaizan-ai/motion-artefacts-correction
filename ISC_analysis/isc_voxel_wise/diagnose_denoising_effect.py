"""
diagnose_denoising_effect.py
===============================
Tests whether denoising's voxelwise ISC drop (see plot_voxel_wise_isc.py) is
explained by the model introducing subject-idiosyncratic reconstruction noise,
by checking two things per voxel, using the same order-A segments already
used for the ISC computation:

1. Fidelity: within-subject correlation between a subject's own raw and
   denoised timeseries (same segment, same timepoints). Low fidelity means
   the model changed that voxel's temporal pattern substantially, not just
   its amplitude/noise floor.
2. Temporal SNR (mean/std over time) for raw and denoised separately -- does
   denoising actually reduce noise, or reduce signal along with it?

Both are then correlated against the ISC delta map (denoised - raw) already
saved by isc_voxel_wise_for_order.py: if low-fidelity voxels are where ISC
dropped, that supports the "idiosyncratic reconstruction noise" hypothesis.
"""
import os
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isc_voxel_wise_for_order as isc

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VOXEL_WISE_ISC_ROOT = os.path.join(REPO_ROOT, "motion_denoising_pipeline_level_evaluation", "voxel_wise_isc")
ORDER = "A"


def segment_key(s):
    return (s["subject"], s["session"], s["run"], s["segment_num"])


def tsnr(x):
    return x.mean(axis=1) / (x.std(axis=1) + 1e-9)


def main():
    print("collecting matched raw/denoised order-A segments...", flush=True)
    raw_segments = isc.collect_segments(ORDER, isc.CHUNK_METADATA_CSV)
    den_segments = isc.collect_segments(ORDER, isc.CHUNK_METADATA_CSV, isc.DENOISED_ROOT, denoised_suffix=True)

    raw_by_key = {segment_key(s): s for s in raw_segments}
    den_by_key = {segment_key(s): s for s in den_segments}
    common_keys = sorted(set(raw_by_key) & set(den_by_key))
    print(f"{len(common_keys)} matched segments", flush=True)

    mask = nib.load(os.path.join(VOXEL_WISE_ISC_ROOT, "order_a", "order_A_mask.nii.gz")).get_fdata().astype(bool)
    t_min = min(
        min(raw_by_key[k]["data"].shape[-1] for k in common_keys),
        min(den_by_key[k]["data"].shape[-1] for k in common_keys),
    )
    print(f"segment length used: {t_min}", flush=True)

    fidelity_maps, tsnr_raw_maps, tsnr_den_maps = [], [], []
    for k in common_keys:
        r = raw_by_key[k]["data"][..., :t_min][mask]
        d = den_by_key[k]["data"][..., :t_min][mask]
        fidelity_maps.append(isc.voxelwise_corr(r, d))
        tsnr_raw_maps.append(tsnr(r))
        tsnr_den_maps.append(tsnr(d))

    fidelity = np.mean(fidelity_maps, axis=0)
    tsnr_raw = np.mean(tsnr_raw_maps, axis=0)
    tsnr_den = np.mean(tsnr_den_maps, axis=0)
    tsnr_delta = tsnr_den - tsnr_raw

    isc_raw = nib.load(os.path.join(VOXEL_WISE_ISC_ROOT, "order_a", "order_A_group_isc_map.nii.gz")).get_fdata()[mask]
    isc_den = nib.load(os.path.join(VOXEL_WISE_ISC_ROOT, "order_a_denoised", "order_A_group_isc_map.nii.gz")).get_fdata()[mask]
    isc_delta = isc_den - isc_raw

    print(flush=True)
    print(f"fidelity (raw vs. denoised, same subject): mean={fidelity.mean():.4f} median={np.median(fidelity):.4f} "
          f"min={fidelity.min():.4f} max={fidelity.max():.4f}", flush=True)
    print(f"tSNR raw: mean={tsnr_raw.mean():.3f}  tSNR denoised: mean={tsnr_den.mean():.3f}  "
          f"delta mean={tsnr_delta.mean():.3f}  pct voxels tSNR improved={100 * (tsnr_delta > 0).mean():.1f}%", flush=True)
    print(flush=True)
    print(f"corr(fidelity, ISC delta) across voxels: {np.corrcoef(fidelity, isc_delta)[0, 1]:.4f}", flush=True)
    print(f"corr(tSNR delta, ISC delta) across voxels: {np.corrcoef(tsnr_delta, isc_delta)[0, 1]:.4f}", flush=True)

    print(flush=True)
    print("mean ISC delta by fidelity quartile (low fidelity = model changed the signal a lot):", flush=True)
    quartiles = np.quantile(fidelity, [0.25, 0.5, 0.75])
    bins = np.digitize(fidelity, quartiles)
    for q in range(4):
        sel = bins == q
        print(f"  Q{q+1} (fidelity {fidelity[sel].min():.3f}-{fidelity[sel].max():.3f}, n={sel.sum()}): "
              f"mean ISC delta = {isc_delta[sel].mean():.4f}", flush=True)

    out_dir = os.path.join(VOXEL_WISE_ISC_ROOT, "order_a_denoising_diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    affine = raw_segments[0]["affine"]
    for name, vec in [("fidelity", fidelity), ("tsnr_raw", tsnr_raw), ("tsnr_denoised", tsnr_den), ("tsnr_delta", tsnr_delta)]:
        full = np.full(mask.shape, np.nan, dtype=np.float32)
        full[mask] = vec
        nib.save(nib.Nifti1Image(full, affine), os.path.join(out_dir, f"order_A_{name}.nii.gz"))
    print(f"\nsaved maps -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
