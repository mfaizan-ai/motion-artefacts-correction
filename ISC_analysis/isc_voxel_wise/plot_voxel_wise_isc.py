"""
plot_voxel_wise_isc.py
=========================
Ortho (sagittal/coronal/axial) slice plots of the group voxelwise ISC maps
produced by isc_voxel_wise_for_order.py -- one row per order, same slice
coordinates and same color scale across every row so orders are directly
comparable.

No permutation/FDR/cluster correction has been run on these maps yet (see
isc_voxel_wise_for_order.py's docstring) -- --threshold only hides voxels
below a raw ISC magnitude, it is not a statistical significance mask.
"""
import argparse
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import plotting

import isc_voxel_wise_for_order as isc

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VOXEL_WISE_ISC_ROOT = os.path.join(REPO_ROOT, "motion_denoising_pipeline_level_evaluation", "voxel_wise_isc")

DEFAULT_CMAP = "hot"
DEFAULT_VMIN = 0.0
DEFAULT_THRESHOLD = 0.0


def group_map_dir(order, source):
    suffix = "_denoised" if source == "denoised" else ""
    return os.path.join(VOXEL_WISE_ISC_ROOT, f"order_{order.lower()}{suffix}")


def group_map_path(order, source):
    return os.path.join(group_map_dir(order, source), f"order_{order}_group_isc_map.nii.gz")


def mask_path(order, source):
    return os.path.join(group_map_dir(order, source), f"order_{order}_mask.nii.gz")


def load_masked_map(order, source):
    img = nib.load(group_map_path(order, source))
    mask = nib.load(mask_path(order, source)).get_fdata().astype(bool)
    data = img.get_fdata()
    data[~mask] = np.nan
    return nib.Nifti1Image(data, img.affine)


def build_background_image():
    """Mean-over-time of one real BOLD run -- there's no separate anatomical
    template on this exact (60,72,56) cropped grid, and nilearn's own default
    (adult MNI152) would be badly misaligned with a 2mo infant brain."""
    runs = isc.select_runs(isc.CHUNK_METADATA_CSV)
    row = runs.iloc[0]
    vol = np.asarray(nib.load(row.source_volume_path).dataobj, dtype=np.float32)
    affine = nib.load(row.source_volume_path).affine
    return nib.Nifti1Image(vol.mean(axis=-1), affine)


def shared_vmax(orders, source, percentile=99.5):
    values = []
    for order in orders:
        data = nib.load(group_map_path(order, source)).get_fdata()
        values.append(data[np.isfinite(data)])
    return float(np.percentile(np.abs(np.concatenate(values)), percentile))


def print_stats(order, source, data):
    v = data[np.isfinite(data)]
    print(
        f"order {order} ({source}): mean={v.mean():.4f} median={np.median(v):.4f} "
        f"std={v.std():.4f} min={v.min():.4f} max={v.max():.4f} pct>0={100 * (v > 0).mean():.1f}%",
        flush=True,
    )


def plot_orders(orders, source, vmin, vmax, threshold, cmap, cut_coords, output_path):
    bg_img = build_background_image()
    if vmax is None:
        vmax = shared_vmax(orders, source)

    # Fix cut_coords once and reuse for every row -- nilearn auto-picks per map otherwise,
    # which breaks comparability. Always derived from the RAW map of the first order (even
    # when plotting denoised), so separate raw/denoised script runs land on identical
    # coordinates too, not just rows within one call.
    if cut_coords is None:
        ref_img = load_masked_map(orders[0], "raw")
        ref_data = np.nan_to_num(ref_img.get_fdata())
        cut_coords = plotting.find_xyz_cut_coords(nib.Nifti1Image(ref_data, ref_img.affine))

    fig, axes = plt.subplots(len(orders), 1, figsize=(15, 5 * len(orders)))
    axes = np.atleast_1d(axes)

    for ax, order in zip(axes, orders):
        stat_img = load_masked_map(order, source)
        print_stats(order, source, stat_img.get_fdata())
        display = plotting.plot_stat_map(
            stat_img, bg_img=bg_img, display_mode="ortho", cut_coords=cut_coords,
            cmap=cmap, vmin=vmin, vmax=vmax, threshold=threshold,
            colorbar=True, axes=ax, title=f"Order {order} ({source})",
        )
        display._colorbar_ax.set_ylabel("Mean ISC (Pearson r)", labelpad=18, rotation=270)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {output_path}")


def parse_args():
    ap = argparse.ArgumentParser(description="Ortho slice plots of group voxelwise ISC maps, one row per order")
    ap.add_argument("--orders", nargs="+", default=["A"], choices=list(isc.ALL_ORDERS))
    ap.add_argument("--source", choices=["raw", "denoised"], default="raw")
    ap.add_argument("--vmin", type=float, default=DEFAULT_VMIN)
    ap.add_argument("--vmax", type=float, default=None, help="Default: shared 99.5th percentile across --orders")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Hide |ISC| below this (not a stats mask)")
    ap.add_argument("--cmap", default=DEFAULT_CMAP)
    ap.add_argument("--cut_coords", type=float, nargs=3, default=None, help="x y z in mm; default: nilearn auto-picks")
    ap.add_argument("--output", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    output = args.output or os.path.join(
        VOXEL_WISE_ISC_ROOT, f"voxelwise_isc_{args.source}_{'-'.join(args.orders)}.png"
    )
    plot_orders(args.orders, args.source, args.vmin, args.vmax, args.threshold, args.cmap, args.cut_coords, output)


if __name__ == "__main__":
    main()
