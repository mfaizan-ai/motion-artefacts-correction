import argparse
import csv
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import numpy as np

DEST_ROOT = "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/isc_segmenting/isc_comparison_data_cyclegans"
ATLAS_PATH = "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/templates/rois/Schaefer2018_400Parcels_7Networks_order_space-nihpd-02-05_2mm.nii.gz"
LUT_PATH = "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/templates/rois/Schaefer2018_400Parcels_7Networks_order.lut"

# ATLAS_PATH is on the native (97,116,79) grid. Chunks built from the cropped/hfiltered
# training data are (60,72,56) instead -- --cropped_2mo_atlas crops the atlas to match, using
# the exact 2mo window from build_dataset.py (verified in experimental_notebooks/
# isc_train_data_raw_denoised_atlas_overlay.ipynb: recovers all 400 ROIs, correct anatomical
# overlay on both raw and denoised). ROI IDs are unchanged by cropping (plain array slice), so
# the LUT-based name/network lookup below applies to the cropped atlas without any changes.
AGE_CROP_2MO = {"x": (17, 77), "y": (26, 98), "z": (11, 67)}


def load_roi_network_map(lut_path):
    roi_names = {}
    roi_networks = {}
    with open(lut_path) as f:
        for line in f:
            parts = line.split()
            roi_id = int(parts[0])
            label = parts[-1]
            roi_names[roi_id] = label
            roi_networks[roi_id] = label.split("_")[2]
    n_roi = len(roi_names)
    network_names = sorted(set(roi_networks.values()))
    network_per_roi = np.array([roi_networks[i] for i in range(1, n_roi + 1)])
    return roi_names, network_per_roi, network_names


def load_atlas(atlas_path, crop=None):
    atlas_img = nib.load(atlas_path)
    atlas_data = np.asarray(atlas_img.dataobj).astype(np.int32)
    if crop is not None:
        atlas_data = atlas_data[crop["x"][0]:crop["x"][1], crop["y"][0]:crop["y"][1], crop["z"][0]:crop["z"][1]]
    return atlas_data


def load_manifest(order, source, exclude_suffix="A"):
    manifest_path = os.path.join(DEST_ROOT, order, f"order_{order}_manifest.csv")
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if not r["subject"].endswith(exclude_suffix)]

    if source != "raw":
        # source names any DEST_ROOT/<source>/<order>/... tree produced by
        # motion_correct.py's --out_dir_name (e.g. "motion_corrected",
        # "motion_correction_with_residual").
        raw_prefix = os.path.join(DEST_ROOT, order)
        corrected_prefix = os.path.join(DEST_ROOT, source, order)
        for r in rows:
            r["chunk_path"] = r["chunk_path"].replace(raw_prefix, corrected_prefix, 1)

    return rows


def extract_roi_time_courses(bold_path, atlas_data, n_roi):
    bold_data = nib.load(bold_path).get_fdata(dtype=np.float32)
    n_t = bold_data.shape[-1]

    flat_labels = atlas_data.reshape(-1)
    flat_bold = bold_data.reshape(-1, n_t)

    in_roi = flat_labels > 0
    labels = flat_labels[in_roi]
    voxels = flat_bold[in_roi]

    sums = np.zeros((n_roi + 1, n_t), dtype=np.float64)
    np.add.at(sums, labels, voxels)
    counts = np.bincount(labels, minlength=n_roi + 1)

    roi_tc = sums[1:] / counts[1:, None]
    return roi_tc.astype(np.float32)


def aggregate_network_time_courses(roi_tc, network_per_roi, network_names):
    network_tc = np.stack([
        roi_tc[network_per_roi == net].mean(axis=0) for net in network_names
    ])
    return network_tc.astype(np.float32)


def build_output_paths(row, order, tc_root):
    run_dir = os.path.join(tc_root, f"sub-{row['subject']}", f"ses-{row['session']}", f"run-{int(row['run']):03d}")
    os.makedirs(run_dir, exist_ok=True)
    stem = f"sub-{row['subject']}_ses-{row['session']}_run-{int(row['run']):03d}_order-{order}_seg-{row['segment_num']}"
    roi_path = os.path.join(run_dir, f"{stem}_roi_tc.npy")
    network_path = os.path.join(run_dir, f"{stem}_network_tc.npy")
    return roi_path, network_path


def process_row(row, order, tc_root, atlas_data, n_roi, network_per_roi, network_names):
    roi_tc = extract_roi_time_courses(row["chunk_path"], atlas_data, n_roi)
    network_tc = aggregate_network_time_courses(roi_tc, network_per_roi, network_names)

    roi_path, network_path = build_output_paths(row, order, tc_root)
    np.save(roi_path, roi_tc)
    np.save(network_path, network_tc)

    return {
        "subject": row["subject"], "session": row["session"], "run": row["run"],
        "order_label": order, "segment_num": row["segment_num"], "video": row["video"],
        "chunk_path": row["chunk_path"], "roi_tc_path": roi_path, "network_tc_path": network_path,
    }


def write_reference_files(tc_root, roi_names, network_per_roi, network_names):
    with open(os.path.join(tc_root, "roi_labels.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["roi_index", "roi_name", "network"])
        for i in range(1, len(roi_names) + 1):
            writer.writerow([i, roi_names[i], network_per_roi[i - 1]])

    with open(os.path.join(tc_root, "network_names.json"), "w") as f:
        json.dump(network_names, f, indent=2)


def write_manifest(tc_root, order, records):
    manifest_path = os.path.join(tc_root, f"order_{order}_timecourses_manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    return manifest_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", default="A")
    ap.add_argument("--source", default="raw",
                     help="'raw', or a DEST_ROOT subdirectory name produced by motion_correct.py's "
                          "--out_dir_name (e.g. 'motion_corrected', 'motion_correction_with_residual').")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--cropped_2mo_atlas", action="store_true",
        help="Crop the atlas to the 2mo (60,72,56) window instead of using it on the native "
             "(97,116,79) grid -- required for chunks built from cropped/hfiltered training data.",
    )
    args = ap.parse_args()

    roi_names, network_per_roi, network_names = load_roi_network_map(LUT_PATH)
    n_roi = len(roi_names)
    atlas_data = load_atlas(ATLAS_PATH, crop=AGE_CROP_2MO if args.cropped_2mo_atlas else None)

    rows = load_manifest(args.order, args.source)
    print(f"Rows for order {args.order}, source={args.source} (excluding A-suffix subjects): {len(rows)}", flush=True)

    root = DEST_ROOT if args.source == "raw" else os.path.join(DEST_ROOT, args.source)
    tc_root = os.path.join(root, args.order, "time_courses")
    os.makedirs(tc_root, exist_ok=True)
    write_reference_files(tc_root, roi_names, network_per_roi, network_names)

    records = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_row, row, args.order, tc_root, atlas_data, n_roi, network_per_roi, network_names): row
            for row in rows
        }
        for fut in as_completed(futures):
            records.append(fut.result())
            done += 1
            if done % 10 == 0 or done == len(rows):
                print(f"  {done}/{len(rows)} chunks processed", flush=True)

    manifest_path = write_manifest(tc_root, args.order, records)
    print(f"\nWrote {len(records)} time course pairs under {tc_root}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
