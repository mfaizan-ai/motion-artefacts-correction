import argparse
import csv
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib

CSV_PATH = "/lustre/disk/home/users/mfaizan/motion_correction/prototyping/motion-artefacts-correction/ISC_analysis/mapping_csv_for_isc/segments_mapping_each_sub_usable_filtered.csv"
DEST_ROOT = "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/isc_segmenting/isc_comparison_data_cyclegans"

# The mapping CSV's bold_path always points at this (uncropped, unfiltered) root. To pull
# chunks from a different derivative instead (e.g. the model's actual training data), pass
# --source_root and this prefix gets swapped for it -- everything after the root (subject/
# run subdirs, filename) is identical across derivatives, so a plain prefix replace works.
CSV_BOLD_ROOT = "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/normalized_to_common_space"


def load_rows(order, source_root=None, denoised_suffix=False):
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["order_label"] == order]
    if source_root:
        for r in rows:
            r["bold_path"] = r["bold_path"].replace(CSV_BOLD_ROOT, source_root, 1)
    if denoised_suffix:
        # Matches denoise_runs.py's output_path() naming exactly: basename's ".nii.gz" -> "_corrected.nii.gz".
        for r in rows:
            r["bold_path"] = r["bold_path"].replace(".nii.gz", "_corrected.nii.gz")
    return rows


def process_source_file(bold_path, segs, order, dest_root):
    """Load one source nifti once, write out every requested chunk from it."""
    img = nib.load(bold_path)
    results = []
    for r in segs:
        subject, session, run, seg, start, end, video = r
        run_dir = os.path.join(dest_root, order, f"sub-{subject}", f"ses-{session}", f"run-{int(run):03d}")
        os.makedirs(run_dir, exist_ok=True)
        out_name = f"sub-{subject}_ses-{session}_run-{int(run):03d}_order-{order}_seg-{seg}_bold.nii.gz"
        out_path = os.path.join(run_dir, out_name)

        chunk = img.slicer[..., start:end]
        nib.save(chunk, out_path)

        results.append({
            "subject": subject, "session": session, "run": run, "order_label": order,
            "segment_num": seg, "video": video, "scan_start_idx": start, "scan_end_idx": end,
            "n_frames": end - start, "source_bold_path": bold_path, "chunk_path": out_path,
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", default="A")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--source_root", default=None,
        help="Replace the mapping CSV's normalized_to_common_space root with this one, e.g. to "
             "pull chunks from the model's actual (cropped, hfiltered) training data instead.",
    )
    ap.add_argument(
        "--out_dir_name", default=None,
        help="Write chunks under DEST_ROOT/<out_dir_name>/<order>/... instead of DEST_ROOT/<order>/... "
             "-- keeps a --source_root run from colliding with the default raw chunks.",
    )
    ap.add_argument(
        "--denoised_suffix", action="store_true",
        help="Append denoise_runs.py's '_corrected' suffix to each bold_path's filename, e.g. "
             "to pull chunks from motion_corrected_st_v4_all_video_runs via --source_root.",
    )
    args = ap.parse_args()

    rows = load_rows(args.order, args.source_root, args.denoised_suffix)
    print(f"Rows for order {args.order}: {len(rows)}", flush=True)

    # Group by source bold_path so a file shared by >1 segment (same subject/run,
    # two viewings of the same video) is decompressed and loaded only once.
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["bold_path"]].append((
            r["subject"], r["session"], r["run"], int(float(r["segment_num"])),
            int(float(r["scan_start_idx"])), int(float(r["scan_end_idx"])), r["first_video_name"],
        ))
    print(f"Unique source files: {len(by_source)}", flush=True)

    dest_root = os.path.join(DEST_ROOT, args.out_dir_name) if args.out_dir_name else DEST_ROOT
    order_dir = os.path.join(dest_root, args.order)
    os.makedirs(order_dir, exist_ok=True)

    manifest_rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_source_file, path, segs, args.order, dest_root): path
            for path, segs in by_source.items()
        }
        for fut in as_completed(futures):
            manifest_rows.extend(fut.result())
            done += 1
            if done % 10 == 0 or done == len(by_source):
                print(f"  {done}/{len(by_source)} source files processed", flush=True)

    manifest_path = os.path.join(order_dir, f"order_{args.order}_manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nWrote {len(manifest_rows)} chunk files under {order_dir}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
