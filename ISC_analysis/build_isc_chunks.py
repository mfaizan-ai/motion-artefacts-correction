import argparse
import csv
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib

CSV_PATH = "/lustre/disk/home/users/mfaizan/motion_correction/prototyping/motion-artefacts-correction/ISC_analysis/mapping_csv_for_isc/segments_mapping_each_sub_usable_filtered.csv"
DEST_ROOT = "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/isc_segmenting/isc_comparison_data_cyclegans"


def load_rows(order):
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        return [r for r in reader if r["order_label"] == order]


def process_source_file(bold_path, segs, order):
    """Load one source nifti once, write out every requested chunk from it."""
    img = nib.load(bold_path)
    results = []
    for r in segs:
        subject, session, run, seg, start, end, video = r
        run_dir = os.path.join(DEST_ROOT, order, f"sub-{subject}", f"ses-{session}", f"run-{int(run):03d}")
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
    args = ap.parse_args()

    rows = load_rows(args.order)
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

    order_dir = os.path.join(DEST_ROOT, args.order)
    os.makedirs(order_dir, exist_ok=True)

    manifest_rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_source_file, path, segs, args.order): path for path, segs in by_source.items()}
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
