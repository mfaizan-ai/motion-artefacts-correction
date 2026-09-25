"""
isc_voxel_wise_for_order.py
==============================
Voxelwise leave-one-out ISC for a single video-clip order, 2mo subjects, on
the raw (cropped, brain-masked, high-pass-filtered) training data.

Segment detection (order via attention-getter events, vid1a/vid1b onset
splitting) is adapted from isc_wf.py -- its confound-regression/GLM part is
not needed here, only the raw BOLD segments.
"""
import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd

CHUNK_METADATA_CSV = (
    "/lustre/disk/home/users/mfaizan/motion_correction/cycleGANS_on_2d_images/"
    "pytorch-CycleGAN-and-pix2pix/data_preprocessing/motion_grades_chunk_5_dataset_hfiltered/"
    "chunk_metadata.csv"
)
BIDS_ROOT = "/lustre/disk/home/shared/cusacklab/foundcog/bids"
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# chunk_metadata_csv's source_volume_path always points at this (raw, cropped, brain-masked,
# high-pass-filtered) root. --source_root swaps it for a different derivative, e.g. the
# st_v4-denoised volumes -- everything after the root is identical, so a prefix replace works.
RAW_ROOT = (
    "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/faizan_motion_correction_dataset/"
    "brain_masked_cropped_hfiltered_normalized_to_common_space"
)
DENOISED_ROOT = (
    "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/faizan_motion_correction_dataset/"
    "motion_corrected_st_v4_all_video_runs"
)

TR = 0.61
SEGMENT_LENGTH = 234
MIN_SEGMENT_LENGTH = 200  # drop truncated segments (scan ended early) rather than let one collapse t_min for everyone
ALL_ORDERS = {
    "A": ["minions_supermarket", "new_orleans", "bathsong", "dog", "moana", "forest"],
    "B": ["bathsong", "dog", "moana", "forest", "minions_supermarket", "new_orleans"],
    "C": ["new_orleans", "minions_supermarket", "dog", "bathsong", "forest", "moana"],
    "D": ["moana", "forest", "minions_supermarket", "new_orleans", "bathsong", "dog"],
    "E": ["forest", "moana", "new_orleans", "minions_supermarket", "dog", "bathsong"],
    "F": ["dog", "bathsong", "forest", "moana", "new_orleans", "minions_supermarket"],
}
FIRST_CLIP_TO_ORDER = {clips[0]: order for order, clips in ALL_ORDERS.items()}


def select_runs(chunk_metadata_csv, source_root=None, denoised_suffix=False):
    meta = pd.read_csv(chunk_metadata_csv, dtype={"session_id": str, "run_id": str})
    v = meta[(meta["task"] == "videos") & (meta["age_group"] == "2mo")]
    v = v[~v["subject_id"].str.endswith("A")]
    cols = ["subject_id", "session_id", "run_id", "source_volume_path"]
    runs = v[cols].drop_duplicates().sort_values(["subject_id", "session_id", "run_id"]).copy()

    if source_root:
        runs["source_volume_path"] = runs["source_volume_path"].str.replace(RAW_ROOT, source_root, regex=False)
    if denoised_suffix:
        runs["source_volume_path"] = runs["source_volume_path"].str.replace(".nii.gz", "_corrected.nii.gz", regex=False)
    return runs


def events_path_for(subject_id, session_id, run_id):
    fname = f"sub-{subject_id}_ses-{session_id}_task-videos_dir-AP_run-{run_id}_events.tsv"
    return os.path.join(BIDS_ROOT, f"sub-{subject_id}", f"ses-{session_id}", "func", fname)


def find_order_segments(events_path, order, tr=TR):
    """Returns [(segment_num, start_frame), ...] for every segment in this run
    whose clip sequence matches `order`."""
    events = pd.read_csv(events_path, sep="\t")
    events["trial_type"] = events["trial_type"].str.replace(".mp4", "", regex=False)
    if len(events) < 7:
        return []

    ag_idxs = events.index[events["trial_type"].str.contains("attention_getter")].tolist()
    matches = []
    for seg_num, ag_idx in enumerate(ag_idxs, start=1):
        if ag_idx + 1 >= len(events):
            continue
        first_clip = events["trial_type"].iloc[ag_idx + 1]
        if FIRST_CLIP_TO_ORDER.get(first_clip) != order:
            continue
        onset = events["onset"].iloc[ag_idx + 1]
        matches.append((seg_num, int(round(onset / tr))))
    return matches


def collect_segments(order, chunk_metadata_csv, source_root=None, denoised_suffix=False):
    runs = select_runs(chunk_metadata_csv, source_root, denoised_suffix)
    segments = []
    for row in runs.itertuples(index=False):
        ev_path = events_path_for(row.subject_id, row.session_id, row.run_id)
        if not os.path.exists(ev_path):
            continue
        matches = find_order_segments(ev_path, order)
        if not matches:
            continue

        data = np.asarray(nib.load(row.source_volume_path).dataobj, dtype=np.float32)
        affine = nib.load(row.source_volume_path).affine
        for seg_num, start in matches:
            end = min(start + SEGMENT_LENGTH, data.shape[-1])
            if end - start < MIN_SEGMENT_LENGTH:
                continue
            segments.append({
                "subject": row.subject_id, "session": row.session_id, "run": row.run_id,
                "segment_num": seg_num, "data": data[..., start:end], "affine": affine,
            })
    return segments


def zscore_time(x):
    """x: (n_voxels, T) -> per-voxel z-score across T."""
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (x - mean) / std


def voxelwise_corr(a, b):
    """a, b: (n_voxels, T) -> per-voxel Pearson r."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
    return num / den


def fisher_z(r):
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def inverse_fisher_z(z):
    return np.tanh(z)


def unmask(vec, mask, fill=np.nan):
    full = np.full(mask.shape, fill, dtype=np.float32)
    full[mask] = vec
    return full


def run_order(order, chunk_metadata_csv, output_dir, source_root=None, denoised_suffix=False):
    print(f"collecting order-{order} segments...", flush=True)
    segments = collect_segments(order, chunk_metadata_csv, source_root, denoised_suffix)
    if not segments:
        raise RuntimeError(f"no order-{order} segments found")

    t_min = min(s["data"].shape[-1] for s in segments)
    for s in segments:
        s["data"] = s["data"][..., :t_min]
    affine = segments[0]["affine"]

    mask = np.ones(segments[0]["data"].shape[:3], dtype=bool)
    for s in segments:
        mask &= s["data"][..., 0] != 0
    print(f"{len(segments)} segments, {len(set(s['subject'] for s in segments))} subjects, "
          f"segment length={t_min}, mask voxels={mask.sum()}", flush=True)

    for s in segments:
        s["z"] = zscore_time(s["data"][mask])

    by_subject = {}
    for s in segments:
        by_subject.setdefault(s["subject"], []).append(s)
    subjects = sorted(by_subject)
    n_subjects = len(subjects)
    print(f"{n_subjects} unique subjects", flush=True)

    representative = {subj: np.mean([s["z"] for s in entries], axis=0) for subj, entries in by_subject.items()}
    total = sum(representative.values())

    segment_maps, segment_rows = [], []
    subject_maps, subject_rows = [], []
    for subj in subjects:
        loo_ref = (total - representative[subj]) / (n_subjects - 1)

        seg_rs = []
        for s in by_subject[subj]:
            r = voxelwise_corr(s["z"], loo_ref)
            seg_rs.append(r)
            segment_maps.append(unmask(r, mask))
            segment_rows.append({"subject": subj, "session": s["session"], "run": s["run"], "segment_num": s["segment_num"]})

        r_subj = inverse_fisher_z(fisher_z(np.stack(seg_rs)).mean(axis=0))
        subject_maps.append(unmask(r_subj, mask))
        subject_rows.append({"subject": subj})

    subject_z_stack = fisher_z(np.stack([m[mask] for m in subject_maps]))
    group_z = subject_z_stack.mean(axis=0)
    group_r = inverse_fisher_z(group_z)

    save_outputs(
        output_dir, order, affine, mask,
        segment_maps, segment_rows, subject_maps, subject_rows,
        unmask(group_z, mask), unmask(group_r, mask),
    )


def save_outputs(output_dir, order, affine, mask, segment_maps, segment_rows,
                  subject_maps, subject_rows, group_z_map, group_r_map):
    os.makedirs(output_dir, exist_ok=True)

    nib.save(
        nib.Nifti1Image(np.stack(segment_maps, axis=-1), affine),
        os.path.join(output_dir, f"order_{order}_segment_isc_maps.nii.gz"),
    )
    pd.DataFrame(segment_rows).to_csv(os.path.join(output_dir, f"order_{order}_segment_manifest.csv"), index=False)

    nib.save(
        nib.Nifti1Image(np.stack(subject_maps, axis=-1), affine),
        os.path.join(output_dir, f"order_{order}_subject_isc_maps.nii.gz"),
    )
    pd.DataFrame(subject_rows).to_csv(os.path.join(output_dir, f"order_{order}_subject_manifest.csv"), index=False)

    nib.save(nib.Nifti1Image(group_z_map, affine), os.path.join(output_dir, f"order_{order}_group_isc_fisher_z_map.nii.gz"))
    nib.save(nib.Nifti1Image(group_r_map, affine), os.path.join(output_dir, f"order_{order}_group_isc_map.nii.gz"))
    nib.save(nib.Nifti1Image(mask.astype(np.float32), affine), os.path.join(output_dir, f"order_{order}_mask.nii.gz"))

    print(f"saved -> {output_dir}", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description="Voxelwise leave-one-out ISC for one video-clip order, 2mo subjects")
    ap.add_argument("--order", default="A", choices=list(ALL_ORDERS))
    ap.add_argument("--chunk_metadata_csv", default=CHUNK_METADATA_CSV)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument(
        "--source_root", default=None,
        help="Replace the raw (hfiltered/cropped/masked) root with this one, e.g. DENOISED_ROOT "
             "to run on the st_v4-denoised volumes instead.",
    )
    ap.add_argument(
        "--denoised_suffix", action="store_true",
        help="Append denoise_runs.py's '_corrected' filename suffix -- use together with "
             "--source_root DENOISED_ROOT.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(OUTPUT_ROOT, f"order_{args.order}")
    run_order(args.order, args.chunk_metadata_csv, output_dir, args.source_root, args.denoised_suffix)


if __name__ == "__main__":
    main()
