#!/usr/bin/env python3
"""
build_cyclegan_dataset.py

Usage
-----
    python build_cyclegan_dataset.py \\
        --corrupted_csv          corrupted_chunks.csv \\
        --motion_free_video_csv  video_state_clean_chunk_dataset.csv \\
        --motion_free_rest_csv   resting_state_clean_chunk_dataset.csv \\
        --output_dir             cyclegan_dataset \\
        --seed                   42
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Dataset split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Domain names for directory structure
DOMAIN_CORRUPTED  = "A_corrupted"
DOMAIN_MOTFREE    = "B_motion_free"

# split names 
SPLITS = ["train", "val", "test"]

CHUNK_FILENAME_TEMPLATE = (
    "sub-{subject_id}_ses-{session_id}_task-{task}"
    "_run-{run_id}_chunk-{chunk_start}-{chunk_end}.nii.gz"
)


# Logger setup
def setup_logging() -> logging.Logger:
    log = logging.getLogger("cyclegan_dataset")
    log.setLevel(logging.INFO)
    if not log.handlers:
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%H:%M:%S",
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        log.addHandler(ch)
    return log


log = setup_logging()


# Command line argument parsing
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description     = "Build CycleGAN-ready fMRI chunk dataset.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--corrupted_csv", required=True, type=Path, metavar="CSV",
        help="CSV of motion-corrupted chunks.",
    )
    parser.add_argument(
        "--motion_free_video_csv", required=True, type=Path, metavar="CSV",
        help="CSV of motion-free video task chunks.",
    )
    parser.add_argument(
        "--motion_free_rest_csv", required=True, type=Path, metavar="CSV",
        help="CSV of motion-free resting-state chunks.",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("cyclegan_dataset"),
        metavar="DIR", help="Root output directory.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--skip_extraction", action="store_true",
        help="Build metadata only — skip NIfTI chunk extraction.",
    )

    args = parser.parse_args()

    for csv_path in [args.corrupted_csv,
                     args.motion_free_video_csv,
                     args.motion_free_rest_csv]:
        if not csv_path.exists():
            parser.error(f"CSV not found: {csv_path}")

    return args



# Data loading and validation
def load_and_validate(csv_path: Path,
                      label:    str) -> pd.DataFrame:
    """
    Load a chunk CSV and validate required columns are present.

    Parameters
    ----------
    csv_path : Path
    label    : str   human-readable name for error messages

    Returns
    -------
    pd.DataFrame
    """
    required = {"subject_id", "session_id", "run_id", "task",
                "preprocessed_bold_file", "chunk_start", "chunk_end"}

    df = pd.read_csv(csv_path)
    missing = required - set(df.columns)
    if missing:
        log.error(f"[{label}] Missing required columns: {missing}")
        sys.exit(1)

    log.info(f"[{label}] Loaded {len(df)} rows from {csv_path.name}")
    return df


def filter_existing_files(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Keep only rows where preprocessed_bold_file exists on disk.
    Also applies the preprocessed_exists flag if present.

    Parameters
    ----------
    df    : pd.DataFrame
    label : str

    Returns
    -------
    pd.DataFrame
    """
    n_before = len(df)

    # Apply preprocessed_exists flag if column present
    if "preprocessed_exists" in df.columns:
        df = df[df["preprocessed_exists"] == True].copy()

    # Hard check: file must actually exist
    exists_mask = df["preprocessed_bold_file"].apply(
        lambda p: Path(p).exists()
    )
    df = df[exists_mask].copy()

    n_after  = len(df)
    n_dropped = n_before - n_after
    if n_dropped > 0:
        log.warning(
            f"[{label}] Dropped {n_dropped} rows with missing files "
            f"({n_after} remaining)"
        )
    return df.reset_index(drop=True)



# Subject-level splitting
def split_subjects(all_subjects: list[str],
                   seed:         int
                   ) -> tuple[list[str], list[str], list[str]]:
    """
    Split subjects into train / val / test using stratified ratios.

    Parameters
    ----------
    all_subjects : list of unique subject IDs
    seed         : int

    Returns
    -------
    train_subs, val_subs, test_subs : three lists of subject IDs
    """
    subjects = sorted(all_subjects)

    # First split: train vs (val + test)
    train_subs, valtest_subs = train_test_split(
        subjects,
        test_size  = VAL_RATIO + TEST_RATIO,
        random_state = seed,
    )

    # Second split: val vs test from the remaining pool
    relative_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_subs, test_subs = train_test_split(
        valtest_subs,
        test_size    = relative_test,
        random_state = seed,
    )

    return train_subs, val_subs, test_subs


def assign_splits(df_corrupted:  pd.DataFrame,
                  df_motfree:    pd.DataFrame,
                  seed:          int
                  ) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Assign subjects to splits and partition both dataframes accordingly.

    Parameters
    ----------
    df_corrupted : pd.DataFrame   motion-corrupted chunks
    df_motfree   : pd.DataFrame   motion-free chunks
    seed         : int

    Returns
    -------
    dict:  { split: { 'corrupted': df, 'motfree': df } }
    """
    # Union of subjects across both domains
    all_subs = sorted(
        set(df_corrupted["subject_id"].unique()) |
        set(df_motfree["subject_id"].unique())
    )
    log.info(f"Total unique subjects across both domains: {len(all_subs)}")

    train_subs, val_subs, test_subs = split_subjects(all_subs, seed)

    log.info(f"Split sizes — train={len(train_subs)}  "
             f"val={len(val_subs)}  test={len(test_subs)}")

    split_map = {
        "train": train_subs,
        "val"  : val_subs,
        "test" : test_subs,
    }

    result = {}
    for split, subs in split_map.items():
        result[split] = {
            "corrupted": df_corrupted[
                df_corrupted["subject_id"].isin(subs)
            ].copy().reset_index(drop=True),
            "motfree"  : df_motfree[
                df_motfree["subject_id"].isin(subs)
            ].copy().reset_index(drop=True),
            "subjects" : subs,
        }

    return result, split_map



# Balanced sampling
def balanced_sample(df_corrupted: pd.DataFrame,
                    n_target:     int,
                    seed:         int) -> pd.DataFrame:
    """
    Randomly sample n_target rows from df_corrupted.
    If df_corrupted has fewer rows than n_target, return all rows.

    Parameters
    ----------
    df_corrupted : pd.DataFrame
    n_target     : int
    seed         : int

    Returns
    -------
    pd.DataFrame
    """
    if len(df_corrupted) <= n_target:
        log.warning(
            f"  Corrupted pool ({len(df_corrupted)}) < target ({n_target}). "
            f"Using all corrupted chunks."
        )
        return df_corrupted.copy()

    return df_corrupted.sample(n=n_target, random_state=seed).reset_index(
        drop=True
    )


# Single chunk extraction
def extract_chunk(bold_path:    Path,
                  chunk_start:  int,
                  chunk_end:    int,
                  output_path:  Path) -> dict:
    """
    Extract a contiguous set of volumes from a 4D NIfTI and save as .nii.gz.

    Validates that chunk indices are within the time dimension.
    Preserves the original affine and header.

    Parameters
    ----------
    bold_path   : Path   source 4D NIfTI
    chunk_start : int    first volume index (inclusive)
    chunk_end   : int    last  volume index (inclusive)
    output_path : Path   destination .nii.gz

    Returns
    -------
    dict  with keys: status ('ok' / 'error'), message
    """
    try:
        img  = nib.load(str(bold_path))
        data = np.asarray(img.dataobj, dtype=np.float32)
    except Exception as exc:
        return {"status": "error", "message": f"Load failed: {exc}"}

    if data.ndim != 4:
        return {"status": "error",
                "message": f"Expected 4D, got shape {data.shape}"}

    n_vols = data.shape[3]

    # Validate chunk indices
    if chunk_start < 0 or chunk_end >= n_vols or chunk_start > chunk_end:
        return {
            "status" : "error",
            "message": (
                f"Invalid chunk [{chunk_start}:{chunk_end}] "
                f"for file with {n_vols} volumes"
            ),
        }

    chunk_data = data[..., chunk_start : chunk_end + 1]

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(
        chunk_data,
        affine = img.affine,
        header = img.header,
    )
    out_img.header.set_data_dtype(np.float32)
    out_img.header["dim"][4] = chunk_data.shape[3]
    nib.save(out_img, str(output_path))

    return {"status": "ok", "message": ""}


# Extract all chunks from the dataframe and save to disk
def extract_all_chunks(df:         pd.DataFrame,
                       domain_dir: Path,
                       domain:     str,
                       split:      str,
                       log_rows:   list) -> pd.DataFrame:
    """
    Extract all chunks in a dataframe and save to domain_dir.

    Adds a 'chunk_path' column to the dataframe with the saved file path.
    Skips already-extracted files (idempotent).
    Records all outcomes in log_rows.

    Parameters
    ----------
    df         : pd.DataFrame   chunk metadata
    domain_dir : Path           output directory (e.g. train/A_corrupted/)
    domain     : str            'A_corrupted' or 'B_motion_free'
    split      : str            'train', 'val', or 'test'
    log_rows   : list           appended in-place with log entries

    Returns
    -------
    pd.DataFrame  original df with 'chunk_path' column added
    """
    domain_dir.mkdir(parents=True, exist_ok=True)

    chunk_paths = []
    n_ok = n_skip = n_err = 0

    for _, row in df.iterrows():
        filename = CHUNK_FILENAME_TEMPLATE.format(
            subject_id  = row["subject_id"],
            session_id  = row["session_id"],
            task        = row["task"],
            run_id      = str(row["run_id"]).zfill(3),
            chunk_start = int(row["chunk_start"]),
            chunk_end   = int(row["chunk_end"]),
        )
        output_path = domain_dir / filename
        bold_path   = Path(row["preprocessed_bold_file"])

        # Skip if already extracted
        if output_path.exists():
            chunk_paths.append(str(output_path))
            n_skip += 1
            log_rows.append({
                "split"      : split,
                "domain"     : domain,
                "subject_id" : row["subject_id"],
                "session_id" : row["session_id"],
                "run_id"     : row["run_id"],
                "task"       : row["task"],
                "chunk_start": int(row["chunk_start"]),
                "chunk_end"  : int(row["chunk_end"]),
                "output_path": str(output_path),
                "status"     : "skipped_exists",
                "message"    : "",
                "timestamp"  : datetime.now().isoformat(timespec="seconds"),
            })
            continue

        result = extract_chunk(
            bold_path   = bold_path,
            chunk_start = int(row["chunk_start"]),
            chunk_end   = int(row["chunk_end"]),
            output_path = output_path,
        )

        if result["status"] == "ok":
            chunk_paths.append(str(output_path))
            n_ok += 1
        else:
            chunk_paths.append("")
            n_err += 1
            log.warning(
                f"  [{split}/{domain}] FAILED "
                f"sub-{row['subject_id']} "
                f"chunk [{row['chunk_start']}:{row['chunk_end']}] "
                f"— {result['message']}"
            )

        log_rows.append({
            "split"      : split,
            "domain"     : domain,
            "subject_id" : row["subject_id"],
            "session_id" : row["session_id"],
            "run_id"     : row["run_id"],
            "task"       : row["task"],
            "chunk_start": int(row["chunk_start"]),
            "chunk_end"  : int(row["chunk_end"]),
            "output_path": str(output_path),
            "status"     : result["status"],
            "message"    : result["message"],
            "timestamp"  : datetime.now().isoformat(timespec="seconds"),
        })

    log.info(
        f"  [{split}/{domain}]  "
        f"extracted={n_ok}  skipped={n_skip}  errors={n_err}"
    )

    df = df.copy()
    df["chunk_path"] = chunk_paths
    return df


# Extract metadata 
def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save a dataframe to CSV, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"  Saved metadata → {path.name}  ({len(df)} rows)")


def main() -> None:
    args = parse_args()
    rng  = args.seed

    log.info("=" * 65)
    log.info("  BUILD CYCLEGAN DATASET")
    log.info("=" * 65)
    log.info(f"  Corrupted CSV       : {args.corrupted_csv.name}")
    log.info(f"  Motion-free video   : {args.motion_free_video_csv.name}")
    log.info(f"  Motion-free rest    : {args.motion_free_rest_csv.name}")
    log.info(f"  Output dir          : {args.output_dir}")
    log.info(f"  Random seed         : {rng}")
    log.info(f"  Skip extraction     : {args.skip_extraction}")
    log.info("=" * 65)

    # Output directories 
    meta_dir = args.output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        for domain in [DOMAIN_CORRUPTED, DOMAIN_MOTFREE]:
            (args.output_dir / split / domain).mkdir(parents=True, exist_ok=True)

    # Load CSVs 
    df_corrupted = load_and_validate(args.corrupted_csv,         "corrupted")
    df_vid       = load_and_validate(args.motion_free_video_csv, "motfree_video")
    df_rest      = load_and_validate(args.motion_free_rest_csv,  "motfree_rest")

    # Combine motion-free 
    df_motfree = pd.concat([df_vid, df_rest], ignore_index=True)
    log.info(f"Combined motion-free : {len(df_motfree)} rows "
             f"(video={len(df_vid)}  rest={len(df_rest)})")

    # Filter missing files 
    df_corrupted = filter_existing_files(df_corrupted, "corrupted")
    df_motfree   = filter_existing_files(df_motfree,   "motion_free")

    log.info(f"After filtering — corrupted={len(df_corrupted)}  "
             f"motion_free={len(df_motfree)}")

    #  Subject-level split 
    log.info("\nSplitting subjects...")
    splits, split_map = assign_splits(df_corrupted, df_motfree, rng)

    # Save subject split manifest
    subject_rows = []
    for split, subs in split_map.items():
        for sub in subs:
            subject_rows.append({"subject_id": sub, "split": split})
    save_csv(pd.DataFrame(subject_rows), meta_dir / "subject_split.csv")

    #  Log collector 
    log_rows: list[dict] = []

    # Process each split 
    for split in SPLITS:
        log.info(f"\n{'─'*50}")
        log.info(f"  {split.upper()}")
        log.info(f"{'─'*50}")

        df_c  = splits[split]["corrupted"]
        df_mf = splits[split]["motfree"]
        n_mf  = len(df_mf)
        n_c   = len(df_c)

        log.info(f"  Motion-free chunks : {n_mf}")
        log.info(f"  Corrupted chunks   : {n_c}")

        # ── Train: keep all motion-free + all corrupted
        #          provide a balanced sample for epoch 0
        if split == "train":
            save_csv(df_mf, meta_dir / "train_motion_free.csv")
            save_csv(df_c,  meta_dir / "train_corrupted_all.csv")

            df_c_balanced = balanced_sample(df_c, n_mf, rng)
            save_csv(df_c_balanced,
                     meta_dir / "train_corrupted_balanced_epoch0.csv")
            log.info(
                f"  Balanced train corrupted (epoch 0): {len(df_c_balanced)}"
            )

            # Extract motion-free
            if not args.skip_extraction:
                log.info("  Extracting B_motion_free...")
                df_mf = extract_all_chunks(
                    df         = df_mf,
                    domain_dir = args.output_dir / split / DOMAIN_MOTFREE,
                    domain     = DOMAIN_MOTFREE,
                    split      = split,
                    log_rows   = log_rows,
                )
                # Extract corrupted (all — training DataLoader samples each epoch)
                log.info("  Extracting A_corrupted (all)...")
                df_c = extract_all_chunks(
                    df         = df_c,
                    domain_dir = args.output_dir / split / DOMAIN_CORRUPTED,
                    domain     = DOMAIN_CORRUPTED,
                    split      = split,
                    log_rows   = log_rows,
                )

            # Re-save with chunk_path column
            save_csv(df_mf, meta_dir / "train_motion_free.csv")
            save_csv(df_c,  meta_dir / "train_corrupted_all.csv")

        # ── Val / Test: balance corrupted to match motion-free count
        else:
            df_c_balanced = balanced_sample(df_c, n_mf, rng)
            log.info(
                f"  Balanced corrupted sampled: {len(df_c_balanced)} "
                f"(to match {n_mf} motion-free)"
            )

            save_csv(df_mf,        meta_dir / f"{split}_motion_free.csv")
            save_csv(df_c_balanced, meta_dir / f"{split}_corrupted_balanced.csv")

            if not args.skip_extraction:
                log.info(f"  Extracting B_motion_free...")
                df_mf = extract_all_chunks(
                    df         = df_mf,
                    domain_dir = args.output_dir / split / DOMAIN_MOTFREE,
                    domain     = DOMAIN_MOTFREE,
                    split      = split,
                    log_rows   = log_rows,
                )
                log.info(f"  Extracting A_corrupted (balanced)...")
                df_c_balanced = extract_all_chunks(
                    df         = df_c_balanced,
                    domain_dir = args.output_dir / split / DOMAIN_CORRUPTED,
                    domain     = DOMAIN_CORRUPTED,
                    split      = split,
                    log_rows   = log_rows,
                )

            # Re-save with chunk_path column
            save_csv(df_mf,         meta_dir / f"{split}_motion_free.csv")
            save_csv(df_c_balanced, meta_dir / f"{split}_corrupted_balanced.csv")

    # Save extraction log 
    if log_rows:
        save_csv(pd.DataFrame(log_rows), meta_dir / "extraction_log.csv")

    # summary print 
    log.info("\n" + "=" * 65)
    log.info("  DATASET SUMMARY")
    log.info("=" * 65)

    for split in SPLITS:
        df_c  = splits[split]["corrupted"]
        df_mf = splits[split]["motfree"]
        n_subs = len(splits[split]["subjects"])
        log.info(
            f"  {split:5s}  subjects={n_subs:3d}  "
            f"motion_free={len(df_mf):5d}  corrupted={len(df_c):5d}"
        )

    if log_rows:
        df_log  = pd.DataFrame(log_rows)
        n_ok    = int((df_log["status"] == "ok").sum())
        n_skip  = int((df_log["status"] == "skipped_exists").sum())
        n_err   = int((df_log["status"] == "error").sum())
        log.info(f"\n  Extraction — ok={n_ok}  skipped={n_skip}  errors={n_err}")

    log.info(f"\n  Output → {args.output_dir.resolve()}")
    log.info("=" * 65)


if __name__ == "__main__":
    main()