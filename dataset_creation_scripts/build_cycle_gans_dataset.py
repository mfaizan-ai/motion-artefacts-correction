#!/usr/bin/env python3
"""
Usage
-----
    python build_cyclegan_dataset.py \\
        --corrupted_csv          corrupted_chunks_with_preprocessed.csv \\
        --motion_free_video_csv  video_state_clean_chunk_dataset_with_preprocessed.csv \\
        --motion_free_rest_csv   resting_state_clean_chunk_dataset_with_preprocessed.csv \\
        --output_dir             cyclegan_dataset \\
        --seed                   42 \\
        --n_jobs                 8
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split


# Key parameters
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

DOMAIN_CORRUPTED = "A_corrupted"
DOMAIN_MOTFREE   = "B_motion_free"
ALL_SPLITS       = ["train", "val", "test"]

CHUNK_FILENAME_TEMPLATE = (
    "sub-{subject_id}_ses-{session_id}_task-{task}"
    "_run-{run_id}_chunk-{chunk_start}-{chunk_end}.nii.gz"
)

GZIP_LEVEL = 1



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



# Argument parsing 
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
        metavar="DIR",
        help="Root output directory.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=8,
        help="Parallel workers for chunk extraction (one per source file).",
    )
    parser.add_argument(
        "--skip_extraction", action="store_true",
        help="Build metadata CSVs only — skip NIfTI chunk extraction.",
    )
    parser.add_argument(
        "--splits", nargs="+", choices=ALL_SPLITS,
        default=["val", "test"],
        metavar="SPLIT",
        help=(
            "Which splits to process. Choices: train val test. "
            "Default: val test  (training data already generated). "
            "Example: --splits val test  or  --splits train val test"
        ),
    )

    args = parser.parse_args()

    # Deduplicate and preserve canonical order
    args.splits = [s for s in ALL_SPLITS if s in args.splits]

    for p in [args.corrupted_csv,
              args.motion_free_video_csv,
              args.motion_free_rest_csv]:
        if not p.exists():
            parser.error(f"CSV not found: {p}")

    return args


# PRE-SCAN — build a global set of already-extracted chunk filenames
def build_existing_chunks_set(output_dir: Path) -> set[str]:
    """
    Scan every domain directory across all splits once and return a set
    of already-extracted chunk filenames (stem only, no path).

    Checking membership in this set (O(1) dict lookup) replaces thousands
    of individual Path.exists() filesystem calls.
    """
    existing = set()
    for split in ALL_SPLITS:
        for domain in [DOMAIN_CORRUPTED, DOMAIN_MOTFREE]:
            domain_dir = output_dir / split / domain
            if domain_dir.exists():
                for f in domain_dir.glob("*.nii.gz"):
                    existing.add(f.name)

    log.info(f"  Pre-scan: {len(existing):,} chunks already on disk "
             f"(will be skipped regardless of split/domain)")
    return existing



#  Data loading and validation 
def load_and_validate(csv_path: Path, label: str) -> pd.DataFrame:
    required = {
        "subject_id", "session_id", "run_id", "task",
        "preprocessed_bold_file", "chunk_start", "chunk_end",
    }
    df      = pd.read_csv(csv_path)
    missing = required - set(df.columns)
    if missing:
        log.error(f"[{label}] Missing columns: {missing}")
        sys.exit(1)
    log.info(f"  [{label}] {len(df)} rows  ←  {csv_path.name}")
    return df


def filter_existing_files(df: pd.DataFrame, label: str) -> pd.DataFrame:
    n_before = len(df)
    if "preprocessed_exists" in df.columns:
        df = df[df["preprocessed_exists"] == True].copy()
    exists = df["preprocessed_bold_file"].apply(lambda p: Path(p).exists())
    df     = df[exists].copy().reset_index(drop=True)
    dropped = n_before - len(df)
    if dropped:
        log.warning(f"  [{label}] Dropped {dropped} rows — file not on disk")
    return df


# Subject level split (stratified by subject_id, same split for both domains)
def split_subjects(subjects: list[str],
                   seed: int) -> tuple[list[str], list[str], list[str]]:
    train_subs, valtest = train_test_split(
        subjects, test_size=VAL_RATIO + TEST_RATIO, random_state=seed,
    )
    val_subs, test_subs = train_test_split(
        valtest,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=seed,
    )
    return train_subs, val_subs, test_subs


def assign_splits(df_corrupted: pd.DataFrame,
                  df_motfree:   pd.DataFrame,
                  seed: int) -> tuple[dict, dict]:
    all_subs = sorted(
        set(df_corrupted["subject_id"].unique()) |
        set(df_motfree["subject_id"].unique())
    )
    log.info(f"  Total unique subjects: {len(all_subs)}")

    train_subs, val_subs, test_subs = split_subjects(all_subs, seed)
    log.info(f"  train={len(train_subs)}  val={len(val_subs)}  "
             f"test={len(test_subs)}")

    split_map = {"train": train_subs, "val": val_subs, "test": test_subs}
    splits = {
        split: {
            "corrupted": df_corrupted[
                df_corrupted["subject_id"].isin(subs)
            ].copy().reset_index(drop=True),
            "motfree": df_motfree[
                df_motfree["subject_id"].isin(subs)
            ].copy().reset_index(drop=True),
            "subjects": subs,
        }
        for split, subs in split_map.items()
    }
    return splits, split_map

# Balanced sampling of corrupted chunks to match the number of motion-free chunks
def balanced_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        log.warning(f"  Corrupted pool ({len(df)}) < target ({n}). Using all.")
        return df.copy()
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


# log entry factory 
def _log_entry(split, domain, row, output_path, status, message) -> dict:
    return {
        "split"      : split,
        "domain"     : domain,
        "subject_id" : row["subject_id"],
        "session_id" : row["session_id"],
        "run_id"     : row["run_id"],
        "task"       : row["task"],
        "chunk_start": int(row["chunk_start"]),
        "chunk_end"  : int(row["chunk_end"]),
        "output_path": output_path,
        "status"     : status,
        "message"    : message,
        "timestamp"  : datetime.now().isoformat(timespec="seconds"),
    }


# EXTRACTION — one source file (called in parallel)
def _extract_one_file(bold_path_str:  str,
                      group:          pd.DataFrame,
                      domain_dir:     Path,
                      domain:         str,
                      split:          str,
                      existing_names: set[str]) -> tuple[dict, list[dict]]:
    chunk_map: dict[int, str] = {}
    log_rows:  list[dict]     = []

    bold_path = Path(bold_path_str)

    try:
        img  = nib.load(str(bold_path))
        data = np.asarray(img.dataobj, dtype=np.float32)
    except Exception as exc:
        for idx, row in group.iterrows():
            log_rows.append(
                _log_entry(split, domain, row, "", "error",
                           f"Load failed: {exc}")
            )
        return chunk_map, log_rows

    if data.ndim != 4:
        for idx, row in group.iterrows():
            log_rows.append(
                _log_entry(split, domain, row, "", "error",
                           f"Not 4D: {data.shape}")
            )
        return chunk_map, log_rows

    n_vols = data.shape[3]

    for idx, row in group.iterrows():
        chunk_start = int(row["chunk_start"])
        chunk_end   = int(row["chunk_end"])

        filename = CHUNK_FILENAME_TEMPLATE.format(
            subject_id  = row["subject_id"],
            session_id  = row["session_id"],
            task        = row["task"],
            run_id      = str(row["run_id"]).zfill(3),
            chunk_start = chunk_start,
            chunk_end   = chunk_end,
        )
        output_path = domain_dir / filename

        if filename in existing_names:
            chunk_map[idx] = str(output_path)
            log_rows.append(
                _log_entry(split, domain, row,
                           str(output_path), "skipped_exists", "")
            )
            continue

        if chunk_start < 0 or chunk_end >= n_vols or chunk_start > chunk_end:
            msg = (f"Invalid indices [{chunk_start}:{chunk_end}] "
                   f"— file has {n_vols} volumes")
            log_rows.append(
                _log_entry(split, domain, row, "", "error", msg)
            )
            continue

        try:
            chunk_data = data[..., chunk_start : chunk_end + 1]
            out_img    = nib.Nifti1Image(chunk_data,
                                          affine = img.affine,
                                          header = img.header)
            out_img.header.set_data_dtype(np.float32)
            out_img.header["dim"][4] = chunk_data.shape[3]

            output_path.parent.mkdir(parents=True, exist_ok=True)

            nib.save(out_img, str(output_path),
                     **{"compression": GZIP_LEVEL}
                     if "compression" in nib.save.__code__.co_varnames
                     else {})

            chunk_map[idx] = str(output_path)
            log_rows.append(
                _log_entry(split, domain, row,
                           str(output_path), "ok", "")
            )

        except Exception as exc:
            log_rows.append(
                _log_entry(split, domain, row, "", "error", str(exc))
            )

    return chunk_map, log_rows


# EXTRACTION — all chunks for a dataframe (parallel across source files)
def extract_all_chunks(df:             pd.DataFrame,
                       domain_dir:     Path,
                       domain:         str,
                       split:          str,
                       log_rows:       list,
                       existing_names: set[str],
                       n_jobs:         int) -> pd.DataFrame:
    domain_dir.mkdir(parents=True, exist_ok=True)

    df       = df.copy()
    df.index = range(len(df))
    chunk_paths = [""] * len(df)

    groups = list(df.groupby("preprocessed_bold_file"))

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_extract_one_file)(
            bold_path_str  = bold_path_str,
            group          = group,
            domain_dir     = domain_dir,
            domain         = domain,
            split          = split,
            existing_names = existing_names,
        )
        for bold_path_str, group in groups
    )

    n_ok = n_skip = n_err = 0
    for chunk_map, file_log_rows in results:
        for idx, path in chunk_map.items():
            chunk_paths[idx] = path
        for entry in file_log_rows:
            log_rows.append(entry)
            if entry["status"] == "ok":
                n_ok   += 1
            elif entry["status"] == "skipped_exists":
                n_skip += 1
            else:
                n_err  += 1

    log.info(
        f"  [{split}/{domain}]  "
        f"extracted={n_ok}  skipped={n_skip}  errors={n_err}"
    )

    df["chunk_path"] = chunk_paths
    return df


# Save metadata CSVs
def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"  Saved → {path.name}  ({len(df)} rows)")


def main() -> None:
    args = parse_args()

    log.info("=" * 65)
    log.info("  BUILD CYCLEGAN DATASET")
    log.info("=" * 65)
    log.info(f"  Corrupted CSV       : {args.corrupted_csv.name}")
    log.info(f"  Motion-free video   : {args.motion_free_video_csv.name}")
    log.info(f"  Motion-free rest    : {args.motion_free_rest_csv.name}")
    log.info(f"  Output dir          : {args.output_dir}")
    log.info(f"  Seed                : {args.seed}")
    log.info(f"  Workers             : {args.n_jobs}")
    log.info(f"  Skip extraction     : {args.skip_extraction}")
    log.info(f"  Splits to process   : {args.splits}")
    log.info("=" * 65)

    if "train" not in args.splits:
        log.info("  ⚠  'train' not in --splits — training data will NOT be "
                 "generated or overwritten.")

    #  Create output directories
    meta_dir = args.output_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for split in ALL_SPLITS:
        for domain in [DOMAIN_CORRUPTED, DOMAIN_MOTFREE]:
            (args.output_dir / split / domain).mkdir(parents=True, exist_ok=True)

    # Pre-scan all output directories once
    existing_names: set[str] = set()
    if not args.skip_extraction:
        log.info("\nPre-scanning output directories...")
        existing_names = build_existing_chunks_set(args.output_dir)

    # Load and validate 
    log.info("\nLoading CSVs...")
    df_corrupted = load_and_validate(args.corrupted_csv,         "corrupted")
    df_vid       = load_and_validate(args.motion_free_video_csv, "motfree_video")
    df_rest      = load_and_validate(args.motion_free_rest_csv,  "motfree_rest")

    df_motfree = pd.concat([df_vid, df_rest], ignore_index=True)
    log.info(f"  Combined motion-free: {len(df_motfree)} "
             f"(video={len(df_vid)}  rest={len(df_rest)})")

    # Filter missing files 
    log.info("\nFiltering missing files...")
    df_corrupted = filter_existing_files(df_corrupted, "corrupted")
    df_motfree   = filter_existing_files(df_motfree,   "motion_free")
    log.info(f"  corrupted={len(df_corrupted)}  motion_free={len(df_motfree)}")

    # Subject-level split — always computed from full data to keep
    #    the same subject assignments regardless of --splits value 
    log.info("\nSplitting subjects (full split computed for consistency)...")
    splits, split_map = assign_splits(df_corrupted, df_motfree, args.seed)

    # Save subject split map once (safe to overwrite; content is deterministic)
    save_csv(
        pd.DataFrame([
            {"subject_id": sub, "split": split}
            for split, subs in split_map.items()
            for sub in subs
        ]),
        meta_dir / "subject_split.csv",
    )

    # Process only the requested splits 
    log_rows: list[dict] = []

    for split in ALL_SPLITS:
        if split not in args.splits:
            log.info(f"\n  Skipping '{split}' (not in --splits)")
            continue

        log.info(f"\n{'─' * 55}")
        log.info(f"  {split.upper()}")
        log.info(f"{'─' * 55}")

        df_c  = splits[split]["corrupted"]
        df_mf = splits[split]["motfree"]
        n_mf  = len(df_mf)
        n_c   = len(df_c)

        log.info(f"  Motion-free : {n_mf}  |  Corrupted : {n_c}")

        if split == "train":
            save_csv(df_mf, meta_dir / "train_motion_free.csv")
            save_csv(df_c,  meta_dir / "train_corrupted_all.csv")
            df_c_bal = balanced_sample(df_c, n_mf, args.seed)
            save_csv(df_c_bal, meta_dir / "train_corrupted_balanced_epoch0.csv")
            log.info(f"  Balanced epoch-0 corrupted: {len(df_c_bal)}")

            if not args.skip_extraction:
                log.info("  Extracting B_motion_free...")
                df_mf = extract_all_chunks(
                    df_mf, args.output_dir / split / DOMAIN_MOTFREE,
                    DOMAIN_MOTFREE, split, log_rows,
                    existing_names, args.n_jobs,
                )
                log.info("  Extracting A_corrupted (all)...")
                df_c = extract_all_chunks(
                    df_c, args.output_dir / split / DOMAIN_CORRUPTED,
                    DOMAIN_CORRUPTED, split, log_rows,
                    existing_names, args.n_jobs,
                )

            save_csv(df_mf, meta_dir / "train_motion_free.csv")
            save_csv(df_c,  meta_dir / "train_corrupted_all.csv")

        else:
            df_c_bal = balanced_sample(df_c, n_mf, args.seed)
            log.info(f"  Balanced corrupted: {len(df_c_bal)} "
                     f"(to match {n_mf} motion-free)")

            save_csv(df_mf,    meta_dir / f"{split}_motion_free.csv")
            save_csv(df_c_bal, meta_dir / f"{split}_corrupted_balanced.csv")

            if not args.skip_extraction:
                log.info(f"  Extracting B_motion_free...")
                df_mf = extract_all_chunks(
                    df_mf, args.output_dir / split / DOMAIN_MOTFREE,
                    DOMAIN_MOTFREE, split, log_rows,
                    existing_names, args.n_jobs,
                )
                log.info(f"  Extracting A_corrupted (balanced)...")
                df_c_bal = extract_all_chunks(
                    df_c_bal, args.output_dir / split / DOMAIN_CORRUPTED,
                    DOMAIN_CORRUPTED, split, log_rows,
                    existing_names, args.n_jobs,
                )

            save_csv(df_mf,    meta_dir / f"{split}_motion_free.csv")
            save_csv(df_c_bal, meta_dir / f"{split}_corrupted_balanced.csv")

    #  Save extraction log 
    if log_rows:
        save_csv(pd.DataFrame(log_rows), meta_dir / "extraction_log.csv")

    # print summary table
    log.info("\n" + "=" * 65)
    log.info("  SUMMARY")
    log.info("=" * 65)
    for split in ALL_SPLITS:
        skipped = " [skipped]" if split not in args.splits else ""
        n_sub = len(splits[split]["subjects"])
        n_mf  = len(splits[split]["motfree"])
        n_c   = len(splits[split]["corrupted"])
        log.info(f"  {split:5s}  subjects={n_sub:3d}  "
                 f"motion_free={n_mf:5d}  corrupted={n_c:5d}{skipped}")
    if log_rows:
        df_log = pd.DataFrame(log_rows)
        log.info(
            f"\n  Extraction — "
            f"ok={int((df_log['status']=='ok').sum())}  "
            f"skipped={int((df_log['status']=='skipped_exists').sum())}  "
            f"errors={int((df_log['status']=='error').sum())}"
        )
    log.info(f"\n  Output → {args.output_dir.resolve()}")
    log.info("=" * 65)


if __name__ == "__main__":
    main()