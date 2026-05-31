#!/usr/bin/env python3
"""
add_preprocessed_paths.py
=========================
Add a preprocessed_bold_file column to any chunk CSV file by constructing
the path to the brain-masked BOLD in the CycleGAN preprocessed directory.

Works with any chunk CSV that contains these columns:
    subject_id, session_id, run_id, task

Usage
-----
    # Basic — uses default paths
    python add_preprocessed_paths.py --input video_state_clean_chunk_dataset.csv

    # Custom output path
    python add_preprocessed_paths.py \\
        --input  video_state_clean_chunk_dataset.csv \\
        --output video_state_clean_chunk_dataset_updated.csv

    # Custom preprocessed root directory
    python add_preprocessed_paths.py \\
        --input           resting_state_clean_chunk_dataset.csv \\
        --output          resting_state_clean_chunk_dataset_updated.csv \\
        --preprocessed_root /path/to/preprocessed_data

    # Verify paths exist on disk after adding (slower but safer)
    python add_preprocessed_paths.py \\
        --input   video_state_clean_chunk_dataset.csv \\
        --verify
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


# =============================================================================
# DEFAULT PATHS
# =============================================================================

DEFAULT_PREPROCESSED_ROOT = Path(
    "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives"
    "/normalized_to_common_space_cycleGANS_preprocessed_data"
)


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description     = (
            "Add preprocessed_bold_file column to a chunk CSV. "
            "Works for any chunk CSV with columns: "
            "subject_id, session_id, run_id, task."
        ),
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True, type=Path, metavar="CSV",
        help="Input chunk CSV file.",
    )
    parser.add_argument(
        "--output", type=Path, default=None, metavar="CSV",
        help=(
            "Output CSV path. Defaults to input filename with "
            "'_with_preprocessed' appended."
        ),
    )
    parser.add_argument(
        "--preprocessed_root", type=Path,
        default=DEFAULT_PREPROCESSED_ROOT, metavar="DIR",
        help="Root directory of the CycleGAN preprocessed data.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help=(
            "Check whether each constructed preprocessed path "
            "actually exists on disk. Prints a summary at the end."
        ),
    )

    args = parser.parse_args()

    # Validate
    if not args.input.exists():
        parser.error(f"Input CSV not found: {args.input}")

    # Default output path
    if args.output is None:
        args.output = args.input.parent / (
            args.input.stem + "_with_preprocessed.csv"
        )

    return args


# =============================================================================
# PATH CONSTRUCTION
# =============================================================================

def build_preprocessed_path(subject_id: str,
                              session_id: str,
                              run_id:     str,
                              task:       str,
                              root:       Path) -> str:
    """
    Construct the full path to a preprocessed (brain-masked) BOLD file.

    Directory structure:
        {root}/
        _subject_id_{subject_id}/
        _referencetype_standard/
        _run_{run:03d}_session_{session}_task_name_{task}/
        sub-{subject}_ses-{session}_task-{task}_dir-AP_run-{run:03d}
        _bold_mcf_corrected_flirt_masked.nii.gz

    Parameters
    ----------
    subject_id : str   e.g. 'ICC111'
    session_id : str   e.g. '1'
    run_id     : str   e.g. '3'  — zero-padded to 3 digits automatically
    task       : str   e.g. 'videos' or 'rest10'
    root       : Path  preprocessed data root directory

    Returns
    -------
    str  full path to the preprocessed BOLD file
    """
    run_padded = str(run_id).zfill(3)   # 1 → 001,  3 → 003,  12 → 012
    ses        = str(session_id)
    sub        = str(subject_id)

    run_dir    = f"_run_{run_padded}_session_{ses}_task_name_{task}"
    filename   = (
        f"sub-{sub}_ses-{ses}_task-{task}_dir-AP_run-{run_padded}"
        f"_bold_mcf_corrected_flirt_masked.nii.gz"
    )

    return str(
        root
        / f"_subject_id_{sub}"
        / "_referencetype_standard"
        / run_dir
        / filename
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    # ── Load ───────────────────────────────────────────────────────────────
    df = pd.read_csv(args.input)
    print(f"Loaded   : {args.input}  ({len(df)} rows)")

    # ── Validate required columns ──────────────────────────────────────────
    required = {"subject_id", "session_id", "run_id", "task"}
    missing  = required - set(df.columns)
    if missing:
        print(f"ERROR: CSV is missing required columns: {missing}")
        sys.exit(1)

    # ── Build preprocessed path column ────────────────────────────────────
    df["preprocessed_bold_file"] = df.apply(
        lambda row: build_preprocessed_path(
            subject_id = row["subject_id"],
            session_id = row["session_id"],
            run_id     = row["run_id"],
            task       = row["task"],
            root       = args.preprocessed_root,
        ),
        axis=1,
    )

    # ── Insert column right after bold_file (if it exists) ────────────────
    if "bold_file" in df.columns:
        cols      = df.columns.tolist()
        bold_idx  = cols.index("bold_file")
        cols.insert(bold_idx + 1,
                    cols.pop(cols.index("preprocessed_bold_file")))
        df = df[cols]

    # ── Verify paths exist on disk (optional) ─────────────────────────────
    if args.verify:
        print("Verifying paths on disk...")
        df["preprocessed_exists"] = df["preprocessed_bold_file"].apply(
            lambda p: Path(p).exists()
        )
        n_found   = int(df["preprocessed_exists"].sum())
        n_missing = int((~df["preprocessed_exists"]).sum())

        print(f"\n  Found   : {n_found} / {len(df)}")
        print(f"  Missing : {n_missing} / {len(df)}")

        if n_missing > 0:
            print("\n  Missing files:")
            missing_rows = df[~df["preprocessed_exists"]]
            for _, row in missing_rows.iterrows():
                print(f"    sub-{row['subject_id']}  "
                      f"ses-{row['session_id']}  "
                      f"run-{row['run_id']}  "
                      f"task-{row['task']}")

    # ── Save ───────────────────────────────────────────────────────────────
    df.to_csv(args.output, index=False)
    print(f"\nSaved    : {args.output}")
    print(f"Columns  : {df.columns.tolist()}")


if __name__ == "__main__":
    main()