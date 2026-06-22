#!/usr/bin/env python3
"""
Usage
-----
    python preprocess_cyclegan_data.py \\
        --subject     ICC155 \\
        --input_dir   /path/to/normalized_to_common_space \\
        --output_dir  /path/to/output \\
        --mask_dir    /path/to/templates/mask
"""
import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm


# Default paths (can be overridden by command-line arguments)
DEFAULT_INPUT_DIR = Path(
    "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives"
    "/normalized_to_common_space"
)
DEFAULT_OUTPUT_DIR = Path(
    "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives"
    "/normalized_to_common_space_cycleGANS_preprocessed_data"
)
DEFAULT_MASK_DIR = Path(
    "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives"
    "/templates/mask"
)

MASK_YOUNGER_NAME  = "nihpd_asym_02-05_fcgmask_2mm.nii.gz"
MASK_OLDER_NAME    = "nihpd_asym_08-11_fcgmask_2mm.nii.gz"
INCLUDE_TASK_KEYWORDS = ["videos", "rest"]
SUFFIX_MASKED      = "_masked.nii.gz"


# Argument parsing
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description     = "Brain-mask Foundcog fMRI data for CycleGAN training.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--subject", type=str, metavar="ID",
        help="Single subject ID to process (e.g. ICC155 or ICC155A).",
    )
    group.add_argument(
        "--all", action="store_true",
        help="Process all subjects found in input_dir sequentially.",
    )
    parser.add_argument(
        "--input_dir",  type=Path, default=DEFAULT_INPUT_DIR,  metavar="DIR",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, metavar="DIR",
    )
    parser.add_argument(
        "--mask_dir",   type=Path, default=DEFAULT_MASK_DIR,   metavar="DIR",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        parser.error(f"input_dir does not exist: {args.input_dir}")
    if not args.mask_dir.exists():
        parser.error(f"mask_dir does not exist: {args.mask_dir}")

    return args

# Logger setup
def setup_logging(subject: str = None,
                  log_dir: Path = None) -> logging.Logger:
    name = f"cyclegan_{subject}" if subject else "cyclegan_preproc"
    log  = logging.getLogger(name)
    log.setLevel(logging.INFO)

    if log.handlers:
        return log

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(ch)

    if subject and log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{subject}.log")
        fh.setFormatter(fmt)
        log.addHandler(fh)

    return log


# Utility functions
def get_mask_path(subject_id: str, mask_dir: Path) -> Path:
    """
    Return the correct age-appropriate brain mask for a subject.
    Subjects ending with 'A' are older infants → 08-11 mask.
    All others are younger infants → 02-05 mask.
    """
    name = MASK_OLDER_NAME if subject_id.endswith("A") else MASK_YOUNGER_NAME
    path = mask_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Brain mask not found: {path}")
    return path


def is_included_task(task_name: str) -> bool:
    """Return True for videos and rest tasks, False for pictures."""
    return any(kw in task_name.lower() for kw in INCLUDE_TASK_KEYWORDS)


def discover_files(subject_id: str, input_dir: Path) -> list[dict]:
    """Find all eligible BOLD files for one subject."""
    subject_dir = input_dir / f"_subject_id_{subject_id}"
    if not subject_dir.exists():
        return []

    records = []
    for nii_file in sorted(subject_dir.rglob("*.nii.gz")):
        parent     = nii_file.parent.name
        task_match = re.search(r"task_name_(\w+)", parent)
        if not task_match or not is_included_task(task_match.group(1)):
            continue

        ses_match = re.search(r"session_(\w+)",      parent)
        run_match = re.search(r"_run_(\w+)_session", parent)

        records.append({
            "subject"        : subject_id,
            "session"        : ses_match.group(1) if ses_match else "unknown",
            "run"            : run_match.group(1) if run_match else "unknown",
            "task"           : task_match.group(1),
            "input_path"     : nii_file,
            "output_rel_dir" : nii_file.relative_to(input_dir).parent,
        })

    return records


def list_all_subjects(input_dir: Path) -> list[str]:
    """Return sorted list of all subject IDs in input_dir."""
    return sorted(
        d.name.replace("_subject_id_", "")
        for d in input_dir.iterdir()
        if d.is_dir() and d.name.startswith("_subject_id_")
    )


def load_mask(mask_path: Path) -> np.ndarray:
    """Load mask NIfTI and return boolean 3D array (threshold at 0.5)."""
    data = np.asarray(nib.load(str(mask_path)).dataobj, dtype=np.float32)
    return data > 0.5


# CORE — single file processing
def process_file(input_path:     Path,
                 output_dir_run: Path,
                 mask_path:      Path,
                 mask_cache:     dict,
                 log:            logging.Logger) -> dict:
    """
    Apply brain mask to one BOLD NIfTI file and save the result.

    Preserves the original NIfTI header and affine exactly.
    Skips if output already exists.

    Parameters
    ----------
    input_path     : Path   source normalised BOLD
    output_dir_run : Path   destination directory for this run
    mask_path      : Path   brain mask
    mask_cache     : dict   { str(mask_path): np.ndarray }
    log            : Logger

    Returns
    -------
    dict  stats for the manifest CSV
    """
    output_path = output_dir_run / input_path.name.replace(
        ".nii.gz", SUFFIX_MASKED
    )

    # Skip if already done
    if output_path.exists():
        log.info(f"  SKIP (exists): {output_path.name}")
        return {"status": "skipped", "output_path": str(output_path)}

    # Load / cache mask
    cache_key = str(mask_path)
    if cache_key not in mask_cache:
        mask_cache[cache_key] = load_mask(mask_path)
        log.info(
            f"  Mask loaded : {mask_path.name}  "
            f"({mask_cache[cache_key].sum():,} brain voxels)"
        )
    mask = mask_cache[cache_key]

    # Load BOLD
    try:
        bold_img  = nib.load(str(input_path))
        bold_data = np.asarray(bold_img.dataobj, dtype=np.float32)
    except Exception as exc:
        log.error(f"  LOAD ERROR  {input_path.name}: {exc}")
        return {"status": "load_error", "error": str(exc)}

    if bold_data.ndim != 4:
        log.error(f"  SHAPE ERROR : expected 4D, got {bold_data.shape}")
        return {"status": "shape_error"}

    if bold_data.shape[:3] != mask.shape:
        log.error(
            f"  SHAPE MISMATCH : BOLD {bold_data.shape[:3]} "
            f"vs mask {mask.shape}"
        )
        return {"status": "shape_mismatch"}

    # Apply brain mask — zero non-brain voxels
    masked_data          = bold_data.copy()
    masked_data[~mask]   = 0.0

    # Save
    output_dir_run.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(
        masked_data,
        affine = bold_img.affine,
        header = bold_img.header,
    )
    out_img.header.set_data_dtype(np.float32)
    nib.save(out_img, str(output_path))

    log.info(
        f"  OK  {output_path.name}  "
        f"vols={bold_data.shape[3]}  "
        f"brain_voxels={mask.sum():,}"
    )

    return {
        "status"         : "processed",
        "n_volumes"      : bold_data.shape[3],
        "n_brain_voxels" : int(mask.sum()),
        "output_path"    : str(output_path),
    }


# subject processing loop
def process_subject(subject_id: str,
                    input_dir:  Path,
                    output_dir: Path,
                    mask_dir:   Path) -> list[dict]:
    """Process all eligible runs for one subject."""
    log_dir    = output_dir / "logs"
    log        = setup_logging(subject_id, log_dir)
    mask_path  = get_mask_path(subject_id, mask_dir)
    mask_cache: dict = {}

    log.info("=" * 60)
    log.info(f"  Subject : {subject_id}  |  Mask : {mask_path.name}")
    log.info("=" * 60)

    files = discover_files(subject_id, input_dir)
    if not files:
        log.warning(f"No eligible files found for {subject_id}")
        return []

    log.info(f"Eligible files: {len(files)}")

    rows = []
    for f in tqdm(files, desc=subject_id, unit="run"):
        output_dir_run = output_dir / f["output_rel_dir"]

        stats = process_file(
            input_path     = f["input_path"],
            output_dir_run = output_dir_run,
            mask_path      = mask_path,
            mask_cache     = mask_cache,
            log            = log,
        )

        rows.append({
            "subject"        : subject_id,
            "session"        : f["session"],
            "run"            : f["run"],
            "task"           : f["task"],
            "mask_used"      : mask_path.name,
            "input_path"     : str(f["input_path"]),
            "output_path"    : stats.get("output_path", ""),
            "status"         : stats.get("status"),
            "n_volumes"      : stats.get("n_volumes"),
            "n_brain_voxels" : stats.get("n_brain_voxels"),
            "processed_at"   : datetime.now().isoformat(timespec="seconds"),
        })

    n_done    = sum(1 for r in rows if r["status"] == "processed")
    n_skipped = sum(1 for r in rows if r["status"] == "skipped")
    n_errors  = sum(1 for r in rows
                    if r["status"] not in ("processed", "skipped"))

    log.info(
        f"Done — processed={n_done}  skipped={n_skipped}  errors={n_errors}"
    )
    return rows



# CSV manifest handling
def save_manifest(rows: list[dict], output_dir: Path) -> None:
    """Append rows to manifest CSV, deduplicating by output_path."""
    if not rows:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest_cyclegan_preprocessed.csv"
    df_new        = pd.DataFrame(rows)

    if manifest_path.exists():
        df_existing = pd.read_csv(manifest_path)
        df_out      = (
            pd.concat([df_existing, df_new], ignore_index=True)
            .drop_duplicates(subset=["output_path"], keep="last")
        )
    else:
        df_out = df_new

    df_out.to_csv(manifest_path, index=False)


def main() -> None:
    args = parse_args()

    if args.subject:
        rows = process_subject(
            subject_id = args.subject,
            input_dir  = args.input_dir,
            output_dir = args.output_dir,
            mask_dir   = args.mask_dir,
        )
        save_manifest(rows, args.output_dir)

    elif args.all:
        subjects = list_all_subjects(args.input_dir)
        log      = setup_logging()
        log.info(f"Processing {len(subjects)} subjects sequentially")
        for sub in subjects:
            rows = process_subject(
                subject_id = sub,
                input_dir  = args.input_dir,
                output_dir = args.output_dir,
                mask_dir   = args.mask_dir,
            )
            save_manifest(rows, args.output_dir)
        log.info("All subjects complete.")


if __name__ == "__main__":
    main()