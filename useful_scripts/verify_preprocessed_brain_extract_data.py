#!/usr/bin/env python3
"""
verify_brain_extraction.py
==========================
Verify that brain-masked output files were correctly created for all
video and resting-state runs across all subjects.

Checks per file:
    1. Output file exists
    2. File is loadable (not corrupted)
    3. Dimensions match input (shape unchanged)
    4. Non-brain voxels are zeroed (brain extraction applied)
    5. Brain voxels retain signal (values not all zero)
    6. No all-zero volumes within the brain mask

Outputs:
    verification_report.csv   one row per file with PASS / FAIL / MISSING
    verification_summary.txt  printed summary

Usage
-----
    python verify_brain_extraction.py

    # Custom paths
    python verify_brain_extraction.py \\
        --input_dir  /path/to/normalized_to_common_space \\
        --output_dir /path/to/normalized_to_common_space_cycleGANS_preprocessed_data \\
        --mask_dir   /path/to/templates/mask \\
        --report     /path/to/verification_report.csv
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm


# =============================================================================
# DEFAULT PATHS
# =============================================================================

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
DEFAULT_REPORT = Path("verification_report.csv")

MASK_YOUNGER    = "nihpd_asym_02-05_fcgmask_2mm.nii.gz"
MASK_OLDER      = "nihpd_asym_08-11_fcgmask_2mm.nii.gz"
INCLUDE_TASKS   = ["videos", "rest"]
INPUT_SUFFIX    = ".nii.gz"
OUTPUT_SUFFIX   = "_masked.nii.gz"

# Tolerance: fraction of brain voxels allowed to be zero after masking
# A few edge voxels may genuinely be zero — flag if more than 5%
MAX_ZERO_BRAIN_FRAC = 0.05


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description     = "Verify brain extraction outputs for CycleGAN data.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir",  type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mask_dir",   type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument("--report",     type=Path, default=DEFAULT_REPORT,
                        help="Path to save the verification CSV report.")
    return parser.parse_args()


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
    stream  = sys.stdout,
)
log = logging.getLogger("verify")


# =============================================================================
# UTILITIES
# =============================================================================

def get_mask_path(subject_id: str, mask_dir: Path) -> Path:
    name = MASK_OLDER if subject_id.endswith("A") else MASK_YOUNGER
    return mask_dir / name


def is_included_task(task_name: str) -> bool:
    return any(kw in task_name.lower() for kw in INCLUDE_TASKS)


def load_mask_cached(mask_path: Path, cache: dict) -> np.ndarray:
    key = str(mask_path)
    if key not in cache:
        data       = np.asarray(nib.load(str(mask_path)).dataobj,
                                dtype=np.float32)
        cache[key] = data > 0.5
    return cache[key]


def discover_input_files(input_dir: Path) -> list[dict]:
    """
    Find all video and rest BOLD files in the input directory
    across all subjects.
    """
    records = []
    subject_dirs = sorted(
        d for d in input_dir.iterdir()
        if d.is_dir() and d.name.startswith("_subject_id_")
    )

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name.replace("_subject_id_", "")

        for nii_file in sorted(subject_dir.rglob("*.nii.gz")):
            # Skip any already-masked files that may exist in input
            if "_masked" in nii_file.name:
                continue

            parent     = nii_file.parent.name
            task_match = re.search(r"task_name_(\w+)", parent)
            if not task_match or not is_included_task(task_match.group(1)):
                continue

            ses_match = re.search(r"session_(\w+)",      parent)
            run_match = re.search(r"_run_(\w+)_session", parent)

            # Build expected output path
            rel_dir     = nii_file.relative_to(input_dir).parent
            out_name    = nii_file.name.replace(INPUT_SUFFIX, OUTPUT_SUFFIX)
            output_path = input_dir.parent / (
                input_dir.name.replace(
                    "normalized_to_common_space",
                    "normalized_to_common_space_cycleGANS_preprocessed_data"
                )
            ) / rel_dir / out_name

            records.append({
                "subject"     : subject_id,
                "session"     : ses_match.group(1) if ses_match else "unknown",
                "run"         : run_match.group(1) if run_match else "unknown",
                "task"        : task_match.group(1),
                "input_path"  : nii_file,
                "output_path" : output_path,
                "mask_path"   : get_mask_path(subject_id,
                                              input_dir.parent /
                                              "templates" / "mask"),
            })

    return records


# =============================================================================
# VERIFICATION — single file pair
# =============================================================================

def verify_file(record:     dict,
                mask_cache: dict,
                mask_dir:   Path) -> dict:
    """
    Run all verification checks on one input/output file pair.

    Returns a dict with all check results and a final status:
        PASS    — all checks passed
        FAIL    — file exists but one or more checks failed
        MISSING — output file does not exist
    """
    subject     = record["subject"]
    input_path  = record["input_path"]
    output_path = record["output_path"]
    mask_path   = get_mask_path(subject, mask_dir)

    result = {
        "subject"              : subject,
        "session"              : record["session"],
        "run"                  : record["run"],
        "task"                 : record["task"],
        "input_path"           : str(input_path),
        "output_path"          : str(output_path),
        "mask_used"            : mask_path.name,
        "output_exists"        : False,
        "file_loadable"        : False,
        "dims_match"           : False,
        "n_volumes"            : None,
        "input_shape"          : None,
        "output_shape"         : None,
        "n_brain_voxels"       : None,
        "n_zero_outside_brain" : None,
        "pct_outside_zeroed"   : None,
        "mean_signal_brain"    : None,
        "brain_has_signal"     : False,
        "n_zero_volumes"       : None,
        "status"               : "MISSING",
        "notes"                : "",
    }

    # ── Check 1: output file exists ────────────────────────────────────────
    if not output_path.exists():
        result["status"] = "MISSING"
        return result

    result["output_exists"] = True

    # ── Check 2: file is loadable ──────────────────────────────────────────
    try:
        out_img  = nib.load(str(output_path))
        out_data = np.asarray(out_img.dataobj, dtype=np.float32)
    except Exception as exc:
        result["status"] = "FAIL"
        result["notes"]  = f"Load error: {exc}"
        return result

    result["file_loadable"] = True
    result["output_shape"]  = str(out_data.shape)
    result["n_volumes"]     = out_data.shape[3] if out_data.ndim == 4 else None

    # ── Check 3: dimensions match input ───────────────────────────────────
    try:
        in_data = np.asarray(nib.load(str(input_path)).dataobj,
                             dtype=np.float32)
        result["input_shape"] = str(in_data.shape)
        dims_match            = in_data.shape == out_data.shape
        result["dims_match"]  = dims_match
        if not dims_match:
            result["status"] = "FAIL"
            result["notes"]  = (f"Shape mismatch: input={in_data.shape} "
                                f"output={out_data.shape}")
            return result
    except Exception as exc:
        result["notes"] = f"Input load error: {exc}"

    # ── Load mask ──────────────────────────────────────────────────────────
    try:
        mask = load_mask_cached(mask_path, mask_cache)
    except Exception as exc:
        result["status"] = "FAIL"
        result["notes"]  = f"Mask load error: {exc}"
        return result

    n_brain  = int(mask.sum())
    n_total  = int(np.prod(mask.shape))
    n_outside= n_total - n_brain
    result["n_brain_voxels"] = n_brain

    # ── Check 4: non-brain voxels are zeroed ──────────────────────────────
    # Sample the first volume only — much faster than checking all volumes
    first_vol     = out_data[..., 0]
    outside_vals  = first_vol[~mask]
    n_nonzero_out = int((np.abs(outside_vals) > 1e-6).sum())
    pct_zeroed    = float((n_outside - n_nonzero_out) / n_outside * 100) \
                    if n_outside > 0 else 100.0

    result["n_zero_outside_brain"] = n_nonzero_out
    result["pct_outside_zeroed"]   = round(pct_zeroed, 2)

    outside_ok = n_nonzero_out == 0
    if not outside_ok:
        result["notes"] += (
            f"Non-brain voxels not fully zeroed "
            f"({n_nonzero_out} non-zero outside brain). "
        )

    # ── Check 5: brain voxels have signal ─────────────────────────────────
    brain_vals          = out_data[mask]          # (n_brain, t)
    mean_signal_brain   = float(np.mean(np.abs(brain_vals)))
    result["mean_signal_brain"] = round(mean_signal_brain, 4)
    brain_has_signal    = mean_signal_brain > 1.0  # signal should be well above 0
    result["brain_has_signal"] = brain_has_signal

    if not brain_has_signal:
        result["notes"] += "Brain signal unexpectedly low. "

    # ── Check 6: no all-zero volumes within brain ──────────────────────────
    # For each timepoint, check if all brain voxels are zero
    brain_4d      = out_data[mask]                 # (n_brain_voxels, t)
    vol_means     = np.abs(brain_4d).mean(axis=0)  # mean per timepoint
    n_zero_vols   = int((vol_means < 1e-6).sum())
    result["n_zero_volumes"] = n_zero_vols

    if n_zero_vols > 0:
        result["notes"] += f"{n_zero_vols} all-zero brain volume(s). "

    # ── Final status ───────────────────────────────────────────────────────
    checks_passed = (
        result["file_loadable"]
        and result["dims_match"]
        and outside_ok
        and brain_has_signal
        and n_zero_vols == 0
    )
    result["status"] = "PASS" if checks_passed else "FAIL"

    return result


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    log.info("=" * 60)
    log.info("  BRAIN EXTRACTION VERIFICATION")
    log.info("=" * 60)
    log.info(f"  Input dir  : {args.input_dir}")
    log.info(f"  Output dir : {args.output_dir}")
    log.info(f"  Mask dir   : {args.mask_dir}")
    log.info(f"  Report     : {args.report}")
    log.info("=" * 60)

    # ── Discover all expected input files ──────────────────────────────────
    log.info("Discovering input files...")
    records = discover_input_files(args.input_dir)
    log.info(f"Found {len(records)} expected output files across all subjects")

    # ── Run verification ───────────────────────────────────────────────────
    mask_cache: dict = {}
    results          = []

    for record in tqdm(records, desc="Verifying", unit="file"):
        # Override output path to use args.output_dir
        rel_dir          = record["input_path"].relative_to(args.input_dir).parent
        out_name         = record["input_path"].name.replace(
                               INPUT_SUFFIX, OUTPUT_SUFFIX
                           )
        record["output_path"] = args.output_dir / rel_dir / out_name

        result = verify_file(record, mask_cache, args.mask_dir)
        results.append(result)

    # ── Save CSV report ────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(args.report, index=False)
    log.info(f"\nFull report saved → {args.report}")

    # ── Print summary ──────────────────────────────────────────────────────
    n_total   = len(df)
    n_pass    = int((df["status"] == "PASS").sum())
    n_fail    = int((df["status"] == "FAIL").sum())
    n_missing = int((df["status"] == "MISSING").sum())

    print("\n" + "=" * 60)
    print("  VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  Total expected files : {n_total}")
    print(f"  PASS                 : {n_pass}  ({n_pass/n_total*100:.1f}%)")
    print(f"  FAIL                 : {n_fail}  ({n_fail/n_total*100:.1f}%)")
    print(f"  MISSING              : {n_missing}  ({n_missing/n_total*100:.1f}%)")
    print("=" * 60)

    # Per-task breakdown
    print("\n  Breakdown by task:")
    for task, grp in df.groupby("task"):
        p = int((grp["status"] == "PASS").sum())
        f = int((grp["status"] == "FAIL").sum())
        m = int((grp["status"] == "MISSING").sum())
        print(f"    {task:12s} — PASS={p}  FAIL={f}  MISSING={m}")

    # Show any failures or missing files
    problems = df[df["status"] != "PASS"]
    if not problems.empty:
        print(f"\n  Files needing attention ({len(problems)}):")
        for _, row in problems.iterrows():
            print(f"    [{row['status']:7s}] "
                  f"sub-{row['subject']} "
                  f"ses-{row['session']} "
                  f"run-{row['run']} "
                  f"task-{row['task']}"
                  + (f"  — {row['notes']}" if row["notes"] else ""))
    else:
        print("\n  All files passed verification.")

    print("=" * 60)
    print(f"\n  Full report → {args.report}")


if __name__ == "__main__":
    main()