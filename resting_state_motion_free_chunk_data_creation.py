#!/usr/bin/env python3
"""
safe_zone_chunks.py
===================
Extract low-motion safe-zone chunks from fMRI resting-state runs.

Algorithm per run
-----------------
1. Compute framewise displacement (FD) from FSL MCFLIRT .par file.
2. Mark every volume where FD >= THR_LOW (0.25 mm) as a spike.
3. Exclude BUF_BEFORE volumes before and BUF_AFTER volumes after every spike.
4. The remaining volumes form safe zones (purely low-motion, away from spikes).
5. Extract non-overlapping chunks of CHUNK_SIZE from each contiguous safe zone.
6. Write one row per chunk to the output CSV.

Inputs
------
  --mapping     Path to mapping CSV.
                Required columns: bids_key, video_bold_file, motion_parameter_file
                bids_key must follow BIDS convention:
                  sub-<id>_ses-<id>_task-<name>_run-<id>
  --output      Path for the output chunks CSV.

Optional
--------
  --radius      Brain radius in mm for rotation→mm conversion (default: 35.0).
  --buf_before  Volumes excluded before each spike (default: 5).
  --buf_after   Volumes excluded after  each spike (default: 10).
  --chunk_size  Volumes per chunk (default: 20).
  --thr_low     FD threshold for low/moderate boundary in mm (default: 0.25).

Outputs
-------
  <output>.csv  One row per chunk with columns:
                subject_id, session_id, run_id, task,
                bold_file, motion_parameter_file,
                chunk_start, chunk_end,
                chunk_mean_fd, chunk_max_fd, n_safe_vols_in_run

Usage
-----
  python safe_zone_chunks.py \\
      --mapping  bold_parameters_mapping.csv \\
      --output   safe_zone_chunks.csv \\
      --radius   50.0 \\
      --buf_before 5 \\
      --buf_after  10 \\
      --chunk_size 20
"""

import argparse
import re
import sys
import logging
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# ARGUMENT PARSING
# =============================================================================
def parse_args():
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description = "Extract low-motion safe-zone chunks from fMRI runs.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--mapping", required=True, type=Path,
        metavar="CSV",
        help="Mapping CSV with columns: bids_key, video_bold_file, motion_parameter_file",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        metavar="CSV",
        help="Output chunks CSV path (e.g. safe_zone_chunks.csv)",
    )

    # ── Optional ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--radius", type=float, default=50.0,
        metavar="MM",
        help="Brain radius in mm for rotation→mm conversion (infant: 35, adult: 50)",
    )
    parser.add_argument(
        "--buf_before", type=int, default=5,
        metavar="N",
        help="Volumes to exclude BEFORE each spike",
    )
    parser.add_argument(
        "--buf_after", type=int, default=10,
        metavar="N",
        help="Volumes to exclude AFTER each spike",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=20,
        metavar="N",
        help="Number of volumes per chunk",
    )
    parser.add_argument(
        "--thr_low", type=float, default=0.25,
        metavar="MM",
        help="FD threshold (mm) above which a volume is treated as a spike",
    )

    args = parser.parse_args()

    # ── Validation ─────────────────────────────────────────────────────────
    if not args.mapping.exists():
        parser.error(f"Mapping CSV not found: {args.mapping}")
    if args.radius <= 0:
        parser.error("--radius must be positive.")
    if args.buf_before < 0 or args.buf_after < 0:
        parser.error("--buf_before and --buf_after must be >= 0.")
    if args.chunk_size < 1:
        parser.error("--chunk_size must be >= 1.")
    if args.thr_low <= 0:
        parser.error("--thr_low must be positive.")

    return args


# =============================================================================
# BIDS KEY PARSER
# =============================================================================
def parse_bids_key(bids_key: str) -> dict:
    """
    Extract BIDS entities from a bids_key string.

    Expects keys of the form:
        sub-ICC157_ses-2_task-rest10_run-001

    Returns a dict with keys: subject_id, session_id, task, run_id.
    Falls back to 'unknown' for any missing entity.

    Parameters
    ----------
    bids_key : str

    Returns
    -------
    dict
    """
    patterns = {
        "subject_id" : r"sub-([A-Za-z0-9]+)",
        "session_id" : r"ses-([A-Za-z0-9]+)",
        "task"       : r"task-([A-Za-z0-9]+)",
        "run_id"     : r"run-([A-Za-z0-9]+)",
    }
    entities = {}
    for field, pattern in patterns.items():
        match = re.search(pattern, bids_key)
        entities[field] = match.group(1) if match else "unknown"
    return entities


# =============================================================================
# FD COMPUTATION
# =============================================================================
def compute_fd(par_file: Path, radius_mm: float) -> np.ndarray:
    """
    Compute framewise displacement from an FSL MCFLIRT .par file.

    Parameters
    ----------
    par_file  : Path    Path to the 6-column .par file.
    radius_mm : float   Brain radius for rotation→mm arc-length conversion.

    Returns
    -------
    np.ndarray  FD values, shape (n_volumes,). FD[0] is always 0.

    Raises
    ------
    FileNotFoundError  If the par file does not exist.
    ValueError         If the file does not have exactly 6 columns.
    """
    if not par_file.exists():
        raise FileNotFoundError(f"Par file not found: {par_file}")

    motion = np.loadtxt(par_file)

    if motion.ndim == 1:
        motion = motion[np.newaxis, :]
    if motion.shape[1] != 6:
        raise ValueError(
            f"Expected 6-column .par file, got shape {motion.shape}: {par_file}"
        )

    # Columns: rot_x, rot_y, rot_z (rad), trans_x, trans_y, trans_z (mm)
    rot_mm   = motion[:, :3] * radius_mm   # radians → mm via arc-length
    trans_mm = motion[:,  3:]               # already in mm
    combined = np.hstack([rot_mm, trans_mm])

    diff     = np.diff(combined, axis=0)    # frame-to-frame differences
    fd       = np.sum(np.abs(diff), axis=1) # L1 norm per frame
    fd       = np.concatenate([[0.0], fd])  # volume 0 has no predecessor

    return fd


# =============================================================================
# SAFE ZONE IDENTIFICATION
# =============================================================================
def build_safe_mask(fd: np.ndarray,
                    thr_low:    float,
                    buf_before: int,
                    buf_after:  int) -> np.ndarray:
    """
    Build a boolean mask of volumes that are in a safe (low-motion) zone.

    A volume is safe if:
      - Its own FD < thr_low, AND
      - It is not within buf_before volumes BEFORE any spike, AND
      - It is not within buf_after  volumes AFTER  any spike.

    Where a spike is any volume with FD >= thr_low.

    Parameters
    ----------
    fd         : np.ndarray  FD timeseries, shape (n_volumes,).
    thr_low    : float       Spike threshold in mm.
    buf_before : int         Volumes to exclude before each spike.
    buf_after  : int         Volumes to exclude after  each spike.

    Returns
    -------
    np.ndarray  Boolean mask, True = safe, shape (n_volumes,).
    """
    n        = len(fd)
    excluded = np.zeros(n, dtype=bool)

    spike_indices = np.where(fd >= thr_low)[0]

    for idx in spike_indices:
        lo = max(0,     idx - buf_before)
        hi = min(n - 1, idx + buf_after)
        excluded[lo : hi + 1] = True

    return ~excluded


def get_safe_regions(safe_mask: np.ndarray) -> list[list[int]]:
    """
    Return contiguous runs of safe (True) volume indices.

    Parameters
    ----------
    safe_mask : np.ndarray  Boolean mask, shape (n_volumes,).

    Returns
    -------
    list of lists  Each inner list is a contiguous block of safe indices.
    """
    regions = []
    for is_safe, group in groupby(enumerate(safe_mask), key=lambda x: x[1]):
        if is_safe:
            indices = [i for i, _ in group]
            regions.append(indices)
    return regions


# =============================================================================
# CHUNK EXTRACTION
# =============================================================================
def extract_chunks_from_regions(safe_regions: list[list[int]],
                                 fd:           np.ndarray,
                                 chunk_size:   int) -> list[dict]:
    """
    Extract non-overlapping chunks of `chunk_size` from safe regions.

    Chunks are taken greedily from the start of each region.
    Any trailing volumes shorter than chunk_size are discarded.

    Parameters
    ----------
    safe_regions : list of lists   Output of get_safe_regions().
    fd           : np.ndarray      Full FD timeseries (for per-chunk stats).
    chunk_size   : int             Volumes per chunk.

    Returns
    -------
    list of dict  Each dict has:
        chunk_start    : int    First volume index (inclusive)
        chunk_end      : int    Last  volume index (inclusive)
        chunk_mean_fd  : float  Mean FD within the chunk
        chunk_max_fd   : float  Max  FD within the chunk
    """
    chunks = []

    for region in safe_regions:
        n_chunks = len(region) // chunk_size

        for c in range(n_chunks):
            start = region[c * chunk_size]
            end   = region[c * chunk_size + chunk_size - 1]
            fd_chunk = fd[start : end + 1]
            chunks.append({
                "chunk_start"   : start,
                "chunk_end"     : end,
                "chunk_mean_fd" : float(np.mean(fd_chunk)),
                "chunk_max_fd"  : float(np.max(fd_chunk)),
            })

    return chunks


# =============================================================================
# PER-RUN PROCESSING
# =============================================================================
def process_run(row:        pd.Series,
                args:       argparse.Namespace) -> list[dict]:
    """
    Process one run: compute FD, find safe zones, extract chunks.

    Parameters
    ----------
    row  : pd.Series        One row from the mapping CSV.
    args : argparse.Namespace

    Returns
    -------
    list of dict  One dict per chunk (empty list if run fails or yields none).
    """
    bids_key  = row["bids_key"]
    bold_file = row["video_bold_file"]
    par_file  = Path(row["motion_parameter_file"])
    entities  = parse_bids_key(bids_key)

    # ── Compute FD ─────────────────────────────────────────────────────────
    try:
        fd = compute_fd(par_file, args.radius)
    except (FileNotFoundError, ValueError) as e:
        log.warning(f"[{bids_key}] Skipping — {e}")
        return []

    n_vols = len(fd)

    # ── Build safe mask ────────────────────────────────────────────────────
    safe_mask    = build_safe_mask(fd, args.thr_low, args.buf_before, args.buf_after)
    n_safe_vols  = int(safe_mask.sum())
    safe_regions = get_safe_regions(safe_mask)

    # ── Extract chunks ─────────────────────────────────────────────────────
    chunks = extract_chunks_from_regions(safe_regions, fd, args.chunk_size)

    if not chunks:
        log.info(
            f"[{bids_key}] No chunks extracted "
            f"(n_vols={n_vols}, n_safe={n_safe_vols}, "
            f"need >={args.chunk_size} consecutive safe vols)"
        )
        return []

    # ── Build output rows ──────────────────────────────────────────────────
    records = []
    for chunk in chunks:
        records.append({
            "subject_id"            : entities["subject_id"],
            "session_id"            : entities["session_id"],
            "run_id"                : entities["run_id"],
            "task"                  : entities["task"],
            "bold_file"             : bold_file,
            "motion_parameter_file" : str(par_file),
            "chunk_start"           : chunk["chunk_start"],
            "chunk_end"             : chunk["chunk_end"],
            "chunk_mean_fd"         : round(chunk["chunk_mean_fd"], 6),
            "chunk_max_fd"          : round(chunk["chunk_max_fd"],  6),
            "n_safe_vols_in_run"    : n_safe_vols,
            "n_total_vols_in_run"   : n_vols,
        })

    log.info(
        f"[{bids_key}]  vols={n_vols}  safe={n_safe_vols}  "
        f"regions={len(safe_regions)}  chunks={len(chunks)}"
    )
    return records



def main():
    args = parse_args()

    # ── Print run configuration ────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  SAFE ZONE CHUNK EXTRACTION")
    log.info("=" * 60)
    log.info(f"  Mapping CSV      : {args.mapping}")
    log.info(f"  Output CSV       : {args.output}")
    log.info(f"  Brain radius     : {args.radius} mm")
    log.info(f"  Spike threshold  : FD >= {args.thr_low} mm")
    log.info(f"  Buffer before    : {args.buf_before} volumes")
    log.info(f"  Buffer after     : {args.buf_after} volumes")
    log.info(f"  Chunk size       : {args.chunk_size} volumes")
    log.info("=" * 60)

    # ── Load mapping CSV ───────────────────────────────────────────────────
    mapping = pd.read_csv(args.mapping)
    required_cols = {"bids_key", "video_bold_file", "motion_parameter_file"}
    missing = required_cols - set(mapping.columns)
    if missing:
        log.error(f"Mapping CSV is missing columns: {missing}")
        sys.exit(1)

    log.info(f"Loaded {len(mapping)} runs from {args.mapping}")

    # ── Process all runs ───────────────────────────────────────────────────
    all_records = []

    for _, row in tqdm(mapping.iterrows(),
                       total    = len(mapping),
                       desc     = "Processing runs",
                       unit     = "run"):
        records = process_run(row, args)
        all_records.extend(records)

    # ── Save output CSV ────────────────────────────────────────────────────
    if not all_records:
        log.warning("No chunks extracted from any run. Output CSV not written.")
        sys.exit(0)

    df_out = pd.DataFrame(all_records, columns=[
        "subject_id",
        "session_id",
        "run_id",
        "task",
        "bold_file",
        "motion_parameter_file",
        "chunk_start",
        "chunk_end",
        "chunk_mean_fd",
        "chunk_max_fd",
        "n_safe_vols_in_run",
        "n_total_vols_in_run",
    ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)

    # ── Final summary ──────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  DONE")
    log.info("=" * 60)
    log.info(f"  Runs processed       : {len(mapping)}")
    log.info(f"  Runs with chunks     : {df_out.groupby(['subject_id','session_id','run_id']).ngroups}")
    log.info(f"  Total chunks         : {len(df_out)}")
    log.info(f"  Subjects covered     : {df_out['subject_id'].nunique()}")
    log.info(f"  Mean FD across chunks: {df_out['chunk_mean_fd'].mean():.4f} mm")
    log.info(f"  Output saved         : {args.output}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()