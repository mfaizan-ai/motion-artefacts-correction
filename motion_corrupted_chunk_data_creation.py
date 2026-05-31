#!/usr/bin/env python3
"""
corrupted_chunks.py
===================
Extract motion-corrupted chunks from fMRI runs for CycleGAN training.

A chunk of CHUNK_SIZE volumes is accepted as corrupted if ALL four
criteria are met:

  1. Fraction high   : >= MIN_FRAC_HIGH   of volumes have FD >= THR_HIGH
  2. Sustained event : >= MIN_SUSTAINED   consecutive volumes above THR_HIGH
  3. Mean FD         : chunk mean FD      >= MIN_MEAN_FD
  4. Not catastrophic: chunk max  FD      <  MAX_FD

A sliding window of STEP volumes is used so every possible position
in the run is evaluated — maximising the number of chunks found.

Inputs
------
  --mapping      Mapping CSV: bids_key, video_bold_file, motion_parameter_file
  --output       Output chunks CSV path

Optional
--------
  --radius        Brain radius in mm (default 50.0)
  --chunk_size    Volumes per chunk (default 20)
  --step          Sliding window step in volumes (default 1 = maximum yield)
  --thr_high      FD threshold for a volume to count as high-motion (default 1.0 mm)
  --min_frac_high Min fraction of volumes above thr_high (default 0.5 = 50%)
  --min_sustained Min consecutive volumes above thr_high (default 3)
  --min_mean_fd   Min chunk mean FD in mm (default 1.0)
  --max_fd        Max allowed FD in chunk — catastrophe guard (default 10.0 mm)

Outputs
-------
  CSV with one row per accepted chunk:
    subject_id, session_id, run_id, task,
    bold_file, motion_parameter_file,
    chunk_start, chunk_end,
    chunk_mean_fd, chunk_max_fd,
    n_vols_above_thr, frac_vols_above_thr,
    max_sustained_streak,
    n_total_vols_in_run

Usage
-----
  python corrupted_chunks.py \\
      --mapping  bold_parameters_mapping.csv \\
      --output   corrupted_chunks.csv
"""

import argparse
import logging
import re
import sys
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
        description     = "Extract motion-corrupted chunks from fMRI runs.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--mapping", required=True, type=Path, metavar="CSV",
        help="Mapping CSV: bids_key, video_bold_file, motion_parameter_file",
    )
    parser.add_argument(
        "--output", required=True, type=Path, metavar="CSV",
        help="Output corrupted chunks CSV (e.g. corrupted_chunks.csv)",
    )

    # ── FD computation ─────────────────────────────────────────────────────
    parser.add_argument(
        "--radius", type=float, default=50.0, metavar="MM",
        help="Brain radius in mm for rotation→mm conversion",
    )

    # ── Chunk selection ────────────────────────────────────────────────────
    parser.add_argument(
        "--chunk_size", type=int, default=20, metavar="N",
        help="Number of volumes per chunk",
    )
    parser.add_argument(
        "--step", type=int, default=1, metavar="N",
        help="Sliding window step (1 = maximum chunks, larger = less overlap)",
    )

    # ── Corruption criteria ────────────────────────────────────────────────
    parser.add_argument(
        "--thr_high", type=float, default=1.0, metavar="MM",
        help="FD threshold (mm) for a volume to count as high-motion",
    )
    parser.add_argument(
        "--min_frac_high", type=float, default=0.5, metavar="FRAC",
        help="Min fraction of chunk volumes that must exceed thr_high (0–1)",
    )
    parser.add_argument(
        "--min_sustained", type=int, default=3, metavar="N",
        help="Min consecutive volumes above thr_high required in the chunk",
    )
    parser.add_argument(
        "--min_mean_fd", type=float, default=1.0, metavar="MM",
        help="Min chunk mean FD in mm",
    )
    parser.add_argument(
        "--max_fd", type=float, default=10.0, metavar="MM",
        help="Max allowed FD in chunk — catastrophe guard",
    )

    args = parser.parse_args()

    # ── Validation ─────────────────────────────────────────────────────────
    if not args.mapping.exists():
        parser.error(f"Mapping CSV not found: {args.mapping}")
    if not (0.0 < args.min_frac_high <= 1.0):
        parser.error("--min_frac_high must be between 0 and 1.")
    if args.min_sustained < 1:
        parser.error("--min_sustained must be >= 1.")
    if args.min_mean_fd <= 0:
        parser.error("--min_mean_fd must be positive.")
    if args.max_fd <= args.min_mean_fd:
        parser.error("--max_fd must be greater than --min_mean_fd.")
    if args.step < 1:
        parser.error("--step must be >= 1.")

    return args


# =============================================================================
# BIDS KEY PARSER
# =============================================================================
def parse_bids_key(bids_key: str) -> dict:
    """
    Extract BIDS entities from a bids_key string.

    Handles keys of the form:
        sub-ICC157_ses-2_task-rest10_run-001

    Returns dict with: subject_id, session_id, task, run_id.
    Falls back to 'unknown' for any missing entity.
    """
    patterns = {
        "subject_id" : r"sub-([A-Za-z0-9]+)",
        "session_id" : r"ses-([A-Za-z0-9]+)",
        "task"       : r"task-([A-Za-z0-9]+)",
        "run_id"     : r"run-([A-Za-z0-9]+)",
    }
    return {
        field: (re.search(p, bids_key).group(1)
                if re.search(p, bids_key) else "unknown")
        for field, p in patterns.items()
    }


# =============================================================================
# FD COMPUTATION
# =============================================================================
def compute_fd(par_file: Path, radius_mm: float) -> np.ndarray:
    """
    Compute framewise displacement from an FSL MCFLIRT .par file.

    Parameters
    ----------
    par_file  : Path    6-column motion parameter file.
    radius_mm : float   Brain radius for rotation→mm arc-length conversion.

    Returns
    -------
    np.ndarray  FD values shape (n_volumes,). FD[0] is always 0.
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

    rot_mm   = motion[:, :3] * radius_mm
    trans_mm = motion[:,  3:]
    diff     = np.diff(np.hstack([rot_mm, trans_mm]), axis=0)
    fd       = np.sum(np.abs(diff), axis=1)
    return np.concatenate([[0.0], fd])


# =============================================================================
# CHUNK CRITERIA
# =============================================================================
def max_consecutive_above(fd_chunk: np.ndarray, threshold: float) -> int:
    """
    Return the length of the longest consecutive run of volumes
    where FD >= threshold within fd_chunk.

    Parameters
    ----------
    fd_chunk  : np.ndarray  FD values for one chunk.
    threshold : float       FD threshold.

    Returns
    -------
    int  Longest streak length (0 if none).
    """
    max_streak = current = 0
    for v in fd_chunk:
        if v >= threshold:
            current   += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def is_corrupted_chunk(fd_chunk:     np.ndarray,
                        thr_high:     float,
                        min_frac_high:float,
                        min_sustained:int,
                        min_mean_fd:  float,
                        max_fd:       float) -> tuple[bool, dict]:
    """
    Evaluate whether a chunk meets all corruption criteria.

    Criteria
    --------
    1. frac_high   >= min_frac_high   (50% of volumes above thr_high)
    2. max_streak  >= min_sustained   (3+ consecutive volumes above thr_high)
    3. mean_fd     >= min_mean_fd     (meaningful severity)
    4. max_fd_val  <  max_fd          (not catastrophic)

    Parameters
    ----------
    fd_chunk      : np.ndarray  FD values for the candidate chunk.
    thr_high      : float       High-motion volume threshold (mm).
    min_frac_high : float       Min fraction of volumes above thr_high.
    min_sustained : int         Min consecutive volumes above thr_high.
    min_mean_fd   : float       Min chunk mean FD (mm).
    max_fd        : float       Max allowed FD in chunk (mm).

    Returns
    -------
    (bool, dict)
        bool  : True if all criteria are met.
        dict  : Per-criterion values for logging / CSV columns.
    """
    n             = len(fd_chunk)
    mean_fd       = float(np.mean(fd_chunk))
    max_fd_val    = float(np.max(fd_chunk))
    n_above       = int((fd_chunk >= thr_high).sum())
    frac_high     = n_above / n
    max_streak    = max_consecutive_above(fd_chunk, thr_high)

    stats = {
        "chunk_mean_fd"        : round(mean_fd,    6),
        "chunk_max_fd"         : round(max_fd_val, 6),
        "n_vols_above_thr"     : n_above,
        "frac_vols_above_thr"  : round(frac_high,  4),
        "max_sustained_streak" : max_streak,
    }

    passed = (
        frac_high  >= min_frac_high   # criterion 1 — fraction above threshold
        and max_streak >= min_sustained  # criterion 2 — sustained corruption
        and mean_fd    >= min_mean_fd    # criterion 3 — meaningful severity
        and max_fd_val <  max_fd         # criterion 4 — not catastrophic
    )

    return passed, stats


# =============================================================================
# PER-RUN CHUNK EXTRACTION
# =============================================================================
def extract_corrupted_chunks(fd:           np.ndarray,
                              chunk_size:   int,
                              step:         int,
                              thr_high:     float,
                              min_frac_high:float,
                              min_sustained:int,
                              min_mean_fd:  float,
                              max_fd:       float) -> list[dict]:
    """
    Slide a window across the FD timeseries and collect all chunks
    that satisfy the corruption criteria.

    Parameters
    ----------
    fd            : np.ndarray  Full FD timeseries for one run.
    chunk_size    : int         Volumes per chunk.
    step          : int         Sliding window step size.
    thr_high      : float       High-motion volume threshold (mm).
    min_frac_high : float       Min fraction above thr_high.
    min_sustained : int         Min consecutive volumes above thr_high.
    min_mean_fd   : float       Min chunk mean FD (mm).
    max_fd        : float       Max allowed chunk FD — catastrophe guard (mm).

    Returns
    -------
    list of dict  Each dict has chunk_start, chunk_end, and per-criterion stats.
    """
    n_vols  = len(fd)
    chunks  = []

    for start in range(0, n_vols - chunk_size + 1, step):
        end      = start + chunk_size - 1
        fd_chunk = fd[start : end + 1]

        passed, stats = is_corrupted_chunk(
            fd_chunk, thr_high, min_frac_high,
            min_sustained, min_mean_fd, max_fd,
        )

        if passed:
            chunks.append({
                "chunk_start": start,
                "chunk_end"  : end,
                **stats,
            })

    return chunks


# =============================================================================
# PER-RUN PROCESSING
# =============================================================================
def process_run(row:  pd.Series,
                args: argparse.Namespace) -> list[dict]:
    """
    Process one run: compute FD then extract all corrupted chunks.

    Parameters
    ----------
    row  : pd.Series          One row from the mapping CSV.
    args : argparse.Namespace Parsed arguments.

    Returns
    -------
    list of dict  One dict per accepted chunk (empty list on failure).
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

    # ── Extract corrupted chunks via sliding window ────────────────────────
    chunks = extract_corrupted_chunks(
        fd            = fd,
        chunk_size    = args.chunk_size,
        step          = args.step,
        thr_high      = args.thr_high,
        min_frac_high = args.min_frac_high,
        min_sustained = args.min_sustained,
        min_mean_fd   = args.min_mean_fd,
        max_fd        = args.max_fd,
    )

    if not chunks:
        log.info(f"[{bids_key}] No corrupted chunks found (n_vols={n_vols})")
        return []

    log.info(
        f"[{bids_key}]  vols={n_vols}  "
        f"corrupted_chunks={len(chunks)}  "
        f"mean_fd_range="
        f"{min(c['chunk_mean_fd'] for c in chunks):.3f}–"
        f"{max(c['chunk_mean_fd'] for c in chunks):.3f} mm"
    )

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
            "chunk_mean_fd"         : chunk["chunk_mean_fd"],
            "chunk_max_fd"          : chunk["chunk_max_fd"],
            "n_vols_above_thr"      : chunk["n_vols_above_thr"],
            "frac_vols_above_thr"   : chunk["frac_vols_above_thr"],
            "max_sustained_streak"  : chunk["max_sustained_streak"],
            "n_total_vols_in_run"   : n_vols,
        })

    return records


def main():
    args = parse_args()

    # ── Print configuration ────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  CORRUPTED CHUNK EXTRACTION")
    log.info("=" * 60)
    log.info(f"  Mapping CSV          : {args.mapping}")
    log.info(f"  Output CSV           : {args.output}")
    log.info(f"  Brain radius         : {args.radius} mm")
    log.info(f"  Chunk size           : {args.chunk_size} volumes")
    log.info(f"  Sliding window step  : {args.step} volume(s)")
    log.info(f"  High-motion threshold: {args.thr_high} mm")
    log.info(f"  Min fraction high    : {args.min_frac_high * 100:.0f}%")
    log.info(f"  Min sustained streak : {args.min_sustained} volumes")
    log.info(f"  Min chunk mean FD    : {args.min_mean_fd} mm")
    log.info(f"  Max chunk FD (guard) : {args.max_fd} mm")
    log.info("=" * 60)

    # ── Load mapping CSV ───────────────────────────────────────────────────
    mapping = pd.read_csv(args.mapping)
    required = {"bids_key", "video_bold_file", "motion_parameter_file"}
    missing  = required - set(mapping.columns)
    if missing:
        log.error(f"Mapping CSV is missing columns: {missing}")
        sys.exit(1)

    log.info(f"Loaded {len(mapping)} runs from {args.mapping}")

    # ── Process all runs ───────────────────────────────────────────────────
    all_records = []

    for _, row in tqdm(mapping.iterrows(),
                       total = len(mapping),
                       desc  = "Processing runs",
                       unit  = "run"):
        records = process_run(row, args)
        all_records.extend(records)

    # ── Save output CSV ────────────────────────────────────────────────────
    if not all_records:
        log.warning("No corrupted chunks found across all runs.")
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
        "n_vols_above_thr",
        "frac_vols_above_thr",
        "max_sustained_streak",
        "n_total_vols_in_run",
    ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)

    # ── Final summary ──────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  DONE")
    log.info("=" * 60)
    log.info(f"  Runs processed         : {len(mapping)}")
    log.info(f"  Runs with chunks       : {df_out.groupby(['subject_id','session_id','run_id']).ngroups}")
    log.info(f"  Total corrupted chunks : {len(df_out)}")
    log.info(f"  Subjects covered       : {df_out['subject_id'].nunique()}")
    log.info(f"  Mean FD across chunks  : {df_out['chunk_mean_fd'].mean():.4f} mm")
    log.info(f"  Median streak length   : {df_out['max_sustained_streak'].median():.1f} vols")
    log.info(f"  Output saved           : {args.output}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()