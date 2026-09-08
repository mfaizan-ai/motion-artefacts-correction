"""
create_subject_fc.py
=====================
Build one subject-level 400x400 FC matrix per subject from create_roi_timeseries.py's
per-run outputs: Pearson correlation per run, Fisher-z, inverse-variance-weighted
average across the subject's runs (runs shorter than --min_volumes are dropped),
inverse Fisher-z back to r. Saved under <roi_timeseries_root>/<fc_dirname>/<subject_id>/fc.npy.

Config is a plain dataclass (FCConfig) so this is reusable from other scripts/notebooks
without going through argparse.
"""
import argparse
import os
import sys
from dataclasses import dataclass
from glob import glob
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from losses import _pearson_corr_matrix


@dataclass
class FCConfig:
    roi_timeseries_root: str
    min_volumes: int = 300
    fc_dirname: str = "subject_fc"

    @property
    def fc_output_root(self) -> str:
        return os.path.join(self.roi_timeseries_root, self.fc_dirname)


def find_subject_runs(roi_timeseries_root: str) -> Dict[str, List[str]]:
    """subject_id -> list of roi_timeseries.npy paths under _subject_id_{ID}/..."""
    subjects: Dict[str, List[str]] = {}
    pattern = os.path.join(roi_timeseries_root, "_subject_id_*", "**", "roi_timeseries.npy")
    for path in glob(pattern, recursive=True):
        rel = os.path.relpath(path, roi_timeseries_root)
        subject_dir = rel.split(os.sep)[0]                       # "_subject_id_ICC103"
        subject_id = subject_dir.removeprefix("_subject_id_")
        subjects.setdefault(subject_id, []).append(path)
    return subjects


def fisher_z(r: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def subject_fc(run_paths: List[str], config: FCConfig) -> Optional[np.ndarray]:
    """Inverse-variance-weighted subject-level FC, dropping runs shorter
    than config.min_volumes. Returns None if no run meets the threshold."""
    z_sum, w_sum, n_rois = None, 0.0, None
    for path in run_paths:
        roi_ts = np.load(path)
        n_timepoints = roi_ts.shape[0]
        if n_timepoints < config.min_volumes:
            continue

        r = _pearson_corr_matrix(torch.from_numpy(roi_ts).float()).numpy()
        n_rois = r.shape[0]
        off_diag = ~np.eye(n_rois, dtype=bool)

        z = np.zeros_like(r)
        z[off_diag] = fisher_z(r[off_diag])
        weight = n_timepoints - 3

        z_sum = z * weight if z_sum is None else z_sum + z * weight
        w_sum += weight

    if z_sum is None:
        return None

    fc = np.tanh(z_sum / w_sum)
    np.fill_diagonal(fc, 1.0)
    return fc


def build_all_subject_fc(config: FCConfig) -> None:
    subjects = find_subject_runs(config.roi_timeseries_root)
    print(f"{len(subjects)} subjects found under {config.roi_timeseries_root}")

    os.makedirs(config.fc_output_root, exist_ok=True)
    n_saved, n_skipped = 0, 0
    for subject_id, run_paths in subjects.items():
        fc = subject_fc(run_paths, config)
        if fc is None:
            n_skipped += 1
            continue
        out_dir = os.path.join(config.fc_output_root, subject_id)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "fc.npy"), fc)
        n_saved += 1

    print(f"saved {n_saved} subject FC matrices, "
          f"skipped {n_skipped} (no run >= {config.min_volumes} volumes)")


def parse_args() -> FCConfig:
    parser = argparse.ArgumentParser(description="Build subject-level FC matrices from ROI timeseries")
    parser.add_argument("--roi_timeseries_root", required=True,
                         help="Root dir written by create_roi_timeseries.py")
    parser.add_argument("--min_volumes", type=int, default=300,
                         help="Drop runs shorter than this many timepoints")
    parser.add_argument("--fc_dirname", default="subject_fc",
                         help="Output subdirectory name under roi_timeseries_root")
    args = parser.parse_args()
    return FCConfig(
        roi_timeseries_root=args.roi_timeseries_root,
        min_volumes=args.min_volumes,
        fc_dirname=args.fc_dirname,
    )


if __name__ == "__main__":
    build_all_subject_fc(parse_args())
