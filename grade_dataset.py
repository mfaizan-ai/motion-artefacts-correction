#!/usr/bin/env python3
"""
grade_dataset.py
=================
Unpaired Grade-1 (motion-free reference) vs. pooled Grade-2-6 (motion-corrupted)
dataset over motion_grades_chunk_5_dataset_hfiltered.

Ported from pytorch-CycleGAN-and-pix2pix/fMRI_dataset/fmri_chunk_dataset.py
(FMRIUnpairedGradeDataset) -- same grading/pooling/sampling logic, adapted for
this project's tensor convention:
    original: (C=1, T=5, H=60, W=72, D=56)
    here:     (T=5, H=64, W=72, D=56) -- channel dim dropped, axis 1 (H) zero-
              padded 60->64 so the 3-stage stride-2 downsampling in
              models/st_model.py divides evenly (see
              experimental_notebooks/scheafar_overlay_age_group_over_cropped_
              brain_mask_high_pass_filtered_data.ipynb for the padding
              derivation -- same crop+pad this project's SchaeferAtlasCropped
              uses, so a chunk tensor here and atlas_fc.SchaeferAtlasCropped's
              output are on the identical voxel grid).

Domain A = grades_b pooled (motion-corrupted), defaults to ALL of Grade 2-6 (any
motion severity, deliberately pooled -- not split by severity). Domain B =
Grade 1 (very low motion, the clean reference). Matches this project's
A=corrupted/B=motion-free convention everywhere else (dataset.py's
A_corrupted/B_motion_free, models/model.py's G_B: A->B corrupted->clean,
D_B: MotionFreeDisc judging domain B) -- train.py's `x_a = batch["A"]  #
corrupted` / `x_b = batch["B"]  # motion-free` is correct as literally
written for this dataset too, not just the flat one. Unpaired: A and B
chunks are never assumed to correspond to the same subject/run/timepoint.
"""
import csv
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

GRADE_A = "Grade 1"
GRADES_B_ALL = ("Grade 2", "Grade 3", "Grade 4", "Grade 5", "Grade 6")
SPLITS = ("train", "val", "test")

PADDED_H = 64  # matches atlas_fc.PADDED_SPATIAL[0]

DEFAULT_CHUNK_METADATA_CSV = (
    "/lustre/disk/home/users/mfaizan/motion_correction/cycleGANS_on_2d_images/"
    "pytorch-CycleGAN-and-pix2pix/data_preprocessing/"
    "motion_grades_chunk_5_dataset_hfiltered/chunk_metadata.csv"
)
DEFAULT_RUN_STATS_CSV = (
    "/lustre/disk/home/users/mfaizan/motion_correction/prototyping/"
    "motion-artefacts-correction/run_normalization_stats_hfiltered.csv"
)


def _load_run_stats(run_stats_csv):
    """(subject_id, session_id, run_id, task) -> (median, scale)."""
    stats = {}
    with open(run_stats_csv) as f:
        for row in csv.DictReader(f):
            key = (row["subject_id"], row["session_id"], row["run_id"], row["task"])
            stats[key] = (float(row["median"]), float(row["scale"]))
    return stats


def _load_chunk_rows(chunk_metadata_csv, split, task=None):
    """All chunk_metadata.csv rows for this split (optionally one task), grouped by grade."""
    rows_by_grade = {}
    with open(chunk_metadata_csv) as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            if task is not None and row["task"] != task:
                continue
            rows_by_grade.setdefault(row["grade"], []).append(row)
    return rows_by_grade


def _run_key(row):
    return (row["subject_id"], row["session_id"], row["run_id"], row["task"])


def _pad_h(tensor, target=PADDED_H):
    """tensor: (T, H, W, D) with H=60 -> (T, target, W, D), zero-padded symmetrically on H."""
    H = tensor.shape[1]
    total = target - H
    lo, hi = total // 2, total - total // 2
    return torch.nn.functional.pad(tensor, (0, 0, 0, 0, lo, hi), mode="constant", value=0.0)


class FMRIUnpairedGradeDataset(Dataset):
    """
    Unpaired (grades_b pooled, domain A) <-> (Grade 1, domain B) dataset for one split.

    See module docstring for the tensor-shape adaptation from the original
    FMRIUnpairedGradeDataset. Sampling/pooling logic is otherwise identical:
      - epoch length = min(A_size, B_size); the SMALLER domain is used in
        full every epoch (relies on DataLoader(shuffle=True)); the LARGER
        domain gets a fresh non-repeating permutation each epoch via
        set_epoch() -- must be called manually before the DataLoader
        iterator is created (see set_epoch()'s docstring for why).
      - full_coverage=True (recommended for val/test): deterministic
        index % size cycling through ALL of both domains every pass.

    Returns {"A", "B", "A_paths", "B_paths", "A_meta", "B_meta"} --
    A_meta/B_meta carry subject_id/age_group directly (used for per-sample
    Schaefer atlas selection in train.py, no filename parsing needed).
    """

    def __init__(self, split, chunk_metadata_csv=DEFAULT_CHUNK_METADATA_CSV,
                 run_stats_csv=DEFAULT_RUN_STATS_CSV, task=None, flip_prob=0.0,
                 base_seed=0, full_coverage=False, grades_b=GRADES_B_ALL):
        super().__init__()
        assert split in SPLITS, f"split must be one of {SPLITS}, got {split!r}"
        assert 0.0 <= flip_prob <= 1.0
        grades_b = tuple(grades_b)
        assert grades_b, "grades_b must be non-empty"
        invalid = set(grades_b) - set(GRADES_B_ALL)
        assert not invalid, f"grades_b contains invalid grade(s) {invalid}"
        self.split = split
        self.flip_prob = flip_prob
        self.base_seed = base_seed
        self.full_coverage = full_coverage
        self.grades_b = grades_b

        self.run_stats = _load_run_stats(run_stats_csv)

        rows_by_grade = _load_chunk_rows(chunk_metadata_csv, split, task=task)
        # A = corrupted (pooled grades_b), B = clean (Grade 1) -- see module docstring for why
        # this order, not GRADE_A/grades_b's own name order.
        self.A_rows = [row for g in self.grades_b for row in rows_by_grade.get(g, [])]
        self.B_rows = rows_by_grade.get(GRADE_A, [])
        self.A_size = len(self.A_rows)
        self.B_size = len(self.B_rows)
        if self.A_size == 0 or self.B_size == 0:
            raise ValueError(
                f"split={split!r} task={task!r} grades_b={self.grades_b}: "
                f"A_size={self.A_size}, B_size={self.B_size} -- both must be > 0"
            )

        missing = {_run_key(r) for r in self.A_rows + self.B_rows if _run_key(r) not in self.run_stats}
        if missing:
            raise ValueError(
                f"{len(missing)} run(s) referenced by chunks in split={split!r} have no entry "
                f"in {run_stats_csv}. Example missing key: {next(iter(missing))}"
            )

        self._b_is_larger = self.B_size >= self.A_size
        self._epoch_len = max(self.A_size, self.B_size) if full_coverage else min(self.A_size, self.B_size)
        self._larger_size = self.B_size if self._b_is_larger else self.A_size
        self._epoch = None
        self.set_epoch(0)

    def __len__(self):
        return self._epoch_len

    def set_epoch(self, epoch):
        """Call once per epoch, before creating that epoch's DataLoader iterator (not inside the
        loop body after iteration has started) -- see FMRIUnpairedGradeDataset's original
        docstring (fMRI_dataset/fmri_chunk_dataset.py) for the full num_workers>0 rationale."""
        self._epoch = epoch
        if self.full_coverage:
            return
        g = torch.Generator().manual_seed(self.base_seed + epoch)
        self._larger_epoch_indices = torch.randperm(self._larger_size, generator=g)[: self._epoch_len].tolist()

    def _load_chunk_tensor(self, row):
        """Load one chunk NIfTI, normalize with its source run's (median, scale), pad H, flip.
        Returns a (T, PADDED_H, W, D) float32 tensor."""
        median, scale = self.run_stats[_run_key(row)]

        img = nib.load(row["chunk_path"])
        data = np.asarray(img.dataobj, dtype=np.float32)  # (H, W, D, T) = (60, 72, 56, 5)
        mask = data[..., 0] != 0

        normalized = np.zeros_like(data, dtype=np.float32)
        normalized[mask] = (data[mask] - median) / scale

        tensor = torch.from_numpy(normalized).permute(3, 0, 1, 2)  # (T, H, W, D)
        tensor = _pad_h(tensor)  # (T, PADDED_H, W, D)

        if self.flip_prob > 0.0 and random.random() < self.flip_prob:
            tensor = torch.flip(tensor, dims=[1])  # dim 1 = H = left-right axis

        return tensor, median, scale

    def _meta(self, row, median, scale):
        return dict(
            subject_id=row["subject_id"], age_group=row["age_group"],
            session_id=row["session_id"], run_id=row["run_id"], task=row["task"],
            grade=row["grade"], chunk_start=int(row["chunk_start"]), chunk_end=int(row["chunk_end"]),
            chunk_mean_fd=float(row["chunk_mean_fd"]), chunk_max_fd=float(row["chunk_max_fd"]),
            source_volume_path=row["source_volume_path"], median=median, scale=scale,
        )

    def __getitem__(self, index):
        if self.full_coverage:
            A_row = self.A_rows[index % self.A_size]
            B_row = self.B_rows[index % self.B_size]
        elif self._b_is_larger:
            A_row = self.A_rows[index]
            B_row = self.B_rows[self._larger_epoch_indices[index]]
        else:
            B_row = self.B_rows[index]
            A_row = self.A_rows[self._larger_epoch_indices[index]]

        A_tensor, A_median, A_scale = self._load_chunk_tensor(A_row)
        B_tensor, B_median, B_scale = self._load_chunk_tensor(B_row)

        return {
            "A": A_tensor,
            "B": B_tensor,
            "A_paths": A_row["chunk_path"],
            "B_paths": B_row["chunk_path"],
            "A_meta": self._meta(A_row, A_median, A_scale),
            "B_meta": self._meta(B_row, B_median, B_scale),
        }


def denormalize_values(values: Tensor, median: Tensor, scale: Tensor) -> Tensor:
    """Inverse of `(x - median) / scale` from `_load_chunk_tensor`. Exact, since the
    normalization is unclipped."""
    return values * scale + median


def denormalize_chunk(chunk: Tensor, median: Tensor, scale: Tensor, mask: Tensor) -> Tensor:
    """
    Denormalize a batch of chunks back to raw BOLD intensity, per-sample median/scale.

    Args:
        chunk  : (B, T, H, W, D) normalized tensor
        median : (B,) per-sample median used at normalization time (e.g. batch["A_meta"]["median"])
        scale  : (B,) per-sample scale used at normalization time (e.g. batch["A_meta"]["scale"])
        mask   : (B, T, H, W, D) bool, True where `chunk` is real brain signal. Must be the
                 GROUND-TRUTH mask (e.g. `x_a != 0`) -- not derived from `chunk` itself. A
                 generator has no constraint forcing background to stay exactly 0, so an
                 untrained (or imperfectly trained) model's output is nonzero almost
                 everywhere; deriving the mask from such a `chunk` would treat that background
                 noise as real tissue and denormalize it to a full-magnitude fake signal.

    Returns:
        (B, T, H, W, D) tensor in raw BOLD units. Voxels outside `mask` are left at 0 rather
        than denormalized -- denormalizing them would turn small (or even zero) values into
        `median`, inventing signal where there is none.
    """
    B = chunk.shape[0]
    median = median.to(chunk.dtype).view(B, 1, 1, 1, 1).expand_as(chunk)
    scale  = scale.to(chunk.dtype).view(B, 1, 1, 1, 1).expand_as(chunk)

    out = torch.zeros_like(chunk)
    out[mask] = denormalize_values(chunk[mask], median[mask], scale[mask])
    return out


def worker_init_fn(worker_id):
    """Pass to DataLoader(..., worker_init_fn=worker_init_fn) when num_workers > 0 -- without
    this every worker inherits the same RNG state from the fork."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
