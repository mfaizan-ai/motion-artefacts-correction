"""
Dataset and DataLoader definitions for CycleGAN training, validation, and testing.
"""

import math
import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# Small constant to avoid division by zero in near-zero-mean voxels
_PSC_EPS = 1e-6


# Helper functions for loading and normalising NIfTI files, used by all datasets
def _load_nifti(path: str) -> Tensor:
    """
    Load a NIfTI file → float32 tensor.
    Disk shape  : (X, Y, Z, T)
    Output shape: (1, T, X, Y, Z)
    """
    img  = nib.load(path, mmap=True)
    data = np.asarray(img.dataobj, dtype=np.float32)   # (X, Y, Z, T)
    data = data.transpose(3, 0, 1, 2)                  # (T, X, Y, Z)
    return torch.from_numpy(data)        # (1, T, X, Y, Z)


def _psc_normalise(x: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Percent Signal Change normalisation.

    Args:
        x : raw BOLD tensor  (1, T, X, Y, Z)

    Returns:
        psc      : normalised tensor  (1, T, X, Y, Z)
                   values are fractional deviations from temporal mean
        mean_vol : voxel-wise temporal mean  (1, 1, X, Y, Z)
                   save this — needed to recover original scale at inference

    Denormalise with:
        x_original = (psc * (mean_vol + eps)) + mean_vol
    """
    # Temporal mean: average over time dimension (dim=1), keep dim for broadcast
    mean_vol = x.mean(dim=0, keepdim=True)                  # (1, 1, X, Y, Z)

    # Warn if any voxel has a near-zero mean (background / outside mask)
    psc = (x - mean_vol) / (mean_vol.abs() + _PSC_EPS)     # (1, T, X, Y, Z)
    return psc, mean_vol


def psc_denormalise(psc: Tensor, mean_vol: Tensor) -> Tensor:
    """
    Recover original BOLD intensity from PSC-normalised tensor.

    Args:
        psc      : model output in PSC space  (B, 1, T, X, Y, Z) or (1, T, X, Y, Z)
        mean_vol : temporal mean saved during normalisation  (same batch shape)

    Returns:
        Tensor in original BOLD intensity units.

    Usage in inference loop:
        corrected_psc  = generator(batch["A"])
        corrected_bold = psc_denormalise(corrected_psc, batch["mean_vol_A"])
    """
    return (psc * (mean_vol.abs() + _PSC_EPS)) + mean_vol


def _collect_files(directory: str) -> List[str]:
    """Sorted list of .nii / .nii.gz files in a directory."""
    p = Path(directory)
    files = sorted(str(f) for f in p.iterdir() if ".nii" in f.name)
    if not files:
        raise FileNotFoundError(f"No NIfTI files found in {directory}")
    return files



# Training dataset  (imbalance-aware, unpaired)
class TrainFMRIDataset(Dataset):
    """
    Unpaired CycleGAN training dataset. Per-epoch size = len(B).

    Batch dict keys
    ---------------
    "A"          : PSC-normalised corrupted chunk     (1, T, X, Y, Z)
    "B"          : PSC-normalised motion-free chunk   (1, T, X, Y, Z)
    "mean_vol_A" : temporal mean of raw A chunk       (1, 1, X, Y, Z)
    "mean_vol_B" : temporal mean of raw B chunk       (1, 1, X, Y, Z)
    "path_A"     : source path string
    "path_B"     : source path string

    mean_vol_A / mean_vol_B are not used during training but are included
    so the same loader can be used for validation with denormalisation.
    """

    def __init__(
        self,
        root_dir: str,
        augment: bool = False,
        cache_limit: int = 0,
    ):
        super().__init__()
        self.augment     = augment
        self.cache_limit = cache_limit

        self.files_A = _collect_files(os.path.join(root_dir, "A_corrupted"))
        self.files_B = _collect_files(os.path.join(root_dir, "B_motion_free"))

        self._len_A      = len(self.files_A)
        self._len_B      = len(self.files_B)
        self._epoch_size = self._len_B          # 1:1 balanced per epoch

        # Global A queue — full shuffle consumed front-to-back, refilled on empty
        self._a_queue: List[int]        = []
        self._a_epoch_indices: List[int] = []

        # B order for this epoch
        self._b_order: List[int] = list(range(self._len_B))

        # RAM cache: path → (psc, mean_vol)
        self._cache: Dict[str, Tuple[Tensor, Tensor]] = {}

        self.on_epoch_start()   # initialise first epoch

        print(
            f"[TrainDataset] A={self._len_A}  B={self._len_B}  "
            f"per-epoch={self._epoch_size}  "
            f"full-A-coverage every ~{math.ceil(self._len_A / self._len_B)} epochs"
        )

    # ------------------------------------------------------------------
    def _refill_a_queue(self) -> None:
        indices = list(range(self._len_A))
        random.shuffle(indices)
        self._a_queue = indices

    def on_epoch_start(self) -> None:
        """
        Resample A subsample and reshuffle B.
        Must be called at the start of every training epoch:

            for epoch in range(n_epochs):
                loaders["train"].dataset.on_epoch_start()
                for batch in loaders["train"]:
                    real_A       = batch["A"].to(device)
                    real_B       = batch["B"].to(device)
                    mean_vol_A   = batch["mean_vol_A"].to(device)
                    ...
        """
        if len(self._a_queue) < self._epoch_size:
            self._refill_a_queue()

        self._a_epoch_indices = self._a_queue[: self._epoch_size]
        self._a_queue         = self._a_queue[self._epoch_size :]

        self._b_order = list(range(self._len_B))
        random.shuffle(self._b_order)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._epoch_size

    def _load(self, path: str) -> Tuple[Tensor, Tensor]:
        """Return (psc, mean_vol), using cache if available."""
        if path in self._cache:
            return self._cache[path]
        raw = _load_nifti(path)
        psc, mean_vol = _psc_normalise(raw)
        if len(self._cache) < self.cache_limit:
            self._cache[path] = (psc, mean_vol)
        return psc, mean_vol

    def __getitem__(self, idx: int) -> Dict[str, object]:
        path_A = self.files_A[self._a_epoch_indices[idx]]
        path_B = self.files_B[self._b_order[idx]]

        psc_A, mean_A = self._load(path_A)
        psc_B, mean_B = self._load(path_B)

        if self.augment:
            psc_A, mean_A, psc_B, mean_B = self._random_flip(
                psc_A, mean_A, psc_B, mean_B
            )

        return {
            "A":          psc_A,    # (1, T, X, Y, Z)  PSC-normalised
            "B":          psc_B,    # (1, T, X, Y, Z)  PSC-normalised
            "mean_vol_A": mean_A,   # (1, 1, X, Y, Z)  for denormalisation
            "mean_vol_B": mean_B,   # (1, 1, X, Y, Z)  for denormalisation
            "path_A":     path_A,
            "path_B":     path_B,
        }

    @staticmethod
    def _random_flip(
        psc_A: Tensor, mean_A: Tensor,
        psc_B: Tensor, mean_B: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Independent L-R flip per domain; applied to both psc and mean_vol."""
        if random.random() > 0.5:
            psc_A  = torch.flip(psc_A,  dims=[2])
            mean_A = torch.flip(mean_A, dims=[2])
        if random.random() > 0.5:
            psc_B  = torch.flip(psc_B,  dims=[2])
            mean_B = torch.flip(mean_B, dims=[2])
        return psc_A, mean_A, psc_B, mean_B



# Val dataset  (full, balanced, unpaired)
class ValFMRIDataset(Dataset):
    """
    Validation dataset. Full dataset, both domains, no subsampling.
    B is shuffled once at init (fixed order across val epochs for consistency).

    Batch dict keys: same as TrainFMRIDataset — "A", "B", "mean_vol_A",
    "mean_vol_B", "path_A", "path_B".
    """

    def __init__(self, root_dir: str, cache_limit: int = 0):
        super().__init__()
        self.cache_limit = cache_limit

        self.files_A = _collect_files(os.path.join(root_dir, "A_corrupted"))
        self.files_B = _collect_files(os.path.join(root_dir, "B_motion_free"))

        self._b_order = list(range(len(self.files_B)))
        random.shuffle(self._b_order)

        self._cache: Dict[str, Tuple[Tensor, Tensor]] = {}

        print(
            f"[ValDataset]   A={len(self.files_A)}  B={len(self.files_B)}"
        )

    def __len__(self) -> int:
        return min(len(self.files_A), len(self.files_B))

    def _load(self, path: str) -> Tuple[Tensor, Tensor]:
        if path in self._cache:
            return self._cache[path]
        raw = _load_nifti(path)
        psc, mean_vol = _psc_normalise(raw)
        if len(self._cache) < self.cache_limit:
            self._cache[path] = (psc, mean_vol)
        return psc, mean_vol

    def __getitem__(self, idx: int) -> Dict[str, object]:
        path_A = self.files_A[idx]
        path_B = self.files_B[self._b_order[idx]]

        psc_A, mean_A = self._load(path_A)
        psc_B, mean_B = self._load(path_B)

        return {
            "A":          psc_A,
            "B":          psc_B,
            "mean_vol_A": mean_A,
            "mean_vol_B": mean_B,
            "path_A":     path_A,
            "path_B":     path_B,
        }


# Test dataset  (domain A only, full, sorted)
class TestFMRIDataset(Dataset):
    """
    Test / inference dataset. Domain A only, full set, sorted order.
    No augmentation, no shuffling.

    Batch dict keys
    ---------------
    "A"          : PSC-normalised corrupted chunk  (1, T, X, Y, Z)
    "mean_vol_A" : temporal mean                  (1, 1, X, Y, Z)
    "path_A"     : source path

    Inference pattern:
        for batch in loaders["test"]:
            corrected_psc  = G_AB(batch["A"].to(device))
            corrected_bold = psc_denormalise(
                corrected_psc, batch["mean_vol_A"].to(device)
            )
            # save corrected_bold as NIfTI
    """

    def __init__(self, root_dir: str):
        self.files_A = _collect_files(os.path.join(root_dir, "A_corrupted"))
        print(f"[TestDataset]  A={len(self.files_A)}")

    def __len__(self) -> int:
        return len(self.files_A)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        path = self.files_A[idx]
        raw  = _load_nifti(path)
        psc, mean_vol = _psc_normalise(raw)
        return {
            "A":          psc,
            "mean_vol_A": mean_vol,
            "path_A":     path,
        }



# Factory datasets and dataloaders for any subset of splits, with sensible defaults.
def build_dataloaders(
    dataset_root: str,
    splits: Optional[List[str]] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
    cache_limit: int = 0,
    augment_train: bool = True,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    prefetch_factor: int = 2,
) -> Dict[str, DataLoader]:
    """
    Build DataLoaders for any subset of train / val / test splits.

    Args:
        dataset_root   : path to cyclegans_dataset/
        splits         : e.g. ["train", "val"] or ["test"]. Default: all three.
        batch_size     : chunks per batch (1 recommended for 3-D fMRI volumes)
        num_workers    : parallel CPU workers (4–8 on HPC)
        pin_memory     : faster CPU→GPU transfer; set False on CPU-only nodes
        cache_limit    : max volumes cached in RAM per dataset (0 = off)
        augment_train  : random L-R flips on training data
        distributed    : True for multi-GPU with torch.distributed / SLURM
        world_size     : total number of DDP processes
        rank           : this process rank
        prefetch_factor: batches prefetched per worker

    Returns:
        Dict[str, DataLoader] for the requested splits.

    Examples:
        # Full pipeline
        loaders = build_dataloaders("/path/to/cyclegans_dataset")

        # Train + val only
        loaders = build_dataloaders(root, splits=["train", "val"])

        # Inference only
        loaders = build_dataloaders(root, splits=["test"],
                                    batch_size=1, num_workers=2)
    """
    if splits is None:
        splits = ["train", "val", "test"]

    _dataset_builders = {
        "train": lambda: TrainFMRIDataset(
            os.path.join(dataset_root, "train"),
            augment=augment_train,
            cache_limit=cache_limit,
        ),
        "val": lambda: ValFMRIDataset(
            os.path.join(dataset_root, "val"),
            cache_limit=cache_limit,
        ),
        "test": lambda: TestFMRIDataset(
            os.path.join(dataset_root, "test"),
        ),
    }

    loaders: Dict[str, DataLoader] = {}

    for split in splits:
        if split not in _dataset_builders:
            raise ValueError(
                f"Unknown split '{split}'. Choose from: train, val, test"
            )

        dataset  = _dataset_builders[split]()
        is_train = split == "train"

        if distributed and split != "test":
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=is_train,
                drop_last=is_train,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = is_train

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=is_train,
            persistent_workers=(num_workers > 0),
            prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        )

    return loaders



# sanity check
if __name__ == "__main__":
    import argparse
    from collections import Counter

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
                "faizan_motion_correction_dataset/cyclegans_dataset",
    )
    parser.add_argument("--splits",  nargs="+", default=["train", "val", "test"])
    parser.add_argument("--workers", type=int,  default=4)
    parser.add_argument("--batch",   type=int,  default=1)
    args = parser.parse_args()

    print(f"Root   : {args.root}")
    print(f"Splits : {args.splits}\n")

    loaders = build_dataloaders(
        dataset_root=args.root,
        splits=args.splits,
        batch_size=args.batch,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        augment_train=True,
    )

    for split, loader in loaders.items():
        batch = next(iter(loader))
        a     = batch["A"]
        mv_a  = batch["mean_vol_A"]

        print(
            f"[{split}]  batches/epoch={len(loader)} | "
            f"A: {tuple(a.shape)}  psc_range=[{a.min():.3f}, {a.max():.3f}] | "
            f"mean_vol_A: {tuple(mv_a.shape)}  range=[{mv_a.min():.1f}, {mv_a.max():.1f}]"
        )
        if "B" in batch:
            print(f"         B: {tuple(batch['B'].shape)}")

        # Verify round-trip: denormalise should recover original signal
        raw_recovered = psc_denormalise(a, mv_a)
        # Re-load raw to compare
        raw_orig = _load_nifti(batch["path_A"][0] if isinstance(batch["path_A"], list)
                               else batch["path_A"])
        max_err = (raw_recovered - raw_orig).abs().max().item()
        print(f"         Denormalise round-trip max error: {max_err:.6f}  "
              f"({'✓ PASS' if max_err < 1e-3 else '✗ FAIL'})")

    # A-queue coverage check
    if "train" in loaders:
        print("\n[Coverage] Simulating 10 epochs...")
        ds   = loaders["train"].dataset
        seen: Counter = Counter()
        for _ in range(10):
            ds.on_epoch_start()
            seen.update(ds._a_epoch_indices)
        vals = list(seen.values())
        print(f"  Unique A chunks seen: {len(seen)} / {ds._len_A}")
        print(f"  Appearances per chunk: min={min(vals)}  max={max(vals)}  "
              f"mean={sum(vals)/len(vals):.1f}")