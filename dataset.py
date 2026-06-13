"""
Dataset and DataLoader definitions for CycleGAN training, validation, and testing.
"""

import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# Target spatial shape the model expects: (X, Y, Z)
TARGET_SPATIAL: Tuple[int, int, int] = (80, 96, 72)

# Small constant to avoid division by zero in near-zero-mean voxels
_PSC_EPS = 1e-6


def _spatial_resample(x: Tensor, target: Tuple[int, int, int]) -> Tensor:
    """
    Trilinear resample a (T, X, Y, Z) tensor to (T, *target).

    T is the temporal/channel dimension and is left untouched.
    A dummy batch dim of 1 is added so F.interpolate sees
    (1, T, X, Y, Z) → (1, T, tX, tY, tZ), where T acts as channels
    and X, Y, Z are the spatial dims being resampled.

    Args:
        x      : raw BOLD tensor  (T, X, Y, Z)
        target : desired spatial size (tX, tY, tZ)

    Returns:
        Resampled tensor  (T, tX, tY, tZ)
    """
    x = x.unsqueeze(0)                          # (1, T, X, Y, Z)  dummy batch
    x = F.interpolate(
        x,
        size=target,
        mode="trilinear",
        align_corners=False,
    )
    return x.squeeze(0)                          # (T, tX, tY, tZ)


def _load_nifti(path: str) -> Tuple[Tensor, Tuple[int, int, int]]:
    """
    Load a NIfTI file, spatially resample to TARGET_SPATIAL.

    Disk shape        : (X, Y, Z, T)
    After transpose   : (T, X, Y, Z)
    After resample    : (T, *TARGET_SPATIAL)  i.e. (T, 80, 96, 72)

    Returns
    -------
    data       : float32 tensor  (T, 80, 96, 72)
    orig_shape : original spatial dims  (X, Y, Z)  — save for upsampling at inference
    """
    img  = nib.load(path, mmap=True)
    data = np.asarray(img.dataobj, dtype=np.float32)   # (X, Y, Z, T)
    data = data.transpose(3, 0, 1, 2)                  # (T, X, Y, Z)

    orig_shape: Tuple[int, int, int] = (data.shape[1], data.shape[2], data.shape[3])

    tensor = torch.from_numpy(data)                    # (T, X, Y, Z)

    if orig_shape != TARGET_SPATIAL:
        tensor = _spatial_resample(tensor, TARGET_SPATIAL)

    return tensor, orig_shape                          # (T, 80, 96, 72), (X, Y, Z)


def _psc_normalise(x: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Percent Signal Change normalisation with brain mask.

    Data is assumed to be brain-extracted: voxels outside the brain are
    exactly zero, voxels inside have positive BOLD signal.  A binary brain
    mask is derived from the temporal mean (mean > 0) and applied to the
    PSC tensor so background voxels never produce blow-up values from
    near-zero division.

    Args:
        x : raw BOLD tensor  (T, X, Y, Z)  — already at TARGET_SPATIAL,
            brain-extracted (background = 0)

    Returns:
        psc      : normalised tensor  (T, X, Y, Z)
                   brain voxels : fractional deviation from temporal mean
                   background   : exactly 0
        mean_vol : voxel-wise temporal mean  (1, X, Y, Z)  unmasked
                   save this — needed to recover original scale at inference

    Denormalise with:
        x_original = (psc * (mean_vol.abs() + eps)) + mean_vol
        background voxels recover to ~0 since mean_vol is 0 there.
    """
    mean_vol   = x.mean(dim=0, keepdim=True)                      # (1, X, Y, Z)
    # Relative threshold: 1% of the maximum mean signal in this chunk.
    # True brain voxels have mean in the hundreds; trilinear interpolation
    # at the brain boundary can produce tiny non-zero artefacts (e.g. 0.001).
    # A fixed threshold (e.g. > 0 or > 1.0) is fragile across scanner scalings;
    # 1% of max scales with the actual data range and cleanly separates
    # real brain signal from boundary artefacts regardless of intensity units.
    brain_mask = mean_vol > (mean_vol.max() * 0.01)               # (1, X, Y, Z) bool
    psc        = (x - mean_vol) / (mean_vol.abs() + _PSC_EPS)     # (T, X, Y, Z)
    psc        = psc * brain_mask                                  # zero background
    return psc, mean_vol


def psc_denormalise(psc: Tensor, mean_vol: Tensor) -> Tensor:
    """
    Recover BOLD intensity from PSC-normalised tensor.

    Args:
        psc      : model output in PSC space
                   (B, T, X, Y, Z)  or  (T, X, Y, Z)
        mean_vol : temporal mean saved during normalisation
                   must broadcast against psc

    Returns:
        Tensor in original BOLD intensity units  (same shape as psc).

    Usage in inference loop:
        corrected_psc  = G_AB(batch["A"].to(device))
        corrected_bold = psc_denormalise(corrected_psc, batch["mean_vol_A"].to(device))
        # optionally upsample corrected_bold back to orig_shape_A
    """
    return (psc * (mean_vol.abs() + _PSC_EPS)) + mean_vol


def upsample_to_original(
    x: Tensor,
    orig_shape: Tuple[int, int, int],
) -> Tensor:
    """
    Upsample corrected BOLD back to source resolution after inference.

    F.interpolate requires (B, C, D, H, W) i.e. a batch dim must be present.
    Accepts both batched (B, T, X, Y, Z) and unbatched (T, X, Y, Z) input.
    T is the temporal dimension treated as channels by the model.

    Args:
        x          : corrected BOLD tensor  (B, T, X, Y, Z) or (T, X, Y, Z)
                     at TARGET_SPATIAL resolution, in original BOLD units
                     (i.e. after psc_denormalise)
        orig_shape : (X_orig, Y_orig, Z_orig) from batch["orig_shape_A"]

    Returns:
        Tensor  (B, T, X_orig, Y_orig, Z_orig) or (T, X_orig, Y_orig, Z_orig)
        matching the input rank.

    Example (inference):
        corrected_bold = psc_denormalise(corrected_psc, mean_vol_A)
        corrected_bold = upsample_to_original(corrected_bold, orig_shape)
        # save corrected_bold as NIfTI
    """
    unbatched = x.dim() == 4              # (T, X, Y, Z) — add dummy batch dim
    if unbatched:
        x = x.unsqueeze(0)               # (1, T, X, Y, Z)

    x = F.interpolate(
        x,
        size=orig_shape,
        mode="trilinear",
        align_corners=False,
    )                                     # (B, T, X_orig, Y_orig, Z_orig)

    if unbatched:
        x = x.squeeze(0)                 # (T, X_orig, Y_orig, Z_orig)
    return x


def _collect_files(directory: str) -> List[str]:
    """Sorted list of .nii / .nii.gz files in a directory."""
    p = Path(directory)
    files = sorted(str(f) for f in p.iterdir() if ".nii" in f.name)
    if not files:
        raise FileNotFoundError(f"No NIfTI files found in {directory}")
    return files


# Train loader 
class TrainFMRIDataset(Dataset):
    """
    Unpaired CycleGAN training dataset.  Per-epoch size = len(B).

    Batch dict keys
    ---------------
    "A"            : PSC-normalised corrupted chunk    (T, 80, 96, 72)
    "B"            : PSC-normalised motion-free chunk  (T, 80, 96, 72)
    "mean_vol_A"   : temporal mean of A at model res   (1, 80, 96, 72)
    "mean_vol_B"   : temporal mean of B at model res   (1, 80, 96, 72)
    "orig_shape_A" : original spatial dims of A        (X, Y, Z)  tuple
    "orig_shape_B" : original spatial dims of B        (X, Y, Z)  tuple
    "path_A"       : source path string
    "path_B"       : source path string

    orig_shape_* and mean_vol_* are not used during training losses but are
    carried so the same loader works for validation with full denormalisation
    and upsampling back to source resolution.
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
        self._epoch_size = self._len_B

        self._a_queue: List[int]         = []
        self._a_epoch_indices: List[int] = []
        self._b_order: List[int]         = list(range(self._len_B))

        self._cache: Dict[str, Tuple[Tensor, Tensor, Tuple[int, int, int]]] = {}

        self.on_epoch_start()

        print(
            f"[TrainDataset] A={self._len_A}  B={self._len_B}  "
            f"per-epoch={self._epoch_size}  "
            f"full-A-coverage every ~{math.ceil(self._len_A / self._len_B)} epochs  "
            f"target-spatial={TARGET_SPATIAL}"
        )

    def _refill_a_queue(self) -> None:
        indices = list(range(self._len_A))
        random.shuffle(indices)
        self._a_queue = indices

    def on_epoch_start(self) -> None:
        """Advance A queue and reshuffle B. Call at the start of every epoch."""
        if len(self._a_queue) < self._epoch_size:
            self._refill_a_queue()

        self._a_epoch_indices = self._a_queue[: self._epoch_size]
        self._a_queue         = self._a_queue[self._epoch_size :]

        self._b_order = list(range(self._len_B))
        random.shuffle(self._b_order)

    def __len__(self) -> int:
        return self._epoch_size

    def _load(
        self, path: str
    ) -> Tuple[Tensor, Tensor, Tuple[int, int, int]]:
        """
        Return (psc, mean_vol, orig_shape), using RAM cache if available.

        Cache stores post-resample, post-PSC tensors so the spatial
        interpolation is only paid once per unique file.
        """
        if path in self._cache:
            return self._cache[path]

        raw, orig_shape = _load_nifti(path)           # (T, 80, 96, 72), (X,Y,Z)
        psc, mean_vol   = _psc_normalise(raw)         # both (T/1, 80, 96, 72)

        if len(self._cache) < self.cache_limit:
            self._cache[path] = (psc, mean_vol, orig_shape)

        return psc, mean_vol, orig_shape

    def __getitem__(self, idx: int) -> Dict[str, object]:
        path_A = self.files_A[self._a_epoch_indices[idx]]
        path_B = self.files_B[self._b_order[idx]]

        psc_A, mean_A, orig_A = self._load(path_A)
        psc_B, mean_B, orig_B = self._load(path_B)

        if self.augment:
            psc_A, mean_A, psc_B, mean_B = self._random_flip(
                psc_A, mean_A, psc_B, mean_B
            )

        return {
            "A":            psc_A,    # (T, 80, 96, 72)  PSC
            "B":            psc_B,    # (T, 80, 96, 72)  PSC
            "mean_vol_A":   mean_A,   # (1, 80, 96, 72)  for denormalisation
            "mean_vol_B":   mean_B,   # (1, 80, 96, 72)  for denormalisation
            "orig_shape_A": orig_A,   # (X, Y, Z)  for upsampling at inference
            "orig_shape_B": orig_B,
            "path_A":       path_A,
            "path_B":       path_B,
        }

    @staticmethod
    def _random_flip(
        psc_A: Tensor, mean_A: Tensor,
        psc_B: Tensor, mean_B: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Independent L-R flip per domain; applied to both psc and mean_vol."""
        if random.random() > 0.5:
            psc_A  = torch.flip(psc_A,  dims=[1])   # flip X dim (T, X, Y, Z)
            mean_A = torch.flip(mean_A, dims=[1])   # flip X dim (1, X, Y, Z)
        if random.random() > 0.5:
            psc_B  = torch.flip(psc_B,  dims=[1])
            mean_B = torch.flip(mean_B, dims=[1])
        return psc_A, mean_A, psc_B, mean_B



# Val dataset 
class ValFMRIDataset(Dataset):
    """
    Validation dataset. Full dataset, both domains, no subsampling.
    B is shuffled once at init (fixed order across val epochs).

    Batch dict keys: same as TrainFMRIDataset.
    """

    def __init__(self, root_dir: str, cache_limit: int = 0):
        super().__init__()
        self.cache_limit = cache_limit

        self.files_A = _collect_files(os.path.join(root_dir, "A_corrupted"))
        self.files_B = _collect_files(os.path.join(root_dir, "B_motion_free"))

        self._b_order = list(range(len(self.files_B)))
        random.shuffle(self._b_order)

        self._cache: Dict[str, Tuple[Tensor, Tensor, Tuple[int, int, int]]] = {}

        print(
            f"[ValDataset]   A={len(self.files_A)}  B={len(self.files_B)}  "
            f"target-spatial={TARGET_SPATIAL}"
        )

    def __len__(self) -> int:
        return min(len(self.files_A), len(self.files_B))

    def _load(
        self, path: str
    ) -> Tuple[Tensor, Tensor, Tuple[int, int, int]]:
        if path in self._cache:
            return self._cache[path]
        raw, orig_shape = _load_nifti(path)
        psc, mean_vol   = _psc_normalise(raw)
        if len(self._cache) < self.cache_limit:
            self._cache[path] = (psc, mean_vol, orig_shape)
        return psc, mean_vol, orig_shape

    def __getitem__(self, idx: int) -> Dict[str, object]:
        path_A = self.files_A[idx]
        path_B = self.files_B[self._b_order[idx]]

        psc_A, mean_A, orig_A = self._load(path_A)
        psc_B, mean_B, orig_B = self._load(path_B)

        return {
            "A":            psc_A,
            "B":            psc_B,
            "mean_vol_A":   mean_A,
            "mean_vol_B":   mean_B,
            "orig_shape_A": orig_A,
            "orig_shape_B": orig_B,
            "path_A":       path_A,
            "path_B":       path_B,
        }



# Test dataset  
class TestFMRIDataset(Dataset):
    """
    Test / inference dataset. Domain A only, full set, sorted order.
    No augmentation, no shuffling.

    Batch dict keys
    ---------------
    "A"            : PSC-normalised corrupted chunk  (T, 80, 96, 72)
    "mean_vol_A"   : temporal mean                  (1, 80, 96, 72)
    "orig_shape_A" : original spatial dims           (X, Y, Z)  tuple
    "path_A"       : source path string

    Inference pattern
    -----------------
        for batch in loaders["test"]:
            corrected_psc  = G_AB(batch["A"].to(device))
            corrected_bold = psc_denormalise(
                corrected_psc, batch["mean_vol_A"].to(device)
            )
            # optionally restore source resolution:
            corrected_bold = upsample_to_original(
                corrected_bold, batch["orig_shape_A"][0]   # unpack from batch tuple
            )
            # save corrected_bold as NIfTI
    """

    def __init__(self, root_dir: str):
        self.files_A = _collect_files(os.path.join(root_dir, "A_corrupted"))
        print(
            f"[TestDataset]  A={len(self.files_A)}  "
            f"target-spatial={TARGET_SPATIAL}"
        )

    def __len__(self) -> int:
        return len(self.files_A)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        path = self.files_A[idx]
        raw, orig_shape = _load_nifti(path)
        psc, mean_vol   = _psc_normalise(raw)
        return {
            "A":            psc,
            "mean_vol_A":   mean_vol,
            "orig_shape_A": orig_shape,
            "path_A":       path,
        }



# DataLoader factory
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


if __name__ == "__main__":
    import argparse
    from collections import Counter

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
                "faizan_motion_correction_dataset/cyclegans_dataset",
    )
    parser.add_argument("--splits",   nargs="+", default=["train", "val", "test"])
    parser.add_argument("--workers",  type=int,  default=4)
    parser.add_argument("--batch",    type=int,  default=1)
    parser.add_argument("--psc_batches", type=int, default=50,
                        help="Batches to sample per domain for PSC range analysis")
    args = parser.parse_args()

    print(f"Root          : {args.root}")
    print(f"Splits        : {args.splits}")
    print(f"Target spatial: {TARGET_SPATIAL}\n")

    loaders = build_dataloaders(
        dataset_root=args.root,
        splits=args.splits,
        batch_size=args.batch,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        augment_train=True,
    )
    
    train_loader = loaders.get("train", None)
    batch = next(iter(train_loader))
    psc_A = batch["A"]
    psc_B = batch["B"]
    print(f'psc_A shape: {psc_A.shape}, psc_B shape: {psc_B.shape}')
    