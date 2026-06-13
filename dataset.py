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

    Epoch loop example
    ------------------
        for epoch in range(n_epochs):
            loaders["train"].dataset.on_epoch_start()
            for batch in loaders["train"]:
                real_A     = batch["A"].to(device)          # (B, T, 80, 96, 72)
                real_B     = batch["B"].to(device)
                mean_vol_A = batch["mean_vol_A"].to(device) # (B, 1, 80, 96, 72)
                ...
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



# Sanity check
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


    # Per-split shape / round-trip / upsample checks
    for split, loader in loaders.items():
        batch = next(iter(loader))
        a     = batch["A"]
        mv_a  = batch["mean_vol_A"]
        orig  = batch["orig_shape_A"]

        print(
            f"[{split}]  batches/epoch={len(loader)} | "
            f"A: {tuple(a.shape)}  psc_range=[{a.min():.3f}, {a.max():.3f}] | "
            f"mean_vol_A: {tuple(mv_a.shape)}  "
            f"range=[{mv_a.min():.1f}, {mv_a.max():.1f}] | "
            f"orig_shape_A: {orig}"
        )
        if "B" in batch:
            b = batch["B"]
            print(
                f"         B: {tuple(b.shape)}  "
                f"psc_range=[{b.min():.3f}, {b.max():.3f}]"
            )

        # PSC round-trip — compare only brain voxels since masking
        # intentionally zeroes background, making denormalisation inexact there
        raw_ref, _ = _load_nifti(
            batch["path_A"][0] if isinstance(batch["path_A"], (list, tuple))
            else batch["path_A"]
        )
        psc_ref, mean_ref = _psc_normalise(raw_ref)
        raw_recovered     = psc_denormalise(psc_ref, mean_ref)
        brain_mask_ref    = mean_ref > (mean_ref.max() * 0.01)   # (1, X, Y, Z)
        max_err = ((raw_recovered - raw_ref) * brain_mask_ref).abs().max().item()
        print(
            f"         PSC round-trip max error : {max_err:.6f}  "
            f"({'✓ PASS' if max_err < 1e-3 else '✗ FAIL'})"
        )

        # Upsample shape check
        if isinstance(orig[0], torch.Tensor):
            orig_tuple = tuple(int(o[0].item()) for o in orig)
        else:
            orig_tuple = tuple(int(o) for o in orig)
        upsampled = upsample_to_original(raw_ref, orig_tuple)
        print(
            f"         Upsampled shape           : {tuple(upsampled.shape)}  "
            f"(expected T={a.shape[1]}, spatial={orig_tuple})"
        )

    # -----------------------------------------------------------------------
    # PSC range analysis across N batches per domain per split
    #
    # What we expect to see for brain-extracted data:
    #
    #   A (corrupted)
    #     - PSC range wider, positive tail >> negative tail
    #     - Motion spikes push signal upward → asymmetry > 1.0
    #     - Typical corrupted BOLD: range roughly [-1, +3] or wider
    #
    #   B (motion-free)
    #     - PSC range narrow and roughly symmetric
    #     - Typical clean BOLD fluctuations: ~±5% (PSC ±0.05)
    #     - No large positive outliers
    #
    #   Cross-domain
    #     - A range > B range
    #     - A std   > B std
    #     - B asymmetry close to 1.0
    #
    #   Brain mask (derived from mean_vol > 0)
    #     - Mask fraction should be consistent across A and B in same split
    #     - Expect ~20-60% of voxels to be brain for infant fMRI
    #     - Large discrepancy between A and B mask fractions = domain mismatch
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PSC RANGE ANALYSIS  (brain-extracted data)")
    print(f"Sampling {args.psc_batches} batches per domain per split")
    print("=" * 70)

    for split, loader in loaders.items():
        a_mins, a_maxs, a_stds, a_mask_fracs = [], [], [], []
        b_mins, b_maxs, b_stds, b_mask_fracs = [], [], [], []

        for i, batch in enumerate(loader):
            if i >= args.psc_batches:
                break

            a    = batch["A"]
            mv_a = batch["mean_vol_A"]

            a_mins.append(a.min().item())
            a_maxs.append(a.max().item())
            a_stds.append(a.std().item())
            # brain mask fraction: voxels with mean > 0
            a_mask_fracs.append((mv_a > 0).float().mean().item())

            if "B" in batch:
                b    = batch["B"]
                mv_b = batch["mean_vol_B"]
                b_mins.append(b.min().item())
                b_maxs.append(b.max().item())
                b_stds.append(b.std().item())
                b_mask_fracs.append((mv_b > 0).float().mean().item())

        # --- Domain A ---
        a_min     = min(a_mins)
        a_max     = max(a_maxs)
        a_std     = sum(a_stds) / len(a_stds)
        a_asym    = abs(a_max) / (abs(a_min) + 1e-9)
        a_mask_pc = 100 * sum(a_mask_fracs) / len(a_mask_fracs)

        print(f"\n  [{split}]  A (corrupted)")
        print(f"    PSC range      : [{a_min:.4f},  {a_max:.4f}]")
        print(f"    Mean std       : {a_std:.4f}")
        print(f"    Asymmetry      : {a_asym:.3f}  "
              f"({'✓ right-skewed as expected' if a_asym > 1.1 else '⚠ unexpectedly symmetric — check domain labels'})")
        print(f"    Brain coverage : {a_mask_pc:.1f}%  "
              f"({'✓ plausible' if 10 < a_mask_pc < 70 else '⚠ unexpected — check brain extraction'})")

        # PSC magnitude check: after brain masking, range should be bounded
        a_bounded = abs(a_min) < 5.0 and abs(a_max) < 10.0
        print(f"    PSC bounded    : "
              f"{'✓ PASS  (no background blowup)' if a_bounded else '⚠ FAIL  (possible unmasked background voxels)'}")

        if b_mins:
            b_min     = min(b_mins)
            b_max     = max(b_maxs)
            b_std     = sum(b_stds) / len(b_stds)
            b_asym    = abs(b_max) / (abs(b_min) + 1e-9)
            b_mask_pc = 100 * sum(b_mask_fracs) / len(b_mask_fracs)

            print(f"\n  [{split}]  B (motion-free)")
            print(f"    PSC range      : [{b_min:.4f},  {b_max:.4f}]")
            print(f"    Mean std       : {b_std:.4f}")
            print(f"    Asymmetry      : {b_asym:.3f}  "
                  f"({'✓ roughly symmetric' if b_asym < 1.8 else '⚠ unexpected skew — check domain labels'})")
            print(f"    Brain coverage : {b_mask_pc:.1f}%  "
                  f"({'✓ plausible' if 10 < b_mask_pc < 70 else '⚠ unexpected — check brain extraction'})")

            b_bounded = abs(b_min) < 5.0 and abs(b_max) < 5.0
            print(f"    PSC bounded    : "
                  f"{'✓ PASS  (no background blowup)' if b_bounded else '⚠ FAIL  (possible unmasked background voxels)'}")

            # Cross-domain
            range_A    = a_max - a_min
            range_B    = b_max - b_min
            wider      = range_A > range_B
            std_higher = a_std > b_std
            mask_match = abs(a_mask_pc - b_mask_pc) < 15.0   # within 15%

            print(f"\n  [{split}]  Cross-domain")
            print(f"    A range ({range_A:.4f}) > B range ({range_B:.4f})  : "
                  f"{'✓ PASS' if wider      else '⚠ FAIL — corrupted should have wider PSC range'}")
            print(f"    A std   ({a_std:.4f}) > B std   ({b_std:.4f})    : "
                  f"{'✓ PASS' if std_higher else '⚠ FAIL — corrupted should have higher variance'}")
            print(f"    Mask coverage match ({a_mask_pc:.1f}% vs {b_mask_pc:.1f}%) : "
                  f"{'✓ PASS' if mask_match else '⚠ FAIL — A and B have very different brain coverage, check extraction'}")

    # A-queue coverage check
    if "train" in loaders:
        print("\n" + "=" * 70)
        print("A-QUEUE COVERAGE  (10 epochs)")
        print("=" * 70)
        ds   = loaders["train"].dataset
        seen: Counter = Counter()
        for _ in range(10):
            ds.on_epoch_start()
            seen.update(ds._a_epoch_indices)
        vals = list(seen.values())
        print(f"  Unique A chunks seen : {len(seen)} / {ds._len_A}  "
              f"({100*len(seen)/ds._len_A:.1f}%)")
        print(
            f"  Appearances per chunk: min={min(vals)}  max={max(vals)}  "
            f"mean={sum(vals)/len(vals):.1f}"
        )