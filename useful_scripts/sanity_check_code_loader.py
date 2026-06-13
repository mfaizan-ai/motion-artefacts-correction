# # Sanity check


# if __name__ == "__main__":
#     import argparse
#     from collections import Counter

#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--root",
#         default="/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
#                 "faizan_motion_correction_dataset/cyclegans_dataset",
#     )
#     parser.add_argument("--splits",   nargs="+", default=["train", "val", "test"])
#     parser.add_argument("--workers",  type=int,  default=4)
#     parser.add_argument("--batch",    type=int,  default=1)
#     parser.add_argument("--psc_batches", type=int, default=50,
#                         help="Batches to sample per domain for PSC range analysis")
#     args = parser.parse_args()

#     print(f"Root          : {args.root}")
#     print(f"Splits        : {args.splits}")
#     print(f"Target spatial: {TARGET_SPATIAL}\n")

#     loaders = build_dataloaders(
#         dataset_root=args.root,
#         splits=args.splits,
#         batch_size=args.batch,
#         num_workers=args.workers,
#         pin_memory=torch.cuda.is_available(),
#         augment_train=True,
#     )


#     # Per-split shape / round-trip / upsample checks
#     for split, loader in loaders.items():
#         batch = next(iter(loader))
#         a     = batch["A"]
#         mv_a  = batch["mean_vol_A"]
#         orig  = batch["orig_shape_A"]

#         print(
#             f"[{split}]  batches/epoch={len(loader)} | "
#             f"A: {tuple(a.shape)}  psc_range=[{a.min():.3f}, {a.max():.3f}] | "
#             f"mean_vol_A: {tuple(mv_a.shape)}  "
#             f"range=[{mv_a.min():.1f}, {mv_a.max():.1f}] | "
#             f"orig_shape_A: {orig}"
#         )
#         if "B" in batch:
#             b = batch["B"]
#             print(
#                 f"         B: {tuple(b.shape)}  "
#                 f"psc_range=[{b.min():.3f}, {b.max():.3f}]"
#             )

#         # PSC round-trip — compare only brain voxels since masking
#         # intentionally zeroes background, making denormalisation inexact there
#         raw_ref, _ = _load_nifti(
#             batch["path_A"][0] if isinstance(batch["path_A"], (list, tuple))
#             else batch["path_A"]
#         )
#         psc_ref, mean_ref = _psc_normalise(raw_ref)
#         raw_recovered     = psc_denormalise(psc_ref, mean_ref)
#         brain_mask_ref    = mean_ref > (mean_ref.max() * 0.01)   # (1, X, Y, Z)
#         max_err = ((raw_recovered - raw_ref) * brain_mask_ref).abs().max().item()
#         print(
#             f"         PSC round-trip max error : {max_err:.6f}  "
#             f"({'✓ PASS' if max_err < 1e-3 else '✗ FAIL'})"
#         )

#         # Upsample shape check
#         if isinstance(orig[0], torch.Tensor):
#             orig_tuple = tuple(int(o[0].item()) for o in orig)
#         else:
#             orig_tuple = tuple(int(o) for o in orig)
#         upsampled = upsample_to_original(raw_ref, orig_tuple)
#         print(
#             f"         Upsampled shape           : {tuple(upsampled.shape)}  "
#             f"(expected T={a.shape[1]}, spatial={orig_tuple})"
#         )

#     # -----------------------------------------------------------------------
#     # PSC range analysis across N batches per domain per split
#     #
#     # What we expect to see for brain-extracted data:
#     #
#     #   A (corrupted)
#     #     - PSC range wider, positive tail >> negative tail
#     #     - Motion spikes push signal upward → asymmetry > 1.0
#     #     - Typical corrupted BOLD: range roughly [-1, +3] or wider
#     #
#     #   B (motion-free)
#     #     - PSC range narrow and roughly symmetric
#     #     - Typical clean BOLD fluctuations: ~±5% (PSC ±0.05)
#     #     - No large positive outliers
#     #
#     #   Cross-domain
#     #     - A range > B range
#     #     - A std   > B std
#     #     - B asymmetry close to 1.0
#     #
#     #   Brain mask (derived from mean_vol > 0)
#     #     - Mask fraction should be consistent across A and B in same split
#     #     - Expect ~20-60% of voxels to be brain for infant fMRI
#     #     - Large discrepancy between A and B mask fractions = domain mismatch
#     # -----------------------------------------------------------------------
#     print("\n" + "=" * 70)
#     print("PSC RANGE ANALYSIS  (brain-extracted data)")
#     print(f"Sampling {args.psc_batches} batches per domain per split")
#     print("=" * 70)

#     for split, loader in loaders.items():
#         a_mins, a_maxs, a_stds, a_mask_fracs = [], [], [], []
#         b_mins, b_maxs, b_stds, b_mask_fracs = [], [], [], []

#         for i, batch in enumerate(loader):
#             if i >= args.psc_batches:
#                 break

#             a    = batch["A"]
#             mv_a = batch["mean_vol_A"]

#             a_mins.append(a.min().item())
#             a_maxs.append(a.max().item())
#             a_stds.append(a.std().item())
#             # brain mask fraction: voxels with mean > 0
#             a_mask_fracs.append((mv_a > 0).float().mean().item())

#             if "B" in batch:
#                 b    = batch["B"]
#                 mv_b = batch["mean_vol_B"]
#                 b_mins.append(b.min().item())
#                 b_maxs.append(b.max().item())
#                 b_stds.append(b.std().item())
#                 b_mask_fracs.append((mv_b > 0).float().mean().item())

#         # --- Domain A ---
#         a_min     = min(a_mins)
#         a_max     = max(a_maxs)
#         a_std     = sum(a_stds) / len(a_stds)
#         a_asym    = abs(a_max) / (abs(a_min) + 1e-9)
#         a_mask_pc = 100 * sum(a_mask_fracs) / len(a_mask_fracs)

#         print(f"\n  [{split}]  A (corrupted)")
#         print(f"    PSC range      : [{a_min:.4f},  {a_max:.4f}]")
#         print(f"    Mean std       : {a_std:.4f}")
#         print(f"    Asymmetry      : {a_asym:.3f}  "
#               f"({'✓ right-skewed as expected' if a_asym > 1.1 else '⚠ unexpectedly symmetric — check domain labels'})")
#         print(f"    Brain coverage : {a_mask_pc:.1f}%  "
#               f"({'✓ plausible' if 10 < a_mask_pc < 70 else '⚠ unexpected — check brain extraction'})")

#         # PSC magnitude check: after brain masking, range should be bounded
#         a_bounded = abs(a_min) < 5.0 and abs(a_max) < 10.0
#         print(f"    PSC bounded    : "
#               f"{'✓ PASS  (no background blowup)' if a_bounded else '⚠ FAIL  (possible unmasked background voxels)'}")

#         if b_mins:
#             b_min     = min(b_mins)
#             b_max     = max(b_maxs)
#             b_std     = sum(b_stds) / len(b_stds)
#             b_asym    = abs(b_max) / (abs(b_min) + 1e-9)
#             b_mask_pc = 100 * sum(b_mask_fracs) / len(b_mask_fracs)

#             print(f"\n  [{split}]  B (motion-free)")
#             print(f"    PSC range      : [{b_min:.4f},  {b_max:.4f}]")
#             print(f"    Mean std       : {b_std:.4f}")
#             print(f"    Asymmetry      : {b_asym:.3f}  "
#                   f"({'✓ roughly symmetric' if b_asym < 1.8 else '⚠ unexpected skew — check domain labels'})")
#             print(f"    Brain coverage : {b_mask_pc:.1f}%  "
#                   f"({'✓ plausible' if 10 < b_mask_pc < 70 else '⚠ unexpected — check brain extraction'})")

#             b_bounded = abs(b_min) < 5.0 and abs(b_max) < 5.0
#             print(f"    PSC bounded    : "
#                   f"{'✓ PASS  (no background blowup)' if b_bounded else '⚠ FAIL  (possible unmasked background voxels)'}")

#             # Cross-domain
#             range_A    = a_max - a_min
#             range_B    = b_max - b_min
#             wider      = range_A > range_B
#             std_higher = a_std > b_std
#             mask_match = abs(a_mask_pc - b_mask_pc) < 15.0   # within 15%

#             print(f"\n  [{split}]  Cross-domain")
#             print(f"    A range ({range_A:.4f}) > B range ({range_B:.4f})  : "
#                   f"{'✓ PASS' if wider      else '⚠ FAIL — corrupted should have wider PSC range'}")
#             print(f"    A std   ({a_std:.4f}) > B std   ({b_std:.4f})    : "
#                   f"{'✓ PASS' if std_higher else '⚠ FAIL — corrupted should have higher variance'}")
#             print(f"    Mask coverage match ({a_mask_pc:.1f}% vs {b_mask_pc:.1f}%) : "
#                   f"{'✓ PASS' if mask_match else '⚠ FAIL — A and B have very different brain coverage, check extraction'}")

#     # A-queue coverage check
#     if "train" in loaders:
#         print("\n" + "=" * 70)
#         print("A-QUEUE COVERAGE  (10 epochs)")
#         print("=" * 70)
#         ds   = loaders["train"].dataset
#         seen: Counter = Counter()
#         for _ in range(10):
#             ds.on_epoch_start()
#             seen.update(ds._a_epoch_indices)
#         vals = list(seen.values())
#         print(f"  Unique A chunks seen : {len(seen)} / {ds._len_A}  "
#               f"({100*len(seen)/ds._len_A:.1f}%)")
#         print(
#             f"  Appearances per chunk: min={min(vals)}  max={max(vals)}  "
#             f"mean={sum(vals)/len(vals):.1f}"
#         )