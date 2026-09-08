"""
debug_st_model_shapes.py
=========================
Small script to step through SpatioTemporalCycleGAN (the factorized R(3+1)D
model with TemporalConv1D in the generator -- models/st_model.py, the
architecture used by the st_v2_ddp run) stage by stage, with your updated
dataset shape: chunk size T=5, spatial (56, 72, 60).

(56, 72, 60) doesn't survive the 3 stride-2 downsample / 3 upsample stages
unchanged -- 60 isn't a multiple of 8 (60 -> 30 -> 15 -> 7 down, but only
7 -> 14 -> 28 -> 56 back up), so the residual add `x_a + decoder(x_a)` used
by --residual would fail on the last axis. This script pads the real input
up to the nearest multiple of 8 (56, 72, 60) -> (56, 72, 64) before feeding
the model, does the residual add in that padded space, then center-crops
back down to (56, 72, 60) afterwards -- i.e. padding/unpadding wraps the
model as pure pre/post-processing, no change needed to st_model.py itself.

Doesn't touch the full model's forward() -- calls E_c / E_a / G_B / G_A's
own sub-layers directly (down1, down2, res[i], up1, ...) and prints the
tensor shape after each one, so you can see exactly where things change.

Run flat (CPU is fine for this -- batch=1, just checking shapes):
    python -m tests.debug_st_model_shapes

Run interactively to keep every intermediate tensor in scope afterwards
(x_a, c_a, a_global, x_hat_b, ...):
    python -i -m tests.debug_st_model_shapes
"""
import math

import torch
import torch.nn.functional as F

from models.st_model import (
    STContentEncoder,
    STArtefactEncoder,
    STMotionFreeDecoder,
    STMotionCorruptedDecoder,
)

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH         = 1
IN_T          = 5
ORIG_SPATIAL  = (56, 72, 60)   # (D, H, W) -- your real dataset spatial shape
PAD_MULTIPLE  = 8              # 3 stride-2 stages -> every dim must divide by 8
PAD_MODE      = "replicate"    # edge-replicate rather than zero-fill for real volumes


def shape(t):
    return tuple(t.shape)


def section(title):
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def pad_to_multiple(x, multiple=PAD_MULTIPLE, mode=PAD_MODE):
    """
    Symmetric-pad the last 3 (spatial) dims of x up to the next multiple of
    `multiple`, so 3 stride-2 downsample stages round-trip exactly.

    x: (..., D, H, W)  ->  (padded_x, pad) where `pad` is the 6-tuple in
    F.pad order (W_lo, W_hi, H_lo, H_hi, D_lo, D_hi), so `unpad(padded_x,
    pad)` below reverses it exactly.
    """
    D, H, W = x.shape[-3], x.shape[-2], x.shape[-1]

    def amounts(size):
        target = math.ceil(size / multiple) * multiple
        total  = target - size
        lo     = total // 2
        hi     = total - lo
        return lo, hi

    d_lo, d_hi = amounts(D)
    h_lo, h_hi = amounts(H)
    w_lo, w_hi = amounts(W)
    pad = (w_lo, w_hi, h_lo, h_hi, d_lo, d_hi)  # F.pad pads last dim first
    return F.pad(x, pad, mode=mode), pad


def unpad(x, pad):
    """Reverse pad_to_multiple: center-crop back to the original spatial size."""
    w_lo, w_hi, h_lo, h_hi, d_lo, d_hi = pad
    D, H, W = x.shape[-3], x.shape[-2], x.shape[-1]
    return x[..., d_lo:D - d_hi, h_lo:H - h_hi, w_lo:W - w_hi]


if __name__ == "__main__":
    torch.manual_seed(0)
    print(f"Device: {DEVICE}   T={IN_T}   orig spatial={ORIG_SPATIAL}")

    x_a_orig = torch.randn(BATCH, IN_T, *ORIG_SPATIAL, device=DEVICE)  # motion-corrupted
    x_b_orig = torch.randn(BATCH, IN_T, *ORIG_SPATIAL, device=DEVICE)  # motion-free

    # -----------------------------------------------------------------
    section("Pad up to a multiple of 8 (needed for the 3 stride-2 stages)")
    x_a, pad = pad_to_multiple(x_a_orig)
    x_b, pad_b = pad_to_multiple(x_b_orig)
    assert pad == pad_b  # same original shape -> same pad on both domains
    print(f"  x_a_orig  {shape(x_a_orig)}  ->  x_a (padded)  {shape(x_a)}")
    print(f"  x_b_orig  {shape(x_b_orig)}  ->  x_b (padded)  {shape(x_b)}")
    print(f"  pad (W_lo,W_hi,H_lo,H_hi,D_lo,D_hi) = {pad}   mode={PAD_MODE!r}")

    E_c = STContentEncoder().to(DEVICE)
    E_a = STArtefactEncoder().to(DEVICE)
    G_B = STMotionFreeDecoder().to(DEVICE)
    G_A = STMotionCorruptedDecoder().to(DEVICE)

    # -----------------------------------------------------------------
    section("E_c -- content encoder, stage by stage (on padded x_a)")
    x = x_a.unsqueeze(2)  # (B, T, 1, D, H, W)
    print(f"  input (unsqueezed)      {shape(x)}")
    x = E_c.down1(x); print(f"  after down1              {shape(x)}")
    x = E_c.down2(x); print(f"  after down2              {shape(x)}")
    x = E_c.down3(x); print(f"  after down3 (bottleneck) {shape(x)}")
    for i, blk in enumerate(E_c.res):
        x = blk(x)
        print(f"  after res[{i}]              {shape(x)}")
    c_a = x
    print(f"  --> content c_a          {shape(c_a)}")

    # Drop into a debugger here if you want to poke at c_a directly:
    # import pdb; pdb.set_trace()

    # -----------------------------------------------------------------
    section("E_a -- artefact encoder, stage by stage (on padded x_a)")
    xa = x_a.unsqueeze(2)
    xa = E_a.down1(xa); print(f"  after down1              {shape(xa)}")
    xa = E_a.down2(xa); print(f"  after down2              {shape(xa)}")
    xa = E_a.down3(xa); print(f"  after down3 (bottleneck) {shape(xa)}")

    B, T, C3, D_, H_, W_ = xa.shape
    xa_bt     = xa.reshape(B * T, C3, D_, H_, W_)
    a_global  = E_a.global_mlp(E_a.global_pool(xa_bt)).reshape(B, T, -1)
    a_spatial = E_a.spatial_branch(xa_bt).reshape(B, T, -1, D_, H_, W_)
    print(f"  a_global                 {shape(a_global)}")
    print(f"  a_spatial                {shape(a_spatial)}")

    # -----------------------------------------------------------------
    section("G_B -- motion-free decoder (A -> B clean), stage by stage")
    x = c_a
    for i, blk in enumerate(G_B.res):
        x = blk(x)
        print(f"  after res[{i}]              {shape(x)}")
    x = G_B.up1(x); print(f"  after up1                {shape(x)}")
    x = G_B.up2(x); print(f"  after up2                {shape(x)}")
    x = G_B.up3(x); print(f"  after up3                {shape(x)}")

    B, T, C, D, H, W = x.shape
    x_flat      = x.reshape(B * T, C, D, H, W)
    x_hat_b_raw = G_B.out_conv(x_flat).squeeze(1).reshape(B, T, D, H, W)
    print(f"  decoder raw output       {shape(x_hat_b_raw)}")

    # residual add happens in padded space (mirrors --residual in training)
    x_hat_b_padded = x_a + x_hat_b_raw
    print(f"  x_a + raw (padded space) {shape(x_hat_b_padded)}   -- matches x_a padded, OK")

    # crop back down to the real dataset resolution before using/evaluating it
    x_hat_b = unpad(x_hat_b_padded, pad)
    print(f"  --> x_hat_b (unpadded)   {shape(x_hat_b)}   (x_a_orig was {shape(x_a_orig)})")
    assert tuple(x_hat_b.shape) == tuple(x_a_orig.shape), "unpad didn't recover original shape!"
    print("  shapes match. ")

    # -----------------------------------------------------------------
    section("G_A -- motion-corrupted decoder (content + artefact), stage by stage")
    x = c_a
    for i, blk in enumerate(G_A.adain_blocks):
        x = blk(x, a_global)
        print(f"  after adain_blocks[{i}]     {shape(x)}")

    B, T = x.shape[:2]
    _, _, _, Da, Ha, Wa = a_spatial.shape
    x_bt = x.reshape(B * T, -1, Da, Ha, Wa)
    a_bt = a_spatial.reshape(B * T, -1, Da, Ha, Wa)
    x_bt = G_A.spatial_merge(torch.cat([x_bt, a_bt], dim=1))
    x    = x_bt.reshape(B, T, -1, Da, Ha, Wa)
    print(f"  after spatial_merge          {shape(x)}")
    x = G_A.up1(x); print(f"  after up1                {shape(x)}")
    x = G_A.up2(x); print(f"  after up2                {shape(x)}")
    x = G_A.up3(x); print(f"  after up3                {shape(x)}")

    B, T, C, D, H, W = x.shape
    x_flat      = x.reshape(B * T, C, D, H, W)
    x_hat_a_raw = G_A.out_conv(x_flat).squeeze(1).reshape(B, T, D, H, W)
    print(f"  decoder raw output       {shape(x_hat_a_raw)}")

    x_hat_a_padded = x_b + x_hat_a_raw
    print(f"  x_b + raw (padded space) {shape(x_hat_a_padded)}   -- matches x_b padded, OK")

    x_hat_a = unpad(x_hat_a_padded, pad)
    print(f"  --> x_hat_a (unpadded)   {shape(x_hat_a)}   (x_b_orig was {shape(x_b_orig)})")
    assert tuple(x_hat_a.shape) == tuple(x_b_orig.shape), "unpad didn't recover original shape!"
    print("  shapes match. ")

    print("\nDone. Re-run with `python -i -m tests.debug_st_model_shapes` to keep")
    print("x_a_orig, x_a, c_a, a_global, a_spatial, x_hat_b, x_hat_a, pad in scope.")
    print("\nTo wire this into real training/inference: pad x_a/x_b right after")
    print("loading a batch (before encode), and unpad the final residual output")
    print("right before computing losses / writing NIfTI back out -- st_model.py")
    print("itself needs no changes, this is pure pre/post-processing.")
