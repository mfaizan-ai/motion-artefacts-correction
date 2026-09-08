"""
test_st_model.py
================
Sanity-check the SpatioTemporalCycleGAN forward pass, loss computation,
and backward pass with dummy tensors matching the fix_v5 training config.

Run on a SLURM GPU node:
    sbatch slurm/run_test_st_model.sh
"""

import torch
import torch.nn.functional as F

from models.st_model import SpatioTemporalCycleGAN
from losses import LossWeights, generator_loss

# ---------------------------------------------------------------------------
# Config — mirrors run_train.sh / fix_v5
# ---------------------------------------------------------------------------
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH        = 2
IN_T         = 5
SPATIAL      = (80, 96, 72)
NUM_SCALES   = 2
RESIDUAL     = True


def make_batch():
    shape = (BATCH, IN_T, *SPATIAL)
    return (
        torch.randn(*shape, device=DEVICE),   # x_a  motion-corrupted
        torch.randn(*shape, device=DEVICE),   # x_b  motion-free
    )


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


if __name__ == "__main__":
    print(f"Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # -----------------------------------------------------------------------
    # 1. Build model
    # -----------------------------------------------------------------------
    section("1. Building SpatioTemporalCycleGAN")
    model = SpatioTemporalCycleGAN(
        in_timepoints    = IN_T,
        spatial_dims     = SPATIAL,
        content_base_ch  = 64,
        content_n_res    = 5,
        artefact_base_ch = 64,
        global_code_dim  = 64,
        spatial_code_ch  = 32,
        disc_base_ch     = 64,
        num_disc_scales  = NUM_SCALES,
        temporal_k       = 3,
        residual         = RESIDUAL,
    ).to(DEVICE)

    counts = model.count_parameters()
    for name, n in counts.items():
        print(f"  {name:<40} {n:>12,}")

    # -----------------------------------------------------------------------
    # 2. Generator forward pass
    # -----------------------------------------------------------------------
    section("2. Generator forward pass")
    x_a, x_b = make_batch()
    print(f"  x_a : {tuple(x_a.shape)}   x_b : {tuple(x_b.shape)}")

    out = model(x_a, x_b, detach_fakes_for_D=False)

    print("\n  ModelOutputs shapes:")
    for field, val in vars(out).items():
        if isinstance(val, torch.Tensor):
            print(f"    {field:<20} {tuple(val.shape)}")
        elif isinstance(val, list):
            print(f"    {field:<20} list of {[tuple(v.shape) for v in val]}")

    # -----------------------------------------------------------------------
    # 3. Generator loss + backward
    # -----------------------------------------------------------------------
    section("3. Generator loss + backward")
    weights  = LossWeights(adv=1.0, cyc=10.0, idt=5.0)
    g_losses = generator_loss(out, x_a, x_b, weights)
    for k, v in g_losses.items():
        print(f"  {k:<10} {v.item():.6f}")

    g_losses["total"].backward()
    print("\n  backward() — OK")

    # -----------------------------------------------------------------------
    # 4. Discriminator forward pass (detached fakes)
    # -----------------------------------------------------------------------
    section("4. Discriminator forward pass")
    model.zero_grad()
    x_a, x_b = make_batch()
    out_d = model(x_a, x_b, detach_fakes_for_D=True)

    def show_scores(name, scores):
        if isinstance(scores, list):
            for i, s in enumerate(scores):
                print(f"  {name}[scale {i}] {tuple(s.shape)}  mean={s.mean().item():.4f}")
        else:
            print(f"  {name} {tuple(scores.shape)}  mean={scores.mean().item():.4f}")

    show_scores("score_real_b", out_d.score_real_b)
    show_scores("score_fake_b", out_d.score_fake_b)
    show_scores("score_real_a", out_d.score_real_a)
    show_scores("score_fake_a", out_d.score_fake_a)

    def lsgan_d(sr, sf):
        if isinstance(sr, list):
            n = len(sr)
            return sum(
                0.5 * (F.mse_loss(r, torch.ones_like(r)) +
                       F.mse_loss(f, torch.zeros_like(f)))
                for r, f in zip(sr, sf)
            ) / n
        return 0.5 * (F.mse_loss(sr, torch.ones_like(sr)) +
                      F.mse_loss(sf, torch.zeros_like(sf)))

    L_D_B = lsgan_d(out_d.score_real_b, out_d.score_fake_b)
    L_D_A = lsgan_d(out_d.score_real_a, out_d.score_fake_a)
    print(f"\n  L_D_B = {L_D_B.item():.4f}  (expect ~0.25 at equilibrium)")
    print(f"  L_D_A = {L_D_A.item():.4f}  (expect ~0.25 at equilibrium)")
    (L_D_B + L_D_A).backward()
    print("  backward() — OK")

    # -----------------------------------------------------------------------
    # 5. Inference path
    # -----------------------------------------------------------------------
    section("5. Inference (model.correct)")
    model.eval()
    with torch.no_grad():
        x_test    = torch.randn(1, IN_T, *SPATIAL, device=DEVICE)
        x_out     = model.correct(x_test)
    print(f"  input     : {tuple(x_test.shape)}")
    print(f"  corrected : {tuple(x_out.shape)}")
    assert x_out.shape == x_test.shape, "Shape mismatch!"
    print("  shape check — OK")

    # -----------------------------------------------------------------------
    # 6. GPU memory
    # -----------------------------------------------------------------------
    if DEVICE.type == "cuda":
        section("6. GPU memory")
        print(f"  Allocated : {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  Reserved  : {torch.cuda.memory_reserved(0)  / 1e9:.2f} GB")

    print("\n✓ All checks passed.\n")
