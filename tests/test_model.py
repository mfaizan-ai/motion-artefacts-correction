"""
test_model.py
=============
Sanity-check the DisentangledCycleGAN forward pass, loss computation, and
backward pass with dummy tensors matching the real training configuration.

Run on a SLURM GPU node — there is no CPU on the login node.
    sbatch slurm/run_test_model.sh
"""

import torch
import torch.nn.functional as F

from models.model import DisentangledCycleGAN
from losses import LossWeights, generator_loss

# ---------------------------------------------------------------------------
# Configuration — must match run_train.sh / fix_v5
# ---------------------------------------------------------------------------
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH        = 2          # small batch for the smoke test
IN_T         = 5          # --in_timepoints
SPATIAL      = (80, 96, 72)  # (D, H, W)  target spatial dims
NUM_SCALES   = 2          # --num_disc_scales
RESIDUAL     = True       # --residual


def make_batch():
    """Random (B, T, D, H, W) tensors on the target device."""
    shape = (BATCH, IN_T, *SPATIAL)
    x_a = torch.randn(*shape, device=DEVICE)  # motion-corrupted
    x_b = torch.randn(*shape, device=DEVICE)  # motion-free
    return x_a, x_b


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


if __name__ == "__main__":
    print(f"Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # -----------------------------------------------------------------------
    # 1. Build model
    # -----------------------------------------------------------------------
    print_section("1. Building model")
    model = DisentangledCycleGAN(
        in_timepoints    = IN_T,
        spatial_dims     = SPATIAL,
        # encoder / decoder defaults match training
        content_ch       = 384,
        content_base_ch  = 64,
        content_n_res    = 5,
        artefact_base_ch = 64,
        global_code_dim  = 64,
        spatial_code_ch  = 32,
        # discriminator
        disc_base_ch        = 64,
        num_disc_scales     = NUM_SCALES,
        disc_temporal_diffs = False,
        residual            = RESIDUAL,
    ).to(DEVICE)

    counts = model.count_parameters()
    for name, n in counts.items():
        print(f"  {name:<35} {n:>12,}")

    # -----------------------------------------------------------------------
    # 2. Generator forward pass — returns ModelOutputs
    # -----------------------------------------------------------------------
    print_section("2. Generator forward pass")
    x_a, x_b = make_batch()
    print(f"  x_a shape : {tuple(x_a.shape)}  (motion-corrupted)")
    print(f"  x_b shape : {tuple(x_b.shape)}  (motion-free)")

    out = model(x_a, x_b, detach_fakes_for_D=False)

    print("\n  ModelOutputs tensor shapes:")
    for field, val in vars(out).items():
        if isinstance(val, torch.Tensor):
            print(f"    {field:<20} {tuple(val.shape)}")
        elif isinstance(val, list):
            shapes = [tuple(v.shape) for v in val]
            print(f"    {field:<20} list{shapes}  (multi-scale)")

    # -----------------------------------------------------------------------
    # 3. Generator loss + backward
    # -----------------------------------------------------------------------
    print_section("3. Generator loss")
    weights = LossWeights(adv=1.0, cyc=10.0, idt=5.0)
    g_losses = generator_loss(out, x_a, x_b, weights)
    for k, v in g_losses.items():
        print(f"  {k:<10} {v.item():.6f}")

    g_losses["total"].backward()
    print("\n  backward() on G total loss — OK")

    # -----------------------------------------------------------------------
    # 4. Discriminator forward pass (detached fakes)
    # -----------------------------------------------------------------------
    print_section("4. Discriminator forward pass")
    model.zero_grad()
    x_a, x_b = make_batch()
    out_d = model(x_a, x_b, detach_fakes_for_D=True)

    def _show_scores(name, scores):
        if isinstance(scores, list):
            for i, s in enumerate(scores):
                print(f"  {name}[scale {i}] : {tuple(s.shape)}  "
                      f"mean={s.mean().item():.4f}")
        else:
            print(f"  {name} : {tuple(scores.shape)}  "
                  f"mean={scores.mean().item():.4f}")

    _show_scores("score_real_b", out_d.score_real_b)
    _show_scores("score_fake_b", out_d.score_fake_b)
    _show_scores("score_real_a", out_d.score_real_a)
    _show_scores("score_fake_a", out_d.score_fake_a)

    # Quick LSGAN D loss check
    def _lsgan_d(sr, sf):
        if isinstance(sr, list):
            n = len(sr)
            return sum(
                0.5 * (F.mse_loss(r, torch.ones_like(r)) +
                       F.mse_loss(f, torch.zeros_like(f)))
                for r, f in zip(sr, sf)
            ) / n
        return 0.5 * (F.mse_loss(sr, torch.ones_like(sr)) +
                      F.mse_loss(sf, torch.zeros_like(sf)))

    L_D_B = _lsgan_d(out_d.score_real_b, out_d.score_fake_b)
    L_D_A = _lsgan_d(out_d.score_real_a, out_d.score_fake_a)
    print(f"\n  L_D_B = {L_D_B.item():.4f}  (expect ~0.25 at equilibrium)")
    print(f"  L_D_A = {L_D_A.item():.4f}  (expect ~0.25 at equilibrium)")

    (L_D_B + L_D_A).backward()
    print("  backward() on D total loss — OK")

    # -----------------------------------------------------------------------
    # 5. Inference path (correction only — no artefact encoder)
    # -----------------------------------------------------------------------
    print_section("5. Inference path (model.correct)")
    model.eval()
    with torch.no_grad():
        x_test = torch.randn(1, IN_T, *SPATIAL, device=DEVICE)
        x_corrected = model.correct(x_test)
    print(f"  input     : {tuple(x_test.shape)}")
    print(f"  corrected : {tuple(x_corrected.shape)}")
    assert x_corrected.shape == x_test.shape, "Shape mismatch in correct()"
    print("  shape check — OK")

    # -----------------------------------------------------------------------
    # 6. Memory summary
    # -----------------------------------------------------------------------
    if DEVICE.type == "cuda":
        print_section("6. GPU memory")
        alloc  = torch.cuda.memory_allocated(0)  / 1e9
        reserv = torch.cuda.memory_reserved(0)   / 1e9
        print(f"  Allocated : {alloc:.2f} GB")
        print(f"  Reserved  : {reserv:.2f} GB")

    print("\n✓ All checks passed.\n")
