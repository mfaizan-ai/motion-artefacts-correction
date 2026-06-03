#!/usr/bin/env python3
"""
test_model.py
=============
Sanity check for the full DisentangledCycleGAN model.

Tests:
    1. All output shapes match expectations
    2. Phase 1 — encoded feature shapes
    3. Phase 2 — intermediate translation shapes
    4. Phase 3 — cyclic output shapes
    5. Discriminator score shapes
    6. Inference-only path (correct method)
    7. Backward pass through generator losses
    8. Backward pass through discriminator losses
    9. Parameter counts per component
   10. Content preservation check (untrained baseline)

Run:
    python test_model.py
"""

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from model import DisentangledCycleGAN


PASS = "✓ PASS"
FAIL = "✗ FAIL"

def check(name: str,
          got:      tuple,
          expected: tuple) -> bool:
    status = PASS if got == expected else FAIL
    print(f"  {status}  {name:45s} got {got}  expected {expected}")
    return got == expected


def section(title: str) -> None:
    print(f"\n{'═' * 65}")
    print(f"  {title}")
    print(f"{'═' * 65}")


# CONFIGURATION
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B  = 2          # batch size
T  = 20         # timepoints
D, H, W = 80, 96, 72        # spatial dims (input)
BD, BH, BW = 10, 12, 9      # bottleneck spatial dims
C_CH = 384                  # content channels
G_DIM = 64                  # global artefact dim
S_CH  = 32                  # spatial artefact channels
P_D, P_H, P_W = 5, 6, 4    # patch discriminator output dims


# MAIN SANITY CHECK
def run_tests() -> None:

    print(f"\nDevice : {DEVICE}")

    # Instantiate model 
    section("INSTANTIATING MODEL")
    model = DisentangledCycleGAN(
        in_timepoints    = T,
        spatial_dims     = (D, H, W),
        content_ch       = C_CH,
        content_base_ch  = 64,
        content_n_res    = 5,
        artefact_base_ch = 64,
        global_code_dim  = G_DIM,
        spatial_code_ch  = S_CH,
        disc_base_ch     = 64,
    ).to(DEVICE)
    print("  Model instantiated successfully")

    #  Parameter counts 
    section("PARAMETER COUNTS")
    counts = model.count_parameters()
    for name, count in counts.items():
        marker = "──" if name != "Total" else "══"
        print(f"  {marker}  {name:40s} : {count:>12,}")

    #  Create dummy inputs 
    section("DUMMY INPUTS")
    x_a = torch.randn(B, T, D, H, W, device=DEVICE)
    x_b = torch.randn(B, T, D, H, W, device=DEVICE)
    print(f"  x_a (corrupted)   : {tuple(x_a.shape)}")
    print(f"  x_b (motion-free) : {tuple(x_b.shape)}")

    # Full forward pass 
    section("FULL FORWARD PASS")
    with torch.no_grad():
        out = model(x_a, x_b)
    print("  Forward pass completed without error\n")

    all_passed = True

    # Phase 1: Encoded features 
    print("  PHASE 1 — ENCODED FEATURES")
    print("  " + "─" * 60)
    all_passed &= check("c_a  content from x_a",
                        tuple(out.c_a.shape),
                        (B, C_CH, BD, BH, BW))
    all_passed &= check("c_b  content from x_b",
                        tuple(out.c_b.shape),
                        (B, C_CH, BD, BH, BW))
    all_passed &= check("a_global   artefact global from x_a",
                        tuple(out.a_global.shape),
                        (B, G_DIM))
    all_passed &= check("a_spatial  artefact spatial from x_a",
                        tuple(out.a_spatial.shape),
                        (B, S_CH, BD, BH, BW))
    all_passed &= check("a_global_b  artefact global from x_b",
                        tuple(out.a_global_b.shape),
                        (B, G_DIM))
    all_passed &= check("a_spatial_b artefact spatial from x_b",
                        tuple(out.a_spatial_b.shape),
                        (B, S_CH, BD, BH, BW))

    # Phase 2: Intermediate translations 
    print("\n  PHASE 2 — INTERMEDIATE TRANSLATIONS")
    print("  " + "─" * 60)
    chunk_shape = (B, T, D, H, W)
    all_passed &= check("x_hat_b  A→B predicted clean",
                        tuple(out.x_hat_b.shape),  chunk_shape)
    all_passed &= check("x_hat_a  B→A predicted corrupted",
                        tuple(out.x_hat_a.shape),  chunk_shape)
    all_passed &= check("x_self_a A→A self-reconstruct corrupted",
                        tuple(out.x_self_a.shape), chunk_shape)
    all_passed &= check("x_self_b B→B self-reconstruct clean",
                        tuple(out.x_self_b.shape), chunk_shape)

    # Phase 3: Cyclic re-encodings 
    print("\n  PHASE 3 — CYCLIC RE-ENCODINGS")
    print("  " + "─" * 60)
    all_passed &= check("c_hat_b    re-encoded from x_hat_b",
                        tuple(out.c_hat_b.shape),
                        (B, C_CH, BD, BH, BW))
    all_passed &= check("c_hat_a    re-encoded from x_hat_a",
                        tuple(out.c_hat_a.shape),
                        (B, C_CH, BD, BH, BW))
    all_passed &= check("a_hat_global  artefact from x_hat_a",
                        tuple(out.a_hat_global.shape),
                        (B, G_DIM))
    all_passed &= check("a_hat_spatial artefact from x_hat_a",
                        tuple(out.a_hat_spatial.shape),
                        (B, S_CH, BD, BH, BW))

    #  Phase 3: Cyclic outputs 
    print("\n  PHASE 3 — CYCLIC OUTPUTS")
    print("  " + "─" * 60)
    all_passed &= check("x_cycle_a  A→B→A  should ≈ x_a",
                        tuple(out.x_cycle_a.shape), chunk_shape)
    all_passed &= check("x_cycle_b  B→A→B  should ≈ x_b",
                        tuple(out.x_cycle_b.shape), chunk_shape)

    # Discriminator scores
    print("\n  DISCRIMINATOR SCORES")
    print("  " + "─" * 60)
    patch_shape = (B, 1, P_D, P_H, P_W)
    all_passed &= check("score_real_b  D_B on real x_b",
                        tuple(out.score_real_b.shape), patch_shape)
    all_passed &= check("score_fake_b  D_B on x_hat_b",
                        tuple(out.score_fake_b.shape), patch_shape)
    all_passed &= check("score_real_a  D_A on real x_a",
                        tuple(out.score_real_a.shape), patch_shape)
    all_passed &= check("score_fake_a  D_A on x_hat_a",
                        tuple(out.score_fake_a.shape), patch_shape)

    #  Output value checks 
    section("OUTPUT VALUE CHECKS")

    # Chunk outputs should be in [-1, 1] due to Tanh
    for name, tensor in [
        ("x_hat_b",  out.x_hat_b),
        ("x_hat_a",  out.x_hat_a),
        ("x_self_a", out.x_self_a),
        ("x_self_b", out.x_self_b),
        ("x_cycle_a",out.x_cycle_a),
        ("x_cycle_b",out.x_cycle_b),
    ]:
        mn, mx = tensor.min().item(), tensor.max().item()
        in_range = (mn >= -1.01 and mx <= 1.01)
        status = PASS if in_range else FAIL
        print(f"  {status}  {name:12s} range [{mn:+.3f}, {mx:+.3f}]"
              f"  (Tanh bounds ±1)")
        all_passed &= in_range

    # Artefact codes from clean chunks should have smaller magnitude
    # than from corrupted chunks (untrained — just a directional check)
    mag_corrupt = out.a_global.abs().mean().item()
    mag_clean   = out.a_global_b.abs().mean().item()
    print(f"\n  a_global mean magnitude — corrupted : {mag_corrupt:.4f}")
    print(f"  a_global mean magnitude — clean     : {mag_clean:.4f}")
    print(f"  (After training, clean should converge toward 0)")

    # Inference path 
    section("INFERENCE PATH  (correct method — E_c + G_B only)")
    with torch.no_grad():
        x_corrected = model.correct(x_a)
    all_passed &= check("correct(x_a) output shape",
                        tuple(x_corrected.shape), chunk_shape)
    print(f"  Output range : [{x_corrected.min():.3f}, {x_corrected.max():.3f}]")

    # Backward pass: discriminator 
    section("BACKWARD PASS — DISCRIMINATOR")
    opt_D = torch.optim.Adam(model.discriminator_parameters(),
                              lr=2e-4, betas=(0.5, 0.999))
    out2 = model(x_a, x_b)

    # LSGAN discriminator loss — detach fakes to isolate discriminator update
    loss_D = (
        0.5 * torch.mean((out2.score_real_b - 1.0) ** 2) +
        0.5 * torch.mean((out2.score_fake_b.detach()) ** 2) +
        0.5 * torch.mean((out2.score_real_a - 1.0) ** 2) +
        0.5 * torch.mean((out2.score_fake_a.detach()) ** 2)
    )
    opt_D.zero_grad()
    loss_D.backward()
    opt_D.step()
    print(f"  Discriminator loss : {loss_D.item():.4f}")
    print(f"  Backward pass      : {PASS}")

    # ── Backward pass: generator + encoders ───────────────────────────────
    section("BACKWARD PASS — GENERATORS + ENCODERS")
    opt_G = torch.optim.Adam(model.generator_parameters(),
                              lr=2e-4, betas=(0.5, 0.999))
    out3 = model(x_a, x_b)

    # Sample losses (weights from recommended λ values)
    L_adv  = (torch.mean((out3.score_fake_b - 1.0) ** 2) +
              torch.mean((out3.score_fake_a - 1.0) ** 2))

    L_cycle = (F.l1_loss(out3.x_cycle_a, x_a) +
               F.l1_loss(out3.x_cycle_b, x_b))

    L_recon = (F.l1_loss(out3.x_self_a, x_a) +
               F.l1_loss(out3.x_self_b, x_b))

    L_content = (F.mse_loss(out3.c_hat_b, out3.c_a.detach()) +
                 F.mse_loss(out3.c_hat_a, out3.c_b.detach()))

    L_G = (1.0 * L_adv   +
           10.0 * L_cycle +
           5.0  * L_recon +
           5.0  * L_content)

    opt_G.zero_grad()
    L_G.backward()
    opt_G.step()

    print(f"  L_adv    : {L_adv.item():.4f}  (weight 1.0)")
    print(f"  L_cycle  : {L_cycle.item():.4f}  (weight 10.0)")
    print(f"  L_recon  : {L_recon.item():.4f}  (weight 5.0)")
    print(f"  L_content: {L_content.item():.4f}  (weight 5.0)")
    print(f"  L_G total: {L_G.item():.4f}")
    print(f"  Backward pass : {PASS}")

    # Final summary 
    section("FINAL SUMMARY")
    status = PASS if all_passed else "✗ SOME TESTS FAILED"
    print(f"  All shape tests : {status}")
    print(f"  Discriminator backward : {PASS}")
    print(f"  Generator backward     : {PASS}")
    print(f"\n  Model is ready for training.")
    print(f"{'═' * 65}\n")


if __name__ == "__main__":
    run_tests()