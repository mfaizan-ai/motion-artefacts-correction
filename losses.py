
#!/usr/bin/env python3
"""
losses.py
Baseline losses for Disentangled CycleGAN fMRI motion artefact correction.

Losses:
    1. Adversarial       (LSGAN)
    2. Cycle-consistency (L1)
    3. Identity          (L1)
    4. Content           (MSE in feature space)
    5. Artefact suppression (MSE → 0)
"""

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

# Weights for each loss term
@dataclass
class LossWeights:
    """Configurable weights for each generator loss term."""
    adv:     float = 1.0
    cyc:     float = 10.0
    idt:     float = 5.0
    content: float = 1.0
    art:     float = 0.1


# Manages all outputs from the model, including intermediate features and final outputs.
@dataclass
class ModelOutputs:
    # Phase 1: encoded features
    c_a:          Tensor   # (B, 384, D', H', W')  content from x_a
    c_b:          Tensor   # (B, 384, D', H', W')  content from x_b
    a_global:     Tensor   # (B, 64)               artefact global from x_a
    a_spatial:    Tensor   # (B, 32, D', H', W')   artefact spatial from x_a
    a_global_b:   Tensor   # (B, 64)               artefact global from x_b  → 0
    a_spatial_b:  Tensor   # (B, 32, D', H', W')   artefact spatial from x_b → 0
    
    
    # Phase 2: translations and self-reconstructions
    x_hat_b:      Tensor   # (B, T, X, Y, Z)  A→B predicted clean
    x_hat_a:      Tensor   # (B, T, X, Y, Z)  B→A predicted corrupted
    x_self_a:     Tensor   # (B, T, X, Y, Z)  A→A self-reconstruction
    x_self_b:     Tensor   # (B, T, X, Y, Z)  B→B self-reconstruction
    
    
    # Phase 3: cyclic re-encodings
    c_hat_b:      Tensor   # (B, 384, D', H', W')  re-encoded from x_hat_b
    c_hat_a:      Tensor   # (B, 384, D', H', W')  re-encoded from x_hat_a
    a_hat_global: Tensor   # (B, 64)
    a_hat_spatial:Tensor   # (B, 32, D', H', W')
    
    
    # Phase 3: cyclic outputs
    x_cycle_a:    Tensor   # (B, T, X, Y, Z)  A→B→A should ≈ x_a
    x_cycle_b:    Tensor   # (B, T, X, Y, Z)  B→A→B should ≈ x_b
    
    
    # Discriminator patch score maps
    score_real_b: Tensor   # (B, 1, d, h, w)  D_B on real x_b  (motion free real)
    score_fake_b: Tensor   # (B, 1, d, h, w)  D_B on x_hat_b.  (motion free fake)
    score_real_a: Tensor   # (B, 1, d, h, w)  D_A on real x_a. (motion corrupted real)
    score_fake_a: Tensor   # (B, 1, d, h, w)  D_A on x_hat_a.  (motion corrupted fake)



# 1. Adversarial discriminator loss (LSGAN)
def adversarial_loss_discriminator(out: ModelOutputs) -> Dict[str, Tensor]:
    """
    LSGAN discriminator loss.
    Real → 1, fake → 0 for both D_A and D_B.
    """
    ones_b  = torch.ones_like(out.score_real_b)
    zeros_b = torch.zeros_like(out.score_fake_b)
    ones_a  = torch.ones_like(out.score_real_a)
    zeros_a = torch.zeros_like(out.score_fake_a)

    L_D_B = (0.5 * F.mse_loss(out.score_real_b, ones_b) +
             0.5 * F.mse_loss(out.score_fake_b, zeros_b))

    L_D_A = (0.5 * F.mse_loss(out.score_real_a, ones_a) +
             0.5 * F.mse_loss(out.score_fake_a, zeros_a))

    return {"D_B": L_D_B, "D_A": L_D_A, "total": L_D_B + L_D_A}

# Adversarial generator loss (LSGAN)
def adversarial_loss_generator(out: ModelOutputs) -> Tensor:
    """
    LSGAN generator loss.
    Generator tries to make discriminator output 1 on fakes.
    """
    ones_b = torch.ones_like(out.score_fake_b)
    ones_a = torch.ones_like(out.score_fake_a)

    return (0.5 * F.mse_loss(out.score_fake_b, ones_b) +
            0.5 * F.mse_loss(out.score_fake_a, ones_a))


# 2. Cycle-consistency loss (L1)
def cycle_consistency_loss(out: ModelOutputs,
                            x_a: Tensor,
                            x_b: Tensor) -> Tensor:
    """
    L1 cycle loss. A→B→A should recover x_a; B→A→B should recover x_b.
    L1 over L2: preserves sharp temporal transitions in PSC-normalised data.
    """
    return F.l1_loss(out.x_cycle_a, x_a) + F.l1_loss(out.x_cycle_b, x_b)



# 3. Identity / self-reconstruction loss (L1)
def identity_loss(out: ModelOutputs,
                  x_a: Tensor,
                  x_b: Tensor) -> Tensor:
    """
    L1 identity loss. Self-reconstruction should be a no-op.
    Prevents generators from changing input that does not need changing.
    """
    return F.l1_loss(out.x_self_a, x_a) + F.l1_loss(out.x_self_b, x_b)


# 4. Content consistency loss (MSE in feature space)
def content_loss(out: ModelOutputs) -> Tensor:
    """
    MSE on content feature maps.
    Re-encoding x_hat_b should recover c_a; re-encoding x_hat_a should recover c_b.
    .detach() on targets so gradients flow only through the re-encoding path.
    """
    return (F.mse_loss(out.c_hat_b, out.c_a.detach()) +
            F.mse_loss(out.c_hat_a, out.c_b.detach()))


# 5. Artefact suppression loss (MSE → 0)
def artefact_suppression_loss(out: ModelOutputs) -> Tensor:
    """
    Pushes artefact codes from clean chunks x_b toward zero.
    Clean data should have no encodable corruption signature.
    """
    return (F.mse_loss(out.a_global_b,  torch.zeros_like(out.a_global_b))  +
            F.mse_loss(out.a_spatial_b, torch.zeros_like(out.a_spatial_b)))



# Combined losses
def generator_loss(out: ModelOutputs,
                   x_a: Tensor,
                   x_b: Tensor,
                   weights: LossWeights = LossWeights()) -> Dict[str, Tensor]:
    """
    Combined weighted generator loss.
    Returns dict with individual terms and 'total' for .backward().
    """
    L_adv     = adversarial_loss_generator(out)
    L_cyc     = cycle_consistency_loss(out, x_a, x_b)
    L_idt     = identity_loss(out, x_a, x_b)
    L_content = content_loss(out)
    L_art     = artefact_suppression_loss(out)

    total = (weights.adv     * L_adv     +
             weights.cyc     * L_cyc     +
             weights.idt     * L_idt     +
             weights.content * L_content +
             weights.art     * L_art)

    return {
        "adv":     L_adv,
        "cyc":     L_cyc,
        "idt":     L_idt,
        "content": L_content,
        "art":     L_art,
        "total":   total,
    }


def discriminator_loss(out: ModelOutputs) -> Dict[str, Tensor]:
    """
    Combined discriminator loss.
    Returns dict with D_A, D_B and 'total' for .backward().
    """
    return adversarial_loss_discriminator(out)





# Sanity checks
if __name__ == "__main__":
    import sys

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy dimensions — match your actual data
    B, T, X, Y, Z   = 2, 20, 97, 116, 79
    C_feat           = 384   # content channels
    D_, H_, W_       = 12, 14, 9   # spatial dims after 8x downsampling
    A_glob, A_spat   = 64, 32
    patch            = (5, 6, 4)  # discriminator output patch

    def rand(*shape): return torch.randn(*shape, device=device, requires_grad=True)
    def ones(*shape): return torch.ones(*shape, device=device)

    print("Building dummy ModelOutputs ...")
    out = ModelOutputs(
        c_a           = rand(B, C_feat, D_, H_, W_),
        c_b           = rand(B, C_feat, D_, H_, W_),
        a_global      = rand(B, A_glob),
        a_spatial     = rand(B, A_spat, D_, H_, W_),
        a_global_b    = rand(B, A_glob),
        a_spatial_b   = rand(B, A_spat, D_, H_, W_),
        x_hat_b       = rand(B, T, X, Y, Z),
        x_hat_a       = rand(B, T, X, Y, Z),
        x_self_a      = rand(B, T, X, Y, Z),
        x_self_b      = rand(B, T, X, Y, Z),
        c_hat_b       = rand(B, C_feat, D_, H_, W_),
        c_hat_a       = rand(B, C_feat, D_, H_, W_),
        a_hat_global  = rand(B, A_glob),
        a_hat_spatial = rand(B, A_spat, D_, H_, W_),
        x_cycle_a     = rand(B, T, X, Y, Z),
        x_cycle_b     = rand(B, T, X, Y, Z),
        score_real_b  = rand(B, 1, *patch),
        score_fake_b  = rand(B, 1, *patch),
        score_real_a  = rand(B, 1, *patch),
        score_fake_a  = rand(B, 1, *patch),
    )

    x_a = rand(B, T, X, Y, Z)
    x_b = rand(B, T, X, Y, Z)

    print("Running individual loss checks ...")
    tests_passed = 0

    def check(name, fn):
        global tests_passed
        result = fn()
        # Scalar check
        assert result.shape == torch.Size([]), \
            f"{name}: expected scalar, got {result.shape}"
        # Finite check
        assert torch.isfinite(result), \
            f"{name}: non-finite value {result.item()}"
        # Non-negative check (all our losses are non-negative by construction)
        assert result.item() >= 0.0, \
            f"{name}: negative value {result.item()}"
        # Grad check — result must carry a grad_fn
        assert result.requires_grad or result.grad_fn is not None, \
            f"{name}: no gradient — loss cannot be backpropagated"
        print(f"  [PASS] {name:<35} value={result.item():.6f}")
        tests_passed += 1

    check("adversarial_generator",
          lambda: adversarial_loss_generator(out))

    check("cycle_consistency",
          lambda: cycle_consistency_loss(out, x_a, x_b))

    check("identity",
          lambda: identity_loss(out, x_a, x_b))

    check("content",
          lambda: content_loss(out))

    check("artefact_suppression",
          lambda: artefact_suppression_loss(out))

    # Discriminator loss
    d = adversarial_loss_discriminator(out)
    assert set(d.keys()) == {"D_A", "D_B", "total"}, \
        "discriminator_loss: missing keys"
    for k, v in d.items():
        assert torch.isfinite(v), f"discriminator_loss[{k}]: non-finite"
        assert v.item() >= 0.0,   f"discriminator_loss[{k}]: negative"
    print(f"  [PASS] {'discriminator_loss (all keys)':<35} "
          f"D_A={d['D_A'].item():.4f}  D_B={d['D_B'].item():.4f}")
    tests_passed += 1

    # Combined generator loss
    weights = LossWeights()
    g = generator_loss(out, x_a, x_b, weights)
    expected_keys = {"adv", "cyc", "idt", "content", "art", "total"}
    assert set(g.keys()) == expected_keys, \
        f"generator_loss: missing keys {expected_keys - set(g.keys())}"

    # Verify total equals weighted sum
    expected_total = (weights.adv     * g["adv"]     +
                      weights.cyc     * g["cyc"]     +
                      weights.idt     * g["idt"]     +
                      weights.content * g["content"] +
                      weights.art     * g["art"])
    assert torch.allclose(g["total"], expected_total, atol=1e-5), \
        "generator_loss: total does not match weighted sum of terms"
    print(f"  [PASS] {'generator_loss (weighted sum)':<35} "
          f"total={g['total'].item():.6f}")
    tests_passed += 1

    # Backward pass check — total must backpropagate without error
    g["total"].backward()
    print(f"  [PASS] {'generator_loss backward()':<35} no errors")
    tests_passed += 1

    # Custom weights check
    custom = LossWeights(adv=2.0, cyc=5.0, idt=2.5, content=0.5, art=0.05)
    g_custom = generator_loss(out, x_a, x_b, custom)
    assert torch.isfinite(g_custom["total"]), \
        "generator_loss with custom weights: non-finite total"
    print(f"  [PASS] {'generator_loss (custom weights)':<35} "
          f"total={g_custom['total'].item():.6f}")
    tests_passed += 1

    print(f"\nAll {tests_passed} tests passed.")
    sys.exit(0)