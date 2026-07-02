
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
import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

log = logging.getLogger(__name__)

# Weights for each loss term
@dataclass
class LossWeights:
    """Configurable weights for each generator loss term."""
    adv:      float = 1.0
    cyc:      float = 10.0
    idt:      float = 5.0
    content:  float = 1.0
    art:      float = 0.1
    temporal: float = 0.0
    fc:       float = 0.0


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
def adversarial_loss_discriminator(out: ModelOutputs,
                                    real_target: float = 1.0,
                                    fake_target: float = 0.0) -> Dict[str, Tensor]:
    """
    LSGAN discriminator loss with label smoothing.
    Real → real_target, fake → fake_target for both D_A and D_B.
    """
    real_b = torch.full_like(out.score_real_b, real_target)
    fake_b = torch.full_like(out.score_fake_b, fake_target)
    real_a = torch.full_like(out.score_real_a, real_target)
    fake_a = torch.full_like(out.score_fake_a, fake_target)

    L_D_B = (0.5 * F.mse_loss(out.score_real_b, real_b) +
             0.5 * F.mse_loss(out.score_fake_b, fake_b))

    L_D_A = (0.5 * F.mse_loss(out.score_real_a, real_a) +
             0.5 * F.mse_loss(out.score_fake_a, fake_a))

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


# 6. Temporal consistency loss (sequence mode)
def temporal_consistency_loss(input_seq: Tensor, corrected_seq: Tensor) -> Tensor:
    """
    Preserve frame-to-frame BOLD dynamics through motion correction.

    Concatenates the S chunks into one continuous T_total = S*T volume
    sequence (temporal order preserved throughout, including the chunk
    boundaries) and compares first-order temporal differences:
        dY_t = Y_{t+1} - Y_t   (corrected)
        dX_t = X_{t+1} - X_t   (input)
        L_temp = mean(| dY - dX |)

    This encourages the generator to preserve real signal change (the
    underlying neural dynamics) while still being free to remove motion
    artefacts — it does not push the corrected signal to be flat, only to
    change in step with the original.

    Args:
        input_seq:     (S, T, X, Y, Z)  original chunks, temporal order
        corrected_seq: (S, T, X, Y, Z)  corrected chunks, same order

    Returns:
        Scalar L1 loss over all S*T-1 consecutive volume pairs.
    """
    S, T = input_seq.shape[:2]
    x = input_seq.reshape(S * T, *input_seq.shape[2:]).detach()
    y = corrected_seq.reshape(S * T, *corrected_seq.shape[2:])

    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    return F.l1_loss(dy, dx)

# ROIs with std below this (in PSC units) are treated as degenerate —
# e.g. an ROI that falls entirely in PSC-zeroed background after nearest-
# neighbour atlas downsampling has an exactly-constant timeseries. Real
# BOLD PSC signal varies on the order of 1e-2 to 1; 1e-4 comfortably
# separates "no real signal" from "real but low-variance" ROIs while
# staying well clear of float32 catastrophic-cancellation territory.
_MIN_ROI_STD = 1e-4
def _pearson_corr_matrix(X: Tensor) -> Tensor:
    """
    Pearson correlation matrix from a (T, N) timeseries matrix.
    Fully differentiable.

    ROI columns with near-zero variance (degenerate — e.g. constant or
    all-zero timeseries) have mathematically undefined correlation. By
    convention they are treated as uncorrelated with every other ROI
    (zeroed off-diagonal) but self-correlated (diagonal forced to 1.0),
    so the result is always a well-formed correlation matrix — valid
    input to validate_fc_matrix — regardless of degenerate ROIs.
    """
    X_centered = X - X.mean(dim=0, keepdim=True)
    std = X_centered.std(dim=0, keepdim=True)              # (1, N)
    degenerate = (std.squeeze(0) < _MIN_ROI_STD)            # (N,) bool

    X_norm = X_centered / std.clamp(min=_MIN_ROI_STD)
    X_norm = X_norm.masked_fill(degenerate.unsqueeze(0), 0.0)

    corr = (X_norm.T @ X_norm) / max(X.shape[0] - 1, 1)

    if degenerate.any():
        # corr's diagonal is already exactly 0 at degenerate positions
        # (zeroed column dotted with itself); add 1.0 there out-of-place
        # so autograd never sees an in-place write to a graph tensor.
        diag_fix = torch.diag(degenerate.to(corr.dtype))
        corr = corr + diag_fix
    return corr


def validate_fc_matrix(fc: Tensor, name: str = "fc", tol: float = 1e-2) -> None:
    """
    Sanity-check a Pearson FC matrix: square (N, N), symmetric, unit diagonal.
    Raises AssertionError on failure. Intended to run on a detached tensor.
    """
    assert fc.dim() == 2 and fc.shape[0] == fc.shape[1], (
        f"{name}: expected square (N, N) matrix, got {tuple(fc.shape)}"
    )
    asym = (fc - fc.T).abs().max().item()
    assert asym < tol, f"{name}: not symmetric (max |fc - fc.T| = {asym:.2e})"

    diag = torch.diagonal(fc)
    diag_err = (diag - 1.0).abs().max().item()
    assert diag_err < tol, f"{name}: diagonal not ~1 (max |diag - 1| = {diag_err:.2e})"


# --- FC connection-strength masking (modular, pluggable strategies) ---
def _off_diagonal_mask(n: int, device: torch.device) -> Tensor:
    """Boolean (N, N) mask selecting each off-diagonal ROI pair once."""
    return torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)


def _mask_by_threshold(strength: Tensor, off_diag: Tensor, *,
                        threshold: float = 0.3, **_) -> Tensor:
    """Keep off-diagonal pairs with |r| >= threshold."""
    return (strength >= threshold) & off_diag


def _mask_by_topk(strength: Tensor, off_diag: Tensor, *,
                   top_k: Optional[int] = None, **_) -> Tensor:
    """Keep the top_k strongest |r| off-diagonal pairs."""
    if top_k is None:
        raise ValueError(
            "fc_mask_strategy='topk' requires top_k to be set (got None) — "
            "pass --fc_top_k on the CLI"
        )
    vals = strength[off_diag]
    k = min(top_k, vals.numel())
    if k == 0:
        return torch.zeros_like(off_diag)
    cutoff = torch.topk(vals, k).values.min()
    return (strength >= cutoff) & off_diag


def _mask_by_percentile(strength: Tensor, off_diag: Tensor, *,
                         percentile: Optional[float] = None, **_) -> Tensor:
    """
    Keep off-diagonal pairs with |r| at or above the given percentile.
    e.g. percentile=90 keeps the strongest 10% of connections.
    """
    if percentile is None:
        raise ValueError(
            "fc_mask_strategy='percentile' requires percentile to be set "
            "(got None) — pass --fc_percentile on the CLI"
        )
    vals = strength[off_diag]
    cutoff = torch.quantile(vals, percentile / 100.0)
    return (strength >= cutoff) & off_diag


# Registry: add new masking strategies here without touching call sites.
FC_MASK_STRATEGIES = {
    "threshold":  _mask_by_threshold,
    "topk":       _mask_by_topk,
    "percentile": _mask_by_percentile,
}

def compute_fc_strength_mask(
    fc_reference: Tensor,
    strategy: str = "threshold",
    **kwargs,
) -> Tensor:
    """
    Build a boolean (N, N) off-diagonal mask selecting the "meaningful"
    ROI-to-ROI connections of a reference FC matrix, by connection
    strength (|r|). Diagonal is always excluded.

    Args:
        fc_reference: (N, N) Pearson FC matrix used to judge strength
        strategy:     key into FC_MASK_STRATEGIES
                      ("threshold", "topk", "percentile")
        **kwargs:     forwarded to the selected strategy function
                      (e.g. threshold=0.3, top_k=500, percentile=90.0)

    Returns:
        (N, N) boolean mask, True for retained pairs (upper triangle only).
    """
    if strategy not in FC_MASK_STRATEGIES:
        raise ValueError(
            f"Unknown FC masking strategy '{strategy}'. "
            f"Choose from: {list(FC_MASK_STRATEGIES)}"
        )
    n = fc_reference.shape[0]
    off_diag = _off_diagonal_mask(n, fc_reference.device)
    strength = fc_reference.abs()
    return FC_MASK_STRATEGIES[strategy](strength, off_diag, **kwargs)

# 7. Functional connectivity preservation loss (sequence mode)
def fc_loss(input_roi_ts: Tensor,
            corrected_roi_ts: Tensor,
            mask_strategy: str = "threshold",
            fc_threshold: float = 0.3,
            top_k: Optional[int] = None,
            percentile: Optional[float] = None,
            validate: bool = True,
            stats: Optional[dict] = None) -> Tensor:
    """
    Regularise the generator to preserve the subject's strong (likely
    genuine) ROI-to-ROI functional connections, while leaving weak /
    borderline correlations — more likely motion-driven noise — out of
    the loss so the model is free to alter them during correction.

    The retention mask is built from the *motion-corrupted input* FC
    matrix's connection strength (not the corrected output), since the
    input represents the subject's actual (if noisy) underlying network.

        L_FC = (1/|M|) * sum_{(i,j) in M} |FC_input(i,j) - FC_pred(i,j)|

    where M is the set of retained ROI pairs (diagonal always excluded).

    Args:
        input_roi_ts:     (T_total, n_rois)  ROI mean timeseries, input
        corrected_roi_ts: (T_total, n_rois)  ROI mean timeseries, corrected
        mask_strategy:     "threshold" | "topk" | "percentile"
                            (see compute_fc_strength_mask / FC_MASK_STRATEGIES)
        fc_threshold:       |r| cutoff, used when mask_strategy="threshold"
        top_k:              pair count, used when mask_strategy="topk"
        percentile:         cutoff (0-100), used when mask_strategy="percentile"
        validate:           run FC-matrix sanity checks (shape/symmetry/diag)
        stats:              optional dict, populated in-place with
                             {"n_retained", "n_total_pairs", "retained_fraction"}

    Returns:
        Scalar L1 loss over the retained ROI pairs only. Returns 0 if the
        mask retains nothing (logged as a warning).
    """
    assert input_roi_ts.shape == corrected_roi_ts.shape, (
        f"fc_loss: shape mismatch input={tuple(input_roi_ts.shape)} "
        f"corrected={tuple(corrected_roi_ts.shape)}"
    )

    fc_inp = _pearson_corr_matrix(input_roi_ts)
    fc_cor = _pearson_corr_matrix(corrected_roi_ts)

    if validate:
        validate_fc_matrix(fc_inp.detach(), name="fc_input")
        validate_fc_matrix(fc_cor.detach(), name="fc_corrected")

    mask = compute_fc_strength_mask(
        fc_inp.detach(), strategy=mask_strategy,
        threshold=fc_threshold, top_k=top_k, percentile=percentile,
    )

    n = fc_inp.shape[0]
    n_total_pairs = n * (n - 1) // 2
    n_retained = int(mask.sum().item())
    retained_fraction = n_retained / max(n_total_pairs, 1)

    if stats is not None:
        stats["n_retained"] = n_retained
        stats["n_total_pairs"] = n_total_pairs
        stats["retained_fraction"] = retained_fraction

    log.info(
        f"fc_loss: retained {n_retained}/{n_total_pairs} ROI pairs "
        f"({retained_fraction:.1%})  strategy={mask_strategy}"
    )

    if n_retained == 0:
        log.warning("fc_loss: mask retained 0 ROI pairs — returning zero loss")
        return torch.zeros((), device=fc_inp.device, dtype=fc_inp.dtype)

    return F.l1_loss(fc_cor[mask], fc_inp[mask].detach())

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


def discriminator_loss(out: ModelOutputs,
                       real_target: float = 1.0,
                       fake_target: float = 0.0) -> Dict[str, Tensor]:
    """
    Combined discriminator loss with label smoothing.
    Returns dict with D_A, D_B and 'total' for .backward().
    """
    return adversarial_loss_discriminator(out, real_target, fake_target)


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