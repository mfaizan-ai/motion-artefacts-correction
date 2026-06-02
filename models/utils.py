"""
utils.py
========

Utility functions for the model, such as shape printing and sanity checks etc. Not part of the core model architecture, but useful for debugging and verification.
"""

import torch
from encoders import ContentEncoder, ArtefactEncoder
from decoders import MotionFreeDecoder, MotionCorruptedDecoder

def print_shape(name: str, tensor: torch.Tensor) -> None:
    print(f"  {name:20s} : {tuple(tensor.shape)}")

def run_sanity_check_encoders() -> None:
    """
    Instantiate both encoders, run a random input through them,
    and verify all output shapes match expectations.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
 
    #  Input 
    B   = 2          # batch size
    T   = 20         # timepoints (channels)
    D, H, W = 80, 96, 72   # spatial dims
 
    x = torch.randn(B, T, D, H, W, device=device)
    print(f"\nInput shape: {tuple(x.shape)}")
 
    #  Content Encoder
    print("\n" + "=" * 55)
    print("  CONTENT ENCODER")
    print("=" * 55)
 
    content_enc = ContentEncoder(
        in_channels   = T,
        base_channels = 64,
        n_res_blocks  = 5,
    ).to(device)
 
    with torch.no_grad():
        content = content_enc(x)
 
    print_shape("Input",   x)
    print_shape("Content", content)
 
    expected_content = (B, 384, 10, 12, 9)
    status = "✓ PASS" if tuple(content.shape) == expected_content else "✗ FAIL"
    print(f"\n  Expected : {expected_content}")
    print(f"  Got      : {tuple(content.shape)}")
    print(f"  Status   : {status}")
 
    # Parameter count
    n_params = sum(p.numel() for p in content_enc.parameters())
    print(f"  Parameters: {n_params:,}")
 
    # Artefact Encoder 
    print("\n" + "=" * 55)
    print("  ARTEFACT ENCODER")
    print("=" * 55)
 
    artefact_enc = ArtefactEncoder(
        in_channels     = T,
        base_channels   = 64,
        global_code_dim = 64,
        spatial_code_ch = 32,
    ).to(device)
 
    with torch.no_grad():
        a_global, a_spatial = artefact_enc(x)
 
    print_shape("Input",      x)
    print_shape("a_global",   a_global)
    print_shape("a_spatial",  a_spatial)
 
    expected_global  = (B, 64)
    expected_spatial = (B, 32, 10, 12, 9)
 
    ok_global  = tuple(a_global.shape)  == expected_global
    ok_spatial = tuple(a_spatial.shape) == expected_spatial
 
    print(f"\n  a_global  expected : {expected_global}  "
          f"got : {tuple(a_global.shape)}  "
          f"{'✓ PASS' if ok_global  else '✗ FAIL'}")
    print(f"  a_spatial expected : {expected_spatial}  "
          f"got : {tuple(a_spatial.shape)}  "
          f"{'✓ PASS' if ok_spatial else '✗ FAIL'}")
 
    n_params = sum(p.numel() for p in artefact_enc.parameters())
    print(f"  Parameters: {n_params:,}")
 
    # Forward pass with both encoders 
    print("\n" + "=" * 55)
    print("  BOTH ENCODERS — FULL FORWARD PASS")
    print("=" * 55)
 
    # Simulate one training step: encode a corrupted (x_a) and clean (x_b)
    x_a = torch.randn(B, T, D, H, W, device=device)   # corrupted domain
    x_b = torch.randn(B, T, D, H, W, device=device)   # motion-free domain
 
    with torch.no_grad():
        c_a              = content_enc(x_a)
        c_b              = content_enc(x_b)
        a_global, a_spat = artefact_enc(x_a)
        _,        a_b    = artefact_enc(x_b)   # should be ~0 for clean
 
    print(f"\n  From corrupted chunk x_a:")
    print_shape("  c_a (content)",     c_a)
    print_shape("  a_global",          a_global)
    print_shape("  a_spatial",         a_spat)
 
    print(f"\n  From motion-free chunk x_b:")
    print_shape("  c_b (content)",     c_b)
    print_shape("  a_spatial_b",       a_b)
 
    print(f"\n  a_global mean  : {a_global.mean().item():.4f}  "
          f"(not yet trained — should converge toward 0 for clean chunks)")
    print(f"  a_global std   : {a_global.std().item():.4f}")
    print(f"  content range  : [{c_a.min().item():.3f}, "
          f"{c_a.max().item():.3f}]")
 
    print("\n" + "=" * 55)
    print("  All shapes verified.")
    print("=" * 55)
 
 
def run_sanity_check_decoders() -> None:
    """
    Instantiate both decoders, run random inputs through them,
    and verify all output shapes match expectations.
    Also simulates the full A→B and B→A translation paths.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
 
    # ── Encoder output shapes (as produced by ContentEncoder/ArtefactEncoder)
    B             = 2
    CONTENT_CH    = 384
    BOT_D, BOT_H, BOT_W = 10, 12, 9    # bottleneck spatial dims
    GLOBAL_DIM    = 64
    SPATIAL_CH    = 32
    OUT_CH        = 20
    FULL_D, FULL_H, FULL_W = 80, 96, 72  # full spatial dims
 
    # Simulate encoder outputs
    content   = torch.randn(B, CONTENT_CH, BOT_D, BOT_H, BOT_W, device=device)
    a_global  = torch.randn(B, GLOBAL_DIM,                        device=device)
    a_spatial = torch.randn(B, SPATIAL_CH, BOT_D, BOT_H, BOT_W,  device=device)
 
    # ── Motion-Free Decoder ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MOTION-FREE DECODER")
    print("=" * 60)
 
    mf_decoder = MotionFreeDecoder(
        content_ch   = CONTENT_CH,
        out_channels = OUT_CH,
        n_res_blocks = 4,
    ).to(device)
 
    with torch.no_grad():
        chunk_clean = mf_decoder(content)
 
    print_shape("Input  content",       content)
    print_shape("Output chunk (clean)", chunk_clean)
 
    expected = (B, OUT_CH, FULL_D, FULL_H, FULL_W)
    status   = "✓ PASS" if tuple(chunk_clean.shape) == expected else "✗ FAIL"
    print(f"\n  Expected : {expected}")
    print(f"  Got      : {tuple(chunk_clean.shape)}")
    print(f"  Status   : {status}")
    print(f"  Output range  : [{chunk_clean.min():.3f}, {chunk_clean.max():.3f}]"
          f"  (Tanh bounds to [-1, 1])")
 
    n_params = sum(p.numel() for p in mf_decoder.parameters())
    print(f"  Parameters    : {n_params:,}")
 
    # ── Motion-Corrupted Decoder ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MOTION-CORRUPTED DECODER")
    print("=" * 60)
 
    mc_decoder = MotionCorruptedDecoder(
        content_ch     = CONTENT_CH,
        artefact_dim   = GLOBAL_DIM,
        spatial_art_ch = SPATIAL_CH,
        out_channels   = OUT_CH,
        n_adain_blocks = 4,
    ).to(device)
 
    with torch.no_grad():
        chunk_corrupt = mc_decoder(content, a_global, a_spatial)
 
    print_shape("Input  content",           content)
    print_shape("Input  a_global",          a_global)
    print_shape("Input  a_spatial",         a_spatial)
    print_shape("Output chunk (corrupted)", chunk_corrupt)
 
    status = "✓ PASS" if tuple(chunk_corrupt.shape) == expected else "✗ FAIL"
    print(f"\n  Expected : {expected}")
    print(f"  Got      : {tuple(chunk_corrupt.shape)}")
    print(f"  Status   : {status}")
    print(f"  Output range  : [{chunk_corrupt.min():.3f}, {chunk_corrupt.max():.3f}]")
 
    n_params = sum(p.numel() for p in mc_decoder.parameters())
    print(f"  Parameters    : {n_params:,}")
 
    # ── Full Translation Paths ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FULL TRANSLATION PATHS")
    print("=" * 60)
 
    # Simulate what the training step does:
    # c_a, a_global, a_spatial come from encoding a corrupted chunk x_a
    # c_b comes from encoding a clean chunk x_b
    c_a       = torch.randn(B, CONTENT_CH, BOT_D, BOT_H, BOT_W, device=device)
    c_b       = torch.randn(B, CONTENT_CH, BOT_D, BOT_H, BOT_W, device=device)
    a_g       = torch.randn(B, GLOBAL_DIM,                       device=device)
    a_s       = torch.randn(B, SPATIAL_CH, BOT_D, BOT_H, BOT_W, device=device)
 
    with torch.no_grad():
        # A → B : corrupted → clean  (main inference path)
        x_hat_b    = mf_decoder(c_a)
 
        # B → A : clean + artefact → corrupted  (synthetic corruption)
        x_hat_a    = mc_decoder(c_b, a_g, a_s)
 
        # Identity paths
        x_self_b   = mf_decoder(c_b)             # clean → clean
        x_self_a   = mc_decoder(c_a, a_g, a_s)   # corrupted → corrupted
 
    print(f"\n  A → B  (corrupted → clean)     x_hat_b  : {tuple(x_hat_b.shape)}")
    print(f"  B → A  (clean → corrupted)     x_hat_a  : {tuple(x_hat_a.shape)}")
    print(f"  Identity B → B                 x_self_b : {tuple(x_self_b.shape)}")
    print(f"  Identity A → A                 x_self_a : {tuple(x_self_a.shape)}")
 
    all_correct = all(
        t.shape == torch.Size([B, OUT_CH, FULL_D, FULL_H, FULL_W])
        for t in [x_hat_b, x_hat_a, x_self_b, x_self_a]
    )
    print(f"\n  All translation outputs correct shape : "
          f"{'✓ PASS' if all_correct else '✗ FAIL'}")
 
    # ── AdaIN behaviour check ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ADAIN BEHAVIOUR CHECK")
    print("=" * 60)
    print("  Verify that different artefact codes produce different outputs")
    print("  from the same content features (AdaIN is doing its job)\n")
 
    a_strong = torch.ones( B, GLOBAL_DIM, device=device) * 5.0   # large code
    a_weak   = torch.zeros(B, GLOBAL_DIM, device=device)          # near-zero code
    a_sp     = torch.zeros(B, SPATIAL_CH, BOT_D, BOT_H, BOT_W, device=device)
 
    with torch.no_grad():
        out_strong = mc_decoder(c_a, a_strong, a_sp)
        out_weak   = mc_decoder(c_a, a_weak,   a_sp)
 
    diff = (out_strong - out_weak).abs().mean().item()
    print(f"  Strong artefact code output std : {out_strong.std():.4f}")
    print(f"  Weak   artefact code output std : {out_weak.std():.4f}")
    print(f"  Mean absolute difference        : {diff:.4f}")
    print(f"  AdaIN modulation working        : "
          f"{'✓ YES' if diff > 0.01 else '✗ NO — check AdaIN'}")
 
    print("\n" + "=" * 60)
    print("  All checks complete.")
    print("=" * 60)
 
 
 
if __name__ == "__main__":
    run_sanity_check_decoders()