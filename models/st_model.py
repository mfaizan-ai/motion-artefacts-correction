"""
st_model.py
===========
Factorized R(3+1)D Disentangled CycleGAN for fMRI motion artefact correction.

Drop-in replacement for DisentangledCycleGAN (same ModelOutputs, same
forward/correct signatures).  The only difference is internal: every
encoder/decoder/discriminator stage now interleaves a 3D spatial convolution
(on B*T frames flattened into the batch axis) with a 1D temporal convolution
(mixing T timepoints at every spatial location).

This lets the model explicitly reason about cross-timepoint dynamics at every
resolution scale, instead of treating T as opaque channel indices.

Key shapes
----------
External:   (B, T, D, H, W)           — identical to original model
Internal:   (B, T, C, D, H, W)        — T kept as a separate dimension
Per-volume: (B*T, C, D, H, W)         — flattened for 3D spatial ops
Temporal:   (B*D*H*W, C, T)           — flattened for 1D temporal ops
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List

from models.st_building_blocks import (
    FactorizedDownBlock,
    FactorizedResBlock,
    FactorizedUpBlock,
    FactorizedAdaINResBlock,
    FactorizedDiscBlock,
)
from models.model import ModelOutputs


# ─────────────────────────────────────────────────────────────────────────────
# Encoders
# ─────────────────────────────────────────────────────────────────────────────

class STContentEncoder(nn.Module):
    """
    Spatiotemporal content encoder.

    Each fMRI volume in a chunk is treated as a single-channel (C=1) 3D image.
    Three FactorizedDownBlocks halve spatial resolution at each stage.
    Five FactorizedResBlocks process the bottleneck with temporal mixing.
    InstanceNorm is used throughout to strip mean/variance (content-only).

    Input  : (B, T, D, H, W)
    Output : (B, T, 384, D/8, H/8, W/8)
    """

    def __init__(self, base_ch: int = 64, n_res: int = 5, temporal_k: int = 3):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 6  # 64, 128, 384
        self.down1 = FactorizedDownBlock(1,  c1, use_norm=True, temporal_k=temporal_k)
        self.down2 = FactorizedDownBlock(c1, c2, use_norm=True, temporal_k=temporal_k)
        self.down3 = FactorizedDownBlock(c2, c3, use_norm=True, temporal_k=temporal_k)
        self.res   = nn.ModuleList([
            FactorizedResBlock(c3, temporal_k=temporal_k) for _ in range(n_res)
        ])

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, D, H, W)  →  (B, T, 384, D/8, H/8, W/8)"""
        x = x.unsqueeze(2)          # (B, T, 1, D, H, W)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        for blk in self.res:
            x = blk(x)
        return x


class STArtefactEncoder(nn.Module):
    """
    Spatiotemporal artefact encoder.

    Same spatial structure as STContentEncoder but uses BatchNorm instead of
    InstanceNorm to preserve intensity/variance statistics (the artefact signal).
    After spatial feature extraction, T is collapsed by mean-pooling so that
    a_global and a_spatial summarise the motion pattern across the whole chunk.

    Input  : (B, T, D, H, W)
    Output : a_global   (B, T, global_code_dim)
             a_spatial  (B, T, spatial_code_ch, D/8, H/8, W/8)
    """

    def __init__(
        self,
        base_ch:         int = 64,
        global_code_dim: int = 64,
        spatial_code_ch: int = 32,
        temporal_k:      int = 3,
    ):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4  # 64, 128, 256
        # use_norm=False → BatchNorm (preserves mean/var for artefact detection)
        # use_temporal=False → each TR encoded independently, no cross-TR contamination
        self.down1 = FactorizedDownBlock(1,  c1, use_norm=False, temporal_k=temporal_k, use_temporal=False)
        self.down2 = FactorizedDownBlock(c1, c2, use_norm=False, temporal_k=temporal_k, use_temporal=False)
        self.down3 = FactorizedDownBlock(c2, c3, use_norm=False, temporal_k=temporal_k, use_temporal=False)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.global_mlp  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, base_ch * 2),
            nn.ReLU(inplace=True),
            nn.Linear(base_ch * 2, global_code_dim),
        )
        self.spatial_branch = nn.Sequential(
            nn.Conv3d(c3, spatial_code_ch, kernel_size=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: Tensor):
        """x: (B, T, D, H, W)  →  a_global (B, T, 64), a_spatial (B, T, 32, D', H', W')"""
        B, T = x.shape[:2]
        x = x.unsqueeze(2)          # (B, T, 1, D, H, W)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)           # (B, T, C3, D', H', W')
        _, _, C3, D_, H_, W_ = x.shape
        # Apply heads per-timepoint so each TR gets its own artefact code
        x_bt = x.reshape(B * T, C3, D_, H_, W_)
        a_global  = self.global_mlp(self.global_pool(x_bt)).reshape(B, T, -1)          # (B, T, 64)
        a_spatial = self.spatial_branch(x_bt).reshape(B, T, -1, D_, H_, W_)           # (B, T, 32, D', H', W')
        return a_global, a_spatial


# ─────────────────────────────────────────────────────────────────────────────
# Decoders
# ─────────────────────────────────────────────────────────────────────────────
class STMotionFreeDecoder(nn.Module):
    """
    Spatiotemporal motion-free decoder (G_B).

    Takes per-timepoint content features and reconstructs T independent but
    temporally-mixed fMRI volumes.  No artefact conditioning.

    Input  : (B, T, 384, D', H', W')
    Output : (B, T, D, H, W)
    """

    def __init__(self, content_ch: int = 384, n_res: int = 4, temporal_k: int = 3):
        super().__init__()
        ch1 = content_ch // 2   # 192
        ch2 = content_ch // 4   #  96
        ch3 = content_ch // 8   #  48

        self.res  = nn.ModuleList([
            FactorizedResBlock(content_ch, temporal_k=temporal_k) for _ in range(n_res)
        ])
        self.up1  = FactorizedUpBlock(content_ch, ch1, temporal_k=temporal_k)
        self.up2  = FactorizedUpBlock(ch1,        ch2, temporal_k=temporal_k)
        self.up3  = FactorizedUpBlock(ch2,        ch3, temporal_k=temporal_k)
        # Collapse C_out=48 → 1 channel per volume, then squeeze to (B, T, D, H, W)
        self.out_conv = nn.Sequential(
            nn.Conv3d(ch3, 1, kernel_size=3, padding=1, bias=True),
            nn.Tanh(),
        )

    def forward(self, content: Tensor) -> Tensor:
        """content: (B, T, 384, D', H', W')  →  (B, T, D, H, W)"""
        x = content
        for blk in self.res:
            x = blk(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)                             # (B, T, 48, D, H, W)
        B, T, C, D, H, W = x.shape
        x = x.reshape(B * T, C, D, H, W)
        x = self.out_conv(x)                        # (B*T, 1, D, H, W)
        return x.squeeze(1).reshape(B, T, D, H, W)

class STMotionCorruptedDecoder(nn.Module):
    """
    Spatiotemporal motion-corrupted decoder (G_A).

    Injects per-timepoint artefact codes into content features:
        1. FactorizedAdaINResBlocks conditioned on a_global (per-timepoint)
        2. Spatial merge with a_spatial (per-timepoint)
        3. Factorized upsampling back to full resolution

    Input  : content   (B, T, 384, D', H', W')
             a_global  (B, T, 64)
             a_spatial (B, T, 32, D', H', W')
    Output : (B, T, D, H, W)
    """

    def __init__(
        self,
        content_ch:     int = 384,
        artefact_dim:   int = 64,
        spatial_art_ch: int = 32,
        n_adain:        int = 4,
        temporal_k:     int = 3,
    ):
        super().__init__()
        ch1 = content_ch // 2
        ch2 = content_ch // 4
        ch3 = content_ch // 8

        self.adain_blocks = nn.ModuleList([
            FactorizedAdaINResBlock(content_ch, artefact_dim, temporal_k=temporal_k)
            for _ in range(n_adain)
        ])
        # 1×1×1 conv to fuse content (384ch) + spatial artefact (32ch) → 384ch
        self.spatial_merge = nn.Sequential(
            nn.Conv3d(content_ch + spatial_art_ch, content_ch, 1, bias=False),
            nn.InstanceNorm3d(content_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.up1 = FactorizedUpBlock(content_ch, ch1, temporal_k=temporal_k)
        self.up2 = FactorizedUpBlock(ch1,        ch2, temporal_k=temporal_k)
        self.up3 = FactorizedUpBlock(ch2,        ch3, temporal_k=temporal_k)
        self.out_conv = nn.Sequential(
            nn.Conv3d(ch3, 1, 3, padding=1, bias=True),
            nn.Tanh(),
        )

    def forward(self, content: Tensor, a_global: Tensor, a_spatial: Tensor) -> Tensor:
        """
        content  : (B, T, 384, D', H', W')
        a_global : (B, T, 64)
        a_spatial: (B, T, 32, D', H', W')
        """
        B, T = content.shape[:2]
        _, _, _, D_, H_, W_ = a_spatial.shape

        x = content
        for blk in self.adain_blocks:
            x = blk(x, a_global)               # (B, T, 384, D', H', W')

        # a_spatial is per-timepoint — reshape directly, no expand needed
        x_bt  = x.reshape(B * T, -1, D_, H_, W_)
        a_bt  = a_spatial.reshape(B * T, -1, D_, H_, W_)
        x_bt  = self.spatial_merge(torch.cat([x_bt, a_bt], dim=1))
        x     = x_bt.reshape(B, T, -1, D_, H_, W_)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)                         # (B, T, 48, D, H, W)
        B, T, C, D, H, W = x.shape
        x = x.reshape(B * T, C, D, H, W)
        x = self.out_conv(x)
        return x.squeeze(1).reshape(B, T, D, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Discriminator
# ─────────────────────────────────────────────────────────────────────────────
class _STScaleCNN(nn.Module):
    """Single-scale factorized PatchGAN CNN.

    Processes (B, T, 1, D, H, W) through four FactorizedDiscBlocks, then
    applies a final spectral-norm Conv3d per timepoint and averages the
    resulting patch scores across T → (B, 1, d, h, w).

    Temporal mixing (TemporalConv1D) is enabled only on the last (coarsest,
    cheapest) block -- same cost-driven placement principle as the generator's
    bottleneck-only temporal blocks. The first three blocks run at much higher
    spatial resolution, where a temporal conv over B*D*H*W sequences would be
    far more expensive for comparatively little benefit.
    """

    def __init__(self, base_ch: int, temporal_k: int):
        super().__init__()
        c = base_ch
        self.blocks = nn.ModuleList([
            FactorizedDiscBlock(1,      c,      use_norm=False, temporal_k=temporal_k),
            FactorizedDiscBlock(c,      c * 2,  use_norm=True,  temporal_k=temporal_k),
            FactorizedDiscBlock(c * 2,  c * 4,  use_norm=True,  temporal_k=temporal_k),
            FactorizedDiscBlock(c * 4,  c * 8,  use_norm=True,  temporal_k=temporal_k, use_temporal=True),
        ])
        self.out_conv = nn.utils.spectral_norm(
            nn.Conv3d(c * 8, 1, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, 1, D, H, W)  →  (B, 1, d, h, w)"""
        for blk in self.blocks:
            x = blk(x)                             # (B, T, C, D', H', W')
        B, T, C, D, H, W = x.shape
        x = x.reshape(B * T, C, D, H, W)
        x = self.out_conv(x)                       # (B*T, 1, d, h, w)
        _, _, d, h, w = x.shape
        x = x.reshape(B, T, 1, d, h, w)
        return x.mean(dim=1)                        # (B, 1, d, h, w)


class STMultiScaleDiscriminator(nn.Module):
    """
    Spatiotemporal multi-scale PatchGAN discriminator.

    Runs num_scales independent factorized CNNs at progressively coarser
    resolutions (AvgPool3d between scales). Returns a list of score maps
    (finest first) — compatible with the existing multi-scale LSGAN losses.

    Input : (B, T, D, H, W)
    Output: list of (B, 1, d, h, w) score tensors, finest first
    """

    def __init__(self, base_ch: int = 64, num_scales: int = 2, temporal_k: int = 3):
        super().__init__()
        self.num_scales = num_scales
        self.downsample = nn.AvgPool3d(3, stride=2, padding=1, count_include_pad=False)
        self.cnns = nn.ModuleList([
            _STScaleCNN(base_ch, temporal_k) for _ in range(num_scales)
        ])

    def forward(self, x: Tensor) -> List[Tensor]:
        """x: (B, T, D, H, W)  →  list of (B, 1, d, h, w)"""
        B, T, D, H, W = x.shape
        x_in = x.unsqueeze(2)       # (B, T, 1, D, H, W)
        outputs = []
        for i, cnn in enumerate(self.cnns):
            outputs.append(cnn(x_in))
            if i < self.num_scales - 1:
                # Spatial downsample, keeping T intact
                x_sp = x_in.reshape(B * T, 1, D, H, W)
                x_sp = self.downsample(x_sp)
                _, _, D, H, W = x_sp.shape
                x_in = x_sp.reshape(B, T, 1, D, H, W)
        return outputs


class SpatioTemporalCycleGAN(nn.Module):
    """
    Factorized R(3+1)D Disentangled CycleGAN.

    Drop-in replacement for DisentangledCycleGAN.  Same forward(x_a, x_b)
    signature, same ModelOutputs, same correct(x_a) inference path.

    Parameters
    ----------
    in_timepoints   : int   T (chunks of T fMRI volumes per sample)
    spatial_dims    : tuple input spatial shape (D, H, W)
    content_base_ch : int   content encoder first-conv channels (default 64)
    content_n_res   : int   content encoder residual blocks (default 5)
    artefact_base_ch: int   artefact encoder base channels (default 64)
    global_code_dim : int   global artefact code dimension (default 64)
    spatial_code_ch : int   spatial artefact map channels (default 32)
    disc_base_ch    : int   discriminator base channels (default 64)
    num_disc_scales : int   multi-scale discriminator scales (default 2)
    temporal_k      : int   temporal Conv1d kernel size (default 3)
    residual        : bool  decoder predicts delta from input (default False)
    """

    def __init__(
        self,
        in_timepoints:   int   = 5,
        spatial_dims:    tuple = (80, 96, 72),
        content_base_ch: int   = 64,
        content_n_res:   int   = 5,
        artefact_base_ch: int  = 64,
        global_code_dim: int   = 64,
        spatial_code_ch: int   = 32,
        disc_base_ch:    int   = 64,
        num_disc_scales: int   = 2,
        temporal_k:      int   = 3,
        residual:        bool  = False,
    ):
        super().__init__()
        self.in_timepoints = in_timepoints
        self.spatial_dims  = spatial_dims
        self.residual      = residual
        content_ch = content_base_ch * 6   # 384

        self.E_c = STContentEncoder(
            base_ch=content_base_ch, n_res=content_n_res, temporal_k=temporal_k,
        )
        self.E_a = STArtefactEncoder(
            base_ch=artefact_base_ch,
            global_code_dim=global_code_dim,
            spatial_code_ch=spatial_code_ch,
            temporal_k=temporal_k,
        )
        self.G_B = STMotionFreeDecoder(
            content_ch=content_ch, n_res=4, temporal_k=temporal_k,
        )
        self.G_A = STMotionCorruptedDecoder(
            content_ch=content_ch,
            artefact_dim=global_code_dim,
            spatial_art_ch=spatial_code_ch,
            n_adain=4,
            temporal_k=temporal_k,
        )
        self.D_B = STMultiScaleDiscriminator(
            base_ch=disc_base_ch, num_scales=num_disc_scales, temporal_k=temporal_k,
        )
        self.D_A = STMultiScaleDiscriminator(
            base_ch=disc_base_ch, num_scales=num_disc_scales, temporal_k=temporal_k,
        )

    def generator_parameters(self):
        return (list(self.E_c.parameters()) + list(self.E_a.parameters()) +
                list(self.G_B.parameters()) + list(self.G_A.parameters()))

    def discriminator_parameters(self):
        return list(self.D_A.parameters()) + list(self.D_B.parameters())

    def encode(self, x: Tensor):
        content            = self.E_c(x)
        a_global, a_spatial = self.E_a(x)
        return content, a_global, a_spatial

    @staticmethod
    def _masked_residual(base: Tensor, delta: Tensor) -> Tensor:
        """
        base + delta, with delta zeroed outside base's brain mask (base != 0).

        Without this, the residual correction is added everywhere, including background --
        the decoder has no constraint keeping delta at 0 there, so background (exactly 0 in
        every real chunk, by construction of the normalization) picks up whatever the decoder
        happens to output. Masking forces background to stay exactly 0 through every residual
        connection, matching the real data instead of relying on training to approximate it.
        """
        return base + delta * (base != 0)

    def forward(
        self,
        x_a: Tensor,
        x_b: Tensor,
        detach_fakes_for_D: bool = False,
    ) -> ModelOutputs:
        # PHASE 1 — ENCODE
        c_a, a_global,   a_spatial   = self.encode(x_a)
        c_b, a_global_b, a_spatial_b = self.encode(x_b)

        # PHASE 2 — TRANSLATE + SELF-RECONSTRUCT
        x_hat_b  = self.G_B(c_a)
        x_hat_a  = self.G_A(c_b, a_global, a_spatial)
        x_self_a = self.G_A(c_a, a_global, a_spatial)
        x_self_b = self.G_B(c_b)

        if self.residual:
            x_hat_b  = self._masked_residual(x_a, x_hat_b)
            x_hat_a  = self._masked_residual(x_b, x_hat_a)
            x_self_a = self._masked_residual(x_a, x_self_a)
            x_self_b = self._masked_residual(x_b, x_self_b)

        # PHASE 3 — CYCLIC RE-ENCODING
        c_hat_b = self.E_c(x_hat_b)
        c_hat_a, a_hat_global, a_hat_spatial = self.encode(x_hat_a)

        x_cycle_a = self.G_A(c_hat_b, a_hat_global, a_hat_spatial)
        x_cycle_b = self.G_B(c_hat_a)

        if self.residual:
            x_cycle_a = self._masked_residual(x_hat_b, x_cycle_a)
            x_cycle_b = self._masked_residual(x_hat_a, x_cycle_b)

        # DISCRIMINATOR SCORES
        fake_b = x_hat_b.detach() if detach_fakes_for_D else x_hat_b
        fake_a = x_hat_a.detach() if detach_fakes_for_D else x_hat_a

        score_real_b = self.D_B(x_b)
        score_fake_b = self.D_B(fake_b)
        score_real_a = self.D_A(x_a)
        score_fake_a = self.D_A(fake_a)

        return ModelOutputs(
            c_a=c_a,             c_b=c_b,
            a_global=a_global,   a_spatial=a_spatial,
            a_global_b=a_global_b, a_spatial_b=a_spatial_b,
            x_hat_b=x_hat_b,     x_hat_a=x_hat_a,
            x_self_a=x_self_a,   x_self_b=x_self_b,
            c_hat_b=c_hat_b,     c_hat_a=c_hat_a,
            a_hat_global=a_hat_global, a_hat_spatial=a_hat_spatial,
            x_cycle_a=x_cycle_a, x_cycle_b=x_cycle_b,
            score_real_b=score_real_b, score_fake_b=score_fake_b,
            score_real_a=score_real_a, score_fake_a=score_fake_a,
        )

    def correct(self, x_a: Tensor) -> Tensor:
        """Inference-only: E_c → G_B, no artefact encoder."""
        c_a = self.E_c(x_a)
        out = self.G_B(c_a)
        if self.residual:
            out = self._masked_residual(x_a, out)
        return out
    

    def count_parameters(self) -> dict:
        def n(m): return sum(p.numel() for p in m.parameters())
        return {
            "E_c (STContentEncoder)":          n(self.E_c),
            "E_a (STArtefactEncoder)":         n(self.E_a),
            "G_B (STMotionFreeDecoder)":       n(self.G_B),
            "G_A (STMotionCorruptedDecoder)":  n(self.G_A),
            "D_B (STMultiScaleDisc)":          n(self.D_B),
            "D_A (STMultiScaleDisc)":          n(self.D_A),
            "Total":                            n(self),
        }