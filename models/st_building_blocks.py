"""
st_building_blocks.py
=====================
Factorized R(3+1)D spatiotemporal building blocks.

Each block separates spatial and temporal processing:
    Step A (Spatial)  — existing 3D conv on (B*T, C, D, H, W)
                        T timepoints are treated as independent batch elements.
    Step B (Temporal) — 1D conv along T on (B*D*H*W, C, T)
                        every spatial position mixes information across time.

The two steps are interleaved at every network stage.

Internal tensor convention: (B, T, C, D, H, W)
External interface:         (B, T, D, H, W)      — same as the original model
"""

import torch
import torch.nn as nn
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Core temporal mixer
# ─────────────────────────────────────────────────────────────────────────────

class TemporalConv1D(nn.Module):
    """
    Lightweight 1D temporal convolution for (B, T, C, D, H, W) tensors.

    Reshapes to (B*D*H*W, C, T), applies Conv1d + InstanceNorm1d + LeakyReLU,
    then reshapes back. A residual skip ensures spatial features are preserved.
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size,
            padding=kernel_size // 2, bias=False,
        )
        self.norm = nn.InstanceNorm1d(channels, affine=True)
        self.act  = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, C, D, H, W)"""
        B, T, C, D, H, W = x.shape
        xt = x.permute(0, 3, 4, 5, 2, 1).reshape(B * D * H * W, C, T)
        xt = self.act(self.norm(self.conv(xt)))
        xt = xt.reshape(B, D, H, W, C, T).permute(0, 5, 4, 1, 2, 3)
        return x + xt



# Factorized encoder blocks
# ───────────────────────────────────────------------------------------------------
class FactorizedDownBlock(nn.Module):
    """
    Factorized R(3+1)D strided downsampling block.

    Spatial: (B*T, C_in, D, H, W) → stride-2 Conv3d → (B*T, C_out, D/2, H/2, W/2)
    Temporal (optional): TemporalConv1D on (B, T, C_out, D/2, H/2, W/2)

    use_norm=True   → InstanceNorm3d  (content encoder — removes mean/var info)
    use_norm=False  → BatchNorm3d     (artefact encoder — preserves statistics)
    use_temporal    → False by default: at large spatial resolutions the temporal
                      conv operates on B×D×H×W sequences which is prohibitively
                      expensive. Temporal mixing is reserved for the bottleneck
                      FactorizedResBlocks where spatial dims are 10×12×9.
    """

    def __init__(self, in_ch: int, out_ch: int, use_norm: bool = True,
                 temporal_k: int = 3, use_temporal: bool = False):
        super().__init__()
        layers = [nn.Conv3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not use_norm)]
        if use_norm:
            layers.append(nn.InstanceNorm3d(out_ch, affine=True))
        else:
            layers.append(nn.BatchNorm3d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.spatial      = nn.Sequential(*layers)
        self.use_temporal = use_temporal
        self.temporal     = TemporalConv1D(out_ch, kernel_size=temporal_k) if use_temporal else None

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, C_in, D, H, W)"""
        B, T, C, D, H, W = x.shape
        x = x.reshape(B * T, C, D, H, W)
        x = self.spatial(x)
        _, C_out, D_, H_, W_ = x.shape
        x = x.reshape(B, T, C_out, D_, H_, W_)
        if self.use_temporal:
            x = self.temporal(x)
        return x

class FactorizedResBlock(nn.Module):
    """
    Factorized R(3+1)D residual block.

    Spatial: two 3×3×3 Conv3d layers with InstanceNorm + spatial residual skip.
    Temporal: TemporalConv1D with residual skip.

    Input/Output: (B, T, C, D, H, W)
    """

    def __init__(self, channels: int, temporal_k: int = 3):
        super().__init__()
        self.spatial_block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
        )
        self.act      = nn.LeakyReLU(0.2, inplace=True)
        self.temporal = TemporalConv1D(channels, kernel_size=temporal_k)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, C, D, H, W)"""
        B, T, C, D, H, W = x.shape
        xs = x.reshape(B * T, C, D, H, W)
        xs = self.act(xs + self.spatial_block(xs))
        x  = xs.reshape(B, T, C, D, H, W)
        return self.temporal(x)


# ─────────────────────────────────────────────────────────────────────────────
# Factorized decoder blocks
# ─────────────────────────────────────────────────────────────────────────────

class FactorizedUpBlock(nn.Module):
    """
    Factorized R(3+1)D upsampling block.

    Spatial: trilinear ×2 upsample + Conv3d + InstanceNorm on (B*T, …)
    Temporal (optional): TemporalConv1D — off by default since upsampled spatial
                         dims are large (20×24×18 → 40×48×36 → 80×96×72) making
                         temporal conv expensive. Enable only when needed.

    Input:  (B, T, C_in, D,   H,   W  )
    Output: (B, T, C_out, D*2, H*2, W*2)
    """

    def __init__(self, in_ch: int, out_ch: int, temporal_k: int = 3, use_temporal: bool = False):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.use_temporal = use_temporal
        self.temporal     = TemporalConv1D(out_ch, kernel_size=temporal_k) if use_temporal else None

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, C_in, D, H, W)"""
        B, T, C, D, H, W = x.shape
        x = x.reshape(B * T, C, D, H, W)
        x = self.spatial(x)
        _, C_out, D_, H_, W_ = x.shape
        x = x.reshape(B, T, C_out, D_, H_, W_)
        if self.use_temporal:
            x = self.temporal(x)
        return x


class FactorizedAdaINResBlock(nn.Module):
    """
    Factorized R(3+1)D AdaIN residual block for the MotionCorruptedDecoder.

    a_global (B, T, artefact_dim) carries a per-timepoint motion code so each
    frame is modulated by its own artefact signature.

    Input:  x (B, T, C, D, H, W),  a_global (B, T, artefact_dim)
    Output: (B, T, C, D, H, W)
    """

    def __init__(self, channels: int, artefact_dim: int = 64, temporal_k: int = 3):
        super().__init__()
        from models.building_blocks import AdaIN3D
        self.conv1    = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.adain1   = AdaIN3D(channels, artefact_dim)
        self.act1     = nn.LeakyReLU(0.2, inplace=True)
        self.conv2    = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.adain2   = AdaIN3D(channels, artefact_dim)
        self.act2     = nn.LeakyReLU(0.2, inplace=True)
        self.temporal = TemporalConv1D(channels, kernel_size=temporal_k)

    def forward(self, x: Tensor, a_global: Tensor) -> Tensor:
        """x: (B, T, C, D, H, W),  a_global: (B, T, artefact_dim)"""
        B, T, C, D, H, W = x.shape
        ag = a_global.reshape(B * T, -1)
        xs = x.reshape(B * T, C, D, H, W)
        residual = xs
        xs = self.act1(self.adain1(self.conv1(xs), ag))
        xs = self.adain2(self.conv2(xs), ag)
        xs = self.act2(xs + residual)
        x  = xs.reshape(B, T, C, D, H, W)
        return self.temporal(x)


# ─────────────────────────────────────────────────────────────────────────────
# Factorized discriminator block
# ─────────────────────────────────────────────────────────────────────────────
class FactorizedDiscBlock(nn.Module):
    """
    Factorized R(3+1)D discriminator block.

    Spectral-norm strided Conv3d (spatial) + TemporalConv1D.

    Input:  (B, T, C_in, D, H, W)
    Output: (B, T, C_out, D/2, H/2, W/2)
    """

    def __init__(self, in_ch: int, out_ch: int, use_norm: bool = True,
                 temporal_k: int = 3, use_temporal: bool = False):
        super().__init__()
        conv = nn.utils.spectral_norm(
            nn.Conv3d(in_ch, out_ch, 4, stride=2, padding=1, bias=not use_norm)
        )
        layers = [conv]
        if use_norm:
            layers.append(nn.InstanceNorm3d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.spatial      = nn.Sequential(*layers)
        self.use_temporal = use_temporal
        self.temporal     = TemporalConv1D(out_ch, kernel_size=temporal_k) if use_temporal else None

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, C_in, D, H, W)"""
        B, T, C, D, H, W = x.shape
        x = x.reshape(B * T, C, D, H, W)
        x = self.spatial(x)
        _, C_out, D_, H_, W_ = x.shape
        x = x.reshape(B, T, C_out, D_, H_, W_)
        if self.use_temporal:
            x = self.temporal(x)
        return x
