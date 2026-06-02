#!/usr/bin/env python3
"""
encoders.py
===========
Content and Artefact encoders for the disentangled CycleGAN fMRI
motion artefact correction model.
 
Input  : (B, 20, 80, 96, 72)  — 20-volume fMRI chunk, spatial dims downsampled
Outputs:
    ContentEncoder  → content feature map  : (B, 384, 10, 12, 9)
    ArtefactEncoder → global artefact code : (B, 64)
                    → spatial artefact map : (B, 32, 10, 12, 9)
 
Based on:
    MUNIT   (Huang et al. 2018)  — content/style split, IN in content encoder
    DRIT++  (Lee et al.   2020)  — dual-branch artefact encoder
    StyleGAN(Karras et al.2019)  — MLP mapping for compact style code
    V-Net   (Milletari et al.2016) — 3D strided conv design
    Dr-CycleGAN (Dewey et al. 2020) — medical image disentanglement
"""

import torch
import torch.nn as nn
from building_blocks import ResBlock3D, StridedConvBlock

class ContentEncoder(nn.Module):
    """
    Extracts motion-invariant BOLD content features from a 4D fMRI chunk.

    Architecture (MUNIT-style, extended to 3D):
        3 strided conv blocks  → spatial downsampling ×8
        5 residual blocks      → feature refinement at bottleneck

    Instance Normalisation throughout — IN removes per-channel mean and
    variance, discarding intensity/style information and keeping only
    structural/spatial content. This is the key property that makes the
    content code motion-invariant (MUNIT, Huang et al. 2018). Here we 
    treat the 20 timepoints as channels. 

    Input  : (B, 20,  80, 96, 72)   — 20 timepoints as channels
    Output : (B, 384, 10, 12,  9)   — spatial content feature map

    Spatial path:
        (80, 96, 72) ----->   input spatial dimensions
      → (40, 48, 36) ----->   after block 1 x2 downsampling
      → (20, 24, 18) ----->   after block 2 x2 downsampling
      → (10, 12,  9) ----->   after block 3 x2 downsampling  ← bottleneck resolution
    """

    def __init__(
        self,
        in_channels: int = 20,
        base_channels: int = 64,
        n_res_blocks: int = 5,
        down_channel_multipliers: tuple[int, int, int] = (1, 2, 6),
        use_norm: bool = True,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        instance_affine: bool = True,
        negative_slope: float = 0.2,
        inplace: bool = True,
        res_kernel_size: int = 3,
        res_padding: int = 1,
        res_bias: bool = False,
        res_affine: bool = True,
    ):
        """
        Parameters
        ----------
        in_channels              : int              number of input timepoints
        base_channels            : int              channels after first strided conv
        n_res_blocks             : int              residual blocks at bottleneck
        down_channel_multipliers : tuple[int, int, int]
            channel multipliers for the three downsampling blocks
        use_norm                 : bool             if True uses InstanceNorm
        kernel_size              : int              strided conv kernel size
        stride                   : int              strided conv stride
        padding                  : int              strided conv padding
        instance_affine          : bool             affine parameter for InstanceNorm3d
        negative_slope           : float            LeakyReLU negative slope
        inplace                  : bool             LeakyReLU inplace flag
        res_kernel_size          : int              residual block conv kernel size
        res_padding              : int              residual block conv padding
        res_bias                 : bool             residual block conv bias
        res_affine               : bool             affine parameter for residual InstanceNorm3d
        """
        super().__init__()

        c = base_channels

        c1 = c * down_channel_multipliers[0]
        c2 = c * down_channel_multipliers[1]
        c3 = c * down_channel_multipliers[2]

        self.down1 = StridedConvBlock(
            in_channels,
            c1,
            use_norm=use_norm,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            instance_affine=instance_affine,
            negative_slope=negative_slope,
            inplace=inplace,
        )

        self.down2 = StridedConvBlock(
            c1,
            c2,
            use_norm=use_norm,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            instance_affine=instance_affine,
            negative_slope=negative_slope,
            inplace=inplace,
        )

        self.down3 = StridedConvBlock(
            c2,
            c3,
            use_norm=use_norm,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            instance_affine=instance_affine,
            negative_slope=negative_slope,
            inplace=inplace,
        )

        self.res_blocks = nn.Sequential(
            *[
                ResBlock3D(
                    c3,
                    kernel_size=res_kernel_size,
                    padding=res_padding,
                    bias=res_bias,
                    negative_slope=negative_slope,
                    affine=res_affine,
                    inplace=inplace,
                )
                for _ in range(n_res_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 20, 80, 96, 72)

        Returns
        -------
        content : (B, 384, 10, 12, 9)
        """
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.res_blocks(x)
        return x
    
    
# Artefact Encoder with dual global and spatial branches 
class ArtefactEncoder(nn.Module):
    """
    Extracts motion artefact characteristics from a 4D fMRI chunk.

    Architecture (DRIT++ / StyleGAN-inspired, extended to 3D):
        3 strided conv blocks  → spatial downsampling ×8  (no InstanceNorm)
        Global branch          → GlobalAvgPool → MLP → compact code (B, 64)
        Spatial branch         → 1×1×1 conv → spatial map (B, 32, 10, 12, 9)

    NO Instance Normalisation in the conv blocks — IN removes per-channel
    mean and variance, which are the primary image-domain signatures of
    motion artefacts.

    Input  : (B, 20,  80, 96, 72)
    Output : a_global  (B, 64)              global motion severity code
             a_spatial (B, 32, 10, 12,  9)  spatially varying artefact map
    """

    def __init__(
        self,
        in_channels: int = 20,
        base_channels: int = 64,
        global_code_dim: int = 64,
        spatial_code_ch: int = 32,
        down_channel_multipliers: tuple[int, int, int] = (1, 2, 4),
        use_norm: bool = False,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        instance_affine: bool = True,
        negative_slope: float = 0.2,
        inplace: bool = True,
        mlp_hidden_multiplier: int = 2,
        spatial_kernel_size: int = 1,
        spatial_bias: bool = True,
    ):
        """
        Parameters
        ----------
        in_channels              : int              number of input timepoints
        base_channels            : int              channels after first strided conv
        global_code_dim          : int              dimension of global artefact code
        spatial_code_ch          : int              channels of spatial artefact map
        down_channel_multipliers : tuple[int, int, int]
            channel multipliers for the three downsampling blocks
        use_norm                 : bool             if False uses BatchNorm in StridedConvBlock
        kernel_size              : int              strided conv kernel size
        stride                   : int              strided conv stride
        padding                  : int              strided conv padding
        instance_affine          : bool             affine parameter if InstanceNorm is used
        negative_slope           : float            LeakyReLU negative slope
        inplace                  : bool             activation inplace flag
        mlp_hidden_multiplier    : int              hidden MLP channel multiplier
        spatial_kernel_size      : int              spatial branch conv kernel size
        spatial_bias             : bool             spatial branch conv bias
        """
        super().__init__()

        c = base_channels

        c1 = c * down_channel_multipliers[0]
        c2 = c * down_channel_multipliers[1]
        c3 = c * down_channel_multipliers[2]

        self.down1 = StridedConvBlock(
            in_channels,
            c1,
            use_norm=use_norm,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            instance_affine=instance_affine,
            negative_slope=negative_slope,
            inplace=inplace,
        )

        self.down2 = StridedConvBlock(
            c1,
            c2,
            use_norm=use_norm,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            instance_affine=instance_affine,
            negative_slope=negative_slope,
            inplace=inplace,
        )

        self.down3 = StridedConvBlock(
            c2,
            c3,
            use_norm=use_norm,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            instance_affine=instance_affine,
            negative_slope=negative_slope,
            inplace=inplace,
        )

        mlp_hidden_dim = c * mlp_hidden_multiplier

        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self.global_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, mlp_hidden_dim),
            nn.ReLU(inplace=inplace),
            nn.Linear(mlp_hidden_dim, global_code_dim),
        )

        self.spatial_branch = nn.Sequential(
            nn.Conv3d(
                c3,
                spatial_code_ch,
                kernel_size=spatial_kernel_size,
                bias=spatial_bias,
            ),
            nn.LeakyReLU(negative_slope, inplace=inplace),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 20, 80, 96, 72)

        Returns
        -------
        a_global  : (B, 64)              global artefact code
        a_spatial : (B, 32, 10, 12,  9)  spatial artefact map
        """
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        a_global = self.global_pool(x)
        a_global = self.global_mlp(a_global)

        a_spatial = self.spatial_branch(x)

        return a_global, a_spatial


# Test the ContentEncoder with a dummy input
if __name__ == "__main__":
    # Test the ContentEncoder with a dummy input
    encoder = ContentEncoder()
    dummy_input = torch.randn(4, 20, 80, 96, 72)  # (B=4, C=20, D=80, H=96, W=72)
    output = encoder(dummy_input)
    print("ContentEncoder output shape:", output.shape)