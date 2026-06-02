#!/usr/bin/env python3
"""
decoders.py
===========
Motion-free and motion-corrupted decoders for the disentangled CycleGAN
fMRI motion artefact correction model.
 
Inputs:
    MotionFreeDecoder
        content   : (B, 384, 10, 12,  9)   ← from ContentEncoder
 
    MotionCorruptedDecoder
        content   : (B, 384, 10, 12,  9)   ← from ContentEncoder
        a_global  : (B, 64)                ← from ArtefactEncoder global branch
        a_spatial : (B,  32, 10, 12,  9)   ← from ArtefactEncoder spatial branch
 
Output (both decoders):
    chunk     : (B,  20, 80, 96, 72)   ← reconstructed 20-volume fMRI chunk
 
Spatial upsampling path (mirrors encoder downsampling):
    (10, 12,  9)  →  (20, 24, 18)  →  (40, 48, 36)  →  (80, 96, 72)
 
Based on:
    MUNIT        (Huang et al. 2018)  — AdaIN residual blocks, no skip connections
    StyleGAN     (Karras et al. 2019) — multi-scale AdaIN style injection
    DRIT++       (Lee et al.   2020)  — spatial style concatenation
    pix2pix      (Isola et al. 2017)  — trilinear upsample avoids checkerboard

"""
import torch
import torch.nn as nn
from building_blocks import ResBlock3D, UpBlock3D, AdaIN3D, AdaINResBlock3D


class MotionFreeDecoder(nn.Module):
    """
    Decodes content features into a motion-free fMRI chunk.

    Receives ONLY content features — no artefact information.

    Input  : content (B, content_ch, D, H, W)
    Output : chunk   (B, out_channels, upsampled_D, upsampled_H, upsampled_W)
    """

    def __init__(
        self,
        content_ch: int = 384,
        out_channels: int = 20,
        n_res_blocks: int = 4,
        up_channel_divisors: tuple[int, int, int] = (2, 4, 8),
        res_kernel_size: int = 3,
        res_padding: int = 1,
        res_bias: bool = False,
        res_affine: bool = True,
        kernel_size: int = 3,
        padding: int = 1,
        up_scale_factor: int = 2,
        up_mode: str = "trilinear",
        up_align_corners: bool = False,
        up_bias: bool = False,
        up_affine: bool = True,
        negative_slope: float = 0.2,
        inplace: bool = True,
        out_kernel_size: int = 3,
        out_padding: int = 1,
        out_bias: bool = True,
        use_tanh: bool = True,
    ):
        """
        Parameters
        ----------
        content_ch          : int              input content feature channels
        out_channels        : int              output channels / timepoints
        n_res_blocks        : int              residual blocks at bottleneck
        up_channel_divisors : tuple[int, int, int]
            channel divisors for progressive upsampling blocks
        res_kernel_size     : int              residual block conv kernel size
        res_padding         : int              residual block conv padding
        res_bias            : bool             residual block conv bias
        res_affine          : bool             residual InstanceNorm affine flag
        kernel_size         : int              upsampling conv kernel size
        padding             : int              upsampling conv padding
        up_scale_factor     : int              upsampling scale factor
        up_mode             : str              upsampling interpolation mode
        up_align_corners    : bool             align_corners for interpolation
        up_bias             : bool             upsampling conv bias
        up_affine           : bool             upsampling InstanceNorm affine flag
        negative_slope      : float            LeakyReLU negative slope
        inplace             : bool             activation inplace flag
        out_kernel_size     : int              output conv kernel size
        out_padding         : int              output conv padding
        out_bias            : bool             output conv bias
        use_tanh            : bool             if True applies Tanh at output
        """
        super().__init__()

        ch1 = content_ch // up_channel_divisors[0]
        ch2 = content_ch // up_channel_divisors[1]
        ch3 = content_ch // up_channel_divisors[2]

        self.res_blocks = nn.Sequential(
            *[
                ResBlock3D(
                    content_ch,
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

        self.up1 = UpBlock3D(
            content_ch,
            ch1,
            scale_factor=up_scale_factor,
            mode=up_mode,
            align_corners=up_align_corners,
            kernel_size=kernel_size,
            padding=padding,
            bias=up_bias,
            affine=up_affine,
            negative_slope=negative_slope,
            inplace=inplace,
        )

        self.up2 = UpBlock3D(
            ch1,
            ch2,
            scale_factor=up_scale_factor,
            mode=up_mode,
            align_corners=up_align_corners,
            kernel_size=kernel_size,
            padding=padding,
            bias=up_bias,
            affine=up_affine,
            negative_slope=negative_slope,
            inplace=inplace,
        )

        self.up3 = UpBlock3D(
            ch2,
            ch3,
            scale_factor=up_scale_factor,
            mode=up_mode,
            align_corners=up_align_corners,
            kernel_size=kernel_size,
            padding=padding,
            bias=up_bias,
            affine=up_affine,
            negative_slope=negative_slope,
            inplace=inplace,
        )

        out_layers = [
            nn.Conv3d(
                ch3,
                out_channels,
                kernel_size=out_kernel_size,
                padding=out_padding,
                bias=out_bias,
            )
        ]

        if use_tanh:
            out_layers.append(nn.Tanh())

        self.out_conv = nn.Sequential(*out_layers)

    def forward(self, content: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        content : (B, content_ch, D, H, W)

        Returns
        -------
        chunk : (B, out_channels, upsampled_D, upsampled_H, upsampled_W)
        """
        x = self.res_blocks(content)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.out_conv(x)
    
    
class MotionCorruptedDecoder(nn.Module):
    """
    Decodes content features + artefact codes into a corrupted fMRI chunk.

    Injects motion corruption into clean content features via:
        1. AdaIN bottleneck residual blocks conditioned on a_global
        2. Spatial artefact concatenation and 1x1x1 merge before upsampling

    Input  : content   (B, content_ch, D, H, W)
             a_global  (B, artefact_dim)
             a_spatial (B, spatial_art_ch, D, H, W)
    Output : chunk     (B, out_channels, upsampled_D, upsampled_H, upsampled_W)
    """

    def __init__(
        self,
        content_ch: int = 384,
        artefact_dim: int = 64,
        spatial_art_ch: int = 32,
        out_channels: int = 20,
        n_adain_blocks: int = 4,
        up_channel_divisors: tuple[int, int, int] = (2, 4, 8),
        adain_kernel_size: int = 3,
        adain_padding: int = 1,
        adain_bias: bool = False,
        adain_negative_slope: float = 0.2,
        adain_inplace: bool = True,
        adain_norm_affine: bool = False,
        adain_linear_bias: bool = True,
        adain_mean_init: float | None = None,
        adain_std_init: float | None = None,
        merge_kernel_size: int = 1,
        merge_bias: bool = False,
        merge_affine: bool = True,
        merge_negative_slope: float = 0.2,
        merge_inplace: bool = True,
        up_scale_factor: int = 2,
        up_mode: str = "trilinear",
        up_align_corners: bool = False,
        up_kernel_size: int = 3,
        up_padding: int = 1,
        up_bias: bool = False,
        up_affine: bool = True,
        up_negative_slope: float = 0.2,
        up_inplace: bool = True,
        out_kernel_size: int = 3,
        out_padding: int = 1,
        out_bias: bool = True,
        use_tanh: bool = True,
    ):
        """
        Parameters
        ----------
        content_ch            : int              content feature channels
        artefact_dim          : int              global artefact code dimension
        spatial_art_ch        : int              spatial artefact map channels
        out_channels          : int              output channels / timepoints
        n_adain_blocks        : int              number of AdaIN residual blocks
        up_channel_divisors   : tuple[int, int, int]
            channel divisors for progressive upsampling blocks
        adain_kernel_size     : int              AdaIN residual conv kernel size
        adain_padding         : int              AdaIN residual conv padding
        adain_bias            : bool             AdaIN residual conv bias
        adain_negative_slope  : float            AdaIN block LeakyReLU negative slope
        adain_inplace         : bool             AdaIN block activation inplace flag
        adain_norm_affine     : bool             affine flag inside AdaIN InstanceNorm
        adain_linear_bias     : bool             bias flag for AdaIN linear layers
        adain_mean_init       : float | None     optional AdaIN mean bias init
        adain_std_init        : float | None     optional AdaIN std bias init
        merge_kernel_size     : int              spatial merge conv kernel size
        merge_bias            : bool             spatial merge conv bias
        merge_affine          : bool             spatial merge InstanceNorm affine flag
        merge_negative_slope  : float            spatial merge LeakyReLU negative slope
        merge_inplace         : bool             spatial merge activation inplace flag
        up_scale_factor       : int              upsampling scale factor
        up_mode               : str              upsampling interpolation mode
        up_align_corners      : bool             align_corners for interpolation
        up_kernel_size        : int              upsampling conv kernel size
        up_padding            : int              upsampling conv padding
        up_bias               : bool             upsampling conv bias
        up_affine             : bool             upsampling InstanceNorm affine flag
        up_negative_slope     : float            upsampling LeakyReLU negative slope
        up_inplace            : bool             upsampling activation inplace flag
        out_kernel_size       : int              output conv kernel size
        out_padding           : int              output conv padding
        out_bias              : bool             output conv bias
        use_tanh              : bool             if True applies Tanh at output
        """
        super().__init__()

        ch1 = content_ch // up_channel_divisors[0]
        ch2 = content_ch // up_channel_divisors[1]
        ch3 = content_ch // up_channel_divisors[2]

        self.adain_blocks = nn.ModuleList(
            [
                AdaINResBlock3D(
                    content_ch,
                    artefact_dim=artefact_dim,
                    kernel_size=adain_kernel_size,
                    padding=adain_padding,
                    bias=adain_bias,
                    negative_slope=adain_negative_slope,
                    inplace=adain_inplace,
                    adain_affine=adain_norm_affine,
                    adain_bias=adain_linear_bias,
                    mean_init=adain_mean_init,
                    std_init=adain_std_init,
                )
                for _ in range(n_adain_blocks)
            ]
        )

        self.spatial_merge = nn.Sequential(
            nn.Conv3d(
                content_ch + spatial_art_ch,
                content_ch,
                kernel_size=merge_kernel_size,
                bias=merge_bias,
            ),
            nn.InstanceNorm3d(content_ch, affine=merge_affine),
            nn.LeakyReLU(merge_negative_slope, inplace=merge_inplace),
        )

        self.up1 = UpBlock3D(
            content_ch,
            ch1,
            scale_factor=up_scale_factor,
            mode=up_mode,
            align_corners=up_align_corners,
            kernel_size=up_kernel_size,
            padding=up_padding,
            bias=up_bias,
            affine=up_affine,
            negative_slope=up_negative_slope,
            inplace=up_inplace,
        )

        self.up2 = UpBlock3D(
            ch1,
            ch2,
            scale_factor=up_scale_factor,
            mode=up_mode,
            align_corners=up_align_corners,
            kernel_size=up_kernel_size,
            padding=up_padding,
            bias=up_bias,
            affine=up_affine,
            negative_slope=up_negative_slope,
            inplace=up_inplace,
        )

        self.up3 = UpBlock3D(
            ch2,
            ch3,
            scale_factor=up_scale_factor,
            mode=up_mode,
            align_corners=up_align_corners,
            kernel_size=up_kernel_size,
            padding=up_padding,
            bias=up_bias,
            affine=up_affine,
            negative_slope=up_negative_slope,
            inplace=up_inplace,
        )

        out_layers = [
            nn.Conv3d(
                ch3,
                out_channels,
                kernel_size=out_kernel_size,
                padding=out_padding,
                bias=out_bias,
            )
        ]

        if use_tanh:
            out_layers.append(nn.Tanh())

        self.out_conv = nn.Sequential(*out_layers)

    def forward(
        self,
        content: torch.Tensor,
        a_global: torch.Tensor,
        a_spatial: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        content   : (B, content_ch, D, H, W)
        a_global  : (B, artefact_dim)
        a_spatial : (B, spatial_art_ch, D, H, W)

        Returns
        -------
        chunk : (B, out_channels, upsampled_D, upsampled_H, upsampled_W)
        """
        x = content

        for block in self.adain_blocks:
            x = block(x, a_global)

        x = torch.cat([x, a_spatial], dim=1)
        x = self.spatial_merge(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        return self.out_conv(x)