#!/usr/bin/env python3
"""
building_blocks.py
==================

a torch module containing building blocks for the model, such as convolutional layers, activation functions, and normalization layers.
"""
import torch
import torch.nn as nn


# 3D Residual Block
class ResBlock3D(nn.Module):
    """
    3D Residual block with Instance Normalisation.

    Used in the ContentEncoder bottleneck.
    Preserves feature map dimensions — no spatial change.

    Input / Output: (B, channels, D, H, W)
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = False,
        negative_slope: float = 0.2,
        affine: bool = True,
        inplace: bool = True,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
            nn.InstanceNorm3d(channels, affine=affine),
            nn.LeakyReLU(negative_slope, inplace=inplace),
            nn.Conv3d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
            nn.InstanceNorm3d(channels, affine=affine),
        )

        self.activation = nn.LeakyReLU(negative_slope, inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))
    
# 3D Strided Convolution Block 
class StridedConvBlock(nn.Module):
    """
    3D strided convolution block for spatial downsampling.

    Each block halves spatial dimensions:
        (D, H, W) → (D/2, H/2, W/2)

    Parameters
    ----------
    in_ch       : int   input channels
    out_ch      : int   output channels
    use_norm    : bool  if True uses InstanceNorm (content encoder)
                        if False uses BatchNorm (artefact encoder)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        use_norm: bool = True,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        instance_affine: bool = True,
        negative_slope: float = 0.2,
        inplace: bool = True,
    ):
        super().__init__()

        layers = [
            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not use_norm,
            ),
        ]

        if use_norm:
            # InstanceNorm: normalises per sample per channel
            # Removes mean/variance info → encourages content-only encoding
            layers.append(nn.InstanceNorm3d(out_ch, affine=instance_affine))
        else:
            # BatchNorm: preserves cross-sample statistics
            # Keeps mean/variance info → artefact encoder retains
            # intensity shift and variance spike signatures of motion
            layers.append(nn.BatchNorm3d(out_ch))

        layers.append(nn.LeakyReLU(negative_slope, inplace=inplace))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    
    
class UpBlock3D(nn.Module):
    """
    3D upsampling block: interpolation + Conv + InstanceNorm + LReLU.

    Each block increases spatial dimensions according to scale_factor:
        (D, H, W) → (scale_factor*D, scale_factor*H, scale_factor*W)

    Input / Output channels are configurable.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        scale_factor: int = 2,
        mode: str = "trilinear",
        align_corners: bool = False,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = False,
        affine: bool = True,
        negative_slope: float = 0.2,
        inplace: bool = True,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Upsample(
                scale_factor=scale_factor,
                mode=mode,
                align_corners=align_corners,
            ),
            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
            nn.InstanceNorm3d(out_ch, affine=affine),
            nn.LeakyReLU(negative_slope, inplace=inplace),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    
    
class AdaIN3D(nn.Module):
    """
    Adaptive Instance Normalisation for 3D feature maps.

    Replaces fixed IN affine parameters with artefact-code-conditioned ones.

        AdaIN(x, a) = σ(a) · InstanceNorm(x) + μ(a)

    Parameters
    ----------
    channels      : int   number of feature map channels to modulate
    artefact_dim  : int   dimension of the global artefact code
    """

    def __init__(
        self,
        channels: int,
        artefact_dim: int = 64,
        affine: bool = False,
        bias: bool = True,
        mean_init: float | None = None,
        std_init: float | None = None,
    ):
        super().__init__()

        self.norm = nn.InstanceNorm3d(channels, affine=affine)
        self.mlp_mean = nn.Linear(artefact_dim, channels, bias=bias)
        self.mlp_std = nn.Linear(artefact_dim, channels, bias=bias)

        if mean_init is not None:
            nn.init.constant_(self.mlp_mean.bias, mean_init)

        if std_init is not None:
            nn.init.constant_(self.mlp_std.bias, std_init)

    def forward(
        self,
        x: torch.Tensor,
        a_global: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x        : (B, C, D, H, W)   feature map to modulate
        a_global : (B, artefact_dim) global artefact code

        Returns
        -------
        (B, C, D, H, W)  modulated feature map
        """
        mean = self.mlp_mean(a_global)
        std = self.mlp_std(a_global)

        mean = mean[:, :, None, None, None]
        std = std[:, :, None, None, None]

        return std * self.norm(x) + mean
    
    
class AdaINResBlock3D(nn.Module):
    """
    3D residual block where both IN layers are replaced with AdaIN.

    Each residual block applies two AdaIN operations, both conditioned
    on the same global artefact code.

    Input / Output: (B, channels, D, H, W)
    """

    def __init__(
        self,
        channels: int,
        artefact_dim: int = 64,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = False,
        negative_slope: float = 0.2,
        inplace: bool = True,
        adain_affine: bool = False,
        adain_bias: bool = True,
        mean_init: float | None = None,
        std_init: float | None = None,
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
        self.adain1 = AdaIN3D(
            channels,
            artefact_dim=artefact_dim,
            affine=adain_affine,
            bias=adain_bias,
            mean_init=mean_init,
            std_init=std_init,
        )
        self.act1 = nn.LeakyReLU(negative_slope, inplace=inplace)

        self.conv2 = nn.Conv3d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
        self.adain2 = AdaIN3D(
            channels,
            artefact_dim=artefact_dim,
            affine=adain_affine,
            bias=adain_bias,
            mean_init=mean_init,
            std_init=std_init,
        )
        self.act2 = nn.LeakyReLU(negative_slope, inplace=inplace)

    def forward(
        self,
        x: torch.Tensor,
        a_global: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x        : (B, C, D, H, W)
        a_global : (B, artefact_dim)
        """
        residual = x

        x = self.conv1(x)
        x = self.adain1(x, a_global)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.adain2(x, a_global)

        return self.act2(x + residual)
    

def compute_temporal_diffs(x: torch.Tensor) -> torch.Tensor:
    """
    Compute frame-to-frame temporal difference maps.
 
    For a chunk with T timepoints, produces T-1 difference maps:
        Δx[t] = x[t+1] - x[t]   for t = 0, 1, ..., T-2
 
    These capture the rate of temporal change — motion artefacts produce
    sudden large differences while clean BOLD signal changes slowly and
    smoothly. By appending these as extra input channels the discriminator
    can directly evaluate temporal dynamics rather than only spatial patterns.
 
    Parameters
    ----------
    x : (B, T, D, H, W)   fMRI chunk, T timepoints as channels
 
    Returns
    -------
    diffs : (B, T-1, D, H, W)   frame-to-frame differences
    """
    # x[:, 1:] = volumes 1..T-1
    # x[:, :-1] = volumes 0..T-2
    # diff[t] = volume[t+1] - volume[t]
    return x[:, 1:, ...] - x[:, :-1, ...]
 
 
def append_temporal_diffs(x: torch.Tensor) -> torch.Tensor:
    """
    Append temporal difference maps to the chunk as extra channels.
 
    Concatenates the original T timepoints with T-1 difference maps,
    giving a (B, 2T-1, D, H, W) tensor as discriminator input.
 
    For T=20: input becomes (B, 39, D, H, W)
        channels  0–19  : original volumes
        channels 20–38  : frame-to-frame differences
 
    Parameters
    ----------
    x : (B, T, D, H, W)
 
    Returns
    -------
    (B, 2T-1, D, H, W)
    """
    diffs = compute_temporal_diffs(x)       # (B, T-1, D, H, W)
    return torch.cat([x, diffs], dim=1)     # (B, 2T-1, D, H, W)
 
 
class DiscConvBlock(nn.Module):
    """
    Single discriminator convolutional block.
 
    Conv3D (stride 2) + optional InstanceNorm + LeakyReLU.
    Spectral normalisation is applied to the conv weight matrix to
    constrain the Lipschitz constant of the discriminator and prevent
    it from dominating the generator during training (Miyato et al. 2018).
 
    The first block omits normalisation following the pix2pix convention —
    normalising the first layer can destabilise early training when the
    input statistics are still widely varying.
 
    Parameters
    ----------
    in_ch    : int   input channels
    out_ch   : int   output channels
    use_norm : bool  apply InstanceNorm after conv (False for first block)
    """
 
    def __init__(self, in_ch: int, out_ch: int, use_norm: bool = True):
        super().__init__()
 
        # Spectral normalisation wraps the Conv3d weight matrix
        conv = nn.utils.spectral_norm(
            nn.Conv3d(in_ch, out_ch,
                      kernel_size=4, stride=2, padding=1, bias=not use_norm)
        )
        layers = [conv]
 
        if use_norm:
            layers.append(nn.InstanceNorm3d(out_ch, affine=True))
 
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    
    
