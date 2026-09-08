"""
discriminators.py
=================
3D PatchGAN discriminators for the disentangled CycleGAN fMRI motion
artefact correction model.

import torch
from torch import nn
from building_blocks import DiscConvBlock, append_temporal_diffs


Key design choices:
    1. 3D PatchGAN  — spatially specific patch-level real/fake judgements
                      rather than a single global score per chunk.
                      Provides localised gradient feedback to the generator.
 
    2. Temporal difference channels — frame-to-frame differences appended
                      as extra input channels. Gives the discriminator direct
                      access to temporal dynamics, making it harder for the
                      generator to produce unrealistic temporal jumps or
                      artificially smooth timeseries.
 
    3. Spectral normalisation — applied to every conv layer to prevent
                      the discriminator from becoming too powerful too quickly,
                      avoiding vanishing generator gradients.
                      
This single discrimintor coudl be used for both domains (motion-corrupted and clean) since they share the same architecture and input format.
"""

import torch
from torch import nn
from typing import List, Union
from models.building_blocks import DiscConvBlock, append_temporal_diffs


def _make_scale_cnn(in_ch: int, base_ch: int) -> nn.Sequential:
    """Single-scale PatchGAN CNN: 4 strided-conv blocks + spectral-norm output conv."""
    c = base_ch
    return nn.Sequential(
        DiscConvBlock(in_ch, c,     use_norm=False),
        DiscConvBlock(c,     c * 2, use_norm=True),
        DiscConvBlock(c * 2, c * 4, use_norm=True),
        DiscConvBlock(c * 4, c * 8, use_norm=True),
        nn.utils.spectral_norm(nn.Conv3d(c * 8, 1, kernel_size=3, padding=1, bias=True)),
    )


class MultiScalePatchDiscriminator3D(nn.Module):
    """
    Multi-scale 3D PatchGAN discriminator (adapted from DC-MS-GANs MsImageDis).

    Runs `num_scales` independent PatchGAN CNNs on the input at progressively
    coarser resolutions. Between scales the input is halved with AvgPool3d.
    Returns a list of score maps (finest scale first).

    Temporal difference channels are OFF by default — appending T-1 diff maps
    gave the discriminator a large information advantage over the generator,
    making it trivially easy to detect temporal discontinuities and causing
    D to dominate (use_temporal_diffs=True to restore old behaviour).

    Parameters
    ----------
    in_timepoints      : number of input timepoints / channels
    base_ch            : channels after first conv in each scale CNN
    num_scales         : number of scales (default 2)
    use_temporal_diffs : if True, prepend T-1 frame-diff channels (default False)
    """

    def __init__(self,
                 in_timepoints:      int  = 20,
                 base_ch:            int  = 64,
                 num_scales:         int  = 2,
                 use_temporal_diffs: bool = False):
        super().__init__()

        self.use_temporal_diffs = use_temporal_diffs
        self.num_scales         = num_scales
        self.downsample = nn.AvgPool3d(3, stride=2, padding=1, count_include_pad=False)

        in_ch = (in_timepoints + (in_timepoints - 1)) if use_temporal_diffs else in_timepoints
        self.cnns = nn.ModuleList([_make_scale_cnn(in_ch, base_ch) for _ in range(num_scales)])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.use_temporal_diffs:
            x = append_temporal_diffs(x)

        outputs = []
        for cnn in self.cnns:
            outputs.append(cnn(x))
            x = self.downsample(x)
        return outputs   # list of score maps, finest first


class MotionFreeDiscriminator(MultiScalePatchDiscriminator3D):
    """Discriminator for the motion-free domain (D_B)."""
    def __init__(self, in_timepoints=20, base_ch=64, num_scales=2, use_temporal_diffs=False):
        super().__init__(in_timepoints, base_ch, num_scales, use_temporal_diffs)


class MotionCorruptedDiscriminator(MultiScalePatchDiscriminator3D):
    """Discriminator for the motion-corrupted domain (D_A)."""
    def __init__(self, in_timepoints=20, base_ch=64, num_scales=2, use_temporal_diffs=False):
        super().__init__(in_timepoints, base_ch, num_scales, use_temporal_diffs)
 
 
 
 
 
 # sanity check with dummy input
if __name__ == "__main__":
    B, T, D, H, W = 2, 20, 80, 96, 72
    dummy_input = torch.randn(B, T, D, H, W)
    disc = PatchDiscriminator3D(in_timepoints=T)
    output = disc(dummy_input)
    print("Output shape:", output.shape)  # should be (B, 1, 5, 6, 4)
    assert output.shape == (B, 1, 5, 6, 4), "Output shape mismatch!"