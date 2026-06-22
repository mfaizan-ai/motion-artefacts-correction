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
from models.building_blocks import DiscConvBlock, append_temporal_diffs

class PatchDiscriminator3D(nn.Module):
    """
    3D PatchGAN discriminator with temporal difference augmentation.
 
    Produces a spatial map of patch-level real/fake scores rather than
    a single global score. Each value in the output grid independently
    judges a local overlapping patch of the input chunk.
 
    Input preparation:
        Original chunk (B, T, D, H, W) is augmented with T-1 temporal
        difference maps to give (B, 2T-1, D, H, W) before the first conv.
 
    Architecture (4 strided conv blocks + output conv):
        Block 1 : (B, 2T-1, 80, 96, 72) → (B,  64, 40, 48, 36)  no norm
        Block 2 : (B,  64,  40, 48, 36) → (B, 128, 20, 24, 18)  InstanceNorm
        Block 3 : (B, 128,  20, 24, 18) → (B, 256, 10, 12,  9)  InstanceNorm
        Block 4 : (B, 256,  10, 12,  9) → (B, 512,  5,  6,  4)  InstanceNorm
        Out conv: (B, 512,   5,  6,  4) → (B,   1,  5,  6,  4)  no norm
 
    Output: (B, 1, 5, 6, 4)
        120 independent patch scores per sample.
        Each patch covers ~16×16×18 voxels in the original volume.
 
    Loss (LSGAN):
        Discriminator: E[(D(real)-1)²] + E[(D(fake))²]
        Generator:     E[(D(fake)-1)²]
        Implemented externally in the training loop.
 
    Parameters
    ----------
    in_timepoints : int   number of input timepoints (default 20)
    base_ch       : int   channels after first conv block (default 64)
    """
 
    def __init__(self,
                 in_timepoints: int = 20,
                 base_ch:       int = 64,
                 use_temporal_diffs: bool = True):
        super().__init__()

        self.use_temporal_diffs = use_temporal_diffs

        if use_temporal_diffs:
            in_ch = in_timepoints + (in_timepoints - 1)   # 2T - 1 = 39
        else:
            in_ch = in_timepoints                          # T = 20

        c = base_ch   # 64

        # Four strided conv blocks — each halves spatial dimensions
        self.block1 = DiscConvBlock(in_ch, c,     use_norm=False)  # no norm first
        self.block2 = DiscConvBlock(c,     c * 2, use_norm=True)
        self.block3 = DiscConvBlock(c * 2, c * 4, use_norm=True)
        self.block4 = DiscConvBlock(c * 4, c * 8, use_norm=True)

        self.out_conv = nn.utils.spectral_norm(
            nn.Conv3d(c * 8, 1,
                      kernel_size=3, padding=1, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_temporal_diffs:
            x = append_temporal_diffs(x)       # (B, 2T-1, D, H, W)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        return self.out_conv(x)
 
 
  
class MotionFreeDiscriminator(PatchDiscriminator3D):
    """Discriminator for the motion-free domain (D_B)."""
    def __init__(self, in_timepoints=20, base_ch=64, use_temporal_diffs=True):
        super().__init__(in_timepoints, base_ch, use_temporal_diffs)


class MotionCorruptedDiscriminator(PatchDiscriminator3D):
    """Discriminator for the motion-corrupted domain (D_A)."""
    def __init__(self, in_timepoints=20, base_ch=64, use_temporal_diffs=True):
        super().__init__(in_timepoints, base_ch, use_temporal_diffs)
 
 
 
 
 
 # sanity check with dummy input
if __name__ == "__main__":
    B, T, D, H, W = 2, 20, 80, 96, 72
    dummy_input = torch.randn(B, T, D, H, W)
    disc = PatchDiscriminator3D(in_timepoints=T)
    output = disc(dummy_input)
    print("Output shape:", output.shape)  # should be (B, 1, 5, 6, 4)
    assert output.shape == (B, 1, 5, 6, 4), "Output shape mismatch!"