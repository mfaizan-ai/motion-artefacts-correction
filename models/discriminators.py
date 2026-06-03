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
from building_blocks import DiscConvBlock, append_temporal_diffs

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
                 base_ch:       int = 64):
        super().__init__()
 
        # Input channels = original timepoints + temporal differences
        # T=20 → 20 + 19 = 39 input channels
        in_ch = in_timepoints + (in_timepoints - 1)   # 2T - 1 = 39
 
        c = base_ch   # 64
 
        # Four strided conv blocks — each halves spatial dimensions
        self.block1 = DiscConvBlock(in_ch, c,     use_norm=False)  # no norm first
        self.block2 = DiscConvBlock(c,     c * 2, use_norm=True)
        self.block3 = DiscConvBlock(c * 2, c * 4, use_norm=True)
        self.block4 = DiscConvBlock(c * 4, c * 8, use_norm=True)
 
        # Output conv — produces patch score map
        # Spectral norm applied here too for consistency
        # kernel=3 padding=1: preserves spatial dims at (5, 6, 4)
        # no normalisation, no activation — raw logits for LSGAN
        self.out_conv = nn.utils.spectral_norm(
            nn.Conv3d(c * 8, 1,
                      kernel_size=3, padding=1, bias=True)
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, D, H, W)   fMRI chunk, T timepoints as channels
                               e.g. (B, 20, 80, 96, 72)
 
        Returns
        -------
        patch_scores : (B, 1, 5, 6, 4)   patch-level real/fake logits
                       positive = real, negative = fake (before sigmoid)
        """
        # Augment with temporal differences before any conv
        x = append_temporal_diffs(x)       # (B, 39, 80, 96, 72)
 
        x = self.block1(x)                 # (B,  64, 40, 48, 36)
        x = self.block2(x)                 # (B, 128, 20, 24, 18)
        x = self.block3(x)                 # (B, 256, 10, 12,  9)
        x = self.block4(x)                 # (B, 512,  5,  6,  4)
 
        return self.out_conv(x)            # (B,   1,  5,  6,  4)
 
 
 # sanity check with dummy input
if __name__ == "__main__":
    B, T, D, H, W = 2, 20, 80, 96, 72
    dummy_input = torch.randn(B, T, D, H, W)
    disc = PatchDiscriminator3D(in_timepoints=T)
    output = disc(dummy_input)
    print("Output shape:", output.shape)  # should be (B, 1, 5, 6, 4)
    assert output.shape == (B, 1, 5, 6, 4), "Output shape mismatch!"