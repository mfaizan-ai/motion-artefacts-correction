#!/usr/bin/env python3
"""
model.py
========
Disentangled CycleGAN for fMRI motion artefact correction.

Assembles four components:
    ContentEncoder          E_c
    ArtefactEncoder         E_a
    MotionFreeDecoder       G_B
    MotionCorruptedDecoder  G_A
    MotionFreeDiscriminator       D_B
    MotionCorruptedDiscriminator  D_A

All intermediate tensors are returned in a ModelOutputs dataclass so
the training loop can compute any loss without re-running the model.

Based on:
    Dr-CycleGAN  (Dewey et al. 2020)
    MUNIT        (Huang et al. 2018)
    CycleGAN     (Zhu et al.   2017)
"""

from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn

# Local component imports 
from encoders       import ContentEncoder,   ArtefactEncoder
from decoders       import MotionFreeDecoder, MotionCorruptedDecoder
from discriminators import (MotionFreeDiscriminator,
                             MotionCorruptedDiscriminator)



# Model outputs dataclass to hold all intermediate tensors from a forward pass
@dataclass
class ModelOutputs:
    """
    All intermediate tensors produced by one forward pass.

    Naming convention:
        c_*      content feature maps
        a_*      artefact codes (global or spatial)
        x_hat_*  domain-translated outputs (intermediate stage)
        x_self_* self-reconstructions
        x_cycle_*cyclic reconstructions
        score_*  discriminator patch score maps

    All chunk tensors have shape (B, 20, 80, 96, 72).
    Content maps have shape (B, 384, 10, 12, 9).
    """

    #  Phase 1: encoded features
    c_a        : torch.Tensor   # (B, 384, 10, 12, 9)  content from x_a. where x_a is the motion-corrupted input and x_b is the the motino free input. 
    c_b        : torch.Tensor   # (B, 384, 10, 12, 9)  content from x_b
    a_global   : torch.Tensor   # (B, 64)              artefact global from x_a
    a_spatial  : torch.Tensor   # (B, 32, 10, 12, 9)   artefact spatial from x_a
    a_global_b : torch.Tensor   # (B, 64)              artefact global from x_b (→ 0)
    a_spatial_b: torch.Tensor   # (B, 32, 10, 12, 9)   artefact spatial from x_b (→ 0)

    # Phase 2: intermediate translations 
    x_hat_b    : torch.Tensor   # (B, 20, 80, 96, 72)  A→B predicted clean
    x_hat_a    : torch.Tensor   # (B, 20, 80, 96, 72)  B→A predicted corrupted
    x_self_a   : torch.Tensor   # (B, 20, 80, 96, 72)  A→A self-reconstruct corrupted 
    x_self_b   : torch.Tensor   # (B, 20, 80, 96, 72)  B→B self-reconstruct clean 

    # Phase 3: cyclic re-encodings
    c_hat_b    : torch.Tensor   # (B, 384, 10, 12, 9)  re-encoded from x_hat_b
    c_hat_a    : torch.Tensor   # (B, 384, 10, 12, 9)  re-encoded from x_hat_a
    a_hat_global : torch.Tensor # (B, 64)              artefact from x_hat_a
    a_hat_spatial: torch.Tensor # (B, 32, 10, 12, 9)   artefact from x_hat_a

    #  Phase 4: cyclic outputs 
    x_cycle_a  : torch.Tensor   # (B, 20, 80, 96, 72)  A→B→A should ≈ x_a
    x_cycle_b  : torch.Tensor   # (B, 20, 80, 96, 72)  B→A→B should ≈ x_b

    # Discriminator scores 
    score_real_b : torch.Tensor  # (B, 1, 5, 6, 4)  D_B on real x_b
    score_fake_b : torch.Tensor  # (B, 1, 5, 6, 4)  D_B on x_hat_b
    score_real_a : torch.Tensor  # (B, 1, 5, 6, 4)  D_A on real x_a
    score_fake_a : torch.Tensor  # (B, 1, 5, 6, 4)  D_A on x_hat_a


# Distengled CycleGAN model for fMRI motion artefact correction
class DisentangledCycleGAN(nn.Module):
    """
    Disentangled CycleGAN for fMRI motion artefact correction.

    Contains all six components:
        E_c  — content encoder        (shared across both domains)
        E_a  — artefact encoder       (encodes corruption signature)
        G_B  — motion-free decoder    (content only → clean chunk)
        G_A  — corrupted decoder      (content + artefact → corrupted chunk)
        D_B  — motion-free discriminator
        D_A  — corrupted discriminator

    Parameters
    ----------
    in_timepoints   : int   number of input volumes per chunk (default 20)
    spatial_dims    : tuple input spatial dimensions (default (80, 96, 72))
    content_ch      : int   content encoder bottleneck channels (default 384)
    content_base_ch : int   content encoder first conv channels (default 64)
    content_n_res   : int   content encoder residual blocks (default 5)
    artefact_base_ch: int   artefact encoder base channels (default 64)
    global_code_dim : int   global artefact code dimension (default 64)
    spatial_code_ch : int   spatial artefact map channels (default 32)
    disc_base_ch    : int   discriminator base channels (default 64)
    """

    def __init__(self,
                 in_timepoints:    int   = 20,
                 spatial_dims:     tuple = (80, 96, 72),
                 content_ch:       int   = 384,
                 content_base_ch:  int   = 64,
                 content_n_res:    int   = 5,
                 artefact_base_ch: int   = 64,
                 global_code_dim:  int   = 64,
                 spatial_code_ch:  int   = 32,
                 disc_base_ch:     int   = 64):
        super().__init__()

        self.in_timepoints  = in_timepoints
        self.spatial_dims   = spatial_dims
        self.global_code_dim = global_code_dim
        self.spatial_code_ch = spatial_code_ch

        #  Content encoder — shared across both domains
        self.E_c = ContentEncoder(
            in_channels   = in_timepoints,
            base_channels = content_base_ch,
            n_res_blocks  = content_n_res,
        )

        #  Artefact encoder 
        self.E_a = ArtefactEncoder(
            in_channels     = in_timepoints,
            base_channels   = artefact_base_ch,
            global_code_dim = global_code_dim,
            spatial_code_ch = spatial_code_ch,
        )

        #  Motion-free decoder G_B 
        self.G_B = MotionFreeDecoder(
            content_ch   = content_ch,
            out_channels = in_timepoints,
            n_res_blocks = 4,
        )

        #  Motion-corrupted decoder G_A 
        self.G_A = MotionCorruptedDecoder(
            content_ch     = content_ch,
            artefact_dim   = global_code_dim,
            spatial_art_ch = spatial_code_ch,
            out_channels   = in_timepoints,
            n_adain_blocks = 4,
        )

        #  Discriminators 
        self.D_B = MotionFreeDiscriminator(
            in_timepoints = in_timepoints,
            base_ch       = disc_base_ch,
        )
        self.D_A = MotionCorruptedDiscriminator(
            in_timepoints = in_timepoints,
            base_ch       = disc_base_ch,
        )

    # Convenience: parameter groups for separate optimisers
    def generator_parameters(self):
        """
        Returns parameters of E_c, E_a, G_B, G_A.
        Used for the generator + encoder optimiser.
        """
        return (list(self.E_c.parameters()) +
                list(self.E_a.parameters()) +
                list(self.G_B.parameters()) +
                list(self.G_A.parameters()))

    def discriminator_parameters(self):
        """
        Returns parameters of D_A, D_B.
        Used for the discriminator optimiser.
        """
        return (list(self.D_A.parameters()) +
                list(self.D_B.parameters()))

    # Core encoding helper
    def encode(self, x: torch.Tensor):
        """
        Encode a chunk through both content and artefact encoders.

        Parameters
        ----------
        x : (B, T, D, H, W)

        Returns
        -------
        content   : (B, 384, 10, 12, 9)
        a_global  : (B, 64)
        a_spatial : (B, 32, 10, 12, 9)
        """
        content            = self.E_c(x)
        a_global, a_spatial = self.E_a(x)
        return content, a_global, a_spatial

    #  Main forward pass 
    def forward(self,
                x_a: torch.Tensor,
                x_b: torch.Tensor) -> ModelOutputs:
        """
        Full Dr-CycleGAN forward pass.

        Phase 1  — encode both inputs
        Phase 2  — intermediate domain translation + self-reconstruction
        Phase 3  — cyclic re-encoding and reconstruction

        Parameters
        ----------
        x_a : (B, T, D, H, W)   motion-corrupted chunk
        x_b : (B, T, D, H, W)   motion-free chunk

        Returns
        -------
        ModelOutputs dataclass containing all intermediate tensors
        """

    
        # PHASE 1 — ENCODE
        
        # Encode corrupted input — get both content and artefact
        c_a, a_global, a_spatial  = self.encode(x_a)

        # Encode motion-free input — content only needed for translation
        # but encode artefact too so training can enforce it → 0
        c_b, a_global_b, a_spatial_b = self.encode(x_b)

        # PHASE 2 — INTERMEDIATE TRANSLATION
    
        # A → B : corrupted content → clean prediction
        # Only content features reach G_B — no artefact information
        # This is the primary inference path
        x_hat_b = self.G_B(c_a)

        # B → A : clean content + corrupted artefact → synthetic corrupted
        # Artefact codes from x_a are transplanted onto content from x_b
        x_hat_a = self.G_A(c_b, a_global, a_spatial)

        # A → A : self-reconstruct corrupted input
        # Same content + same artefact should reproduce x_a 
        x_self_a = self.G_A(c_a, a_global, a_spatial)

        # B → B : self-reconstruct clean input
        # Content encoder + clean decoder should reproduce x_b 
        x_self_b = self.G_B(c_b)


        # PHASE 3 — CYCLIC TRANSLATION

        # Re-encode the predicted clean output
        # c_hat_b should match c_a if no hallucination occurred
        c_hat_b = self.E_c(x_hat_b)

        # Re-encode the predicted corrupted output through both encoders
        # c_hat_a should match c_b (content preserved through corruption)
        # a_hat should resemble a_global
        c_hat_a, a_hat_global, a_hat_spatial = self.encode(x_hat_a)

        # A→B→A cycle: content from x_hat_b + artefact from x_hat_a
        # Should recover x_a
        x_cycle_a = self.G_A(c_hat_b, a_hat_global, a_hat_spatial)

        # B→A→B cycle: content from x_hat_a decoded cleanly
        # Should recover x_b
        x_cycle_b = self.G_B(c_hat_a)

    
        # DISCRIMINATOR SCORES

        # D_B: judges motion-free domain
        score_real_b = self.D_B(x_b)          # real motion-free
        score_fake_b = self.D_B(x_hat_b)      # predicted clean

        # D_A: judges corrupted domain
        score_real_a = self.D_A(x_a)          # real corrupted
        score_fake_a = self.D_A(x_hat_a)      # predicted corrupted

        return ModelOutputs(
            # Phase 1
            c_a          = c_a,
            c_b          = c_b,
            a_global     = a_global,
            a_spatial    = a_spatial,
            a_global_b   = a_global_b,
            a_spatial_b  = a_spatial_b,
            # Phase 2
            x_hat_b      = x_hat_b,
            x_hat_a      = x_hat_a,
            x_self_a     = x_self_a,
            x_self_b     = x_self_b,
            # Phase 3 re-encodings
            c_hat_b      = c_hat_b,
            c_hat_a      = c_hat_a,
            a_hat_global  = a_hat_global,
            a_hat_spatial = a_hat_spatial,
            # Phase 3 cyclic
            x_cycle_a    = x_cycle_a,
            x_cycle_b    = x_cycle_b,
            # Discriminator
            score_real_b = score_real_b,
            score_fake_b = score_fake_b,
            score_real_a = score_real_a,
            score_fake_a = score_fake_a,
        )

    # Inference only (no artefact encoder needed)
    def correct(self, x_a: torch.Tensor) -> torch.Tensor:
        """
        Inference-time motion correction.

        Only E_c and G_B are used — artefact encoder is discarded.
        Returns the predicted motion-free chunk.

        Parameters
        ----------
        x_a : (B, T, D, H, W)   motion-corrupted chunk

        Returns
        -------
        x_hat_b : (B, T, D, H, W)   predicted motion-free chunk
        """
        c_a = self.E_c(x_a)
        return self.G_B(c_a)

    #  Parameter count summary 
    def count_parameters(self) -> dict:
        """Return parameter counts for each component."""
        def n(module): return sum(p.numel() for p in module.parameters())
        return {
            "E_c (ContentEncoder)"         : n(self.E_c),
            "E_a (ArtefactEncoder)"        : n(self.E_a),
            "G_B (MotionFreeDecoder)"      : n(self.G_B),
            "G_A (MotionCorruptedDecoder)" : n(self.G_A),
            "D_B (MotionFreeDisc)"         : n(self.D_B),
            "D_A (MotionCorruptedDisc)"    : n(self.D_A),
            "Total"                        : n(self),
        }