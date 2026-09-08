"""
test_roi_discriminator.py
==========================
Small sanity check for MultiScaleROITemporalDiscriminator (models/roi_discriminator.py)
with dummy ROI-timeseries input.

Run either way:
    python tests/test_roi_discriminator.py
    python -m tests.test_roi_discriminator
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.roi_discriminator import MultiScaleROITemporalDiscriminator

if __name__ == "__main__":
    B, n_rois, T = 2, 400, 5
    x = torch.randn(B, n_rois, T)

    D = MultiScaleROITemporalDiscriminator(n_rois=n_rois)
    out = D(x)

    print(f"input        : {tuple(x.shape)}")
    print(f"roi scores   : {tuple(out['roi'].shape)}")
    print(f"global score : {tuple(out['global'].shape)}")
