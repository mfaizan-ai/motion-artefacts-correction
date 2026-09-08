import torch 
import torch.nn as nn

class MultiScaleROITemporalDiscriminator(nn.Module):
    def __init__(
        self,
        n_rois=400,
        temporal_features=32
    ):
        super().__init__()

        self.n_rois = n_rois
        self.temporal_features = temporal_features

        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(
                16,
                temporal_features,
                kernel_size=3,
                padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Separate classifier for every ROI
        self.roi_weight = nn.Parameter(
            torch.randn(n_rois, temporal_features) * 0.02
        )
        self.roi_bias = nn.Parameter(
            torch.zeros(n_rois)
        )

        # Whole-brain classifier
        self.global_head = nn.Sequential(
            nn.Linear(
                n_rois * temporal_features,
                256
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # x: (B, 400, 5)
        B, R, T = x.shape

        # ROI-wise temporal encoding
        z = x.reshape(B * R, 1, T)
        z = self.temporal_encoder(z)

        # (B*R, 32, 5) -> (B*R, 32)
        z = z.mean(dim=-1)

        # (B, 400, 32)
        z = z.reshape(B, R, self.temporal_features)

        # Individual ROI scores: (B, 400)
        roi_scores = torch.einsum(
            "brf,rf->br",
            z,
            self.roi_weight
        )
        roi_scores = roi_scores + self.roi_bias

        # Whole-brain score: (B, 1)
        global_scores = self.global_head(
            z.flatten(start_dim=1)
        )

        return {
            "roi": roi_scores,
            "global": global_scores
        }