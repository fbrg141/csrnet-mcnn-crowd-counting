"""MCNN: Multi-Column Convolutional Neural Network for crowd counting.

Reference: Zhang et al., "Single-Image Crowd Counting via Multi-Column
Convolutional Neural Network", CVPR 2016.

The network runs three parallel conv columns with different filter sizes
(large / medium / small) to handle heads at very different scales, then
fuses their features with a 1x1 conv into a single-channel density map.
Output spatial size is 1/4 of the input (two 2x2 max-pools).
"""

from __future__ import annotations

import torch
from torch import nn


class MCNN(nn.Module):
    """Three-column density-map regressor.

    Input:  (B, 3, H, W) image tensor.
    Output: (B, 1, H/4, W/4) predicted density map. Sum over the spatial
            dims gives the predicted head count for each image.
    """

    def __init__(self) -> None:
        super().__init__()

        # Column 1 — large filters, for nearby / large heads.
        # Two conv+pool blocks. Receptive field grows large enough to cover
        # a big head in one filter footprint. Few channels (filters are
        # parameter-expensive at this size).
        self.col1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Column 2 — medium filters, for mid-distance heads.
        self.col2 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 40, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Column 3 — small filters, for far / small heads. More channels:
        # small filters have few parameters each, so we can afford more of
        # them, which keeps the three columns roughly balanced in capacity.
        self.col3 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Fusion: a 1x1 conv is a learned per-pixel linear combination of
        # the 120 concatenated channels. It learns where to trust which
        # column (dense crowd -> small filters, sparse -> large filters).
        self.fusion = nn.Conv2d(32 + 40 + 48, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.col1(x)  # (B, 32, H/4, W/4)
        out2 = self.col2(x)  # (B, 40, H/4, W/4)
        out3 = self.col3(x)  # (B, 48, H/4, W/4)
        out = torch.cat([out1, out2, out3], dim=1)  # (B, 120, H/4, W/4)
        out = self.fusion(out)  # (B, 1, H/4, W/4)
        return out