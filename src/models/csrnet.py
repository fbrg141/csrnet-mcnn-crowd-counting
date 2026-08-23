"""CSRNet for density-map based crowd counting.

Reference: Li et al., "CSRNet: Dilated Convolutional Neural Networks for
Understanding the Highly Congested Scenes", CVPR 2018.

The frontend reuses VGG16 through ``conv4_3`` (three pooling stages), while
the configuration-B backend uses six dilation-2 convolutions. The resulting
density map has one eighth of the input image's spatial resolution.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import VGG16_Weights, vgg16


class CSRNet(nn.Module):
    """VGG16 frontend followed by the full CSRNet-B dilated backend.

    Args:
        weights: VGG16 ImageNet weights used to initialize the frontend.
            Pass ``None`` for offline tests or training from scratch.
    """

    output_stride = 8

    def __init__(
        self,
        weights: VGG16_Weights | None = VGG16_Weights.IMAGENET1K_V1,
    ) -> None:
        super().__init__()

        vgg = vgg16(weights=weights)
        # features[:23] ends at conv4_3 + ReLU and excludes pool4.
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])
        # Drop VGG16's unused classifier before allocating the large backend.
        del vgg

        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self._initialize_new_layers()

    @staticmethod
    def _initialize_convolution(layer: nn.Conv2d) -> None:
        """Apply the initialization used by the original CSRNet code."""
        nn.init.normal_(layer.weight, std=0.01)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

    def _initialize_new_layers(self) -> None:
        """Initialize backend/output without overwriting VGG16 weights."""
        for layer in self.backend.modules():
            if isinstance(layer, nn.Conv2d):
                self._initialize_convolution(layer)
        self._initialize_convolution(self.output_layer)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Predict a single-channel density map at one-eighth resolution."""
        features = self.frontend(images)
        features = self.backend(features)
        return self.output_layer(features)