"""Quick verification that ImageNet normalization yields ~zero mean / unit std.

Builds a synthetic (3, H, W) tensor whose per-channel mean and std match the
ImageNet statistics, runs it through `normalize_imagenet`, and asserts the
output is standardized. Run with:

    python -m scripts.verify_normalization
"""

import sys

import torch

from src.datasets.dataset import IMAGENET_MEAN, IMAGENET_STD, normalize_imagenet


def main() -> int:
    h = w = 64
    # Construct a tensor in [0, 1] whose per-channel mean/std match ImageNet
    # stats: half the pixels at mean - std, half at mean + std (clamped to
    # [0, 1]). After normalization this gives mean ~0 and std ~1 per channel.
    image = torch.empty(3, h, w)
    for c in range(3):
        low = max(0.0, IMAGENET_MEAN[c] - IMAGENET_STD[c])
        high = min(1.0, IMAGENET_MEAN[c] + IMAGENET_STD[c])
        half = h * w // 2
        image[c].fill_(low)
        image[c].view(-1)[:half] = high

    normalized = normalize_imagenet(image)
    mean = normalized.mean(dim=(1, 2))
    std = normalized.std(dim=(1, 2))

    print("per-channel mean:", mean.tolist())
    print("per-channel std :", std.tolist())
    assert torch.allclose(mean, torch.zeros(3), atol=1e-5), f"mean not ~0: {mean}"
    assert torch.allclose(std, torch.ones(3), atol=1e-2), f"std not ~1: {std}"
    print("OK: normalized output has approximately zero mean and unit std.")
    return 0


if __name__ == "__main__":
    sys.exit(main())