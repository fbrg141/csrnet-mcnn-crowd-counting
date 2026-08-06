"""Tests for ImageNet normalization in CrowdCountingDataset."""

from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from PIL import Image

from src.datasets.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    CrowdCountingDataset,
    normalize_imagenet,
)


def _make_fake_shanghaitech(root: Path, part: str = "A", n: int = 4) -> None:
    """Create a minimal fake ShanghaiTech layout on disk."""
    img_dir = root / f"part_{part}" / "train_data" / "images"
    gt_dir = root / f"part_{part}" / "train_data" / "ground-truth"
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(img).save(img_dir / f"IMG_{i + 1}.jpg")
        points = rng.uniform(0, 64, size=(3, 2))
        sio.savemat(
            gt_dir / f"GT_IMG_{i + 1}.mat",
            {"image_info": np.array([[[[[points]]]]], dtype=float)},
        )


def test_normalize_false_is_unchanged(tmp_path: Path) -> None:
    """With normalize=False the output is only divided by 255.0."""
    _make_fake_shanghaitech(tmp_path)
    ds = CrowdCountingDataset(tmp_path, part="A", split="train", normalize=False)
    img, _ = ds[0]
    assert img.shape == (3, 64, 64)
    assert img.min() >= 0.0
    assert img.max() <= 1.0


def test_normalize_true_applies_imagenet_stats(tmp_path: Path) -> None:
    """With normalize=True the output equals (x/255 - mean) / std per channel."""
    _make_fake_shanghaitech(tmp_path)
    ds_raw = CrowdCountingDataset(tmp_path, part="A", split="train", normalize=False)
    ds_norm = CrowdCountingDataset(tmp_path, part="A", split="train", normalize=True)
    raw, _ = ds_raw[0]
    norm, _ = ds_norm[0]
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    expected = (raw - mean) / std
    assert torch.allclose(norm, expected, atol=1e-6)


def test_normalize_imagenet_constant_image_zero_mean() -> None:
    """A constant image at the ImageNet mean normalizes to ~zero per channel."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    const = mean.expand(3, 8, 8).clone()
    out = normalize_imagenet(const)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_normalize_imagenet_standardizes_imagenet_stats() -> None:
    """An image with ImageNet per-channel mean/std normalizes to ~0 mean, ~1 std."""
    h = w = 64
    image = torch.empty(3, h, w)
    for c in range(3):
        low = max(0.0, IMAGENET_MEAN[c] - IMAGENET_STD[c])
        high = min(1.0, IMAGENET_MEAN[c] + IMAGENET_STD[c])
        half = h * w // 2
        image[c].fill_(low)
        image[c].view(-1)[:half] = high
    out = normalize_imagenet(image)
    assert torch.allclose(out.mean(dim=(1, 2)), torch.zeros(3), atol=1e-5)
    assert torch.allclose(out.std(dim=(1, 2)), torch.ones(3), atol=1e-2)