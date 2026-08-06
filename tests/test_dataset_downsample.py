"""Tests for the downsample_factor parameter in CrowdCountingDataset."""

from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image

from src.datasets.dataset import CrowdCountingDataset
from src.datasets.density_map import fixed_sigma_density_map


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


def test_downsample_factor_one_is_unchanged(tmp_path: Path) -> None:
    """factor=1 keeps density at full image resolution (backward compat)."""
    _make_fake_shanghaitech(tmp_path)
    ds = CrowdCountingDataset(tmp_path, part="A", split="train", downsample_factor=1)
    _, dens = ds[0]
    assert dens.shape == (1, 64, 64)


def test_downsample_factor_reduces_density_resolution(tmp_path: Path) -> None:
    """factor>1 produces a density map at (H//factor, W//factor)."""
    _make_fake_shanghaitech(tmp_path)
    for factor, expected in ((2, 32), (4, 16), (8, 8)):
        ds = CrowdCountingDataset(tmp_path, part="A", split="train", downsample_factor=factor)
        img, dens = ds[0]
        # Image stays at full resolution; only the density map shrinks.
        assert img.shape == (3, 64, 64)
        assert dens.shape == (1, expected, expected), (
            f"factor={factor}: density {dens.shape} != (1,{expected},{expected})"
        )


def test_downsample_sum_approximates_head_count(tmp_path: Path) -> None:
    """The downsampled density-map sum stays close to the head count."""
    n_heads = 10
    root = tmp_path
    img_dir = root / "part_A" / "train_data" / "images"
    gt_dir = root / "part_A" / "train_data" / "ground-truth"
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    Image.fromarray(img).save(img_dir / "IMG_1.jpg")
    points = rng.uniform(0, 64, size=(n_heads, 2))
    sio.savemat(
        gt_dir / "GT_IMG_1.mat",
        {"image_info": np.array([[[[[points]]]]], dtype=float)},
    )
    ds = CrowdCountingDataset(
        root, part="A", split="train", downsample_factor=4, sigma=8.0
    )
    _, dens = ds[0]
    assert dens.shape == (1, 16, 16)
    rel_err = abs(float(dens.sum()) - n_heads) / n_heads
    assert rel_err < 0.20, f"density sum {dens.sum():.2f} too far from {n_heads}"


def test_downsample_factor_matches_manual_path(tmp_path: Path) -> None:
    """The dataset's downsampled density equals a manual scale-and-generate path."""
    _make_fake_shanghaitech(tmp_path)
    sigma = 15.0
    factor = 4

    # Dataset path
    ds = CrowdCountingDataset(
        tmp_path, part="A", split="train", downsample_factor=factor, sigma=sigma
    )
    _, dens = ds[0]

    # Manual path: load the same points, scale them, generate at reduced size.
    gt_path = ds.gts[0]
    # Re-derive points the same way the dataset does (factor=1 dataset).
    ds_base = CrowdCountingDataset(
        tmp_path, part="A", split="train", downsample_factor=1, sigma=sigma
    )
    from src.datasets.dataset import _load_annotations

    points = _load_annotations(gt_path)
    pts = points.copy()
    pts[:, 0] /= factor
    pts[:, 1] /= factor
    expected = fixed_sigma_density_map(pts, 64 // factor, 64 // factor, sigma=sigma / factor)

    assert dens.shape == (1, *expected.shape)
    np.testing.assert_allclose(dens.squeeze(0), expected, atol=1e-8)


def test_invalid_downsample_factor_raises(tmp_path: Path) -> None:
    """Non-positive or non-integer downsample_factor is rejected."""
    _make_fake_shanghaitech(tmp_path)
    import pytest

    for bad in (0, -1, 1.5, "4"):
        with pytest.raises((ValueError, TypeError)):
            CrowdCountingDataset(tmp_path, part="A", split="train", downsample_factor=bad)  # type: ignore[arg-type]