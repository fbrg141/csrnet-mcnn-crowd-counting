"""Tests for the on-disk density-map cache in CrowdCountingDataset.

Uses the project's fake-ShanghaiTech fixture pattern (tiny synthetic images +
.mat annotations under tmp_path) so no real dataset is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image

from src.datasets.dataset import CrowdCountingDataset
from src.datasets import density_map as dm_mod


def _make_fake_shanghaitech(root: Path, part: str = "A", n: int = 4) -> None:
    """Create a minimal fake ShanghaiTech layout on disk."""
    img_dir = root / f"part_{part}" / "train_data" / "images"
    gt_dir = root / f"part_{part}" / "train_data" / "ground-truth"
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        Image.fromarray(
            rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        ).save(img_dir / f"IMG_{i + 1}.jpg")
        points = rng.uniform(0, 64, size=(6, 2))
        sio.savemat(
            gt_dir / f"GT_IMG_{i + 1}.mat",
            {"image_info": np.array([[[[[points]]]]], dtype=float)},
        )


def _ds(root, **kw):
    """Build a train dataset with caching on, cache under tmp root."""
    cache = kw.pop("cache_dir", root / "cache")
    return CrowdCountingDataset(
        root=root, part="A", split="train", downsample_factor=4, sigma=8.0,
        use_cache=True, cache_dir=cache, **kw,
    )


def test_cache_matches_no_cache(tmp_path: Path) -> None:
    """Cached density equals on-the-fly density for every image (numerically)."""
    _make_fake_shanghaitech(tmp_path)
    cached = _ds(tmp_path)
    fresh = CrowdCountingDataset(
        tmp_path, part="A", split="train", downsample_factor=4, sigma=8.0,
        use_cache=False,
    )
    for i in range(len(cached)):
        _, c = cached[i]
        _, f = fresh[i]
        np.testing.assert_allclose(c.squeeze(0), f.squeeze(0), atol=1e-8)


def test_cache_file_created_on_access(tmp_path: Path) -> None:
    """Touching an item writes a .npy cache file keyed by the image stem."""
    _make_fake_shanghaitech(tmp_path)
    ds = _ds(tmp_path)
    ds[0]
    path = ds._cache_path("IMG_1")
    assert path.exists(), f"expected cache file at {path}"
    assert path.suffix == ".npy"


def test_cache_hit_skips_generation(tmp_path: Path, monkeypatch) -> None:
    """A second dataset over the same cache_dir loads without regenerating."""
    _make_fake_shanghaitech(tmp_path)

    calls = {"n": 0}
    orig = dm_mod.fixed_sigma_density_map

    def counting_fixed(points, h, w, sigma=15.0):
        calls["n"] += 1
        return orig(points, h, w, sigma=sigma)

    monkeypatch.setattr(dm_mod, "fixed_sigma_density_map", counting_fixed)
    # NOTE: dataset imports the name directly, so patch the dataset's binding too.
    import src.datasets.dataset as dmod
    monkeypatch.setattr(dmod, "fixed_sigma_density_map", counting_fixed)

    # First dataset: cold cache -> generates.
    ds1 = _ds(tmp_path)
    for i in range(len(ds1)):
        ds1[i]
    cold = calls["n"]
    assert cold == len(ds1), f"cold cache should generate once per image, got {cold}"

    # Second dataset, same cache_dir: warm -> no generation.
    calls["n"] = 0
    ds2 = _ds(tmp_path)
    for i in range(len(ds2)):
        ds2[i]
    assert calls["n"] == 0, f"warm cache regenerated {calls['n']} times"


def test_cache_key_separates_downsample_factor(tmp_path: Path) -> None:
    """MCNN (ds4) and CSRNet (ds8) cache to different paths (different stride)."""
    _make_fake_shanghaitech(tmp_path)
    a = CrowdCountingDataset(tmp_path, part="A", split="train", downsample_factor=4,
                             use_cache=True, cache_dir=tmp_path / "c")
    b = CrowdCountingDataset(tmp_path, part="A", split="train", downsample_factor=8,
                             use_cache=True, cache_dir=tmp_path / "c")
    a[0]; b[0]
    assert a._cache_path("IMG_1") != b._cache_path("IMG_1")
    assert a._cache_path("IMG_1").exists()
    assert b._cache_path("IMG_1").exists()


def test_cache_key_separates_density_mode(tmp_path: Path) -> None:
    """Fixed vs adaptive produce different cache paths."""
    _make_fake_shanghaitech(tmp_path)
    fixed = CrowdCountingDataset(tmp_path, part="A", split="train", downsample_factor=4,
                                 density_mode="fixed", use_cache=True,
                                 cache_dir=tmp_path / "c")
    adapt = CrowdCountingDataset(tmp_path, part="A", split="train", downsample_factor=4,
                                 density_mode="adaptive", use_cache=True,
                                 cache_dir=tmp_path / "c")
    fixed[0]; adapt[0]
    assert fixed._cache_path("IMG_1") != adapt._cache_path("IMG_1")
    assert adapt._cache_path("IMG_1").exists()


def test_no_cache_writes_nothing(tmp_path: Path) -> None:
    """use_cache=False never writes to disk (backward-compat behaviour)."""
    _make_fake_shanghaitech(tmp_path)
    cache_root = tmp_path / "cache"
    ds = CrowdCountingDataset(tmp_path, part="A", split="train", downsample_factor=4,
                              use_cache=False, cache_dir=cache_root)
    for i in range(len(ds)):
        ds[i]
    assert not cache_root.exists() or not any(cache_root.rglob("*.npy"))


def test_cache_path_contains_version(tmp_path: Path) -> None:
    """Cache path embeds CACHE_VERSION so bumping it invalidates the cache."""
    _make_fake_shanghaitech(tmp_path)
    from src.config import CACHE_VERSION
    ds = _ds(tmp_path)
    path = ds._cache_path("IMG_1")
    assert f"v{CACHE_VERSION}_" in str(path), path


def test_atomic_write_no_tmp_leftover(tmp_path: Path) -> None:
    """No .tmp files remain after caching (atomic write cleans up)."""
    _make_fake_shanghaitech(tmp_path)
    ds = _ds(tmp_path)
    for i in range(len(ds)):
        ds[i]
    tmps = list((tmp_path / "cache").rglob("*.tmp"))
    assert tmps == [], f"leftover tmp files: {tmps}"


def test_precompute_script_warms_cache(tmp_path: Path) -> None:
    """The precompute script writes a .npy for every train image."""
    _make_fake_shanghaitech(tmp_path, n=3)
    from scripts.precompute_density_maps import main as precompute_main
    cache = tmp_path / "cache"
    precompute_main([
        "--parts", "A", "--splits", "train", "--models", "mcnn",
        "--root", str(tmp_path), "--cache-dir", str(cache),
    ])
    npy = list(cache.rglob("*.npy"))
    # from_config applies val_split, so train keeps N - held_out images.
    expected = len(CrowdCountingDataset.from_config(
        part="A", split="train", root=tmp_path,
        downsample_factor=4, use_cache=False,
    ))
    assert len(npy) == expected, f"expected {expected} cached maps, got {len(npy)}"
    # Loading one back through a fresh caching dataset must match a fresh build.
    ds = CrowdCountingDataset(tmp_path, part="A", split="train", downsample_factor=4,
                              use_cache=True, cache_dir=cache)
    _, cached = ds[0]
    fresh = CrowdCountingDataset(tmp_path, part="A", split="train", downsample_factor=4,
                                 use_cache=False)
    _, f = fresh[0]
    np.testing.assert_allclose(cached.squeeze(0), f.squeeze(0), atol=1e-8)