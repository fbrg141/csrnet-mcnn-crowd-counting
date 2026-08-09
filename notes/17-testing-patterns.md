# 17 — Testing Patterns for PyTorch Datasets

This note documents how we test the dataset code without requiring the real ShanghaiTech dataset. Grounded in `tests/test_dataset_downsample.py` and `tests/test_dataset_normalization.py`.

---

## 17.1 The Core Challenge

Testing a dataset class that loads images and `.mat` files from disk needs files on disk. But we don't want to:

- Require the 333 MB dataset download for tests
- Depend on network access in CI
- Make tests slow (loading real images)

**Solution:** create a fake mini-dataset on disk with the same directory structure, using tiny synthetic images and random annotations.

---

## 17.2 The Fake Fixture Pattern

```python
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
```

### What it creates

```
root/
  part_A/
    train_data/
      images/IMG_1.jpg ... IMG_4.jpg       (64×64 random RGB noise)
      ground-truth/GT_IMG_1.mat ... GT_IMG_4.mat  (3 heads each)
```

### Key design choices

**Tiny images (64×64)** — fast to create, load, and process. Real ShanghaiTech images are ~1024×768; 64×64 is enough to test the pipeline logic.

**Random noise images** — the dataset class doesn't inspect image content, it only needs valid JPEGs. Noise works fine.

**3 heads per image** — enough to test density map generation without being slow. Some tests use more (10) when testing sum accuracy.

**Fixed seed (`rng = np.random.default_rng(0)`)** — deterministic. The same fake images and annotations every run. Tests must be reproducible.

**`.mat` nesting** — the `np.array([[[[[points]]]]]` replicates the exact MATLAB struct nesting that `scipy.io.loadmat` produces, so `_load_annotations` can peel it:

```python
mat["image_info"][0, 0][0, 0][0]  # → the (N, 2) points array
```

This is the trickiest part — getting the nesting right so the production code's indexing works.

---

## 17.3 The `tmp_path` Fixture

```python
def test_downsample_factor_one_is_unchanged(tmp_path: Path) -> None:
    _make_fake_shanghaitech(tmp_path)
    ds = CrowdCountingDataset(tmp_path, part="A", split="train", downsample_factor=1)
    ...
```

**`tmp_path`** is a pytest built-in fixture. Each test gets a fresh, empty temporary directory:

- Automatically created before the test
- Automatically deleted after the test
- Unique per test (no interference between tests)
- A `Path` object, so you can use `/` directly

No manual cleanup, no `setUp`/`tearDown` boilerplate. pytest handles it.

---

## 17.4 Test Through the Real Class

```python
# GOOD: test through the real CrowdCountingDataset
ds = CrowdCountingDataset(tmp_path, part="A", split="train", downsample_factor=4)
img, dens = ds[0]

# BAD: reimplement the downsample math (can drift from the real code)
pts[:, 0] /= factor
pts[:, 1] /= factor
sig = sigma / factor
dens = fixed_sigma_density_map(pts, h_d, w_d, sigma=sig)
```

**The key principle:** test the real code, not a copy of it. If you reimplement the logic in the test, the test can pass while the real code is broken (or vice versa — the test's copy drifts from the implementation).

The deleted `verify_downsample.py` made this mistake — it reimplemented the downsample math instead of calling through `CrowdCountingDataset`. The pytest tests do it correctly.

### Exception: when manual comparison is the test

```python
def test_downsample_factor_matches_manual_path(tmp_path: Path) -> None:
    # Dataset path (the real code)
    ds = CrowdCountingDataset(tmp_path, ..., downsample_factor=4, sigma=sigma)
    _, dens = ds[0]

    # Manual path (independent computation)
    points = _load_annotations(gt_path)
    pts = points.copy()
    pts[:, 0] /= factor
    expected = fixed_sigma_density_map(pts, 64 // factor, 64 // factor, sigma=sigma / factor)

    # They must match
    np.testing.assert_allclose(dens.squeeze(0), expected, atol=1e-8)
```

Here the manual path is **the expected result** — an independent computation to verify the real code against. This is the strongest form of test: two independent implementations that must agree.

---

## 17.5 What We Test

### Shape tests

```python
assert dens.shape == (1, 64, 64)          # factor=1, full res
assert dens.shape == (1, expected, expected)  # factor>1, reduced
assert img.shape == (3, 64, 64)            # image stays full res
```

Verify the tensor shapes are correct — the most common bug in image pipelines.

### Sum accuracy

```python
rel_err = abs(float(dens.sum()) - n_heads) / n_heads
assert rel_err < 0.20
```

The density map sum should approximate the head count. Tolerance accounts for edge clipping (Gaussians near image borders lose some mass).

### Exact match

```python
np.testing.assert_allclose(dens.squeeze(0), expected, atol=1e-8)
```

Compare against an independent computation — must match to 8 decimal places. Catches subtle numerical bugs.

### Backward compatibility

```python
# factor=1 must be identical to the original path (no regression)
ds_factor1 = CrowdCountingDataset(tmp_path, ..., downsample_factor=1)
_, dens = ds_factor1[0]
assert dens.shape == (1, 64, 64)  # unchanged from before the feature
```

New features must not break existing behavior. The default path is tested as carefully as the new path.

### Error handling

```python
for bad in (0, -1, 1.5, "4"):
    with pytest.raises((ValueError, TypeError)):
        CrowdCountingDataset(tmp_path, ..., downsample_factor=bad)
```

Bad inputs must be rejected, not silently accepted. Tests both the validation and that it raises the right exception type.

---

## 17.6 pytest Patterns Used

### `with pytest.raises(...)`

```python
with pytest.raises(ValueError):
    CrowdCountingDataset(..., downsample_factor=0)  # must raise
```

Asserts that the code inside the `with` block raises the specified exception. If no exception is raised, the test **fails** — "I expected an error but didn't get one."

Can accept a tuple of exception types:

```python
with pytest.raises((ValueError, TypeError)):
    CrowdCountingDataset(..., downsample_factor="4")  # TypeError or ValueError
```

### `np.testing.assert_allclose`

```python
np.testing.assert_allclose(actual, expected, atol=1e-8)
```

"Almost equal" with absolute tolerance. Floating-point operations can produce tiny differences due to order of operations, so exact equality (`==`) is too strict. `1e-8` is tight enough to catch real bugs but loose enough for float rounding.

### `np.array_equal` vs `assert_allclose`

```python
np.array_equal(a, b)              # exact match (bit-for-bit)
np.testing.assert_allclose(a, b)  # approximate match (within tolerance)
```

Use `array_equal` when the operations are deterministic and identical. Use `assert_allclose` when different code paths might produce tiny float differences.

---

## 17.7 Determinism in Tests

```python
rng = np.random.default_rng(0)    # fixed seed
points = rng.uniform(0, 64, size=(3, 2))
```

Every test uses a fixed random seed. This means:

- The same fake data every run
- Tests are reproducible — a failure today is a failure tomorrow
- No flaky tests (tests that pass sometimes, fail other times)

Without a fixed seed, random data would differ each run, making failures impossible to reproduce and debug.

---

## 17.8 The `from __future__ import annotations` Pattern

```python
from __future__ import annotations
```

Enables postponed evaluation of type hints. In Python 3.9 and earlier, type hints like `tuple[int, int]` (lowercase) would crash at import time because `tuple` didn't support subscripting. This import makes all annotations strings (evaluated lazily), so modern type syntax works on older Pythons.

In Python 3.10+ this is the default behavior, but the import is harmless and makes the code work everywhere.

---

## 17.9 Summary

| Pattern | Purpose |
|---|---|
| **Fake fixture** | Create synthetic data on disk — test without the real dataset |
| **`tmp_path`** | Fresh temp directory per test, auto-cleaned |
| **Fixed seeds** | Deterministic, reproducible tests |
| **Test through real class** | No logic duplication — test what users actually call |
| **Shape assertions** | Catch the most common image pipeline bugs |
| **Sum accuracy** | Verify density maps preserve head count |
| **Exact match** | Independent computation must agree |
| **`pytest.raises`** | Verify bad inputs are rejected |
| **`assert_allclose`** | Float comparison with tolerance |