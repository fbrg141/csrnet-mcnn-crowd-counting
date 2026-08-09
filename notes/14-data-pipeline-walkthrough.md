# 14 — The Data Pipeline (Code Walkthrough)

This note documents the actual data pipeline implemented in `src/datasets/dataset.py` and `src/datasets/density_map.py`. It traces one sample from raw files on disk to model-ready tensors.

---

## 14.1 The Three Components

```
scripts/download_data.py   ← gets files onto disk (run once)
src/datasets/density_map.py  ← pure functions: coordinates → density map
src/datasets/dataset.py     ← CrowdCountingDataset: ties everything together
```

The pipeline is triggered by `CrowdCountingDataset.__getitem__(index)`, which the DataLoader calls for each sample.

---

## 14.2 On-Disk Layout

After `download_data.py` runs:

```
data/raw/ShanghaiTech/
  part_A/
    train_data/
      images/        IMG_1.jpg ... IMG_300.jpg
      ground-truth/  GT_IMG_1.mat ... GT_IMG_300.mat
    test_data/
      images/        IMG_1.jpg ... IMG_182.jpg
      ground-truth/  GT_IMG_1.mat ... GT_IMG_182.mat
  part_B/
    train_data/ ...
    test_data/ ...
```

Each `.mat` file contains `(x, y)` head coordinates for one image. Loaded via:

```python
mat = sio.loadmat(gt_path)
points = mat["image_info"][0, 0][0, 0][0]  # (N, 2) array
```

---

## 14.3 `__init__` — Setup (called once)

### Parameters

```python
CrowdCountingDataset(
    root,                  # path to ShanghaiTech/
    part="A",              # "A" (dense) or "B" (sparse)
    split="train",         # "train", "val", or "test"
    density_mode="fixed",  # "fixed" or "adaptive"
    sigma=15.0,            # Gaussian width for fixed mode
    k=4, beta=0.3,         # adaptive mode parameters
    target_size=None,      # (H, W) resize, or None for original
    val_split=0.0,         # fraction of train held out for val
    downsample_factor=1,   # reduce density map resolution (4 for MCNN, 8 for CSRNet)
    normalize=False,      # ImageNet normalization for pretrained VGG
)
```

### File discovery

```python
split_name = "train_data" if split in ("train", "val") else "test_data"
data_dir = self.root / f"part_{part}" / split_name
self.img_dir = data_dir / "images"
self.gt_dir = data_dir / "ground-truth"

all_images = sorted(self.img_dir.glob("*.jpg"))
all_gts = sorted(self.gt_dir.glob("*.mat"))
```

Both lists are sorted alphabetically so `IMG_1.jpg` always pairs with `GT_IMG_1.mat` at the same index. Sorting guarantees deterministic, reproducible indexing across machines.

### Train/val split

```python
n_val = max(1, int(len(all_images) * val_split))

# "val"   → last n_val images
# "train" → first len-n_val images
# "test"  → all images (no split)
```

The split is deterministic (no random seed) — the last 10% of the sorted train set becomes validation.

---

## 14.4 `__getitem__` — Load One Sample (called per sample, per epoch)

### Step 1: Load image and annotations

```python
image = Image.open(img_path).convert("RGB")   # PIL, lazy decode
orig_w, orig_h = image.size                    # PIL returns (W, H)
points = _load_annotations(gt_path)             # (N, 2) numpy array
```

`.convert("RGB")` guarantees 3 channels regardless of the original format (grayscale, RGBA, etc.).

### Step 2: Resize (if target_size is set)

```python
if self.target_size is not None:
    target_h, target_w = self.target_size       # (H, W) — torch convention
    image = image.resize((target_w, target_h))  # (W, H) — PIL convention!

    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    points[:, 0] *= scale_x     # scale x coordinates
    points[:, 1] *= scale_y     # scale y coordinates

    h, w = target_h, target_w
```

The image and the head coordinates are scaled **together** so they stay consistent. Note the PIL vs torch dimension order mismatch — PIL takes `(W, H)`, our `target_size` is `(H, W)`.

### Step 3: Downsample density map (if downsample_factor > 1)

```python
if self.downsample_factor > 1:
    factor = self.downsample_factor
    h_dens = h // factor       # e.g. 768 // 8 = 96
    w_dens = w // factor        # e.g. 1024 // 8 = 128

    points[:, 0] /= factor     # coordinates move to reduced grid
    points[:, 1] /= factor
    sigma = self.sigma / factor  # Gaussian width shrinks proportionally
else:
    h_dens, w_dens = h, w
    sigma = self.sigma
```

The **image stays at full resolution** — only the density target is reduced to match the model's output stride. The sigma must shrink too, otherwise the Gaussian would be wider than the entire reduced image.

### Step 4: Generate density map

```python
if self.density_mode == "fixed":
    density = fixed_sigma_density_map(points, h_dens, w_dens, sigma=sigma)
elif self.density_mode == "adaptive":
    density = adaptive_density_map(points, h_dens, w_dens, k=self.k, beta=self.beta)
```

Both functions return a `(h_dens, w_dens)` numpy array where `sum(density) ≈ number of heads`.

- **Fixed:** place 1 at each head, blur with `gaussian_filter(sigma)`. Fast (~5ms).
- **Adaptive:** per-head sigma from nearest-neighbour distances. Slow (~500ms) but better for dense crowds with scale variation.

### Step 5: Convert to tensors

```python
# Image: PIL (H, W, C) uint8 → tensor (C, H, W) float32 in [0, 1]
image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

# Optional ImageNet normalization for pretrained VGG (CSRNet)
if self.normalize:
    image_tensor = normalize_imagenet(image_tensor)

# Density: numpy (H, W) float32 → tensor (1, H, W) float32
density_tensor = torch.from_numpy(density).unsqueeze(0).float()

return image_tensor, density_tensor
```

The image goes through: decode → numpy → permute to channels-first → float → scale to [0,1] → optionally normalize.

The density map only needs: numpy → tensor → add channel dimension. No scaling — the sum must be preserved (it equals the head count).

---

## 14.5 Output Shapes

For `target_size=(768, 1024)`, `downsample_factor=8` (CSRNet):

```
__getitem__ returns:
  image_tensor:   (3, 768, 1024)    — full resolution, fed to model
  density_tensor: (1, 96, 128)       — reduced, matches model output

DataLoader with batch_size=4 produces:
  images:         (4, 3, 768, 1024)
  densities:      (4, 1, 96, 128)
```

The model takes `(4, 3, 768, 1024)` and outputs `(4, 1, 96, 128)` — same shape as the target, so MSE loss works without resizing.

---

## 14.6 Complete Trace for One Sample

```
Parameters: part="A", split="train", density_mode="fixed",
            target_size=(768, 1024), downsample_factor=8,
            normalize=True, sigma=15

__getitem__(42):

  Load:    image = PIL (1024×768 W×H), points = (1546, 2)

  Resize:  image = PIL (1024×768), points scaled to match
           h=768, w=1024

  Downsample (factor=8):
           h_dens=96, w_dens=128
           points = (1546, 2) ÷ 8
           sigma = 15 / 8 = 1.875

  Density: fixed_sigma_density_map(points, 96, 128, sigma=1.875)
           → (96, 128) numpy, sum ≈ 1546

  Tensors:
           image_tensor   = (3, 768, 1024)  normalized
           density_tensor = (1, 96, 128)    sum ≈ 1546

  Return:  (image_tensor, density_tensor)
```

---

## 14.7 The Two PR Fixes

Two changes were added after initial implementation:

### Downsample factor (issue #8)

Without this, the density map was full-resolution `(768, 1024)` but the model outputs `(96, 128)`. MSE loss crashed with a shape mismatch. The fix scales coordinates, resolution, and sigma by `1/factor` so the target matches the model output.

### ImageNet normalization (issue #9)

CSRNet uses a pretrained VGG16 frontend that expects inputs normalized with ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). Without normalization, the pretrained features are meaningless. The fix adds an optional `normalize=True` flag applied after the `/255.0` step.

---

## 14.8 Known Limitations

| Limitation | Issue | Impact |
|---|---|---|
| No density map caching | #12 | Adaptive mode regenerates ~500ms per image every epoch |
| No data augmentation | #14 | Only 300 images → overfitting risk |
| Config not wired to dataset | #13 | Config values exist but dataset doesn't use them automatically |
| Single-head edge case | #10 | Adaptive sigma crashes on 1 head (doesn't occur in ShanghaiTech) |
| `Image.BILINEAR` deprecated | #11 | Works now, will break in future Pillow versions |

These are tracked as GitHub issues and don't block training — the pipeline produces correctly-shaped tensors.