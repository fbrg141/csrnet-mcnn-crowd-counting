# 13 — Image Preprocessing Conventions

Image processing libraries disagree on dimension ordering. This note documents the conventions and the common bugs they cause.

---

## 13.1 The Three Dimension Orders

Different libraries use different orders for the same image:

| Library | Dimension order | Example |
|---|---|---|
| **PIL (Pillow)** | `(width, height)` | `image.size` → `(1024, 768)` |
| **NumPy** | `(height, width, channels)` | `np.array(image).shape` → `(768, 1024, 3)` |
| **PyTorch** | `(channels, height, width)` | `tensor.shape` → `(3, 768, 1024)` |
| **OpenCV** | `(height, width, channels)` | same as numpy |

```
PIL:         (W, H)          = (1024, 768)
NumPy:       (H, W, C)       = (768, 1024, 3)
PyTorch:     (C, H, W)       = (3, 768, 1024)
```

This is the #1 source of bugs in image pipelines.

---

## 13.2 The Conversion Chain

In our pipeline, an image goes through three formats:

```
Disk (.jpg)
    │
    │  Image.open(path).convert("RGB")
    ▼
PIL Image:  size = (1024, 768)  ← (width, height)
    │
    │  np.array(image)
    ▼
NumPy array: shape = (768, 1024, 3)  ← (height, width, channels)
    │
    │  torch.from_numpy(arr).permute(2, 0, 1)
    ▼
PyTorch tensor: shape = (3, 768, 1024)  ← (channels, height, width)
    │
    │  DataLoader adds batch dim
    ▼
Batch tensor: shape = (4, 3, 768, 1024)  ← (batch, channels, height, width)
```

### Each step explained

**PIL → NumPy:**
```python
image = Image.open("IMG_1.jpg").convert("RGB")
print(image.size)              # (1024, 768)  — width, height

arr = np.array(image)
print(arr.shape)               # (768, 1024, 3)  — height, width, channels
```

PIL stores size as `(width, height)` but when converted to numpy, the array is `(height, width, channels)`. PIL handles this internally — the conversion just flips the representation.

**NumPy → PyTorch:**
```python
# numpy: (H, W, C) = (768, 1024, 3)  — channels last
# torch: (C, H, W) = (3, 768, 1024)  — channels first

tensor = torch.from_numpy(arr).permute(2, 0, 1)
#                                ↑  ↑  ↑
#                         new axis order: C, H, W
#                         old was:        H, W, C
```

`permute(2, 0, 1)` means: "take old axis 2 (channels) and make it axis 0, take old axis 0 (height) and make it axis 1, take old axis 1 (width) and make it axis 2."

```
Old:  axis 0 = height,  axis 1 = width,  axis 2 = channels
New:  axis 0 = channels, axis 1 = height, axis 2 = width
```

PyTorch uses "channels first" (NCHW format) because it's more efficient for GPU convolution — accessing all channels of one pixel is a contiguous memory read.

---

## 13.3 The PIL Resize Trap

PIL's `resize` takes `(width, height)`, not `(height, width)`:

```python
image = Image.open("IMG_1.jpg")   # size = (1024, 768)

# CORRECT: pass (width, height)
image.resize((1024, 768))   # 1024 wide, 768 tall  ✓

# WRONG: passing (height, width) by habit
image.resize((768, 1024))  # 768 wide, 1024 tall  ✗ (transposed!)
```

In our code, `target_size` is stored as `(height, width)` to match PyTorch conventions, but PIL needs `(width, height)`:

```python
target_h, target_w = self.target_size    # (768, 1024) — H, W
image = image.resize((target_w, target_h))  # (1024, 768) — W, H for PIL
```

This is why we unpack into separate named variables — it makes the order explicit and prevents silent transposition bugs.

---

## 13.4 Coordinate Scaling

When an image is resized, head coordinates must scale proportionally:

```
Original image: 640×480 (W×H)
Head at (320, 240)

Resized to: 1024×768 (W×H)

scale_x = 1024 / 640 = 1.6
scale_y = 768  / 480 = 1.6

new_x = 320 × 1.6 = 512
new_y = 240 × 1.6 = 384
```

The head moves from center of the old image to center of the new image — same relative position.

### Why separate scale_x and scale_y?

If the aspect ratio changes, x and y scale differently:

```
Original: 800×600
Resized:   1024×500  (squished vertically)

scale_x = 1024 / 800 = 1.28   (wider)
scale_y = 500  / 600 = 0.833  (shorter)
```

A head at (400, 300):
```
new_x = 400 × 1.28 = 512
new_y = 300 × 0.833 = 250
```

If we used a single scale, the head would be in the wrong vertical position.

---

## 13.5 Normalization

### The two-step process

```python
# Step 1: Scale from [0, 255] to [0, 1]
image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

# Step 2: Normalize with ImageNet stats
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
image_tensor = (image_tensor - mean) / std
```

### Why divide by 255?

Image pixels are stored as `uint8` (0-255). Neural networks work best with small float values centered around zero:

```
uint8:    [0, 255]     — range 255, integers
float:    [0.0, 1.0]   — range 1.0, floats
```

Large input values cause large gradients, which cause unstable training. Dividing by 255 normalizes the range to [0, 1].

### Why ImageNet normalization?

Pretrained models (like VGG16) were trained on ImageNet with inputs normalized to have **mean=0, std=1** per channel. If you feed them [0, 1] inputs instead, the first layer's features are computed in a different distribution than what it was trained for — the pretrained weights become useless.

```python
# After /255:      [0, 1]  — mean ~0.5, std ~0.3
# After normalize:  ~mean 0, std 1  — standardized per channel
```

### The `.view(3, 1, 1)` trick

```python
mean = torch.tensor([0.485, 0.456, 0.406])  # shape (3,)
mean = mean.view(3, 1, 1)                    # shape (3, 1, 1)
```

The image is `(3, H, W)`. The mean is `(3,)`. To subtract a per-channel mean from every pixel, reshape to `(3, 1, 1)` so broadcasting stretches it across H and W:

```
Image:  (3, 768, 1024)
Mean:   (3, 1, 1)        ← broadcasts to (3, 768, 1024)
Result: (3, 768, 1024)    ← same shape, per-channel subtraction
```

Without `.view(3, 1, 1)`, subtracting a `(3,)` tensor from a `(3, 768, 1024)` tensor would fail or do the wrong thing.

---

## 13.6 Common Bugs

### Bug 1: Transposed image (W/H swap)

```python
# WRONG: using target_size (H, W) directly with PIL
image.resize((768, 1024))   # PIL interprets as (W=768, H=1024) → transposed!

# CORRECT: swap to (W, H) for PIL
image.resize((1024, 768))   # PIL interprets as (W=1024, H=768) → correct
```

### Bug 2: Forgetting .permute()

```python
# WRONG: feeding (H, W, C) to a model that expects (C, H, W)
tensor = torch.from_numpy(np.array(image))  # (768, 1024, 3)
model(tensor)  # RuntimeError: expected 3 channels, got 1024

# CORRECT: permute to (C, H, W)
tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)  # (3, 768, 1024)
```

### Bug 3: Forgetting to scale coordinates after resize

```python
# WRONG: resize image but keep original coordinates
image = image.resize((1024, 768))
points = load_annotations(...)   # still in original pixel space!
# → density map puts heads in wrong locations

# CORRECT: scale coordinates to match
scale_x = 1024 / orig_w
scale_y = 768 / orig_h
points[:, 0] *= scale_x
points[:, 1] *= scale_y
```

### Bug 4: Not converting to float before division

```python
# WRONG: integer division
arr = np.array(image)  # uint8
result = arr / 255     # integer division! 200/255 = 0 (not 0.784)

# CORRECT: convert to float first
result = arr.astype(np.float32) / 255.0  # 200/255 = 0.784
# or use torch:
result = torch.from_numpy(arr).float() / 255.0
```

---

## 13.7 Summary

| Convention | Order | Where used |
|---|---|---|
| **PIL** | `(width, height)` | `image.size`, `image.resize()` |
| **NumPy** | `(height, width, channels)` | `np.array(image).shape` |
| **PyTorch** | `(channels, height, width)` | `tensor.shape`, model input |
| **PyTorch batch** | `(batch, channels, height, width)` | DataLoader output |

| Operation | Code |
|---|---|
| PIL → tensor | `torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0` |
| Add channel dim | `density.unsqueeze(0)` — `(H, W)` → `(1, H, W)` |
| Per-channel normalize | `(image - mean.view(3,1,1)) / std.view(3,1,1)` |
| Scale coordinates | `points[:, 0] *= new_w / old_w; points[:, 1] *= new_h / old_h` |