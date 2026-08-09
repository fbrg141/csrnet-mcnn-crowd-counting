# 10 — Tensors

A **tensor** is a multi-dimensional array. It's the generalized form of scalars, vectors, and matrices — any array of numbers with any number of dimensions is a tensor.

---

## 10.1 The Tensor Hierarchy

| Dimensions | Name | Math term | Example | Shape |
|---|---|---|---|---|
| 0D | scalar | scalar | `5` | `()` |
| 1D | vector | vector | `[1, 2, 3]` | `(3,)` |
| 2D | matrix | matrix | `[[1, 2], [3, 4]]` | `(2, 2)` |
| 3D | tensor | — | a color image (channels × height × width) | `(3, 768, 1024)` |
| 4D | tensor | — | a batch of images | `(4, 3, 768, 1024)` |
| nD | tensor | — | any number of dimensions | `(...)` |

Every tensor has:
- **`shape`** — the size of each dimension
- **`ndim`** — the number of dimensions
- **`dtype`** — the data type (float32, int64, etc.)
- **`device`** — where it lives (CPU or GPU)

```
Scalar:     5                    shape: ()         ndim: 0
Vector:     [1, 2, 3]            shape: (3,)       ndim: 1
Matrix:     [[1, 2],             shape: (2, 2)     ndim: 2
             [3, 4]]
3D tensor:  [[[1], [2]],         shape: (2, 2, 1)  ndim: 3
              [[3], [4]]]
```

---

## 10.2 Why "Tensor" and Not "Array"

Numpy already uses `np.array`. PyTorch created `torch.Tensor` because tensors need three things numpy arrays can't do:

1. **Track gradients** — for automatic differentiation (backpropagation)
2. **Run on GPU** — numpy arrays stay on CPU
3. **Integrate with the autograd engine** — makes `loss.backward()` work

A PyTorch tensor is a numpy-like array that also knows how to compute gradients and move to GPU.

---

## 10.3 Numpy vs PyTorch — API Comparison

They are almost identical:

```python
# Numpy
import numpy as np
arr = np.zeros((3, 768, 1024))
arr[0, 0, 0]       # indexing
arr.shape           # (3, 768, 1024)
arr.transpose()     # swap axes
arr + arr           # element-wise addition

# PyTorch
import torch
t = torch.zeros((3, 768, 1024))
t[0, 0, 0]         # indexing (same syntax)
t.shape             # torch.Size([3, 768, 1024])
t.transpose(0, 1)   # swap axes
t + t               # element-wise addition (same)
```

### Key differences

```python
# GPU
arr              # always on CPU
t.cuda()          # moves to GPU
t.to('cuda')      # same, more explicit

# Gradients
arr              # no gradient tracking
t.requires_grad = True   # tracks operations for backprop
t.sum().backward()        # computes gradients

# Conversion (shares memory — no copy)
t = torch.from_numpy(arr)   # numpy → torch
arr = t.numpy()             # torch → numpy (CPU only)
```

---

## 10.4 Shapes in Our Pipeline

### Single sample (what `__getitem__` returns)

```
Image tensor:   shape (3, 768, 1024)    ndim 3
                 ↑  ↑    ↑
                channels height width
                (3 = R, G, B)

Density tensor: shape (1, 96, 128)      ndim 3
                 ↑  ↑   ↑
                channels height width
                (1 = single density channel)
```

### Batched (what DataLoader produces)

```
Batched images:   shape (4, 3, 768, 1024)   ndim 4
                   ↑  ↑  ↑    ↑
                  batch ch  h    w

Batched densities: shape (4, 1, 96, 128)   ndim 4
                   ↑  ↑  ↑   ↑
                  batch ch  h   w
```

The DataLoader adds a **batch dimension** as the first axis. A batch of 4 images is `(4, 3, 768, 1024)`, not 4 separate `(3, 768, 1024)` tensors.

---

## 10.5 Common Tensor Operations in Our Pipeline

### `permute` — reorder dimensions

```python
# PIL image → numpy array: shape (H, W, C) = (768, 1024, 3)
# PyTorch model expects:   shape (C, H, W) = (3, 768, 1024)

image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)
#                                              ↑  ↑  ↑
#                                           new axis order: (C, H, W)
#                                           original was:   (H, W, C)
```

`permute(2, 0, 1)` means: "make the old axis 2 the new axis 0, old axis 0 the new axis 1, old axis 1 the new axis 2."

### `unsqueeze` — add a dimension

```python
# Density map: shape (96, 128) — 2D
# Model expects: shape (1, 96, 128) — 3D with channel dimension

density_tensor = torch.from_numpy(density).unsqueeze(0)
#                                              ↑
#                                        add dimension at position 0

# (96, 128) → (1, 96, 128)
```

### `.float()` — convert dtype

```python
image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
#                                                     ↑
#                              convert from uint8 (0-255) to float32 (0.0-1.0)
```

Numpy image arrays are `uint8` (integers 0-255). Division by 255 requires floats.

### Arithmetic — broadcasting

```python
# Image: (3, H, W), values in [0, 255]
# Divide every pixel by 255 → values in [0, 1]
image_tensor / 255.0

# The 255.0 is a scalar — it "broadcasts" to every element
# (3, H, W) / scalar = (3, H, W)
```

---

## 10.6 Memory Sharing

`torch.from_numpy()` and `.numpy()` **share memory** — no copy is made:

```python
arr = np.zeros((3, 768, 1024))
t = torch.from_numpy(arr)

arr[0, 0, 0] = 42
print(t[0, 0, 0])  # → 42  (same memory!)

t[0, 0, 1] = 99
print(arr[0, 0, 1])  # → 99  (same memory!)
```

This is fast but dangerous — modifying one modifies the other. In our pipeline, the numpy array goes out of scope after `__getitem__` returns, so there's no risk of accidental mutation.

---

## 10.7 Tensors on GPU

```python
# Create on CPU
t = torch.zeros((3, 768, 1024))

# Move to GPU
t = t.cuda()
# or
t = t.to('cuda')

# All operations now run on GPU
result = t @ t.T   # matrix multiply on GPU

# Models must also be on GPU
model = CSRNet().cuda()

# Both model and input must be on the same device
output = model(image.cuda())   # image must be on cuda too
```

**Common bug:** model on GPU, input on CPU (or vice versa):

```python
model = CSRNet().cuda()
image = torch.zeros((3, 768, 1024))   # still on CPU!

output = model(image)
# RuntimeError: Expected all tensors to be on the same device
```

---

## 10.8 Summary

| Concept | Key point |
|---|---|
| **Tensor** | Multi-dimensional array (generalized scalar/vector/matrix) |
| **Shape** | Size of each dimension, e.g. `(3, 768, 1024)` |
| **ndim** | Number of dimensions |
| **vs numpy** | Same API + gradients + GPU |
| **permute** | Reorder dimensions (e.g. HWC → CHW) |
| **unsqueeze** | Add a dimension (e.g. add channel axis) |
| **from_numpy** | Convert numpy → torch (shares memory) |
| **requires_grad** | Enable gradient tracking for backprop |
| **cuda()** | Move to GPU |