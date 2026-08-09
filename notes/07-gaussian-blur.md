# 07 — Gaussian Blur (Gaussian Filter)

**Gaussian blur** is a convolution operation that smooths an image by averaging each pixel with its neighbours, weighted by a Gaussian (bell curve) distribution.

---

## 7.1 The Gaussian Function



$$G(x,y)=\frac{1}{2\pi\sigma^2}e^{-\frac{x^2+y^2}{2\sigma^2}}$$


Where:
- **σ (sigma)** — controls the width of the curve (how much spread)
- **x, y** — distance from the center of the kernel
- The function is **normalized** so the sum of all values = 1

### Visual shape

```
σ = 1:                    σ = 3:
                             ___
     ╱╲                     ╱   ╲
    ╱  ╲                   ╱     ╲
   ╱    ╲                 ╱       ╲
  ╱______╲               ╱         ╲
   narrow                   wide
 (sharp blur)           (smooth blur)
```

---

## 7.2 The Gaussian Kernel

A discrete kernel is sampled from the Gaussian function. For σ = 1, a 5×5 kernel:

```
Raw values (before normalization):
    0.00  0.08  0.14  0.08  0.00
    0.08  0.37  0.61  0.37  0.08
    0.14  0.61  1.00  0.61  0.14
    0.08  0.37  0.61  0.37  0.08
    0.00  0.08  0.14  0.08  0.00

Normalized (sum = 1):
    0.00  0.02  0.04  0.02  0.00
    0.02  0.11  0.18  0.11  0.02
    0.04  0.18  0.29  0.18  0.04
    0.02  0.11  0.18  0.11  0.02
    0.00  0.02  0.04  0.02  0.00
```

**Key property:** the center has the highest weight. Values drop off smoothly. Pixels far from the center contribute almost nothing.

---

## 7.3 How Gaussian Blur Works

It's a convolution — same operation as a CNN layer, but with fixed weights:

```
Input patch (5×5):        Gaussian kernel (5×5):       Output:
  0   0   0   0   0         0.00 0.02 0.04 0.02 0.00
  0   0   1   0   0         0.02 0.11 0.18 0.11 0.02
  0   0   0   0   0    ×    0.04 0.18 0.29 0.18 0.04  =  0.29
  0   0   0   0   0         0.02 0.11 0.18 0.11 0.02
  0   0   0   0   0         0.00 0.02 0.04 0.02 0.00

  (1 at center)            (kernel)                   (weighted sum)
```

The kernel slides across the entire image, producing a blurred output.

---

## 7.4 The Role of Sigma (σ)

| σ | Kernel size needed | Effect |
|---|---|---|
| 1 | 5×5 | Light blur, preserves edges |
| 3 | 13×13 | Medium blur |
| 5 | 21×21 | Heavy blur |
| 15 | 61×61 | Very heavy blur (used for ShanghaiTech Part B) |

**Rule of thumb:** kernel radius = 3σ. Values beyond 3σ are ≈ 0 and can be ignored.

### In crowd counting

- **σ = 15** (ShanghaiTech Part B): wide Gaussian, heads blend smoothly
- **σ = 4** (adaptive, dense crowd): sharp Gaussian, heads remain distinct
- **σ = 2** (fallback for isolated heads): very sharp, almost a dot

---

## 7.5 Why Gaussian and Not Something Else

| Filter | Shape | Why not |
|---|---|---|
| **Gaussian** | ╱╲ Smooth bell curve | ✅ Natural blur, preserves sum |
| **Box (average)** | ⎯ Uniform square | ❌ Creates blocky artifacts, hard edges |
| **Median** | Takes middle value | ❌ Non-linear, sum not preserved |
| **Triangle** | ╱╲ Linear falloff | ❌ Less natural, same cost as Gaussian |

**The Gaussian is the only filter that:**
- Is **smooth** (no hard edges → no artifacts)
- Is **normalized** (sum = 1 → total count preserved)
- Is **separable** (2D = 1D horizontal × 1D vertical → fast)
- **Matches real-world optics** (point sources blur into Gaussians naturally)

---

## 7.6 Separability

A 2D Gaussian can be decomposed into two 1D passes:

```
G(x,y) = G(x) × G(y)

2D kernel (5×5):     =    1D horizontal (1×5)  ×  1D vertical (5×1)
  0.00 0.02 0.04 0.02 0.00
  0.02 0.11 0.18 0.11 0.02          [0.00 0.02 0.04 0.02 0.00]
  0.04 0.18 0.29 0.18 0.04     =          ↑
  0.02 0.11 0.18 0.11 0.02          [0.00 0.02 0.04 0.02 0.00]ᵀ
  0.00 0.02 0.04 0.02 0.00
```

**Why this matters:** a 61×61 2D kernel = 3,721 multiplications per pixel. Two 1D passes = 61 + 61 = 122 multiplications per pixel. **~30× faster.**

---

## 7.7 Gaussian Blur in scipy

```python
from scipy.ndimage import gaussian_filter

# Fixed sigma
density = gaussian_filter(delta_map, sigma=15)

# Different sigma per axis (if image is not square)
density = gaussian_filter(delta_map, sigma=(15, 15))

# Mode controls edge handling
density = gaussian_filter(delta_map, sigma=15, mode='constant')
# mode='constant' — pad with zeros outside image bounds
# mode='reflect'  — mirror the image at edges (default)
```

The `gaussian_filter` function handles kernel creation, separability optimization, and edge padding automatically.
