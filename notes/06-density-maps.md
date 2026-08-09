# 06 — Density Maps for Crowd Counting

A **density map** is a 2D heatmap where each pixel value represents the density of people at that location. Summing all pixel values gives the total crowd count.

---

## 6.1 From Dots to Density

The raw annotation for a crowd image is a set of (x, y) coordinates — one dot per head:

```
Image with annotations:        What the model sees:
┌──────────────────────┐       ┌──────────────────────┐
│    .        .   .    │       │    ?        ?   ?    │
│          .           │       │          ?           │
│  .   .        .      │       │  ?   ?        ?      │
│        .   .         │       │        ?   ?         │
│  .              .    │       │  ?              ?    │
└──────────────────────┘       └──────────────────────┘
  Ground truth: dots            Model output: density map
  (not useful for training)     (smooth, learnable)
```

A single dot at (x, y) is useless as a training target — the model would need to predict an exact 1 at one pixel and 0 everywhere else. That's nearly impossible and gives no gradient signal for pixels even slightly off.

**Solution:** spread each dot into a Gaussian blob. Now the target is smooth, and the model gets useful gradient information everywhere.

---

## 6.2 Anatomy of a Density Map

```
Density map (H × W × 1):

  0.0  0.0  0.1  0.3  0.1  0.0  0.0
  0.0  0.2  0.5  0.8  0.5  0.2  0.0
  0.0  0.5  1.2  1.5  1.2  0.5  0.0    ← peak at head location
  0.0  0.2  0.5  0.8  0.5  0.2  0.0
  0.0  0.0  0.1  0.3  0.1  0.0  0.0

  Sum ≈ 10.0  →  10 people in this region
```

**Properties:**
- Values are floats, not integers (a pixel can be 0.3 — "part of a person")
- Nearby heads overlap → their Gaussians add up → higher peak values
- Sum of all pixels ≈ total number of heads (slight loss due to clipping at edges)

---

## 6.3 How Ground Truth Density Maps Are Made

**Step 1:** Create an empty zero matrix of the same size as the image.

**Step 2:** For each (x, y) head coordinate, place a 1 at that position.

```
height, width = image.shape[:2]
density = np.zeros((height, width))

for (x, y) in head_coordinates:
    ix, iy = int(round(x)), int(round(y))
    density[iy, ix] = 1.0
```

**Step 3:** Convolve with a Gaussian kernel (blur).

```python
from scipy.ndimage import gaussian_filter
density = gaussian_filter(density, sigma=15)
```

**Step 4:** The result is the ground truth density map.

---

## 6.4 Fixed vs Adaptive Sigma

| Mode | Sigma | When to use |
|---|---|---|
| **Fixed** | Same σ for all heads (e.g. 15) | Consistent head sizes, simpler, faster |
| **Adaptive** | σ ∝ distance to k nearest neighbours | Varying head sizes (near vs far), more accurate |

**Fixed sigma** is simpler and commonly used for ShanghaiTech Part B (sparse, consistent scale).

**Adaptive sigma** (geometry-adaptive) is better for Part A (dense crowds with perspective distortion). Each head gets its own sigma:

```
σ_i = β × mean_distance_to_k_nearest_neighbours
```

Where β ≈ 0.3 and k ≈ 4. Dense regions get sharp peaks, sparse regions get wide peaks.

---

## 6.5 Why Density Maps Work for Training

1. **Smooth gradients** — every pixel has a non-zero target value, so every pixel contributes to the loss. The model learns *where* to put mass, not just *that* mass exists.

2. **Spatial awareness** — the model must learn the relationship between image features (heads, shoulders, background) and density. It can't just guess a global number.

3. **Handles occlusion** — overlapping heads blend naturally in the density map. The model doesn't need to separate them — it just needs to predict the right total mass.

4. **Interpretable** — you can visualize the predicted density map and see exactly where the model thinks people are.

---

## 6.6 Density Map vs Other Approaches

| Approach | Output | Pros | Cons |
|---|---|---|---|
| **Global count** | Single number | Simple | No spatial info, terrible accuracy |
| **Detection (bbox)** | Bounding boxes | Precise location | Fails in dense crowds (occlusion) |
| **Density map** | 2D heatmap | Handles density, interpretable | Needs Gaussian generation step |

Density maps are the standard for crowd counting because detection fails in dense crowds (too much overlap) and global counting loses too much information.
