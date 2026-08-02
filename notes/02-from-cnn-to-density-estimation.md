# 02 — From CNN to Crowd Counting: The Density Map

A standard CNN classifier outputs a single label. For crowd counting, we need a **count** — and ideally a **spatial distribution** of where people are.

---

## 2.1 The Naive Approach (Why It Fails)

Take a standard CNN, replace the softmax with a single neuron, and train it to output a number.

```
Input → Conv → Pool → Conv → Pool → Conv → Flatten → Dense → Dense → 1 neuron → count
```

**Problems:**

1. **No spatial awareness** — the network doesn't know *where* people are. It just sees a global feature vector and guesses a number. Move people around → same count, but the network sees completely different features.

2. **Scale ambiguity** — a head at 5 pixels and a head at 50 pixels look completely different. A single filter can't capture both.

3. **Pooling destroys location** — after 3-4 pooling layers, a 512×512 image becomes 32×32 or smaller. You've lost the fine-grained spatial info needed to distinguish "10 people close together" from "1 person far away."

4. **No interpretability** — you get a number. You can't see *where* the network thinks people are. If it's wrong, you have no idea why.

**Result:** terrible accuracy, especially in dense crowds.

---

## 2.2 The Key Insight: Density Maps

Instead of predicting a single number, predict a **density map** — an image of the same spatial size as the input, where each pixel value represents "how many people are at this location."

```
Input image (512×512)          Density map (512×512)
     [RGB pixels]                  [0.0  0.0  0.0  0.0  0.0]
                                    [0.0  0.3  0.4  0.0  0.0]
                                    [0.0  0.0  0.0  0.5  0.0]
                                    [0.0  0.2  0.3  0.0  0.0]
                                    [0.0  0.0  0.0  0.0  0.0]
```

**Sum of all pixel values = total count.**

### How ground truth density maps are made

1. Each person is annotated with a single dot at the center of their head: `(x, y)`
2. This dot is convolved with a Gaussian kernel to spread the mass:

```
Ground truth:     After Gaussian:
. . . . . .      0.0 0.0 0.0 0.0 0.0
. . . . . .      0.0 0.1 0.2 0.1 0.0
. . X . . .  →   0.0 0.2 0.5 0.2 0.0
. . . . . .      0.0 0.1 0.2 0.1 0.0
. . . . . .      0.0 0.0 0.0 0.0 0.0
```

The Gaussian spread accounts for:
- Head size variation (larger sigma for bigger heads)
- Overlap between nearby heads (they blend together naturally)
- The fact that a head occupies an area, not a point

**Geometry-adaptive kernels** (used in MCNN): sigma is proportional to the distance to the nearest neighbor, so dense crowds get smaller Gaussians (sharper peaks) and sparse crowds get larger ones.

---

## 2.3 Fully Convolutional Networks (FCN)

To output a density map, the network must be **fully convolutional** — no dense layers, no flattening. Everything is conv, pool, conv, pool, conv, and the final layer is a 1×1 conv that reduces channels to 1.

```
Input (H×W×3) → Conv → Pool → Conv → Pool → Conv → Conv(1×1) → Density map (H'×W'×1)
```

**Why no dense layers?**
- Dense layers require fixed input size (flatten produces a fixed-length vector)
- Dense layers destroy spatial structure (2D → 1D)
- Convolution preserves spatial relationships

**The 1×1 conv trick:** a convolution with a 1×1 kernel and 1 output channel. It's equivalent to a dense layer applied independently at each spatial position — it mixes information across channels without mixing across space.

---

## 2.4 The Resolution Problem

Here's the tension:

| We want | But pooling does |
|---|---|
| Large receptive field (see the whole image) | Shrinks the map |
| High-resolution output (sharp density map) | Loses detail |
| Deep network (rich features) | More pooling |

**Example:** A VGG-style network with 5 pooling layers turns 512×512 → 16×16. The density map is tiny and blurry.

**Two strategies to solve this:**

| Strategy | Representative | How it works |
|---|---|---|
| **Multi-column** | MCNN | Three parallel columns with different filter sizes, no deep pooling |
| **Dilated convolution** | CSRNet | Remove pooling, use dilated convs to grow receptive field |

These are the subjects of the next two files.

---

## 2.5 Evaluation Metrics

Crowd counting models are evaluated on:

**MAE (Mean Absolute Error):**
```
MAE = (1/N) × Σ |predicted_count - actual_count|
```
Average error in count. Lower is better.

**MSE (Mean Squared Error):**
```
MSE = (1/N) × Σ (predicted_count - actual_count)²
```
Penalizes large errors more heavily. Lower is better.

**Pixels vs count:** MAE/MSE are computed on the *sum* of the density map, not on individual pixels. Pixel-wise loss (MSE between predicted and ground truth density maps) is used during training, but evaluation is count-based.

---

## 2.6 Summary: The Evolution Path

```
Standard CNN (classifier)
    │
    ▼
Fully Convolutional Network (density map output)
    │
    ├──→ MCNN: multiple columns for scale variation
    │
    └──→ CSRNet: dilated convs for resolution preservation
```

The next two files dive into each.
