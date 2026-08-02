# 01 — CNN Fundamentals

A **Convolutional Neural Network (CNN)** is a neural network designed for grid-structured data — most commonly images. Instead of every neuron connecting to every input (dense layer), CNNs use small **filters** that slide across the image, detecting local patterns.

---

## 1.1 The Convolution Operation

A small matrix (filter/kernel) slides across the input image. At each position, element-wise multiplication + sum produces one output number.

### Example

```
Input image (5×5):          Filter (3×3):         Output (3×3):
1  1  0  0  1              1  0  1                ?  ?  ?
1  0  0  1  1              0  1  0                ?  ?  ?
0  0  1  1  0              1  0  1                ?  ?  ?
0  1  1  0  0
1  0  0  0  1
```

**Step 1:** Place filter at top-left corner (covers rows 1-3, cols 1-3):

```
1  1  0         1  0  1
1  0  0   ×     0  1  0    =  (1×1)+(1×0)+(0×1)+(1×0)+(0×1)+(0×0)+(0×1)+(0×0)+(1×1)
0  0  1         1  0  1    =  1 + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 1 = 2
```

**Step 2:** Slide filter one column to the right (stride=1), repeat. After covering all positions:

```
Output (3×3):
2  3  2
3  4  3
2  3  2
```

Each value tells us: "how much did the filter pattern match at this location?"

### Key properties

- **Local connectivity:** each neuron sees only a small patch (e.g. 3×3), not the whole image
- **Weight sharing:** the same filter slides everywhere — translation invariant, far fewer parameters
- **Hierarchical features:** first layers detect edges → middle layers detect textures/patterns → deep layers detect objects/parts

---

## 1.2 The Anatomy of a CNN Layer

```
Input (H×W×C_in) → Conv(filter_size, C_in→C_out) → Output (H'×W'×C_out)
```

**Hyperparameters:**

| Parameter | What it does | Typical value |
|---|---|---|
| Kernel size | Size of the sliding window | 3×3, 5×5 |
| Stride | How many pixels the filter moves each step | 1 (dense), 2 (skip) |
| Padding | Zeros added around the border | "same" (output = input size), "valid" (no padding) |
| Channels (depth) | Number of filters in the layer | 64, 128, 256, 512 |

**Output size formula:**

```
Output_size = (Input_size - Kernel_size + 2×Padding) / Stride + 1
```

Example: 224×224 input, 3×3 kernel, stride=1, padding=1 → 224×224 output (same size).

---

## 1.3 Activation Function — ReLU

After every convolution, apply ReLU: `f(x) = max(0, x)`

```
Before ReLU:    -2  3  -1  0  5  -4
After ReLU:      0  3   0  0  5   0
```

Why? Convolution is linear (multiply + add). Stacking linear operations is still linear. ReLU adds non-linearity so the network can learn complex patterns.

---

## 1.4 Pooling

Downsamples the feature map — reduces width/height, keeps depth.

### Max Pooling (2×2, stride=2)

```
Input (4×4):                    Output (2×2):
1  3 │ 2  1                     max(1,3,2,4)=4 │ max(2,1,1,0)=2
2  4 │ 1  0                     ───────────────┼───────────────
─── ┼───                        max(0,1,1,2)=2 │ max(5,3,4,2)=5
0  1 │ 5  3
1  2 │ 4  2
```

**Why pool?**
- Reduces computation (smaller maps)
- Controls overfitting
- Adds translation invariance (small shifts don't change the max)
- Increases receptive field of later layers

**Trade-off:** you lose spatial precision — you know *that* a feature appeared, but not exactly *where*.

---

## 1.5 Fully Connected (Dense) Layer

At the end of a classification CNN, the 2D feature maps are **flattened** into a 1D vector and fed into regular dense layers.

```
Feature maps (7×7×512) → Flatten (25088) → Dense(4096) → Dense(4096) → Dense(1000)
```

Each neuron connects to every neuron in the previous layer. This is where the network makes the final decision.

---

## 1.6 Complete CNN Example — Image Classifier

```
Input (224×224×3)
│
├─ Conv(3×3, 3→64) + ReLU     → 224×224×64
├─ Conv(3×3, 64→64) + ReLU    → 224×224×64
├─ MaxPool(2×2, stride=2)      → 112×112×64
│
├─ Conv(3×3, 64→128) + ReLU    → 112×112×128
├─ Conv(3×3, 128→128) + ReLU   → 112×112×128
├─ MaxPool(2×2, stride=2)      → 56×56×128
│
├─ Conv(3×3, 128→256) + ReLU   → 56×56×256
├─ Conv(3×3, 256→256) + ReLU   → 56×56×256
├─ Conv(3×3, 256→256) + ReLU   → 56×56×256
├─ MaxPool(2×2, stride=2)      → 28×28×256
│
├─ Flatten                     → 200704
├─ Dense(4096) + ReLU
├─ Dropout(0.5)
├─ Dense(4096) + ReLU
├─ Dropout(0.5)
└─ Dense(1000) + Softmax      → class probabilities
```

This is essentially VGG11 — a simplified VGG. Each block doubles the channels and halves the spatial size.

---

## 1.7 What a CNN Learns (Visualized)

```
Layer 1 filters:     edges at different orientations
  [ -1  0  1 ]       [ -1 -1 -1 ]       [  1  0 -1 ]
  [ -1  0  1 ]       [  0  0  0 ]       [  1  0 -1 ]
  [ -1  0  1 ]       [  1  1  1 ]       [  1  0 -1 ]
  (vertical edge)     (horizontal edge)  (diagonal edge)

Layer 2-3:           textures, corners, repeating patterns
Layer 4-5:           object parts (eyes, wheels, windows)
Layer 6+:            whole objects (faces, cars, buildings)
```

This hierarchy emerges automatically from training — you don't design it.

---

## 1.8 CNN vs Dense Network — Parameter Comparison

For a 224×224 RGB image:

| Layer | Dense network | CNN (3×3 conv) |
|---|---|---|
| First layer | 224×224×3 = 150,528 inputs → each neuron has 150,528 weights | 3×3×3 = 27 weights per filter |
| 64 neurons/filters | 150,528 × 64 ≈ **9.6M params** | 27 × 64 = **1,728 params** |
| Translation invariance | No — move the cat 1 pixel, all activations change | Yes — filter slides, same response |

This is why CNNs are practical for images and dense networks are not.
