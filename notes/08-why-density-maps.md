# 08 — Why Density Maps? The Evolution of the Idea

Why go through the trouble of generating density maps instead of just counting dots directly? This note traces the reasoning.

---

## 8.1 The Naive Approach: Count the Dots

You have (x, y) head annotations. Why not just count them and train the model to output a single number?

```
Input image → CNN → Dense layer → 1 neuron → "247 people"
```

**This fails for three reasons:**

1. **No spatial signal** — the model sees the whole image and guesses a number. It has no idea *where* people are. Move everyone 10 pixels left → same count, completely different pixels → the model sees it as a different image.

2. **Scale blindness** — 100 people far away look like 10 people close up. A global count can't distinguish these.

3. **No interpretability** — the model outputs "247." If it's wrong, you have no idea why. Did it miss a crowd? Double-count? Guess?

**Result:** terrible accuracy, especially in dense crowds.

---

## 8.2 Step 1: From Classification to Regression

Standard CNNs classify (cat vs dog). Crowd counting is a regression problem (output a number).

```
Classification:  CNN → Softmax → [0.1, 0.7, 0.2] → "dog"
Regression:      CNN → Linear   → 247.0           → "247 people"
```

But a single number is still too lossy. The model needs spatial information.

---

## 8.3 Step 2: From Global to Local — The Density Map Insight

**Key realization:** instead of predicting one number, predict a map where the *sum* of the map is the count.

```
Global count:    247  (one number, no spatial info)
Density map:     ┌──────────────────────┐
                 │ 0.0 0.3 0.5 0.0 0.0  │
                 │ 0.2 0.8 1.2 0.4 0.0  │  ← each pixel = local density
                 │ 0.0 0.5 0.3 0.0 0.0  │
                 │ 0.0 0.0 0.0 0.1 0.0  │
                 └──────────────────────┘
                 Sum = 247.0
```

**Why this is better:**
- The model learns *where* people are
- Every pixel provides a training signal (not just one number)
- The output is interpretable (visualize where the model sees crowds)
- Occlusion is handled naturally (overlapping heads → higher density)

---

## 8.4 Step 3: From Hard Labels to Soft Targets

If we just placed 1s at head positions and 0s everywhere else:

```
Hard target (delta map):            Soft target (density map):
┌──────────────────────┐            ┌──────────────────────┐
│ 0 0 0 0 0 0 0 0 0 0  │            │ 0.0 0.0 0.1 0.3 0.1  │
│ 0 0 0 0 0 0 0 0 0 0  │            │ 0.0 0.2 0.5 0.8 0.5  │
│ 0 0 0 1 0 0 0 0 0 0  │            │ 0.0 0.5 1.2 1.5 1.2  │
│ 0 0 0 0 0 0 0 0 0 0  │            │ 0.0 0.2 0.5 0.8 0.5  │
│ 0 0 0 0 0 0 0 0 0 0  │            │ 0.0 0.0 0.1 0.3 0.1  │
└──────────────────────┘            └──────────────────────┘
  Loss: 0 everywhere except         Loss: smooth gradient everywhere
  at the exact pixel                    → model learns gradually
  → no gradient, can't learn
```

**Hard targets** (delta peaks) are impossible to learn — the model must predict exactly 1 at one pixel and 0 everywhere else. Any off-by-one prediction gives the same loss as being completely wrong.

**Soft targets** (Gaussian blobs) give the model useful gradient information everywhere. Predict slightly off-center? The loss is slightly higher, and the gradient points toward the correct location.

---

## 8.5 Step 4: From Fixed to Adaptive Gaussians

Early methods used a fixed sigma for all heads. But heads at different distances have different sizes:

```
Nearby head (30×30 px):       Far head (5×5 px):
  ╱╲                           ╱╲
 ╱  ╲                         ╱  ╲
╱    ╲                       ╱    ╲
      wide Gaussian                narrow Gaussian
      σ = 15                        σ = 4
```

**Adaptive sigma** solves this by making σ proportional to the distance to the nearest neighbour:

```
σ_i = β × avg_distance_to_k_nearest
```

Dense crowds → small σ (sharp peaks, heads stay distinct)
Sparse crowds → large σ (wide peaks, smooth coverage)

---

## 8.6 The Full Evolution

```
Raw annotations (dots)
    │
    ▼
Hard delta map (1s at head positions)
    │
    ├── Problem: impossible to learn (no gradient)
    │
    ▼
Fixed-sigma density map (Gaussian blur with constant σ)
    │
    ├── Problem: doesn't handle scale variation
    │
    ▼
Adaptive density map (σ ∝ nearest neighbour distance)
    │
    ├── Best for dense crowds with perspective
    │
    ▼
What the model learns: image → density map → sum = count
```

---

## 8.7 Summary: Why Density Maps

| Reason | Explanation |
|---|---|
| **Spatial supervision** | Every pixel teaches the model where people are |
| **Smooth gradients** | Gaussian blur creates learnable targets (not impossible delta peaks) |
| **Sum = count** | Simple evaluation: sum the map, compare to ground truth |
| **Interpretable** | Visualize the map to see where the model is confident or wrong |
| **Handles occlusion** | Overlapping heads blend naturally — no need to separate them |
| **Scale robust** | Adaptive sigma handles near and far heads in one framework |

Without density maps, crowd counting would require object detection in scenes where individual heads are barely visible. Density maps make the problem tractable by reframing it as density estimation instead of detection.
