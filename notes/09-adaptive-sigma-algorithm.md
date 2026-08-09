# 09 — Adaptive Sigma Algorithm

The **adaptive sigma** (geometry-adaptive) algorithm generates a density map where each head gets its own Gaussian kernel width based on local crowd density. Dense clusters get sharp peaks, isolated heads get wide peaks.

---

## 9.1 The Problem with Fixed Sigma

Fixed sigma uses the same Gaussian width for every head:

```
Fixed σ = 15 for all heads:

Dense cluster (50 heads):     Sparse area (2 heads):
  ╱╲ ╱╲ ╱╲ ╱╲ ╱╲              ●       ●
 ╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲
╱    ╲    ╲    ╲    ╲    ╲
  All merge into one blob     Fine, but unnecessarily wide
  → count information lost
```

In dense crowds, wide Gaussians overlap and smear together — the model can't distinguish individual heads. The density map loses count accuracy.

---

## 9.2 The Adaptive Sigma Formula

```
σ_i = β × d_i
```

Where:
- **σ_i** — sigma for head i
- **β** — scaling factor (typically 0.3)
- **d_i** — average distance to the k nearest neighbours of head i

$$
d_i = \frac{1}{k}\times\sum_{j=1..k} distance(head_i, neighbour_i)
$$

**Default parameters** (from MCNN paper):
- k = 4 (number of neighbours)
- β = 0.3 (scaling factor)
- fallback σ = 2.0 (when avg distance ≈ 0)

---

## 9.3 Step-by-Step Walkthrough

### Setup

Given a set of N head coordinates:

```
points = [[x₁, y₁], [x₂, y₂], ..., [xₙ, yₙ]]
```

### Step 1: Build a KDTree

A KDTree organizes points so nearest-neighbour queries are fast — O(log N) instead of O(N).

```python
from scipy.spatial import KDTree
tree = KDTree(points)
```

### Step 2: For each head, find k+1 nearest neighbours

We query k+1 because the head itself is always distance 0 (nearest neighbour to itself). We skip it.

```python
distances, indices = tree.query((x, y), k=k+1)
# distances[0] ≈ 0 (self)
neighbour_distances = distances[1:]  # skip self
```

### Step 3: Compute average distance

```python
avg_dist = np.mean(neighbour_distances)
```

### Step 4: Compute sigma

```python
sigma = beta * avg_dist
if sigma < 1e-5:
    sigma = fallback_sigma  # 2.0
```

### Step 5: Place Gaussian at (x, y) with this sigma

```python
radius = int(3 * sigma)
for dy in range(-radius, radius + 1):
    for dx in range(-radius, radius + 1):
        yy = y + dy
        xx = x + dx
        if 0 <= yy < height and 0 <= xx < width:
            density[yy, xx] += (1 / (2πσ²)) × exp(-(dx² + dy²) / (2σ²))
```

---

## 9.4 Worked Example

Five heads in a crowd:

```
Coordinates:
  A: (100, 100)
  B: (150, 110)
  C: (80,  150)
  D: (160, 140)
  E: (200, 130)
```

**For head B (150, 110):**

```
Neighbours (k=4, excluding self):
  D (160, 140):  √((160-150)² + (140-110)²) = √(100 + 900) = 31.6
  A (100, 100):  √((100-150)² + (100-110)²) = √(2500 + 100) = 51.0
  E (200, 130):  √((200-150)² + (130-110)²) = √(2500 + 400) = 53.9
  C (80, 150):   √((80-150)² + (150-110)²)  = √(4900 + 1600) = 80.6

avg_dist = (31.6 + 51.0 + 53.9 + 80.6) / 4 = 54.3
σ = 0.3 × 54.3 = 16.3
```

**For head D (160, 140):**

```
Neighbours:
  B (150, 110):  31.6
  E (200, 130):  41.2
  A (100, 100):  72.1
  C (80, 150):   80.0

avg_dist = (31.6 + 41.2 + 72.1 + 80.0) / 4 = 56.2
σ = 0.3 × 56.2 = 16.9
```

**For an isolated head** (if one existed far from others):

```
Neighbours are far → large avg_dist → large σ → wide Gaussian
```

---

## 9.5 Visual Intuition

```
Dense cluster (small σ):         Sparse area (large σ):

  ╱╲                              _____╱╲_____
 ╱  ╲                           ╱             ╲
╱    ╲                         ╱               ╲
  Sharp peak                    Wide, flat peak
  Stays distinct               Blends smoothly
  Preserves count               Fills the area
```

**The effect on the density map:**

```
Fixed σ = 15:                    Adaptive σ:
┌──────────────────────┐         ┌──────────────────────┐
│  ╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲    │         │   ╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲   │
│ ╱                  ╲ │         │  ╱                ╲  │
│╱                    ╲│         │ ╱                  ╲ │
│  All peaks merge     │         │  Peaks stay distinct │
│  into one blob       │         │  in dense areas      │
│  Sum ≈ 45 (lost 5)   │         │  Sum ≈ 50 (correct)  │
└──────────────────────┘         └──────────────────────┘
```

---

## 9.6 Edge Cases

| Scenario | What happens | Why |
|---|---|---|
| **Isolated head** | σ = fallback (2.0) | avg_dist ≈ 0 because the only neighbour is itself |
| **Single head in image** | σ = fallback (2.0) | k+1 > N, so only 1 neighbour (self), distances[1:] is empty |
| **Two heads very close** | Small σ, sharp peak | Correct — they're distinct, keep them separate |
| **Head at image edge** | Gaussian clipped by bounds check | Some mass lost, but negligible |

---

## 9.7 Adaptive vs Fixed — When to Use

| Criterion | Fixed sigma | Adaptive sigma |
|---|---|---|
| **Speed** | ~5 ms per image | ~500 ms per image (100× slower) |
| **Accuracy** | Good for uniform scale | Better for varying scale |
| **Part A (dense, perspective)** | Loses count in dense clusters | Preserves count |
| **Part B (sparse, uniform)** | Works well | Overkill |
| **Complexity** | One-liner | KDTree + per-head loop |

**Recommendation:** use fixed sigma for Part B, adaptive sigma for Part A.

---

## 9.8 Why KDTree?

A naive implementation would compute all pairwise distances:

```python
# O(N²) — terrible for 1500 heads
for each head:
    for each other head:
        compute distance
```

KDTree reduces this to O(N log N) build + O(log N) per query:

```python
# O(N log N) — fast
tree = KDTree(points)
for each head:
    distances = tree.query(head, k=5)  # O(log N)
```

For N = 1500, that's ~2.25M operations vs ~11K operations — a **200× speedup** just from the data structure.

---

## 9.9 The Bottleneck

The slow part is not the KDTree query — it's the per-head Gaussian placement:

```python
for each head:                          # 1500 iterations
    for dy in range(-radius, radius):   # ~6σ iterations
        for dx in range(-radius, radius):  # ~6σ iterations
            density[yy, xx] += ...      # 12.5M total iterations
```

This is pure Python nested loops. For σ ≈ 15, that's ~8,100 pixels per head × 1,500 heads = **12.5 million iterations**. Each iteration does exponentiation, division, and bounds checking.

**Possible optimizations** (see issue #7):
- Vectorize with `np.add.at`
- Use numba's `@njit`
- Precompute Gaussian patch lookup table
