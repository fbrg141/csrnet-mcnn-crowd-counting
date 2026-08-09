# 15 — Evaluation Metrics (MAE and RMSE)

This note documents `src/utils/metrics.py` — the two functions used to evaluate crowd counting models. Both operate on **counts** (one scalar per image), not on density maps directly.

---

## 15.1 The Evaluation Flow

```
For each test image:
  model(image) → predicted density map → sum → predicted count
  ground truth density map → sum → actual count

Collect:
  actual_counts    = [1546, 233, 87, ...]   (one per image)
  predicted_counts = [1520, 240, 95, ...]   (one per image)

Compute:
  MAE(actual_counts, predicted_counts)
  RMSE(actual_counts, predicted_counts)
```

The metrics compare **counts**, not density maps. You sum each density map to get a count, then compare the counts across all test images.

---

## 15.2 MAE — Mean Absolute Error

```
MAE = (1/N) × Σ |predicted_i - actual_i|
```

For each image, take the absolute difference between predicted and actual count, then average across all images.

### Example

```
Actual:    [150,  200, 50, 1000, 300]
Predicted: [145,  210, 48, 980,  290]
Error:     [  5,   10,  2,   20,   10]   (absolute)

MAE = (5 + 10 + 2 + 20 + 10) / 5 = 9.4
```

**Interpretation:** "On average, the model is off by 9.4 people."

### Properties

- **Linear penalty:** an error of 20 is 20× worse than an error of 1
- **Robust to outliers:** a single bad prediction doesn't dominate
- **Same units as the count:** MAE of 9.4 means "9.4 people"
- **Easy to interpret:** directly answers "how wrong is the model on a typical image?"

---

## 15.3 RMSE — Root Mean Squared Error

```
RMSE = sqrt( (1/N) × Σ (predicted_i - actual_i)² )
```

Same idea as MAE, but squares the errors first, averages, then takes the square root.

### Example

```
Actual:    [150,  200, 50, 1000, 300]
Predicted: [145,  210, 48, 980,  290]
Error:     [  5,   10,  2,   20,   10]

Squared:  [ 25,  100,  4,   400,  100]

MSE  = (25 + 100 + 4 + 400 + 100) / 5 = 125.8
RMSE = sqrt(125.8) = 11.2
```

**Interpretation:** "The model's typical error is 11.2, weighted toward the worst predictions."

### Properties

- **Quadratic penalty:** an error of 20 is 400× worse than an error of 1 (squared)
- **Sensitive to outliers:** large errors dominate the score
- **Same units as the count** (after the square root): RMSE of 11.2 means "11.2 people"
- **Always ≥ MAE** (by Jensen's inequality) — if they're equal, all errors are the same size

---

## 15.4 MAE vs RMSE — What They Tell You

| Metric | Question it answers | Sensitivity |
|---|---|---|
| **MAE** | "How accurate is the model on a typical image?" | Linear — all errors contribute proportionally |
| **RMSE** | "How bad are the worst predictions?" | Quadratic — large errors are heavily penalized |

```
MAE = 9.4,  RMSE = 11.2   → consistent, small spread in errors
MAE = 9.4,  RMSE = 50.0   → most images are fine, but some are disasters
```

If MAE and RMSE are close, the model is consistent across images. If RMSE >> MAE, the model has occasional large failures — it gets most images right but completely misses on some.

### Why both?

Crowd counting papers report both because they tell different stories:
- **MAE** — overall accuracy, what a user would experience on average
- **RMSE** — stability, whether the model can be trusted on hard images

A model with low MAE but high RMSE is risky: it's good on average but unreliable on individual images. A model with both low is trustworthy.

---

## 15.5 The Code

```python
from math import sqrt
from typing import Iterable


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true_values = list(y_true)
    pred_values = list(y_pred)
    if len(true_values) != len(pred_values):
        raise ValueError("y_true and y_pred must have the same length")
    if not true_values:
        return 0.0
    return sum(abs(t - p) for t, p in zip(true_values, pred_values)) / len(true_values)


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    true_values = list(y_true)
    pred_values = list(y_pred)
    if len(true_values) != len(pred_values):
        raise ValueError("y_true and y_pred must have the same length")
    if not true_values:
        return 0.0
    mse = sum((t - p) ** 2 for t, p in zip(true_values, pred_values)) / len(true_values)
    return sqrt(mse)
```

### Key implementation details

**`list(y_true)`** — converts the input to a list. Needed because:
- Generators have no `len()` until consumed
- Generators can only be iterated once — we need two passes (length + sum)

**Length check** — fail fast if predictions and ground truth don't align:

```python
if len(true_values) != len(pred_values):
    raise ValueError(...)
```

**Empty check** — prevent division by zero:

```python
if not true_values:
    return 0.0
```

**`zip(true_values, pred_values)`** — pairs up corresponding elements:

```python
zip([150, 200, 50], [145, 210, 48])
# → [(150, 145), (200, 210), (50, 48)]
```

**`abs(t - p)` vs `(t - p) ** 2`** — the only difference between MAE and RMSE:
- MAE: absolute error (linear penalty)
- RMSE: squared error (quadratic penalty), then square root at the end

**`math.sqrt`** not `numpy.sqrt` — we're taking the square root of a single scalar (MSE), not an array. `math.sqrt` is lighter and more appropriate.

---

## 15.6 Why Pure Python (Not NumPy)

These functions use Python loops (`sum(... for ... in ...)`) instead of numpy vectorized operations. This is intentional:

```
Inputs:  182 counts (ShanghaiTech Part A test set)
         316 counts (ShanghaiTech Part B test set)

Pure Python:  sum of 300 numbers  →  microseconds
NumPy:       overhead of array creation  →  similar or slower
```

The bottleneck is **model inference** (running images through the network), not metric computation. For a few hundred scalar counts, pure Python is fast enough and keeps the code dependency-free.

We never compute MAE/RMSE on pixel-level density maps — crowd counting evaluation is always count-level. So the pure Python version will never need to be replaced.

---

## 15.7 How These Will Be Used

```python
from src.utils.metrics import mae, rmse

# After running inference on all test images
actual_counts = []
predicted_counts = []

for image, density in test_loader:
    prediction = model(image)
    actual_counts.append(density.sum().item())
    predicted_counts.append(prediction.sum().item())

print(f"MAE:  {mae(actual_counts, predicted_counts):.2f}")
print(f"RMSE: {rmse(actual_counts, predicted_counts):.2f}")
```

`.sum().item()` — sum the density map (get the count), then `.item()` extracts the scalar from the tensor (tensor → Python float).

---

## 15.8 Typical Results (ShanghaiTech, from papers)

| Model | Part A MAE | Part A RMSE | Part B MAE | Part B RMSE |
|---|---|---|---|---|
| MCNN (2016) | 110.2 | 173.2 | 26.4 | 41.3 |
| CSRNet (2018) | 68.2 | 115.0 | 10.6 | 16.0 |

Part A is harder (dense crowds, up to 3000 people) so errors are larger. Part B is easier (sparse, up to 500 people) so errors are smaller.

CSRNet outperforms MCNN on both parts — lower MAE and lower RMSE. The gap is larger on Part A (68 vs 110) because CSRNet's dilated convolutions handle dense crowds better than MCNN's pooling-based approach.