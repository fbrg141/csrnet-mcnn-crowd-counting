# 12 — PyTorch Dataset & DataLoader

PyTorch separates **what data you have** (Dataset) from **how you feed it to the model** (DataLoader). This separation makes the pipeline modular and efficient.

---

## 12.1 The Two Concepts

**Dataset** — defines how to get one sample. You write this.

**DataLoader** — handles batching, shuffling, and parallel loading. PyTorch provides this.

```
Dataset:    "give me sample #42"  →  (image, density_map)
DataLoader: "give me a batch of 4 shuffled samples"  →  (4 images, 4 density_maps)
```

---

## 12.2 The Dataset Class

To create a custom dataset, subclass `torch.utils.data.Dataset` and implement two methods:

```python
from torch.utils.data import Dataset

class CrowdCountingDataset(Dataset):
    def __init__(self, ...):
        # Setup: find files, store config
        # Called ONCE when the dataset is created
        ...

    def __len__(self) -> int:
        # How many samples exist?
        # Called by DataLoader to know the dataset size
        return len(self.images)

    def __getitem__(self, index: int) -> tuple:
        # Load and return ONE sample
        # Called for every sample, every epoch
        ...
        return image_tensor, density_tensor
```

### When each method is called

| Method | When | How often |
|---|---|---|
| `__init__` | When you create the dataset object | Once |
| `__len__` | When DataLoader needs the total count | Occasionally (progress bars, epoch sizing) |
| `__getitem__` | When DataLoader needs a specific sample | Once per sample per epoch |

`__init__` does the **cheap** work — find file paths, store configuration. `__getitem__` does the **expensive** work — load image from disk, generate density map, convert to tensor. This lazy loading means you don't load 1,000 images into memory at startup; you load them one at a time as needed.

---

## 12.3 The DataLoader

```python
from torch.utils.data import DataLoader

dataset = CrowdCountingDataset(root="data/raw/ShanghaiTech", part="A", split="train")
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

for images, densities in loader:
    # images:    (4, 3, 768, 1024)  — batch of images
    # densities: (4, 1, 96, 128)    — batch of density maps
    predictions = model(images)
    loss = loss_fn(predictions, densities)
    loss.backward()
    optimizer.step()
```

### What DataLoader does

| Feature | What it handles |
|---|---|
| **Batching** | Calls `__getitem__` for multiple indices, stacks results into a batch tensor |
| **Shuffling** | Randomizes sample order each epoch (only for training) |
| **Parallel loading** | Uses multiple worker processes to load samples in parallel |
| **Memory efficiency** | Only keeps the current batch in memory, not the whole dataset |
| **Iteration** | Makes the dataset iterable — `for batch in loader:` |

### Key parameters

```python
DataLoader(
    dataset,
    batch_size=4,      # samples per batch
    shuffle=True,      # randomize order (train only, NOT val/test)
    num_workers=4,     # parallel processes for loading
    drop_last=False,   # drop incomplete last batch? (True for training)
)
```

**`shuffle=True`** — critical for training. Without shuffling, the model sees samples in the same order every epoch and can learn spurious patterns. Set `shuffle=False` for validation/test — you want consistent evaluation.

**`num_workers=4`** — spawns 4 background processes, each loading samples independently. This overlaps data loading with GPU computation: while the GPU trains on batch N, the workers prepare batch N+1.

---

## 12.4 The Batching Mechanism

The DataLoader calls `__getitem__` for each index in the batch, then stacks the results:

```
DataLoader wants batch [3, 17, 42, 8]:

  __getitem__(3)  →  image: (3, 768, 1024),  density: (1, 96, 128)
  __getitem__(17) →  image: (3, 768, 1024),  density: (1, 96, 128)
  __getitem__(42) →  image: (3, 768, 1024),  dimension: (1, 96, 128)
  __getitem__(8)  →  image: (3, 768, 1024),  density: (1, 96, 128)

  Stack along new dimension 0:
  images:    (4, 3, 768, 1024)
  densities: (4, 1, 96, 128)
```

This is why **every sample must have the same shape**. If image #3 is 768×1024 but image #17 is 640×480, the stack fails. This is why our dataset resizes all images to `target_size=(768, 1024)`.

---

## 12.5 The Dimension Convention

PyTorch always uses **(batch, channels, height, width)** — the "NCHW" format:

```
Single image:   (C, H, W)    = (3, 768, 1024)
                 ↑  ↑  ↑
              channels height width

Batched:        (N, C, H, W) = (4, 3, 768, 1024)
                 ↑  ↑  ↑  ↑
              batch channels height width
```

This is why our `__getitem__` does:

```python
# PIL image is (H, W, C) → we need (C, H, W)
image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)
#                                              channels first

# Density map is (H, W) → we need (1, H, W) — add channel dim
density_tensor = torch.from_numpy(density).unsqueeze(0)
#                                            add channel dim
```

The DataLoader then adds the batch dim: `(C, H, W)` → `(N, C, H, W)`.

---

## 12.6 The Full Pipeline Flow

```
1. Create dataset (once)
   dataset = CrowdCountingDataset(root=..., part="A", split="train", ...)
   → __init__ runs: finds 300 images, stores paths

2. Create loader (once)
   loader = DataLoader(dataset, batch_size=4, shuffle=True)

3. Training loop (repeated)
   for images, densities in loader:
   │
   ├─ DataLoader calls __getitem__ for 4 random indices
   │  └─ Each call: load image, make density map, return tensors
   │
   ├─ DataLoader stacks 4 samples into batch tensors
   │  └─ (4, 3, 768, 1024), (4, 1, 96, 128)
   │
   ├─ Forward pass
   │  └─ predictions = model(images)  → (4, 1, 96, 128)
   │
   ├─ Loss
   │  └─ loss = mse_loss(predictions, densities)
   │
   ├─ Backward
   │  └─ loss.backward()  → gradients computed
   │
   └─ Update
      └─ optimizer.step()  → weights updated
```

---

## 12.7 Train vs Val vs Test Loaders

```python
# Training — shuffle, maybe drop_last
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)

# Validation — no shuffle, keep all samples
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

# Test — no shuffle, keep all samples
test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)
```

**Why no shuffle for val/test?** Evaluation must be deterministic and reproducible. Shuffling doesn't affect metrics (you're averaging over all samples anyway), but deterministic order makes debugging easier.

**Why `drop_last=True` for training?** If you have 298 samples and batch_size=4, the last batch has only 2 samples. Smaller batches can cause noisy gradients and BatchNorm issues. Dropping it keeps all training batches the same size.

---

## 12.8 Performance Considerations

### The bottleneck

For our dataset, the expensive part of `__getitem__` is **density map generation**:

```
Fixed sigma:     ~5ms per image     (fast — just a Gaussian filter)
Adaptive sigma:  ~500ms per image  (slow — KDTree + per-head loop)
```

With `num_workers=4` and `batch_size=4`:
- Fixed: 4 images × 5ms / 4 workers ≈ 5ms per batch (fast enough)
- Adaptive: 4 images × 500ms / 4 workers ≈ 500ms per batch (bottleneck!)

This is why issue #12 (caching) matters — precompute density maps once, load from disk instead of regenerating every epoch.

### `num_workers` and multiprocessing

```
num_workers=0   → load in main process (blocks training)
num_workers=4   → 4 background processes load in parallel
num_workers=8   → more parallelism, but more memory and CPU overhead
```

Each worker is a **separate process** (not thread) with its own copy of the dataset. They load samples in advance and put them in a queue. The main process (which runs the GPU) pulls from the queue — no waiting.

---

## 12.9 Summary

| Concept | Key point |
|---|---|
| **Dataset** | Defines `__len__` and `__getitem__` — how to get one sample |
| **DataLoader** | Batches, shuffles, and parallelizes loading |
| **`__init__`** | Setup — find files, store config (called once) |
| **`__getitem__`** | Load one sample — image + density map tensors (called per sample) |
| **NCHW** | PyTorch dimension order: (batch, channels, height, width) |
| **shuffle=True** | For training only — randomizes order each epoch |
| **num_workers** | Parallel processes for faster loading |
| **Uniform shapes** | All samples must have the same shape to batch |