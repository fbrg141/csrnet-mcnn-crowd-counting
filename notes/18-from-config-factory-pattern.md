# 18 — The from_config Factory Pattern

This note documents how `config.py` is wired to `CrowdCountingDataset` without coupling — the factory method pattern we implemented for issue #13.

---

## 18.1 The Problem

Two ways to configure a class, both bad:

### Option A: Hardcode defaults in the constructor

```python
class CrowdCountingDataset(Dataset):
    def __init__(self, root, sigma=15.0, k=4, beta=0.3, target_size=None, ...):
        self.sigma = sigma
        ...
```

**Problem:** `config.py` defines `FIXED_SIGMA = 15.0` but the constructor ignores it. The config is dead code. Change the config, nothing changes. The two values can silently drift apart.

### Option B: Import config into the constructor

```python
from src.config import FIXED_SIGMA, ADAPTIVE_K, ...

class CrowdCountingDataset(Dataset):
    def __init__(self, root, sigma=FIXED_SIGMA, k=ADAPTIVE_K, ...):
        ...
```

**Problem:** the class is now hard-coupled to `config.py`. Tests can't create a dataset with custom values without overriding config. And if config imports something that imports the dataset, you get a circular import.

---

## 18.2 The Solution: Factory Method

Keep the constructor decoupled (with its own defaults), but add a **factory method** that reads from config:

```python
class CrowdCountingDataset(Dataset):
    def __init__(self, root, sigma=15.0, k=4, beta=0.3, target_size=None,
                 downsample_factor=1, normalize=False, ...):
        # Constructor stays decoupled — own defaults, no config import
        ...

    @classmethod
    def from_config(cls, part="A", split="train",
                    downsample_factor=1, normalize=False, root=None):
        """Create a dataset using defaults from src.config."""
        return cls(
            root=root or SHANGHAITECH_DIR,
            part=part,
            split=split,
            density_mode=DEFAULT_DENSITY_MODE,
            sigma=FIXED_SIGMA,
            k=ADAPTIVE_K,
            beta=ADAPTIVE_BETA,
            target_size=DEFAULT_IMAGE_SIZE,
            val_split=VAL_SPLIT,
            downsample_factor=downsample_factor,
            normalize=normalize,
        )
```

### How it works

- **Constructor** — stays pure. Takes explicit arguments, has its own defaults, no config dependency. Tests use this directly with custom values.
- **`from_config()`** — a `@classmethod` that imports config values and passes them to the constructor. Production code uses this for convenience.

### Why this is better

| | Hardcoded (A) | Config-in-constructor (B) | Factory method |
|---|---|---|---|
| Config is used | ❌ | ✅ | ✅ |
| Class is decoupled | ✅ | ❌ | ✅ |
| Tests can override | ✅ | ❌ | ✅ |
| One import for all config | ❌ | ✅ | ✅ |
| No circular import risk | ✅ | ❌ | ✅ |

The factory method gets the best of all options.

---

## 18.3 `@classmethod` Explained

```python
@classmethod
def from_config(cls, ...) -> "CrowdCountingDataset":
    return cls(...)
```

**`@classmethod`** — the method receives the **class** (`cls`) as its first argument instead of an instance (`self`). This means:

- You call it on the class, not an instance: `CrowdCountingDataset.from_config(...)`
- `cls` is `CrowdCountingDataset` — so `cls(...)` calls the constructor
- It can return a new instance of the class

### Why not a regular function?

```python
# Could be a standalone function:
def make_dataset_from_config(part="A", ...):
    return CrowdCountingDataset(...)

# But @classmethod is better because:
# - It's discoverable: CrowdCountingDataset.from_config(...) is found via autocomplete
# - It's namespaced: lives on the class, not floating in the module
# - It works with inheritance: subclasses can override from_config
```

### The return type annotation

```python
def from_config(cls, ...) -> "CrowdCountingDataset":
```

The return type is a string (`"CrowdCountingDataset"`) because the class isn't fully defined yet when the annotation is parsed. This is a **forward reference** — Python evaluates it lazily. Without the quotes, it would be a `NameError` because `CrowdCountingDataset` doesn't exist yet inside its own body.

---

## 18.4 Which Parameters Come from Config, Which Stay Explicit

```python
def from_config(cls, part="A", split="train",
                downsample_factor=1, normalize=False, root=None):
```

### From config (the "always the same" values)

| Parameter | Config value | Why from config |
|---|---|---|
| `root` | `SHANGHAITECH_DIR` | Dataset path never changes |
| `density_mode` | `DEFAULT_DENSITY_MODE` | Project-wide choice |
| `sigma` | `FIXED_SIGMA` | Hyperparameter, set once |
| `k`, `beta` | `ADAPTIVE_K`, `ADAPTIVE_BETA` | From the MCNN paper, never change |
| `target_size` | `DEFAULT_IMAGE_SIZE` | Project-wide resize |
| `val_split` | `VAL_SPLIT` | Project-wide split ratio |

### Explicit (the "depends on context" values)

| Parameter | Why explicit |
|---|---|
| `part` | "A" or "B" — you choose per experiment |
| `split` | "train", "val", or "test" — different per loader |
| `downsample_factor` | Model-specific: 4 for MCNN, 8 for CSRNet |
| `normalize` | Model-specific: False for MCNN, True for CSRNet |

**The principle:** global hyperparameters come from config; per-model or per-experiment values stay explicit. If a value changes between MCNN and CSRNet, it shouldn't be in config.

---

## 18.5 The `root or SHANGHAITECH_DIR` Pattern

```python
return cls(
    root=root or SHANGHAITECH_DIR,
    ...
)
```

**`root or SHANGHAITECH_DIR`** — if `root` is `None` (the default), use `SHANGHAITECH_DIR` from config. If someone passes a custom `root`, use that instead.

This is the Python `or` short-circuit:
```python
None or SHANGHAITECH_DIR   → SHANGHAITECH_DIR  (None is falsy)
"/custom/path" or SHANGHAITECH_DIR  → "/custom/path"  (string is truthy)
```

Why allow a custom root? Tests. The pytest tests create a fake dataset in `tmp_path`, so they need to override the root:

```python
def test_something(tmp_path):
    _make_fake_shanghaitech(tmp_path)
    ds = CrowdCountingDataset(tmp_path, ...)  # custom root, not config
```

Without the `root or` pattern, `from_config` would always use `SHANGHAITECH_DIR` and tests couldn't use fake data.

---

## 18.6 Usage

### Production code (train.py)

```python
from src.datasets import CrowdCountingDataset

# CSRNet
train_ds = CrowdCountingDataset.from_config(
    part="A", split="train", downsample_factor=8, normalize=True
)
val_ds = CrowdCountingDataset.from_config(
    part="A", split="val", downsample_factor=8, normalize=True
)

# MCNN
train_ds = CrowdCountingDataset.from_config(
    part="A", split="train", downsample_factor=4, normalize=False
)
```

One import, all hyperparameters consistent. Change `FIXED_SIGMA` in config, the whole pipeline updates.

### Tests (with custom values)

```python
def test_custom_sigma(tmp_path):
    _make_fake_shanghaitech(tmp_path)
    # Use the constructor directly with custom values, bypassing config
    ds = CrowdCountingDataset(tmp_path, sigma=8.0, downsample_factor=4)
    ...
```

Tests use the constructor directly when they need specific values that differ from config. The factory method is for production convenience; the constructor is for full control.

---

## 18.7 When to Use This Pattern

Use the factory method pattern when:

- You have a **config module** with project-wide defaults
- The class needs to be **testable with custom values** (decoupled from config)
- You want **one import** to get all config values in production code
- There's a risk of **circular imports** if config is imported in the constructor

Don't use it when:

- The class has no config — just use constructor defaults
- The class is never used outside one place — inline the config import
- The class needs config-dependent behavior at import time (rare)

---

## 18.8 Summary

| Concept | Key point |
|---|---|
| **Factory method** | `@classmethod` that reads config and calls the constructor |
| **Constructor stays pure** | No config import — tests can use custom values |
| **`from_config()` for production** | One import, all config values, convenience |
| **`root or DEFAULT`** | Allow override while defaulting to config |
| **Explicit params** | Model-specific values (downsample_factor, normalize) stay explicit |
| **Config params** | Global hyperparameters (sigma, k, beta, target_size) from config |
| **`@classmethod`** | Receives the class (`cls`), not an instance (`self`) |
| **Forward reference** | `"CrowdCountingDataset"` as a string — class not defined yet |