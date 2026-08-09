# 16 — Project Structure

This note documents how the repository is organized and why — the separation between exploration, production code, utilities, and tests.

---

## 16.1 The Directory Layout

```
crowd-counting/
│
├── notebooks/          ← EXPLORATION (done once, manually)
│   ├── 01_dataset_inspection.ipynb
│   └── 02_density_maps.ipynb
│
├── src/                ← PRODUCTION CODE (used by training/eval)
│   ├── config.py              ← all hyperparameters in one place
│   ├── datasets/
│   │   ├── dataset.py         ← CrowdCountingDataset
│   │   └── density_map.py      ← fixed + adaptive density functions
│   ├── models/
│   │   ├── mcnn.py             ← MCNN model
│   │   └── csrnet.py           ← CSRNet model
│   ├── utils/
│   │   ├── metrics.py          ← MAE, RMSE
│   │   └── visualization.py    ← plotting helpers
│   ├── train.py               ← training loop
│   └── evaluate.py            ← evaluation pipeline
│
├── scripts/            ← ONE-OFF UTILITIES (run manually, not imported)
│   └── download_data.py
│
├── tests/              ← UNIT TESTS (run by pytest/CI)
│   ├── test_dataset_downsample.py
│   └── test_dataset_normalization.py
│
├── notes/              ← THEORY NOTES (reference, not code)
│   └── 01-18 *.md
│
├── data/               ← DATASET (gitignored, downloaded by script)
│   └── raw/ShanghaiTech/...
│
└── reports/            ← RESULTS AND REPORT DRAFTS
```

---

## 16.2 The Four Layers

### Notebooks — Exploration

**Purpose:** answer questions about the data and prototype ideas before committing to production code.

- `01_dataset_inspection.ipynb` — "what does the data look like?" (head counts, distributions, image sizes)
- `02_density_maps.ipynb` — "does density map generation work?" (fixed vs adaptive sigma, visual comparison)

**Characteristics:**
- Run once, manually, to look at output
- Not imported by anything
- Not part of the training pipeline
- Can be messy — they're lab notes, not production code

**Once an idea is validated in a notebook, the logic is extracted into `src/` as reusable functions.** The notebooks stay as a record of the exploration; the production code lives separately.

---

### `src/` — Production Code

**Purpose:** the real pipeline that training and evaluation use.

```
train.py
  → imports CrowdCountingDataset from src/datasets/dataset.py
  → imports MCNN or CSRNet from src/models/
  → loads images + generates density maps
  → trains the model
  → evaluate.py computes MAE/RMSE using src/utils/metrics.py
```

**Characteristics:**
- Imported by `train.py` and `evaluate.py`
- Clean, tested, documented
- No exploration code — just the reusable pipeline

The notebooks proved density maps work. `src/datasets/density_map.py` contains the same logic as **reusable functions** that the training pipeline calls automatically.

---

### `scripts/` — One-off Utilities

**Purpose:** standalone scripts run manually from the command line, not imported by training code.

- `download_data.py` — run once to download the dataset

**Characteristics:**
- Run manually: `python scripts/download_data.py`
- Not imported by `src/` or `tests/`
- Self-contained — doesn't depend on `src.config` (works before `src` is importable)
- Uses `Path(__file__).resolve().parent.parent` to find the project root itself

**Why separate from `src/`?** These are utilities, not pipeline components. The download script must work standalone — it's the first thing you run after cloning, before `src` is even importable.

---

### `tests/` — Unit Tests

**Purpose:** verify the production code works correctly, run automatically by pytest or CI.

- `test_dataset_downsample.py` — tests the `downsample_factor` parameter
- `test_dataset_normalization.py` — tests the `normalize` parameter

**Characteristics:**
- Run with `python -m pytest tests/`
- Use fake fixtures (no real dataset needed)
- Test through the real `CrowdCountingDataset` class (no logic duplication)
- Deterministic (fixed random seeds)

---

## 16.3 Why This Separation Matters

| Layer | If it breaks... | Who runs it |
|---|---|---|
| Notebooks | nothing else breaks — they're standalone | Developer, once |
| `src/` | training/eval break — this is the pipeline | `train.py`, `evaluate.py` |
| `scripts/` | download breaks — can't get data | Developer, manually |
| `tests/` | CI breaks — don't catch regressions | pytest / CI automatically |

The key principle: **exploration code (notebooks) is disposable, production code (`src/`) is not.** Don't put logic you need for training in a notebook — it won't be importable and it'll drift from the real code.

---

## 16.4 The Data Directory

```
data/
  raw/        ← .gitignore (downloaded, never committed)
  processed/  ← .gitignore (generated from raw)
  README.md   ← committed (documents source + download instructions)
```

**Why gitignored?**
- Size — ShanghaiTech is ~333 MB; git stores every version
- Git is for code, not data — binary files don't diff
- Reproducibility — the dataset should be downloaded by a script, not bundled
- Licensing — some datasets prohibit redistribution

**What's committed instead:**
- `data/README.md` — documents the source and download command
- `scripts/download_data.py` — reproduces the dataset with one command

---

## 16.5 The `.mat` Annotation Format

ShanghaiTech annotations are MATLAB `.mat` files. Each contains head coordinates for one image:

```
GT_IMG_1.mat
  └── image_info (MATLAB struct)
        └── location → (N, 2) array of (x, y) head coordinates
```

Loaded with `scipy.io.loadmat`:

```python
mat = sio.loadmat(gt_path)
points = mat["image_info"][0, 0][0, 0][0]  # peel through MATLAB struct nesting
```

The indexing `mat["image_info"][0, 0][0, 0][0]` peels through nested MATLAB struct representation:

```
mat["image_info"]     → numpy array shape (1, 1), dtype=object
  [0, 0]              → another nested array
    [0, 0]            → another level
      [0]             → the actual (N, 2) coordinate array
```

scipy represents MATLAB structs as nested numpy arrays of objects, so you peel through the nesting to reach the raw data. No MATLAB license needed — `scipy.io.loadmat` reads the binary format directly.

---

## 16.6 The kagglehub Download Pattern

`scripts/download_data.py` uses `kagglehub` to download the dataset:

```python
import kagglehub

DATASET = "tthien/shanghaitech"  # Kaggle dataset slug
path = kagglehub.dataset_download(DATASET)  # downloads to ~/.cache/kagglehub/
```

`kagglehub` downloads to its own cache directory, not to our `data/raw/`. The script then copies the relevant subdirectory:

```python
src = Path(path) / "ShanghaiTech"
dst = RAW_DATA_DIR / "ShanghaiTech"

if dst.exists():
    shutil.rmtree(dst)     # clean any previous download
shutil.copytree(src, dst)  # copy into data/raw/
```

**`shutil.rmtree`** — recursively delete (like `rm -rf`).
**`shutil.copytree`** — recursively copy (like `cp -R`).

After copying, the script counts files to verify the download was complete:

```python
part_a_train = len(list(dst.glob("part_A/train_data/images/*.jpg")))
```

The shebang line `#!/usr/bin/env python3` allows running the script directly (`./download_data.py`) on Unix — `env` searches `PATH` for the active Python, including virtual environments.