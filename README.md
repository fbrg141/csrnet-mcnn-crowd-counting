# Comparative Analysis of CSRNet and MCNN Models for Crowd Counting

## Team Members
- Uroš Dimitrijević
- Miloš Kutlešić

## Project Description
This project addresses the problem of **crowd counting in images**, focusing on a comparative analysis of two neural network architectures:
- **MCNN (Multi-Column Convolutional Neural Network)**
- **CSRNet (Congested Scene Recognition Network)**

The goal is to investigate how these two architectures perform on the task of estimating the number of people in crowd images, using an appropriate crowd counting dataset.

## Motivation
Unlike classical object detection, where each person is localized with a bounding box, the crowd counting approach is better suited for scenes with high people density, partial occlusion, and difficult detection of individual objects.
For this reason, models such as MCNN and CSRNet are more suitable than standard YOLO approaches for this problem.

## Dataset
Dataset used:
- **ShanghaiTech** (Part A & Part B): https://www.kaggle.com/datasets/tthien/shanghaitech

| Part | Images | Description |
|------|--------|-------------|
| Part A | 482 (300 train, 182 test) | Dense crowds, up to ~3000 people per image |
| Part B | 716 (400 train, 316 test) | Sparser scenes, up to ~500 people per image |

Annotations are (x, y) head coordinates stored in `.mat` files.

## Project Goals
The main objectives of the project are:
1. analyze the crowd counting dataset,
2. implement or adapt the **MCNN** and **CSRNet** models,
3. train the models under the same conditions,
4. compare their performance using standard metrics,
5. draw a conclusion about which model gives better results for the problem at hand.

## Research Question
Which model, **MCNN** or **CSRNet**, gives better results on the task of crowd counting in images, in terms of estimation accuracy and stability on the selected dataset?

## Models
### MCNN
MCNN uses multiple parallel convolutional branches with different receptive fields to handle scenes with varying crowd densities.

Paper: [Single-Image Crowd Counting via Multi-Column Convolutional Neural Network](https://openaccess.thecvf.com/content_cvpr_2016/html/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.html) (Zhang et al., CVPR 2016)

### CSRNet
CSRNet uses a deeper architecture with dilated convolutions and is known for strong results on crowd counting tasks, particularly in scenes with high crowd density.

Paper: [CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes](https://arxiv.org/abs/1802.10062) (Li et al., CVPR 2018)

## Evaluation Metrics
The following metrics will be used to compare the models:
- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**

Additionally, the following may be considered:
- training time,
- inference time,
- number of model parameters.

## Work Plan
1. Download and analyze the dataset
2. Prepare the data loading and annotation pipeline
3. Implement/adapt the MCNN model
4. Implement/adapt the CSRNet model
5. Train and evaluate the models
6. Comparative analysis of results
7. Write the report and prepare the presentation

## Project Structure
```text
data/             - dataset and data preparation
notebooks/        - exploratory analysis and visualizations
src/              - main project source code
reports/          - notes, images, results, and report drafts
pyproject.toml    - project metadata and direct dependencies
uv.lock           - exact, reproducible dependency versions
.python-version   - Python version used by uv
```

## Running the Project
This project uses [uv](https://docs.astral.sh/uv/) to manage Python, the
virtual environment, and locked dependencies. The supported platforms are
Linux x86_64 and Apple Silicon macOS, using Python 3.11.

### 1. Clone the repository
```bash
git clone https://github.com/fbrg141/csrnet-mcnn-crowd-counting.git
cd csrnet-mcnn-crowd-counting
```

### 2. Install uv

If `uv` is not already installed, follow the
[official installation guide](https://docs.astral.sh/uv/getting-started/installation/).
For Linux and macOS, the installer can be run with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify the installation:

```bash
uv --version
```

### 3. Set up the project

Create `.venv`, install Python 3.11 if necessary, and install the exact
versions recorded in `uv.lock`:

```bash
uv sync --locked
```

The default sync includes the runtime dependencies and the `dev`, `notebooks`,
and `dataset` dependency groups. The environment does not need to be activated
when commands are run with `uv run`.

If activation is preferred, use:

```bash
source .venv/bin/activate
```

After activation, `python` points to `.venv/bin/python`. To leave the
environment, run `deactivate`.

### 4. Download the dataset

The `dataset` dependency group provides `kagglehub`, which is used by the
download script:

```bash
uv run --locked python scripts/download_data.py
```

The dataset is copied to `data/raw/ShanghaiTech/`. See `data/README.md` for
the expected directory structure.

### 5. Run project commands

Use `uv run --locked` to run commands in the project environment without
allowing an implicit lockfile update:

```bash
# Run the test suite
uv run --locked python -m pytest tests/

# Start Jupyter Notebook
uv run --locked jupyter notebook

# Run a Python script
uv run --locked python path/to/script.py
```

Alternatively, activate `.venv` and run the same commands directly with
`python`, `pytest`, or `jupyter`.

### Updating dependencies

Use `uv add` and `uv remove` instead of editing `uv.lock` manually:

```bash
# Runtime dependency
uv add <package>

# Test and development dependency
uv add --dev <package>

# Notebook or dataset tooling
uv add --group notebooks <package>
uv add --group dataset <package>

# Remove a dependency from a group
uv remove --group notebooks <package>
```

These commands update both `pyproject.toml` and `uv.lock`. Commit both files
together. After pulling dependency changes, run `uv sync --locked` again.

To install only the runtime dependencies, without the default tool groups:

```bash
uv sync --locked --no-default-groups
```

The `.venv` directory is local and ignored by Git.

### Compute device support

Final training targets Linux x86_64 with an NVIDIA RTX 4070. The lockfile pins
PyTorch 2.13.0 and torchvision 0.28.0; their CUDA 13 dependencies are installed
only on Linux. Verify the Linux training environment with:

```bash
uv run --locked python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

On Apple Silicon with macOS 14 or newer, the same lockfile installs the macOS
PyTorch wheel without NVIDIA packages. The Mac can be used for development,
tests, notebooks, and small MPS/CPU smoke runs; final training is performed on
the Linux GPU. Intel Macs are not supported by the pinned PyTorch version.

## Training and Evaluation

Training and evaluation are model-agnostic entrypoints driven by a per-model
config (`src/config.py` → `MODEL_CONFIGS`) that sets the output stride, input
normalization, and learning rate. This keeps MCNN and CSRNet trained under
comparable, paper-faithful conditions:

| Model | Output stride | Input norm | Learning rate |
|-------|--------------|------------|---------------|
| MCNN  | 4 (two 2×2 pools) | raw [0,1] (from scratch) | 1e-6 |
| CSRNet | 8 (VGG + dilated) | ImageNet (pretrained VGG frontend) | 1e-5 |

### Train a model

```bash
# MCNN on ShanghaiTech Part A (defaults: 50 epochs, batch 4, lr from MODEL_CONFIGS)
uv run --locked python -m src.train --model mcnn --part A

# Smoke test: 1 epoch on a tiny fake dataset (no download needed)
uv run --locked python -m src.train --model mcnn --part A --smoke
```

Loss is pixel-wise MSE on the density map (the paper's Euclidean loss);
optimizer is SGD with momentum 0.95. Per epoch it logs train loss/MAE/RMSE
and validation MAE/RMSE, and checkpoints the best-validation-MAE `state_dict` to
`reports/checkpoints/<model>_part<part>_best.pth`.

### Density-map caching

Adaptive density maps take ~500 ms per image to generate. To avoid
regenerating identical targets every epoch (issue #15), density maps are cached
to disk under `data/processed/density_maps/` (git-ignored) and loaded on later
accesses. Caching is on by default for training and evaluation.

Precompute the whole cache once before a long run so every epoch is fast:

```bash
uv run --locked python scripts/precompute_density_maps.py            # all parts/splits/models
uv run --locked python scripts/precompute_density_maps.py --parts A --models mcnn
uv run --locked python scripts/precompute_density_maps.py --force       # rebuild from scratch
```

The cache key encodes every parameter that affects the density output (mode,
sigma/k+beta, target size, output stride, `CACHE_VERSION` in `src/config.py`),
so a stale or mismatched cache is never loaded. Bump `CACHE_VERSION` after any
change to the density-map generation logic. Flags: `--no-cache` regenerates every
epoch; `--cache-dir <path>` overrides the cache root.

### Evaluate a checkpoint

```bash
uv run --locked python -m src.evaluate --model mcnn --part A \
    --ckpt reports/checkpoints/mcnn_partA_best.pth
```

Loads the checkpoint, runs the test split, and writes
`reports/<model>_part<part>_metrics.json` (`mae`, `rmse`, `n_params`, ...) for
the later comparison table. The loader sanity-checks that the checkpoint's
model name matches the requested model.

### Counting convention

Both training and evaluation recover the head count as the integral of the
density map (`density.sum()`), for predictions and ground truth alike, so train
and eval measure the same quantity. MAE and RMSE are computed on these
per-image counts.

## Current Status
- [x] topic defined
- [x] dataset selected (ShanghaiTech)
- [x] dataset download script (`scripts/download_data.py`)
- [x] annotation analysis and density-map generation (`src/datasets/density_map.py`, `src/datasets/dataset.py`)
- [x] exploratory notebooks (`notebooks/01_dataset_inspection.ipynb`, `notebooks/02_density_maps.ipynb`)
- [x] dataset loader + normalization/downsample tests
- [x] adaptive density-map single-head edge case (`src/datasets/density_map.py`, `tests/test_density_map.py`)
- [x] MCNN implementation (`src/models/mcnn.py`, `tests/test_mcnn.py`) — verified against Zhang et al. CVPR 2016
- [ ] CSRNet implementation (currently a placeholder in `src/models/csrnet.py`)
- [x] training loop (`src/train.py`) — MSE loss, SGD, per-model config (stride/normalize/lr)
- [x] evaluation entrypoint (`src/evaluate.py`) — checkpoint load + MAE/RMSE
- [x] density-map caching (issue #15) — `src/datasets/dataset.py` + `scripts/precompute_density_maps.py`, cache under `data/processed/density_maps/`
- [ ] data augmentation (issue #14)
- [ ] real training run + results (needs dataset download)
- [ ] final report

## Expected Outcome
By the end of the project, the expected deliverables are:
- a functional implementation of both models,
- an experimental comparison of their performance,
- a clear conclusion about the advantages and disadvantages of the MCNN and CSRNet approaches on the selected dataset.

## Note
This repository is a student project for a machine learning course and serves as an experimental comparative analysis of crowd counting models.
