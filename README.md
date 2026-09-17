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
- **ShanghaiTech** (final experiments: Part A only; Part B available but not evaluated): https://www.kaggle.com/datasets/tthien/shanghaitech

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

## Project status and roadmap

[PLAN.md](PLAN.md) is the maintained status/roadmap: completed Part A experiments
and deliverables, required handover checks, and optional future experiments.
It does not claim Part B training, additional CSRNet seeds, or submission.

## Project Structure
```text
data/             - dataset and data preparation
notebooks/        - exploratory analysis and visualizations
src/              - main project source code
reports/          - final report, curated evidence, shared figures and local runs
presentation/     - 16-slide PDF, TeX source and speaker notes
configs/          - reproducible MCNN search grid
docs/             - experiment and Colab reproduction guide
notes/            - hand-authored study notes and historical pilots
PLAN.md           - current status, handover checks and optional future work
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
a shared pipeline with explicit model-specific differences (not an exact paper reproduction):

| Model | Output stride | Input norm | Learning rate |
|-------|--------------|------------|---------------|
| MCNN  | 4 (two 2×2 pools) | raw [0,1] (from scratch) | 1e-6 |
| CSRNet | 8 (VGG + dilated) | ImageNet (pretrained VGG frontend) | 1e-5 |

### Train a model

```bash
# MCNN on ShanghaiTech Part A (defaults: 50 epochs, batch 4, seed 42)
uv run --locked python -m src.train --model mcnn --part A --seed 42

# Smoke test: 1 epoch on a tiny fake dataset (no download needed)
uv run --locked python -m src.train --model mcnn --part A --smoke
```

Loss is pixel-wise MSE on the density map (the paper's Euclidean loss);
optimizer is SGD with momentum 0.95. Per epoch it logs train loss/MAE/RMSE
and validation MAE/RMSE, and checkpoints the best-validation-MAE `state_dict` to
`reports/checkpoints/<model>_part<part>_seed<seed>_best.pth`. The checkpoint also
records the seed and complete configuration. Use a separate output directory
for each model/part/seed/config; `--resume` verifies or resumes an existing run.
Smoke runs add a `_smoke` suffix, preventing a smoke checkpoint from replacing
a real experiment with the same model, part, and seed.

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
    --ckpt reports/checkpoints/mcnn_partA_seed42_best.pth
```

Loads the checkpoint, runs the test split, and writes
`reports/<model>_part<part>_seed<seed>_metrics.json` (`mae`, `rmse`, `n_params`,
`seed`, ...) for the later comparison table. The loader sanity-checks that the
checkpoint's model name matches the requested model and that seed metadata is
present. It also checks the ShanghaiTech part stored in the checkpoint, avoiding
an accidental Part A checkpoint evaluation labeled as Part B (or vice versa).

### Experimental protocol

The completed scope is **ShanghaiTech Part A only**:

- MCNN: 12 validation-ranked grid trials at seed 42 and four confirmation runs
  for baseline/winner at seeds 123 and 2026; all completed 50 epochs;
- CSRNet: one completed 50-epoch run, seed 42; no CSRNet search or multi-seed claim;
- 270 training / 30 validation images, the final 10% of lexicographically sorted
  official training files; 182 official test images;
- fixed sigma 15 density maps, corrected cache v2, input 768×1024, batch 4,
  mean pixel MSE and SGD; no augmentation, scheduler or early stopping;
- checkpoint/MCNN configuration selection by validation MAE only;
- primary comparison uses seed 42 for all three configurations; MCNN three-seed
  mean ± sample SD is reported separately (denominator n−1).

The validation split is a known limitation: because it is not randomly sampled,
it may not represent the full training distribution and may select a suboptimal
checkpoint. It is nevertheless fixed across all models and seeds, so the
controlled comparison remains consistent. Final metrics are computed on the
official test split, excluded from hyperparameter selection. Historical pilot
test results have already been inspected; this is not a globally unseen test set.

### Reproducible hyperparameter search and updated Colab notebook

See [the search protocol and commands](docs/hyperparameter-search.md) and
[`03_hyperparameters.ipynb`](notebooks/03_hyperparameters.ipynb).
The default JSON grid has 12 MCNN trials (lr, momentum, weight decay), ranks
only by best validation MAE, and supports resumable Drive checkpoints, optional
baseline/winner confirmation seeds, and explicitly gated test evaluation.
`--momentum`, `--weight-decay`, and `--resume` are available in `src.train`.
Each run needs its own output directory; changed configs never overwrite it.
Histories, summaries, hashes and optimizer/RNG state are persisted automatically.
Old model/part/seed checkpoints remain compatible with evaluation.

Before the changes are pushed, build/upload the allowlisted source zip rather
than expecting a Git clone to contain new code:

```bash
uv run --locked python scripts/build_colab_bundle.py --output /absolute/artifacts/path
```

The notebook has explicit long-run flags (off by default), absolute `/content`
paths, Drive dataset restore, locked setup, GPU/cache-v2 checks, validation-only
search and final CSV/JSON/plot exports. The real MCNN search and confirmations
are complete; curated original reports and final analysis are included below.
Synthetic smoke runs are never accuracy results.

### Historical Google Colab pilot (prefer the updated notebook above)

The first full run is MCNN on Part A with seed 42. Dataset files and density-map
cache can remain in Colab's temporary `/content` storage; persist only the
checkpoint, metrics, and training log in Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")

OUTPUT_ROOT = "/content/drive/MyDrive/csrnet-mcnn-results"
```

```python
!git pull --ff-only
!mkdir -p "{OUTPUT_ROOT}/checkpoints" "{OUTPUT_ROOT}/metrics" "{OUTPUT_ROOT}/logs"

!uv run --locked python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

!uv run --locked python scripts/precompute_density_maps.py

!uv run --locked python -m src.train --model mcnn --part A --epochs 50 --seed 42 --out-dir "{OUTPUT_ROOT}/checkpoints" 2>&1 | tee "{OUTPUT_ROOT}/logs/mcnn_partA_seed42.log"

!uv run --locked python -m src.evaluate --model mcnn --part A --ckpt "{OUTPUT_ROOT}/checkpoints/mcnn_partA_seed42_best.pth" --out "{OUTPUT_ROOT}/metrics/mcnn_partA_seed42_metrics.json"
```

### Counting convention

Both training and evaluation recover the head count as the integral of the
density map (`density.sum()`), for predictions and ground truth alike, so train
and eval measure the same quantity. MAE and RMSE are computed on these
per-image counts.

## Final collected results

Part A, official test set (182 images), **seed 42**:

| Configuration | Test MAE | Test RMSE | Best epoch |
|---|---:|---:|---:|
| MCNN baseline (lr1e-6, momentum .95, wd0) | 2509.3384 | 2531.8389 | 50 |
| MCNN tuned (lr1e-5, momentum .95, wd1e-4) | 316.3083 | 431.9880 | 50 |
| CSRNet (lr1e-5, momentum .95, wd0) | 233.4080 | 376.9634 | 31 |

CSRNet has 26.21% lower MAE and 12.74% lower RMSE than tuned MCNN for this
seed, not a statistical-significance or universal-architecture claim.
Its best validation MAE is 360.9125; recorded training time is 1583.2287 seconds
(local RTX 4070, CUDA FP32), not inference time or a cross-hardware benchmark.

**MCNN only**, seeds 42/123/2026, mean ± sample SD:

| Configuration | Test MAE | Test RMSE |
|---|---:|---:|
| Baseline | 1458.7485 ± 927.4737 | 1542.6522 ± 867.3692 |
| Tuned | 317.9492 ± 69.4665 | 435.9658 ± 100.2936 |

All 12 grid trials select epoch 50: the baseline is undertrained and convergence
is not established. Winner versus identical lr/momentum without weight decay
improves validation MAE by only 0.0135; do not claim meaningful regularization
benefit. Different pretraining, strides, normalization and tuning budgets mean
this comparison does not isolate architecture alone. CSRNet has no multi-seed SD.

- [Full Serbian report](reports/report.md)
- [Presentation PDF](presentation/presentation.pdf), [source](presentation/presentation.tex),
  [speaker notes](presentation/speaker-notes.md)
- [Original lightweight evidence and integrity limits](reports/evidence/README.md)
- [Derived machine-readable summary](reports/results_summary.json)

Regenerate every results chart and summary **without training, dataset or GPU**:

```bash
uv run --locked python scripts/plot_training_curves.py
# Install Tectonic separately; its first build downloads TeX resources:
(cd presentation && tectonic --keep-logs presentation.tex)
# Alternatively use a complete XeLaTeX installation (two passes).
```

The script verifies original hashes and derives results from JSON histories and
test records, never hard-coded curve arrays. The MCNN archive contains no weights;
rerunning inference requires original Drive checkpoints or a new training run.
Historical notes 19/20 are retained and labeled as superseded pilot evidence.

## Repository and artifact policy

- [PLAN.md](PLAN.md) owns the completed/required/optional checklist; the report
  owns scientific conclusions, and `docs/hyperparameter-search.md` owns run commands.
- `reports/evidence/` contains curated original lightweight records, while
  `reports/figures/` is the single figure source used by both report and slides.
- Preserve local datasets, checkpoints, full run directories and historical logs.
  They are ignored, not deleted; do not force-add them to Git. Curate reviewed
  small records into `reports/evidence/` with provenance instead.
- Build notebook/source deliveries outside the repository. The training-only zip
  intentionally omits final-report evidence and its plot/test pair; use the full
  repository to regenerate the final report figures.
- Exploratory notebooks and hand-authored study notes remain as learning history.
  Their sketches are not exact reproductions of papers or the current pipeline.
  The trained local MCNN has two convolutions per column and a 120-channel fusion;
  model names alone do not imply architecture/protocol equivalence with literature.

## Note
This repository is a student project for a machine learning course and serves as an experimental comparative analysis of crowd counting models.
