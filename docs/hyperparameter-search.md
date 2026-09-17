# MCNN hyperparameter search

## Completed evidence (Part A only)

All 12 seed-42 trials and four confirmation runs completed 50 epochs. The
validation winner is `(lr=1e-5, momentum=.95, weight_decay=1e-4)`, validation MAE
467.0918; the no-weight-decay counterpart gives 467.1052 (delta 0.0135).
All grid best epochs are 50; this is fixed-epoch-budget selection, not convergence.
Six MCNN test records cover baseline/winner × seeds 42/123/2026. The separate
local CSRNet seed-42 run is also complete (test MAE/RMSE 233.4080/376.9634,
best epoch 31), unlike the incomplete historical notebook output described below.
No Part B or additional CSRNet seeds are claimed.

See [the final report](../reports/report.md) for seed-matched comparison and
separate MCNN sample-SD analysis. Original lightweight evidence is stored in
`reports/evidence/`; regenerate charts with
`uv run --locked python scripts/plot_training_curves.py`. Archived MCNN weights
are absent, so report reproducibility is distinct from training resume.


## Protocol and interpretation

`configs/mcnn_search.json` defines MCNN Part A, seed 42, 50 epochs, batch 4:
learning rates `[1e-6, 3e-6, 1e-5]`, momenta `[0.90, 0.95]`, weight decay
`[0, 1e-4]`. The Cartesian product is **12 trials / 600 training epochs**.
The default baseline `(1e-6, .95, 0)` is included. Part B is configurable.
CSRNet retains its fixed baseline `(1e-5, .95, 0)`; it is not tuned here.

All other preprocessing/training settings stay fixed: corrected density maps
(`CACHE_VERSION=2`), model-specific stride/normalization, mean pixel MSE,
SGD, no augmentation/scheduler/early stopping, train `drop_last=True`, fixed
lexicographic-tail validation split. Report the validation split limitation.
The training loss denominator counts processed samples, not dropped samples.

Rank by **best validation MAE only** within the equal epoch budget; ties use
trial ID ascending. Validation RMSE is descriptive, never a tie-breaker.
The test split is not loaded by search or confirmation. All grid trials must
complete with finite metrics before selection is frozen in `selection.json`.
An invalid/diverged/interrupted trial stops search, leaving completed work;
fix the cause, then resume. If you change the search design/budget, use a new
output directory. Do not silently drop unfavorable trials.

The original notebook's 50-epoch baseline was still improving (validation
MAE 2629.85); its separate lr=1e-5 pilot obtained validation MAE 464.74 and
reported test MAE/RMSE 315.2272/429.1577. It also reused/overwrote the same
checkpoint filename. Independent review replaced the proposed 3e-7 rate with
1e-5 before experiments: the pilot's validation evidence favors testing the
higher range, while preserving the baseline and 12-trial budget. This does not
guarantee improvement or use the historical test score for ranking. All 1e-5
combinations now participate in validation-only selection. Start a new experiment
directory rather than resume the earlier grid. Fifty epochs compares
fixed compute budgets, not converged optima. The pilot already exposed test
results; claim exclusion from this search, not globally unseen test data.
Original CSRNet output stopped at epoch 48, so it is not completion evidence.

## Commands

Use the same config, source snapshot, root path, device option and output
directory after restarts. Keep the dataset/cache immutable and dedicated to
this dataset; the inherited density cache filenames do not identify arbitrary
alternative dataset roots. Only one runner may write an output directory.

```bash
uv run --locked python -m src.search --config configs/mcnn_search.json --stage dry-run
# On Colab use /content/drive/MyDrive/... for --out-dir, not local reports/.
uv run --locked python -m src.search --config configs/mcnn_search.json --out-dir reports/search/mcnn-A --stage baseline
uv run --locked python -m src.search --config configs/mcnn_search.json --out-dir reports/search/mcnn-A --stage search
# Repeat the same search command after a disconnect: verifies/skips finished runs,
# resumes unfinished runs from last.pth (at epoch boundaries).
uv run --locked python -m src.search --config configs/mcnn_search.json --out-dir reports/search/mcnn-A --stage confirm
# Only AFTER hyperparameters and seed list are fixed:
uv run --locked python -m src.search --config configs/mcnn_search.json --out-dir reports/search/mcnn-A --stage evaluate --with-confirmation
uv run --locked python -m src.search --config configs/mcnn_search.json --out-dir reports/search/mcnn-A --stage report
```

Flags `--root`, `--cache-dir`, `--device {auto,cpu,cuda}` apply to all stages.
Default stage is dry-run. Training never evaluates test implicitly.
Optional confirmation fixes the baseline and validation winner, then trains
seeds 123 and 2026 (at most four additional runs, 200 epochs). Seed 42 is reused
from search, not retrained. Identical baseline/winner shares artifacts.
To report only seed 42, omit confirmation and `--with-confirmation`; standard
deviation will be null, not zero. The evaluation plan is immutable: choose
single-seed versus three-seed **before** the first test evaluation. To avoid
selective reporting, never choose the seed plan based on its test results.
All requested confirmation runs must finish before three-seed evaluation.

Standalone models use the same persistence features:

```bash
uv run --locked python -m src.train --model csrnet --part A --seed 42 \
  --lr 1e-5 --momentum .95 --weight-decay 0 --epochs 50 --batch-size 4 \
  --out-dir reports/baselines/csrnet-A-seed42 --resume
uv run --locked python -m src.evaluate --model csrnet --part A \
  --ckpt reports/baselines/csrnet-A-seed42/csrnet_partA_seed42_best.pth \
  --out reports/baselines/csrnet-A-seed42/test_metrics.json
```

## Persistence, integrity and resume

Every trial lives in its own content-derived `trial-<hash>` directory.
`config.json` binds optimizer, batch/epoch budget, seed, dataset content hashes,
source/lockfile fingerprint, preprocessing, split, device and PyTorch version.
Changing any of these refuses reuse instead of silently mixing experiments.
Selection also checks that all trials share the same dataset and fixed training
configuration; final evaluation validates confirmation configurations and seeds
against the frozen search before accessing test images.
Trial IDs encode the requested training config; the enclosing protocol also
binds source and root. Old checkpoints still load in `src.evaluate` but cannot
be used for optimizer resume; start a new output directory for old pilots.

- `history.json`: per-epoch train loss/count MAE/RMSE and validation MAE/RMSE.
- `<model>_part<part>_seed<seed>_best.pth`: best-validation checkpoint and config.
- `last.pth`: completed epoch, model, optimizer, RNG states, full history,
  elapsed time and a copy of best weights for interrupted-write recovery.
- `summary.json`: best epoch/validation metrics, budget, runtime and checkpoint.
- `complete.json`: configuration/artifact SHA256 hashes. Skipping a run requires
  verified complete files, not just a checkpoint filename. Corrupt manifests
  fail closed; inspect/recover from a trusted backup rather than bypass checks.

Writes use temporary files, flush/fsync and replace; completion is written last.
A disconnect can lose the current epoch, not require restarting the whole run.
Google Drive FUSE/cloud synchronization is not a transactional filesystem;
verify artifacts in Drive before discarding a runtime and keep backups.
Don't modify code halfway through a run. Cross-GPU/OS/PyTorch bitwise identity
is not guaranteed; local tests verify same-device epoch-resume equivalence.
Only load checkpoints you trust: PyTorch optimizer/RNG checkpoints use pickle.

## Results (generated, never fabricated)

Search produces `selection.json`, `search_results.csv` and
`validation_curves.png`. Explicit evaluation first freezes
`evaluation_plan.json` (labels/seeds/checkpoint hashes), then writes per-run
`test_metrics.json`, `final_metrics.json`, `test_results.csv`, `aggregate.json`.
Aggregate MAE/RMSE are means and **sample standard deviations** (n−1) over the
fixed seeds, not per-image standard errors. The search winner is not changed
by these test results. Test evaluation can be rerun to recover interrupted
report generation, but must use the same frozen plan and unchanged dataset.
Notebook export excludes datasets/checkpoints and collects report JSON/CSV/PNG.
Copy only reviewed real results into repository reports after the actual Colab
runs; the implementation/smoke tests do not complete the experimental report.

## Colab source bundle (works before commit/push)

Canonical notebook: `notebooks/03_hyperparameters.ipynb`.
Build the exact local source bundle outside the repository:

```bash
uv run --locked python scripts/build_colab_bundle.py --output /absolute/artifacts/path
```

This emits the notebook, `crowd-hyperparameters-source.zip`, and SHA256 file.
An explicit allowlist includes Python source/scripts/tests, configuration,
README/guide, Python version and locked dependency files. The final-report plot
script and its evidence-dependent tests are excluded together: they require the
full repository, while the training-only bundle runs every included test offline.
Report/evidence links in the README refer to the full repository, not bundle members.
No `.git`, `.venv`,
secrets, datasets, checkpoints, results or PLAN.md. The bundle manifest records
base revision **and dirty working-tree status**, not an invented remote commit.
The exported notebook pins the paired zip SHA256; a stale/different Drive bundle
is refused. Change `BUNDLE_ON_DRIVE` to a new path to upload a new paired bundle.
Open the notebook in Colab and upload the trusted zip when prompted. It is
saved on Drive for reconnects and extracted into an absolute content-addressed
`/content` directory. Existing differing source files are never overwritten.
The notebook mounts Drive, installs with `uv sync --locked`, checks GPU/cache
v2, restores/downloads data, and offers gated baseline/search/confirmation/test
cells. All expensive steps require explicit flags; read operation order first.

## Local smoke verification

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run --locked python -m pytest tests -q
uv run --locked python -m src.search --smoke --device cpu --out-dir /tmp/crowd-smoke --stage search
uv run --locked python -m src.search --smoke --device cpu --out-dir /tmp/crowd-smoke --stage confirm
uv run --locked python -m src.search --smoke --device cpu --out-dir /tmp/crowd-smoke --stage evaluate --with-confirmation
```

Smoke uses real MCNN forward/backward/SGD with synthetic images/annotations,
one epoch and 64×64 images; it disables density caching and labels results
synthetic. It checks execution, **not model accuracy**. CSRNet smoke avoids
pretrained downloads; real CSRNet initialization remains pretrained. Local
checks cannot verify Colab account access, actual GPU quotas or Drive sync.
