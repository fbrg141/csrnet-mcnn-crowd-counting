#!/usr/bin/env python3
"""Precompute and cache density maps for the ShanghaiTech dataset.

Density-map generation (especially the adaptive, geometry-adaptive kernels) is
expensive — ~500ms per image. During training these maps are regenerated on every
`__getitem__`, wasting ~150s per epoch on identical targets (issue #15).

This script pays that cost once: for each (part, split, model) combination it
builds a caching dataset and touches every item, writing the density maps to
`data/processed/density_maps/`. Subsequent training/evaluation runs then load
the cached `.npy` files instead of regenerating.

Run:
    python scripts/precompute_density_maps.py                 # all parts/splits/models
    python scripts/precompute_density_maps.py --parts A --models mcnn
    python scripts/precompute_density_maps.py --force          # delete cache first
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

# Ensure the project root is importable whether this file is run as a direct
# path (``python scripts/precompute_density_maps.py``, which puts ``scripts/``
# on sys.path[0]) or as a module (``python -m scripts.precompute_density_maps``).
# The project is not an installed package (pyproject ``package = false``), so
# ``src`` is only reachable when the project root is on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODEL_CONFIGS, SHANGHAITECH_DIR, DENSITY_CACHE_DIR
from src.datasets.dataset import CrowdCountingDataset

PARTS = ("A", "B")
SPLITS = ("train", "test")
# "val" reuses train_data with the held-out tail; cache it too so training's
# validation pass is also fast. It is distinct from "train" only via val_split.
VAL_SPLIT = "val"


def warm_one(model: str, part: str, split: str, root: str | Path,
                cache_dir: str | Path) -> tuple[int, float]:
    """Build a caching dataset for one combo and touch every item.

    Returns (n_items, seconds)."""
    cfg = MODEL_CONFIGS[model]
    ds = CrowdCountingDataset.from_config(
        part=part, split=split, root=root,
        downsample_factor=cfg["downsample_factor"],
        normalize=cfg["normalize"],
        use_cache=True,
        cache_dir=cache_dir,
    )
    start = time.perf_counter()
    for i in range(len(ds)):
        ds[i]
    return len(ds), time.perf_counter() - start


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Precompute density-map caches.")
    parser.add_argument("--parts", nargs="+", default=list(PARTS),
                        choices=PARTS, help="dataset parts to warm")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS) + [VAL_SPLIT],
                        choices=list(SPLITS) + [VAL_SPLIT],
                        help="splits to warm (default train, test, val)")
    parser.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS),
                        choices=list(MODEL_CONFIGS),
                        help="models (each caches its own output stride)")
    parser.add_argument("--root", default=str(SHANGHAITECH_DIR))
    parser.add_argument("--cache-dir", default=str(DENSITY_CACHE_DIR))
    parser.add_argument("--force", action="store_true",
                        help="delete the existing cache before warming")
    args = parser.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    if args.force and cache_dir.exists():
        print(f"[force] removing existing cache at {cache_dir}")
        shutil.rmtree(cache_dir)

    print(f"[cache] {cache_dir}")
    total_items, total_s = 0, 0.0
    for model in args.models:
        for part in args.parts:
            for split in args.splits:
                n, s = warm_one(model, part, split, args.root, args.cache_dir)
                total_items += n
                total_s += s
                print(f"  {model:6s} part{part} {split:5s}: {n:4d} items in {s:6.2f}s")

    print(f"[done] cached {total_items} density maps in {total_s:.2f}s -> {cache_dir}")


if __name__ == "__main__":
    main()