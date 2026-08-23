"""Evaluation entrypoint for crowd counting models.

Loads a trained checkpoint and reports MAE/RMSE on the test split, plus the
total parameter count (for the report's comparison table). Reuses the same
evaluate() and density_to_count() as training so train and eval measure the
same thing.

Run:
    python -m src.evaluate --model mcnn --part A \\
        --ckpt reports/checkpoints/mcnn_partA_best.pth
    python -m src.evaluate --model mcnn --part A --smoke   # no checkpoint needed
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.config import DENSITY_CACHE_DIR, MODEL_CONFIGS, REPORTS_DIR, SHANGHAITECH_DIR
from src.datasets.dataset import CrowdCountingDataset
from src.models import build_model
from src.train import evaluate, get_device


def load_checkpoint(path: str | Path, model_name: str) -> dict:
    """Load a checkpoint dict and sanity-check it matches the requested model.

    Verifies the stored `model` field equals the one requested, so you can't
    accidentally evaluate an MCNN checkpoint with a CSRNet model (the
    state_dict would load but with silently-wrong semantics).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("model") != model_name:
        raise ValueError(
            f"checkpoint model={ckpt.get('model')!r} != requested {model_name!r}"
        )
    return ckpt


def build_test_loader(model_name: str, part: str, root: str | Path,
                        use_cache: bool = True,
                        cache_dir: str | Path = DENSITY_CACHE_DIR) -> DataLoader:
    """Build the test-split DataLoader with the model's stride/normalization."""
    cfg = MODEL_CONFIGS[model_name]
    test_ds = CrowdCountingDataset.from_config(
        part=part, split="test", root=root,
        downsample_factor=cfg["downsample_factor"],
        normalize=cfg["normalize"],
        use_cache=use_cache,
        cache_dir=cache_dir,
    )
    # batch_size=1: counts are per-image; no padding/drop so every test image counts.
    return DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)


# --------------------------------------------------------------------------------------
# Fake dataset for smoke test (mirrors the train.py / test fixtures)
# --------------------------------------------------------------------------------------
def _make_fake_dataset(root: str | Path, part: str = "A", n: int = 6) -> str:
    root = Path(root)
    img_dir = root / f"part_{part}" / "test_data" / "images"
    gt_dir = root / f"part_{part}" / "test_data" / "ground-truth"
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for i in range(n):
        Image.fromarray(
            rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
        ).save(img_dir / f"IMG_{i + 1}.jpg")
        pts = rng.uniform(0, 256, size=(10, 2))
        sio.savemat(
            gt_dir / f"GT_IMG_{i + 1}.mat",
            {"image_info": np.array([[[[[pts]]]]], dtype=float)},
        )
    return str(root)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--model", default="mcnn", choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--part", default="A", choices=["A", "B"])
    parser.add_argument("--ckpt", default=None,
                        help="path to checkpoint (required unless --smoke)")
    parser.add_argument("--root", default=str(SHANGHAITECH_DIR))
    parser.add_argument("--no-cache", action="store_true",
                        help="disable density-map disk caching")
    parser.add_argument("--cache-dir", default=str(DENSITY_CACHE_DIR),
                        help="density-map cache root (default data/processed/density_maps)")
    parser.add_argument("--out", default=None,
                        help="json file to write metrics to (default: reports/<model>_part<part>_metrics.json)")
    parser.add_argument("--smoke", action="store_true",
                        help="evaluate a random-init model on a tiny fake test set")
    args = parser.parse_args(argv)

    device = get_device()

    model = build_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    if args.smoke:
        # No real checkpoint: build fake data, use a fresh random-init model,
        # just to exercise the full load/eval/report path. Numbers are meaningless.
        root = _make_fake_dataset(tempfile.mkdtemp(), part=args.part, n=6)
        args.root = root
        ckpt_info = {"epoch": 0, "val_mae": None, "val_rmse": None}
    else:
        if args.ckpt is None:
            parser.error("--ckpt is required unless --smoke is set")
        ckpt = load_checkpoint(args.ckpt, args.model)
        model.load_state_dict(ckpt["state_dict"])
        ckpt_info = {k: ckpt.get(k) for k in ("epoch", "val_mae", "val_rmse")}

    test_loader = build_test_loader(args.model, args.part, args.root,
                                    use_cache=not args.no_cache,
                                    cache_dir=args.cache_dir)
    print(f"[device] {device}")
    print(f"[model] {args.model} params={n_params} "
          f"(ckpt epoch={ckpt_info['epoch']} val_mae={ckpt_info['val_mae']})")
    print(f"[data] test={len(test_loader.dataset)} images")

    mae, rmse = evaluate(model, test_loader, device)
    print(f"[result] MAE={mae:.4f}  RMSE={rmse:.4f}")

    out_path = Path(args.out) if args.out else (
        REPORTS_DIR / f"{args.model}_part{args.part}_metrics.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": args.model,
        "part": args.part,
        "n_test": len(test_loader.dataset),
        "n_params": n_params,
        "mae": mae,
        "rmse": rmse,
        "ckpt_epoch": ckpt_info["epoch"],
        "ckpt_val_mae": ckpt_info["val_mae"],
        "ckpt_val_rmse": ckpt_info["val_rmse"],
    }
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()