"""Training entrypoint for crowd counting models.

Run:
    python -m src.train --model mcnn --part A --epochs 50
    python -m src.train --model mcnn --part A --smoke   # 1 epoch, tiny fake set

The loop is intentionally model-agnostic: the per-model knobs (output stride,
input normalization, learning rate, momentum) come from src.config.MODEL_CONFIGS
so MCNN and CSRNet are trained under comparable, paper-faithful conditions.
"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_EPOCHS,
    MODEL_CONFIGS,
    DENSITY_CACHE_DIR,
    REPORTS_DIR,
    SHANGHAITECH_DIR,
    VAL_SPLIT,
)
from src.datasets.dataset import CrowdCountingDataset
from src.models import build_model


# --------------------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------------------
def get_device() -> torch.device:
    """Pick the best available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def build_dataloaders(
    model_name: str,
    part: str,
    root: str | Path,
    batch_size: int,
    num_workers: int,
    use_cache: bool = True,
    cache_dir: str | Path = DENSITY_CACHE_DIR,
) -> tuple[DataLoader, DataLoader]:
    """Build train + validation DataLoaders for a given model.

    The model-specific downsample_factor and normalize flags are pulled from
    MODEL_CONFIGS so the GT density map matches the model's output stride and
    the input scaling matches what the model was designed for.

    use_cache (default True) persists density maps to disk so they are
    generated once instead of every epoch (see issue #15). cache_dir defaults
    to config.DENSITY_CACHE_DIR (data/processed/density_maps).
    """
    cfg = MODEL_CONFIGS[model_name]
    # from_config hardcodes target_size=DEFAULT_IMAGE_SIZE and val_split=VAL_SPLIT,
    # which is exactly what we want for the batched training path (uniform size).
    common = dict(
        root=root,
        downsample_factor=cfg["downsample_factor"],
        normalize=cfg["normalize"],
        use_cache=use_cache,
        cache_dir=cache_dir,
    )
    train_ds = CrowdCountingDataset.from_config(part=part, split="train", **common)
    val_ds = CrowdCountingDataset.from_config(part=part, split="val", **common)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=num_workers,
    )
    return train_loader, val_loader


# --------------------------------------------------------------------------------------
# Counting convention
# --------------------------------------------------------------------------------------
def density_to_count(density: torch.Tensor) -> torch.Tensor:
    """A density map's integral is the head count: sum over spatial dims.

    Input:  (B, 1, H', W') density tensor.
    Output: (B,) per-image counts.

    GT and predictions both use this. Using density.sum() for the GT count
    (rather than the raw annotation length) matches standard MCNN reference
    implementations and is what the dataset tests verify to be accurate.
    """
    return density.sum(dim=(1, 2, 3))


# --------------------------------------------------------------------------------------
# One epoch
# --------------------------------------------------------------------------------------
def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Run one training epoch.

    Returns (avg_pixel_mse, train_mae, train_rmse) over the epoch. The pixel MSE
    is the optimized loss; the MAE/RMSE are computed on counts and are the
    metrics the project actually reports (and what the paper reports).
    """
    model.train()
    total_loss = 0.0
    errs: list[float] = []

    for images, density in loader:
        images = images.to(device)
        density = density.to(device)

        optimizer.zero_grad()
        pred = model(images)
        loss = criterion(pred, density)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * images.size(0)

        with torch.no_grad():
            pred_counts = density_to_count(pred).cpu()
            gt_counts = density_to_count(density).cpu()
            errs.extend((pred_counts - gt_counts).tolist())

    n = len(loader.dataset)
    avg_loss = total_loss / n
    mae = float(np.mean(np.abs(errs)))
    rmse = float(math.sqrt(np.mean(np.square(errs))))
    return avg_loss, mae, rmse


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Compute (MAE, RMSE) on counts over a dataloader (val or test)."""
    model.eval()
    errs: list[float] = []
    for images, density in loader:
        images = images.to(device)
        density = density.to(device)
        pred = model(images)
        pred_counts = density_to_count(pred).cpu()
        gt_counts = density_to_count(density).cpu()
        errs.extend((pred_counts - gt_counts).tolist())
    mae = float(np.mean(np.abs(errs)))
    rmse = float(math.sqrt(np.mean(np.square(errs))))
    return mae, rmse


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a crowd counting model.")
    parser.add_argument("--model", default="mcnn", choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--part", default="A", choices=["A", "B"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=None,
                        help="override lr from MODEL_CONFIGS")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true",
                        help="disable density-map disk caching (regenerate every epoch)")
    parser.add_argument("--cache-dir", default=str(DENSITY_CACHE_DIR),
                        help="density-map cache root (default data/processed/density_maps)")
    parser.add_argument("--root", default=str(SHANGHAITECH_DIR))
    parser.add_argument("--out-dir", default=str(REPORTS_DIR / "checkpoints"))
    parser.add_argument("--smoke", action="store_true",
                        help="1-epoch smoke test on a tiny fake dataset")
    args = parser.parse_args(argv)

    device = get_device()
    print(f"[device] {device}")

    cfg = MODEL_CONFIGS[args.model]
    lr = args.lr if args.lr is not None else cfg["lr"]
    print(f"[config] model={args.model} part={args.part} "
          f"downsample={cfg['downsample_factor']} normalize={cfg['normalize']} "
          f"lr={lr} momentum={cfg['momentum']}")

    # --- smoke mode: build a throwaway fake dataset so we can run without the
    # real download. Exercises the entire loop end-to-end. Cache is forced off
    # so fake 'IMG_*' stems never pollute the real on-disk cache.
    use_cache = not args.no_cache
    if args.smoke:
        root = _make_fake_dataset(tempfile.mkdtemp(), part=args.part, n=8)
        args.root = root
        args.epochs = 1
        use_cache = False

    model = build_model(args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.model} params={n_params}")

    train_loader, val_loader = build_dataloaders(
        args.model, args.part, args.root,
        batch_size=args.batch_size, num_workers=args.num_workers,
        use_cache=use_cache, cache_dir=args.cache_dir,
    )
    print(f"[data] train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=cfg["momentum"],
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{args.model}_part{args.part}_best.pth"

    best_val_mae = math.inf
    for epoch in range(1, args.epochs + 1):
        train_loss, train_mae, train_rmse = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
        )
        val_mae, val_rmse = evaluate(model, val_loader, device)
        print(
            f"[epoch {epoch:3d}] "
            f"loss={train_loss:.6f} train_mae={train_mae:.2f} train_rmse={train_rmse:.2f} "
            f"| val_mae={val_mae:.2f} val_rmse={val_rmse:.2f}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(
                {"epoch": epoch, "model": args.model, "state_dict": model.state_dict(),
                 "val_mae": val_mae, "val_rmse": val_rmse},
                ckpt_path,
            )
            print(f"  -> saved {ckpt_path.name} (val_mae={val_mae:.2f})")

    print(f"[done] best val_mae={best_val_mae:.2f} -> {ckpt_path}")


# --------------------------------------------------------------------------------------
# Fake dataset for smoke tests (mirrors tests/test_dataset_*.py fixture)
# --------------------------------------------------------------------------------------
def _make_fake_dataset(root: str | Path, part: str = "A", n: int = 8) -> str:
    """Create a tiny fake ShanghaiTech layout on disk and return its root."""
    root = Path(root)
    img_dir = root / f"part_{part}" / "train_data" / "images"
    gt_dir = root / f"part_{part}" / "train_data" / "ground-truth"
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
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


if __name__ == "__main__":
    main()