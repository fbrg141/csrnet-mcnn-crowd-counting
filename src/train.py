"""Training entrypoint for crowd counting models.

Run:
    python -m src.train --model mcnn --part A --epochs 50 --seed 42
    python -m src.train --model mcnn --part A --smoke   # 1 epoch, tiny fake set

The loop is intentionally model-agnostic: the per-model knobs (output stride,
input normalization, learning rate, momentum) come from src.config.MODEL_CONFIGS
so MCNN and CSRNet are trained under comparable, paper-faithful conditions.
"""

from __future__ import annotations

import argparse
import math
import json
import time
import random
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
from src.config import CACHE_VERSION, DEFAULT_IMAGE_SIZE
from src.runs import (atomic_checkpoint, atomic_json, completed, digest, file_hash,
                      mark_complete, source_hash, validate_hparams)


# --------------------------------------------------------------------------------------
# Experiment identity and reproducibility
# --------------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed the random generators used by the training pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def experiment_stem(model_name: str, part: str, seed: int, smoke: bool = False) -> str:
    """Return the shared filename stem for one experimental run."""
    stem = f"{model_name}_part{part}_seed{seed}"
    return f"{stem}_smoke" if smoke else stem


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
        if not torch.isfinite(loss):
            raise ValueError('non-finite training loss; trial stopped')
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * images.size(0)

        with torch.no_grad():
            pred_counts = density_to_count(pred).cpu()
            gt_counts = density_to_count(density).cpu()
            errs.extend((pred_counts - gt_counts).tolist())

    n = len(errs)
    if not n:
        raise ValueError('empty training loader (check dataset and batch size)')
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
    if not errs or not np.isfinite(errs).all():
        raise ValueError('empty evaluation loader or non-finite predictions')
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
    parser.add_argument('--momentum', type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--resume', action='store_true', help='resume last epoch or verify/skip a completed run')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default: 42)")
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

    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or not 0 <= args.seed < 2**32:
        parser.error('positive epochs/batch size, nonnegative workers and uint32 seed required')
    cfg = MODEL_CONFIGS[args.model]
    lr = args.lr if args.lr is not None else cfg['lr']
    momentum = args.momentum if args.momentum is not None else cfg['momentum']
    validate_hparams(lr, momentum, args.weight_decay)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = get_device() if args.device == 'auto' else torch.device(args.device)
    print(f'[device] {device}')
    use_cache = not args.no_cache and not args.smoke
    # TemporaryDirectory cleans smoke inputs even when a run fails.
    with tempfile.TemporaryDirectory() as temporary:
        if args.smoke:
            args.root = _make_fake_dataset(temporary, part=args.part, n=8)
            args.epochs = 1
        train_loader, val_loader = build_dataloaders(
            args.model, args.part, args.root, args.batch_size, args.num_workers,
            use_cache=use_cache, cache_dir=args.cache_dir,
        )
        if args.smoke:
            for loader in (train_loader, val_loader):
                loader.dataset.target_size = (64, 64)
        if not len(train_loader) or not len(val_loader):
            raise ValueError('empty train/validation loader; check dataset and batch size')
        data = {}
        for split, loader in [('train', train_loader), ('val', val_loader)]:
            data[split] = [(p.name, file_hash(p)) for p in
                           loader.dataset.images + loader.dataset.gts]
        config = {
            'model': args.model, 'part': args.part, 'seed': args.seed,
            'epochs': args.epochs, 'batch_size': args.batch_size,
            'lr': lr, 'momentum': momentum, 'weight_decay': args.weight_decay,
            'num_workers': args.num_workers, 'smoke': args.smoke,
            'root': 'synthetic-v1' if args.smoke else str(Path(args.root).resolve()),
            'dataset_hash': digest('synthetic-v1') if args.smoke else digest(data), 'source_hash': source_hash(),
            'cache_version': CACHE_VERSION, 'use_cache': use_cache,
            'image_size': list(train_loader.dataset.target_size),
            'density_mode': train_loader.dataset.density_mode,
            'sigma': train_loader.dataset.sigma, 'k': train_loader.dataset.k,
            'beta': train_loader.dataset.beta,
            'val_split': VAL_SPLIT, 'split_policy': 'lexicographic-tail',
            'downsample_factor': cfg['downsample_factor'], 'normalize': cfg['normalize'],
            'optimizer': 'SGD', 'loss': 'pixel_mean_MSE', 'drop_last': True,
            'device': str(device), 'torch_version': str(torch.__version__),
            'n_train': len(train_loader.dataset), 'n_val': len(val_loader.dataset),
        }
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        config_path = out_dir / 'config.json'
        if config_path.exists():
            if json.loads(config_path.read_text()) != config:
                raise ValueError('run configuration changed; use a new output directory')
            if not args.resume:
                raise ValueError('run exists; use --resume or a new output directory')
        elif any(out_dir.glob('*.pth')):
            raise ValueError('legacy checkpoints in output directory; use a new directory')
        if completed(out_dir, config):
            print('[skip] verified completed run')
            return
        atomic_json(config_path, config)
        ckpt_path = out_dir / f"{experiment_stem(args.model, args.part, args.seed, smoke=args.smoke)}_best.pth"
        last_path = out_dir / 'last.pth'
        model = build_model(args.model, pretrained=not args.smoke and not last_path.exists()).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                    weight_decay=args.weight_decay)
        history = []
        best = None
        elapsed = 0.0
        if args.resume and last_path.exists():
            last = torch.load(last_path, map_location='cpu', weights_only=False)
            if last['config'] != config:
                raise ValueError('resume checkpoint configuration mismatch')
            model.load_state_dict(last['state_dict'])
            optimizer.load_state_dict(last['optimizer'])
            history, best, elapsed = last['history'], last['best'], last['elapsed_seconds']
            random.setstate(last['rng']['python'])
            np.random.set_state(last['rng']['numpy'])
            torch.set_rng_state(last['rng']['torch'])
            if last['rng']['cuda'] is not None:
                torch.cuda.set_rng_state_all(last['rng']['cuda'])
            # Recover the best checkpoint if a disconnect lost that separate write.
            atomic_checkpoint(ckpt_path, best)
            print(f"[resume] epoch {len(history) + 1}")
        start = time.perf_counter()
        for epoch in range(len(history) + 1, args.epochs + 1):
            train_loss, train_mae, train_rmse = train_one_epoch(
                model, train_loader, optimizer, torch.nn.MSELoss(), device)
            val_mae, val_rmse = evaluate(model, val_loader, device)
            row = dict(epoch=epoch, train_loss=train_loss, train_mae=train_mae,
                       train_rmse=train_rmse, val_mae=val_mae, val_rmse=val_rmse)
            if not all(math.isfinite(v) for v in row.values()):
                raise ValueError('non-finite epoch metrics; trial stopped')
            history.append(row)
            print(f'[epoch {epoch}] {row}', flush=True)
            if best is None or val_mae < best['val_mae']:
                best = dict(epoch=epoch, model=args.model, part=args.part, seed=args.seed,
                            val_mae=val_mae, val_rmse=val_rmse, config=config,
                            state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                atomic_checkpoint(ckpt_path, best)
            atomic_checkpoint(last_path, {
                'config': config, 'epoch': epoch, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'history': history, 'best': best,
                'elapsed_seconds': elapsed + time.perf_counter() - start,
                'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                        'torch': torch.get_rng_state(),
                        'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None},
            })
            atomic_json(out_dir / 'history.json', history)
        atomic_json(out_dir / 'history.json', history)
        summary = dict(config=config, checkpoint=ckpt_path.name, best_epoch=best['epoch'],
                       val_mae=best['val_mae'], val_rmse=best['val_rmse'],
                       epochs_completed=len(history), elapsed_seconds=elapsed + time.perf_counter() - start)
        atomic_json(out_dir / 'summary.json', summary)
        mark_complete(out_dir, summary)
        print(f"[done] best val_mae={best['val_mae']:.4f} -> {ckpt_path}")


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