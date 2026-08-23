"""PyTorch dataset for ShanghaiTech crowd counting."""

import os
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.config import (
    ADAPTIVE_BETA,
    ADAPTIVE_K,
    CACHE_VERSION,
    DEFAULT_DENSITY_MODE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_PART,
    DENSITY_CACHE_DIR,
    FIXED_SIGMA,
    SHANGHAITECH_DIR,
    VAL_SPLIT,
)
from src.datasets.density_map import adaptive_density_map, fixed_sigma_density_map

# ImageNet statistics expected by pretrained VGG frontends (e.g. CSRNet's
# VGG16 backbone). Inputs are first scaled to [0, 1] via /255.0, then
# standardized with these per-channel mean/std (RGB order).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_imagenet(image: torch.Tensor) -> torch.Tensor:
    """Apply per-channel ImageNet normalization to a (C, H, W) float tensor.

    The input is assumed to already be in [0, 1] (i.e. after /255.0).
    Returns a new tensor standardized with ImageNet mean/std.
    """
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"expected a (3, H, W) tensor, got shape {tuple(image.shape)}")
    mean = torch.tensor(IMAGENET_MEAN, device=image.device, dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device, dtype=image.dtype).view(3, 1, 1)
    return (image - mean) / std


# Map from part name to (train_count, test_count)
PART_INFO = {
    "A": (300, 182),
    "B": (400, 316),
}


def _load_annotations(gt_path: Path) -> np.ndarray:
    """Load (x, y) head coordinates from a ShanghaiTech .mat file."""
    mat = sio.loadmat(gt_path)
    points = mat["image_info"][0, 0][0, 0][0]  # (N, 2) float array
    return points


class CrowdCountingDataset(Dataset):
    """ShanghaiTech crowd counting dataset.

    Args:
        root: Path to the dataset root (data/raw/ShanghaiTech).
        part: "A" or "B".
        split: "train" or "test".
        density_mode: "fixed" or "adaptive".
        sigma: Fixed sigma for Gaussian kernel (used when mode="fixed").
        k: Number of nearest neighbours (used when mode="adaptive").
        beta: Scaling factor for adaptive sigma.
        target_size: If set, resize image and scale coordinates to this (H, W).
                     If None, keep original size.
        val_split: Fraction of training set to hold out for validation (0.0 = no val).
        downsample_factor: Factor by which to reduce the density-map resolution
                     relative to the (possibly resized) image, to match model
                     output strides (e.g. 4 for MCNN, 8 for CSRNet). When > 1,
                     head coordinates are scaled by 1/factor, the density map
                     is generated at (H//factor, W//factor), and the fixed-mode
                     sigma is scaled by 1/factor so the Gaussian width matches.
                     When 1 (default), behaviour is unchanged.
        normalize: If True, apply ImageNet normalization (mean=[0.485, 0.456,
            0.406], std=[0.229, 0.224, 0.225]) after dividing by 255.0. Required
            for CSRNet's pretrained VGG16 frontend; leave False otherwise.
        use_cache: If True, persist generated density maps to disk as .npy and
            load them on later accesses instead of regenerating. The cache key
            encodes every parameter that affects the density output (mode,
            sigma/k/beta, target_size, downsample_factor, CACHE_VERSION), so a
            stale or mismatched cache is never loaded. Default False to keep
            the historical on-the-fly behaviour.
        cache_dir: Root directory for the density cache (only used when
            use_cache=True). If None with use_cache=True, defaults to
            config.DENSITY_CACHE_DIR (data/processed/density_maps).
    """

    def __init__(
        self,
        root: str | Path,
        part: str = "A",
        split: str = "train",
        density_mode: str = "fixed",
        sigma: float = 15.0,
        k: int = 4,
        beta: float = 0.3,
        target_size: tuple[int, int] | None = None,
        val_split: float = 0.0,
        downsample_factor: int = 1,
        normalize: bool = False,
        use_cache: bool = False,
        cache_dir: str | Path | None = None,
    ) -> None:
        part = part.upper()
        if part not in ("A", "B"):
            raise ValueError(f"part must be 'A' or 'B', got {part!r}")
        if split not in ("train", "test", "val"):
            raise ValueError(f"split must be 'train', 'test', or 'val', got {split!r}")
        if not isinstance(downsample_factor, int) or downsample_factor < 1:
            raise ValueError(
                f"downsample_factor must be a positive integer, got {downsample_factor!r}"
            )

        self.root = Path(root)
        self.part = part
        self.split = split
        self.density_mode = density_mode
        self.sigma = sigma
        self.k = k
        self.beta = beta
        self.target_size = target_size
        self.downsample_factor = downsample_factor
        self.normalize = normalize
        self.use_cache = use_cache
        self._cache_dir = self._resolve_cache_dir(cache_dir)
        self._cache_subdir = self._compute_cache_subdir() if self.use_cache else None

        # Paths
        split_name = "train_data" if split in ("train", "val") else "test_data"
        data_dir = self.root / f"part_{part}" / split_name
        self.img_dir = data_dir / "images"
        self.gt_dir = data_dir / "ground-truth"

        # List all image files (sorted for reproducibility)
        all_images = sorted(self.img_dir.glob("*.jpg"))
        all_gts = sorted(self.gt_dir.glob("*.mat"))

        if len(all_images) != len(all_gts):
            raise RuntimeError(
                f"Mismatch: {len(all_images)} images vs {len(all_gts)} annotations"
            )

        # Train/val split
        if split == "val" and val_split > 0.0:
            n_val = max(1, int(len(all_images) * val_split))
            self.images = all_images[-n_val:]
            self.gts = all_gts[-n_val:]
        elif split == "train" and val_split > 0.0:
            n_val = max(1, int(len(all_images) * val_split))
            self.images = all_images[:-n_val]
            self.gts = all_gts[:-n_val]
        else:
            self.images = all_images
            self.gts = all_gts

    @classmethod
    def from_config(
        cls,
        part: str = DEFAULT_PART,
        split: str = "train",
        downsample_factor: int = 1,
        normalize: bool = False,
        use_cache: bool = False,
        cache_dir: str | Path | None = None,
        root: str | Path | None = None,
    ) -> "CrowdCountingDataset":
        """Create a dataset using defaults from src.config.

        Model-specific parameters (downsample_factor, normalize) must be
        passed explicitly since they depend on the model:
        - MCNN:    downsample_factor=4,  normalize=False
        - CSRNet:  downsample_factor=8,  normalize=True

        Args:
            part: "A" or "B" (defaults to config.DEFAULT_PART).
            split: "train", "val", or "test".
            downsample_factor: Reduce density map resolution (4 for MCNN, 8 for CSRNet).
            normalize: Apply ImageNet normalization (True for CSRNet, False for MCNN).
            use_cache: Persist density maps to disk and load on later accesses.
            cache_dir: Cache root (defaults to config.DENSITY_CACHE_DIR).
            root: Dataset root path (defaults to config.SHANGHAITECH_DIR).
        """
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
            use_cache=use_cache,
            cache_dir=cache_dir or DENSITY_CACHE_DIR,
        )

    def __len__(self) -> int:
        return len(self.images)

    # ------------------------------------------------------------------
    # Density-map caching
    # ------------------------------------------------------------------
    def _resolve_cache_dir(self, cache_dir: str | Path | None) -> Path | None:
        """Resolve the cache root directory (only meaningful when use_cache)."""
        if not self.use_cache:
            return None
        return Path(cache_dir) if cache_dir is not None else DENSITY_CACHE_DIR

    def _compute_cache_subdir(self) -> Path:
        """Build the cache subdirectory encoding every param that affects the
        density output. A change in any of these yields a different path, so a
        stale or mismatched cache file is never loaded.
        """
        if self.target_size is None:
            sz = "orig"
        else:
            sz = f"{self.target_size[0]}x{self.target_size[1]}"
        if self.density_mode == "fixed":
            params = f"sigma{self.sigma}"
        elif self.density_mode == "adaptive":
            params = f"k{self.k}_beta{self.beta}"
        else:
            params = f"mode{self.density_mode}"
        tag = (
            f"v{CACHE_VERSION}_{self.density_mode}"
            f"_ds{self.downsample_factor}_sz{sz}_{params}"
        )
        return Path(f"part_{self.part}") / self.split / tag

    def _cache_path(self, stem: str) -> Path:
        """Full .npy path for one image's density map (use_cache must be True)."""
        return self._cache_dir / self._cache_subdir / f"{stem}.npy"

    def _generate_density(
        self, points: np.ndarray, h_dens: int, w_dens: int, sigma: float
    ) -> np.ndarray:
        """Compute the density map from scratch (the historical path)."""
        if self.density_mode == "fixed":
            return fixed_sigma_density_map(points, h_dens, w_dens, sigma=sigma)
        if self.density_mode == "adaptive":
            return adaptive_density_map(
                points, h_dens, w_dens, k=self.k, beta=self.beta
            )
        raise ValueError(f"Unknown density_mode: {self.density_mode}")

    def _load_or_compute_density(
        self, points: np.ndarray, h_dens: int, w_dens: int, sigma: float, stem: str
    ) -> np.ndarray:
        """Load the density map from disk cache if present, else generate and
        atomically persist it. Falls back to plain generation when caching is off.
        """
        if not self.use_cache:
            return self._generate_density(points, h_dens, w_dens, sigma)
        path = self._cache_path(stem)
        if path.exists():
            return np.load(path)
        density = self._generate_density(points, h_dens, w_dens, sigma)
        self._save_npy_atomic(density, path)
        return density

    @staticmethod
    def _save_npy_atomic(arr: np.ndarray, path: Path) -> None:
        """Write arr to path via a temp file + atomic rename.

        The file-handle form of np.save is used so np.save does not append a
        stray '.npy' to the temp name. os.replace is atomic on POSIX/Windows,
        so concurrent DataLoader workers can't observe a half-written file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, arr)
        os.replace(tmp, path)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.images[index]
        gt_path = self.gts[index]

        # Load image
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Load annotations
        points = _load_annotations(gt_path)

        # Resize if requested
        if self.target_size is not None:
            target_h, target_w = self.target_size
            image = image.resize((target_w, target_h), Image.Resampling.BILINEAR)
            # Scale coordinates
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h
            points = points.copy()
            points[:, 0] *= scale_x
            points[:, 1] *= scale_y
            h, w = target_h, target_w
        else:
            h, w = orig_h, orig_w

        # Downsample the density map to match the model output stride.
        # The image itself stays at full resolution; only the target density
        # map (and the head coordinates used to build it) are scaled down.
        if self.downsample_factor > 1:
            factor = self.downsample_factor
            h_dens = h // factor
            w_dens = w // factor
            if h_dens < 1 or w_dens < 1:
                raise ValueError(
                    f"downsample_factor={factor} reduces {h}x{w} below 1 pixel; "
                    "use a smaller factor or a larger target_size"
                )
            points = points.copy()
            points[:, 0] /= factor
            points[:, 1] /= factor
            sigma = self.sigma / factor
        else:
            h_dens, w_dens = h, w
            sigma = self.sigma

        # Generate density map (or load from disk cache when enabled)
        density = self._load_or_compute_density(
            points, h_dens, w_dens, sigma, img_path.stem
        )

        # Convert to tensors
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # ImageNet normalization for pretrained VGG frontend (CSRNet).
        # Applied after /255.0 so the values are in [0, 1] before normalizing.
        if self.normalize:
            image_tensor = normalize_imagenet(image_tensor)

        density_tensor = torch.from_numpy(density).unsqueeze(0).float()

        return image_tensor, density_tensor
