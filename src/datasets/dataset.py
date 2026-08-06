"""PyTorch dataset for ShanghaiTech crowd counting."""

from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import Dataset

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

    def __len__(self) -> int:
        return len(self.images)

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
            image = image.resize((target_w, target_h), Image.BILINEAR)
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

        # Generate density map
        if self.density_mode == "fixed":
            density = fixed_sigma_density_map(points, h_dens, w_dens, sigma=sigma)
        elif self.density_mode == "adaptive":
            density = adaptive_density_map(points, h_dens, w_dens, k=self.k, beta=self.beta)
        else:
            raise ValueError(f"Unknown density_mode: {self.density_mode}")

        # Convert to tensors
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # ImageNet normalization for pretrained VGG frontend (CSRNet).
        # Applied after /255.0 so the values are in [0, 1] before normalizing.
        if self.normalize:
            image_tensor = normalize_imagenet(image_tensor)

        density_tensor = torch.from_numpy(density).unsqueeze(0).float()

        return image_tensor, density_tensor
