"""Central configuration for the crowd counting project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
DENSITY_CACHE_DIR = PROCESSED_DATA_DIR / "density_maps"

# Bump this when the density-map *generation logic* changes (e.g. editing
# fixed_sigma_density_map / adaptive_density_map), so on-disk caches built
# with the old logic are no longer loaded. It is embedded in every cache
# path, so a bump silently invalidates the whole cache.
CACHE_VERSION = 1

# Dataset
SHANGHAITECH_DIR = RAW_DATA_DIR / "ShanghaiTech"
DEFAULT_PART = "A"
DEFAULT_DENSITY_MODE = "fixed"  # "fixed" or "adaptive"
FIXED_SIGMA = 15.0
ADAPTIVE_K = 4
ADAPTIVE_BETA = 0.3

# Image preprocessing
DEFAULT_IMAGE_SIZE = (768, 1024)  # (H, W) — resize to this
VAL_SPLIT = 0.1  # fraction of training set for validation

# Training
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_EPOCHS = 50
DEFAULT_LEARNING_RATE = 1e-4

# Per-model settings. These are the values the paper / standard references use.
# They must NOT be collapsed into a single global default, because the two
# models differ in crucial ways:
#   - downsample_factor: matches each model's output stride (MCNN=4, CSRNet=8)
#     so the GT density map aligns with the prediction.
#   - normalize: CSRNet uses a pretrained VGG16 frontend (needs ImageNet norm);
#     MCNN is trained from scratch (raw [0,1] inputs).
#   - lr: MCNN is from-scratch and fragile (paper: 1e-6); CSRNet fine-tunes a
#     pretrained backbone and tolerates a larger lr (1e-5).
MODEL_CONFIGS = {
    "mcnn": {
        "downsample_factor": 4,
        "normalize": False,
        "lr": 1e-6,
        "momentum": 0.95,
    },
    "csrnet": {
        "downsample_factor": 8,
        "normalize": True,
        "lr": 1e-5,
        "momentum": 0.95,
    },
}

# SGD momentum used for all models (kept here for clarity; also in MODEL_CONFIGS).
DEFAULT_MOMENTUM = 0.95
