"""Central configuration for the crowd counting project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

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
