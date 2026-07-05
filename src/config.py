"""Central configuration for the crowd counting project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_IMAGE_SIZE = (768, 1024)
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_EPOCHS = 50
DEFAULT_LEARNING_RATE = 1e-4
