"""Model definitions for crowd counting."""

from __future__ import annotations

from src.models.csrnet import CSRNet
from src.models.mcnn import MCNN

# Registry of available models. Used by train/evaluate to pick the right
# architecture and to look up the per-model config in src.config.MODEL_CONFIGS.
MODELS: dict[str, type] = {
    "mcnn": MCNN,
    "csrnet": CSRNet,
}


def build_model(name: str):
    """Instantiate a model by name.

    Args:
        name: one of the keys in MODELS (e.g. "mcnn", "csrnet").

    Raises:
        ValueError: if the name is not a registered model.
    """
    key = name.lower()
    if key not in MODELS:
        raise ValueError(
            f"unknown model {name!r}; available: {sorted(MODELS)}"
        )
    return MODELS[key]()