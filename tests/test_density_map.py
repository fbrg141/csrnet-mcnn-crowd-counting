"""Tests for density-map generation edge cases."""

import numpy as np
import pytest

from src.datasets.density_map import adaptive_density_map


def test_adaptive_density_map_single_head_uses_fallback_sigma() -> None:
    """A single head produces one finite Gaussian instead of crashing."""
    points = np.array([[32.0, 32.0]], dtype=np.float64)

    density = adaptive_density_map(
        points,
        height=64,
        width=64,
        fallback_sigma=2.0,
    )

    assert density.shape == (64, 64)
    assert density.dtype == np.float32
    assert np.isfinite(density).all()
    assert (density >= 0).all()
    assert np.unravel_index(np.argmax(density), density.shape) == (32, 32)
    assert float(density.sum()) == pytest.approx(1.0, abs=0.02)
