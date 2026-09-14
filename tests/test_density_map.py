"""Tests for density-map generation edge cases."""

import numpy as np
import pytest

from src.datasets.density_map import adaptive_density_map, fixed_sigma_density_map


@pytest.mark.parametrize(
    "points, expected_count",
    [
        pytest.param([[10.1, 10.1], [10.1, 10.1]], 2, id="identical-points"),
        pytest.param([[10.1, 10.1], [10.2, 10.2]], 2, id="rounded-collision"),
        pytest.param([], 0, id="empty"),
        pytest.param([[10.0, 10.0], [20.0, 20.0]], 2, id="separate-pixels"),
        pytest.param(
            [[10.0, 10.0], [-0.6, 10.0], [31.6, 10.0],
             [10.0, -0.6], [10.0, 31.6]],
            1,
            id="rounded-indices-outside-image",
        ),
    ],
)
def test_fixed_sigma_density_map_preserves_valid_count(points, expected_count) -> None:
    """Every valid annotation contributes, even when rounded pixels collide."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)

    density = fixed_sigma_density_map(points, height=32, width=32, sigma=2.0)

    assert density.shape == (32, 32)
    assert density.dtype == np.float32
    assert np.isfinite(density).all()
    assert (density >= 0).all()
    assert float(density.sum()) == pytest.approx(expected_count, abs=1e-5)


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
