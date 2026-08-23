"""Density map generation functions for crowd counting."""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree


def fixed_sigma_density_map(
    points: np.ndarray,
    height: int,
    width: int,
    sigma: float = 15.0,
) -> np.ndarray:
    """Generate a density map using a fixed-size Gaussian kernel.

    Each head coordinate is placed as a delta peak, then convolved
    with a Gaussian of fixed sigma.

    Args:
        points: (N, 2) array of (x, y) head coordinates.
        height: Image height in pixels.
        width: Image width in pixels.
        sigma: Standard deviation of the Gaussian kernel.

    Returns:
        (height, width) density map. Sum ≈ number of heads.
    """
    density = np.zeros((height, width), dtype=np.float32)
    for x, y in points:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < width and 0 <= iy < height:
            density[iy, ix] = 1.0

    density = gaussian_filter(density, sigma=sigma)
    return density


def adaptive_density_map(
    points: np.ndarray,
    height: int,
    width: int,
    k: int = 4,
    beta: float = 0.3,
    fallback_sigma: float = 2.0,
) -> np.ndarray:
    """Generate a density map using geometry-adaptive Gaussian kernels.

    Each head gets its own sigma proportional to the average distance
    to its k nearest neighbours. Dense crowds get sharper peaks,
    sparse crowds get wider peaks.

    Args:
        points: (N, 2) array of (x, y) head coordinates.
        height: Image height in pixels.
        width: Image width in pixels.
        k: Number of nearest neighbours to average (excluding self).
        beta: Scaling factor for sigma = beta * avg_distance.
        fallback_sigma: Sigma used when avg_distance ≈ 0 (isolated point).

    Returns:
        (height, width) density map. Sum ≈ number of heads.
    """
    density = np.zeros((height, width), dtype=np.float32)
    n_points = len(points)

    if n_points == 0:
        return density

    tree = KDTree(points)
    # At least 2 neighbours so we can exclude self
    n_neighbours = min(k + 1, n_points)

    for x, y in points:
        distances, _ = tree.query((x, y), k=n_neighbours)
        distances = np.atleast_1d(distances)
        # distances[0] is distance to self (≈ 0)
        neighbour_distances = distances[1:]
        sigma = (
            beta * np.mean(neighbour_distances)
            if neighbour_distances.size > 0
            else fallback_sigma
        )

        if sigma < 1e-5:
            sigma = fallback_sigma

        # Place Gaussian centred at (x, y)
        cx, cy = int(round(x)), int(round(y))
        radius = int(3 * sigma)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                yy = cy + dy
                xx = cx + dx
                if 0 <= yy < height and 0 <= xx < width:
                    density[yy, xx] += (
                        (1.0 / (2 * np.pi * sigma**2))
                        * np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
                    )

    return density
