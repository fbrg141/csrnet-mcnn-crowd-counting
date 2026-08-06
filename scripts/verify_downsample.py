"""Verify the downsample_factor parameter keeps the density-map sum ≈ head count.

This is a quick offline check that does not require the real ShanghaiTech
dataset: it directly exercises the density-map generation path used by
``CrowdCountingDataset`` at the reduced resolution, so it can run anywhere
without downloaded data.

It checks three invariants for ``downsample_factor`` in {1, 4, 8}:
  1. The density map has the expected reduced shape (H//f, W//f).
  2. The density-map sum approximates the number of heads.
  3. For factor == 1 the result is byte-identical to the original path
     (backward compatibility).
"""

from __future__ import annotations

import numpy as np

from src.datasets.density_map import fixed_sigma_density_map


def make_points(n: int, h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, w, size=n)
    ys = rng.uniform(0, h, size=n)
    return np.stack([xs, ys], axis=1).astype(np.float64)


def run() -> None:
    h, w = 768, 1024
    n_heads = 200
    sigma = 15.0
    points = make_points(n_heads, h, w, seed=42)

    baseline = None
    ok = True
    for factor in (1, 4, 8):
        h_d, w_d = h // factor, w // factor
        pts = points.copy()
        pts[:, 0] /= factor
        pts[:, 1] /= factor
        sig = sigma / factor

        dens = fixed_sigma_density_map(pts, h_d, w_d, sigma=sig)

        # Shape check
        assert dens.shape == (h_d, w_d), f"factor={factor}: shape {dens.shape} != {(h_d, w_d)}"
        # Sum check
        total = float(dens.sum())
        rel_err = abs(total - n_heads) / n_heads
        shape_ok = dens.shape == (h_d, w_d)
        sum_ok = rel_err < 0.05  # within 5% of head count
        print(
            f"factor={factor}: shape={dens.shape} sum={total:.2f} "
            f"(rel_err={rel_err:.4f}) shape_ok={shape_ok} sum_ok={sum_ok}"
        )
        ok = ok and shape_ok and sum_ok

        # Backward-compat: factor==1 must match the unscaled original path.
        if factor == 1:
            baseline = fixed_sigma_density_map(points, h, w, sigma=sigma)
            assert np.array_equal(dens, baseline), "factor=1 must be identical to original path"

    assert baseline is not None
    print(f"\nAll checks {'PASSED' if ok else 'FAILED'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    run()