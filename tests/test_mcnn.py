"""Shape and behaviour tests for the MCNN model.

No real dataset needed — we feed random tensors and check the
input/output contract (channels, stride, count-from-sum, batch handling).
"""

from __future__ import annotations

import torch

from src.models.mcnn import MCNN


def test_output_shape_and_stride() -> None:
    """Output is 1-channel and 1/4 spatial size of the input (two 2x2 pools)."""
    model = MCNN().eval()
    x = torch.randn(1, 3, 256, 320)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 64, 80), f"got {tuple(out.shape)}"


def test_output_channels_are_one() -> None:
    """The fusion 1x1 conv collapses 120 channels to a single density channel."""
    model = MCNN().eval()
    out = model(torch.randn(2, 3, 128, 128))
    assert out.shape[1] == 1


def test_batch_handling() -> None:
    """The model handles batch sizes > 1 with no broadcasting surprises."""
    model = MCNN().eval()
    b = 4
    out = model(torch.randn(b, 3, 64, 64))
    assert out.shape[0] == b
    assert out.shape[2:] == (16, 16)


def test_odd_spatial_dims_floor_divide() -> None:
    """H/W not divisible by 4: pooling floors, output is H//4 x W//4."""
    model = MCNN().eval()
    out = model(torch.randn(1, 3, 66, 70))
    assert out.shape == (1, 1, 16, 17), f"got {tuple(out.shape)}"


def test_param_count_matches_columns() -> None:
    """Parameter count reflects the three columns + the 1x1 fusion."""
    model = MCNN()
    n = sum(p.numel() for p in model.parameters())
    # MCNN is a small net: three columns + a 1x1 fusion ~= 64k params.
    assert 60_000 < n < 70_000, f"param count {n} outside expected range"


def test_gradient_flows_to_all_columns() -> None:
    """A backward pass updates every column (no detached branch)."""
    model = MCNN()
    out = model(torch.randn(1, 3, 64, 64))
    out.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"


def test_count_from_density_sum() -> None:
    """Summing the output map yields a per-image scalar (the count proxy)."""
    model = MCNN().eval()
    out = model(torch.randn(3, 3, 128, 128))  # (B,1,H/4,W/4)
    counts = out.sum(dim=(1, 2, 3))  # (B,)
    assert counts.shape == (3,)