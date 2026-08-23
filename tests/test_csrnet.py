"""Architecture and behaviour tests for the CSRNet model."""

from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn
from torchvision.models import VGG16_Weights, vgg16
from torchvision.models.vgg import VGG

import src.models.csrnet as csrnet_module
from src.models.csrnet import CSRNet


def build_offline_model() -> CSRNet:
    """Build CSRNet without downloading ImageNet weights."""
    return CSRNet(weights=None)


def test_default_uses_pinned_imagenet_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """The constructor forwards V1 and preserves loaded frontend weights."""
    default = inspect.signature(CSRNet).parameters["weights"].default
    reference = vgg16(weights=None)
    sentinel = 0.125
    first_convolution = next(
        layer for layer in reference.features if isinstance(layer, nn.Conv2d)
    )
    nn.init.constant_(first_convolution.weight, sentinel)
    captured_weights: list[VGG16_Weights | None] = []

    def fake_vgg16(*, weights: VGG16_Weights | None) -> VGG:
        captured_weights.append(weights)
        return reference

    monkeypatch.setattr(csrnet_module, "vgg16", fake_vgg16)
    model = CSRNet()
    loaded_first_convolution = model.frontend[0]

    assert default is VGG16_Weights.IMAGENET1K_V1
    assert captured_weights == [VGG16_Weights.IMAGENET1K_V1]
    assert isinstance(loaded_first_convolution, nn.Conv2d)
    assert torch.all(loaded_first_convolution.weight == sentinel)


def test_frontend_matches_vgg16_conv4_3() -> None:
    """The frontend exactly matches torchvision VGG16 features through conv4_3."""
    model = build_offline_model()
    reference = list(vgg16(weights=None).features.children())[:23]
    convolutions = [layer for layer in model.frontend if isinstance(layer, nn.Conv2d)]
    pools = [layer for layer in model.frontend if isinstance(layer, nn.MaxPool2d)]

    assert len(model.frontend) == len(reference)
    for actual, expected in zip(model.frontend, reference):
        assert type(actual) is type(expected)
        if isinstance(actual, nn.Conv2d) and isinstance(expected, nn.Conv2d):
            assert actual.in_channels == expected.in_channels
            assert actual.out_channels == expected.out_channels
            assert actual.kernel_size == expected.kernel_size
            assert actual.stride == expected.stride
            assert actual.padding == expected.padding
            assert actual.dilation == expected.dilation
        if isinstance(actual, nn.MaxPool2d) and isinstance(expected, nn.MaxPool2d):
            assert actual.kernel_size == expected.kernel_size
            assert actual.stride == expected.stride
            assert actual.padding == expected.padding

    assert len(convolutions) == 10
    assert len(pools) == 3
    assert convolutions[-1].out_channels == 512
    assert all(parameter.requires_grad for parameter in model.frontend.parameters())


def test_backend_matches_full_configuration_b() -> None:
    """CSRNet-B uses six dilation-2 convolutions ending in 64 channels."""
    model = build_offline_model()
    convolutions = [layer for layer in model.backend if isinstance(layer, nn.Conv2d)]

    assert [layer.in_channels for layer in convolutions] == [512, 512, 512, 512, 256, 128]
    assert [layer.out_channels for layer in convolutions] == [512, 512, 512, 256, 128, 64]
    assert all(layer.kernel_size == (3, 3) for layer in convolutions)
    assert all(layer.dilation == (2, 2) for layer in convolutions)
    assert all(layer.padding == (2, 2) for layer in convolutions)


def test_output_layer_is_linear_one_by_one_convolution() -> None:
    """The density head is Conv(64->1, 1x1) with no output activation."""
    model = build_offline_model()

    assert isinstance(model.output_layer, nn.Conv2d)
    assert model.output_layer.in_channels == 64
    assert model.output_layer.out_channels == 1
    assert model.output_layer.kernel_size == (1, 1)

    inputs = torch.randn(1, 3, 32, 40)
    with torch.no_grad():
        expected = model.output_layer(model.backend(model.frontend(inputs)))
        actual = model(inputs)
    torch.testing.assert_close(actual, expected)


def test_output_shape_has_stride_eight() -> None:
    """Three frontend pools produce a density map at one-eighth resolution."""
    model = build_offline_model().eval()
    inputs = torch.randn(2, 3, 64, 80)

    with torch.no_grad():
        output = model(inputs)

    assert output.shape == (2, 1, 8, 10)


def test_odd_spatial_dimensions_floor_divide() -> None:
    """Pooling floors non-divisible dimensions consistently with dataset targets."""
    model = build_offline_model().eval()

    with torch.no_grad():
        output = model(torch.randn(1, 3, 66, 70))

    assert output.shape == (1, 1, 66 // 8, 70 // 8)


def test_gradients_flow_through_all_csrnet_stages() -> None:
    """A backward pass reaches the frontend, backend, and output layer."""
    model = build_offline_model()
    output = model(torch.randn(1, 3, 32, 32))

    output.sum().backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient for {name}"


def test_new_layers_use_original_csrnet_initialization() -> None:
    """Backend/output weights use N(0, 0.01²), with zero biases."""
    model = build_offline_model()
    layers = [
        layer
        for layer in (*model.backend, model.output_layer)
        if isinstance(layer, nn.Conv2d)
    ]
    weights = torch.cat([layer.weight.detach().flatten() for layer in layers])

    assert float(weights.mean()) == pytest.approx(0.0, abs=2e-4)
    assert float(weights.std()) == pytest.approx(0.01, rel=0.05)
    for layer in layers:
        assert layer.bias is not None
        assert torch.count_nonzero(layer.bias) == 0
