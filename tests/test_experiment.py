import random

import numpy as np
import torch

from src.train import experiment_stem, set_seed


def _random_sample() -> tuple[float, float, torch.Tensor]:
    return random.random(), float(np.random.random()), torch.rand(3)


def test_set_seed_repeats_python_numpy_and_torch_sequences() -> None:
    set_seed(123)
    first = _random_sample()

    set_seed(123)
    second = _random_sample()

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_experiment_stem_includes_model_part_and_seed() -> None:
    assert experiment_stem("mcnn", "A", 42) == "mcnn_partA_seed42"
    assert experiment_stem("csrnet", "B", 2026) == "csrnet_partB_seed2026"
