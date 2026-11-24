
import copy

import pytest
import torch
from torch import nn

from src.fedresearch.utils.experiment import (
    configure_reproducibility,
    state_dict_sha256,
    save_experiment,
    load_verified_checkpoint,
)


def make_model():
    return nn.Linear(4, 2)


def test_seed_reproducibility():
    configure_reproducibility(42)
    first = torch.randn(5)

    configure_reproducibility(42)
    second = torch.randn(5)

    assert torch.equal(first, second)


def test_model_hash_stable():
    model = make_model()

    first = state_dict_sha256(model.state_dict())
    second = state_dict_sha256(model.state_dict())

    assert first == second


def test_model_hash_changes():
    model = make_model()

    before = state_dict_sha256(model.state_dict())

    with torch.no_grad():
        model.weight[0, 0] += 1.0

    after = state_dict_sha256(model.state_dict())

    assert before != after


def test_checkpoint_roundtrip(tmp_path):
    configure_reproducibility(42)

    model = make_model()
    original = copy.deepcopy(model.state_dict())

    output_dir = tmp_path / "experiment"

    config = {
        "seed": 42,
        "rounds": 3,
        "algorithm": "fedavg",
    }

    save_experiment(
        output_dir,
        model,
        config,
        history=[{"round": 1, "accuracy": 0.5}],
    )

    restored = make_model()

    loaded_config = load_verified_checkpoint(
        output_dir / "checkpoint.pt",
        restored,
    )

    assert loaded_config == config

    for name, tensor in original.items():
        assert torch.equal(
            tensor,
            restored.state_dict()[name],
        )


def test_no_overwrite(tmp_path):
    model = make_model()
    output_dir = tmp_path / "experiment"

    save_experiment(
        output_dir,
        model,
        {"seed": 42},
        [],
    )

    with pytest.raises(FileExistsError):
        save_experiment(
            output_dir,
            model,
            {"seed": 42},
            [],
        )


def test_tampered_checkpoint_rejected(tmp_path):
    model = make_model()
    output_dir = tmp_path / "experiment"

    save_experiment(
        output_dir,
        model,
        {"seed": 42},
        [],
    )

    path = output_dir / "checkpoint.pt"

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    checkpoint["model_state_dict"]["weight"][0, 0] += 1

    torch.save(checkpoint, path)

    with pytest.raises(ValueError):
        load_verified_checkpoint(
            path,
            make_model(),
        )
