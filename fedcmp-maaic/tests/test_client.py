
import copy

import pytest
import torch
from torch.utils.data import TensorDataset

from src.fedresearch.fl.client import (
    train_client,
    set_seed,
)
from src.fedresearch.models.cnn import MNISTCNN


@pytest.fixture
def small_dataset():
    images = torch.randn(16, 1, 28, 28)
    labels = torch.arange(16) % 10
    return TensorDataset(images, labels)


def test_global_model_unchanged(small_dataset):
    set_seed(42)
    global_model = MNISTCNN()

    before = copy.deepcopy(global_model.state_dict())

    train_client(
        client_id=0,
        global_model=global_model,
        dataset=small_dataset,
        batch_size=8,
    )

    for name, tensor in global_model.state_dict().items():
        assert torch.equal(tensor, before[name])


def test_update_shapes(small_dataset):
    set_seed(42)
    model = MNISTCNN()

    result = train_client(
        client_id=1,
        global_model=model,
        dataset=small_dataset,
        batch_size=8,
    )

    for name, parameter in model.state_dict().items():
        assert result.update[name].shape == parameter.shape


def test_update_nonzero(small_dataset):
    set_seed(42)
    model = MNISTCNN()

    result = train_client(
        client_id=2,
        global_model=model,
        dataset=small_dataset,
        batch_size=8,
    )

    total_magnitude = sum(
        tensor.abs().sum().item()
        for tensor in result.update.values()
    )

    assert total_magnitude > 0


def test_client_metadata(small_dataset):
    set_seed(42)
    model = MNISTCNN()

    result = train_client(
        client_id=3,
        global_model=model,
        dataset=small_dataset,
        batch_size=8,
    )

    assert result.client_id == 3
    assert result.num_samples == 16
    assert result.mean_loss > 0


def test_invalid_dataset():
    model = MNISTCNN()

    empty = TensorDataset(
        torch.empty(0, 1, 28, 28),
        torch.empty(0, dtype=torch.long),
    )

    with pytest.raises(ValueError):
        train_client(
            client_id=0,
            global_model=model,
            dataset=empty,
        )
