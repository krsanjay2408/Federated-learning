
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from src.fedresearch.fl.client import set_seed
from src.fedresearch.fl.trainer import (
    select_clients,
    run_federated_training,
)


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def toy_data():
    generator = torch.Generator().manual_seed(42)

    x = torch.randn(40, 4, generator=generator)
    y = (x[:, 0] > 0).long()

    dataset = TensorDataset(x, y)

    partitions = {
        0: list(range(0, 10)),
        1: list(range(10, 20)),
        2: list(range(20, 30)),
        3: list(range(30, 40)),
    }

    return dataset, partitions


def test_selection_reproducible():
    first = select_clients(20, 4, 42, 1)
    second = select_clients(20, 4, 42, 1)

    assert first == second
    assert len(set(first)) == 4


def test_selection_invalid():
    with pytest.raises(ValueError):
        select_clients(4, 5, 42, 1)


def test_training_loop_runs(toy_data):
    dataset, partitions = toy_data
    set_seed(42)

    model = TinyClassifier()

    history = run_federated_training(
        model=model,
        train_dataset=dataset,
        validation_dataset=dataset,
        partitions=partitions,
        rounds=2,
        clients_per_round=2,
        batch_size=5,
        lr=0.1,
    )

    assert len(history) == 2

    for result in history:
        assert 0 <= result.validation_accuracy <= 1
        assert result.validation_loss >= 0
        assert len(result.selected_clients) == 2


def test_invalid_rounds(toy_data):
    dataset, partitions = toy_data

    with pytest.raises(ValueError):
        run_federated_training(
            model=TinyClassifier(),
            train_dataset=dataset,
            validation_dataset=dataset,
            partitions=partitions,
            rounds=0,
        )
