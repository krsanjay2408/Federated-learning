
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from src.fedresearch.fl.client import ClientResult
import src.fedresearch.fl.fedcmp_trainer as trainer


def test_false_rejection_rates():
    selected = [0, 1, 2, 3, 4, 5]
    rejected = [1, 3, 5]

    roles = {
        0: "honest_majority",
        1: "honest_majority",
        2: "honest_majority",
        3: "honest_majority",
        4: "honest_minority",
        5: "honest_minority",
    }

    assert trainer.false_rejection_rate(
        selected, rejected, roles, "honest_majority"
    ) == 0.5

    assert trainer.false_rejection_rate(
        selected, rejected, roles, "honest_minority"
    ) == 0.5

    assert trainer.false_rejection_rate(
        selected, rejected, roles
    ) == 0.5


def test_absent_role_returns_none():
    roles = {
        0: "honest_majority",
        1: "honest_majority",
    }

    assert trainer.false_rejection_rate(
        [0, 1],
        [0],
        roles,
        "honest_minority",
    ) is None


def test_invalid_rounds():
    model = nn.Linear(40, 2)

    dataset = TensorDataset(
        torch.zeros(6, 40),
        torch.zeros(6, dtype=torch.long),
    )

    partitions = {
        i: [i] for i in range(6)
    }

    roles = {
        i: "honest_majority" for i in range(6)
    }

    with pytest.raises(ValueError):
        trainer.run_fedcmp_training(
            model=model,
            train_dataset=dataset,
            validation_dataset=dataset,
            partitions=partitions,
            roles=roles,
            rounds=0,
            clients_per_round=6,
            parameter_name="weight",
        )


def test_invalid_participation():
    model = nn.Linear(40, 2)

    dataset = TensorDataset(
        torch.zeros(6, 40),
        torch.zeros(6, dtype=torch.long),
    )

    partitions = {
        i: [i] for i in range(6)
    }

    roles = {
        i: "honest_majority" for i in range(6)
    }

    with pytest.raises(ValueError):
        trainer.run_fedcmp_training(
            model=model,
            train_dataset=dataset,
            validation_dataset=dataset,
            partitions=partitions,
            roles=roles,
            clients_per_round=2,
            parameter_name="weight",
        )


def test_fedcmp_training_integration(monkeypatch):
    model = nn.Linear(40, 2, bias=False)

    with torch.no_grad():
        model.weight.zero_()

    dataset = TensorDataset(
        torch.zeros(6, 40),
        torch.zeros(6, dtype=torch.long),
    )

    partitions = {
        i: [i] for i in range(6)
    }

    roles = {
        i: (
            "honest_minority" if i >= 4
            else "honest_majority"
        )
        for i in range(6)
    }

    # Distinct, separated synthetic client deltas.
    values = [-5.0, -5.0, 0.0, 0.0, 5.0, 5.0]

    def fake_train_client(
        client_id,
        global_model,
        dataset,
        **kwargs,
    ):
        delta = torch.full_like(
            global_model.weight,
            values[client_id],
        )

        return ClientResult(
            client_id=client_id,
            num_samples=len(dataset),
            mean_loss=1.0,
            update={"weight": delta},
        )

    def fake_evaluate_model(model, dataset, device):
        return SimpleNamespace(
            loss=0.8,
            accuracy=0.6,
        )

    monkeypatch.setattr(
        trainer,
        "train_client",
        fake_train_client,
    )

    monkeypatch.setattr(
        trainer,
        "evaluate_model",
        fake_evaluate_model,
    )

    history = trainer.run_fedcmp_training(
        model=model,
        train_dataset=dataset,
        validation_dataset=dataset,
        partitions=partitions,
        roles=roles,
        rounds=1,
        clients_per_round=6,
        parameter_name="weight",
        k=40,
        gamma=3,
        representation="delta",
    )

    assert len(history) == 1

    result = history[0]

    assert result.selected_clients == list(range(6))

    retained = set(result.retained_clients)
    rejected = set(result.rejected_clients)

    assert retained.isdisjoint(rejected)
    assert retained | rejected == set(range(6))
    assert len(retained) > 0

    assert result.validation_accuracy == 0.6
    assert result.mean_client_loss == 1.0

    assert len(result.vote_counts) == 6
    assert all(
        0 <= count <= 40
        for count in result.vote_counts.values()
    )

    # The global model must equal the mean of the
    # retained synthetic updates, not all updates.
    expected_delta = np.mean(
        [values[i] for i in result.retained_clients]
    )

    torch.testing.assert_close(
        model.weight,
        torch.full_like(model.weight, expected_delta),
    )


def test_role_mismatch_rejected():
    model = nn.Linear(40, 2)

    dataset = TensorDataset(
        torch.zeros(6, 40),
        torch.zeros(6, dtype=torch.long),
    )

    partitions = {
        i: [i] for i in range(6)
    }

    with pytest.raises(ValueError):
        trainer.run_fedcmp_training(
            model=model,
            train_dataset=dataset,
            validation_dataset=dataset,
            partitions=partitions,
            roles={0: "honest_majority"},
            clients_per_round=6,
            parameter_name="weight",
        )
