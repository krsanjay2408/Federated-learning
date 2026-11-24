"""End-to-end integration test for the MAAIC training path."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from src.fedresearch.fl.client import ClientResult
import src.fedresearch.fl.fedcmp_trainer as trainer


def test_maaic_training_integration(monkeypatch):
    model = nn.Linear(40, 2, bias=False)

    with torch.no_grad():
        model.weight.zero_()

    dataset = TensorDataset(
        torch.zeros(6, 40),
        torch.zeros(6, dtype=torch.long),
    )

    partitions = {i: [i] for i in range(6)}

    roles = {
        i: "honest_minority" if i >= 4 else "honest_majority"
        for i in range(6)
    }

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
        return SimpleNamespace(loss=0.8, accuracy=0.6)

    monkeypatch.setattr(
        trainer, "train_client", fake_train_client
    )
    monkeypatch.setattr(
        trainer, "evaluate_model", fake_evaluate_model
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
        degenerate_policy="adaptive",
        defense_mode="maaic",
    )

    assert len(history) == 1

    result = history[0]

    retained = set(result.retained_clients)
    rejected = set(result.rejected_clients)

    assert retained.isdisjoint(rejected)
    assert retained | rejected == set(range(6))
    assert retained

    assert result.validation_accuracy == pytest.approx(0.6)
    assert result.mean_client_loss == pytest.approx(1.0)

    # Verify that aggregation uses the final MAAIC
    # retained set, not all participating clients.
    expected_delta = sum(
        values[i] for i in retained
    ) / len(retained)

    torch.testing.assert_close(
        model.weight,
        torch.full_like(model.weight, expected_delta),
    )


def test_invalid_defense_mode():
    model = nn.Linear(40, 2)

    dataset = TensorDataset(
        torch.zeros(6, 40),
        torch.zeros(6, dtype=torch.long),
    )

    partitions = {i: [i] for i in range(6)}
    roles = {i: "honest_majority" for i in range(6)}

    with pytest.raises(ValueError):
        trainer.run_fedcmp_training(
            model=model,
            train_dataset=dataset,
            validation_dataset=dataset,
            partitions=partitions,
            roles=roles,
            rounds=1,
            clients_per_round=6,
            parameter_name="weight",
            defense_mode="invalid",
        )
