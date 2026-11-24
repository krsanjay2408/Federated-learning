
from dataclasses import dataclass
from typing import Callable

import numpy as np
from torch import nn
from torch.utils.data import Dataset, Subset

from src.fedresearch.fl.aggregation import (
    fedavg_aggregate,
    apply_global_state,
)
from src.fedresearch.fl.client import train_client
from src.fedresearch.fl.evaluation import evaluate_model


@dataclass
class RoundResult:
    round_number: int
    selected_clients: list[int]
    mean_client_loss: float
    validation_loss: float
    validation_accuracy: float


def select_clients(
    num_clients: int,
    clients_per_round: int,
    seed: int,
    round_number: int,
) -> list[int]:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if not 1 <= clients_per_round <= num_clients:
        raise ValueError("Invalid clients_per_round")

    rng = np.random.default_rng(seed + round_number)

    return sorted(
        rng.choice(
            num_clients,
            size=clients_per_round,
            replace=False,
        ).tolist()
    )


def run_federated_training(
    model: nn.Module,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    partitions: dict[int, list[int]],
    rounds: int = 3,
    clients_per_round: int = 4,
    local_epochs: int = 1,
    batch_size: int = 64,
    lr: float = 0.01,
    seed: int = 42,
    device: str = "cpu",
    on_round_end: Callable | None = None,
) -> list[RoundResult]:
    """Run synchronous FedAvg training."""
    if rounds <= 0:
        raise ValueError("rounds must be positive")

    num_clients = len(partitions)

    if set(partitions) != set(range(num_clients)):
        raise ValueError(
            "Client IDs must be consecutive from 0"
        )

    history = []

    for round_number in range(1, rounds + 1):
        selected = select_clients(
            num_clients,
            clients_per_round,
            seed,
            round_number,
        )

        client_results = []

        for client_id in selected:
            client_dataset = Subset(
                train_dataset,
                partitions[client_id],
            )

            result = train_client(
                client_id=client_id,
                global_model=model,
                dataset=client_dataset,
                epochs=local_epochs,
                batch_size=batch_size,
                lr=lr,
                seed=seed + round_number * 1000,
                device=device,
            )

            client_results.append(result)

        new_state = fedavg_aggregate(
            model,
            client_results,
        )
        apply_global_state(model, new_state)

        evaluation = evaluate_model(
            model,
            validation_dataset,
            device=device,
        )

        mean_client_loss = sum(
            r.mean_loss * r.num_samples
            for r in client_results
        ) / sum(
            r.num_samples for r in client_results
        )

        round_result = RoundResult(
            round_number=round_number,
            selected_clients=selected,
            mean_client_loss=mean_client_loss,
            validation_loss=evaluation.loss,
            validation_accuracy=evaluation.accuracy,
        )

        history.append(round_result)

        if on_round_end is not None:
            on_round_end(round_result)

    return history
