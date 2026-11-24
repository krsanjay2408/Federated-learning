"""FedCmp federated training with optional Byzantine attacks."""

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.utils.data import Dataset, Subset

from src.fedresearch.attacks.sign_flip import sign_flip_update
from src.fedresearch.fl.client import train_client
from src.fedresearch.fl.aggregation import (
    fedavg_aggregate,
    apply_global_state,
)
from src.fedresearch.fl.evaluation import evaluate_model
from src.fedresearch.fl.trainer import select_clients

from src.fedresearch.defenses.fedcmp_sampling import (
    sample_parameter_matrix,
)
from src.fedresearch.defenses.fedcmp_voting import (
    cluster_and_vote,
)
from src.fedresearch.defenses.fedcmp_classification import (
    classify_clients,
)
from src.fedresearch.defenses.maaic_classification import (
    classify_clients_maaic,
)


@dataclass
class FedCmpRoundResult:
    round_number: int
    selected_clients: list[int]
    retained_clients: list[int]
    rejected_clients: list[int]
    vote_counts: dict[int, int]
    client_sample_counts: dict[int, int]
    mean_client_loss: float
    validation_loss: float
    validation_accuracy: float
    minority_frr: float | None
    majority_frr: float | None
    overall_frr: float
    used_fallback: bool

    constant_columns: int = 0
    two_value_columns: int = 0
    normal_columns: int = 0

    # Attack-aware diagnostics.
    actual_malicious_clients: list[int] | None = None
    attack_detection_rate: float | None = None
    malicious_retained: int = 0
    malicious_rejected: int = 0


def false_rejection_rate(
    selected_clients: list[int],
    rejected_clients: list[int],
    roles: dict[int, str],
    target_role: str | None = None,
    actual_malicious_clients: list[int] | None = None,
) -> float | None:
    """
    Fraction of eligible honest participating clients rejected.

    With target_role=None, all non-malicious clients are eligible.
    With a target role, only honest clients of that role are eligible.

    Return None when the eligible group is empty.
    """
    malicious = set(actual_malicious_clients or [])

    if target_role is None:
        eligible = [
            client_id
            for client_id in selected_clients
            if client_id not in malicious
        ]
    else:
        eligible = [
            client_id
            for client_id in selected_clients
            if client_id not in malicious
            and roles[client_id] == target_role
        ]

    if not eligible:
        return None

    rejected = set(rejected_clients)

    return sum(
        client_id in rejected
        for client_id in eligible
    ) / len(eligible)


def attack_detection_rate(
    actual_malicious_clients: list[int],
    rejected_clients: list[int],
) -> float | None:
    """
    Fraction of participating malicious clients rejected.

    Return None when no malicious clients participated.
    """
    if not actual_malicious_clients:
        return None

    rejected = set(rejected_clients)

    return sum(
        client_id in rejected
        for client_id in actual_malicious_clients
    ) / len(actual_malicious_clients)


def run_fedcmp_training(
    model: nn.Module,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    partitions: dict[int, list[int]],
    roles: dict[int, str],
    rounds: int = 1,
    clients_per_round: int = 20,
    local_epochs: int = 1,
    batch_size: int = 64,
    lr: float = 0.01,
    seed: int = 42,
    device: str = "cpu",
    parameter_name: str = "classifier.3.weight",
    k: int = 40,
    gamma: int = 3,
    representation: str = "local_weight",
    degenerate_policy: str = "strict",
    on_round_end: Callable | None = None,
    attack_mode: str = "none",
    attack_scale: float = 1.0,
    defense_mode: str = "fedcmp",
    maaic_direction_threshold: float = 0.0,
    maaic_max_norm_ratio: float = 3.0,
) -> list[FedCmpRoundResult]:
    """
    Run FedCmp training with optional sign-flipping attacks.

    Attacks are applied to designated Byzantine clients after
    local training and before FedCmp sampling and aggregation.

    The adaptive degenerate-column policy is a reproduction
    extension, not a verified original-paper rule.
    """

    if rounds < 1:
        raise ValueError("rounds must be positive")

    num_clients = len(partitions)

    if set(partitions) != set(range(num_clients)):
        raise ValueError(
            "Client IDs must be consecutive from zero"
        )

    if set(roles) != set(partitions):
        raise ValueError(
            "Roles and partitions must contain the same client IDs"
        )

    if not 3 <= clients_per_round <= num_clients:
        raise ValueError(
            "clients_per_round must be between 3 and N"
        )

    if not 2 <= gamma <= clients_per_round:
        raise ValueError(
            "gamma must be between 2 and clients_per_round"
        )

    if local_epochs < 1:
        raise ValueError("local_epochs must be positive")

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    if lr <= 0:
        raise ValueError("lr must be positive")

    if k < 1:
        raise ValueError("k must be positive")

    if representation not in {"delta", "local_weight"}:
        raise ValueError("Invalid representation")

    if degenerate_policy not in {"strict", "adaptive"}:
        raise ValueError("Invalid degenerate_policy")

    if attack_mode not in {"none", "sign_flip"}:
        raise ValueError("Invalid attack_mode")

    if defense_mode not in {"fedcmp", "maaic"}:
        raise ValueError("Invalid defense_mode")

    if not 0 < attack_scale < float("inf"):
        raise ValueError("attack_scale must be finite and positive")

    global_state = model.state_dict()

    if parameter_name not in global_state:
        raise KeyError(
            f"Model has no parameter: {parameter_name}"
        )

    parameter_count = global_state[parameter_name].numel()

    if k > parameter_count:
        raise ValueError(
            f"k={k} exceeds parameter count={parameter_count}"
        )

    history = []

    for round_number in range(1, rounds + 1):

        # 1. Select participating clients.
        selected = select_clients(
            num_clients=num_clients,
            clients_per_round=clients_per_round,
            seed=seed,
            round_number=round_number,
        )

        # 2. Train clients, then optionally transform their updates.
        client_results = []
        actual_malicious_clients = []

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

            if (
                attack_mode == "sign_flip"
                and roles[client_id] == "byzantine"
            ):
                result.update = sign_flip_update(
                    result.update,
                    scale=attack_scale,
                )
                actual_malicious_clients.append(client_id)

            client_results.append(result)

        updates = {
            result.client_id: result.update
            for result in client_results
        }

        # 3. Snapshot current global parameters.
        current_global_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }

        # 4. Sample aligned parameter coordinates.
        parameter_matrix = sample_parameter_matrix(
            client_updates=updates,
            parameter_name=parameter_name,
            k=k,
            seed=seed + round_number,
            client_ids=selected,
            global_state=current_global_state,
            representation=representation,
        )

        # 5. Cluster parameter values and calculate votes.
        voting = cluster_and_vote(
            parameter_matrix,
            gamma=gamma,
            seed=seed + round_number,
            degenerate_policy=degenerate_policy,
        )

        constant_columns = voting.constant_columns
        two_value_columns = voting.two_value_columns

        normal_columns = (
            len(voting.column_results)
            - constant_columns
            - two_value_columns
        )

        print(
            f"Round {round_number}: "
            f"constant columns={constant_columns}, "
            f"two-value columns={two_value_columns}, "
            f"normal columns={normal_columns}",
            flush=True,
        )

        # 6. FedCmp classification, optionally followed by MAAIC.
        initial_classification = classify_clients(
            voting,
            seed=seed + round_number,
        )

        if defense_mode == "maaic":
            classification = classify_clients_maaic(
                initial_classification=initial_classification,
                client_updates=updates,
                direction_threshold=maaic_direction_threshold,
                max_norm_ratio=maaic_max_norm_ratio,
            )
        else:
            classification = initial_classification

        retained_ids = set(
            classification.retained_client_ids
        )

        retained_results = [
            result
            for result in client_results
            if result.client_id in retained_ids
        ]

        if not retained_results:
            raise RuntimeError(
                "FedCmp rejected every participating client"
            )

        # 7. Aggregate retained updates.
        new_state = fedavg_aggregate(
            model,
            retained_results,
        )

        apply_global_state(
            model,
            new_state,
        )

        # 8. Evaluate updated global model.
        evaluation = evaluate_model(
            model,
            validation_dataset,
            device=device,
        )

        total_samples = sum(
            result.num_samples
            for result in client_results
        )

        mean_client_loss = sum(
            result.mean_loss * result.num_samples
            for result in client_results
        ) / total_samples

        # 9. Record votes and sample counts.
        vote_counts = {
            client_id: int(votes)
            for client_id, votes in zip(
                voting.client_ids,
                voting.vote_counts,
            )
        }

        client_sample_counts = {
            result.client_id: result.num_samples
            for result in client_results
        }

        # 10. Calculate attack-aware metrics.
        rejected_clients = classification.rejected_client_ids

        minority_frr = false_rejection_rate(
            selected_clients=selected,
            rejected_clients=rejected_clients,
            roles=roles,
            target_role="honest_minority",
            actual_malicious_clients=actual_malicious_clients,
        )

        majority_frr = false_rejection_rate(
            selected_clients=selected,
            rejected_clients=rejected_clients,
            roles=roles,
            target_role="honest_majority",
            actual_malicious_clients=actual_malicious_clients,
        )

        overall_frr_value = false_rejection_rate(
            selected_clients=selected,
            rejected_clients=rejected_clients,
            roles=roles,
            actual_malicious_clients=actual_malicious_clients,
        )

        detection_rate = attack_detection_rate(
            actual_malicious_clients=actual_malicious_clients,
            rejected_clients=rejected_clients,
        )

        rejected_set = set(rejected_clients)

        malicious_rejected = sum(
            client_id in rejected_set
            for client_id in actual_malicious_clients
        )

        malicious_retained = (
            len(actual_malicious_clients) - malicious_rejected
        )

        # 11. Save round results.
        round_result = FedCmpRoundResult(
            round_number=round_number,
            selected_clients=selected,
            retained_clients=classification.retained_client_ids,
            rejected_clients=rejected_clients,
            vote_counts=vote_counts,
            client_sample_counts=client_sample_counts,
            mean_client_loss=mean_client_loss,
            validation_loss=evaluation.loss,
            validation_accuracy=evaluation.accuracy,
            minority_frr=minority_frr,
            majority_frr=majority_frr,
            overall_frr=overall_frr_value,
            used_fallback=initial_classification.used_fallback,
            constant_columns=constant_columns,
            two_value_columns=two_value_columns,
            normal_columns=normal_columns,
            actual_malicious_clients=actual_malicious_clients,
            attack_detection_rate=detection_rate,
            malicious_retained=malicious_retained,
            malicious_rejected=malicious_rejected,
        )

        history.append(round_result)

        if on_round_end is not None:
            on_round_end(round_result)

    return history
