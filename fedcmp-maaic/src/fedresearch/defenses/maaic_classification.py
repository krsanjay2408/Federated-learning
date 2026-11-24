"""Experimental minority-aware classification for MAAIC.

MAAIC starts with FedCmp's classification and reviews clients
that FedCmp rejected. A rejected client is rescued only when
its update direction and magnitude pass consistency checks.

No ground-truth client roles are used in the decision.
"""

from dataclasses import dataclass
from collections.abc import Mapping

import torch

from src.fedresearch.defenses.fedcmp_classification import (
    ClassificationResult,
)
from src.fedresearch.defenses.maaic_consistency import (
    ConsistencyResult,
    check_update_consistency,
)


@dataclass
class MAAICClassificationResult:
    client_ids: list[int]
    retained_client_ids: list[int]
    rejected_client_ids: list[int]
    rescued_client_ids: list[int]
    initially_retained_client_ids: list[int]
    initially_rejected_client_ids: list[int]
    consistency: ConsistencyResult


def classify_clients_maaic(
    initial_classification: ClassificationResult,
    client_updates: Mapping[int, Mapping[str, torch.Tensor]],
    direction_threshold: float = 0.0,
    max_norm_ratio: float = 3.0,
) -> MAAICClassificationResult:
    """Review initially rejected clients using update consistency."""
    ids = list(initial_classification.client_ids)

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate client IDs")

    if set(client_updates) != set(ids):
        raise ValueError(
            "Client updates must match classification client IDs"
        )

    initial_retained = list(
        initial_classification.retained_client_ids
    )
    initial_rejected = list(
        initial_classification.rejected_client_ids
    )

    retained_set = set(initial_retained)
    rejected_set = set(initial_rejected)

    if retained_set & rejected_set:
        raise ValueError(
            "Initially retained and rejected sets overlap"
        )

    if retained_set | rejected_set != set(ids):
        raise ValueError(
            "Initial classification does not cover all clients"
        )

    if not initial_retained:
        raise ValueError(
            "At least one initially retained client is required"
        )

    consistency = check_update_consistency(
        client_updates=client_updates,
        reference_client_ids=initial_retained,
        direction_threshold=direction_threshold,
        max_norm_ratio=max_norm_ratio,
    )

    rescued = [
        client_id
        for client_id in initial_rejected
        if consistency.direction_consistent[client_id]
        and consistency.magnitude_consistent[client_id]
    ]

    final_retained_set = retained_set | set(rescued)

    final_retained = [
        client_id
        for client_id in ids
        if client_id in final_retained_set
    ]

    final_rejected = [
        client_id
        for client_id in ids
        if client_id not in final_retained_set
    ]

    return MAAICClassificationResult(
        client_ids=ids,
        retained_client_ids=final_retained,
        rejected_client_ids=final_rejected,
        rescued_client_ids=rescued,
        initially_retained_client_ids=initial_retained,
        initially_rejected_client_ids=initial_rejected,
        consistency=consistency,
    )
