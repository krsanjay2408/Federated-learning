"""Update consistency diagnostics for the proposed MAAIC defense.

This module uses only submitted client updates and FedCmp's
initial retained set. It never uses ground-truth client roles.

The scores are diagnostics, not proof of benign or malicious
behavior. No client is rescued or rejected by this module.
"""

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
import torch


@dataclass
class ConsistencyResult:
    client_ids: list[int]
    reference_client_ids: list[int]
    cosine_scores: dict[int, float]
    norm_ratios: dict[int, float]
    direction_consistent: dict[int, bool]
    magnitude_consistent: dict[int, bool]


def _flatten_update(
    update: Mapping[str, torch.Tensor],
    parameter_names: list[str],
) -> torch.Tensor:
    """Flatten a client's parameter deltas in a fixed order."""
    pieces = []

    for name in parameter_names:
        if name not in update:
            raise ValueError(f"Missing update parameter: {name}")

        tensor = update[name]

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Update parameter is not a tensor: {name}")

        if not torch.is_floating_point(tensor):
            raise TypeError(f"Update parameter is not floating: {name}")

        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"Non-finite update parameter: {name}")

        pieces.append(
            tensor.detach().to(
                device="cpu",
                dtype=torch.float64,
            ).reshape(-1)
        )

    return torch.cat(pieces)


def check_update_consistency(
    client_updates: Mapping[int, Mapping[str, torch.Tensor]],
    reference_client_ids: list[int],
    direction_threshold: float = 0.0,
    max_norm_ratio: float = 3.0,
) -> ConsistencyResult:
    """
    Compare client updates with the reference group's mean update.

    Direction:
        cosine(update, reference_mean) >= direction_threshold

    Magnitude:
        norm(update) / median(reference_norms) <= max_norm_ratio

    The reference group is usually FedCmp's initially retained
    clients. It may contain malicious clients, so these scores
    must not be treated as ground-truth security labels.
    """
    if not client_updates:
        raise ValueError("client_updates cannot be empty")

    if not reference_client_ids:
        raise ValueError("reference_client_ids cannot be empty")

    if len(set(reference_client_ids)) != len(reference_client_ids):
        raise ValueError("Duplicate reference client IDs")

    if not np.isfinite(direction_threshold):
        raise ValueError("direction_threshold must be finite")

    if not -1.0 <= direction_threshold <= 1.0:
        raise ValueError("direction_threshold must be in [-1, 1]")

    if not np.isfinite(max_norm_ratio) or max_norm_ratio <= 0:
        raise ValueError("max_norm_ratio must be finite and positive")

    client_ids = list(client_updates)

    if not set(reference_client_ids).issubset(client_updates):
        raise ValueError("Reference clients must have updates")

    first_update = client_updates[client_ids[0]]

    if not first_update:
        raise ValueError("Client updates cannot be empty")

    parameter_names = list(first_update)

    vectors = {}

    for client_id in client_ids:
        update = client_updates[client_id]

        if set(update) != set(parameter_names):
            raise ValueError("Client update parameter names differ")

        vectors[client_id] = _flatten_update(
            update,
            parameter_names,
        )

        if vectors[client_id].numel() == 0:
            raise ValueError("Update vectors cannot be empty")

    reference_vectors = torch.stack([
        vectors[client_id]
        for client_id in reference_client_ids
    ])

    reference_mean = reference_vectors.mean(dim=0)
    reference_mean_norm = torch.linalg.vector_norm(reference_mean)

    reference_norms = torch.linalg.vector_norm(
        reference_vectors,
        dim=1,
    )

    median_reference_norm = torch.median(reference_norms)

    cosine_scores = {}
    norm_ratios = {}
    direction_consistent = {}
    magnitude_consistent = {}

    for client_id in client_ids:
        vector = vectors[client_id]
        vector_norm = torch.linalg.vector_norm(vector)

        if vector_norm.item() == 0 or reference_mean_norm.item() == 0:
            cosine = 0.0
            direction_ok = False
        else:
            cosine = float(
                torch.dot(vector, reference_mean)
                / (vector_norm * reference_mean_norm)
            )
            cosine = max(-1.0, min(1.0, cosine))
            direction_ok = cosine >= direction_threshold

        if median_reference_norm.item() == 0:
            ratio = 0.0 if vector_norm.item() == 0 else float("inf")
        else:
            ratio = float(vector_norm / median_reference_norm)

        cosine_scores[client_id] = cosine
        norm_ratios[client_id] = ratio
        direction_consistent[client_id] = direction_ok
        magnitude_consistent[client_id] = ratio <= max_norm_ratio

    return ConsistencyResult(
        client_ids=client_ids,
        reference_client_ids=reference_client_ids.copy(),
        cosine_scores=cosine_scores,
        norm_ratios=norm_ratios,
        direction_consistent=direction_consistent,
        magnitude_consistent=magnitude_consistent,
    )
