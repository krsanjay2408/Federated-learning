
from typing import Dict

import numpy as np


def iid_partition(
    num_samples: int,
    num_clients: int,
    seed: int = 42,
) -> Dict[int, list[int]]:
    """Split sample positions into disjoint, nearly equal groups."""

    if num_samples < num_clients or num_clients <= 0:
        raise ValueError("Need at least one sample per client")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)

    splits = np.array_split(indices, num_clients)

    return {
        client_id: split.tolist()
        for client_id, split in enumerate(splits)
    }


def dirichlet_partition(
    labels,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
    min_samples: int = 10,
    max_attempts: int = 1000,
) -> Dict[int, list[int]]:
    """Partition sample positions by class-wise Dirichlet sampling."""

    labels = np.asarray(labels)

    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if num_clients <= 0 or alpha <= 0:
        raise ValueError("num_clients and alpha must be positive")
    if min_samples < 0:
        raise ValueError("min_samples must be nonnegative")
    if len(labels) < num_clients * min_samples:
        raise ValueError("Not enough samples for min_samples")

    rng = np.random.default_rng(seed)
    classes = np.unique(labels)

    for _ in range(max_attempts):
        partitions = {
            i: [] for i in range(num_clients)
        }

        for cls in classes:
            cls_indices = np.flatnonzero(labels == cls)
            rng.shuffle(cls_indices)

            proportions = rng.dirichlet(
                np.full(num_clients, alpha)
            )

            counts = rng.multinomial(
                len(cls_indices), proportions
            )

            start = 0

            for client_id, count in enumerate(counts):
                end = start + count

                partitions[client_id].extend(
                    cls_indices[start:end].tolist()
                )

                start = end

        if min(map(len, partitions.values())) >= min_samples:
            for indices in partitions.values():
                rng.shuffle(indices)

            return partitions

    raise RuntimeError(
        "Could not satisfy min_samples; "
        "increase alpha, reduce min_samples, "
        "or increase max_attempts."
    )


def validate_partition(
    partitions: Dict[int, list[int]],
    num_samples: int,
) -> None:
    """Assert that every position is assigned exactly once."""

    all_indices = [
        index
        for indices in partitions.values()
        for index in indices
    ]

    assert len(all_indices) == num_samples
    assert len(set(all_indices)) == num_samples
    assert set(all_indices) == set(range(num_samples))
