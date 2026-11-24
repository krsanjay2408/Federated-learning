
from dataclasses import dataclass

import numpy as np

from src.fedresearch.data.partition import validate_partition


@dataclass
class MinorityPartition:
    partitions: dict[int, list[int]]
    roles: dict[int, str]
    counts: np.ndarray


def minority_partition(
    labels,
    seed: int = 42,
    preference: float = 8.0,
) -> MinorityPartition:
    """
    Controlled 20-client MNIST partition.

    Clients 0-13: honest majority, prefer 0-3
    Clients 14-17: honest minority, prefer 7-9
    Clients 18-19: Byzantine-designated, no preference

    Every client receives exactly the same number of samples.
    All input positions are assigned once.
    """
    labels = np.asarray(labels, dtype=np.int64)

    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if len(labels) == 0 or len(labels) % 20 != 0:
        raise ValueError("Sample count must be positive and divisible by 20")
    if np.any((labels < 0) | (labels > 9)):
        raise ValueError("Labels must be MNIST digits 0-9")
    if not np.isfinite(preference) or preference <= 0:
        raise ValueError("preference must be positive and finite")

    rng = np.random.default_rng(seed)
    num_clients = 20
    samples_per_client = len(labels) // num_clients

    roles = {
        i: (
            "honest_majority" if i < 14
            else "honest_minority" if i < 18
            else "byzantine"
        )
        for i in range(num_clients)
    }

    weights = np.ones((num_clients, 10), dtype=float)
    weights[:14, :4] = preference
    weights[14:18, 7:10] = preference

    # Build one shuffled pool of positions for each digit.
    pools = []
    for digit in range(10):
        positions = np.flatnonzero(labels == digit)
        rng.shuffle(positions)
        pools.append(positions.tolist())

    remaining = np.array(
        [len(pool) for pool in pools],
        dtype=np.int64,
    )

    partitions = {i: [] for i in range(num_clients)}

    # Round-robin allocation avoids permanently prioritizing
    # the clients with the lowest IDs.
    for _ in range(samples_per_client):
        client_order = rng.permutation(num_clients)

        for client_id in client_order:
            probabilities = weights[client_id] * remaining
            probabilities /= probabilities.sum()

            digit = int(
                rng.choice(10, p=probabilities)
            )

            position = pools[digit].pop()
            partitions[client_id].append(position)
            remaining[digit] -= 1

    validate_partition(partitions, len(labels))

    counts = np.zeros((num_clients, 10), dtype=np.int64)

    for client_id, positions in partitions.items():
        counts[client_id] = np.bincount(
            labels[positions],
            minlength=10,
        )

    assert np.all(counts.sum(axis=1) == samples_per_client)
    assert np.array_equal(
        counts.sum(axis=0),
        np.bincount(labels, minlength=10),
    )

    return MinorityPartition(
        partitions=partitions,
        roles=roles,
        counts=counts,
    )
