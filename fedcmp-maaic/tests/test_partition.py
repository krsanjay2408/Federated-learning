
import numpy as np

from src.fedresearch.data.partition import (
    iid_partition,
    dirichlet_partition,
    validate_partition,
)


def test_iid_partition():
    partitions = iid_partition(
        num_samples=54000,
        num_clients=20,
        seed=42,
    )

    validate_partition(partitions, 54000)

    sizes = [len(p) for p in partitions.values()]

    assert max(sizes) - min(sizes) <= 1


def test_dirichlet_partition():
    labels = np.repeat(np.arange(10), 1000)

    partitions = dirichlet_partition(
        labels=labels,
        num_clients=20,
        alpha=0.5,
        seed=42,
    )

    validate_partition(partitions, len(labels))

    assert min(map(len, partitions.values())) >= 10


def test_reproducibility():
    labels = np.repeat(np.arange(10), 100)

    first = dirichlet_partition(
        labels, num_clients=10, alpha=0.5, seed=42
    )

    second = dirichlet_partition(
        labels, num_clients=10, alpha=0.5, seed=42
    )

    assert first == second


def test_invalid_alpha():
    labels = np.arange(100) % 10

    try:
        dirichlet_partition(
            labels, num_clients=10, alpha=0
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
