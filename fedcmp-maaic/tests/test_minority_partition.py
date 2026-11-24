
import numpy as np
import pytest

from src.fedresearch.data.minority_partition import (
    minority_partition,
)
from src.fedresearch.data.partition import validate_partition


@pytest.fixture(scope="module")
def sample_result():
    # Balanced synthetic digits, 200 examples per digit.
    labels = np.repeat(np.arange(10), 200)
    return labels, minority_partition(
        labels,
        seed=42,
        preference=8.0,
    )


def test_complete_disjoint_assignment(sample_result):
    labels, result = sample_result

    validate_partition(result.partitions, len(labels))
    assert len(result.partitions) == 20


def test_equal_client_sizes(sample_result):
    labels, result = sample_result

    expected = len(labels) // 20
    assert all(
        len(indices) == expected
        for indices in result.partitions.values()
    )


def test_client_roles(sample_result):
    _, result = sample_result

    assert sum(
        role == "honest_majority"
        for role in result.roles.values()
    ) == 14

    assert sum(
        role == "honest_minority"
        for role in result.roles.values()
    ) == 4

    assert sum(
        role == "byzantine"
        for role in result.roles.values()
    ) == 2


def test_class_coverage(sample_result):
    labels, result = sample_result

    actual = result.counts.sum(axis=0)
    expected = np.bincount(labels, minlength=10)

    np.testing.assert_array_equal(actual, expected)


def test_reproducibility(sample_result):
    labels, first = sample_result

    second = minority_partition(
        labels,
        seed=42,
        preference=8.0,
    )

    assert first.partitions == second.partitions
    np.testing.assert_array_equal(
        first.counts,
        second.counts,
    )


def test_preference_effect(sample_result):
    _, result = sample_result

    counts = result.counts
    majority_fraction = (
        counts[:14, :4].sum() / counts[:14].sum()
    )
    minority_fraction = (
        counts[14:18, 7:10].sum()
        / counts[14:18].sum()
    )

    # On balanced synthetic labels, preferences should
    # increase both fractions above their global baselines.
    assert majority_fraction > 0.4
    assert minority_fraction > 0.3


def test_invalid_preference():
    labels = np.repeat(np.arange(10), 20)

    with pytest.raises(ValueError):
        minority_partition(labels, preference=0)
