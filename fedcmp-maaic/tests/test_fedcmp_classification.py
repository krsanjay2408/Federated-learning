
import numpy as np
import pytest

from src.fedresearch.defenses.fedcmp_voting import (
    VotingResult,
)
from src.fedresearch.defenses.fedcmp_classification import (
    classify_clients,
)


def make_voting_result(votes, k=40):
    votes = np.asarray(votes, dtype=np.int64)

    if np.any(votes < 0) or np.any(votes > k):
        raise ValueError("Invalid synthetic vote count")

    matrix = np.zeros(
        (len(votes), k),
        dtype=np.int64,
    )

    for i, count in enumerate(votes):
        matrix[i, :count] = 1

    return VotingResult(
        client_ids=list(range(len(votes))),
        vote_counts=votes,
        vote_matrix=matrix,
        column_results=[],
    )


def test_larger_zero_vote_group_retained():
    voting = make_voting_result(
        [40] * 7 + [0] * 13
    )

    result = classify_clients(voting)

    assert result.retained_client_ids == list(range(7, 20))
    assert result.rejected_client_ids == list(range(7))
    assert len(result.retained_client_ids) == 13


def test_larger_high_vote_group_retained():
    voting = make_voting_result(
        [0] * 6 + [40] * 14
    )

    result = classify_clients(voting)

    assert result.retained_client_ids == list(range(6, 20))
    assert result.rejected_client_ids == list(range(6))


def test_equal_size_selects_higher_vote_group():
    voting = make_voting_result(
        [0] * 10 + [40] * 10
    )

    result = classify_clients(voting)

    assert result.cluster_sizes.tolist() == [10, 10]
    assert result.selected_cluster == 1
    assert result.retained_client_ids == list(range(10, 20))


def test_identical_votes_retains_everyone():
    voting = make_voting_result([20] * 20)

    result = classify_clients(voting)

    assert result.used_fallback
    assert result.retained_client_ids == list(range(20))
    assert result.rejected_client_ids == []
    assert result.benign_mask.all()


def test_reproducibility():
    voting = make_voting_result(
        [2] * 6 + [18] * 8 + [39] * 6
    )

    first = classify_clients(voting, seed=42)
    second = classify_clients(voting, seed=42)

    np.testing.assert_array_equal(
        first.cluster_labels,
        second.cluster_labels,
    )
    assert (
        first.retained_client_ids
        == second.retained_client_ids
    )


def test_masks_partition_all_clients():
    voting = make_voting_result(
        [0] * 7 + [40] * 13
    )

    result = classify_clients(voting)

    retained = set(result.retained_client_ids)
    rejected = set(result.rejected_client_ids)

    assert retained.isdisjoint(rejected)
    assert retained | rejected == set(range(20))
    assert result.benign_mask.sum() == len(retained)


def test_invalid_negative_votes():
    voting = make_voting_result([0, 10, 20])
    voting.vote_counts[0] = -1

    with pytest.raises(ValueError):
        classify_clients(voting)


def test_mismatched_vote_counts():
    voting = make_voting_result([0, 10, 20])
    voting.vote_counts[1] = 11

    with pytest.raises(ValueError):
        classify_clients(voting)


def test_duplicate_client_ids():
    voting = make_voting_result([0, 10, 20])
    voting.client_ids = [0, 0, 2]

    with pytest.raises(ValueError):
        classify_clients(voting)


def test_single_client_rejected():
    voting = make_voting_result([10])

    with pytest.raises(ValueError):
        classify_clients(voting)


def test_invalid_n_init():
    voting = make_voting_result([0, 10, 20])

    with pytest.raises(ValueError):
        classify_clients(voting, n_init=0)


def test_nonbinary_vote_matrix():
    voting = make_voting_result([0, 10, 20])
    voting.vote_matrix[1, 0] = 2

    with pytest.raises(ValueError):
        classify_clients(voting)
