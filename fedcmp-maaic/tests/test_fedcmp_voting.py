
import numpy as np
import pytest

from src.fedresearch.defenses.fedcmp_sampling import (
    ParameterMatrix,
)
from src.fedresearch.defenses.fedcmp_voting import (
    cluster_and_vote,
    merge_ordered_clusters,
)


def make_matrix(n=20, k=40):
    rng = np.random.default_rng(42)

    # Three separated synthetic parameter groups.
    values = np.concatenate(
        [
            rng.normal(-5.0, 0.1, size=(7, k)),
            rng.normal(0.0, 0.1, size=(7, k)),
            rng.normal(5.0, 0.1, size=(n - 14, k)),
        ],
        axis=0,
    )

    return ParameterMatrix(
        client_ids=list(range(n)),
        parameter_name="synthetic.weight",
        coordinate_indices=np.arange(k),
        values=values,
    )


def test_merge_balanced_boundary():
    # Cluster sizes: 4, 9, 7
    labels = np.array(
        [0] * 4 + [1] * 9 + [2] * 7
    )
    centers = np.array([-5.0, 0.0, 5.0])

    mask, boundary, _, sizes = merge_ordered_clusters(
        labels, centers
    )

    assert boundary == 2
    assert mask.sum() == 13
    np.testing.assert_array_equal(sizes, [4, 9, 7])


def test_merge_sorts_centroids():
    # Cluster IDs are deliberately not in centroid order.
    labels = np.array(
        [0] * 7 + [1] * 4 + [2] * 9
    )
    centers = np.array([5.0, -5.0, 0.0])

    mask, boundary, sorted_centers, sizes = (
        merge_ordered_clusters(labels, centers)
    )

    np.testing.assert_array_equal(
        sorted_centers, [-5.0, 0.0, 5.0]
    )
    np.testing.assert_array_equal(sizes, [4, 9, 7])

    assert boundary == 2
    assert mask.sum() == 13


def test_merge_tie_uses_first_boundary():
    # Sizes 5, 10, 5:
    # boundary 1 -> 5 vs 15
    # boundary 2 -> 15 vs 5
    labels = np.array(
        [0] * 5 + [1] * 10 + [2] * 5
    )
    centers = np.array([-5.0, 0.0, 5.0])

    mask, boundary, _, _ = merge_ordered_clusters(
        labels, centers
    )

    assert boundary == 1
    assert mask.sum() == 5


def test_vote_matrix_shape():
    result = cluster_and_vote(make_matrix())

    assert result.vote_matrix.shape == (20, 40)
    assert result.vote_counts.shape == (20,)
    assert len(result.column_results) == 40


def test_votes_are_binary():
    result = cluster_and_vote(make_matrix())

    assert np.isin(
        result.vote_matrix, [0, 1]
    ).all()


def test_vote_counts_match_matrix():
    result = cluster_and_vote(make_matrix())

    np.testing.assert_array_equal(
        result.vote_counts,
        result.vote_matrix.sum(axis=1),
    )


def test_vote_counts_in_range():
    result = cluster_and_vote(make_matrix())

    assert np.all(result.vote_counts >= 0)
    assert np.all(result.vote_counts <= 40)


def test_voting_reproducible():
    matrix = make_matrix()

    first = cluster_and_vote(matrix, seed=42)
    second = cluster_and_vote(matrix, seed=42)

    np.testing.assert_array_equal(
        first.vote_matrix,
        second.vote_matrix,
    )
    np.testing.assert_array_equal(
        first.vote_counts,
        second.vote_counts,
    )



def test_lower_group_receives_votes():
    result = cluster_and_vote(make_matrix())

    # Three well-separated groups have sizes 7, 7, 6.
    #
    # Boundary 1: 7 vs 13 -> difference 6
    # Boundary 2: 14 vs 6 -> difference 8
    #
    # Boundary 1 is more balanced, so only the
    # lowest-centroid cluster receives votes.

    assert np.all(result.vote_counts[:7] == 40)
    assert np.all(result.vote_counts[7:] == 0)



def test_invalid_gamma():
    with pytest.raises(ValueError):
        cluster_and_vote(make_matrix(), gamma=1)

    with pytest.raises(ValueError):
        cluster_and_vote(make_matrix(), gamma=21)


def test_nan_rejected():
    matrix = make_matrix()
    matrix.values[0, 0] = np.nan

    with pytest.raises(ValueError):
        cluster_and_vote(matrix)


def test_insufficient_distinct_values():
    matrix = make_matrix()
    matrix.values[:, 0] = 1.0

    with pytest.raises(ValueError):
        cluster_and_vote(matrix, gamma=3)
