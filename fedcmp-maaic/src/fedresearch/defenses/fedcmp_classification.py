
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from src.fedresearch.defenses.fedcmp_voting import (
    VotingResult,
)


@dataclass
class ClassificationResult:
    client_ids: list[int]
    vote_counts: np.ndarray
    retained_client_ids: list[int]
    rejected_client_ids: list[int]
    benign_mask: np.ndarray
    cluster_labels: np.ndarray
    cluster_centers: np.ndarray
    cluster_sizes: np.ndarray
    selected_cluster: int
    used_fallback: bool


def classify_clients(
    voting_result: VotingResult,
    seed: int = 42,
    n_init: int = 10,
) -> ClassificationResult:
    """
    Classify clients using two-cluster K-Means on vote counts.

    Clusters are relabeled by ascending vote centroid:
      cluster 0 = lower vote centroid
      cluster 1 = higher vote centroid

    Select the larger cluster as benign.
    Equal cluster sizes select cluster 1.

    If all vote counts are identical, retain all clients.
    This fallback is an explicit reproduction assumption.
    """
    ids = list(voting_result.client_ids)
    votes = np.asarray(voting_result.vote_counts)

    if votes.ndim != 1:
        raise ValueError("vote_counts must be one-dimensional")

    n = len(ids)

    if n < 2:
        raise ValueError("At least two clients are required")

    if len(set(ids)) != n:
        raise ValueError("Duplicate client IDs")

    if len(votes) != n:
        raise ValueError("Client IDs and votes must match")

    if not np.issubdtype(votes.dtype, np.number):
        raise ValueError("Votes must be numeric")

    if not np.isfinite(votes).all():
        raise ValueError("Votes contain NaN or infinity")

    if np.any(votes < 0):
        raise ValueError("Vote counts cannot be negative")

    if not np.all(votes == np.floor(votes)):
        raise ValueError("Vote counts must be integers")

    vote_matrix = np.asarray(voting_result.vote_matrix)

    if vote_matrix.ndim != 2 or vote_matrix.shape[0] != n:
        raise ValueError("Invalid vote matrix shape")

    if vote_matrix.shape[1] == 0:
        raise ValueError("Vote matrix has no columns")

    if not np.isin(vote_matrix, [0, 1]).all():
        raise ValueError("Vote matrix must be binary")

    if not np.array_equal(
        votes, vote_matrix.sum(axis=1)
    ):
        raise ValueError(
            "Vote counts do not match the vote matrix"
        )

    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be nonnegative")

    if (
        not isinstance(n_init, int)
        or isinstance(n_init, bool)
        or n_init <= 0
    ):
        raise ValueError("n_init must be a positive integer")

    # Edge case: two distinct clusters cannot be formed.
    # We retain everyone instead of fabricating a split.
    if np.unique(votes).size == 1:
        return ClassificationResult(
            client_ids=ids,
            vote_counts=votes.copy(),
            retained_client_ids=ids.copy(),
            rejected_client_ids=[],
            benign_mask=np.ones(n, dtype=bool),
            cluster_labels=np.zeros(n, dtype=np.int64),
            cluster_centers=np.array(
                [float(votes[0]), float(votes[0])]
            ),
            cluster_sizes=np.array([n, 0]),
            selected_cluster=0,
            used_fallback=True,
        )

    model = KMeans(
        n_clusters=2,
        random_state=seed,
        n_init=n_init,
    )

    raw_labels = model.fit_predict(
        votes.astype(np.float64).reshape(-1, 1)
    )

    raw_centers = model.cluster_centers_.reshape(-1)

    # K-Means cluster IDs have no inherent meaning.
    # Sort by centroid to obtain stable semantic labels.
    order = np.argsort(raw_centers, kind="stable")

    labels = np.empty(n, dtype=np.int64)

    for ordered_id, original_id in enumerate(order):
        labels[raw_labels == original_id] = ordered_id

    centers = raw_centers[order]
    sizes = np.bincount(labels, minlength=2)

    if np.any(sizes == 0):
        raise RuntimeError(
            "K-Means produced an empty cluster"
        )

    # Paper-style size rule, with our documented
    # higher-centroid tie convention.
    selected = 0 if sizes[0] > sizes[1] else 1

    benign_mask = labels == selected

    retained = [
        client_id
        for client_id, keep in zip(ids, benign_mask)
        if keep
    ]

    rejected = [
        client_id
        for client_id, keep in zip(ids, benign_mask)
        if not keep
    ]

    return ClassificationResult(
        client_ids=ids,
        vote_counts=votes.copy(),
        retained_client_ids=retained,
        rejected_client_ids=rejected,
        benign_mask=benign_mask,
        cluster_labels=labels,
        cluster_centers=centers.copy(),
        cluster_sizes=sizes,
        selected_cluster=selected,
        used_fallback=False,
    )
