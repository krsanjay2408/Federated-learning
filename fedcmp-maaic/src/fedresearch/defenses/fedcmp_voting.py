
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from src.fedresearch.defenses.fedcmp_sampling import (
    ParameterMatrix,
)



@dataclass
class ColumnClustering:
    column_index: int
    centroids: np.ndarray
    cluster_sizes: np.ndarray
    merge_boundary: int
    votes: np.ndarray
    status: str = "normal"


@dataclass
class VotingResult:
    client_ids: list[int]
    vote_counts: np.ndarray
    vote_matrix: np.ndarray
    column_results: list[ColumnClustering]
    constant_columns: int = 0
    two_value_columns: int = 0



def merge_ordered_clusters(
    labels: np.ndarray,
    centers: np.ndarray,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """
    Sort clusters by centroid and merge adjacent clusters
    into two groups with the smallest size difference.

    Returns:
        alpha_mask: clients receiving one vote
        boundary: number of low-centroid clusters in alpha
        sorted_centers
        sorted_cluster_sizes
    """
    labels = np.asarray(labels)
    centers = np.asarray(centers, dtype=np.float64).reshape(-1)

    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")

    if not np.isfinite(centers).all():
        raise ValueError("centers must be finite")

    gamma = len(centers)

    if gamma < 2:
        raise ValueError("At least two clusters required")

    if len(labels) < gamma:
        raise ValueError("Fewer clients than clusters")

    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Cluster labels must be integers")

    if np.any(labels < 0) or np.any(labels >= gamma):
        raise ValueError("Invalid cluster labels")

    sizes = np.bincount(labels, minlength=gamma)

    if np.any(sizes == 0):
        raise ValueError("Every initial cluster must be nonempty")

    # Stable ordering also defines centroid-tie behavior.
    order = np.argsort(centers, kind="stable")

    sorted_centers = centers[order]
    sorted_sizes = sizes[order]

    cumulative_sizes = np.cumsum(sorted_sizes)
    total = len(labels)

    # Each boundary leaves at least one cluster on each side.
    differences = np.abs(
        2 * cumulative_sizes[:-1] - total
    )

    # np.argmin chooses the first boundary on a tie.
    boundary = int(np.argmin(differences) + 1)

    alpha_cluster_ids = order[:boundary]

    alpha_mask = np.isin(
        labels,
        alpha_cluster_ids,
    )

    return (
        alpha_mask,
        boundary,
        sorted_centers,
        sorted_sizes,
    )



def cluster_and_vote(
    parameter_matrix: ParameterMatrix,
    gamma: int = 3,
    seed: int = 42,
    n_init: int = 10,
    degenerate_policy: str = "strict",
) -> VotingResult:
    """
    Cluster each sampled parameter column and accumulate votes.

    strict:
        Raise if a column has fewer than gamma distinct values.

    adaptive:
        1 distinct value: assign zero votes to everyone.
        2 distinct values: split by exact distinct value,
                           vote for the lower-valued group.
        >= gamma values: original gamma-cluster procedure.

    Adaptive behavior is a documented reproduction extension,
    not a confirmed rule from the original FedCmp paper.
    """
    values = np.asarray(
        parameter_matrix.values,
        dtype=np.float64,
    )

    if values.ndim != 2:
        raise ValueError("Parameter matrix must be 2D")

    n_clients, k = values.shape

    if n_clients == 0 or k == 0:
        raise ValueError("Parameter matrix cannot be empty")

    if len(parameter_matrix.client_ids) != n_clients:
        raise ValueError("Client IDs do not match matrix rows")

    if len(set(parameter_matrix.client_ids)) != n_clients:
        raise ValueError("Duplicate client IDs")

    if not np.isfinite(values).all():
        raise ValueError(
            "Parameter matrix contains NaN or infinity"
        )

    if (
        not isinstance(gamma, int)
        or isinstance(gamma, bool)
        or not 2 <= gamma <= n_clients
    ):
        raise ValueError("gamma must be between 2 and N")

    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or seed < 0
    ):
        raise ValueError("seed must be nonnegative")

    if (
        not isinstance(n_init, int)
        or isinstance(n_init, bool)
        or n_init <= 0
    ):
        raise ValueError("n_init must be a positive integer")

    if degenerate_policy not in {"strict", "adaptive"}:
        raise ValueError("Invalid degenerate_policy")

    vote_matrix = np.zeros(
        (n_clients, k),
        dtype=np.int64,
    )

    column_results = []
    constant_columns = 0
    two_value_columns = 0

    for column_index in range(k):
        column = values[:, column_index]
        unique_values = np.unique(column)
        unique_count = len(unique_values)

        if unique_count < gamma:
            if degenerate_policy == "strict":
                raise ValueError(
                    f"Column {column_index} has only "
                    f"{unique_count} distinct values; "
                    f"gamma={gamma} is not feasible"
                )

            if unique_count == 1:
                constant_columns += 1

                column_results.append(
                    ColumnClustering(
                        column_index=column_index,
                        centroids=unique_values.copy(),
                        cluster_sizes=np.array(
                            [n_clients],
                            dtype=np.int64,
                        ),
                        merge_boundary=0,
                        votes=np.zeros(
                            n_clients,
                            dtype=np.int64,
                        ),
                        status="constant",
                    )
                )
                continue

            if unique_count == 2:
                two_value_columns += 1

                # Exact two-value split avoids unnecessary
                # K-Means numerical instability.
                votes = (
                    column == unique_values[0]
                ).astype(np.int64)

                sizes = np.array(
                    [
                        int(votes.sum()),
                        int(n_clients - votes.sum()),
                    ],
                    dtype=np.int64,
                )

                vote_matrix[:, column_index] = votes

                column_results.append(
                    ColumnClustering(
                        column_index=column_index,
                        centroids=unique_values.copy(),
                        cluster_sizes=sizes,
                        merge_boundary=1,
                        votes=votes.copy(),
                        status="two_value",
                    )
                )
                continue

            # For gamma > 3, this policy deliberately
            # does not invent an additional merge rule.
            raise ValueError(
                f"Adaptive policy does not support "
                f"{unique_count} distinct values "
                f"with gamma={gamma}"
            )

        model = KMeans(
            n_clusters=gamma,
            random_state=seed + column_index,
            n_init=n_init,
        )

        labels = model.fit_predict(
            column.reshape(-1, 1)
        )

        centers = model.cluster_centers_.reshape(-1)

        # Even when unique_count >= gamma, numerical
        # degeneracy may yield fewer fitted clusters.
        if np.unique(labels).size != gamma:
            raise ValueError(
                f"Column {column_index}: K-Means formed "
                f"fewer than gamma={gamma} clusters. "
                "Numerical degeneracy requires inspection."
            )

        alpha_mask, boundary, centers, sizes = (
            merge_ordered_clusters(
                labels,
                centers,
            )
        )

        votes = alpha_mask.astype(np.int64)
        vote_matrix[:, column_index] = votes

        column_results.append(
            ColumnClustering(
                column_index=column_index,
                centroids=centers.copy(),
                cluster_sizes=sizes.copy(),
                merge_boundary=boundary,
                votes=votes.copy(),
                status="normal",
            )
        )

    return VotingResult(
        client_ids=list(parameter_matrix.client_ids),
        vote_counts=vote_matrix.sum(axis=1),
        vote_matrix=vote_matrix,
        column_results=column_results,
        constant_columns=constant_columns,
        two_value_columns=two_value_columns,
    )

