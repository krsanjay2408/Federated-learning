
import numpy as np

from src.fedresearch.defenses.fedcmp_sampling import (
    ParameterMatrix,
)
from src.fedresearch.defenses.fedcmp_voting import (
    cluster_and_vote,
)


def main():
    rng = np.random.default_rng(42)

    k = 40

    # Synthetic client parameter values.
    # These are not actual trained model updates.
    values = np.concatenate(
        [
            rng.normal(-5, 0.1, size=(7, k)),
            rng.normal(0, 0.1, size=(7, k)),
            rng.normal(5, 0.1, size=(6, k)),
        ],
        axis=0,
    )

    matrix = ParameterMatrix(
        client_ids=list(range(20)),
        parameter_name="synthetic.weight",
        coordinate_indices=np.arange(k),
        values=values,
    )

    result = cluster_and_vote(
        matrix,
        gamma=3,
        seed=42,
    )

    print("Parameter matrix:", matrix.values.shape)
    print("Vote matrix:", result.vote_matrix.shape)
    print("Vote vector:", result.vote_counts.shape)

    print("\nClient vote counts:")

    for client_id, votes in zip(
        result.client_ids,
        result.vote_counts,
    ):
        print(f"Client {client_id:02d}: {votes:2d} / {k}")

    first = result.column_results[0]

    print("\nFirst parameter column:")
    print("Sorted centroids:", first.centroids)
    print("Cluster sizes:", first.cluster_sizes)
    print("Merge boundary:", first.merge_boundary)
    print("Clients receiving votes:", int(first.votes.sum()))

    assert result.vote_matrix.shape == (20, 40)
    assert result.vote_counts.shape == (20,)

    print("\nFedCmp clustering and voting: PASSED")


if __name__ == "__main__":
    main()
