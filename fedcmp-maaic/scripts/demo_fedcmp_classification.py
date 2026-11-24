
import numpy as np

from src.fedresearch.defenses.fedcmp_sampling import (
    ParameterMatrix,
)
from src.fedresearch.defenses.fedcmp_voting import (
    cluster_and_vote,
)
from src.fedresearch.defenses.fedcmp_classification import (
    classify_clients,
)


def main():
    rng = np.random.default_rng(42)
    k = 40

    # Synthetic parameter values for 20 clients.
    # These are NOT trained MNIST updates.
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

    voting = cluster_and_vote(
        matrix,
        gamma=3,
        seed=42,
    )

    classification = classify_clients(
        voting,
        seed=42,
    )

    print("Parameter matrix:", matrix.values.shape)
    print("Vote matrix:", voting.vote_matrix.shape)
    print("Vote vector:", voting.vote_counts.shape)

    print("\nClient classification:")
    for i, client_id in enumerate(
        classification.client_ids
    ):
        status = (
            "RETAINED"
            if classification.benign_mask[i]
            else "REJECTED"
        )

        print(
            f"Client {client_id:02d} | "
            f"Votes: {voting.vote_counts[i]:2d} | "
            f"{status}"
        )

    print(
        "\nCluster centroids:",
        classification.cluster_centers,
    )
    print(
        "Cluster sizes:",
        classification.cluster_sizes,
    )

    print(
        "Retained clients:",
        classification.retained_client_ids,
    )
    print(
        "Rejected clients:",
        classification.rejected_client_ids,
    )

    print(
        "Identical-vote fallback:",
        classification.used_fallback,
    )

    assert classification.retained_client_ids == list(
        range(7, 20)
    )
    assert classification.rejected_client_ids == list(
        range(7)
    )

    print("\nFedCmp final classification: PASSED")


if __name__ == "__main__":
    main()
