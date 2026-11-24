
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.fedresearch.data.mnist import load_mnist
from src.fedresearch.data.minority_partition import (
    minority_partition,
)


def main():
    seed = 42
    preference = 8.0

    data = load_mnist(seed=seed)

    labels = np.asarray(
        data.train.dataset.targets,
        dtype=np.int64,
    )[data.train.indices]

    result = minority_partition(
        labels,
        seed=seed,
        preference=preference,
    )

    output = Path("results/minority")
    output.mkdir(parents=True, exist_ok=True)

    counts = result.counts
    proportions = counts / counts.sum(
        axis=1, keepdims=True
    )

    fig, ax = plt.subplots(figsize=(11, 8))
    image = ax.imshow(
        proportions,
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    ax.axhline(13.5, color="white", linewidth=1.5)
    ax.axhline(17.5, color="white", linewidth=1.5)

    ax.set_xticks(range(10))
    ax.set_yticks(range(20))
    ax.set_xlabel("MNIST digit")
    ax.set_ylabel("Client ID")
    ax.set_title(
        f"Controlled majority/minority partition "
        f"(preference={preference}, seed={seed})"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Fraction of client samples",
    )

    fig.tight_layout()
    fig.savefig(
        output / "minority_heatmap.png",
        dpi=200,
    )
    plt.close(fig)

    with (output / "class_counts.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ["client_id", "role", "total"]
            + [f"digit_{d}" for d in range(10)]
        )

        for client_id in range(20):
            writer.writerow(
                [
                    client_id,
                    result.roles[client_id],
                    int(counts[client_id].sum()),
                ]
                + counts[client_id].tolist()
            )

    # Partition positions refer to data.train.
    # Convert them to original MNIST training indices.
    manifest = {
        "seed": seed,
        "preference": preference,
        "train_indices": {
            str(client_id): [
                int(data.train_indices[position])
                for position in positions
            ]
            for client_id, positions
            in result.partitions.items()
        },
        "roles": {
            str(client_id): role
            for client_id, role in result.roles.items()
        },
    }

    with (output / "manifest.json").open("w") as file:
        json.dump(manifest, file, indent=2)

    majority_fraction = (
        counts[:14, :4].sum()
        / counts[:14].sum()
    )
    minority_fraction = (
        counts[14:18, 7:10].sum()
        / counts[14:18].sum()
    )

    print("Clients:", len(result.partitions))
    print("Assigned samples:", counts.sum())
    print("Samples per client:", counts.sum(axis=1)[0])
    print(
        "Majority preferred-digit fraction:",
        round(float(majority_fraction), 4),
    )
    print(
        "Minority preferred-digit fraction:",
        round(float(minority_fraction), 4),
    )
    print("Output directory:", output)


if __name__ == "__main__":
    main()
