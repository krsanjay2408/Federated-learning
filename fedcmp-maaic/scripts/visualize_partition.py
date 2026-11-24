
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.fedresearch.data.mnist import load_mnist
from src.fedresearch.data.partition import (
    iid_partition,
    dirichlet_partition,
    validate_partition,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", choices=["iid", "dirichlet"],
        default="dirichlet"
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = load_mnist(seed=args.seed)

    # Labels must correspond to positions in data.train.
    labels = np.asarray(data.train.dataset.targets)[
        data.train.indices
    ]

    if args.method == "iid":
        partitions = iid_partition(
            len(labels), args.clients, args.seed
        )
    else:
        partitions = dirichlet_partition(
            labels, args.clients, args.alpha, args.seed
        )

    validate_partition(partitions, len(labels))

    counts = np.zeros((args.clients, 10), dtype=int)

    for client_id, positions in partitions.items():
        counts[client_id] = np.bincount(
            labels[positions], minlength=10
        )

    # Normalize rows to show each client's class proportions.
    totals = counts.sum(axis=1, keepdims=True)
    proportions = counts / np.maximum(totals, 1)

    fig, ax = plt.subplots(figsize=(11, 8))
    image = ax.imshow(
        proportions, aspect="auto",
        interpolation="nearest", cmap="viridis",
        vmin=0, vmax=1
    )

    ax.set_xlabel("MNIST digit")
    ax.set_ylabel("Client ID")
    ax.set_title(
        f"{args.method.upper()} MNIST distribution "
        f"(alpha={args.alpha}, seed={args.seed})"
    )
    ax.set_xticks(range(10))
    ax.set_yticks(range(args.clients))
    fig.colorbar(
        image, ax=ax, label="Fraction of client samples"
    )
    fig.tight_layout()

    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (
        f"{args.method}_clients{args.clients}"
        f"_alpha{args.alpha}_seed{args.seed}.png"
    )

    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Saved heatmap: {output_path}")
    print(f"Clients: {args.clients}")
    print(f"Total assigned samples: {counts.sum()}")
    print(f"Minimum client samples: {totals.min()}")
    print(f"Maximum client samples: {totals.max()}")


if __name__ == "__main__":
    main()
