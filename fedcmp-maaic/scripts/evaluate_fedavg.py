
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.fedresearch.data.mnist import load_mnist
from src.fedresearch.fl.detailed_evaluation import (
    evaluate_detailed,
)
from src.fedresearch.models.cnn import MNISTCNN


def main():
    checkpoint_path = Path(
        "results/fedavg/global_model.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model = MNISTCNN()
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    seed = int(checkpoint["seed"])
    data = load_mnist(seed=seed)

    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    result = evaluate_detailed(
        model=model,
        dataset=data.test,
        batch_size=256,
        device=device,
    )

    output_dir = Path("results/fedavg/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "checkpoint": str(checkpoint_path),
        "seed": seed,
        "rounds": int(checkpoint["rounds"]),
        "test_samples": result.total,
        "test_loss": result.loss,
        "test_accuracy": result.accuracy,
        "majority_digit_accuracy": (
            result.majority_digit_accuracy
        ),
        "minority_digit_accuracy": (
            result.minority_digit_accuracy
        ),
        "per_digit_accuracy": {
            str(k): v
            for k, v in result.per_digit_accuracy.items()
        },
    }

    with (output_dir / "report.json").open("w") as f:
        json.dump(report, f, indent=2)

    with (output_dir / "per_digit_accuracy.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["digit", "accuracy"])

        for digit, accuracy in (
            result.per_digit_accuracy.items()
        ):
            writer.writerow([digit, accuracy])

    confusion = result.confusion_matrix

    np.savetxt(
        output_dir / "confusion_matrix.csv",
        confusion,
        fmt="%d",
        delimiter=",",
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(
        confusion,
        cmap="Blues",
        interpolation="nearest",
    )

    ax.set_title("FedAvg MNIST Confusion Matrix")
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

    fig.colorbar(
        image,
        ax=ax,
        label="Number of images",
    )

    fig.tight_layout()
    fig.savefig(
        output_dir / "confusion_matrix.png",
        dpi=200,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))

    digits = list(range(10))
    accuracies = [
        100 * result.per_digit_accuracy[d]
        for d in digits
    ]

    bars = ax.bar(digits, accuracies)

    for digit in [7, 8, 9]:
        bars[digit].set_color("teal")

    ax.set_xticks(digits)
    ax.set_ylim(0, 100)
    ax.set_xlabel("MNIST digit")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("FedAvg Per-Digit Test Accuracy")

    fig.tight_layout()
    fig.savefig(
        output_dir / "per_digit_accuracy.png",
        dpi=200,
    )
    plt.close(fig)

    print("Device:", device)
    print("Test samples:", result.total)
    print(f"Test loss: {result.loss:.4f}")
    print(
        f"Overall test accuracy: "
        f"{100 * result.accuracy:.2f}%"
    )

    print("\nPer-digit accuracy:")

    for digit, accuracy in (
        result.per_digit_accuracy.items()
    ):
        print(f"  Digit {digit}: {100 * accuracy:.2f}%")

    print(
        "\nMajority digits (0-3): "
        f"{100 * result.majority_digit_accuracy:.2f}%"
    )
    print(
        "Minority digits (7-9): "
        f"{100 * result.minority_digit_accuracy:.2f}%"
    )

    print("\nSaved evaluation to:", output_dir)


if __name__ == "__main__":
    main()
