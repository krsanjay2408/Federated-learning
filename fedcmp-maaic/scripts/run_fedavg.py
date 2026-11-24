
import argparse
import csv
from pathlib import Path

import torch

from src.fedresearch.data.mnist import load_mnist
from src.fedresearch.data.minority_partition import (
    minority_partition,
)
from src.fedresearch.fl.client import set_seed
from src.fedresearch.fl.trainer import (
    run_federated_training,
)
from src.fedresearch.models.cnn import MNISTCNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--clients-per-round", type=int, default=4)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    data = load_mnist(seed=args.seed)

    labels = torch.as_tensor(
        data.train.dataset.targets
    )[data.train.indices].numpy()

    partition = minority_partition(
        labels,
        seed=args.seed,
        preference=8.0,
    )

    model = MNISTCNN()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)
    print("Rounds:", args.rounds)
    print("Clients per round:", args.clients_per_round)

    def log_round(result):
        print(
            f"Round {result.round_number:02d} | "
            f"Clients {result.selected_clients} | "
            f"Client loss {result.mean_client_loss:.4f} | "
            f"Validation loss {result.validation_loss:.4f} | "
            f"Validation accuracy "
            f"{100 * result.validation_accuracy:.2f}%",
            flush=True,
        )

    history = run_federated_training(
        model=model,
        train_dataset=data.train,
        validation_dataset=data.validation,
        partitions=partition.partitions,
        rounds=args.rounds,
        clients_per_round=args.clients_per_round,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=device,
        on_round_end=log_round,
    )

    output_dir = Path("results/fedavg")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "history.csv"

    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "round",
            "selected_clients",
            "mean_client_loss",
            "validation_loss",
            "validation_accuracy",
        ])

        for result in history:
            writer.writerow([
                result.round_number,
                ";".join(
                    map(str, result.selected_clients)
                ),
                result.mean_client_loss,
                result.validation_loss,
                result.validation_accuracy,
            ])

    checkpoint_path = output_dir / "global_model.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seed": args.seed,
            "rounds": args.rounds,
            "clients_per_round": args.clients_per_round,
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
        },
        checkpoint_path,
    )

    print("Saved:", csv_path)
    print("Saved:", checkpoint_path)


if __name__ == "__main__":
    main()
