"""Run FedCmp experiments with optional Byzantine attacks."""

import argparse
import csv
from pathlib import Path

import torch

from src.fedresearch.data.mnist import load_mnist
from src.fedresearch.data.minority_partition import minority_partition
from src.fedresearch.fl.client import set_seed
from src.fedresearch.fl.fedcmp_trainer import run_fedcmp_training
from src.fedresearch.models.cnn import MNISTCNN


def main():
    parser = argparse.ArgumentParser(
        description="FedCmp MNIST experiment with optional attacks."
    )

    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--clients-per-round", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--gamma", type=int, default=3)

    parser.add_argument(
        "--representation",
        choices=["delta", "local_weight"],
        default="local_weight",
    )

    parser.add_argument(
        "--degenerate-policy",
        choices=["strict", "adaptive"],
        default="strict",
    )

    parser.add_argument(
        "--parameter-name",
        type=str,
        default="classifier.3.weight",
    )

    parser.add_argument(
        "--attack-mode",
        choices=["none", "sign_flip"],
        default="none",
    )

    parser.add_argument(
        "--attack-scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/fedcmp",
    )

    parser.add_argument(
        "--defense-mode",
        choices=["fedcmp", "maaic"],
        default="fedcmp",
    )

    parser.add_argument(
        "--maaic-direction-threshold",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--maaic-max-norm-ratio",
        type=float,
        default=3.0,
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Never silently overwrite an existing experiment.
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new --output-dir."
        )

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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"{args.defense_mode.upper()} MNIST Experiment")
    print("=" * 60)
    print("Device:", device)
    print("Rounds:", args.rounds)
    print("Clients per round:", args.clients_per_round)
    print("Local epochs:", args.local_epochs)
    print("Batch size:", args.batch_size)
    print("Learning rate:", args.lr)
    print("Seed:", args.seed)
    print("Sampled parameters (K):", args.k)
    print("Initial clusters (gamma):", args.gamma)
    print("Parameter:", args.parameter_name)
    print("Representation:", args.representation)
    print("Degenerate policy:", args.degenerate_policy)
    print("Defense mode:", args.defense_mode)
    print("MAAIC direction threshold:", args.maaic_direction_threshold)
    print("MAAIC max norm ratio:", args.maaic_max_norm_ratio)
    print("Attack mode:", args.attack_mode)
    print("Attack scale:", args.attack_scale)
    print("=" * 60, flush=True)

    def log_round(result):
        print(
            f"\nRound {result.round_number:02d} | "
            f"Participating: {len(result.selected_clients)} | "
            f"Retained: {len(result.retained_clients)} | "
            f"Rejected: {len(result.rejected_clients)}",
            flush=True,
        )

        print(
            f"  Validation accuracy: "
            f"{100 * result.validation_accuracy:.2f}%"
        )
        print(f"  Minority FRR: {result.minority_frr}")
        print(f"  Majority FRR: {result.majority_frr}")
        print(f"  Honest overall FRR: {result.overall_frr}")
        print(
            f"  Actual malicious clients: "
            f"{result.actual_malicious_clients}"
        )
        print(
            f"  Attack detection rate: "
            f"{result.attack_detection_rate}"
        )
        print(
            f"  Malicious rejected: {result.malicious_rejected}"
        )
        print(
            f"  Malicious retained: {result.malicious_retained}"
        )
        print(
            f"  Classification fallback: "
            f"{result.used_fallback}",
            flush=True,
        )

    history = run_fedcmp_training(
        model=model,
        train_dataset=data.train,
        validation_dataset=data.validation,
        partitions=partition.partitions,
        roles=partition.roles,
        rounds=args.rounds,
        clients_per_round=args.clients_per_round,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=device,
        parameter_name=args.parameter_name,
        k=args.k,
        gamma=args.gamma,
        representation=args.representation,
        degenerate_policy=args.degenerate_policy,
        on_round_end=log_round,
        attack_mode=args.attack_mode,
        attack_scale=args.attack_scale,
        defense_mode=args.defense_mode,
        maaic_direction_threshold=args.maaic_direction_threshold,
        maaic_max_norm_ratio=args.maaic_max_norm_ratio,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / "history.csv"

    history_fields = [
        "round",
        "participating",
        "retained",
        "rejected",
        "mean_client_loss",
        "validation_loss",
        "validation_accuracy",
        "minority_frr",
        "majority_frr",
        "overall_frr",
        "constant_columns",
        "two_value_columns",
        "normal_columns",
        "used_fallback",
        "actual_malicious_clients",
        "attack_detection_rate",
        "malicious_retained",
        "malicious_rejected",
        "defense_mode",
        "maaic_direction_threshold",
        "maaic_max_norm_ratio",
        "attack_mode",
        "attack_scale",
    ]

    with history_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=history_fields,
        )
        writer.writeheader()

        for result in history:
            writer.writerow({
                "round": result.round_number,
                "participating": ";".join(
                    map(str, result.selected_clients)
                ),
                "retained": ";".join(
                    map(str, result.retained_clients)
                ),
                "rejected": ";".join(
                    map(str, result.rejected_clients)
                ),
                "mean_client_loss": result.mean_client_loss,
                "validation_loss": result.validation_loss,
                "validation_accuracy": result.validation_accuracy,
                "minority_frr": result.minority_frr,
                "majority_frr": result.majority_frr,
                "overall_frr": result.overall_frr,
                "constant_columns": result.constant_columns,
                "two_value_columns": result.two_value_columns,
                "normal_columns": result.normal_columns,
                "used_fallback": result.used_fallback,
                "actual_malicious_clients": ";".join(
                    map(str, result.actual_malicious_clients or [])
                ),
                "attack_detection_rate": (
                    result.attack_detection_rate
                ),
                "malicious_retained": result.malicious_retained,
                "malicious_rejected": result.malicious_rejected,
                "defense_mode": args.defense_mode,
                "maaic_direction_threshold": args.maaic_direction_threshold,
                "maaic_max_norm_ratio": args.maaic_max_norm_ratio,
                "attack_mode": args.attack_mode,
                "attack_scale": args.attack_scale,
            })

    decisions_path = output_dir / "client_decisions.csv"

    decision_fields = [
        "round",
        "client_id",
        "designated_role",
        "actual_malicious",
        "vote_count",
        "retained",
        "num_samples",
        "defense_mode",
        "maaic_direction_threshold",
        "maaic_max_norm_ratio",
        "attack_mode",
    ]

    with decisions_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=decision_fields,
        )
        writer.writeheader()

        for result in history:
            retained_set = set(result.retained_clients)
            malicious_set = set(
                result.actual_malicious_clients or []
            )

            for client_id in result.selected_clients:
                writer.writerow({
                    "round": result.round_number,
                    "client_id": client_id,
                    "designated_role": partition.roles[client_id],
                    "actual_malicious": client_id in malicious_set,
                    "vote_count": result.vote_counts[client_id],
                    "retained": client_id in retained_set,
                    "num_samples": (
                        result.client_sample_counts[client_id]
                    ),
                    "defense_mode": args.defense_mode,
                    "maaic_direction_threshold": args.maaic_direction_threshold,
                    "maaic_max_norm_ratio": args.maaic_max_norm_ratio,
                    "attack_mode": args.attack_mode,
                })

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
            "k": args.k,
            "gamma": args.gamma,
            "parameter_name": args.parameter_name,
            "representation": args.representation,
            "degenerate_policy": args.degenerate_policy,
            "defense_mode": args.defense_mode,
            "maaic_direction_threshold": args.maaic_direction_threshold,
            "maaic_max_norm_ratio": args.maaic_max_norm_ratio,
            "attack_mode": args.attack_mode,
            "attack_scale": args.attack_scale,
        },
        checkpoint_path,
    )

    print("\n" + "=" * 60)
    print(f"{args.defense_mode.upper()} experiment completed")
    print("=" * 60)
    print("Saved:", history_path)
    print("Saved:", decisions_path)
    print("Saved:", checkpoint_path)


if __name__ == "__main__":
    main()
