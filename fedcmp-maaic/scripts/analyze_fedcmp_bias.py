"""Analyze honest-client rejection across FedCmp no-attack runs."""

import csv
from collections import Counter
from pathlib import Path

SEEDS = (42, 43, 44, 45, 46)
OUTPUT_DIR = Path("results/fedcmp_bias_analysis")

ROLES = {
    "honest_minority": 4,
    "honest_majority": 14,
}


def run_directory(seed):
    if seed == 42:
        return Path("results/fedcmp_no_attack")
    return Path(f"results/fedcmp_no_attack_seed{seed}")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def parse_bool(value):
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Unexpected Boolean value: {value!r}")


def rejection_stats(rows, role):
    selected = [row for row in rows if row["designated_role"] == role]
    rejected = sum(not parse_bool(row["retained"]) for row in selected)
    return rejected, len(selected)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summary = []

    for seed in SEEDS:
        directory = run_directory(seed)
        decisions = read_csv(directory / "client_decisions.csv")
        history = read_csv(directory / "history.csv")

        if len(decisions) != 60 or len(history) != 3:
            raise ValueError(f"Seed {seed}: unexpected number of rows")

        for round_number in (1, 2, 3):
            round_rows = [
                row for row in decisions
                if int(row["round"]) == round_number
            ]

            ids = [int(row["client_id"]) for row in round_rows]

            if len(ids) != 20 or sorted(ids) != list(range(20)):
                raise ValueError(
                    f"Seed {seed}, round {round_number}: invalid client IDs"
                )

            if any(parse_bool(row["actual_malicious"]) for row in round_rows):
                raise ValueError(
                    f"Seed {seed}: active attacker found in no-attack data"
                )

            role_counts = Counter(
                row["designated_role"] for row in round_rows
            )

            if role_counts != {
                "honest_majority": 14,
                "honest_minority": 4,
                "byzantine": 2,
            }:
                raise ValueError(
                    f"Seed {seed}, round {round_number}: invalid roles"
                )

        for row in decisions:
            row["seed"] = seed
            all_rows.append(row)

        minority_rejected, minority_total = rejection_stats(
            decisions, "honest_minority"
        )
        majority_rejected, majority_total = rejection_stats(
            decisions, "honest_majority"
        )

        minority_frr = minority_rejected / minority_total
        majority_frr = majority_rejected / majority_total

        summary.append({
            "seed": seed,
            "minority_rejected": minority_rejected,
            "minority_total": minority_total,
            "minority_frr": minority_frr,
            "majority_rejected": majority_rejected,
            "majority_total": majority_total,
            "majority_frr": majority_frr,
            "frr_gap": minority_frr - majority_frr,
            "final_validation_accuracy": float(
                history[-1]["validation_accuracy"]
            ),
        })

    output_path = OUTPUT_DIR / "seed_summary.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    minority_rejected = sum(r["minority_rejected"] for r in summary)
    minority_total = sum(r["minority_total"] for r in summary)
    majority_rejected = sum(r["majority_rejected"] for r in summary)
    majority_total = sum(r["majority_total"] for r in summary)

    minority_frr = minority_rejected / minority_total
    majority_frr = majority_rejected / majority_total

    print("=" * 65)
    print("FedCmp No-Attack Minority Bias Analysis")
    print("=" * 65)

    for row in summary:
        print(
            f"Seed {row['seed']} | "
            f"Minority FRR: {row['minority_frr']:.2%} | "
            f"Majority FRR: {row['majority_frr']:.2%} | "
            f"Gap: {row['frr_gap']:+.2%} | "
            f"Accuracy: {row['final_validation_accuracy']:.2%}"
        )

    print("-" * 65)
    print(f"Total client-round records: {len(all_rows)}")
    print(
        f"Minority rejected: {minority_rejected}/{minority_total} "
        f"({minority_frr:.2%})"
    )
    print(
        f"Majority rejected: {majority_rejected}/{majority_total} "
        f"({majority_frr:.2%})"
    )
    print(f"Pooled FRR gap: {minority_frr - majority_frr:+.2%}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
