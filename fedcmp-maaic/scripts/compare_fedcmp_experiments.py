"""Validate and compare FedCmp no-attack and sign-flipping runs."""

import csv
from pathlib import Path
from statistics import mean

SEEDS = [42, 43, 44, 45, 46]
OUTPUT = Path("results/fedcmp_comparison")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def as_bool(value):
    if value not in {"True", "False"}:
        raise ValueError(f"Unexpected boolean value: {value!r}")
    return value == "True"


def experiment_dir(seed, attack):
    if attack:
        return Path(f"results/fedcmp_signflip_seed{seed}")
    if seed == 42:
        return Path("results/fedcmp_no_attack")
    return Path(f"results/fedcmp_no_attack_seed{seed}")


def analyze(seed, attack):
    directory = experiment_dir(seed, attack)

    history = read_csv(directory / "history.csv")
    decisions = read_csv(directory / "client_decisions.csv")

    if len(history) != 3:
        raise ValueError(
            f"{directory}: expected 3 history rows, got {len(history)}"
        )

    if len(decisions) != 60:
        raise ValueError(
            f"{directory}: expected 60 decisions, got {len(decisions)}"
        )

    minority_rejected = 0
    minority_total = 0
    majority_rejected = 0
    majority_total = 0
    malicious_rejected = 0
    malicious_total = 0

    for round_number in (1, 2, 3):
        rows = [
            row for row in decisions
            if int(row["round"]) == round_number
        ]

        if len(rows) != 20:
            raise ValueError(
                f"{directory}: round {round_number} has "
                f"{len(rows)} decisions"
            )

        ids = [int(row["client_id"]) for row in rows]

        if len(set(ids)) != 20 or set(ids) != set(range(20)):
            raise ValueError(
                f"{directory}: invalid client IDs in round {round_number}"
            )

        expected_malicious = {18, 19} if attack else set()

        actual_malicious = {
            int(row["client_id"])
            for row in rows
            if as_bool(row["actual_malicious"])
        }

        if actual_malicious != expected_malicious:
            raise ValueError(
                f"{directory}: round {round_number} malicious "
                f"clients {actual_malicious}, expected "
                f"{expected_malicious}"
            )

        for row in rows:
            client_id = int(row["client_id"])
            role = row["designated_role"]
            malicious = as_bool(row["actual_malicious"])
            retained = as_bool(row["retained"])
            rejected = not retained

            expected_role = (
                "honest_majority" if client_id < 14
                else "honest_minority" if client_id < 18
                else "byzantine"
            )

            if role != expected_role:
                raise ValueError(
                    f"{directory}: unexpected role for client {client_id}"
                )

            if malicious:
                malicious_total += 1
                malicious_rejected += int(rejected)
            elif role == "honest_minority":
                minority_total += 1
                minority_rejected += int(rejected)
            elif role == "honest_majority":
                majority_total += 1
                majority_rejected += int(rejected)

    final_accuracy = float(history[-1]["validation_accuracy"])

    return {
        "seed": seed,
        "attack_mode": "sign_flip" if attack else "none",
        "minority_rejected": minority_rejected,
        "minority_total": minority_total,
        "minority_frr": minority_rejected / minority_total,
        "majority_rejected": majority_rejected,
        "majority_total": majority_total,
        "majority_frr": majority_rejected / majority_total,
        "malicious_rejected": malicious_rejected,
        "malicious_total": malicious_total,
        "attack_detection_rate": (
            malicious_rejected / malicious_total
            if malicious_total else None
        ),
        "final_validation_accuracy": final_accuracy,
    }


def summarize(rows):
    minority_rejected = sum(r["minority_rejected"] for r in rows)
    minority_total = sum(r["minority_total"] for r in rows)

    majority_rejected = sum(r["majority_rejected"] for r in rows)
    majority_total = sum(r["majority_total"] for r in rows)

    malicious_rejected = sum(r["malicious_rejected"] for r in rows)
    malicious_total = sum(r["malicious_total"] for r in rows)

    return {
        "minority_frr": minority_rejected / minority_total,
        "majority_frr": majority_rejected / majority_total,
        "attack_detection_rate": (
            malicious_rejected / malicious_total
            if malicious_total else None
        ),
        "mean_final_accuracy": mean(
            r["final_validation_accuracy"] for r in rows
        ),
    }


def main():
    baseline = [analyze(seed, False) for seed in SEEDS]
    attacks = [analyze(seed, True) for seed in SEEDS]

    OUTPUT.mkdir(parents=True, exist_ok=True)

    path = OUTPUT / "comparison.csv"

    rows = baseline + attacks

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    baseline_summary = summarize(baseline)
    attack_summary = summarize(attacks)

    print("\nFedCmp five-seed comparison")
    print("=" * 60)

    for label, summary in [
        ("NO ATTACK", baseline_summary),
        ("SIGN FLIP", attack_summary),
    ]:
        print(f"\n{label}")
        print(
            "Minority FRR:",
            f"{summary['minority_frr']:.2%}",
        )
        print(
            "Majority FRR:",
            f"{summary['majority_frr']:.2%}",
        )
        print(
            "Byzantine detection:",
            (
                f"{summary['attack_detection_rate']:.2%}"
                if summary["attack_detection_rate"] is not None
                else "N/A"
            ),
        )
        print(
            "Mean final validation accuracy:",
            f"{summary['mean_final_accuracy']:.2%}",
        )

    print("\nSaved:", path)


if __name__ == "__main__":
    main()
