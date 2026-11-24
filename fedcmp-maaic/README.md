# FedCmp-MAAIC Research Project

## Research Objective

Investigate honest minority client rejection in
Byzantine-robust federated learning under non-IID
data distributions and develop a proposed minority-aware
defense called MAAIC.

## Current Project Status

- MNIST dataset preparation completed.
- Non-IID client partitioning completed.
- Federated learning framework implemented.
- FedCmp defense implemented.
- Five-seed no-attack experiments completed.
- Five-seed sign-flip experiments completed.
- MAAIC prototype implemented.
- Eight trainer tests passed.
- MAAIC command-line integration pending.
- MAAIC comparative experiments pending.

## Experimental Setup

| Parameter | Value |
|---|---|
| Dataset | MNIST |
| Model | CNN |
| Training samples | 54000 |
| Total clients | 20 |
| Honest majority clients | 14 |
| Honest minority clients | 4 |
| Designated Byzantine clients | 2 |
| Samples per client | 2700 |
| Communication rounds | 3 |
| Local epochs | 1 |
| Batch size | 64 |
| Learning rate | 0.01 |
| Sampled parameters | 40 |
| Initial clusters | 3 |
| Seeds | 42, 43, 44, 45, 46 |
| Attack | Sign flip |

## Preliminary FedCmp Results

| Metric | No attack | Sign flip |
|---|---:|---:|
| Minority FRR | 53.33% | 56.67% |
| Majority FRR | 27.14% | 29.05% |
| Rejection gap | 26.19 pp | 27.62 pp |
| Byzantine detection | N/A | 50.00% |
| Mean final validation accuracy | 77.85% | 71.54% |

These are preliminary descriptive results.
Statistical significance has not been established.

## Proposed MAAIC Defense

MAAIC extends FedCmp by reviewing initially rejected
clients using two consistency checks:

1. Cosine similarity with the retained reference update.
2. Update norm relative to the retained reference norms.

Clients satisfying both conditions are reconsidered
for aggregation.

Ground-truth client roles are used only for evaluation,
not for defense decisions.

MAAIC effectiveness has not yet been established
through comparative MNIST experiments.

## Repository Structure

    fedcmp-maaic/
    |-- src/fedresearch/
    |   |-- attacks/
    |   |-- data/
    |   |-- defenses/
    |   |-- fl/
    |   |-- models/
    |   |-- utils/
    |-- scripts/
    |-- tests/
    |-- results/
    |-- requirements.txt
    |-- README.md
    |-- .gitignore

## Installation

Recommended environment: Ubuntu or WSL with Python 3.12.

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

PyTorch installation may need adjustment depending
on the operating system and CPU/CUDA configuration.

The original environment information is available in
results/environment/.

MNIST is downloaded by the dataset loader when needed.

## Run Trainer Tests

    python -m pytest tests/test_fedcmp_trainer.py tests/test_maaic_trainer.py -q

## Run FedCmp Without Attack

    python -m scripts.run_fedcmp \
      --rounds 3 \
      --clients-per-round 20 \
      --local-epochs 1 \
      --batch-size 64 \
      --lr 0.01 \
      --seed 42 \
      --k 40 \
      --gamma 3 \
      --representation local_weight \
      --degenerate-policy adaptive \
      --attack-mode none \
      --output-dir results/team_no_attack_seed42

## Run FedCmp With Sign-Flip Attack

    python -m scripts.run_fedcmp \
      --rounds 3 \
      --clients-per-round 20 \
      --local-epochs 1 \
      --batch-size 64 \
      --lr 0.01 \
      --seed 42 \
      --k 40 \
      --gamma 3 \
      --representation local_weight \
      --degenerate-policy adaptive \
      --attack-mode sign_flip \
      --attack-scale 1.0 \
      --output-dir results/team_signflip_seed42

## View Existing Experimental Results

    python -m scripts.compare_fedcmp_experiments

## Remaining Research Work

- Complete MAAIC command-line integration.
- Verify MAAIC on MNIST.
- Verify attack and rescue behavior.
- Conduct matched five-seed MAAIC experiments.
- Compare FedCmp and MAAIC.
- Perform ablation and robustness studies.
- Prepare final research report.

## Team Collaboration

This repository contains ongoing research.

Team members should create separate Git branches
for new features and experimental changes.

Do not overwrite existing experimental results.
Use separate output directories for new experiments.

Virtual environments, raw datasets, model checkpoints
and credentials are excluded from version control.
