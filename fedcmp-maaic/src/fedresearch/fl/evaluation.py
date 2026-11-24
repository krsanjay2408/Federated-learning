
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class EvaluationResult:
    loss: float
    accuracy: float
    total: int


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int = 256,
    device: str = "cpu",
) -> EvaluationResult:
    """Evaluate a model without changing its parameters."""
    if len(dataset) == 0:
        raise ValueError("Evaluation dataset cannot be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    was_training = model.training
    model.to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    correct = 0
    total = 0

    try:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            total_loss += criterion(
                logits, labels
            ).item()

            predictions = logits.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)
    finally:
        model.train(was_training)

    return EvaluationResult(
        loss=total_loss / total,
        accuracy=correct / total,
        total=total,
    )
