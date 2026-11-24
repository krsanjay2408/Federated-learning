
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class DetailedEvaluation:
    loss: float
    accuracy: float
    total: int
    confusion_matrix: np.ndarray
    per_digit_accuracy: dict[int, float]
    majority_digit_accuracy: float
    minority_digit_accuracy: float


@torch.no_grad()
def evaluate_detailed(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int = 256,
    device: str = "cpu",
) -> DetailedEvaluation:
    """Evaluate a 10-class MNIST model."""

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

    confusion = np.zeros((10, 10), dtype=np.int64)
    total_loss = 0.0
    total = 0

    try:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            if logits.ndim != 2 or logits.shape[1] != 10:
                raise ValueError(
                    "Expected model output shape (batch, 10)"
                )

            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)

            total_loss += loss.item()
            total += labels.size(0)

            actual = labels.cpu().numpy()
            predicted = predictions.cpu().numpy()

            np.add.at(
                confusion,
                (actual, predicted),
                1,
            )
    finally:
        model.train(was_training)

    class_totals = confusion.sum(axis=1)
    correct_per_class = np.diag(confusion)

    per_digit_accuracy = {
        digit: (
            float(correct_per_class[digit] / class_totals[digit])
            if class_totals[digit] > 0
            else float("nan")
        )
        for digit in range(10)
    }

    def group_accuracy(digits):
        group_total = class_totals[digits].sum()

        if group_total == 0:
            return float("nan")

        group_correct = correct_per_class[digits].sum()
        return float(group_correct / group_total)

    return DetailedEvaluation(
        loss=total_loss / total,
        accuracy=float(np.trace(confusion) / total),
        total=total,
        confusion_matrix=confusion,
        per_digit_accuracy=per_digit_accuracy,
        majority_digit_accuracy=group_accuracy([0, 1, 2, 3]),
        minority_digit_accuracy=group_accuracy([7, 8, 9]),
    )
