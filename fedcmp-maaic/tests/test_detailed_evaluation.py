
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from src.fedresearch.fl.detailed_evaluation import (
    evaluate_detailed,
)


class IdentityClassifier(nn.Module):
    def forward(self, x):
        return x


def make_dataset():
    # Ten one-hot inputs, one for each MNIST class.
    labels = torch.arange(10)

    logits = torch.eye(10) * 10.0

    return TensorDataset(logits, labels)


def test_perfect_predictions():
    dataset = make_dataset()
    model = IdentityClassifier()

    result = evaluate_detailed(model, dataset)

    assert result.total == 10
    assert result.accuracy == 1.0

    for digit in range(10):
        assert result.per_digit_accuracy[digit] == 1.0


def test_confusion_matrix_shape():
    result = evaluate_detailed(
        IdentityClassifier(),
        make_dataset(),
    )

    assert result.confusion_matrix.shape == (10, 10)
    assert result.confusion_matrix.sum() == 10

    np.testing.assert_array_equal(
        np.diag(result.confusion_matrix),
        np.ones(10, dtype=np.int64),
    )


def test_group_accuracies():
    result = evaluate_detailed(
        IdentityClassifier(),
        make_dataset(),
    )

    assert result.majority_digit_accuracy == 1.0
    assert result.minority_digit_accuracy == 1.0


def test_model_mode_restored():
    model = IdentityClassifier()
    model.train()

    evaluate_detailed(model, make_dataset())

    assert model.training is True


def test_empty_dataset_rejected():
    empty = TensorDataset(
        torch.empty(0, 10),
        torch.empty(0, dtype=torch.long),
    )

    with pytest.raises(ValueError):
        evaluate_detailed(
            IdentityClassifier(),
            empty,
        )


def test_invalid_batch_size():
    with pytest.raises(ValueError):
        evaluate_detailed(
            IdentityClassifier(),
            make_dataset(),
            batch_size=0,
        )
