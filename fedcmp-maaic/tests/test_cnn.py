
import torch
from torch import nn

from src.fedresearch.models.cnn import MNISTCNN


def test_output_shape():
    model = MNISTCNN()
    images = torch.randn(8, 1, 28, 28)

    logits = model(images)

    assert logits.shape == (8, 10)


def test_backward_pass():
    model = MNISTCNN()
    images = torch.randn(4, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3])

    logits = model(images)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
    )


def test_parameter_count():
    model = MNISTCNN()

    count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert count > 0
    print("Model parameters:", count)
