
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


@dataclass
class MNISTData:
    train: Dataset
    validation: Dataset
    test: Dataset
    train_indices: list[int]
    validation_indices: list[int]


def load_mnist(
    root: str = "./datasets",
    seed: int = 42,
    validation_size: int = 6000,
) -> MNISTData:
    """Load MNIST with a reproducible train/validation split."""

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    full_train = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )

    test = datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=transform,
    )

    if not 0 < validation_size < len(full_train):
        raise ValueError("Invalid validation_size")

    generator = torch.Generator().manual_seed(seed)

    shuffled_indices = torch.randperm(
        len(full_train),
        generator=generator,
    ).tolist()

    validation_indices = shuffled_indices[:validation_size]
    train_indices = shuffled_indices[validation_size:]

    assert set(train_indices).isdisjoint(validation_indices)

    train = Subset(full_train, train_indices)
    validation = Subset(full_train, validation_indices)

    return MNISTData(
        train=train,
        validation=validation,
        test=test,
        train_indices=train_indices,
        validation_indices=validation_indices,
    )
