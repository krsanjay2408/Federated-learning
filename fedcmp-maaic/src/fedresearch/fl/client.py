
from collections import OrderedDict
from dataclasses import dataclass
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class ClientResult:
    client_id: int
    num_samples: int
    mean_loss: float
    update: OrderedDict


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_client(
    client_id: int,
    global_model: nn.Module,
    dataset: Dataset,
    epochs: int = 1,
    batch_size: int = 64,
    lr: float = 0.01,
    seed: int = 42,
    device: str = "cpu",
) -> ClientResult:
    """
    Train a private copy of the global model.

    Returns delta = local_state - global_state.
    The input global_model is never modified.
    """
    if len(dataset) == 0:
        raise ValueError("Client dataset cannot be empty")
    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("Invalid training hyperparameters")

    # Keep global_model unchanged.
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()

    # Snapshot global weights on CPU.
    global_state = OrderedDict(
        (name, tensor.detach().cpu().clone())
        for name, tensor in global_model.state_dict().items()
    )

    generator = torch.Generator().manual_seed(seed + client_id)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    optimizer = torch.optim.SGD(
        local_model.parameters(),
        lr=lr,
    )

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_examples = 0

    for _ in range(epochs):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = local_model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size_actual = labels.size(0)

            total_loss += (
                loss.item() * batch_size_actual
            )
            total_examples += batch_size_actual

    local_state = local_model.state_dict()

    update = OrderedDict()

    for name, local_tensor in local_state.items():
        if not torch.is_floating_point(local_tensor):
            raise TypeError(
                f"Non-floating state entry unsupported: {name}"
            )

        update[name] = (
            local_tensor.detach().cpu()
            - global_state[name]
        ).clone()

    return ClientResult(
        client_id=client_id,
        num_samples=len(dataset),
        mean_loss=total_loss / total_examples,
        update=update,
    )
