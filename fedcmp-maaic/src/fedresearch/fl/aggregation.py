
from collections import OrderedDict

import torch
from torch import nn

from src.fedresearch.fl.client import ClientResult


def fedavg_aggregate(
    global_model: nn.Module,
    client_results: list[ClientResult],
) -> OrderedDict:
    """
    Compute the next global state using weighted FedAvg.

    Update convention:
        delta_i = local_state_i - global_state

    Returns a new state_dict without modifying global_model.
    """
    if not client_results:
        raise ValueError("At least one client result is required")

    total_samples = sum(
        result.num_samples for result in client_results
    )

    if any(r.num_samples <= 0 for r in client_results):
        raise ValueError("Client sample counts must be positive")

    global_state = global_model.state_dict()
    aggregated = OrderedDict()

    for name, global_tensor in global_state.items():
        if not torch.is_floating_point(global_tensor):
            raise TypeError(
                f"Non-floating state entry unsupported: {name}"
            )

        weighted_update = torch.zeros_like(
            global_tensor,
            dtype=torch.float64,
            device="cpu",
        )

        for result in client_results:
            if name not in result.update:
                raise KeyError(
                    f"Client {result.client_id} missing {name}"
                )

            delta = result.update[name]

            if delta.shape != global_tensor.shape:
                raise ValueError(
                    f"Shape mismatch for {name}"
                )

            weight = result.num_samples / total_samples

            weighted_update += (
                weight * delta.detach().cpu().double()
            )

        aggregated[name] = (
            global_tensor.detach().cpu().double()
            + weighted_update
        ).to(dtype=global_tensor.dtype)

    return aggregated


def apply_global_state(
    global_model: nn.Module,
    new_state: OrderedDict,
) -> None:
    """Load an aggregated state into the server model."""
    global_model.load_state_dict(new_state, strict=True)
