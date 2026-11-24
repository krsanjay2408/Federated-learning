"""Controlled sign-flipping attack for federated learning."""

from collections import OrderedDict

import torch


def sign_flip_update(
    update: dict[str, torch.Tensor],
    scale: float = 1.0,
) -> OrderedDict:
    """
    Construct a malicious client update by sign flipping.

    For each parameter:
        malicious_delta = -scale * honest_delta

    The original update is never modified.
    """

    if scale <= 0:
        raise ValueError("Attack scale must be positive")

    if not update:
        raise ValueError("Client update cannot be empty")

    malicious_update = OrderedDict()

    for name, tensor in update.items():
        if not torch.is_floating_point(tensor):
            raise TypeError(
                f"Non-floating update tensor: {name}"
            )

        malicious_update[name] = (
            -scale * tensor.detach().clone()
        )

    return malicious_update
