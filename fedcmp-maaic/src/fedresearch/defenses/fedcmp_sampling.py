
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ParameterMatrix:
    """Aligned parameter values from participating clients."""

    client_ids: list[int]
    parameter_name: str
    coordinate_indices: np.ndarray
    values: np.ndarray


def sample_parameter_matrix(
    client_updates: Mapping[int, Mapping[str, torch.Tensor]],
    parameter_name: str,
    k: int = 40,
    seed: int = 42,
    client_ids: Sequence[int] | None = None,
    global_state: Mapping[str, torch.Tensor] | None = None,
    representation: str = "delta",
) -> ParameterMatrix:
    """
    Construct an N x K FedCmp parameter matrix.

    representation:
        "delta"        : local_model - global_model
        "local_weight" : reconstructed local model weights

    client_updates contains parameter deltas produced by train_client().

    Coordinates are sampled once and reused across all clients.
    """

    if not client_updates:
        raise ValueError("At least one client update is required")

    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")

    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    if representation not in {"delta", "local_weight"}:
        raise ValueError("Invalid representation")

    if representation == "local_weight" and global_state is None:
        raise ValueError(
            "global_state is required for local_weight representation"
        )

    ids = (
        sorted(client_updates)
        if client_ids is None
        else list(client_ids)
    )

    if not ids:
        raise ValueError("client_ids cannot be empty")

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate client IDs")

    if set(ids) != set(client_updates):
        raise ValueError(
            "client_ids must match the supplied client updates"
        )

    first = client_updates[ids[0]]

    if parameter_name not in first:
        raise KeyError(f"Missing parameter: {parameter_name}")

    reference = first[parameter_name]

    if not isinstance(reference, torch.Tensor):
        raise TypeError("Parameter must be a torch.Tensor")

    if not torch.is_floating_point(reference):
        raise TypeError("Parameter must be floating point")

    shape = reference.shape
    parameter_count = reference.numel()

    if k > parameter_count:
        raise ValueError(
            f"k={k} exceeds parameter count={parameter_count}"
        )

    global_flat = None

    if representation == "local_weight":
        if parameter_name not in global_state:
            raise KeyError(
                f"Missing global parameter: {parameter_name}"
            )

        global_tensor = global_state[parameter_name]

        if global_tensor.shape != shape:
            raise ValueError("Global parameter shape mismatch")

        if not torch.is_floating_point(global_tensor):
            raise TypeError("Global parameter must be floating point")

        global_flat = (
            global_tensor.detach()
            .cpu()
            .reshape(-1)
            .to(torch.float64)
            .numpy()
        )

    rng = np.random.default_rng(seed)

    coordinates = np.sort(
        rng.choice(
            parameter_count,
            size=k,
            replace=False,
        )
    )

    matrix = np.empty((len(ids), k), dtype=np.float64)

    for row, client_id in enumerate(ids):
        update = client_updates[client_id]

        if parameter_name not in update:
            raise KeyError(
                f"Client {client_id} missing {parameter_name}"
            )

        tensor = update[parameter_name]

        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Parameter must be a torch.Tensor")

        if tensor.shape != shape:
            raise ValueError(
                f"Client {client_id} parameter shape mismatch"
            )

        if not torch.is_floating_point(tensor):
            raise TypeError("Parameter must be floating point")

        flat = (
            tensor.detach()
            .cpu()
            .reshape(-1)
            .to(torch.float64)
            .numpy()
        )

        sampled = flat[coordinates]

        if global_flat is not None:
            sampled = sampled + global_flat[coordinates]

        if not np.isfinite(sampled).all():
            raise ValueError(
                f"Client {client_id} contains NaN or infinity"
            )

        matrix[row] = sampled

    return ParameterMatrix(
        client_ids=ids,
        parameter_name=parameter_name,
        coordinate_indices=coordinates,
        values=matrix,
    )
