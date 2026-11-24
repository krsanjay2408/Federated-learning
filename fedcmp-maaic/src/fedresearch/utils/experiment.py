
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def configure_reproducibility(seed: int) -> None:
    """Configure common sources of randomness."""
    if seed < 0:
        raise ValueError("seed must be nonnegative")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Raise an error if an unsupported nondeterministic
    # operation is used.
    torch.use_deterministic_algorithms(True)


def state_dict_sha256(state_dict) -> str:
    """Compute a stable hash of model tensors."""
    digest = hashlib.sha256()

    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()

        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())

    return digest.hexdigest()


def save_experiment(
    output_dir: str | Path,
    model: torch.nn.Module,
    config: dict,
    history: list[dict],
) -> None:
    """Save model, configuration, and history."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }

    model_hash = state_dict_sha256(state)

    torch.save(
        {
            "model_state_dict": state,
            "config": config,
            "model_sha256": model_hash,
        },
        output_dir / "checkpoint.pt",
    )

    with (output_dir / "config.json").open("w") as file:
        json.dump(config, file, indent=2, sort_keys=True)

    with (output_dir / "history.json").open("w") as file:
        json.dump(history, file, indent=2)

    with (output_dir / "model_sha256.txt").open("w") as file:
        file.write(model_hash + "\n")


def load_verified_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
) -> dict:
    """Load a checkpoint and verify its model hash."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    expected_hash = checkpoint["model_sha256"]
    actual_hash = state_dict_sha256(
        checkpoint["model_state_dict"]
    )

    if expected_hash != actual_hash:
        raise ValueError("Checkpoint model hash mismatch")

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    return checkpoint["config"]
