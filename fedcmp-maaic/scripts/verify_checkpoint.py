
from pathlib import Path

import torch

from src.fedresearch.models.cnn import MNISTCNN
from src.fedresearch.utils.experiment import (
    state_dict_sha256,
)


def main():
    source = Path("results/fedavg/global_model.pt")

    if not source.exists():
        raise FileNotFoundError(source)

    checkpoint = torch.load(
        source,
        map_location="cpu",
        weights_only=True,
    )

    first = MNISTCNN()
    first.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    first.eval()

    first_hash = state_dict_sha256(first.state_dict())

    output = Path("results/archive/verified_fedavg.pt")
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": first.state_dict(),
            "model_sha256": first_hash,
        },
        output,
    )

    restored = torch.load(
        output,
        map_location="cpu",
        weights_only=True,
    )

    second = MNISTCNN()
    second.load_state_dict(
        restored["model_state_dict"],
        strict=True,
    )
    second.eval()

    second_hash = state_dict_sha256(
        second.state_dict()
    )

    # Fixed synthetic input for prediction comparison.
    generator = torch.Generator().manual_seed(42)

    images = torch.randn(
        8, 1, 28, 28,
        generator=generator,
    )

    with torch.no_grad():
        first_logits = first(images)
        second_logits = second(images)

    assert first_hash == second_hash
    assert torch.equal(first_logits, second_logits)

    print("Checkpoint verification: PASSED")
    print("Model hash:", first_hash)
    print("Predictions identical:", True)
    print("Verified copy:", output)


if __name__ == "__main__":
    main()
