
import numpy as np
import torch

from src.fedresearch.defenses.fedcmp_sampling import (
    sample_parameter_matrix,
)
from src.fedresearch.models.cnn import MNISTCNN


def main():
    torch.manual_seed(42)

    model = MNISTCNN()
    global_state = model.state_dict()

    parameter_name = "classifier.3.weight"

    # Synthetic client deltas for a sampling smoke test.
    # These are NOT trained updates or attack results.
    generator = torch.Generator().manual_seed(123)

    updates = {}

    for client_id in range(20):
        updates[client_id] = {
            parameter_name: (
                0.01
                * torch.randn(
                    global_state[parameter_name].shape,
                    generator=generator,
                )
            )
        }

    result = sample_parameter_matrix(
        client_updates=updates,
        parameter_name=parameter_name,
        k=40,
        seed=42,
        global_state=global_state,
        representation="local_weight",
    )

    print("Participating clients:", len(result.client_ids))
    print("Selected parameter:", result.parameter_name)
    print("Matrix shape:", result.values.shape)
    print("Sampled coordinates:", len(result.coordinate_indices))
    print(
        "Unique coordinates:",
        len(np.unique(result.coordinate_indices)),
    )
    print(
        "All values finite:",
        bool(np.isfinite(result.values).all()),
    )

    assert result.values.shape == (20, 40)
    assert len(np.unique(result.coordinate_indices)) == 40

    print("FedCmp parameter sampling: PASSED")


if __name__ == "__main__":
    main()
