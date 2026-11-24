
import torch
from torch.utils.data import Subset

from src.fedresearch.data.mnist import load_mnist
from src.fedresearch.data.minority_partition import (
    minority_partition,
)
from src.fedresearch.fl.client import (
    set_seed,
    train_client,
)
from src.fedresearch.models.cnn import MNISTCNN


def main():
    seed = 42
    set_seed(seed)

    data = load_mnist(seed=seed)

    labels = torch.as_tensor(
        data.train.dataset.targets
    )[data.train.indices].numpy()

    partition = minority_partition(
        labels,
        seed=seed,
        preference=8.0,
    )

    client_id = 14  # Honest minority client

    client_dataset = Subset(
        data.train,
        partition.partitions[client_id],
    )

    model = MNISTCNN()

    result = train_client(
        client_id=client_id,
        global_model=model,
        dataset=client_dataset,
        epochs=1,
        batch_size=64,
        lr=0.01,
        seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    update_norm = torch.sqrt(
        sum(
            (tensor.float() ** 2).sum()
            for tensor in result.update.values()
        )
    ).item()

    print("Client ID:", result.client_id)
    print("Role:", partition.roles[client_id])
    print("Samples:", result.num_samples)
    print("Mean training loss:", round(result.mean_loss, 4))
    print("Update L2 norm:", round(update_norm, 6))


if __name__ == "__main__":
    main()
