
from src.fedresearch.data.mnist import load_mnist


def test_mnist_split():
    data = load_mnist()

    assert len(data.train) == 54000
    assert len(data.validation) == 6000
    assert len(data.test) == 10000

    train_ids = set(data.train_indices)
    val_ids = set(data.validation_indices)

    assert train_ids.isdisjoint(val_ids)

    assert len(train_ids | val_ids) == 60000


def test_reproducibility():
    first = load_mnist(seed=42)
    second = load_mnist(seed=42)

    assert first.train_indices == second.train_indices
    assert first.validation_indices == second.validation_indices


def test_image_shape():
    data = load_mnist()

    image, label = data.train[0]

    assert image.shape == (1, 28, 28)
    assert image.dtype.is_floating_point
    assert 0 <= label <= 9
