
import numpy as np
import pytest
import torch

from src.fedresearch.defenses.fedcmp_sampling import (
    sample_parameter_matrix,
)


def make_updates():
    return {
        0: {
            "layer.weight": torch.tensor(
                [0.0, 1.0, 2.0, 3.0, 4.0]
            )
        },
        1: {
            "layer.weight": torch.tensor(
                [10.0, 11.0, 12.0, 13.0, 14.0]
            )
        },
        2: {
            "layer.weight": torch.tensor(
                [20.0, 21.0, 22.0, 23.0, 24.0]
            )
        },
    }


def test_matrix_shape():
    result = sample_parameter_matrix(
        make_updates(), "layer.weight", k=3
    )

    assert result.values.shape == (3, 3)
    assert len(result.coordinate_indices) == 3


def test_aligned_coordinates():
    result = sample_parameter_matrix(
        make_updates(), "layer.weight", k=3
    )

    # Each client differs by exactly 10 at every coordinate.
    np.testing.assert_allclose(
        result.values[1] - result.values[0],
        np.full(3, 10.0),
    )

    np.testing.assert_allclose(
        result.values[2] - result.values[1],
        np.full(3, 10.0),
    )


def test_sampling_reproducible():
    first = sample_parameter_matrix(
        make_updates(), "layer.weight", k=3, seed=42
    )

    second = sample_parameter_matrix(
        make_updates(), "layer.weight", k=3, seed=42
    )

    np.testing.assert_array_equal(
        first.coordinate_indices,
        second.coordinate_indices,
    )
    np.testing.assert_array_equal(
        first.values,
        second.values,
    )


def test_no_repeated_coordinates():
    result = sample_parameter_matrix(
        make_updates(), "layer.weight", k=4
    )

    assert len(set(result.coordinate_indices.tolist())) == 4


def test_local_weight_reconstruction():
    global_state = {
        "layer.weight": torch.full((5,), 100.0)
    }

    delta = sample_parameter_matrix(
        make_updates(),
        "layer.weight",
        k=5,
        representation="delta",
    )

    weights = sample_parameter_matrix(
        make_updates(),
        "layer.weight",
        k=5,
        representation="local_weight",
        global_state=global_state,
    )

    np.testing.assert_array_equal(
        delta.coordinate_indices,
        weights.coordinate_indices,
    )

    np.testing.assert_allclose(
        weights.values - delta.values,
        100.0,
    )


def test_invalid_k():
    with pytest.raises(ValueError):
        sample_parameter_matrix(
            make_updates(), "layer.weight", k=6
        )

    with pytest.raises(ValueError):
        sample_parameter_matrix(
            make_updates(), "layer.weight", k=0
        )


def test_missing_parameter():
    with pytest.raises(KeyError):
        sample_parameter_matrix(
            make_updates(), "missing.weight", k=2
        )


def test_shape_mismatch():
    updates = make_updates()
    updates[2]["layer.weight"] = torch.randn(6)

    with pytest.raises(ValueError):
        sample_parameter_matrix(
            updates, "layer.weight", k=3
        )


def test_nan_rejected():
    updates = make_updates()
    updates[1]["layer.weight"][0] = float("nan")

    with pytest.raises(ValueError):
        sample_parameter_matrix(
            updates, "layer.weight", k=5
        )


def test_client_order():
    updates = make_updates()

    result = sample_parameter_matrix(
        updates,
        "layer.weight",
        k=2,
        client_ids=[2, 0, 1],
    )

    assert result.client_ids == [2, 0, 1]
    assert result.values[0].min() >= 20.0
