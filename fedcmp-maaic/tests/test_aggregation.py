
from collections import OrderedDict

import pytest
import torch
from torch import nn

from src.fedresearch.fl.client import ClientResult
from src.fedresearch.fl.aggregation import (
    fedavg_aggregate,
    apply_global_state,
)


def make_result(client_id, samples, delta):
    return ClientResult(
        client_id=client_id,
        num_samples=samples,
        mean_loss=1.0,
        update=OrderedDict(
            weight=torch.tensor([[delta]], dtype=torch.float32)
        ),
    )


def make_model():
    model = nn.Linear(1, 1, bias=False)

    with torch.no_grad():
        model.weight.fill_(10.0)

    return model


def test_weighted_average():
    model = make_model()

    results = [
        make_result(0, 100, 2.0),
        make_result(1, 300, 6.0),
    ]

    state = fedavg_aggregate(model, results)

    # 10 + (0.25 * 2 + 0.75 * 6) = 15
    assert torch.allclose(
        state["weight"],
        torch.tensor([[15.0]]),
    )


def test_global_model_not_modified():
    model = make_model()

    results = [make_result(0, 100, 2.0)]

    fedavg_aggregate(model, results)

    assert torch.allclose(
        model.weight,
        torch.tensor([[10.0]]),
    )


def test_apply_global_state():
    model = make_model()

    results = [make_result(0, 100, 2.0)]
    state = fedavg_aggregate(model, results)

    apply_global_state(model, state)

    assert torch.allclose(
        model.weight,
        torch.tensor([[12.0]]),
    )


def test_empty_results():
    model = make_model()

    with pytest.raises(ValueError):
        fedavg_aggregate(model, [])


def test_invalid_sample_count():
    model = make_model()

    results = [make_result(0, 0, 2.0)]

    with pytest.raises(ValueError):
        fedavg_aggregate(model, results)


def test_missing_parameter():
    model = make_model()

    result = make_result(0, 100, 2.0)
    result.update.clear()

    with pytest.raises(KeyError):
        fedavg_aggregate(model, [result])
