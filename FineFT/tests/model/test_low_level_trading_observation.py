import numpy as np
import pytest
import torch

from common.trading_observation import TradingObservationError
from model.low_level import Qnet, ensemble_Qnet


def _inputs(batch=2, width=11):
    return (
        torch.zeros(batch, 3),
        torch.zeros(batch, 2),
        torch.zeros(batch, 1),
        torch.ones(batch, 3),
        torch.zeros(batch, width),
    )


def test_qnet_and_ensemble_consume_all_eleven_fields():
    model = Qnet(3, 3, 8, 2)
    output = model(*_inputs())
    assert output.shape == (2, 3)
    ensemble = ensemble_Qnet(3, 3, 8, 2, ensemble_number=2)
    assert ensemble(*_inputs()).shape == (2, 2, 3)


def test_width_and_finite_validation_is_strict():
    model = Qnet(3, 3, 8, 2)
    with pytest.raises(TradingObservationError, match="expects"):
        model(*_inputs(width=4))
    bad = list(_inputs())
    bad[-1] = torch.full((2, 11), float("nan"))
    with pytest.raises(TradingObservationError, match="non-finite"):
        model(*bad)


def test_new_field_can_change_q_values():
    torch.manual_seed(7)
    model = Qnet(3, 3, 8, 2)
    for parameter in model.parameters():
        parameter.data.fill_(0.1)
    base = list(_inputs(batch=1))
    changed = list(base)
    changed[-1] = base[-1].clone()
    changed[-1][0, 10] = 1.0
    assert not torch.equal(model(*base), model(*changed))
