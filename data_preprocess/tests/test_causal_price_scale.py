import numpy as np
import polars as pl
import pytest

from operator_futures.scale_describe_save.causal_price_scale import (
    CAUSAL_PRICE_SCALE_COLUMN,
    CAUSAL_PRICE_SCALE_VALID_COLUMN,
    attach_causal_price_statistics,
    prepare_causal_price_statistics,
)


def test_preparation_marks_warmup_and_uses_only_past_rows():
    prices = np.ones(20, dtype=float)
    prices[18] = 10.0
    prices[19] = 12.0
    stats = prepare_causal_price_statistics(prices, np.arange(20))
    assert np.all(stats.scale[:18] == 0)
    assert not stats.valid[:18].any()
    assert stats.valid[18:].tolist() == [True, True]
    assert stats.scale[18] == pytest.approx(9.0 / 18.0)
    assert stats.scale[19] == pytest.approx(11.0 / 18.0)

    changed_future = prices.copy()
    changed_future[-1] = 1000.0
    changed_stats = prepare_causal_price_statistics(changed_future, np.arange(20))
    assert changed_stats.scale[18] == stats.scale[18]


def test_constant_window_is_valid_zero_and_short_partition_stays_invalid():
    constant = prepare_causal_price_statistics(np.ones(19), np.arange(19))
    assert constant.valid[-1]
    assert constant.scale[-1] == 0.0
    short = prepare_causal_price_statistics(np.ones(18), np.arange(18))
    assert not short.valid.any()
    assert np.all(short.scale == 0)


def test_input_quality_and_auxiliary_alignment_are_strict():
    with pytest.raises(ValueError, match="strictly positive"):
        prepare_causal_price_statistics([1.0, 0.0], [0, 1])
    with pytest.raises(ValueError, match="increasing"):
        prepare_causal_price_statistics([1.0, 2.0], [1, 0])
    frame = pl.DataFrame({"mark_price": np.ones(19), "timestamp": np.arange(19)})
    output, stats = attach_causal_price_statistics(frame)
    assert output.columns[-2:] == [CAUSAL_PRICE_SCALE_COLUMN, CAUSAL_PRICE_SCALE_VALID_COLUMN]
    assert output.height == frame.height
    assert stats.valid_row_count == 1
