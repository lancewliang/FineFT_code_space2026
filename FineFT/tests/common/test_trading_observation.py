import numpy as np
import pytest

from common.trading_observation import (
    ExitMemoryRecord,
    TRADING_INFO_DIM,
    TRADING_INFO_KEYS,
    TRADING_INFO_SCHEMA,
    TradingObservationConfig,
    TradingObservationError,
    adjust_causal_warmup_start,
    calculate_causal_price_scale,
    encode_exit_memory,
    load_trading_observation_metadata,
    save_trading_observation_metadata,
    validate_trading_info,
    validate_trading_info_batch,
    validate_trading_metadata,
    validate_trading_state_dict,
)


def test_contract_declares_target_order_without_importing_environment():
    assert TRADING_INFO_DIM == 11
    assert TRADING_INFO_KEYS[:4] == (
        "position_exposure",
        "single_holding_return_rate",
        "single_holding_max_drawdown",
        "current_holding_duration_norm",
    )
    assert TRADING_INFO_KEYS[4] == "unrealized_loss_to_notional_ratio"
    assert TRADING_INFO_KEYS[-1] == "price_drop_since_last_short_exit_tanh"


def test_validation_rejects_legacy_width_and_non_finite_values():
    with pytest.raises(TradingObservationError, match="width 11"):
        validate_trading_info(np.zeros(4, dtype=np.float32))
    invalid = np.zeros(TRADING_INFO_DIM, dtype=np.float32)
    invalid[3] = np.nan
    with pytest.raises(TradingObservationError, match="non-finite"):
        validate_trading_info(invalid)
    assert validate_trading_info_batch(np.zeros((2, TRADING_INFO_DIM))).dtype == np.float32


def test_causal_scale_and_directional_exit_memory_are_causal():
    prices = np.arange(100.0, 120.0)
    assert calculate_causal_price_scale(prices, 19, 5.0) == 5.0
    record = ExitMemoryRecord.create(19, execution_price=110.0, price_scale=5.0)
    encoded = encode_exit_memory(record, current_step=24, current_price=115.0, direction="long")
    assert encoded[:2] == pytest.approx((1.0, 5.0 / 256.0))
    assert encoded[2] == pytest.approx(np.tanh(0.2))
    with pytest.raises(TradingObservationError, match="future"):
        encode_exit_memory(record, current_step=18, current_price=110.0, direction="long")


def test_metadata_and_checkpoint_validation_are_strict(tmp_path):
    config = TradingObservationConfig(exit_price_scale_floor=5.0)
    metadata = config.to_metadata()
    assert metadata["schema"] == TRADING_INFO_SCHEMA
    assert validate_trading_metadata(metadata).exit_price_scale_floor == 5.0
    path = tmp_path / "trading_observation.json"
    save_trading_observation_metadata(path, config)
    assert load_trading_observation_metadata(path).schema == TRADING_INFO_SCHEMA
    assert validate_trading_state_dict({"fc_trading.weight": np.zeros((8, 11))}) == 11
    with pytest.raises(TradingObservationError, match="width must be 11"):
        validate_trading_state_dict({"fc_trading.weight": np.zeros((8, 4))})


def test_partition_warmup_is_explicit():
    assert adjust_causal_warmup_start(0, 0, 30) == (18, 18)
    with pytest.raises(TradingObservationError, match="no decision"):
        adjust_causal_warmup_start(0, 0, 10)
