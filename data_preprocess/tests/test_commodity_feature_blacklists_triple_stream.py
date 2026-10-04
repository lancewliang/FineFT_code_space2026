from __future__ import annotations

import io
from contextlib import redirect_stdout
from operator_futures.feature_selection.blacklists import (
    get_commodity_global_hygiene_blacklist,
    get_commodity_rl_feature_blacklist,
    get_commodity_stream_blacklists,
    get_commodity_vae_slope_feature_blacklist,
    get_commodity_vae_volatility_feature_blacklist,
    main,
)
import sys
from pathlib import Path

FINEFT_ROOT = Path(__file__).resolve().parents[2] / "FineFT"
if str(FINEFT_ROOT) not in sys.path:
    sys.path.insert(0, str(FINEFT_ROOT))

from common.artifacts import ArtifactNames


def test_artifact_names_triple_stream() -> None:
    assert ArtifactNames.VAE_SLOPE_STATE_FEATURES_NPY == "vae_slope_state_features.npy"
    assert (
        ArtifactNames.VAE_VOLATILITY_STATE_FEATURES_NPY
        == "vae_volatility_state_features.npy"
    )
    assert not hasattr(ArtifactNames, "VAE_STATE_FEATURES_NPY")


def test_commodity_stream_blacklists_decoupling() -> None:
    slope_bl, vol_bl, rl_bl = get_commodity_stream_blacklists("10min")

    assert len(slope_bl) > 0
    assert len(vol_bl) > 0
    assert len(rl_bl) > 0

    # Slope blacklist must filter pure volatility features
    assert "realized_vol_zscore_192" in slope_bl
    assert "atr_pct_6" in slope_bl
    assert "bollinger_bandwidth_12_origin" in slope_bl
    assert "historical_volatility_24" in slope_bl

    # Volatility blacklist must filter signed directional features
    assert "wap_1_trend_24" in vol_bl
    assert "log_price_slope_96" in vol_bl
    assert "ema_slope_192" in vol_bl
    assert "beta_16_std_norm_origin" in vol_bl

    # RL blacklist must NOT filter macro trend slopes or volatility
    assert "log_price_slope_96" not in rl_bl
    assert "realized_vol_zscore_192" not in rl_bl


def test_blacklists_cli_queries() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(["--stream", "vae_slope", "--freq", "10min"])
    out_slope = buf.getvalue().splitlines()
    assert "atr_pct_6" in out_slope

    buf = io.StringIO()
    with redirect_stdout(buf):
        main(["--stream", "vae_volatility", "--freq", "10min"])
    out_vol = buf.getvalue().splitlines()
    assert "wap_1_trend_24" in out_vol
