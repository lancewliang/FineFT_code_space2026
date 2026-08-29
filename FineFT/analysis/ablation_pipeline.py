from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import polars as pl
from scipy.stats import t as student_t

from operator_futures.feature_selection.muti_contract.regime_audit import (
    MARKET_STATE_ANCHOR_COLUMNS,
    TARGET_REGIME_BINS,
)


@dataclass
class RegimePerformanceRecord:
    slope_bin: int
    vol_bin: int
    is_target_regime: bool
    step_count: int
    step_ratio: float
    total_reward: float
    mean_reward: float
    trade_count: int
    turnover_rate: float
    total_commission: float
    total_slippage: float
    max_drawdown: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "slope_bin": self.slope_bin,
            "vol_bin": self.vol_bin,
            "is_target_regime": self.is_target_regime,
            "step_count": self.step_count,
            "step_ratio": self.step_ratio,
            "total_reward": self.total_reward,
            "mean_reward": self.mean_reward,
            "trade_count": self.trade_count,
            "turnover_rate": self.turnover_rate,
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
            "max_drawdown": self.max_drawdown,
        }


@dataclass
class AblationGroupResult:
    config_name: str
    stage: str
    total_steps: int
    macro_net_return: float
    macro_trade_count: int
    macro_commission: float
    macro_slippage: float
    macro_max_drawdown: float
    target_four_net_return: float
    target_four_trade_count: int
    target_four_commission: float
    target_four_slippage: float
    regime_records: list[RegimePerformanceRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_name": self.config_name,
            "stage": self.stage,
            "total_steps": self.total_steps,
            "macro_net_return": self.macro_net_return,
            "macro_trade_count": self.macro_trade_count,
            "macro_commission": self.macro_commission,
            "macro_slippage": self.macro_slippage,
            "macro_max_drawdown": self.macro_max_drawdown,
            "target_four_net_return": self.target_four_net_return,
            "target_four_trade_count": self.target_four_trade_count,
            "target_four_commission": self.target_four_commission,
            "target_four_slippage": self.target_four_slippage,
            "regime_records": [rec.to_dict() for rec in self.regime_records],
        }


def compute_paired_bootstrap_ci(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    *,
    num_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Compute paired bootstrap mean difference (B - A) and 95% confidence interval.
    Returns (mean_diff, lower_bound, upper_bound).
    """
    arr_a = np.asarray(sample_a, dtype=float)
    arr_b = np.asarray(sample_b, dtype=float)
    if len(arr_a) != len(arr_b) or len(arr_a) == 0:
        return 0.0, 0.0, 0.0

    diffs = arr_b - arr_a
    mean_diff = float(np.mean(diffs))

    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.zeros(num_bootstraps, dtype=float)

    for i in range(num_bootstraps):
        boot_idx = rng.choice(n, size=n, replace=True)
        boot_means[i] = np.mean(diffs[boot_idx])

    alpha = 1.0 - confidence_level
    lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))

    return mean_diff, lower, upper


def evaluate_regimes_from_detail_df(
    detail_df: pl.DataFrame,
    regime_quantiles: dict[str, list[float]],
    *,
    config_name: str = "A",
    stage: str = "valid",
) -> AblationGroupResult:
    """
    Analyze step-by-step trading detail dataframe and compute 16-regime performance breakdown.
    """
    if detail_df.height == 0:
        return AblationGroupResult(
            config_name=config_name,
            stage=stage,
            total_steps=0,
            macro_net_return=0.0,
            macro_trade_count=0,
            macro_commission=0.0,
            macro_slippage=0.0,
            macro_max_drawdown=0.0,
            target_four_net_return=0.0,
            target_four_trade_count=0,
            target_four_commission=0.0,
            target_four_slippage=0.0,
        )

    # Bin assignment using slope and volatility quantiles
    slope_th = regime_quantiles["slope"]
    vol_th = regime_quantiles["volatility"]

    slopes = detail_df.get_column("log_price_slope_48").to_numpy().astype(float) if "log_price_slope_48" in detail_df.columns else np.zeros(detail_df.height)
    close = detail_df.get_column("close").to_numpy().astype(float) if "close" in detail_df.columns else np.full(detail_df.height, 100.0)

    log_returns = np.diff(np.log(np.maximum(close, 1e-8)))
    vols = np.zeros(detail_df.height, dtype=float)
    if detail_df.height >= 48:
        rolling_returns = np.lib.stride_tricks.sliding_window_view(log_returns, 47)
        vols[47:] = rolling_returns.std(axis=1, ddof=0)

    slope_bins = np.zeros(detail_df.height, dtype=int)
    vol_bins = np.zeros(detail_df.height, dtype=int)

    for i in range(detail_df.height):
        s_val = slopes[i]
        v_val = vols[i]
        slope_bins[i] = 0 if s_val < slope_th[0] else (1 if s_val < slope_th[1] else (2 if s_val < slope_th[2] else 3))
        vol_bins[i] = 0 if v_val < vol_th[0] else (1 if v_val < vol_th[1] else (2 if v_val < vol_th[2] else 3))

    detail_df = detail_df.with_columns([
        pl.Series("slope_bin", slope_bins),
        pl.Series("vol_bin", vol_bins),
    ])

    regime_records: list[RegimePerformanceRecord] = []
    target_set = set(TARGET_REGIME_BINS)
    total_steps = detail_df.height

    for s_bin in range(4):
        for v_bin in range(4):
            sub = detail_df.filter((pl.col("slope_bin") == s_bin) & (pl.col("vol_bin") == v_bin))
            cnt = sub.height
            is_target = (s_bin, v_bin) in target_set

            if cnt == 0:
                regime_records.append(RegimePerformanceRecord(
                    slope_bin=s_bin,
                    vol_bin=v_bin,
                    is_target_regime=is_target,
                    step_count=0,
                    step_ratio=0.0,
                    total_reward=0.0,
                    mean_reward=0.0,
                    trade_count=0,
                    turnover_rate=0.0,
                    total_commission=0.0,
                    total_slippage=0.0,
                    max_drawdown=0.0,
                ))
                continue

            rewards = sub.get_column("step_reward").to_numpy().astype(float) if "step_reward" in sub.columns else np.zeros(cnt)
            trades = sub.get_column("trade_count_step").to_numpy().astype(int) if "trade_count_step" in sub.columns else np.zeros(cnt, dtype=int)
            fees = sub.get_column("commission_fee_step").to_numpy().astype(float) if "commission_fee_step" in sub.columns else np.zeros(cnt)
            slips = sub.get_column("slippage_step").to_numpy().astype(float) if "slippage_step" in sub.columns else np.zeros(cnt)
            margin_bal = sub.get_column("margin_balance").to_numpy().astype(float) if "margin_balance" in sub.columns else np.ones(cnt)

            cum_max = np.maximum.accumulate(margin_bal)
            drawdowns = np.where(cum_max > 0, (cum_max - margin_bal) / cum_max, 0.0)
            max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

            regime_records.append(RegimePerformanceRecord(
                slope_bin=s_bin,
                vol_bin=v_bin,
                is_target_regime=is_target,
                step_count=cnt,
                step_ratio=float(cnt) / float(total_steps),
                total_reward=float(np.sum(rewards)),
                mean_reward=float(np.mean(rewards)),
                trade_count=int(np.sum(trades)),
                turnover_rate=float(np.sum(trades)) / float(cnt),
                total_commission=float(np.sum(fees)),
                total_slippage=float(np.sum(slips)),
                max_drawdown=max_dd,
            ))

    macro_net_return = float(sum(r.total_reward for r in regime_records))
    macro_trade_count = sum(r.trade_count for r in regime_records)
    macro_commission = float(sum(r.total_commission for r in regime_records))
    macro_slippage = float(sum(r.total_slippage for r in regime_records))
    macro_max_dd = max(r.max_drawdown for r in regime_records) if regime_records else 0.0

    target_recs = [r for r in regime_records if r.is_target_regime]
    target_net_return = float(sum(r.total_reward for r in target_recs))
    target_trade_count = sum(r.trade_count for r in target_recs)
    target_commission = float(sum(r.total_commission for r in target_recs))
    target_slippage = float(sum(r.total_slippage for r in target_recs))

    return AblationGroupResult(
        config_name=config_name,
        stage=stage,
        total_steps=total_steps,
        macro_net_return=macro_net_return,
        macro_trade_count=macro_trade_count,
        macro_commission=macro_commission,
        macro_slippage=macro_slippage,
        macro_max_drawdown=macro_max_dd,
        target_four_net_return=target_net_return,
        target_four_trade_count=target_trade_count,
        target_four_commission=target_commission,
        target_four_slippage=target_slippage,
        regime_records=regime_records,
    )
