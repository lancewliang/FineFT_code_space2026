"""Aggregate test_agent_index detail CSVs into trade lifecycle detail, trend summary CSVs, and selection_manifest.json using Polars."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

# Standard header mapping from Chinese/English to canonical internal key
HEADER_MAPPING = {
    "标签": "label",
    "label": "label",
    "数据文件": "df_path",
    "df_path": "df_path",
    "初始动作": "initial_action",
    "initial_action": "initial_action",
    "分箱索引": "bin_index",
    "bin_index": "bin_index",
    "时间步": "timestep",
    "timestep": "timestep",
    "时间戳": "timestamp",
    "timestamp": "timestamp",
    "开盘价": "open",
    "open": "open",
    "最高价": "high",
    "high": "high",
    "最低价": "low",
    "low": "low",
    "收盘价": "close",
    "close": "close",
    "成交量": "volume",
    "volume": "volume",
    "标记价格": "mark_price",
    "mark_price": "mark_price",
    "动作": "action",
    "action": "action",
    "目标仓位": "target_position",
    "target_position": "target_position",
    "目标杠杆": "target_leverage",
    "target_leverage": "target_leverage",
    "执行前仓位": "position_before",
    "position_before": "position_before",
    "执行前杠杆": "leverage_before",
    "leverage_before": "leverage_before",
    "执行后仓位": "position_after",
    "position_after": "position_after",
    "执行后杠杆": "leverage_after",
    "leverage_after": "leverage_after",
    "单步奖励": "step_reward",
    "step_reward": "step_reward",
    "单步实现盈亏": "realized_pnl_step",
    "realized_pnl_step": "realized_pnl_step",
    "累计已实现盈亏": "cumulative_realized_pnl",
    "cumulative_realized_pnl": "cumulative_realized_pnl",
    "单步手续费": "commission_fee_step",
    "commission_fee_step": "commission_fee_step",
    "累计手续费": "cumulative_commission_fee",
    "cumulative_commission_fee": "cumulative_commission_fee",
    "单步滑点": "slippage_step",
    "slippage_step": "slippage_step",
    "累计滑点": "cumulative_slippage",
    "cumulative_slippage": "cumulative_slippage",
    "浮动盈亏": "unrealized_pnl",
    "unrealized_pnl": "unrealized_pnl",
    "保证金余额": "margin_balance",
    "margin_balance": "margin_balance",
    "持仓资产": "notional_asset_value",
    "notional_asset_value": "notional_asset_value",
    "结算总价值": "wallet_balance",
    "wallet_balance": "wallet_balance",
    "浮动总价值": "total_value",
    "total_value": "total_value",
}

# Output bilingual header mappings
LIFECYCLE_HEADER_LABELS = {
    "epoch": "轮次",
    "label": "标签",
    "bin_index": "分箱索引",
    "contract": "合约",
    "df_path": "数据文件",
    "initial_action": "初始动作",
    "trade_id": "交易ID",
    "start_timestep": "开始时间步",
    "end_timestep": "结束时间步",
    "start_timestamp": "开始时间戳",
    "end_timestamp": "结束时间戳",
    "holding_duration": "持仓步数",
    "trade_direction": "交易方向",
    "segment_type": "分片类型",
    "trend_type": "趋势类型",
    "entry_price": "开仓价格",
    "avg_entry_price": "加权开仓价格",
    "exit_price": "平仓价格",
    "realized_pnl_sum": "已实现盈亏合计",
    "tail_unrealized_pnl": "尾部浮动盈亏结算",
    "commission_fee_sum": "手续费合计",
    "slippage_sum": "滑点合计",
    "net_pnl": "净利润",
    "return_rate": "持仓收益率",
    "max_position_abs": "最大绝对仓位",
    "mean_position_abs": "平均绝对仓位",
    "position_change_count": "调仓次数",
    "profitable_increase_count": "盈利同向加仓次数",
    "max_drawdown": "持仓最大回撤",
    "is_tail_forced_close": "是否尾部强平",
}

SUMMARY_HEADER_LABELS = {
    "epoch": "轮次",
    "label": "标签",
    "bin_index": "分箱索引",
    "trend_type": "趋势类型",
    "trade_count": "交易总笔数",
    "long_trade_count": "做多次数",
    "short_trade_count": "做空次数",
    "win_trade_count": "盈利笔数",
    "win_rate": "胜率",
    "long_win_rate": "做多胜率",
    "short_win_rate": "做空胜率",
    "total_holding_steps": "总持仓步数",
    "mean_holding_duration": "平均持仓步数",
    "total_realized_pnl": "总已实现盈亏",
    "total_commission_fee": "总手续费",
    "total_slippage": "总滑点",
    "total_net_pnl": "总净利润",
    "mean_net_pnl": "平均单笔交易利润",
    "pnl_p25": "净利润P25",
    "pnl_p50": "净利润P50",
    "pnl_p75": "净利润P75",
    "profit_factor": "盈亏比",
    "mean_max_position_abs": "平均最大仓位",
    "total_profitable_increase_count": "盈利加仓总次数",
    "mean_max_drawdown": "平均持仓最大回撤",
    "max_drawdown": "最大持仓回撤",
}


def _parse_label_info(label_val: Any) -> tuple[str, str]:
    """Extract standard label name and determine directional segment type."""
    label_str = str(label_val).strip()
    match = re.search(r"(\d+)", label_str)
    if match:
        idx = int(match.group(1))
    else:
        idx = -1

    clean_label = f"label_{idx}" if idx >= 0 else label_str

    # 0, 1, 2: downtrend; 3, 4, 5: uptrend
    if idx in (0, 1, 2):
        segment_type = "下跌分片"
    elif idx in (3, 4, 5):
        segment_type = "上涨分片"
    else:
        segment_type = "未知分片"

    return clean_label, segment_type


def _determine_trend_type(segment_type: str, position: float) -> str:
    """Classify trade position into Trend-Following or Trend-Reversion based on segment."""
    if segment_type == "下跌分片":
        if position < 0:
            return "趋势跟随"
        elif position > 0:
            return "趋势回归"
    elif segment_type == "上涨分片":
        if position > 0:
            return "趋势跟随"
        elif position < 0:
            return "趋势回归"
    return "未分类"


def _extract_contract(df_path: str) -> str:
    """Extract contract symbol from df_path (e.g. 'BTCUSDT/label_0/df_0.feather')."""
    if not df_path or not isinstance(df_path, str):
        return ""
    parts = Path(df_path).parts
    if len(parts) > 0 and parts[0] != ".":
        return parts[0]
    return ""


def _standardize_pl_df(df: pl.DataFrame) -> pl.DataFrame:
    """Standardize Polars DataFrame column names using HEADER_MAPPING."""
    rename_dict = {}
    for col in df.columns:
        clean_col = col.strip()
        if clean_col in HEADER_MAPPING and HEADER_MAPPING[clean_col] != col:
            rename_dict[col] = HEADER_MAPPING[clean_col]
    return df.rename(rename_dict) if rename_dict else df


def extract_trade_lifecycles(
    df: pl.DataFrame,
    default_epoch: int = 0,
    initial_wallet_balance: float = 100000.0,
) -> list[dict[str, Any]]:
    """Extract trade lifecycles (open to close round-trip) from a detail Polars DataFrame."""
    std_df = _standardize_pl_df(df)
    if std_df.is_empty():
        return []

    # Sort by timestep if present
    if "timestep" in std_df.columns:
        std_df = std_df.sort("timestep")

    # Group by (label, df_path, initial_action, bin_index) to process contiguous market execution slices
    group_cols = [c for c in ["label", "df_path", "initial_action", "bin_index"] if c in std_df.columns]

    if not group_cols:
        groups = [("default", std_df)]
    else:
        groups = std_df.group_by(group_cols, maintain_order=True)

    trade_rows: list[dict[str, Any]] = []

    for _, group in groups:
        rows = group.to_dicts()
        if not rows:
            continue

        active_trade: dict[str, Any] | None = None
        trade_counter = 0

        for row in rows:
            pos_before = float(row.get("position_before") or 0.0)
            pos_after = float(row.get("position_after") or 0.0)
            t_step = int(row.get("timestep") or 0)
            t_stamp = str(row.get("timestamp") or "")
            mark_p = row.get("mark_price")
            close_p = row.get("close")
            price = float(mark_p if mark_p is not None and not np.isnan(mark_p) and mark_p > 0 else (close_p or 0.0))

            realized_pnl = float(row.get("realized_pnl_step") or 0.0)
            commission = float(row.get("commission_fee_step") or 0.0)
            slippage = float(row.get("slippage_step") or 0.0)
            unrealized = float(row.get("unrealized_pnl") or 0.0)
            epoch_val = int(row.get("epoch") or default_epoch)

            label_raw = row.get("label") or "label_0"
            clean_label, segment_type = _parse_label_info(label_raw)
            df_path = str(row.get("df_path") or "")
            contract = _extract_contract(df_path)
            bin_index = row.get("bin_index", 0)
            initial_action = row.get("initial_action", 0)

            is_reversal = (pos_before * pos_after < 0) and (pos_before != 0) and (pos_after != 0)

            # 1. Close existing trade if position returned to 0 or reversed
            if active_trade is not None and (pos_after == 0 or is_reversal):
                active_trade["end_timestep"] = t_step
                active_trade["end_timestamp"] = t_stamp
                active_trade["exit_price"] = price
                active_trade["realized_pnl_sum"] += realized_pnl
                active_trade["commission_fee_sum"] += commission
                active_trade["slippage_sum"] += slippage
                active_trade["holding_duration"] = t_step - active_trade["start_timestep"] + 1

                # Net PnL = realized_pnl - commission
                active_trade["net_pnl"] = (
                    active_trade["realized_pnl_sum"]
                    + active_trade["tail_unrealized_pnl"]
                    - active_trade["commission_fee_sum"]
                )

                # Return rate calculation relative to initial wallet balance
                if initial_wallet_balance > 0:
                    active_trade["return_rate"] = active_trade["net_pnl"] / initial_wallet_balance
                else:
                    active_trade["return_rate"] = 0.0

                trade_rows.append(active_trade)
                active_trade = None

            # 2. Open new trade if starting from 0 or after reversal
            if (pos_before == 0 and pos_after != 0) or is_reversal:
                trade_counter += 1
                direction = "Long" if pos_after > 0 else "Short"
                trend_type = _determine_trend_type(segment_type, pos_after)

                active_trade = {
                    "epoch": epoch_val,
                    "label": clean_label,
                    "bin_index": bin_index,
                    "contract": contract,
                    "df_path": df_path,
                    "initial_action": initial_action,
                    "trade_id": trade_counter,
                    "start_timestep": t_step,
                    "end_timestep": t_step,
                    "start_timestamp": t_stamp,
                    "end_timestamp": t_stamp,
                    "holding_duration": 1,
                    "trade_direction": direction,
                    "segment_type": segment_type,
                    "trend_type": trend_type,
                    "entry_price": price,
                    "total_weighted_price": price * abs(pos_after),
                    "total_position_qty": abs(pos_after),
                    "avg_entry_price": price,
                    "exit_price": price,
                    "initial_pos_abs": abs(pos_after),
                    "realized_pnl_sum": 0.0 if is_reversal else realized_pnl,
                    "tail_unrealized_pnl": 0.0,
                    "commission_fee_sum": 0.0 if is_reversal else commission,
                    "slippage_sum": 0.0 if is_reversal else slippage,
                    "net_pnl": 0.0,
                    "return_rate": 0.0,
                    "max_position_abs": abs(pos_after),
                    "position_abs_list": [abs(pos_after)],
                    "position_change_count": 0,
                    "profitable_increase_count": 0,
                    "unrealized_pnl_peak": max(0.0, unrealized),
                    "max_drawdown": 0.0,
                    "last_unrealized_pnl": unrealized,
                    "is_tail_forced_close": False,
                }
            # 3. Update existing ongoing trade
            elif active_trade is not None:
                active_trade["holding_duration"] = t_step - active_trade["start_timestep"] + 1
                active_trade["realized_pnl_sum"] += realized_pnl
                active_trade["commission_fee_sum"] += commission
                active_trade["slippage_sum"] += slippage
                active_trade["position_abs_list"].append(abs(pos_after))
                active_trade["max_position_abs"] = max(active_trade["max_position_abs"], abs(pos_after))

                # Check position change & profitable same-direction position increase
                if pos_after != pos_before:
                    active_trade["position_change_count"] += 1
                    # Same-direction size increase
                    if abs(pos_after) > abs(pos_before) and (pos_after * pos_before > 0):
                        added_qty = abs(pos_after) - abs(pos_before)
                        active_trade["total_weighted_price"] += price * added_qty
                        active_trade["total_position_qty"] += added_qty
                        active_trade["avg_entry_price"] = (
                            active_trade["total_weighted_price"] / active_trade["total_position_qty"]
                        )
                        if active_trade["last_unrealized_pnl"] > 0:
                            active_trade["profitable_increase_count"] += 1

                # Update drawdown tracking
                active_trade["unrealized_pnl_peak"] = max(active_trade["unrealized_pnl_peak"], unrealized)
                dd = active_trade["unrealized_pnl_peak"] - unrealized
                active_trade["max_drawdown"] = max(active_trade["max_drawdown"], dd)
                active_trade["last_unrealized_pnl"] = unrealized

        # 4. Tail forced close if trade is still active at the end of slice
        if active_trade is not None:
            last_row = rows[-1]
            active_trade["end_timestep"] = int(last_row.get("timestep") or 0)
            active_trade["end_timestamp"] = str(last_row.get("timestamp") or "")
            mark_p = last_row.get("mark_price")
            close_p = last_row.get("close")
            last_price = float(mark_p if mark_p is not None and not np.isnan(mark_p) and mark_p > 0 else (close_p or 0.0))

            active_trade["exit_price"] = last_price
            active_trade["tail_unrealized_pnl"] = float(last_row.get("unrealized_pnl") or 0.0)
            active_trade["is_tail_forced_close"] = True
            active_trade["net_pnl"] = (
                active_trade["realized_pnl_sum"]
                + active_trade["tail_unrealized_pnl"]
                - active_trade["commission_fee_sum"]
            )

            if initial_wallet_balance > 0:
                active_trade["return_rate"] = active_trade["net_pnl"] / initial_wallet_balance
            else:
                active_trade["return_rate"] = 0.0

            trade_rows.append(active_trade)

    # Post-process mean_position_abs
    for tr in trade_rows:
        pos_list = tr.pop("position_abs_list", [tr["max_position_abs"]])
        tr["mean_position_abs"] = float(np.mean(pos_list)) if pos_list else float(tr["max_position_abs"])
        tr.pop("total_weighted_price", None)
        tr.pop("total_position_qty", None)
        tr.pop("initial_pos_abs", None)
        tr.pop("unrealized_pnl_peak", None)
        tr.pop("last_unrealized_pnl", None)

    return trade_rows


def aggregate_summary_statistics(lifecycle_rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Aggregate lifecycle trade rows by [epoch, label, bin_index, trend_type] using Polars."""
    if not lifecycle_rows:
        return pl.DataFrame(schema={
            "epoch": pl.Int64,
            "label": pl.String,
            "bin_index": pl.Int64,
            "trend_type": pl.String,
            "trade_count": pl.Int64,
            "long_trade_count": pl.Int64,
            "short_trade_count": pl.Int64,
            "win_trade_count": pl.Int64,
            "win_rate": pl.Float64,
            "long_win_rate": pl.Float64,
            "short_win_rate": pl.Float64,
            "total_holding_steps": pl.Int64,
            "mean_holding_duration": pl.Float64,
            "total_realized_pnl": pl.Float64,
            "total_commission_fee": pl.Float64,
            "total_slippage": pl.Float64,
            "total_net_pnl": pl.Float64,
            "mean_net_pnl": pl.Float64,
            "pnl_p25": pl.Float64,
            "pnl_p50": pl.Float64,
            "pnl_p75": pl.Float64,
            "profit_factor": pl.Float64,
            "mean_max_position_abs": pl.Float64,
            "total_profitable_increase_count": pl.Int64,
            "mean_max_drawdown": pl.Float64,
            "max_drawdown": pl.Float64,
        })

    df = pl.DataFrame(lifecycle_rows)
    group_cols = ["epoch", "label", "bin_index", "trend_type"]
    if "trade_direction" not in df.columns:
        df = df.with_columns(pl.lit("Long").alias("trade_direction"))
    if "max_drawdown" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("max_drawdown"))
    for col in group_cols:
        if col not in df.columns:
            df = df.with_columns(pl.lit("unknown").alias(col))

    long_filter = pl.col("trade_direction") == "Long"
    short_filter = pl.col("trade_direction") == "Short"
    win_filter = pl.col("net_pnl") > 0

    # Aggregation using Polars expressions
    summary_df = (
        df.group_by(group_cols, maintain_order=True)
        .agg([
            pl.len().alias("trade_count"),
            pl.col("trade_direction").filter(long_filter).len().alias("long_trade_count"),
            pl.col("trade_direction").filter(short_filter).len().alias("short_trade_count"),
            (pl.col("net_pnl") > 0).sum().alias("win_trade_count"),
            ((pl.col("net_pnl") > 0).sum() / pl.len()).alias("win_rate"),
            (
                pl.when(pl.col("trade_direction").filter(long_filter).len() > 0)
                .then(pl.col("net_pnl").filter(long_filter & win_filter).len() / pl.col("trade_direction").filter(long_filter).len())
                .otherwise(0.0)
            ).alias("long_win_rate"),
            (
                pl.when(pl.col("trade_direction").filter(short_filter).len() > 0)
                .then(pl.col("net_pnl").filter(short_filter & win_filter).len() / pl.col("trade_direction").filter(short_filter).len())
                .otherwise(0.0)
            ).alias("short_win_rate"),
            pl.col("holding_duration").sum().alias("total_holding_steps"),
            pl.col("holding_duration").mean().alias("mean_holding_duration"),
            pl.col("realized_pnl_sum").sum().alias("total_realized_pnl"),
            pl.col("commission_fee_sum").sum().alias("total_commission_fee"),
            pl.col("slippage_sum").sum().alias("total_slippage"),
            pl.col("net_pnl").sum().alias("total_net_pnl"),
            pl.col("net_pnl").mean().alias("mean_net_pnl"),
            pl.col("net_pnl").quantile(0.25).alias("pnl_p25"),
            pl.col("net_pnl").quantile(0.50).alias("pnl_p50"),
            pl.col("net_pnl").quantile(0.75).alias("pnl_p75"),
            (
                pl.when(pl.col("net_pnl").filter(pl.col("net_pnl") < 0).abs().sum() > 0)
                .then(pl.col("net_pnl").filter(pl.col("net_pnl") > 0).sum() / pl.col("net_pnl").filter(pl.col("net_pnl") < 0).abs().sum())
                .otherwise(pl.col("net_pnl").filter(pl.col("net_pnl") > 0).sum())
            ).alias("profit_factor"),
            pl.col("max_position_abs").mean().alias("mean_max_position_abs"),
            pl.col("profitable_increase_count").sum().alias("total_profitable_increase_count"),
            pl.col("max_drawdown").mean().alias("mean_max_drawdown"),
            pl.col("max_drawdown").max().alias("max_drawdown"),
        ])
    )

    return summary_df


def select_best_agents(summary_df: pl.DataFrame, min_trades: int = 3) -> dict[str, Any]:
    """Select the best agent (epoch, bin_index) for each (label, trend_type) using Option A.

    Option A Rules:
      1. Filter candidates by (trade_count >= min_trades) & (total_net_pnl > 0) & (profit_factor >= 1.0).
      2. Choose candidate with maximum total_net_pnl.
      3. Fallback: If no candidate satisfies all filters, choose max total_net_pnl among trade_count >= 1.
    """
    manifest: dict[str, Any] = {}
    if summary_df.is_empty():
        return manifest

    dicts = summary_df.to_dicts()
    # Group rows by (label, trend_type)
    by_label_trend: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in dicts:
        lbl = str(row.get("label", ""))
        trd = str(row.get("trend_type", ""))
        key = (lbl, trd)
        if key not in by_label_trend:
            by_label_trend[key] = []
        by_label_trend[key].append(row)

    for (label, trend_type), rows in by_label_trend.items():
        if label not in manifest:
            manifest[label] = {}

        # 1. Filter qualified candidates
        qualified = [
            r for r in rows
            if r.get("trade_count", 0) >= min_trades
            and r.get("total_net_pnl", 0.0) > 0
            and r.get("profit_factor", 0.0) >= 1.0
        ]

        if qualified:
            best_row = max(qualified, key=lambda x: float(x.get("total_net_pnl", 0.0)))
            is_fallback = False
        else:
            # Fallback to trade_count >= 1
            valid_trades = [r for r in rows if r.get("trade_count", 0) >= 1]
            if valid_trades:
                best_row = max(valid_trades, key=lambda x: float(x.get("total_net_pnl", 0.0)))
            else:
                best_row = max(rows, key=lambda x: float(x.get("total_net_pnl", 0.0)))
            is_fallback = True

        manifest[label][trend_type] = {
            "best_epoch": int(best_row.get("epoch", 0)),
            "best_bin_index": int(best_row.get("bin_index", 0)),
            "total_net_pnl": float(best_row.get("total_net_pnl", 0.0)),
            "win_rate": float(best_row.get("win_rate", 0.0)),
            "long_win_rate": float(best_row.get("long_win_rate", 0.0)),
            "short_win_rate": float(best_row.get("short_win_rate", 0.0)),
            "profit_factor": float(best_row.get("profit_factor", 0.0)),
            "trade_count": int(best_row.get("trade_count", 0)),
            "long_trade_count": int(best_row.get("long_trade_count", 0)),
            "short_trade_count": int(best_row.get("short_trade_count", 0)),
            "mean_holding_duration": float(best_row.get("mean_holding_duration", 0.0)),
            "mean_max_drawdown": float(best_row.get("mean_max_drawdown", 0.0)),
            "max_drawdown": float(best_row.get("max_drawdown", 0.0)),
            "is_fallback": is_fallback,
        }

    return manifest


def find_trading_detail_csvs(root_dir: Path) -> list[Path]:
    """Find all trading_action_detail_epoch_*.csv files in root_dir."""
    if not root_dir.exists():
        return []
    detail_files: list[Path] = []
    for path in root_dir.rglob("trading_action_detail_epoch_*.csv"):
        if path.is_file():
            detail_files.append(path)
            print(f"Found detail file: {path}", flush=True)
    detail_files = sorted(detail_files)
    print(f"Total detail files found: {len(detail_files)} in {root_dir}", flush=True)
    return detail_files


def aggregate_agents_indexs(
    result_root: Path,
    output_dir: Path,
    initial_wallet_balance: float = 100000.0,
    min_trades: int = 3,
) -> tuple[Path, Path, Path]:
    """Find detail CSVs, extract lifecycle details, calculate summary statistics, and generate selection_manifest.json using Polars."""
    detail_files = find_trading_detail_csvs(result_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_lifecycles: list[dict[str, Any]] = []
    total_files = len(detail_files)
    print(f"Starting aggregation for {total_files} detail files from {result_root}...", flush=True)

    for idx, file_path in enumerate(detail_files, 1):
        match = re.search(r"epoch_(\d+)", file_path.name)
        default_epoch = int(match.group(1)) if match else 0

        print(f"[{idx}/{total_files}] Processing {file_path}...", flush=True)
        try:
            df = pl.read_csv(file_path)
            lifecycles = extract_trade_lifecycles(
                df,
                default_epoch=default_epoch,
                initial_wallet_balance=initial_wallet_balance,
            )
            all_lifecycles.extend(lifecycles)
            print(f"[{idx}/{total_files}] Processed {file_path}: extracted {len(lifecycles)} lifecycles", flush=True)
        except Exception as err:
            print(f"[{idx}/{total_files}] Warning: failed to process {file_path}: {err}", flush=True)

    if all_lifecycles:
        detail_df = pl.DataFrame(all_lifecycles)
    else:
        detail_df = pl.DataFrame()

    summary_df = aggregate_summary_statistics(all_lifecycles)
    selection_manifest = select_best_agents(summary_df, min_trades=min_trades)

    detail_csv_path = output_dir / "agent_trade_lifecycle_detail.csv"
    summary_csv_path = output_dir / "agent_trend_type_summary.csv"
    manifest_json_path = output_dir / "selection_manifest.json"

    if not detail_df.is_empty():
        renamed_detail = detail_df.rename({k: v for k, v in LIFECYCLE_HEADER_LABELS.items() if k in detail_df.columns})
        renamed_detail.write_csv(detail_csv_path)
    else:
        pl.DataFrame().write_csv(detail_csv_path)

    if not summary_df.is_empty():
        renamed_summary = summary_df.rename({k: v for k, v in SUMMARY_HEADER_LABELS.items() if k in summary_df.columns})
        renamed_summary.write_csv(summary_csv_path)
    else:
        pl.DataFrame().write_csv(summary_csv_path)

    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(selection_manifest, f, indent=2, ensure_ascii=False)

    print(f"Generated trade lifecycle detail CSV: {detail_csv_path} ({len(detail_df)} rows)", flush=True)
    print(f"Generated trend type summary CSV: {summary_csv_path} ({len(summary_df)} rows)", flush=True)
    print(f"Generated selection manifest JSON: {manifest_json_path}", flush=True)

    return detail_csv_path, summary_csv_path, manifest_json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate agent trading details into trend lifecycle metrics and selection_manifest using Polars.")
    parser.add_argument("--result_root", type=Path, default=Path("result/DiHFT/low_level"), help="Root directory searching for detail CSVs")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory to save summary, detail CSVs, and selection_manifest.json")
    parser.add_argument("--initial_wallet_balance", type=float, default=100000.0, help="Initial wallet balance used for return_rate calculation")
    parser.add_argument("--min_trades", type=int, default=3, help="Minimum trade count threshold for agent selection filter")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(args, flush=True)
    aggregate_agents_indexs(
        args.result_root,
        args.output_dir,
        initial_wallet_balance=args.initial_wallet_balance,
        min_trades=args.min_trades,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())