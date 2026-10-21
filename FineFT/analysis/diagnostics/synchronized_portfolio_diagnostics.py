"""Time-Synchronized Multi-Contract Portfolio Diagnostics and Action Timeline Tracking.

This module aligns multi-contract trading histories along a unified timestamp axis,
evaluating the combined portfolio where trading is strictly constrained to the
dynamic daily main and sub-main contracts (at most 2 active contracts at any time),
with graceful downgrade on non-main contracts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.calculate_metric.calculate_metric import calculate_metric, assign_trading_days_from_timestamps
from common import ArtifactNames, HistoryArtifactNames

logger = logging.getLogger(__name__)


@dataclass
class ContractMetricSummary:
    contract: str
    initial_capital: float
    gross_pnl: float
    net_pnl: float
    commission: float
    tr: float
    mdd: float
    annual_sr: float | None
    daily_cr: float | None
    daily_sor: float | None
    daily_vol: float | None
    trades: int
    turnover_volume: float
    turnover_ratio: float
    long_pct: float
    short_pct: float
    flat_pct: float
    total_steps: int
    main_role_steps: int
    sub_role_steps: int
    non_main_steps: int
    trading_days_count: int = 0
    daily_mdd: float | None = None


@dataclass
class PortfolioMetricSummary:
    initial_capital: float
    total_gross_pnl: float
    total_net_pnl: float
    total_commission: float
    tr: float
    mdd: float
    annual_sr: float | None
    daily_cr: float | None
    daily_sor: float | None
    daily_vol: float | None
    total_trades: int
    total_turnover_volume: float
    turnover_multiple: float
    max_concurrent_contracts: int
    mean_concurrent_contracts: float
    total_timeline_steps: int
    eval_stage: str
    trading_days_count: int = 0
    daily_mdd: float | None = None


class SynchronizedPortfolioCalculator:
    """Calculates time-synchronized portfolio performance and timeline records."""

    def __init__(
        self,
        result_dir: str | Path,
        data_dir: str | Path,
        output_dir: str | Path | None = None,
        single_contract_capital: float = 6000.0,
        contract_unit: float = 1.0,
        commission_rate: float = 0.0003,
        freq: int = 12,
        eval_stage: str = "valid",
    ) -> None:
        self.result_dir = Path(result_dir)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.result_dir
        self.single_contract_capital = float(single_contract_capital)
        self.contract_unit = float(contract_unit)
        self.commission_rate = float(commission_rate)
        self.freq = int(freq)
        self.eval_stage = eval_stage
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_contract_dirs(self) -> list[tuple[str, Path]]:
        contracts_root = self.result_dir / "contracts"
        if not contracts_root.is_dir():
            return []
        subdirs = [d for d in sorted(contracts_root.iterdir()) if d.is_dir()]
        matched = []
        for d in subdirs:
            feather_path = self.data_dir / f"{d.name}.feather"
            csv_path = self.data_dir / f"{d.name}.csv"
            if feather_path.is_file() or csv_path.is_file():
                matched.append((d.name, d))
        return matched

    def _load_contract_data(
        self, contract_name: str, contract_path: Path
    ) -> tuple[pd.DataFrame, ContractMetricSummary]:
        feather_path = self.data_dir / f"{contract_name}.feather"
        if feather_path.is_file():
            df_market = pd.read_feather(feather_path)
        else:
            df_market = pd.read_csv(self.data_dir / f"{contract_name}.csv")

        steps = len(df_market) - 1
        timestamps = df_market["timestamp"].iloc[:-1].reset_index(drop=True)
        role_tiers = (
            df_market["prev_day_contract_role_tier"].iloc[:-1].to_numpy()
            if "prev_day_contract_role_tier" in df_market.columns
            else np.zeros(steps, dtype=float)
        )
        mark_prices = (
            df_market["mark_price"].to_numpy()
            if "mark_price" in df_market.columns
            else df_market["close"].to_numpy()
        )
        price_diffs = np.diff(mark_prices)

        pos_file = contract_path / HistoryArtifactNames.EXECUTED_POSITION_HISTORY_NPY
        if pos_file.is_file():
            positions = np.load(pos_file)[:steps].astype(float)
        else:
            positions = np.zeros(steps, dtype=float)

        rew_file = contract_path / HistoryArtifactNames.REWARD_HISTORY_NPY
        if rew_file.is_file():
            rewards = np.load(rew_file)[:steps].astype(float)
        else:
            rewards = np.zeros(steps, dtype=float)

        mic_file = contract_path / HistoryArtifactNames.MICRO_ACTION_HISTORY_NPY
        micro_actions = (
            np.load(mic_file)[:steps].astype(int)
            if mic_file.is_file()
            else np.zeros(steps, dtype=int)
        )

        mac_file = contract_path / HistoryArtifactNames.MACRO_ACTION_HISTORY_NPY
        if mac_file.is_file():
            loaded_mac = np.load(mac_file).astype(int)
            if len(loaded_mac) < steps:
                macro_actions = np.r_[np.full(steps - len(loaded_mac), 9, dtype=int), loaded_mac]
            else:
                macro_actions = loaded_mac[:steps]
        else:
            macro_actions = np.full(steps, -1, dtype=int)

        rea_file = contract_path / HistoryArtifactNames.ACTION_DECISION_REASON_HISTORY_NPY
        if rea_file.is_file():
            loaded_rea = np.load(rea_file).astype(int)
            if len(loaded_rea) < steps:
                reasons = np.r_[np.full(steps - len(loaded_rea), 3, dtype=int), loaded_rea]
            else:
                reasons = loaded_rea[:steps]
        else:
            reasons = np.zeros(steps, dtype=int)

        fee_file = contract_path / HistoryArtifactNames.COMMISSION_FEE_HISTORY_NPY
        pos_diffs = np.abs(np.diff(positions, prepend=0.0))
        if fee_file.is_file():
            commissions = np.load(fee_file)[:steps].astype(float) * self.contract_unit
        else:
            commissions = (
                pos_diffs * mark_prices[:-1] * self.contract_unit * self.commission_rate
            )

        gross_pnl = positions * price_diffs * self.contract_unit
        net_pnl = rewards * self.contract_unit

        trade_indices = np.where(pos_diffs > 0)[0]
        trades = int(len(trade_indices))
        turnover_volume = float(
            sum(
                pos_diffs[idx] * mark_prices[idx] * self.contract_unit
                for idx in trade_indices
            )
        )
        turnover_ratio = (
            float(turnover_volume / self.single_contract_capital)
            if self.single_contract_capital > 0
            else 0.0
        )

        trading_days = assign_trading_days_from_timestamps(timestamps)

        if steps >= self.freq:
            tr, daily_vol, mdd, downside_dev, annual_sr, daily_cr, daily_sor = (
                calculate_metric(
                    self.single_contract_capital,
                    np.r_[0.0, net_pnl],
                    freq=self.freq,
                    trading_days=trading_days.to_numpy(),
                    annual_trading_days=242,
                )
            )
            df_contract_daily = pd.DataFrame(
                {"td": trading_days, "equity": self.single_contract_capital + np.cumsum(net_pnl)}
            )
            c_daily_close = df_contract_daily.groupby("td", sort=False)["equity"].last().to_numpy()
            c_daily_seq = np.r_[self.single_contract_capital, c_daily_close]
            c_daily_peaks = np.maximum.accumulate(c_daily_seq)
            contract_daily_mdd = (
                float(np.max((c_daily_peaks - c_daily_seq) / (c_daily_peaks + 1e-12)))
                if len(c_daily_seq) > 0
                else 0.0
            )
            contract_td_count = int(trading_days.nunique())
        else:
            equity = self.single_contract_capital + np.cumsum(net_pnl)
            tr = float((equity[-1] - equity[0]) / (equity[0] + 1e-12))
            peaks = np.maximum.accumulate(equity)
            mdd = float(np.max((peaks - equity) / (peaks + 1e-12)))
            daily_vol = annual_sr = daily_cr = daily_sor = None
            contract_daily_mdd = None
            contract_td_count = int(trading_days.nunique()) if len(trading_days) > 0 else 0

        long_pct = float((positions > 0).mean() * 100.0) if steps > 0 else 0.0
        short_pct = float((positions < 0).mean() * 100.0) if steps > 0 else 0.0
        flat_pct = float((positions == 0).mean() * 100.0) if steps > 0 else 0.0

        contract_summary = ContractMetricSummary(
            contract=contract_name,
            initial_capital=self.single_contract_capital,
            gross_pnl=float(np.sum(gross_pnl)),
            net_pnl=float(np.sum(net_pnl)),
            commission=float(np.sum(commissions)),
            tr=float(tr),
            mdd=float(mdd),
            annual_sr=float(annual_sr) if annual_sr is not None else None,
            daily_cr=float(daily_cr) if daily_cr is not None else None,
            daily_sor=float(daily_sor) if daily_sor is not None else None,
            daily_vol=float(daily_vol) if daily_vol is not None else None,
            trades=trades,
            turnover_volume=turnover_volume,
            turnover_ratio=turnover_ratio,
            long_pct=long_pct,
            short_pct=short_pct,
            flat_pct=flat_pct,
            total_steps=steps,
            main_role_steps=int((role_tiers == 1.0).sum()),
            sub_role_steps=int((role_tiers == 0.5).sum()),
            non_main_steps=int((role_tiers < 0.5).sum()),
            trading_days_count=contract_td_count,
            daily_mdd=contract_daily_mdd,
        )

        step_df = pd.DataFrame(
            {
                "contract": contract_name,
                "timestamp": timestamps,
                "trading_day": trading_days,
                "role_tier": role_tiers,
                "mark_price": mark_prices[:-1],
                "micro_action": micro_actions,
                "macro_action": macro_actions,
                "reason": reasons,
                "position": positions,
                "pos_diff": pos_diffs,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "commission": commissions,
            }
        )

        return step_df, contract_summary

    def run(self) -> dict[str, Any]:
        contract_targets = self._resolve_contract_dirs()
        logger.info(
            "Found %d contract directories to synchronize in %s",
            len(contract_targets),
            self.result_dir,
        )

        all_step_dfs: list[pd.DataFrame] = []
        contract_summaries: list[ContractMetricSummary] = []

        for c_name, c_path in contract_targets:
            step_df, c_summary = self._load_contract_data(c_name, c_path)
            all_step_dfs.append(step_df)
            contract_summaries.append(c_summary)

        if not all_step_dfs:
            logger.warning("No valid contract step data found to synchronize.")
            return {}

        combined_df = pd.concat(all_step_dfs, ignore_index=True)
        unique_timestamps = sorted(combined_df["timestamp"].unique())
        logger.info(
            "Synthesizing synchronized timeline across %d global timestamps...",
            len(unique_timestamps),
        )

        timeline_records = []
        portfolio_initial_capital = self.single_contract_capital * 2.0
        cum_net_pnl = 0.0

        for ts in unique_timestamps:
            sub_step = combined_df[combined_df["timestamp"] == ts]

            # Main contract (tier = 1.0)
            main_row = sub_step[sub_step["role_tier"] == 1.0]
            main_contract = main_row["contract"].iloc[0] if len(main_row) > 0 else None
            main_action = int(main_row["micro_action"].iloc[0]) if len(main_row) > 0 else None
            main_macro = int(main_row["macro_action"].iloc[0]) if len(main_row) > 0 else None
            main_position = float(main_row["position"].iloc[0]) if len(main_row) > 0 else 0.0

            # Sub-main contract (tier = 0.5)
            sub_row = sub_step[sub_step["role_tier"] == 0.5]
            sub_contract = sub_row["contract"].iloc[0] if len(sub_row) > 0 else None
            sub_action = int(sub_row["micro_action"].iloc[0]) if len(sub_row) > 0 else None
            sub_macro = int(sub_row["macro_action"].iloc[0]) if len(sub_row) > 0 else None
            sub_position = float(sub_row["position"].iloc[0]) if len(sub_row) > 0 else 0.0

            # Active holding contracts
            holding_rows = sub_step[sub_step["position"].abs() > 1e-6]
            holding_contracts = holding_rows["contract"].tolist()
            holding_count = len(holding_contracts)

            step_gross = float(sub_step["gross_pnl"].sum())
            step_net = float(sub_step["net_pnl"].sum())
            step_fee = float(sub_step["commission"].sum())
            cum_net_pnl += step_net

            timeline_records.append(
                {
                    "timestamp": str(ts),
                    "main_contract": main_contract,
                    "main_action": main_action,
                    "main_macro_action": main_macro,
                    "main_position": main_position,
                    "sub_contract": sub_contract,
                    "sub_action": sub_action,
                    "sub_macro_action": sub_macro,
                    "sub_position": sub_position,
                    "holding_contract_count": holding_count,
                    "holding_contracts": ",".join(holding_contracts),
                    "step_gross_pnl": step_gross,
                    "step_net_pnl": step_net,
                    "step_commission": step_fee,
                    "portfolio_equity": portfolio_initial_capital + cum_net_pnl,
                }
            )

        df_timeline = pd.DataFrame(timeline_records)
        portfolio_trading_days = assign_trading_days_from_timestamps(df_timeline["timestamp"])
        df_timeline["trading_day"] = portfolio_trading_days

        # Calculate portfolio-level metrics
        step_net_pnls = df_timeline["step_net_pnl"].to_numpy()
        steps = len(df_timeline)

        if steps >= self.freq:
            p_tr, p_daily_vol, p_mdd, p_downside, p_annual_sr, p_daily_cr, p_daily_sor = (
                calculate_metric(
                    portfolio_initial_capital,
                    np.r_[0.0, step_net_pnls],
                    freq=self.freq,
                    trading_days=portfolio_trading_days.to_numpy(),
                    annual_trading_days=242,
                )
            )
            df_port_daily = pd.DataFrame(
                {"td": portfolio_trading_days, "equity": df_timeline["portfolio_equity"].to_numpy()}
            )
            p_daily_close = df_port_daily.groupby("td", sort=False)["equity"].last().to_numpy()
            p_daily_seq = np.r_[portfolio_initial_capital, p_daily_close]
            p_daily_peaks = np.maximum.accumulate(p_daily_seq)
            p_daily_mdd = (
                float(np.max((p_daily_peaks - p_daily_seq) / (p_daily_peaks + 1e-12)))
                if len(p_daily_seq) > 0
                else 0.0
            )
            p_td_count = int(portfolio_trading_days.nunique())
        else:
            equities = df_timeline["portfolio_equity"].to_numpy()
            p_tr = float((equities[-1] - equities[0]) / (equities[0] + 1e-12))
            peaks = np.maximum.accumulate(equities)
            p_mdd = float(np.max((peaks - equities) / (peaks + 1e-12)))
            p_daily_vol = p_annual_sr = p_daily_cr = p_daily_sor = None
            p_daily_mdd = None
            p_td_count = int(portfolio_trading_days.nunique()) if len(portfolio_trading_days) > 0 else 0

        total_trades = sum(c.trades for c in contract_summaries)
        total_turnover = sum(c.turnover_volume for c in contract_summaries)
        max_concurrent = int(df_timeline["holding_contract_count"].max()) if steps > 0 else 0
        mean_concurrent = float(df_timeline["holding_contract_count"].mean()) if steps > 0 else 0.0

        portfolio_summary = PortfolioMetricSummary(
            initial_capital=portfolio_initial_capital,
            total_gross_pnl=float(df_timeline["step_gross_pnl"].sum()),
            total_net_pnl=float(df_timeline["step_net_pnl"].sum()),
            total_commission=float(df_timeline["step_commission"].sum()),
            tr=float(p_tr),
            mdd=float(p_mdd),
            annual_sr=float(p_annual_sr) if p_annual_sr is not None else None,
            daily_cr=float(p_daily_cr) if p_daily_cr is not None else None,
            daily_sor=float(p_daily_sor) if p_daily_sor is not None else None,
            daily_vol=float(p_daily_vol) if p_daily_vol is not None else None,
            total_trades=total_trades,
            total_turnover_volume=total_turnover,
            turnover_multiple=(
                float(total_turnover / portfolio_initial_capital)
                if portfolio_initial_capital > 0
                else 0.0
            ),
            max_concurrent_contracts=max_concurrent,
            mean_concurrent_contracts=mean_concurrent,
            total_timeline_steps=steps,
            eval_stage=self.eval_stage,
            trading_days_count=p_td_count,
            daily_mdd=p_daily_mdd,
        )

        # Save artifacts
        timeline_path = self.output_dir / ArtifactNames.PORTFOLIO_ACTION_TIMELINE_CSV
        df_timeline.to_csv(timeline_path, index=False)
        logger.info("[Artifacts] Saved timeline to %s", timeline_path)

        contracts_data = [asdict(c) for c in contract_summaries]
        portfolio_data = asdict(portfolio_summary)

        full_summary_json = {
            "portfolio": portfolio_data,
            "contracts": contracts_data,
        }

        json_path = self.output_dir / ArtifactNames.SUMMARY_METRICS_JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(full_summary_json, f, indent=2, ensure_ascii=False)
        logger.info("[Artifacts] Saved summary JSON to %s", json_path)

        # Flat CSV summary
        df_contracts = pd.DataFrame(contracts_data)
        csv_path = self.output_dir / ArtifactNames.SUMMARY_METRICS_CSV
        df_contracts.to_csv(csv_path, index=False)
        logger.info("[Artifacts] Saved summary CSV to %s", csv_path)

        # Markdown report
        md_path = self.output_dir / ArtifactNames.PORTFOLIO_SUMMARY_MD
        self._write_markdown_summary(md_path, portfolio_summary, contract_summaries)
        logger.info("[Artifacts] Saved summary Markdown to %s", md_path)

        self._print_terminal_summary(portfolio_summary, contract_summaries)

        return full_summary_json

    def _write_markdown_summary(
        self,
        md_path: Path,
        p: PortfolioMetricSummary,
        contracts: list[ContractMetricSummary],
    ) -> None:
        sr_text = f"{p.annual_sr:.4f}" if p.annual_sr is not None else "N/A"
        cr_text = f"{p.daily_cr:.4f}" if p.daily_cr is not None else "N/A"
        sor_text = f"{p.daily_sor:.4f}" if p.daily_sor is not None else "N/A"
        daily_mdd_text = f"{p.daily_mdd * 100.0:.2f}%" if p.daily_mdd is not None else "N/A"

        lines = [
            "# 双主力动态时间同步投资组合回测总结报告",
            "",
            f"- **评测阶段 (Eval Stage)**: `{p.eval_stage}`",
            f"- **初始组合本金 (Initial Capital)**: `¥{p.initial_capital:,.2f}` (两合约各 ¥{self.single_contract_capital:,.2f})",
            f"- **组合累计净收益 (Net PnL)**: `¥{p.total_net_pnl:+,.2f}`",
            f"- **组合累计总收益率 (Total Return / TR)**: **`{p.tr * 100.0:+.2f}%`**",
            f"- **步级高水位最大回撤 (Step MDD)**: **`{p.mdd * 100.0:.2f}%`** | **日末净值最大回撤 (Daily Close MDD)**: **`{daily_mdd_text}`**",
            f"- **真实交易日天数 (TradingDays)**: `{p.trading_days_count}` 天 (年化基准: 242 交易日)",
            f"- **组合年化夏普比率 (242-Day Annualized Sharpe Ratio)**: **`{sr_text}`**",
            f"- **组合卡玛比率 (Calmar Ratio)**: `{cr_text}` | **索提诺比率 (Sortino Ratio)**: `{sor_text}`",
            f"- **总交易次数 (Total Trades)**: `{p.total_trades}` 次 | **名义换手额**: `¥{p.total_turnover_volume:,.2f}` ({p.turnover_multiple:.1f}x)",
            f"- **总手续费摩擦 (Commission Fee)**: `¥{p.total_commission:,.2f}`",
            f"- **最大并发持仓合约数 (Max Concurrent Contracts)**: **`{p.max_concurrent_contracts}`** (平均: `{p.mean_concurrent_contracts:.2f}`)",
            f"- **时间步总数 (Timeline Steps)**: `{p.total_timeline_steps}` 根 10min Bar",
            "",
            "---",
            "",
            f"## 逐合约独立收益与流动性角色统计 (单合约本金 ¥{self.single_contract_capital:,.2f})",
            "",
            "| 合约代码 | TR (%) | Step MDD (%) | Daily MDD (%) | Annual SR (242d) | Net PnL (元) | 手续费 (元) | 交易次数 | 交易日数 | 主力步数 | 次主力步数 | 非主力步数 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for c in contracts:
            c_sr = f"{c.annual_sr:.2f}" if c.annual_sr is not None else "N/A"
            c_dmdd = f"{c.daily_mdd * 100.0:.2f}%" if c.daily_mdd is not None else "N/A"
            lines.append(
                f"| `{c.contract}` | {c.tr * 100.0:+.2f}% | {c.mdd * 100.0:.2f}% | {c_dmdd} | {c_sr} | "
                f"{c.net_pnl:+,.1f} | {c.commission:,.1f} | {c.trades} | {c.trading_days_count} | "
                f"{c.main_role_steps} | {c.sub_role_steps} | {c.non_main_steps} |"
            )

        md_path.write_text("\n".join(lines), encoding="utf-8")

    def _print_terminal_summary(
        self,
        p: PortfolioMetricSummary,
        contracts: list[ContractMetricSummary],
    ) -> None:
        sr_text = f"{p.annual_sr:.4f}" if p.annual_sr is not None else "N/A"
        daily_mdd_text = f"{p.daily_mdd * 100.0:.4f}%" if p.daily_mdd is not None else "N/A"
        print("=" * 80)
        print("🎉 [双主力动态时间同步组合回测完成 (Synchronized Dual-Main Portfolio)]")
        print(f"  评测阶段: {p.eval_stage}")
        print(f"  初始本金: ¥{p.initial_capital:,.2f} (单合约基准: ¥{self.single_contract_capital:,.2f})")
        print(f"  净收益额: ¥{p.total_net_pnl:+,.2f}")
        print(f"  累计收益率 (TR): {p.tr * 100.0:+.4f}%")
        print(f"  步级最大回撤 (Step MDD): {p.mdd * 100.0:.4f}% | 日末最大回撤 (Daily MDD): {daily_mdd_text}")
        print(f"  真实交易日: {p.trading_days_count} 天 | 年化夏普 (242d Sharpe): {sr_text}")
        print(f"  最大瞬时持仓合约数: {p.max_concurrent_contracts} (平均: {p.mean_concurrent_contracts:.2f})")
        print(f"  总调仓次数: {p.total_trades} 次 | 总手续费: ¥{p.total_commission:,.2f}")
        print("-" * 80)
        print(f"{'合约':<10} {'TR(%)':>10} {'StepMDD(%)':>12} {'DailyMDD(%)':>12} {'Sharpe(242d)':>14} {'Net PnL':>12} {'Trades':>8} {'主力/次主力/非主力步数':>22}")
        for c in contracts:
            c_sr = f"{c.annual_sr:.2f}" if c.annual_sr is not None else "N/A"
            c_dmdd = f"{c.daily_mdd * 100.0:.2f}%" if c.daily_mdd is not None else "N/A"
            steps_str = f"{c.main_role_steps}/{c.sub_role_steps}/{c.non_main_steps}"
            print(f"{c.contract:<10} {c.tr * 100.0:>9.2f}% {c.mdd * 100.0:>11.2f}% {c_dmdd:>11} {c_sr:>13} {c.net_pnl:>12.2f} {c.trades:>8} {steps_str:>22}")
        print("=" * 80)
