"""Comprehensive Trading Diagnostics and Friction Attribution Engine.

Decomposes strategy returns into gross market timing PnL, commission fees,
and order book slippage, benchmarks against 1x/5x Buy & Hold, and profiles
micro-posture, macro VAE routing, and risk-control events (ADR 0045/0047/0048).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.append(".")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from analysis.calculate_metric.calculate_metric import calculate_metric
from common import ActionDecisionReasons, HistoryArtifactNames

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("TradingDiagnostics")


@dataclass
class ContractDiagnostics:
    contract: str
    period: str
    steps: int
    p0: float
    p_end: float
    # Benchmark
    bh_pts: float
    bh_tr: float
    bh_5x_tr: float
    bh_pnl: float
    bh_5x_pnl: float
    # Strategy PnL & Friction
    gross_pts: float
    gross_pnl: float
    net_pnl: float
    friction: float
    friction_ratio: float
    trades: int
    turnover_volume: float
    turnover_ratio: float
    est_commission: float
    est_slippage: float
    req_money: float
    tr: float
    alpha_vs_bh: float
    alpha_vs_bh_5x: float
    # Risk-adjusted
    annual_sr: float
    mdd: float
    daily_cr: float
    daily_sor: float
    daily_vol: float
    # Micro posture
    long_pct: float
    short_pct: float
    flat_pct: float
    mean_holding_bars: float
    median_holding_bars: float
    # Risk controls
    policy_inference_steps: int
    action_persistence_steps: int
    defensive_preemptions: int
    defensive_rule_closes: int
    hard_stop_losses: int
    stop_loss_cooldown_steps: int
    circuit_breaker_suspension_steps: int
    # Macro routing slots
    macro_counts: dict[str, int]


class TradingDiagnosticsCalculator:
    def __init__(
        self,
        result_dir: str,
        data_dir: str,
        output_dir: str | None = None,
        initial_wallet_balance: float = 6000.0,
        commission_rate: float = 0.0003,
        contract_unit: float = 1.0,
        freq: int = 12,
    ) -> None:
        self.result_dir = Path(result_dir)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.result_dir / "diagnostics"
        self.initial_wallet_balance = initial_wallet_balance
        self.commission_rate = commission_rate
        self.contract_unit = contract_unit
        self.freq = freq

    def _resolve_contract_dirs(self) -> list[tuple[str, Path]]:
        contracts_root = self.result_dir / "contracts"
        if contracts_root.is_dir():
            subdirs = [d for d in sorted(contracts_root.iterdir()) if d.is_dir()]
            if subdirs:
                return [(d.name, d) for d in subdirs]
        return [(self.result_dir.name, self.result_dir)]

    def _calculate_holding_durations(self, micro_actions: np.ndarray) -> tuple[float, float]:
        if len(micro_actions) == 0:
            return 0.0, 0.0
        diffs = np.diff(micro_actions)
        durations = []
        curr_len = 1
        for d in diffs:
            if d == 0:
                curr_len += 1
            else:
                durations.append(curr_len)
                curr_len = 1
        durations.append(curr_len)
        return float(np.mean(durations)), float(np.median(durations))

    def evaluate_contract(self, contract_name: str, contract_path: Path) -> ContractDiagnostics:
        # Load feather
        feather_file = self.data_dir / f"{contract_name}.feather"
        if not feather_file.is_file():
            # Try finding single file if data_dir is a file or contains one feather
            feathers = list(self.data_dir.glob("*.feather"))
            if len(feathers) == 1:
                feather_file = feathers[0]
            else:
                raise FileNotFoundError(f"Market feather data not found for contract {contract_name} in {self.data_dir}")

        df = pd.read_feather(feather_file)
        reward_history = np.load(contract_path / HistoryArtifactNames.REWARD_HISTORY_NPY)
        total_asset_history = np.load(contract_path / HistoryArtifactNames.TOTAL_ASSET_HISTORY_NPY)
        micro_actions = np.load(contract_path / HistoryArtifactNames.MICRO_ACTION_HISTORY_NPY)

        macro_file = contract_path / HistoryArtifactNames.MACRO_ACTION_HISTORY_NPY
        if not macro_file.is_file():
            macro_file = contract_path / HistoryArtifactNames.MACRO_ACTION_NPY
        macro_actions = np.load(macro_file) if macro_file.is_file() else np.array([])

        reason_file = contract_path / HistoryArtifactNames.ACTION_DECISION_REASON_HISTORY_NPY
        reasons = np.load(reason_file) if reason_file.is_file() else np.array([])

        steps = len(micro_actions)
        mark_prices = df["mark_price"].values[:steps]
        p0 = float(mark_prices[0])
        p_end = float(mark_prices[-1])
        start_time = str(df["timestamp"].iloc[0])[:10]
        end_time = str(df["timestamp"].iloc[-1])[:10]
        period_str = f"{start_time} ~ {end_time}"

        # 1. Benchmark
        bh_pts = float(p_end - p0)
        bh_tr = float((p_end / p0) - 1.0)
        bh_5x_tr = float(bh_tr * 5.0)
        bh_pnl = float(bh_pts * self.contract_unit)
        bh_5x_pnl = float(bh_pnl * 5.0)

        # 2. Timing Gross PnL
        price_diffs = np.diff(mark_prices)
        gross_pts = 0.0
        for i in range(len(price_diffs)):
            pos = 1 if micro_actions[i] == 2 else (-1 if micro_actions[i] == 0 else 0)
            if pos == 1:
                gross_pts += float(price_diffs[i])
            elif pos == -1:
                gross_pts -= float(price_diffs[i])

        gross_pnl = float(gross_pts * self.contract_unit)
        scaled_reward_history = reward_history * self.contract_unit
        net_pnl = float(np.sum(scaled_reward_history))
        friction = float(gross_pnl - net_pnl)
        friction_ratio = float((friction / gross_pnl * 100.0) if gross_pnl > 0 else 0.0)

        # 3. Turnover & Trades
        pos_diffs = np.abs(np.diff(micro_actions))
        trade_indices = np.where(pos_diffs > 0)[0]
        trades = int(len(trade_indices))
        turnover_volume = float(
            sum(pos_diffs[idx] * mark_prices[idx] * self.contract_unit for idx in trade_indices)
        )

        req_money = float(total_asset_history[0]) if len(total_asset_history) > 0 and float(total_asset_history[0]) > 0 else self.initial_wallet_balance
        turnover_ratio = float(turnover_volume / req_money) if req_money > 0 else 0.0
        est_commission = float(turnover_volume * self.commission_rate)
        est_slippage = float(friction - est_commission)

        # 4. Financial Metrics
        freq_calc = self.freq if steps >= 24 else max(1, steps // 2)
        tr, daily_vol, mdd, downside_dev, annual_sr, daily_cr, daily_sor = calculate_metric(
            req_money, scaled_reward_history, freq=freq_calc
        )

        alpha_vs_bh = float(tr - bh_tr)
        alpha_vs_bh_5x = float(tr - bh_5x_tr)

        # 5. Micro Posture
        micro_cnt = Counter(micro_actions.tolist())
        long_pct = float(micro_cnt.get(2, 0) / steps * 100.0) if steps > 0 else 0.0
        short_pct = float(micro_cnt.get(0, 0) / steps * 100.0) if steps > 0 else 0.0
        flat_pct = float(micro_cnt.get(1, 0) / steps * 100.0) if steps > 0 else 0.0
        mean_hold, med_hold = self._calculate_holding_durations(micro_actions)

        # 6. Risk Controls
        reason_cnt = Counter(reasons.tolist())
        pol_inf = int(reason_cnt.get(ActionDecisionReasons.POLICY_INFERENCE.value, 0))
        act_pers = int(reason_cnt.get(ActionDecisionReasons.ACTION_PERSISTENCE.value, 0))
        def_preempt = int(reason_cnt.get(ActionDecisionReasons.DEFENSIVE_PREEMPTION.value, 0))
        def_rule = int(reason_cnt.get(ActionDecisionReasons.DEFENSIVE_RULE_CLOSE.value, 0))
        hard_stop = int(reason_cnt.get(ActionDecisionReasons.HARD_STOP_LOSS.value, 0))
        stop_cool = int(reason_cnt.get(ActionDecisionReasons.STOP_LOSS_COOLDOWN.value, 0))
        circ_brk = int(reason_cnt.get(ActionDecisionReasons.CIRCUIT_BREAKER_SUSPENSION.value, 0))

        # 7. Macro Routing
        macro_cnt = Counter(macro_actions.tolist())
        macro_dict = {f"Slot_{s}": int(macro_cnt.get(s, 0)) for s in range(10)}

        return ContractDiagnostics(
            contract=contract_name,
            period=period_str,
            steps=steps,
            p0=p0,
            p_end=p_end,
            bh_pts=bh_pts,
            bh_tr=bh_tr,
            bh_5x_tr=bh_5x_tr,
            bh_pnl=bh_pnl,
            bh_5x_pnl=bh_5x_pnl,
            gross_pts=gross_pts,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            friction=friction,
            friction_ratio=friction_ratio,
            trades=trades,
            turnover_volume=turnover_volume,
            turnover_ratio=turnover_ratio,
            est_commission=est_commission,
            est_slippage=est_slippage,
            req_money=req_money,
            tr=float(tr),
            alpha_vs_bh=alpha_vs_bh,
            alpha_vs_bh_5x=alpha_vs_bh_5x,
            annual_sr=float(annual_sr),
            mdd=float(mdd),
            daily_cr=float(daily_cr),
            daily_sor=float(daily_sor),
            daily_vol=float(daily_vol),
            long_pct=long_pct,
            short_pct=short_pct,
            flat_pct=flat_pct,
            mean_holding_bars=mean_hold,
            median_holding_bars=med_hold,
            policy_inference_steps=pol_inf,
            action_persistence_steps=act_pers,
            defensive_preemptions=def_preempt,
            defensive_rule_closes=def_rule,
            hard_stop_losses=hard_stop,
            stop_loss_cooldown_steps=stop_cool,
            circuit_breaker_suspension_steps=circ_brk,
            macro_counts=macro_dict,
        )

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        contract_targets = self._resolve_contract_dirs()
        logger.info("Found %d contract targets to diagnose in %s", len(contract_targets), self.result_dir)

        contract_diagnostics: list[ContractDiagnostics] = []
        for c_name, c_path in contract_targets:
            diag = self.evaluate_contract(c_name, c_path)
            contract_diagnostics.append(diag)

        # Build DataFrames
        df_pnl = pd.DataFrame([
            {
                "contract": d.contract,
                "period": d.period,
                "steps": d.steps,
                "p0": d.p0,
                "p_end": d.p_end,
                "bh_tr_pct": d.bh_tr * 100.0,
                "bh_5x_tr_pct": d.bh_5x_tr * 100.0,
                "gross_pnl": d.gross_pnl,
                "net_pnl": d.net_pnl,
                "friction": d.friction,
                "friction_ratio_pct": d.friction_ratio,
                "trades": d.trades,
                "turnover_volume": d.turnover_volume,
                "turnover_ratio": d.turnover_ratio,
                "est_commission": d.est_commission,
                "est_slippage": d.est_slippage,
                "dihft_tr_pct": d.tr * 100.0,
                "annual_sr": d.annual_sr,
                "mdd_pct": d.mdd * 100.0,
                "daily_cr": d.daily_cr,
                "daily_sor": d.daily_sor,
            }
            for d in contract_diagnostics
        ])

        df_behavior = pd.DataFrame([
            {
                "contract": d.contract,
                "steps": d.steps,
                "long_pct": d.long_pct,
                "short_pct": d.short_pct,
                "flat_pct": d.flat_pct,
                "mean_holding_bars": d.mean_holding_bars,
                "median_holding_bars": d.median_holding_bars,
                "trades": d.trades,
                "policy_inference_steps": d.policy_inference_steps,
                "action_persistence_steps": d.action_persistence_steps,
                "defensive_preemptions": d.defensive_preemptions,
                "defensive_rule_closes": d.defensive_rule_closes,
                "hard_stop_losses": d.hard_stop_losses,
                "stop_loss_cooldown_steps": d.stop_loss_cooldown_steps,
                "circuit_breaker_suspension_steps": d.circuit_breaker_suspension_steps,
            }
            for d in contract_diagnostics
        ])

        macro_rows = []
        for d in contract_diagnostics:
            row = {"contract": d.contract, "steps": d.steps}
            row.update(d.macro_counts)
            macro_rows.append(row)
        df_macro = pd.DataFrame(macro_rows)

        # Summary Portfolio Metrics
        total_gross_pnl = float(df_pnl["gross_pnl"].sum())
        total_net_pnl = float(df_pnl["net_pnl"].sum())
        total_friction = float(df_pnl["friction"].sum())
        total_commission = float(df_pnl["est_commission"].sum())
        total_slippage = float(df_pnl["est_slippage"].sum())
        total_turnover = float(df_pnl["turnover_volume"].sum())
        total_capital = float(sum(d.req_money for d in contract_diagnostics))
        portfolio_return = float(total_net_pnl / total_capital) if total_capital > 0 else 0.0
        portfolio_gross_return = float(total_gross_pnl / total_capital) if total_capital > 0 else 0.0
        win_rate = float(np.mean([1.0 if d.net_pnl > 0 else 0.0 for d in contract_diagnostics]))
        friction_consumption_pct = float(total_friction / total_gross_pnl * 100.0) if total_gross_pnl > 0 else 0.0
        portfolio_turnover_mult = float(total_turnover / total_capital) if total_capital > 0 else 0.0

        summary_data = {
            "portfolio": {
                "contract_count": len(contract_diagnostics),
                "total_steps": int(sum(d.steps for d in contract_diagnostics)),
                "total_capital": total_capital,
                "total_gross_pnl": total_gross_pnl,
                "total_net_pnl": total_net_pnl,
                "portfolio_return_pct": portfolio_return * 100.0,
                "portfolio_gross_return_pct": portfolio_gross_return * 100.0,
                "win_rate_pct": win_rate * 100.0,
                "mean_tr_pct": float(df_pnl["dihft_tr_pct"].mean()),
                "mean_annual_sr": float(df_pnl["annual_sr"].mean()),
                "mean_mdd_pct": float(df_pnl["mdd_pct"].mean()),
                "mean_daily_cr": float(df_pnl["daily_cr"].mean()),
                "mean_daily_sor": float(df_pnl["daily_sor"].mean()),
                "total_turnover_volume": total_turnover,
                "portfolio_turnover_multiple": portfolio_turnover_mult,
                "total_trades": int(df_pnl["trades"].sum()),
                "total_friction": total_friction,
                "total_commission": total_commission,
                "total_slippage": total_slippage,
                "friction_consumption_ratio_pct": friction_consumption_pct,
            },
            "contracts": [asdict(d) for d in contract_diagnostics],
        }

        # Export CSVs
        csv_pnl_path = self.output_dir / "contract_pnl_friction.csv"
        csv_beh_path = self.output_dir / "contract_behavior_risk.csv"
        csv_mac_path = self.output_dir / "macro_routing_distribution.csv"
        df_pnl.to_csv(csv_pnl_path, index=False)
        df_behavior.to_csv(csv_beh_path, index=False)
        df_macro.to_csv(csv_mac_path, index=False)

        # Export JSON
        json_path = self.output_dir / "diagnostics_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        # Export Markdown Report
        md_path = self.output_dir / "diagnostics_summary.md"
        self._write_markdown_report(md_path, summary_data, df_pnl, df_behavior, df_macro)

        logger.info("Saved diagnostics artifacts to %s", self.output_dir)
        return summary_data

    @staticmethod
    def _df_to_markdown(df: pd.DataFrame) -> str:
        headers = [str(c) for c in df.columns]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in df.iterrows():
            vals = [
                f"{val:.2f}" if isinstance(val, (float, np.floating)) else str(val)
                for val in row
            ]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    def _write_markdown_report(
        self,
        md_path: Path,
        summary_data: dict[str, Any],
        df_pnl: pd.DataFrame,
        df_behavior: pd.DataFrame,
        df_macro: pd.DataFrame,
    ) -> None:
        p = summary_data["portfolio"]
        md_lines = [
            "# 全维度交易诊断与归因分析报告 (Trading Diagnostics Report)",
            "",
            "## 1. 投资组合核心绩效与摩擦损耗概览",
            "",
            f"- **评估合约数**：`{p['contract_count']}` 支合约，累计 `{p['total_steps']:,}` 步",
            f"- **初始总本金**：`{p['total_capital']:,.2f}` 元",
            f"- **盘面择时毛利润 (Gross PnL)**：**`+{p['total_gross_pnl']:,.2f}` 元**（**+{p['portfolio_gross_return_pct']:.2f}%**）",
            f"- **最终实现净利润 (Net PnL)**：**`+{p['total_net_pnl']:,.2f}` 元**（**+{p['portfolio_return_pct']:.2f}%**）",
            f"- **合约多合约胜率 (Win Rate)**：**`{p['win_rate_pct']:.2f}%`**",
            f"- **平均年化夏普比率 (Annual SR)**：`{p['mean_annual_sr']:.4f}`",
            f"- **平均最大回撤 (Mean MDD)**：`{p['mean_mdd_pct']:.2f}%`",
            f"- **平均索提诺比率 (Daily SoR)**：`{p['mean_daily_sor']:.2f}`",
            "",
            "### 摩擦成本解耦 (Friction Breakdown)",
            "",
            f"- **总调仓次数**：`{p['total_trades']:,}` 次",
            f"- **总换手成交额**：`{p['total_turnover_volume']:,.2f}` 元（本金换手倍数 **`{p['portfolio_turnover_multiple']:.1f}x`**）",
            f"- **手续费损耗 (Commission)**：`{p['total_commission']:,.2f}` 元",
            f"- **订单簿滑点损耗 (Slippage)**：`{p['total_slippage']:,.2f}` 元",
            f"- **摩擦损耗总额 (Total Friction)**：**`{p['total_friction']:,.2f}` 元**",
            f"- **摩擦损耗吞噬率 (Friction / Gross PnL)**：**`{p['friction_consumption_ratio_pct']:.2f}%`**",
            "",
            "---",
            "",
            "## 2. 逐合约收益、摩擦与换手汇总",
            "",
            self._df_to_markdown(df_pnl[
                [
                    "contract",
                    "bh_tr_pct",
                    "bh_5x_tr_pct",
                    "gross_pnl",
                    "net_pnl",
                    "friction",
                    "friction_ratio_pct",
                    "trades",
                    "turnover_ratio",
                    "dihft_tr_pct",
                    "annual_sr",
                    "mdd_pct",
                ]
            ]),

            "",
            "---",
            "",
            "## 3. 逐合约交易姿态与风控触发统计",
            "",
            self._df_to_markdown(df_behavior[
                [
                    "contract",
                    "long_pct",
                    "short_pct",
                    "flat_pct",
                    "mean_holding_bars",
                    "action_persistence_steps",
                    "defensive_rule_closes",
                    "hard_stop_losses",
                    "circuit_breaker_suspension_steps",
                ]
            ]),

            "",
            "---",
            "",
            "## 4. 逐合约宏观 VAE 路由分布",
            "",
            self._df_to_markdown(df_macro),

            "",
        ]

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive Trading Diagnostics Runner")
    parser.add_argument("--result_dir", type=str, required=True, help="Path to trial or final_result directory")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to market feather data folder")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for diagnostics")
    parser.add_argument("--initial_wallet_balance", type=float, default=6000.0, help="Initial wallet balance per contract")
    parser.add_argument("--commission_rate", type=float, default=0.0003, help="Commission rate")
    parser.add_argument("--contract_unit", type=float, default=1.0, help="Contract unit multiplier")
    parser.add_argument("--freq", type=int, default=12, help="Bar frequency per day for annualization")
    args = parser.parse_args()

    calculator = TradingDiagnosticsCalculator(
        result_dir=args.result_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        initial_wallet_balance=args.initial_wallet_balance,
        commission_rate=args.commission_rate,
        contract_unit=args.contract_unit,
        freq=args.freq,
    )
    calculator.run()


if __name__ == "__main__":
    main()
