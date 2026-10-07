"""Batch Diagnostics and Experiment Aggregation Engine.

Aggregates diagnostics_summary.json, contract PnL/friction CSVs, behavior/risk CSVs,
macro routing distributions, best_result.csv, manifests, and parameter records in
a single pass to prevent token exhaustion and sequential file reading.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.append(".")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from common import ArtifactNames, MetricColumns, RoutingParamColumns


@dataclass
class PortfolioSnapshot:
    contract_count: int
    total_steps: int
    total_capital: float
    total_gross_pnl: float
    total_net_pnl: float
    portfolio_gross_return_pct: float
    portfolio_return_pct: float
    win_rate_pct: float
    mean_annual_sr: float
    mean_mdd_pct: float
    mean_daily_cr: float
    mean_daily_sor: float
    total_turnover_volume: float
    portfolio_turnover_multiple: float
    total_trades: int
    total_commission: float
    total_slippage: float
    total_friction: float
    friction_consumption_ratio_pct: float


@dataclass
class BehaviorSnapshot:
    long_pct_mean: float
    short_pct_mean: float
    flat_pct_mean: float
    mean_holding_bars: float
    median_holding_bars: float
    trades_per_1k_steps: float
    policy_inference_steps: int
    action_persistence_steps: int
    defensive_preemptions: int
    defensive_rule_closes: int
    hard_stop_losses: int
    stop_loss_cooldown_steps: int
    circuit_breaker_suspension_steps: int


@dataclass
class MacroRoutingSnapshot:
    slot_percentages: dict[str, float]
    dominant_slot: str
    dominant_slot_pct: float
    active_slots_count: int


@dataclass
class ContractExtreme:
    contract: str
    net_pnl: float
    gross_pnl: float
    friction: float
    dihft_tr_pct: float
    annual_sr: float
    mdd_pct: float


@dataclass
class DiagnosticVerdict:
    status: str
    primary_bottleneck: str
    flags: list[str]
    advice: list[str]


@dataclass
class BatchExperimentDigest:
    portfolio: PortfolioSnapshot
    behavior: BehaviorSnapshot
    macro_routing: MacroRoutingSnapshot
    best_results_summary: dict[str, Any]
    top_winners: list[ContractExtreme]
    top_losers: list[ContractExtreme]
    verdict: DiagnosticVerdict
    meta: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "portfolio": asdict(self.portfolio),
            "behavior": asdict(self.behavior),
            "macro_routing": asdict(self.macro_routing),
            "best_results_summary": self.best_results_summary,
            "top_winners": [asdict(w) for w in self.top_winners],
            "top_losers": [asdict(l) for l in self.top_losers],
            "verdict": asdict(self.verdict),
            "meta": self.meta,
        }

    def format_text_report(self) -> str:
        p = self.portfolio
        b = self.behavior
        m = self.macro_routing
        v = self.verdict
        lines = [
            "=" * 78,
            f" [DIHFT EXPERIMENT DIAGNOSTIC DIGEST] - {self.meta['experiment_path']}",
            "=" * 78,
            f"Verdict Status      : {v.status}",
            f"Primary Bottleneck  : {v.primary_bottleneck}",
            f"Diagnostic Flags    : {', '.join(v.flags) if v.flags else 'NONE'}",
            "-" * 78,
            "1. FINANCIAL & FRICTION OVERVIEW",
            f"  Contracts / Steps : {p.contract_count} contracts | {p.total_steps:,} steps | Win Rate: {p.win_rate_pct:.1f}%",
            f"  Capital & Returns : Total Capital: {p.total_capital:,.2f} | Net Return: {p.portfolio_return_pct:+.2f}%",
            f"  Gross Market PnL  : {p.total_gross_pnl:+,.2f} ({p.portfolio_gross_return_pct:+.2f}%)",
            f"  Net Realized PnL  : {p.total_net_pnl:+,.2f} ({p.portfolio_return_pct:+.2f}%)",
            f"  Friction Breakdown: Total: {p.total_friction:,.2f} (Commission: {p.total_commission:,.2f} + Slippage: {p.total_slippage:,.2f})",
            f"  Friction Consumed : {p.friction_consumption_ratio_pct:.1f}% of Gross PnL",
            f"  Turnover Multiple : {p.portfolio_turnover_multiple:.1f}x capital ({p.total_trades:,} trades)",
            f"  Risk Profile      : Mean SR: {p.mean_annual_sr:.2f} | Mean MDD: {p.mean_mdd_pct:.2f}% | Daily CR: {p.mean_daily_cr:.2f}",
            "-" * 78,
            "2. MICRO BEHAVIOR & RISK EXECUTION",
            f"  Posture Allocation: Long: {b.long_pct_mean:.1f}% | Short: {b.short_pct_mean:.1f}% | Flat: {b.flat_pct_mean:.1f}%",
            f"  Holding Duration  : Mean: {b.mean_holding_bars:.1f} bars | Median: {b.median_holding_bars:.1f} bars",
            f"  Trading Frequency : {b.trades_per_1k_steps:.1f} trades / 1,000 bars",
            f"  Risk Interventions: Hard Stops: {b.hard_stop_losses} | Cooldown Steps: {b.stop_loss_cooldown_steps}",
            f"                      Circuit Breaks: {b.circuit_breaker_suspension_steps} | Preemptions: {b.defensive_preemptions}",
            "-" * 78,
            "3. MACRO VAE ROUTING DYNAMICS",
            f"  Dominant Slot     : {m.dominant_slot} ({m.dominant_slot_pct:.1f}%) | Active Slots Count: {m.active_slots_count}",
            f"  Slot Distribution : {json.dumps(m.slot_percentages, ensure_ascii=False)}",
            "-" * 78,
            "4. CONTRACT EXTREMES (TAIL IMPACT)",
        ]

        lines.append("  Top Winners:")
        for w in self.top_winners:
            lines.append(
                f"    + {w.contract:<8} Net: {w.net_pnl:+8.2f} | Gross: {w.gross_pnl:+8.2f} | "
                f"Friction: {w.friction:7.2f} | Ret: {w.dihft_tr_pct:+6.2f}% | SR: {w.annual_sr:5.2f}"
            )

        lines.append("  Top Losers:")
        for l in self.top_losers:
            lines.append(
                f"    - {l.contract:<8} Net: {l.net_pnl:+8.2f} | Gross: {l.gross_pnl:+8.2f} | "
                f"Friction: {l.friction:7.2f} | Ret: {l.dihft_tr_pct:+6.2f}% | MDD: {l.mdd_pct:5.2f}%"
            )

        if self.best_results_summary:
            lines.append("-" * 78)
            lines.append("5. BEST HEURISTIC CONFIGURATIONS (from best_result.csv)")
            for ind, row_dict in self.best_results_summary.items():
                tr_val = row_dict.get(MetricColumns.TR, "N/A")
                sr_val = row_dict.get(MetricColumns.ANNUAL_SR, "N/A")
                mdd_val = row_dict.get(MetricColumns.MDD, "N/A")
                lines.append(f"  Target [{ind:<10}]: TR={tr_val} | SR={sr_val} | MDD={mdd_val}")

        lines.append("-" * 78)
        lines.append("6. ACTIONABLE ADJUSTMENT ADVICE")
        for adv in v.advice:
            lines.append(f"  * {adv}")
        lines.append("=" * 78)
        return "\n".join(lines)


class BatchDiagnosticsLoader:
    def __init__(
        self,
        analysis_path: Path,
        final_result_dir: Path | None = None,
        selection_manifest_path: Path | None = None,
        top_k: int = 3,
    ) -> None:
        self.analysis_path = analysis_path
        self.final_result_dir = final_result_dir
        self.selection_manifest_path = selection_manifest_path
        self.top_k = top_k

    def _resolve_paths(self) -> tuple[Path, Path | None, Path | None]:
        if (self.analysis_path / "diagnostics").is_dir():
            diag_dir = self.analysis_path / "diagnostics"
            base_analysis_dir = self.analysis_path
        elif (self.analysis_path / "diagnostics_summary.json").is_file():
            diag_dir = self.analysis_path
            base_analysis_dir = self.analysis_path.parent
        else:
            diag_dir = self.analysis_path
            base_analysis_dir = self.analysis_path

        return diag_dir, base_analysis_dir, self.final_result_dir

    def load(self) -> BatchExperimentDigest:
        diag_dir, base_dir, final_dir = self._resolve_paths()

        summary_json_path = diag_dir / "diagnostics_summary.json"
        if not summary_json_path.is_file():
            raise FileNotFoundError(f"diagnostics_summary.json not found in {diag_dir}")

        with open(summary_json_path, encoding="utf-8") as f:
            summary_raw = json.load(f)

        port_data = summary_raw["portfolio"]
        portfolio = PortfolioSnapshot(
            contract_count=int(port_data["contract_count"]),
            total_steps=int(port_data["total_steps"]),
            total_capital=float(port_data["total_capital"]),
            total_gross_pnl=float(port_data["total_gross_pnl"]),
            total_net_pnl=float(port_data["total_net_pnl"]),
            portfolio_gross_return_pct=float(port_data["portfolio_gross_return_pct"]),
            portfolio_return_pct=float(port_data["portfolio_return_pct"]),
            win_rate_pct=float(port_data["win_rate_pct"]),
            mean_annual_sr=float(port_data["mean_annual_sr"]),
            mean_mdd_pct=float(port_data["mean_mdd_pct"]),
            mean_daily_cr=float(port_data["mean_daily_cr"]),
            mean_daily_sor=float(port_data["mean_daily_sor"]),
            total_turnover_volume=float(port_data["total_turnover_volume"]),
            portfolio_turnover_multiple=float(port_data["portfolio_turnover_multiple"]),
            total_trades=int(port_data["total_trades"]),
            total_commission=float(port_data["total_commission"]),
            total_slippage=float(port_data["total_slippage"]),
            total_friction=float(port_data["total_friction"]),
            friction_consumption_ratio_pct=float(port_data["friction_consumption_ratio_pct"]),
        )

        # 2. Behavior CSV
        csv_beh_path = diag_dir / "contract_behavior_risk.csv"
        df_beh = pd.read_csv(csv_beh_path)
        total_steps = portfolio.total_steps
        trades_per_1k = (portfolio.total_trades / total_steps * 1000.0) if total_steps > 0 else 0.0

        behavior = BehaviorSnapshot(
            long_pct_mean=float(df_beh["long_pct"].mean()),
            short_pct_mean=float(df_beh["short_pct"].mean()),
            flat_pct_mean=float(df_beh["flat_pct"].mean()),
            mean_holding_bars=float(df_beh["mean_holding_bars"].mean()),
            median_holding_bars=float(df_beh["median_holding_bars"].mean()),
            trades_per_1k_steps=trades_per_1k,
            policy_inference_steps=int(df_beh["policy_inference_steps"].sum()),
            action_persistence_steps=int(df_beh["action_persistence_steps"].sum()),
            defensive_preemptions=int(df_beh["defensive_preemptions"].sum()),
            defensive_rule_closes=int(df_beh["defensive_rule_closes"].sum()),
            hard_stop_losses=int(df_beh["hard_stop_losses"].sum()),
            stop_loss_cooldown_steps=int(df_beh["stop_loss_cooldown_steps"].sum()),
            circuit_breaker_suspension_steps=int(df_beh["circuit_breaker_suspension_steps"].sum()),
        )

        # 3. Macro Routing CSV
        csv_mac_path = diag_dir / "macro_routing_distribution.csv"
        df_mac = pd.read_csv(csv_mac_path)
        slot_cols = [c for c in df_mac.columns if c not in ("contract", "steps")]
        slot_totals = df_mac[slot_cols].sum()
        total_slot_steps = slot_totals.sum()
        slot_pcts: dict[str, float] = {}
        if total_slot_steps > 0:
            for s_col in slot_cols:
                slot_pcts[s_col] = round(float(slot_totals[s_col] / total_slot_steps * 100.0), 2)
            dominant_slot = str(slot_totals.idxmax())
            dominant_pct = slot_pcts[dominant_slot]
            active_count = int((slot_totals > 0).sum())
        else:
            dominant_slot = "none"
            dominant_pct = 0.0
            active_count = 0

        macro_routing = MacroRoutingSnapshot(
            slot_percentages=slot_pcts,
            dominant_slot=dominant_slot,
            dominant_slot_pct=dominant_pct,
            active_slots_count=active_count,
        )

        # 4. PnL CSV & Extremes
        csv_pnl_path = diag_dir / "contract_pnl_friction.csv"
        df_pnl = pd.read_csv(csv_pnl_path)
        sorted_pnl = df_pnl.sort_values(by="net_pnl", ascending=False)
        top_winners: list[ContractExtreme] = []
        for _, row in sorted_pnl.head(self.top_k).iterrows():
            top_winners.append(
                ContractExtreme(
                    contract=str(row["contract"]),
                    net_pnl=float(row["net_pnl"]),
                    gross_pnl=float(row["gross_pnl"]),
                    friction=float(row["friction"]),
                    dihft_tr_pct=float(row["dihft_tr_pct"]),
                    annual_sr=float(row["annual_sr"]),
                    mdd_pct=float(row["mdd_pct"]),
                )
            )

        top_losers: list[ContractExtreme] = []
        for _, row in sorted_pnl.tail(self.top_k).sort_values(by="net_pnl").iterrows():
            top_losers.append(
                ContractExtreme(
                    contract=str(row["contract"]),
                    net_pnl=float(row["net_pnl"]),
                    gross_pnl=float(row["gross_pnl"]),
                    friction=float(row["friction"]),
                    dihft_tr_pct=float(row["dihft_tr_pct"]),
                    annual_sr=float(row["annual_sr"]),
                    mdd_pct=float(row["mdd_pct"]),
                )
            )

        # 5. Best Result CSV (if exists in base_dir)
        best_results_summary: dict[str, Any] = {}
        best_csv = base_dir / ArtifactNames.BEST_RESULT_CSV
        if best_csv.is_file():
            df_best = pd.read_csv(best_csv)
            for _, r in df_best.iterrows():
                ind = str(r[MetricColumns.INDICATOR]) if MetricColumns.INDICATOR in r else "default"
                best_results_summary[ind] = r.to_dict()

        # 6. Synthesize Verdict & Advice
        verdict = self._synthesize_verdict(portfolio, behavior, macro_routing, top_losers)

        meta = {
            "analysis_path": str(self.analysis_path),
            "experiment_path": str(base_dir),
            "diagnostics_dir": str(diag_dir),
        }

        return BatchExperimentDigest(
            portfolio=portfolio,
            behavior=behavior,
            macro_routing=macro_routing,
            best_results_summary=best_results_summary,
            top_winners=top_winners,
            top_losers=top_losers,
            verdict=verdict,
            meta=meta,
        )

    @staticmethod
    def _synthesize_verdict(
        p: PortfolioSnapshot,
        b: BehaviorSnapshot,
        m: MacroRoutingSnapshot,
        top_losers: list[ContractExtreme],
    ) -> DiagnosticVerdict:
        flags: list[str] = []
        advice: list[str] = []

        # Check Directional Timing Alpha
        directional_ok = p.total_gross_pnl > 0 and p.portfolio_gross_return_pct > 5.0
        if not directional_ok:
            flags.append("DIRECTIONAL_ALPHA_WEAK")
            advice.append(
                "Gross market timing PnL is weak or negative. Low-level Q-networks or VAE regime "
                "classification failed to predict price direction. Check if test period has strong "
                "regime shift or consider switching selection_metric from TR to annual_sr/daily_cr."
            )

        # Check Friction Ratio
        if p.friction_consumption_ratio_pct > 80.0:
            flags.append("EXCESSIVE_FRICTION_CONSUMPTION")
            advice.append(
                f"Friction consumes {p.friction_consumption_ratio_pct:.1f}% of gross profits. "
                "Turnover multiple is excessive. Increase --action_persistence (e.g., from 3 to 6/12), "
                "tighten hysteresis thresholds, or activate asymmetric turnover penalty (ADR 0050)."
            )
        elif p.portfolio_turnover_multiple > 50.0:
            flags.append("HIGH_TURNOVER_CHURN")
            advice.append(
                f"Portfolio turnover multiple is {p.portfolio_turnover_multiple:.1f}x. "
                "Short holding durations indicate micro-scalping churn. Enforce minimum holding bars."
            )

        # Check Holding Bars
        if b.mean_holding_bars < 5.0:
            flags.append("ULTRA_SHORT_HOLDING")
            advice.append(
                f"Mean holding duration is only {b.mean_holding_bars:.1f} bars. Strategy exits too quickly, "
                "paying spreads without capturing trend moves. Increase action persistence."
            )

        # Check Inaction / Extreme Flat
        if b.flat_pct_mean > 75.0:
            flags.append("CHRONIC_INACATION_FLAT")
            advice.append(
                f"Average flat stance is {b.flat_pct_mean:.1f}%. Strategy is excessively passive. "
                "Check if ood_threshold is too strict or circuit breakers are overly sensitive."
            )

        # Check Tail Risk Loss
        worst_loss = top_losers[0].net_pnl if top_losers else 0.0
        if worst_loss < -0.15 * p.total_capital:
            flags.append("FAT_TAIL_CONTRACT_RISK")
            advice.append(
                f"Single worst contract {top_losers[0].contract} lost {worst_loss:,.2f} "
                f"({worst_loss / p.total_capital * 100.0:.1f}% of capital). "
                "Check stop-loss mechanism. Tighten --stop_loss_return_threshold (e.g. to 0.012) "
                "and extend --circuit_breaker_cooling_steps."
            )

        # Macro Slot Collapse
        if m.dominant_slot_pct > 80.0:
            flags.append("MACRO_SLOT_MONOPOLY")
            advice.append(
                f"Dominant slot {m.dominant_slot} took {m.dominant_slot_pct:.1f}% of all steps. "
                "High-level VAE routing failed to diversify across market regimes. "
                "Tune slope/volatility margin thresholds or calibrate VAE regime bounds."
            )

        # Determine overall Status & Primary Bottleneck
        if p.total_net_pnl > 0 and p.win_rate_pct >= 80.0 and p.friction_consumption_ratio_pct < 60.0:
            status = "HEALTHY"
            bottleneck = "None (Profitable, well-controlled friction)"
        elif p.total_gross_pnl > 0 and p.total_net_pnl <= 0:
            status = "FRICTION_BOUND"
            bottleneck = "Friction Overkill (Strategy predicts well but fees/slippage destroy profits)"
        elif p.total_gross_pnl <= 0:
            status = "DIRECTIONAL_FAILURE"
            bottleneck = "Alpha Signal Invalidation (Low-level agents trade against market trend)"
        elif p.mean_mdd_pct > 25.0:
            status = "TAIL_RISK_BREACH"
            bottleneck = "Excessive Drawdown / Tail Risk (Needs strict stop-loss / circuit breakers)"
        else:
            status = "SUBOPTIMAL"
            bottleneck = "Mixed Inefficiencies (Moderate alpha, elevated friction or tail drag)"

        if not advice:
            advice.append("Performance metrics meet all baseline health criteria. Keep monitoring.")

        return DiagnosticVerdict(
            status=status,
            primary_bottleneck=bottleneck,
            flags=flags,
            advice=advice,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch load and analyze DiHFT experiment diagnostics.")
    parser.add_argument(
        "--analysis_path",
        type=str,
        required=True,
        help="Path to experiment analysis folder or diagnostics subfolder.",
    )
    parser.add_argument(
        "--final_result_dir",
        type=str,
        default=None,
        help="Optional path to final_result execution folder.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (compact human-readable text or structured JSON).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top winning/losing contracts to display.",
    )
    args = parser.parse_args()

    loader = BatchDiagnosticsLoader(
        analysis_path=Path(args.analysis_path),
        final_result_dir=Path(args.final_result_dir) if args.final_result_dir else None,
        top_k=args.top_k,
    )
    digest = loader.load()

    if args.format == "json":
        print(json.dumps(digest.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(digest.format_text_report())


if __name__ == "__main__":
    main()
