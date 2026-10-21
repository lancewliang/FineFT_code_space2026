"""Build diagnostic plots, descriptive factor regression, and input fingerprints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/research/fu_10min_followup_evidence"


def main() -> None:
    probes = pd.concat([
        pd.read_csv(OUT / "state_machine_probes.csv"),
        pd.read_csv(OUT / "state_machine_probes_extra.csv"),
    ], ignore_index=True).drop_duplicates("case", keep="last")
    probes.to_csv(OUT / "all_probes.csv", index=False)
    series = {}
    for case in probes["case"]:
        frame = pd.read_csv(OUT / f"probe_{case}_timeline.csv", parse_dates=["timestamp"])
        series[case] = frame.set_index("timestamp").net_pnl
    # Calendar-date bins, not exchange TradingDay; no annualized alpha claim.
    daily = pd.DataFrame(series).fillna(0).resample("D").sum()
    daily = daily.loc[(pd.DataFrame(series).resample("D").count().sum(axis=1) > 0)]
    y = daily.baseline / 12000
    x = sm.add_constant(daily[["benchmark_long", "benchmark_mom48"]] / 12000)
    fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    result = {"scope": "in-sample descriptive regression; calendar-date cash increments / fixed 12000 capital; not validated alpha", "calendar_dates": len(daily), "r_squared": float(fit.rsquared), "coefficients": fit.params.to_dict(), "hac_t_values": fit.tvalues.to_dict(), "hac_p_values": fit.pvalues.to_dict(), "hac_confidence_intervals": fit.conf_int().to_dict(orient="index"), "intercept_total_pnl": float(fit.params['const']*12000*len(daily))}
    (OUT / "descriptive_factor_regression.json").write_text(json.dumps(result, indent=2))
    daily.to_csv(OUT / "calendar_date_probe_pnl.csv")
    episodes = pd.read_csv(OUT / "episodes.csv")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    colors = {"baseline": "#1765a7", "main_only": "#239169", "delivery_disabled": "#d88a19", "benchmark_long": "#8b8797"}
    for case in colors:
        axes[0, 0].plot(series[case].index, series[case].cumsum()/12000*100, label=case, color=colors[case], linewidth=1.8)
    axes[0, 0].set(title="Cumulative return on frozen valid period", ylabel="Return (%)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].tick_params(axis="x", rotation=20)
    cases = ["baseline", "main_only", "delivery_disabled", "stop_2pct", "trailing_1pct_half", "exit_ratio_1", "hier_margin_002"]
    offsets = {"baseline": (6, -15), "main_only": (-22, 8), "delivery_disabled": (6, 4), "stop_2pct": (6, 0), "trailing_1pct_half": (5, -12), "exit_ratio_1": (8, 4), "hier_margin_002": (8, 4)}
    for case in cases:
        row = probes[probes["case"] == case].iloc[0]
        axes[0, 1].scatter(row.mdd*100, row.tr*100, s=55)
        axes[0, 1].annotate(case, (row.mdd*100, row.tr*100), xytext=offsets[case], textcoords="offset points", fontsize=8)
    axes[0, 1].scatter(5, 50, marker="*", color="#b43141", s=150, label="Target: 50% / 5%")
    axes[0, 1].axvline(5, color="grey", linestyle="--", linewidth=1)
    axes[0, 1].set(title="Actual state-machine probes on valid", xlabel="Maximum drawdown (%)", ylabel="Return (%)", ylim=(-8, 55), xlim=(2.5, 8))
    axes[0, 1].legend(fontsize=8, loc="upper right")
    pnl = episodes.groupby("exit_reason").pnl.sum().sort_values()
    axes[1, 0].barh(pnl.index, pnl, color=["#b43141" if v < 0 else "#239169" for v in pnl])
    axes[1, 0].set(title="Episode PnL by actual exit reason", xlabel="Normalized ledger PnL (unit=1)")
    axes[1, 0].tick_params(axis="y", labelsize=8)
    stopped = episodes.exit_reason.eq("HARD_STOP_LOSS")
    axes[1, 1].scatter(episodes.loc[~stopped, "mfe_net_path"], episodes.loc[~stopped, "pnl"], alpha=.7, s=30, label="Other exits")
    axes[1, 1].scatter(episodes.loc[stopped, "mfe_net_path"], episodes.loc[stopped, "pnl"], color="#b43141", s=35, label="Hard stops")
    axes[1, 1].axhline(0, color="grey", linewidth=1)
    axes[1, 1].set(title="Profit giveback is descriptive, not recoverable alpha", xlabel="Maximum cumulative episode PnL (hindsight)", ylabel="Final episode PnL")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.15)
    fig.savefig(OUT / "diagnostic_overview.png", dpi=180)
    plt.close(fig)
    paths = [ROOT / "docs/research/1.txt", ROOT / "result/DiHFT/final_result/fu/10min_parallel/main_sub/summary_metrics.json", ROOT / "result/DiHFT/final_result/fu/10min_parallel/main_sub/portfolio_action_timeline.csv", ROOT / "analysis_result/DiHFT/low_level/fu/10min_parallel/two_dimensional_selection/model.pth", ROOT / "analysis_result/DiHFT/low_level/fu/10min_parallel/two_dimensional_selection/two_dimensional_selection_manifest.json", ROOT / "FineFT/RL/DiHFT/high_level/vae_routing_util.py", ROOT / "FineFT/env/env_class/base_env.py"]
    paths += sorted((ROOT / "dataset/10min/fu/valid").glob("*.feather"))
    paths += sorted((ROOT / "dataset/10min/fu").glob("*state_features.npy"))
    paths += [ROOT / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/10min/fu/train/feature_selection_manifest.json", ROOT / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/10min/fu/train/distribution_audit_metrics.csv"]
    paths += sorted((ROOT / "docs/research").glob("fu_10min_*.py"))
    manifest = {"date": "2026-10-21", "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "scope": "valid only; test timestamps inspected for split boundaries, test policies not evaluated", "sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    (OUT / "input_fingerprints.json").write_text(json.dumps(manifest, indent=2))
    print(probes[["case", "net_pnl", "tr", "mdd", "trades", "hard_stops"]].to_string(index=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
