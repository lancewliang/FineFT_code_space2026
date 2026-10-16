"""Compact audit reading entry points; full evidence stays in CSV and Parquet."""

import json
import math
import os
from pathlib import Path
from typing import Any

import polars as pl

# Reading examples for distinct review questions, never an automatic selection list.
REVIEW_EXAMPLES = (
    ("ask1_price_trend_2", "1h", "return", "短期方向特征，核对各流取舍"),
    ("ask1_price_log_return_2", "1h", "return", "与趋势特征对照，查漂移和冗余依据"),
    ("vwap", "next_5td", "return", "价格水平类长期关联，查适用限制"),
    ("atr_pct_6", "today", "abs_return", "幅度关联，查各流漂移和黑名单依据"),
    ("night_session", "today", "abs_return", "时段关联，查剩余交易时长影响"),
    ("bid1_price_trend_2", "1h", "return", "买卖侧短期趋势是否重复"),
    ("bid1_price_log_return_2", "1h", "return", "买卖侧收益率是否重复"),
    ("wap_1_log_return_2", "1h", "return", "加权报价与一档趋势的增量区别"),
    ("ask2_price_trend_2", "2h", "return", "二档趋势与一档趋势的冗余"),
    ("ask2_price_log_return_2", "2h", "return", "二档收益率的漂移与冗余"),
    ("bid2_price_trend_2", "1h", "return", "二档买侧趋势的流内取舍"),
    ("sell_wap_log_return_2", "today", "return", "加权卖价收益率的日内关联"),
    ("twap", "4h", "return", "与 VWAP 的价格水平冗余"),
    ("awap", "next_td", "return", "跨日价格水平关联及黑名单依据"),
    ("ask1_price", "next_5td", "return", "原始报价未入选的原因是否有证据"),
    ("index_price", "next_5td", "return", "参考价定义及价格水平影响"),
    ("ma_96_origin", "next_td", "abs_return", "长回看均线的幅度关联"),
    ("mark_price_trend_192", "next_td", "abs_return", "长回看趋势在不同目标上的表现"),
    ("ema_divergence_20_120", "4h", "abs_return", "趋势背离与幅度目标的适配性"),
    ("atr_pct_2", "today", "abs_return", "已入选短回看 ATR 与其他窗口对照"),
    ("atr_pct_12", "today", "abs_return", "ATR 回看跨度与漂移门禁"),
    ("atr_pct_24", "next_15td", "abs_return", "较长期幅度关联的跨窗变号"),
    ("atr_pct_192", "4h", "abs_return", "长回看 ATR 的窗口稳定性"),
    ("garman_klass_volatility_6", "today", "abs_return", "价格区间估计量的日内关联"),
    ("parkinson_volatility_6", "today", "abs_return", "与其他短回看波动特征的冗余"),
    ("rolling_volatility_6", "today", "abs_return", "滚动波动特征被拒的原始依据"),
    ("realized_volatility_6", "today", "abs_return", "已入选波动特征的对照基准"),
    ("garman_klass_volatility_48", "next_15td", "abs_return", "较长期关联的样本覆盖和变号"),
    ("parkinson_volatility_192", "4h", "abs_return", "长回看区间特征的样本与稳定性"),
    ("morning_session", "1h", "abs_return", "上午时段与收益幅度的关系"),
    ("afternoon_session", "4h", "abs_return", "下午时段与预测终点的交互"),
    ("is_opening_30m", "1h", "abs_return", "开盘时段是否解释其他特征关联"),
    ("vsump_16_origin", "1h", "abs_return", "已入选量能指标的对照基准"),
    ("vsumd_16_origin", "1h", "abs_return", "与 vsump 同窗指标的冗余依据"),
    ("vsumn_16_origin", "1h", "abs_return", "反向量能指标是否仅为符号变换"),
    ("vsump_12_origin", "2h", "abs_return", "量能指标跨回看窗口的变化"),
    ("vma_6_std_norm_origin", "1h", "abs_return", "归一化量能指标的分布漂移"),
    ("relative_volume_24", "1h", "abs_return", "已入选相对成交量的关联"),
    ("relative_amount_16", "2h", "abs_return", "相对成交额的流内取舍"),
    ("prev_week_body_to_range", "next_5td", "abs_return", "上周形态与跨周目标的适配"),
    ("prev_week_twap_deviation_pct", "next_5td", "abs_return", "上周偏离特征的跨窗口稳定性"),
    ("prev_week_return", "next_5td", "abs_return", "历史周收益与未来幅度的关系"),
    ("plus_di_14", "next_5td", "abs_return", "方向指标在不同流中的用途"),
    ("rank_96_origin", "4h", "abs_return", "位置秩特征的方向与幅度区别"),
    ("rank_192_origin", "4h", "abs_return", "长回看位置秩的淘汰依据"),
    ("bollinger_upper_192_origin", "next_td", "abs_return", "价格水平与相对位置的区别"),
    ("qtld_192_std_norm_origin", "next_td", "abs_return", "归一化分位特征的取舍"),
    ("max_192_std_norm_origin", "next_td", "abs_return", "极值特征与价格水平类的冗余"),
    ("cm_m1_m2_open_interest_share_m2", "next_15td", "abs_return", "跨月持仓结构、覆盖率及长期稳定性"),
    ("cntp_192_origin", "next_15td", "abs_return", "长期计数特征的跨窗口表现"),
)
STREAM_NAMES = {"slope_vae": "Slope", "vol_vae": "Vol", "rl_decision": "RL", "global": "全局"}
REASON_NAMES = {
    "Feature Blacklist Dropped": "特征黑名单",
    "Stream Blacklist Dropped": "流黑名单",
    "Distribution Drift Dropped": "分布漂移",
    "Correlation Filter Dropped": "相关性过滤",
}


def generate_matrices(metrics_df: pl.DataFrame, result_dir: Path) -> None:
    """Write directional feature × horizon correlation and coverage tables."""
    return_metrics = metrics_df.filter(pl.col("target_kind") == "return")
    for field, filename in (
        ("spearman_corr", "feature_horizon_corr_matrix.csv"),
        ("coverage_ratio", "feature_horizon_coverage_matrix.csv"),
    ):
        return_metrics.pivot(values=field, index="feature", on="horizon").write_csv(
            result_dir / filename
        )


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _number(value: object, *, percent: bool = False) -> str:
    if value is None:
        return "—"
    number = float(value)
    if not math.isfinite(number):
        return "—"
    return f"{number:.1%}" if percent else f"{number:+.4f}"


def render_review_examples(
    metrics: list[dict[str, Any]], annotations: list[dict[str, Any]]
) -> str:
    """Show up to fifty distinct review features with explicit native-sample scope."""
    by_key = {(r["feature"], r["horizon"], r["target_kind"]): r for r in metrics}
    selected = [(key, purpose) for *key, purpose in REVIEW_EXAMPLES if tuple(key) in by_key]
    seen = {key[0] for key, _ in selected}
    for key in sorted(by_key):
        if len(selected) >= 50:
            break
        if key[0] not in seen:
            selected.append((key, "补充阅读示例，核对各流状态与证据"))
            seen.add(key[0])
    by_feature = {row["feature"]: row for row in annotations}
    lines = [
        f"展示 **{len(selected)}** 个不同特征，围绕盘口、价格水平、历史波动、量能/时段和长期状态列举。",
        "每个特征展示预设的一个目标/期限，不自动挑选最高 IC；不足 50 个时按名称补充。",
        "以下是复核问题的示例，非最佳特征榜或推荐入选名单。IC 使用各期限原生有效样本。",
        "",
        "| 特征 | 目标 / 期限 | IC | 覆盖率 | 入选流；已记录的淘汰原因 | 复核重点 |",
        "|---|---|---|---|---|---|",
    ]
    for key, purpose in selected:
        row = by_key[tuple(key)]
        annotation = by_feature.get(row["feature"], row)
        membership = annotation.get("legacy_stream_membership", "unknown")
        membership = "/".join(STREAM_NAMES.get(s, s) for s in membership.split(","))
        membership = {"not_selected": "未入选", "rejected": "未入选", "unknown": "未知", "unassigned": "未知"}.get(
            membership, membership
        )
        drops = json.loads(annotation.get("drop_reasons", "[]") or "[]")
        reasons = "；".join(
            f"{STREAM_NAMES.get(d['scope'], d['scope'])}："
            f"{REASON_NAMES.get(d['reason'], d['reason'])}" for d in drops
        )
        if not reasons:
            reasons = "无已记录原因（不代表所有门禁均通过）"
        cells = (
            f"`{row['feature']}`",
            f"{row['target_kind']} / {row['horizon']}",
            _number(row.get("spearman_corr")),
            _number(row.get("coverage_ratio"), percent=True),
            f"{membership}；{reasons}",
            purpose,
        )
        lines.append("| " + " | ".join(_cell(c) for c in cells) + " |")
    if not selected:
        lines.append("暂无可展示指标。")
    return "\n".join(lines)


def render_artifact_guide(audit_dir: Path, insights_dir: Path, document_dir: Path) -> str:
    """Link evidence artifacts with their grain and a concrete reading instruction."""
    def link(directory: Path, name: str) -> str:
        path = directory / name
        relative = Path(os.path.relpath(path, document_dir)).as_posix()
        label = f"insights/{name}" if directory == insights_dir else name
        if not path.is_file():
            if name in {"audit_run_manifest.json", "_SUCCESS"}:
                return f"`{label}`（随运行成功写入）"
            return f"`{label}`（未生成）"
        return f"[{label}](<{relative}>)"

    rows = [
        (insights_dir, "feature_evidence.csv", "首选：特征 × 期限 × 目标", "筛 feature，读 IC、覆盖率、CI、selected_streams、drop_reasons"),
        (insights_dir, "legacy_selection_comparison.csv", "一特征一行，选择证据", "展开 drop_reasons 的 scope/reason；unknown 表示缺证据，不是统计失败"),
        (audit_dir, "matched_horizon_comparisons.csv", "特征 × 目标，共同样本", "比较 ic_common_1h/1day/5day 和 delta；next_td/next_5td 是 D+1/D+5 收盘"),
        (audit_dir, "window_strata_metrics.csv", "特征 × 期限 × 目标 × 合约 × 窗口", "筛同一特征；评价时排除 is_historical_reference=true，再看变号和 valid_pairs"),
        (audit_dir, "feature_metrics.csv", "上游全量统计表", "核验 contract_equal_spearman、sample_weighted_spearman、null_reason 和 CI 诊断；与选择注释分开追溯"),
        (audit_dir, "feature_horizon_corr_matrix.csv", "特征 × 七期限，仅 return", "横看有符号 IC；各期限样本不同，不用此表直接证明期限优劣"),
        (audit_dir, "feature_horizon_coverage_matrix.csv", "特征 × 七期限，仅 return", "与相关矩阵一起读，定位样本覆盖变化"),
        (audit_dir, "ci_diagnostics.csv", "期限 × 目标，CI 诊断", "看有效交易日、区块长度、有效区块、门槛和 null_reason"),
        (audit_dir, "invalid_reasons_summary.csv", "期限 × 标签无效原因", "区分过期报价、边界未成熟和尾部不足；不是特征淘汰原因"),
        (audit_dir, "row_labels.parquet", "决策点 × 期限，行级标签", "仅异常追查时读取；核验同合约、目标/观察时间、价格和 label_status"),
        (insights_dir, "feature_dynamics_classification.csv", "一特征一行，端点分类", "1h 与 15TD 的探索标签，不等于全期限单调变化或有效性判定"),
        (insights_dir, "selection_dynamics_summary.csv", "分类 × 选择状态 × 原因状态", "看哪些类别待复核；计数不是独立统计试验"),
        (insights_dir, "ci_evidence_summary.csv", "期限 × 目标 × CI 状态计数", "快速确认多少指标行缺区间以及记录的状态"),
        (audit_dir, "legacy_selection_comparison.csv", "原运行保存的选择注释", "与 insights 版本核对来源；版本不同不可无条件覆盖或混用"),
        (audit_dir, "audit_run_manifest.json", "审计运行元数据", "核对品种、频率、协议、配置指纹及运行完成时间"),
        (insights_dir, "analysis_manifest.json", "分析来源和展示参数", "核对 selection_source 路径/指纹、source_artifacts 和 display_parameters"),
        (audit_dir, "_SUCCESS", "运行完成标记", "只表示运行完成，不证明统计充分或全部协议验收通过"),
    ]
    lines = [
        "| 产物 | 是什么 | 怎么看 |",
        "|---|---|---|",
        *(f"| {link(directory, name)} | {grain} | {reading} |" for directory, name, grain, reading in rows),
        "",
        "`insights/top_return_<h>_features.csv` / `top_volatility_<h>_features.csv`：",
        "按展示阈值筛选、按绝对 IC 排序的方向 / 终点绝对收益候选摘录，含窗口汇总；不代表统计通过。",
        "`top_short_horizon_features.csv`、`top_long_horizon_features.csv`、",
        "`top_volatility_15td_features.csv` 分别是 return_1h、return_next_15td、volatility_next_15td 的兼容别名。",
        "`archive_before_evidence_sync/`、`archive_before_report_slimming/` 和 `legacy_annotation_repair/`",
        "是历史报告或修复快照，不作为当前主入口。",
    ]
    return "\n".join(lines)


def generate_audit_report(
    manifest: dict[str, Any],
    metrics_df: pl.DataFrame,
    invalid_summary_df: pl.DataFrame | None,
    output_path: Path,
    ci_diagnostics_df: pl.DataFrame | None = None,
    matched_comparisons_df: pl.DataFrame | None = None,
) -> None:
    """Render a bounded reading guide without copying the full metric tables."""
    audit_dir = output_path.parent
    selection_path = audit_dir / "legacy_selection_comparison.csv"
    annotations = pl.read_csv(selection_path).to_dicts() if selection_path.is_file() else []
    metrics = metrics_df.to_dicts()
    counts: dict[str, int] = {}
    for row in metrics:
        status = row.get("ci_status") or "MISSING_CI_REASON_UNRECORDED"
        counts[status] = counts.get(status, 0) + 1
    ci_statuses = "；".join(f"`{status}` {count:,} 行" for status, count in sorted(counts.items()))
    lines = [
        f"# Multi-Horizon Feature Audit Report：{manifest.get('symbol', '').upper()} / {manifest.get('frequency', '')}",
        "",
        f"审计特征 **{metrics_df['feature'].n_unique():,}** 个；协议 `{manifest.get('protocol_version', 'unknown')}`；",
        f"运行完成时间 `{manifest.get('completed_at', 'unknown')}`。全量统计留在明细文件。",
        "",
        "## 核心特征阅读示例",
        "",
        render_review_examples(metrics, annotations),
        "",
        "## 证据状态",
        "",
        f"基础 CI：{ci_statuses or '无指标'}。计数单位是指标行，不是独立观察。",
        "CI 缺失不等于特征无效；可用区间也仅为未做多重比较调整的探索性证据。",
    ]
    if ci_diagnostics_df is not None and not ci_diagnostics_df.is_empty():
        lines.extend([
            "",
            "Bootstrap Uncertainty & Horizon Block Diagnostics (Protocol Q14)：完整诊断见 ci_diagnostics.csv。",
            "",
            "| 期限 / 目标 | 有效交易日 | 有效区块 / 门槛（基础；双倍） |",
            "|---|---|---|",
        ])
        for row in ci_diagnostics_df.head(14).iter_rows(named=True):
            lines.append(
                f"| {row['horizon']} / {row['target_kind']} | {row['num_valid_days']} | "
                f"{row['effective_blocks']:g} / {row['min_effective_blocks']}；"
                f"{row['effective_blocks_2x']:g} / {row['min_effective_blocks_2x']} |"
            )
    if matched_comparisons_df is not None and not matched_comparisons_df.is_empty():
        row = matched_comparisons_df.row(0, named=True)
        lines.extend([
            "",
            "Three-Way Matched Common-Sample Horizon Comparisons (1h ↔ 1day ↔ 5day)：",
            f"共同决策点 **{row.get('matched_samples', 0):,}** 个，"
            f"保留率 **{row.get('sample_retention_ratio', 0):.2%}**。",
            "共同样本改善可比性，不能单独证明 alpha、因果关系或排除行情阶段影响。",
        ])
    if invalid_summary_df is not None and not invalid_summary_df.is_empty():
        lines.extend(["", "标签无效原因见 invalid_reasons_summary.csv；与原选择器淘汰原因分开阅读。"])
    lines.extend([
        "",
        "## 每个产物是什么、怎么看",
        "",
        render_artifact_guide(audit_dir, audit_dir / "insights", audit_dir),
        "",
        "## 逐特征下钻顺序",
        "",
        "1. 在 feature_evidence.csv 筛选 feature，再分 target_kind 和 horizon 看证据。",
        "2. 在 legacy_selection_comparison.csv 展开三流 scope/reason，核对来源版本；没有记录时保留 unknown。",
        "3. 在 window_strata_metrics.csv 检查各合约、窗口，再用共同样本表比较期限。",
        "4. 只有异常时才进入 CI、缺失标签或行级价格明细。淘汰标签本身不含阈值和被替代特征等完整依据。",
        "",
        "Markdown 负责摘要和导航；搜索、组合筛选、展开三流原因及联动窗口图应交给交互表格。",
        "return 是方向收益，abs_return 是终点绝对收益，不是路径实现波动率。审计不自动修改原入选清单。",
    ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
