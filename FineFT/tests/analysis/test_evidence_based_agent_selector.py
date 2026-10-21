import json
from pathlib import Path
import polars as pl
import pytest

from analysis.pick_agent.FineFT_two_dimensional_agent_selector import (
    SelectionConfig,
    TwoDimensionalAgentSelector,
)


def _setup_mock_environment(
    tmp_path: Path,
    epoch_num: int = 1,
    vol_analysis_rows: list[dict] | None = None,
    slope_analysis_rows: list[dict] | None = None,
    vol_detail_rows: list[dict] | None = None,
    slope_detail_rows: list[dict] | None = None,
    valid_labels: dict[str, dict[str, dict[str, list[str]]]] | None = None,
):
    candidate_root = tmp_path / "candidate"
    valid_root = tmp_path / "valid"
    epoch_path = candidate_root / f"epoch_{epoch_num}"
    (epoch_path / "trained_model.pkl").parent.mkdir(parents=True, exist_ok=True)
    (epoch_path / "trained_model.pkl").write_bytes(b"checkpoint")

    vol_path = epoch_path / "volatility"
    slope_path = epoch_path / "slope"
    vol_path.mkdir(parents=True, exist_ok=True)
    slope_path.mkdir(parents=True, exist_ok=True)

    if vol_analysis_rows:
        pl.DataFrame(vol_analysis_rows).write_csv(vol_path / "analysis_result.csv")
    if slope_analysis_rows:
        pl.DataFrame(slope_analysis_rows).write_csv(slope_path / "analysis_result.csv")
    if vol_detail_rows:
        pl.DataFrame(vol_detail_rows).write_csv(vol_path / f"trading_action_detail_epoch_{epoch_num}.csv")
    if slope_detail_rows:
        pl.DataFrame(slope_detail_rows).write_csv(slope_path / f"trading_action_detail_epoch_{epoch_num}.csv")

    if valid_labels:
        for label_type, contract_map in valid_labels.items():
            for contract, label_dict in contract_map.items():
                for label_name, timestamps in label_dict.items():
                    label_dir = valid_root / label_type / contract / label_name
                    label_dir.mkdir(parents=True, exist_ok=True)
                    pl.DataFrame({"timestamp": timestamps}).write_ipc(label_dir / "df_0.feather")

    return candidate_root, valid_root


def test_conjunctive_gate_rejects_candidate_with_negative_marginal_volatility_slot6(tmp_path: Path) -> None:
    """
    Slot 6 reproduction:
    Candidate 0 (epoch_1:bin_0) has negative marginal volatility return (-0.10),
    but in joint slice (0, 0) it achieved positive reward (+0.50).
    Candidate 1 (epoch_1:bin_1) has positive marginal volatility (+0.30) and positive joint reward (+0.25).

    Under ADR-0059 conjunctive gates:
    Candidate 0 MUST fail gate_marginal_volatility and be rejected.
    Candidate 1 MUST pass all gates and be selected.
    """
    contracts = ["fu2409", "fu2411"]
    joint_timestamps = [f"2026-01-01 {hour:02d}:00:00" for hour in range(9, 13)]     # 4 timestamps
    other_timestamps = [f"2026-01-01 {hour:02d}:00:00" for hour in range(13, 29)]    # 16 timestamps
    all_timestamps = joint_timestamps + other_timestamps                             # 20 timestamps

    # Volatility has 1 label (label_0) covering all 20 timestamps:
    # bin_0: 4 steps * 0.50 + 16 steps * (-0.25) = 2.0 - 4.0 = -2.0 per contract (mean = -0.10)
    # bin_1: 4 steps * 0.25 + 16 steps * (+0.3125) = 1.0 + 5.0 = 6.0 per contract (mean = +0.30)
    vol_analysis_rows = [
        {
            "标签": "label_0",
            "初始动作": 0,
            "分箱索引": 0,
            "合约": json.dumps(contracts),
            "数据文件": json.dumps([f"{c}/label_0/df_0.feather" for c in contracts]),
            "奖励总和": json.dumps([-2.0, -2.0]),
            "数据长度": json.dumps([21, 21]),
            "换手率": json.dumps([0.1, 0.1]),
        },
        {
            "标签": "label_0",
            "初始动作": 0,
            "分箱索引": 1,
            "合约": json.dumps(contracts),
            "数据文件": json.dumps([f"{c}/label_0/df_0.feather" for c in contracts]),
            "奖励总和": json.dumps([6.0, 6.0]),
            "数据长度": json.dumps([21, 21]),
            "换手率": json.dumps([0.1, 0.1]),
        },
        # dummy label_1 rows for 2-label symmetry
        {
            "标签": "label_1",
            "初始动作": 0,
            "分箱索引": 0,
            "合约": json.dumps(contracts),
            "数据文件": json.dumps([f"{c}/label_1/df_0.feather" for c in contracts]),
            "奖励总和": json.dumps([1.0, 1.0]),
            "数据长度": json.dumps([2, 2]),
            "换手率": json.dumps([0.1, 0.1]),
        },
        {
            "标签": "label_1",
            "初始动作": 0,
            "分箱索引": 1,
            "合约": json.dumps(contracts),
            "数据文件": json.dumps([f"{c}/label_1/df_0.feather" for c in contracts]),
            "奖励总和": json.dumps([1.0, 1.0]),
            "数据长度": json.dumps([2, 2]),
            "换手率": json.dumps([0.1, 0.1]),
        },
    ]

    # Slope has 2 labels:
    # label_0: 4 steps (the joint timestamps with vol label_0)
    # label_1: 16 steps (the other timestamps with vol label_0) + 1 dummy step for vol label_1
    slope_analysis_rows = [
        {
            "标签": "label_0",
            "初始动作": 0,
            "分箱索引": 0,
            "合约": json.dumps(contracts),
            "数据文件": json.dumps([f"{c}/label_0/df_0.feather" for c in contracts]),
            "奖励总和": json.dumps([2.0, 2.0]),
            "数据长度": json.dumps([5, 5]),
            "换手率": json.dumps([0.1, 0.1]),
        },
        {
            "标签": "label_0",
            "初始动作": 0,
            "分箱索引": 1,
            "合约": json.dumps(contracts),
            "数据文件": json.dumps([f"{c}/label_0/df_0.feather" for c in contracts]),
            "奖励总和": json.dumps([1.0, 1.0]),
            "数据长度": json.dumps([5, 5]),
            "换手率": json.dumps([0.1, 0.1]),
        },
        {
            "标签": "label_1",
            "初始动作": 0,
            "分箱索引": 0,
            "合约": json.dumps(contracts),
            "数据文件": json.dumps([f"{c}/label_1/df_0.feather" for c in contracts]),
            "奖励总和": json.dumps([-3.0, -3.0]),
            "数据长度": json.dumps([18, 18]),
            "换手率": json.dumps([0.1, 0.1]),
        },
        {
            "标签": "label_1",
            "初始动作": 0,
            "分箱索引": 1,
            "合约": json.dumps(contracts),
            "数据文件": json.dumps([f"{c}/label_1/df_0.feather" for c in contracts]),
            "奖励总和": json.dumps([6.0, 6.0]),
            "数据长度": json.dumps([18, 18]),
            "换手率": json.dumps([0.1, 0.1]),
        },
    ]

    dummy_ts = ["2026-01-02 09:00:00"]
    vol_detail_rows = []
    slope_detail_rows = []

    for c in contracts:
        # Volatility label_0 rows
        for ts in joint_timestamps:
            vol_detail_rows.append({
                "标签": "label_0",
                "数据文件": f"{c}/label_0/df_0.feather",
                "初始动作": 0,
                "分箱索引": 0,
                "时间戳": ts,
                "单步奖励": 0.50,
            })
            vol_detail_rows.append({
                "标签": "label_0",
                "数据文件": f"{c}/label_0/df_0.feather",
                "初始动作": 0,
                "分箱索引": 1,
                "时间戳": ts,
                "单步奖励": 0.25,
            })
        for ts in other_timestamps:
            vol_detail_rows.append({
                "标签": "label_0",
                "数据文件": f"{c}/label_0/df_0.feather",
                "初始动作": 0,
                "分箱索引": 0,
                "时间戳": ts,
                "单步奖励": -0.25,
            })
            vol_detail_rows.append({
                "标签": "label_0",
                "数据文件": f"{c}/label_0/df_0.feather",
                "初始动作": 0,
                "分箱索引": 1,
                "时间戳": ts,
                "单步奖励": 0.3125,
            })
        # Volatility label_1 dummy row
        for ts in dummy_ts:
            vol_detail_rows.append({
                "标签": "label_1",
                "数据文件": f"{c}/label_1/df_0.feather",
                "初始动作": 0,
                "分箱索引": 0,
                "时间戳": ts,
                "单步奖励": 1.0,
            })
            vol_detail_rows.append({
                "标签": "label_1",
                "数据文件": f"{c}/label_1/df_0.feather",
                "初始动作": 0,
                "分箱索引": 1,
                "时间戳": ts,
                "单步奖励": 1.0,
            })

        # Slope label_0 rows
        for ts in joint_timestamps:
            slope_detail_rows.append({
                "标签": "label_0",
                "数据文件": f"{c}/label_0/df_0.feather",
                "初始动作": 0,
                "分箱索引": 0,
                "时间戳": ts,
                "单步奖励": 0.50,
            })
            slope_detail_rows.append({
                "标签": "label_0",
                "数据文件": f"{c}/label_0/df_0.feather",
                "初始动作": 0,
                "分箱索引": 1,
                "时间戳": ts,
                "单步奖励": 0.25,
            })
        # Slope label_1 rows
        for ts in other_timestamps:
            slope_detail_rows.append({
                "标签": "label_1",
                "数据文件": f"{c}/label_1/df_0.feather",
                "初始动作": 0,
                "分箱索引": 0,
                "时间戳": ts,
                "单步奖励": -0.25,
            })
            slope_detail_rows.append({
                "标签": "label_1",
                "数据文件": f"{c}/label_1/df_0.feather",
                "初始动作": 0,
                "分箱索引": 1,
                "时间戳": ts,
                "单步奖励": 0.3125,
            })
        for ts in dummy_ts:
            slope_detail_rows.append({
                "标签": "label_1",
                "数据文件": f"{c}/label_1/df_0.feather",
                "初始动作": 0,
                "分箱索引": 0,
                "时间戳": ts,
                "单步奖励": 1.0,
            })
            slope_detail_rows.append({
                "标签": "label_1",
                "数据文件": f"{c}/label_1/df_0.feather",
                "初始动作": 0,
                "分箱索引": 1,
                "时间戳": ts,
                "单步奖励": 1.0,
            })

    valid_labels = {
        "volatility": {
            c: {
                "label_0": all_timestamps,
                "label_1": dummy_ts,
            }
            for c in contracts
        },
        "slope": {
            c: {
                "label_0": joint_timestamps,
                "label_1": other_timestamps + dummy_ts,
            }
            for c in contracts
        },
    }

    candidate_root, valid_root = _setup_mock_environment(
        tmp_path,
        epoch_num=1,
        vol_analysis_rows=vol_analysis_rows,
        slope_analysis_rows=slope_analysis_rows,
        vol_detail_rows=vol_detail_rows,
        slope_detail_rows=slope_detail_rows,
        valid_labels=valid_labels,
    )

    config = SelectionConfig(
        num_labels=2,
        min_marginal_contracts=2,
        min_joint_contracts=2,
        min_positive_contract_ratio=0.5,
        min_mean_return=0.0,
        min_lcb=0.0,
        lcb_z=1.0,
        enable_conjunctive_gates=True,
    )
    selector = TwoDimensionalAgentSelector(config)
    artifacts = selector.select(candidate_root, valid_root)

    rankings = artifacts.candidate_rankings
    bin_0_row = rankings.filter(
        (pl.col("candidate_id") == "epoch_1:bin_0")
        & (pl.col("volatility_label") == "label_0")
        & (pl.col("slope_label") == "label_0")
    ).row(0, named=True)
    bin_1_row = rankings.filter(
        (pl.col("candidate_id") == "epoch_1:bin_1")
        & (pl.col("volatility_label") == "label_0")
        & (pl.col("slope_label") == "label_0")
    ).row(0, named=True)

    # bin_0 MUST fail marginal volatility gate
    assert not bin_0_row["eligible"]
    assert "marginal_volatility" in bin_0_row["rejection_reasons"] or "marginal" in bin_0_row["rejection_reasons"]

    # bin_1 MUST pass and be selected
    assert bin_1_row["eligible"]
    selected = artifacts.selected_slots.filter(
        (pl.col("volatility_label") == "label_0")
        & (pl.col("slope_label") == "label_0")
    ).row(0, named=True)
    assert selected["candidate_id"] == "epoch_1:bin_1"
    assert selected["kind"] == "model"


def test_traffic_tiering_applies_stricter_contract_threshold_to_core_slots(tmp_path: Path) -> None:
    """
    Traffic tiering:
    Slot has 5 contracts and 50 timestamps (>=10% of total timestamps and >= 5 contracts), so it is a core slot.
    Under core slot tiering, core_slot_min_joint_contracts = 6.
    The slot only has 5 contracts, so it MUST fail due to insufficient_joint_contract_coverage.
    """
    contracts = [f"fu000{i}" for i in range(1, 6)]  # 5 contracts
    timestamps = [f"2026-01-01 {hour:02d}:00:00" for hour in range(9, 19)]  # 10 timestamps each -> 50 total

    analysis_rows = [
        {
            "标签": "label_0",
            "初始动作": 0,
            "分箱索引": 0,
            "合约": json.dumps(contracts),
            "数据文件": json.dumps([f"{c}/label_0/df_0.feather" for c in contracts]),
            "奖励总和": json.dumps([2.0] * 5),
            "数据长度": json.dumps([11] * 5),
            "换手率": json.dumps([0.1] * 5),
        }
    ]

    detail_rows = []
    for c in contracts:
        for ts in timestamps:
            detail_rows.append({
                "标签": "label_0",
                "数据文件": f"{c}/label_0/df_0.feather",
                "初始动作": 0,
                "分箱索引": 0,
                "时间戳": ts,
                "单步奖励": 0.20,
            })

    valid_labels = {
        "volatility": {c: {"label_0": timestamps} for c in contracts},
        "slope": {c: {"label_0": timestamps} for c in contracts},
    }

    candidate_root, valid_root = _setup_mock_environment(
        tmp_path,
        epoch_num=1,
        vol_analysis_rows=analysis_rows,
        slope_analysis_rows=analysis_rows,
        vol_detail_rows=detail_rows,
        slope_detail_rows=detail_rows,
        valid_labels=valid_labels,
    )

    # Core slot requires 6 joint contracts; slot has 5 contracts.
    config = SelectionConfig(
        num_labels=1,
        min_marginal_contracts=1,
        min_joint_contracts=4,
        enable_traffic_tiering=True,
        core_slot_min_steps_ratio=0.10,
        core_slot_min_contracts=5,
        core_slot_min_joint_contracts=6,
        core_slot_min_positive_contract_ratio=0.70,
        peripheral_slot_min_joint_contracts=4,
        missing_joint_policy="empty_model",
    )
    selector = TwoDimensionalAgentSelector(config)
    artifacts = selector.select(candidate_root, valid_root)

    slot_0 = artifacts.manifest.slots[0]
    assert slot_0["traffic_tier"] == "core"
    assert slot_0["kind"] == "empty_model"
    assert "insufficient_joint_contract_coverage" in slot_0["selection_reason"]


def test_hard_stop_loss_gate_rejects_agent_with_excessive_stop_loss_episodes(tmp_path: Path) -> None:
    """
    Hard stop-loss gate:
    Candidate has position episode reaching loss rate 5% >= 3% threshold.
    Stop loss episode ratio = 100% > 10% tolerance.
    Under ADR-0059, candidate MUST be rejected by gate_hard_stop_loss.
    """
    contracts = ["fu2409", "fu2411"]
    timestamps = [f"2026-01-01 {hour:02d}:00:00" for hour in range(9, 13)]

    # 4 steps: sum of rewards = 1.0 - 5.0 + 5.0 + 0.0 = 1.0 per contract
    analysis_rows = [
        {
            "标签": "label_0",
            "初始动作": 0,
            "分箱索引": 0,
            "合约": json.dumps(contracts),
            "数据文件": json.dumps([f"{c}/label_0/df_0.feather" for c in contracts]),
            "奖励总和": json.dumps([1.0, 1.0]),
            "数据长度": json.dumps([5, 5]),
            "换手率": json.dumps([0.1, 0.1]),
        }
    ]

    detail_rows = []
    for c in contracts:
        # ts 09:00: position open, flat pnl
        detail_rows.append({
            "标签": "label_0",
            "数据文件": f"{c}/label_0/df_0.feather",
            "初始动作": 0,
            "分箱索引": 0,
            "时间戳": timestamps[0],
            "单步奖励": 1.0,
            "执行后仓位": 1.0,
            "标记价格": 3000.0,
            "浮动盈亏": 0.0,
            "单步实现盈亏": 0.0,
            "单步手续费": 1.0,
            "单步滑点": 0.0,
        })
        # ts 10:00: position open, deep loss (-150 on 1 hand of 3000 price -> loss_rate = 150/3000 = 5% > 3%)
        detail_rows.append({
            "标签": "label_0",
            "数据文件": f"{c}/label_0/df_0.feather",
            "初始动作": 0,
            "分箱索引": 0,
            "时间戳": timestamps[1],
            "单步奖励": -5.0,
            "执行后仓位": 1.0,
            "标记价格": 3000.0,
            "浮动盈亏": -150.0,
            "单步实现盈亏": 0.0,
            "单步手续费": 0.0,
            "单步滑点": 0.0,
        })
        # ts 11:00: close position
        detail_rows.append({
            "标签": "label_0",
            "数据文件": f"{c}/label_0/df_0.feather",
            "初始动作": 0,
            "分箱索引": 0,
            "时间戳": timestamps[2],
            "单步奖励": 5.0,
            "执行后仓位": 0.0,
            "标记价格": 3000.0,
            "浮动盈亏": 0.0,
            "单步实现盈亏": 20.0,
            "单步手续费": 1.0,
            "单步滑点": 0.0,
        })
        # ts 12:00: flat
        detail_rows.append({
            "标签": "label_0",
            "数据文件": f"{c}/label_0/df_0.feather",
            "初始动作": 0,
            "分箱索引": 0,
            "时间戳": timestamps[3],
            "单步奖励": 0.0,
            "执行后仓位": 0.0,
            "标记价格": 3000.0,
            "浮动盈亏": 0.0,
            "单步实现盈亏": 0.0,
            "单步手续费": 0.0,
            "单步滑点": 0.0,
        })

    valid_labels = {
        "volatility": {c: {"label_0": timestamps} for c in contracts},
        "slope": {c: {"label_0": timestamps} for c in contracts},
    }

    candidate_root, valid_root = _setup_mock_environment(
        tmp_path,
        epoch_num=1,
        vol_analysis_rows=analysis_rows,
        slope_analysis_rows=analysis_rows,
        vol_detail_rows=detail_rows,
        slope_detail_rows=detail_rows,
        valid_labels=valid_labels,
    )

    config = SelectionConfig(
        num_labels=1,
        min_marginal_contracts=2,
        min_joint_contracts=2,
        enable_hard_stop_loss_gate=True,
        hard_stop_loss_threshold=0.03,
        max_hard_stop_loss_episode_ratio=0.10,
    )
    selector = TwoDimensionalAgentSelector(config)
    artifacts = selector.select(candidate_root, valid_root)

    rankings = artifacts.candidate_rankings
    row = rankings.row(0, named=True)
    assert not row["eligible"]
    assert "hard_stop_loss" in row["rejection_reasons"]
    assert artifacts.selected_slots.row(0, named=True)["kind"] == "empty_model"


def test_daily_main_contract_sub_gate_rejects_agent_losing_on_main(tmp_path: Path) -> None:
    """
    Daily main contract sub-gate:
    Valid dataset contains contracts with prev_day_contract_role_tier.
    Candidate is profitable overall, but has negative return on Daily Main steps (tier == 1.0).
    Under ADR-0059, candidate MUST be rejected by gate_daily_main_contract.
    """
    contracts = ["fu2409", "fu2411"]
    timestamps = [f"2026-01-01 {hour:02d}:00:00" for hour in range(9, 13)]

    # fu2409 has 4 steps * (-2.0) = -8.0
    # fu2411 has 4 steps * 10.0 = 40.0
    analysis_rows = [
        {
            "标签": "label_0",
            "初始动作": 0,
            "分箱索引": 0,
            "合约": json.dumps(contracts),
            "数据文件": json.dumps([f"{c}/label_0/df_0.feather" for c in contracts]),
            "奖励总和": json.dumps([-8.0, 40.0]),
            "数据长度": json.dumps([5, 5]),
            "换手率": json.dumps([0.1, 0.1]),
        }
    ]

    # fu2409 is main contract (tier=1.0), but detail rows have net loss (-2.0 per step)
    # fu2411 is sub contract (tier=0.5), has huge net profit (+10.0 per step)
    detail_rows = []
    for ts in timestamps:
        detail_rows.append({
            "标签": "label_0",
            "数据文件": "fu2409/label_0/df_0.feather",
            "初始动作": 0,
            "分箱索引": 0,
            "时间戳": ts,
            "单步奖励": -2.0,
            "单步实现盈亏": -2.0,
            "单步手续费": 0.0,
            "单步滑点": 0.0,
        })
        detail_rows.append({
            "标签": "label_0",
            "数据文件": "fu2411/label_0/df_0.feather",
            "初始动作": 0,
            "分箱索引": 0,
            "时间戳": ts,
            "单步奖励": 10.0,
            "单步实现盈亏": 10.0,
            "单步手续费": 0.0,
            "单步滑点": 0.0,
        })

    valid_labels = {
        "volatility": {c: {"label_0": timestamps} for c in contracts},
        "slope": {c: {"label_0": timestamps} for c in contracts},
    }

    candidate_root, valid_root = _setup_mock_environment(
        tmp_path,
        epoch_num=1,
        vol_analysis_rows=analysis_rows,
        slope_analysis_rows=analysis_rows,
        vol_detail_rows=detail_rows,
        slope_detail_rows=detail_rows,
        valid_labels=valid_labels,
    )

    # Add contract feather files with prev_day_contract_role_tier in valid_root
    for c, tier in [("fu2409", 1.0), ("fu2411", 0.5)]:
        pl.DataFrame({
            "timestamp": timestamps,
            "prev_day_contract_role_tier": [tier] * len(timestamps),
        }).write_ipc(valid_root / f"{c}.feather")

    config = SelectionConfig(
        num_labels=1,
        min_marginal_contracts=2,
        min_joint_contracts=2,
        enable_main_contract_sub_gate=True,
        min_main_steps=2,
    )
    selector = TwoDimensionalAgentSelector(config)
    artifacts = selector.select(candidate_root, valid_root)

    rankings = artifacts.candidate_rankings
    row = rankings.row(0, named=True)
    assert not row["eligible"]
    assert "daily_main_contract" in row["rejection_reasons"]
    assert artifacts.selected_slots.row(0, named=True)["kind"] == "empty_model"


def test_detail_csv_with_mixed_numeric_schema_parses_without_compute_error(tmp_path: Path) -> None:
    """
    Ensure detail CSVs where realized_pnl / commission_fee / slippage contain
    integers followed by floats (e.g. 0 followed by 0.0) do not trigger Polars ComputeError
    during header discovery and detail loading.
    """
    contracts = ["fu2409"]
    timestamps = [f"2026-01-01 {idx:04d}" for idx in range(120)]

    analysis_rows = [
        {
            "标签": "label_0",
            "初始动作": 0,
            "分箱索引": 0,
            "合约": json.dumps(contracts),
            "数据文件": json.dumps([f"{c}/label_0/df_0.feather" for c in contracts]),
            "奖励总和": json.dumps([120.0]),
            "数据长度": json.dumps([121]),
            "换手率": json.dumps([0.1]),
        }
    ]

    # Create 120 detail rows: first 105 rows have integer 0 in '单步实现盈亏',
    # row 106 has float '0.0' or '1.5'
    detail_csv_path = tmp_path / "mock_detail.csv"
    headers = [
        "标签", "数据文件", "初始动作", "分箱索引", "时间戳", "单步奖励",
        "单步实现盈亏", "单步手续费", "单步滑点"
    ]
    lines = [",".join(headers)]
    for i in range(120):
        ts = timestamps[i]
        pnl_val = "0" if i < 105 else "0.0"
        lines.append(f"label_0,fu2409/label_0/df_0.feather,0,0,{ts},1.0,{pnl_val},0.0,0.0")

    candidate_root = tmp_path / "candidate"
    valid_root = tmp_path / "valid"
    epoch_path = candidate_root / "epoch_1"
    (epoch_path / "trained_model.pkl").parent.mkdir(parents=True, exist_ok=True)
    (epoch_path / "trained_model.pkl").write_bytes(b"checkpoint")

    vol_path = epoch_path / "volatility"
    slope_path = epoch_path / "slope"
    vol_path.mkdir(parents=True, exist_ok=True)
    slope_path.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(analysis_rows).write_csv(vol_path / "analysis_result.csv")
    pl.DataFrame(analysis_rows).write_csv(slope_path / "analysis_result.csv")

    detail_content = "\n".join(lines)
    (vol_path / "trading_action_detail_epoch_1.csv").write_text(detail_content, encoding="utf-8")
    (slope_path / "trading_action_detail_epoch_1.csv").write_text(detail_content, encoding="utf-8")

    for label_type in ["volatility", "slope"]:
        label_dir = valid_root / label_type / "fu2409" / "label_0"
        label_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"timestamp": timestamps}).write_ipc(label_dir / "df_0.feather")

    config = SelectionConfig(
        num_labels=1,
        min_marginal_contracts=1,
        min_joint_contracts=1,
    )
    selector = TwoDimensionalAgentSelector(config)
    artifacts = selector.select(candidate_root, valid_root)
    assert artifacts.selected_slots.height > 0


def test_ensemble_traffic_coverage_fail_fast_and_sparse_bypass(tmp_path: Path) -> None:
    contracts = ["fu2409"]
    timestamps = [f"2026-01-01 {hour:02d}:00:00" for hour in range(9, 13)]
    analysis_rows = [{
        "标签": "label_0",
        "初始动作": 0,
        "分箱索引": 0,
        "合约": json.dumps(contracts),
        "数据文件": json.dumps([f"{c}/label_0/df_0.feather" for c in contracts]),
        "奖励总和": json.dumps([0.80]),
        "数据长度": json.dumps([len(timestamps) + 1]),
        "换手率": json.dumps([0.10]),
    }]
    detail_rows = [{
        "标签": "label_0",
        "数据文件": "fu2409/label_0/df_0.feather",
        "初始动作": 0,
        "分箱索引": 0,
        "时间戳": ts,
        "单步奖励": 0.20,
    } for ts in timestamps]

    valid_labels = {
        "volatility": {c: {"label_0": timestamps} for c in contracts},
        "slope": {c: {"label_0": timestamps} for c in contracts},
    }

    candidate_root, valid_root = _setup_mock_environment(
        tmp_path,
        epoch_num=1,
        vol_analysis_rows=analysis_rows,
        slope_analysis_rows=analysis_rows,
        vol_detail_rows=detail_rows,
        slope_detail_rows=detail_rows,
        valid_labels=valid_labels,
    )

    # Requires 4 joint contracts, but only 1 exists -> Slot 0 becomes empty_model -> coverage = 0.0%
    strict_config = SelectionConfig(
        num_labels=1,
        min_marginal_contracts=1,
        min_joint_contracts=4,
        missing_joint_policy="empty_model",
        min_ensemble_traffic_coverage=0.40,
        allow_sparse_ensemble=False,
    )
    with pytest.raises(RuntimeError, match="Ensemble traffic coverage 0.00% is below required minimum"):
        TwoDimensionalAgentSelector(strict_config).select(candidate_root, valid_root)

    # With allow_sparse_ensemble=True, it succeeds
    sparse_config = SelectionConfig(
        num_labels=1,
        min_marginal_contracts=1,
        min_joint_contracts=4,
        missing_joint_policy="empty_model",
        min_ensemble_traffic_coverage=0.40,
        allow_sparse_ensemble=True,
    )
    artifacts = TwoDimensionalAgentSelector(sparse_config).select(candidate_root, valid_root)
    assert artifacts.selected_slots.row(0, named=True)["kind"] == "empty_model"
    assert artifacts.manifest.artifacts.traffic_coverage_ratio == 0.0
