import json
from pathlib import Path

import polars as pl
import pytest
import torch

from analysis.pick_agent.FineFT_two_dimensional_agent_selector import (
    SelectionConfig,
    TwoDimensionalAgentSelector,
    assemble_and_save_ensemble,
)
from model.low_level import ensemble_Qnet


def test_select_ignores_zero_transition_detail_rows(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidate"
    valid_root = tmp_path / "valid"
    epoch_path = candidate_root / "epoch_1"
    (epoch_path / "trained_model.pkl").parent.mkdir(parents=True)
    (epoch_path / "trained_model.pkl").write_bytes(b"checkpoint")

    analysis_row = {
        "标签": "label_0",
        "初始动作": 0,
        "分箱索引": 0,
        "合约": json.dumps(["fu0001", "fu0001"]),
        "数据文件": json.dumps(
            [
                "fu0001/label_0/df_0.feather",
                "fu0001/label_0/df_1.feather",
            ]
        ),
        "奖励总和": json.dumps([2.5, 0.0]),
        "数据长度": json.dumps([2, 1]),
        "换手率": json.dumps([0.0, 0.0]),
    }
    detail_rows = [
        {
            "标签": "label_0",
            "数据文件": "fu0001/label_0/df_0.feather",
            "初始动作": 0,
            "分箱索引": 0,
            "时间戳": "2026-01-01 09:00:00",
            "单步奖励": 2.5,
        },
        {
            "标签": "label_0",
            "数据文件": "fu0001/label_0/df_1.feather",
            "初始动作": 0,
            "分箱索引": 0,
            "时间戳": "2026-01-01 10:00:00",
            "单步奖励": 0.0,
        },
    ]
    for label_type in ("volatility", "slope"):
        result_path = epoch_path / label_type
        result_path.mkdir(parents=True)
        pl.DataFrame([analysis_row]).write_csv(
            result_path / "analysis_result.csv"
        )
        pl.DataFrame(detail_rows).write_csv(
            result_path / "trading_action_detail_epoch_1.csv"
        )

        label_path = valid_root / label_type / "fu0001" / "label_0"
        label_path.mkdir(parents=True)
        pl.DataFrame(
            {
                "timestamp": [
                    "2026-01-01 09:00:00",
                    "2026-01-01 09:30:00",
                ]
            }
        ).write_ipc(label_path / "df_0.feather")
        pl.DataFrame({"timestamp": ["2026-01-01 10:00:00"]}).write_ipc(
            label_path / "df_1.feather"
        )

    artifacts = TwoDimensionalAgentSelector(
        SelectionConfig(
            num_labels=1,
            min_marginal_contracts=1,
            min_joint_contracts=1,
        )
    ).select(candidate_root, valid_root)

    assert artifacts.marginal_metrics["mean_return_per_step"].to_list() == [
        2.5,
        2.5,
    ]
    assert artifacts.marginal_metrics["step_count"].to_list() == [1, 1]
    assert artifacts.joint_metrics["mean_return_per_step"].to_list() == [2.5, 2.5]
    assert artifacts.joint_metrics["step_count"].to_list() == [1, 1]

    output_dir = tmp_path / "artifacts_out"
    written_paths = artifacts.write(output_dir)
    assert all(p.is_file() for p in written_paths.values())


def test_select_selects_eligible_model_slot(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidate"
    valid_root = tmp_path / "valid"
    epoch_path = candidate_root / "epoch_1"
    (epoch_path / "trained_model.pkl").parent.mkdir(parents=True)
    (epoch_path / "trained_model.pkl").write_bytes(b"checkpoint")

    analysis_row = {
        "标签": "label_0",
        "初始动作": 0,
        "分箱索引": 0,
        "合约": json.dumps(["fu0001", "fu0002"]),
        "数据文件": json.dumps(
            [
                "fu0001/label_0/df_0.feather",
                "fu0002/label_0/df_0.feather",
            ]
        ),
        "奖励总和": json.dumps([5.0, 5.0]),
        "数据长度": json.dumps([3, 3]),
        "换手率": json.dumps([0.1, 0.1]),
    }
    detail_rows = [
        {
            "标签": "label_0",
            "数据文件": "fu0001/label_0/df_0.feather",
            "初始动作": 0,
            "分箱索引": 0,
            "时间戳": "2026-01-01 09:00:00",
            "单步奖励": 2.5,
        },
        {
            "标签": "label_0",
            "数据文件": "fu0001/label_0/df_0.feather",
            "初始动作": 0,
            "分箱索引": 0,
            "时间戳": "2026-01-01 09:30:00",
            "单步奖励": 2.5,
        },
        {
            "标签": "label_0",
            "数据文件": "fu0002/label_0/df_0.feather",
            "初始动作": 0,
            "分箱索引": 0,
            "时间戳": "2026-01-01 09:00:00",
            "单步奖励": 2.5,
        },
        {
            "标签": "label_0",
            "数据文件": "fu0002/label_0/df_0.feather",
            "初始动作": 0,
            "分箱索引": 0,
            "时间戳": "2026-01-01 09:30:00",
            "单步奖励": 2.5,
        },
    ]
    for label_type in ("volatility", "slope"):
        result_path = epoch_path / label_type
        result_path.mkdir(parents=True)
        pl.DataFrame([analysis_row]).write_csv(
            result_path / "analysis_result.csv"
        )
        pl.DataFrame(detail_rows).write_csv(
            result_path / "trading_action_detail_epoch_1.csv"
        )

        for contract in ("fu0001", "fu0002"):
            label_path = valid_root / label_type / contract / "label_0"
            label_path.mkdir(parents=True)
            pl.DataFrame(
                {
                    "timestamp": [
                        "2026-01-01 09:00:00",
                        "2026-01-01 09:30:00",
                        "2026-01-01 10:00:00",
                    ]
                }
            ).write_ipc(label_path / "df_0.feather")

    artifacts = TwoDimensionalAgentSelector(
        SelectionConfig(
            num_labels=1,
            min_marginal_contracts=2,
            min_joint_contracts=2,
            min_positive_contract_ratio=0.5,
            min_mean_return=0.0,
            min_lcb=0.0,
        )
    ).select(candidate_root, valid_root)

    assert artifacts.selected_slots["kind"].to_list() == ["model"]
    assert artifacts.selected_slots["candidate_id"].to_list() == ["epoch_1:bin_0"]
    assert artifacts.selected_slots["pair_score"].to_list()[0] > 0.0


def test_select_handles_slope_marginal_fallback(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidate"
    valid_root = tmp_path / "valid"
    epoch_path = candidate_root / "epoch_1"
    (epoch_path / "trained_model.pkl").parent.mkdir(parents=True)
    (epoch_path / "trained_model.pkl").write_bytes(b"checkpoint")

    # Set up 2 labels (2x2 grid = 4 slots) where only label_0 in volatility and slope has data
    analysis_rows = [
        {
            "标签": "label_0",
            "初始动作": 0,
            "分箱索引": 0,
            "合约": json.dumps(["fu0001", "fu0002"]),
            "数据文件": json.dumps(
                [
                    "fu0001/label_0/df_0.feather",
                    "fu0002/label_0/df_0.feather",
                ]
            ),
            "奖励总和": json.dumps([4.0, 4.0]),
            "数据长度": json.dumps([3, 3]),
            "换手率": json.dumps([0.1, 0.1]),
        },
        {
            "标签": "label_1",
            "初始动作": 0,
            "分箱索引": 0,
            "合约": json.dumps(["fu0001", "fu0002"]),
            "数据文件": json.dumps(
                [
                    "fu0001/label_1/df_0.feather",
                    "fu0002/label_1/df_0.feather",
                ]
            ),
            "奖励总和": json.dumps([2.0, 2.0]),
            "数据长度": json.dumps([3, 3]),
            "换手率": json.dumps([0.1, 0.1]),
        },
    ]
    # In valid, timestamps for label_0 in volatility match label_0 in slope;
    # timestamps for label_1 in volatility match label_1 in slope.
    # So slot (label_0, label_1) has 0 joint rows!
    vol_detail_rows = [
        {"标签": "label_0", "数据文件": "fu0001/label_0/df_0.feather", "初始动作": 0, "分箱索引": 0, "时间戳": "2026-01-01 09:00:00", "单步奖励": 2.0},
        {"标签": "label_0", "数据文件": "fu0001/label_0/df_0.feather", "初始动作": 0, "分箱索引": 0, "时间戳": "2026-01-01 09:30:00", "单步奖励": 2.0},
        {"标签": "label_0", "数据文件": "fu0002/label_0/df_0.feather", "初始动作": 0, "分箱索引": 0, "时间戳": "2026-01-01 09:00:00", "单步奖励": 2.0},
        {"标签": "label_0", "数据文件": "fu0002/label_0/df_0.feather", "初始动作": 0, "分箱索引": 0, "时间戳": "2026-01-01 09:30:00", "单步奖励": 2.0},
        {"标签": "label_1", "数据文件": "fu0001/label_1/df_0.feather", "初始动作": 0, "分箱索引": 0, "时间戳": "2026-01-02 09:00:00", "单步奖励": 1.0},
        {"标签": "label_1", "数据文件": "fu0001/label_1/df_0.feather", "初始动作": 0, "分箱索引": 0, "时间戳": "2026-01-02 09:30:00", "单步奖励": 1.0},
        {"标签": "label_1", "数据文件": "fu0002/label_1/df_0.feather", "初始动作": 0, "分箱索引": 0, "时间戳": "2026-01-02 09:00:00", "单步奖励": 1.0},
        {"标签": "label_1", "数据文件": "fu0002/label_1/df_0.feather", "初始动作": 0, "分箱索引": 0, "时间戳": "2026-01-02 09:30:00", "单步奖励": 1.0},
    ]

    for label_type in ("volatility", "slope"):
        result_path = epoch_path / label_type
        result_path.mkdir(parents=True)
        pl.DataFrame(analysis_rows).write_csv(
            result_path / "analysis_result.csv"
        )
        pl.DataFrame(vol_detail_rows).write_csv(
            result_path / "trading_action_detail_epoch_1.csv"
        )

        for contract in ("fu0001", "fu0002"):
            p0 = valid_root / label_type / contract / "label_0"
            p0.mkdir(parents=True)
            pl.DataFrame({"timestamp": ["2026-01-01 09:00:00", "2026-01-01 09:30:00", "2026-01-01 10:00:00"]}).write_ipc(p0 / "df_0.feather")

            p1 = valid_root / label_type / contract / "label_1"
            p1.mkdir(parents=True)
            pl.DataFrame({"timestamp": ["2026-01-02 09:00:00", "2026-01-02 09:30:00", "2026-01-02 10:00:00"]}).write_ipc(p1 / "df_0.feather")

    # When missing_joint_policy="slope_marginal_best", slot (label_0, label_1) should use fallback
    selector_fallback = TwoDimensionalAgentSelector(
        SelectionConfig(
            num_labels=2,
            min_marginal_contracts=2,
            min_joint_contracts=2,
            min_positive_contract_ratio=0.5,
            min_mean_return=0.0,
            min_lcb=0.0,
            missing_joint_policy="slope_marginal_best",
        )
    )
    artifacts = selector_fallback.select(candidate_root, valid_root)
    slots = artifacts.selected_slots.to_dicts()
    slot_01 = [s for s in slots if s["volatility_label"] == "label_0" and s["slope_label"] == "label_1"][0]
    assert slot_01["kind"] == "model"
    assert slot_01["selection_reason"] == "fallback_from_slope_marginal"


def test_select_raises_on_invalid_label(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidate"
    valid_root = tmp_path / "valid"
    epoch_path = candidate_root / "epoch_1"
    (epoch_path / "trained_model.pkl").parent.mkdir(parents=True)
    (epoch_path / "trained_model.pkl").write_bytes(b"checkpoint")

    analysis_row = {
        "标签": "label_99",
        "初始动作": 0,
        "分箱索引": 0,
        "合约": json.dumps(["fu0001"]),
        "数据文件": json.dumps(["fu0001/label_99/df_0.feather"]),
        "奖励总和": json.dumps([2.5]),
        "数据长度": json.dumps([2]),
        "换手率": json.dumps([0.0]),
    }
    for label_type in ("volatility", "slope"):
        result_path = epoch_path / label_type
        result_path.mkdir(parents=True)
        pl.DataFrame([analysis_row]).write_csv(
            result_path / "analysis_result.csv"
        )
        pl.DataFrame([]).write_csv(
            result_path / "trading_action_detail_epoch_1.csv"
        )

    selector = TwoDimensionalAgentSelector(SelectionConfig(num_labels=2))
    with pytest.raises(ValueError, match="invalid or unexpected label"):
        selector.select(candidate_root, valid_root)


def test_assemble_and_save_ensemble_includes_flat_empty_model(
    tmp_path: Path,
) -> None:
    n_states = 3
    n_actions = 5
    hidden_nodes = 4
    time_info_dim = 2
    source = ensemble_Qnet(
        n_states,
        n_actions,
        hidden_nodes,
        time_info_dim,
        ensemble_number=1,
    )
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(0.25)
    source_path = tmp_path / "trained_model.pkl"
    torch.save(source.state_dict(), source_path)

    slots = pl.DataFrame(
        [
            {
                "slot_id": 0,
                "kind": "model",
                "model_path": str(source_path),
                "bin_index": 0,
            },
            {
                "slot_id": 1,
                "kind": "empty_model",
                "model_path": None,
                "bin_index": None,
            },
        ],
        schema={
            "slot_id": pl.Int64,
            "kind": pl.String,
            "model_path": pl.String,
            "bin_index": pl.Int64,
        },
    )
    output_path = tmp_path / "model.pth"

    assemble_and_save_ensemble(
        slots,
        output_path,
        n_states=n_states,
        n_actions=n_actions,
        hidden_nodes=hidden_nodes,
        time_info_dim=time_info_dim,
    )

    assembled = ensemble_Qnet(
        n_states,
        n_actions,
        hidden_nodes,
        time_info_dim,
        ensemble_number=2,
    )
    assembled.load_state_dict(torch.load(output_path, weights_only=True))
    assert torch.equal(
        assembled.qnet_list[0].fc1.weight,
        source.qnet_list[0].fc1.weight,
    )

    batch_size = 3
    q_values = assembled(
        torch.randn(batch_size, n_states),
        torch.randn(batch_size, time_info_dim),
        torch.zeros(batch_size, 1),
        torch.ones(batch_size, n_actions),
        torch.randn(batch_size, 4),
    )
    assert q_values[:, 1].argmax(dim=1).tolist() == [n_actions // 2] * batch_size


def test_contract_weighting_step_weighted_vs_contract_equal(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidate"
    valid_root = tmp_path / "valid"
    epoch_path = candidate_root / "epoch_1"
    (epoch_path / "trained_model.pkl").parent.mkdir(parents=True)
    (epoch_path / "trained_model.pkl").write_bytes(b"checkpoint")

    # fu0001: 10 steps (length 11), reward 10.0 -> return 1.0/step
    # fu0002: 90 steps (length 91), reward 0.0 -> return 0.0/step
    analysis_row = {
        "标签": "label_0",
        "初始动作": 0,
        "分箱索引": 0,
        "合约": json.dumps(["fu0001", "fu0002"]),
        "数据文件": json.dumps(
            [
                "fu0001/label_0/df_0.feather",
                "fu0002/label_0/df_0.feather",
            ]
        ),
        "奖励总和": json.dumps([10.0, 0.0]),
        "数据长度": json.dumps([11, 91]),
        "换手率": json.dumps([0.1, 0.1]),
    }

    fu0001_detail = [
        {
            "标签": "label_0",
            "数据文件": "fu0001/label_0/df_0.feather",
            "初始动作": 0,
            "分箱索引": 0,
            "时间戳": f"2026-01-01 {i:02d}:00:00",
            "单步奖励": 1.0,
        }
        for i in range(10)
    ]
    fu0002_detail = [
        {
            "标签": "label_0",
            "数据文件": "fu0002/label_0/df_0.feather",
            "初始动作": 0,
            "分箱索引": 0,
            "时间戳": f"2026-01-02 {i // 60:02d}:{i % 60:02d}:00",
            "单步奖励": 0.0,
        }
        for i in range(90)
    ]
    detail_rows = fu0001_detail + fu0002_detail

    for label_type in ("volatility", "slope"):
        result_path = epoch_path / label_type
        result_path.mkdir(parents=True)
        pl.DataFrame([analysis_row]).write_csv(
            result_path / "analysis_result.csv"
        )
        pl.DataFrame(detail_rows).write_csv(
            result_path / "trading_action_detail_epoch_1.csv"
        )

        p1 = valid_root / label_type / "fu0001" / "label_0"
        p1.mkdir(parents=True)
        pl.DataFrame(
            {"timestamp": [f"2026-01-01 {i:02d}:00:00" for i in range(11)]}
        ).write_ipc(p1 / "df_0.feather")

        p2 = valid_root / label_type / "fu0002" / "label_0"
        p2.mkdir(parents=True)
        pl.DataFrame(
            {"timestamp": [f"2026-01-02 {i // 60:02d}:{i % 60:02d}:00" for i in range(91)]}
        ).write_ipc(p2 / "df_0.feather")

    # step_weighted (default)
    selector_weighted = TwoDimensionalAgentSelector(
        SelectionConfig(
            num_labels=1,
            min_marginal_contracts=2,
            min_joint_contracts=2,
            contract_weighting="step_weighted",
        )
    )
    artifacts_weighted = selector_weighted.select(candidate_root, valid_root)
    # (1.0 * 10 + 0.0 * 90) / 100 = 0.10
    assert pytest.approx(
        artifacts_weighted.marginal_metrics["mean_return_per_step"][0], abs=1e-6
    ) == 0.10

    # contract_equal
    selector_equal = TwoDimensionalAgentSelector(
        SelectionConfig(
            num_labels=1,
            min_marginal_contracts=2,
            min_joint_contracts=2,
            contract_weighting="contract_equal",
        )
    )
    artifacts_equal = selector_equal.select(candidate_root, valid_root)
    # (1.0 + 0.0) / 2 = 0.50
    assert pytest.approx(
        artifacts_equal.marginal_metrics["mean_return_per_step"][0], abs=1e-6
    ) == 0.50


def test_select_filters_isolated_short_slices_when_min_slice_steps_set(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    valid_root = tmp_path / "valid"
    epoch_path = candidate_root / "epoch_1"
    (epoch_path / "trained_model.pkl").parent.mkdir(parents=True)
    (epoch_path / "trained_model.pkl").write_bytes(b"checkpoint")

    # df_0: length 11 (10 steps) -> filtered by min_slice_steps=30
    # df_1: length 51 (50 steps) -> kept
    analysis_row = {
        "标签": "label_0",
        "初始动作": 0,
        "分箱索引": 0,
        "合约": json.dumps(["fu0001", "fu0001"]),
        "数据文件": json.dumps(
            [
                "fu0001/label_0/df_0.feather",
                "fu0001/label_0/df_1.feather",
            ]
        ),
        "奖励总和": json.dumps([-100.0, 50.0]),
        "数据长度": json.dumps([11, 51]),
        "换手率": json.dumps([0.1, 0.1]),
    }
    df_0_detail = [
        {
            "标签": "label_0",
            "数据文件": "fu0001/label_0/df_0.feather",
            "初始动作": 0,
            "分箱索引": 0,
            "时间戳": f"2026-01-01 {i:02d}:00:00",
            "单步奖励": -10.0,
        }
        for i in range(10)
    ]
    df_1_detail = [
        {
            "标签": "label_0",
            "数据文件": "fu0001/label_0/df_1.feather",
            "初始动作": 0,
            "分箱索引": 0,
            "时间戳": f"2026-01-02 {i // 60:02d}:{i % 60:02d}:00",
            "单步奖励": 1.0,
        }
        for i in range(50)
    ]
    detail_rows = df_0_detail + df_1_detail

    for label_type in ("volatility", "slope"):
        result_path = epoch_path / label_type
        result_path.mkdir(parents=True)
        pl.DataFrame([analysis_row]).write_csv(
            result_path / "analysis_result.csv"
        )
        pl.DataFrame(detail_rows).write_csv(
            result_path / "trading_action_detail_epoch_1.csv"
        )

        label_path = valid_root / label_type / "fu0001" / "label_0"
        label_path.mkdir(parents=True)
        pl.DataFrame(
            {"timestamp": [f"2026-01-01 {i:02d}:00:00" for i in range(11)]}
        ).write_ipc(label_path / "df_0.feather")
        pl.DataFrame(
            {"timestamp": [f"2026-01-02 {i // 60:02d}:{i % 60:02d}:00" for i in range(51)]}
        ).write_ipc(label_path / "df_1.feather")

    selector = TwoDimensionalAgentSelector(
        SelectionConfig(
            num_labels=1,
            min_marginal_contracts=1,
            min_joint_contracts=1,
            min_slice_steps=30,
        )
    )
    artifacts = selector.select(candidate_root, valid_root)

    # df_0 (10 steps) should be excluded; only df_1 (50 steps, reward 50 -> return 1.0/step) remains
    assert artifacts.marginal_metrics["step_count"].to_list() == [50, 50]
    assert artifacts.marginal_metrics["mean_return_per_step"].to_list() == [1.0, 1.0]
    assert artifacts.joint_metrics["step_count"].to_list() == [50, 50]
    assert artifacts.joint_metrics["mean_return_per_step"].to_list() == [1.0, 1.0]


def test_invalid_contract_weighting_or_min_slice_steps() -> None:
    with pytest.raises(ValueError, match="unsupported contract_weighting"):
        TwoDimensionalAgentSelector(
            SelectionConfig(contract_weighting="invalid_mode")
        )
    with pytest.raises(ValueError, match="min_slice_steps must be non-negative"):
        TwoDimensionalAgentSelector(
            SelectionConfig(min_slice_steps=-5)
        )


def test_worst_initial_position_step_weighted_and_volatility_tiered(
    tmp_path: Path,
) -> None:
    candidate_root = tmp_path / "candidate"
    valid_root = tmp_path / "valid"
    epoch_path = candidate_root / "epoch_1"
    (epoch_path / "trained_model.pkl").parent.mkdir(parents=True)
    (epoch_path / "trained_model.pkl").write_bytes(b"checkpoint")

    # 4 labels grid (4x4). We test label_3 (high vol).
    # Contract 1: 10 steps (length 11), reward -20.0 -> -2.0/step for action 0
    # Contract 2: 90 steps (length 91), reward 90.0 -> +1.0/step for action 0
    # Weighted position_return = (-20.0 + 90.0)/100 = +0.70
    # Unweighted position_return = (-2.0 + 1.0)/2 = -0.50
    analysis_row_v3_act0 = {
        "标签": "label_3",
        "初始动作": 0,
        "分箱索引": 0,
        "合约": json.dumps(["fu0001", "fu0002"]),
        "数据文件": json.dumps(
            [
                "fu0001/label_3/df_0.feather",
                "fu0002/label_3/df_0.feather",
            ]
        ),
        "奖励总和": json.dumps([-20.0, 90.0]),
        "数据长度": json.dumps([11, 91]),
        "换手率": json.dumps([0.1, 0.1]),
    }
    # For action 1..4, normal positive returns
    analysis_rows = [analysis_row_v3_act0]
    for act in range(1, 5):
        analysis_rows.append(
            {
                "标签": "label_3",
                "初始动作": act,
                "分箱索引": 0,
                "合约": json.dumps(["fu0001", "fu0002"]),
                "数据文件": json.dumps(
                    [
                        "fu0001/label_3/df_0.feather",
                        "fu0002/label_3/df_0.feather",
                    ]
                ),
                "奖励总和": json.dumps([10.0, 90.0]),
                "数据长度": json.dumps([11, 91]),
                "换手率": json.dumps([0.1, 0.1]),
            }
        )
    # Also add labels 0..2 to satisfy candidate coverage
    for l_idx in range(3):
        for act in range(5):
            analysis_rows.append(
                {
                    "标签": f"label_{l_idx}",
                    "初始动作": act,
                    "分箱索引": 0,
                    "合约": json.dumps(["fu0001", "fu0002"]),
                    "数据文件": json.dumps(
                        [
                            f"fu0001/label_{l_idx}/df_0.feather",
                            f"fu0002/label_{l_idx}/df_0.feather",
                        ]
                    ),
                    "奖励总和": json.dumps([10.0, 90.0]),
                    "数据长度": json.dumps([11, 91]),
                    "换手率": json.dumps([0.1, 0.1]),
                }
            )

    detail_rows = []
    for l_idx in range(4):
        for act in range(5):
            for i in range(10):
                r_val = -2.0 if (l_idx == 3 and act == 0) else 1.0
                detail_rows.append(
                    {
                        "标签": f"label_{l_idx}",
                        "数据文件": f"fu0001/label_{l_idx}/df_0.feather",
                        "初始动作": act,
                        "分箱索引": 0,
                        "时间戳": f"2026-01-0{l_idx+1} {i:02d}:00:00",
                        "单步奖励": r_val,
                    }
                )
            for i in range(90):
                detail_rows.append(
                    {
                        "标签": f"label_{l_idx}",
                        "数据文件": f"fu0002/label_{l_idx}/df_0.feather",
                        "初始动作": act,
                        "分箱索引": 0,
                        "时间戳": f"2026-01-1{l_idx+1} {i // 60:02d}:{i % 60:02d}:00",
                        "单步奖励": 1.0,
                    }
                )

    for label_type in ("volatility", "slope"):
        result_path = epoch_path / label_type
        result_path.mkdir(parents=True)
        pl.DataFrame(analysis_rows).write_csv(
            result_path / "analysis_result.csv"
        )
        pl.DataFrame(detail_rows).write_csv(
            result_path / "trading_action_detail_epoch_1.csv"
        )

        for l_idx in range(4):
            p1 = valid_root / label_type / "fu0001" / f"label_{l_idx}"
            p1.mkdir(parents=True)
            pl.DataFrame(
                {"timestamp": [f"2026-01-0{l_idx+1} {i:02d}:00:00" for i in range(11)]}
            ).write_ipc(p1 / "df_0.feather")

            p2 = valid_root / label_type / "fu0002" / f"label_{l_idx}"
            p2.mkdir(parents=True)
            pl.DataFrame(
                {"timestamp": [f"2026-01-1{l_idx+1} {i // 60:02d}:{i % 60:02d}:00" for i in range(91)]}
            ).write_ipc(p2 / "df_0.feather")

    # Test step-weighted position_return
    selector = TwoDimensionalAgentSelector(
        SelectionConfig(
            num_labels=4,
            min_marginal_contracts=1,
            min_joint_contracts=1,
            min_worst_initial_position_return=-1.0,
            min_worst_initial_position_return_v3=-5.0,
            contract_weighting="step_weighted",
        )
    )
    artifacts = selector.select(candidate_root, valid_root)

    # In label_3 volatility marginal, action 0 weighted position_return = 0.70
    l3_marg = artifacts.marginal_metrics.filter(
        (pl.col("label_type") == "volatility") & (pl.col("label") == "label_3")
    )
    assert pytest.approx(l3_marg["worst_initial_position_return"][0], abs=1e-5) == 0.70


def test_cross_data_vs_slope_marginal_selection(tmp_path: Path) -> None:
    """Verify that slots with cross data select by cross-data profit,

    while slots without cross data select by slope profit.
    """
    candidate_root = tmp_path / "candidate"
    valid_root = tmp_path / "valid"

    # Set up 2 candidates: epoch_1:bin_0 and epoch_2:bin_0
    for ep in (1, 2):
        ep_path = candidate_root / f"epoch_{ep}"
        (ep_path / "trained_model.pkl").parent.mkdir(parents=True)
        (ep_path / "trained_model.pkl").write_bytes(b"checkpoint")

    # Timestamps in valid:
    # label_0 in vol and label_0 in slope have matching timestamps -> slot (0, 0) HAS cross data.
    # label_0 in vol and label_1 in slope have NO matching timestamps -> slot (0, 1) has NO cross data.
    # label_1 in vol and label_0 in slope have NO matching timestamps -> slot (1, 0) has NO cross data.
    # label_1 in vol and label_1 in slope have matching timestamps -> slot (1, 1) HAS cross data.
    for label_type in ("volatility", "slope"):
        for l_idx, ts_day in [(0, "01"), (1, "02")]:
            p = valid_root / label_type / "fu0001" / f"label_{l_idx}"
            p.mkdir(parents=True)
            pl.DataFrame(
                {"timestamp": [f"2026-01-{ts_day} 09:00:00", f"2026-01-{ts_day} 09:30:00"]}
            ).write_ipc(p / "df_0.feather")

    # Candidate 1 (epoch_1):
    # - On label_0 (cross data for slot 0,0): reward = 10.0 (return 10.0/step)
    # - On label_1 (slope marginal): reward = 1.0 (return 1.0/step)
    # Candidate 2 (epoch_2):
    # - On label_0 (cross data for slot 0,0): reward = 2.0 (return 2.0/step)
    # - On label_1 (slope marginal): reward = 8.0 (return 8.0/step)

    # For epoch_1
    ep1_analysis = [
        {
            "标签": "label_0",
            "初始动作": 0,
            "分箱索引": 0,
            "合约": json.dumps(["fu0001"]),
            "数据文件": json.dumps(["fu0001/label_0/df_0.feather"]),
            "奖励总和": json.dumps([10.0]),
            "数据长度": json.dumps([2]),
            "换手率": json.dumps([0.1]),
        },
        {
            "标签": "label_1",
            "初始动作": 0,
            "分箱索引": 0,
            "合约": json.dumps(["fu0001"]),
            "数据文件": json.dumps(["fu0001/label_1/df_0.feather"]),
            "奖励总和": json.dumps([1.0]),
            "数据长度": json.dumps([2]),
            "换手率": json.dumps([0.1]),
        },
    ]
    ep1_detail = [
        {"标签": "label_0", "数据文件": "fu0001/label_0/df_0.feather", "初始动作": 0, "分箱索引": 0, "时间戳": "2026-01-01 09:00:00", "单步奖励": 10.0},
        {"标签": "label_1", "数据文件": "fu0001/label_1/df_0.feather", "初始动作": 0, "分箱索引": 0, "时间戳": "2026-01-02 09:00:00", "单步奖励": 1.0},
    ]

    # For epoch_2
    ep2_analysis = [
        {
            "标签": "label_0",
            "初始动作": 0,
            "分箱索引": 0,
            "合约": json.dumps(["fu0001"]),
            "数据文件": json.dumps(["fu0001/label_0/df_0.feather"]),
            "奖励总和": json.dumps([2.0]),
            "数据长度": json.dumps([2]),
            "换手率": json.dumps([0.1]),
        },
        {
            "标签": "label_1",
            "初始动作": 0,
            "分箱索引": 0,
            "合约": json.dumps(["fu0001"]),
            "数据文件": json.dumps(["fu0001/label_1/df_0.feather"]),
            "奖励总和": json.dumps([8.0]),
            "数据长度": json.dumps([2]),
            "换手率": json.dumps([0.1]),
        },
    ]
    ep2_detail = [
        {"标签": "label_0", "数据文件": "fu0001/label_0/df_0.feather", "初始动作": 0, "分箱索引": 0, "时间戳": "2026-01-01 09:00:00", "单步奖励": 2.0},
        {"标签": "label_1", "数据文件": "fu0001/label_1/df_0.feather", "初始动作": 0, "分箱索引": 0, "时间戳": "2026-01-02 09:00:00", "单步奖励": 8.0},
    ]

    for label_type in ("volatility", "slope"):
        p1 = candidate_root / "epoch_1" / label_type
        p1.mkdir(parents=True)
        pl.DataFrame(ep1_analysis).write_csv(p1 / "analysis_result.csv")
        pl.DataFrame(ep1_detail).write_csv(p1 / "trading_action_detail_epoch_1.csv")

        p2 = candidate_root / "epoch_2" / label_type
        p2.mkdir(parents=True)
        pl.DataFrame(ep2_analysis).write_csv(p2 / "analysis_result.csv")
        pl.DataFrame(ep2_detail).write_csv(p2 / "trading_action_detail_epoch_2.csv")

    selector = TwoDimensionalAgentSelector(
        SelectionConfig(
            num_labels=2,
            min_marginal_contracts=1,
            min_joint_contracts=1,
            min_positive_contract_ratio=0.5,
            min_mean_return=0.0,
            min_lcb=0.0,
            missing_joint_policy="slope_marginal_best",
        )
    )
    artifacts = selector.select(candidate_root, valid_root)
    slots = artifacts.selected_slots.to_dicts()

    # Slot (0, 0): HAS cross data -> selected by cross data profit -> epoch_1 (10.0 > 2.0)
    slot_00 = [s for s in slots if s["volatility_label"] == "label_0" and s["slope_label"] == "label_0"][0]
    assert slot_00["kind"] == "model"
    assert slot_00["candidate_id"] == "epoch_1:bin_0"
    assert pytest.approx(slot_00["pair_score"], abs=1e-5) == 10.0

    # Slot (0, 1): NO cross data -> selected by slope profit (slope label_1) -> epoch_2 (8.0 > 1.0)
    slot_01 = [s for s in slots if s["volatility_label"] == "label_0" and s["slope_label"] == "label_1"][0]
    assert slot_01["kind"] == "model"
    assert slot_01["candidate_id"] == "epoch_2:bin_0"
    assert slot_01["selection_reason"] == "fallback_from_slope_marginal"
    assert pytest.approx(slot_01["pair_score"], abs=1e-5) == 8.0

