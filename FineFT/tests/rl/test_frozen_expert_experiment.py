"""Formal preparation round-trips complete policy and independently sourced scores."""

import copy
import json
import types
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from RL.DiHFT.high_level import frozen_experiment as frozen
from RL.DiHFT.high_level import vae_routing_final_result_macro_action as final
from RL.DiHFT.high_level import vae_routing_optuna as tuning
from RL.DiHFT.high_level import vae_routing_util as vru
from RL.DiHFT.VAE.vae import MLP_VAE
from tests.rl.test_expert_hysteresis_routing import make_router
from tests.rl.test_vae_routing_action_persistence import _sample_manifest_payload


def descriptor(path):
    return {"path": str(path.resolve()), "sha256": frozen.file_sha256(path)}


@pytest.fixture
def experiment(tmp_path):
    dataset = tmp_path / "dataset" / "fu"
    selection_root = tmp_path / "selection_data"
    frames = {}
    for role, directory, date in (
        ("fit", dataset / "train", "2020-01-02"),
        ("selection", selection_root, "2020-01-03"),
        ("calibration", dataset / "valid", "2020-02-02"),
        ("final", dataset / "test", "2020-03-02"),
    ):
        directory.mkdir(parents=True)
        path = directory / "fu2005.feather"
        pd.DataFrame(
            {
                "timestamp": pd.date_range(date, periods=4, freq="10min", tz="UTC"),
                "trading_day": [date] * 4,
                "feature": [0.1, 0.2, 0.3, 0.4],
            }
        ).to_feather(path)
        frames[role] = path
    for field in (
        "rl_state_features",
        "vae_slope_state_features",
        "vae_volatility_state_features",
        "maintenance_margin_ratio_dict",
    ):
        np.save(dataset / f"{field}.npy", np.array(["feature"]))
    model = tmp_path / "specialists.pth"
    model.write_bytes(b"controlled specialist identity")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            _sample_manifest_payload(
                schema_version=1,
                axes={axis: [f"label_{i}" for i in range(3)] for axis in ("slope", "volatility")},
                slot_count=9,
                slot_index_formula="volatility_index * num_labels + slope_index",
                slots=[{"slot_id": i, "kind": "model"} for i in range(9)],
                valid_root=str(selection_root),
                artifacts={"model_assembly": str(model)},
            )
        )
    )
    for axis in ("slope", "volatility"):
        for label in range(3):
            root = tmp_path / "vae" / "fu" / "fixture" / axis / f"label_{label}"
            root.mkdir(parents=True)
            model = MLP_VAE(INPUT_DIM=1, Z_DIM=2, hidden_dims=[8, 4], loss_func="NLL")
            torch.save(model.state_dict(), root / "model_latest.pth")
            np.save(root / "id_logpx.npy", np.array([-2.0, -1.0, 0.0]))
    args = vru.parser.parse_args(
        [
            "--base_path",
            str(tmp_path / "dataset"),
            "--dataset_name",
            "fu",
            "--experiment_name",
            "fixture",
            "--selection_manifest",
            str(selection),
            "--vae_path",
            str(tmp_path / "vae"),
            "--gating_strategy",
            "expert_hysteresis",
            "--slope_window_length",
            "2",
            "--volatility_window_length",
            "3",
            "--slope_gamma",
            "0.9",
            "--volatility_gamma",
            "0.8",
            "--slope_rule_base_threshold",
            "0.6",
            "--volatility_rule_base_threshold",
            "0.8",
            "--hysteresis_exit_ratio",
            "0.5",
            "--expert_switch_lead",
            "0.15",
            "--expert_confirm_steps",
            "2",
            "--eval_stage",
            "valid",
            "--z_dim",
            "2",
        ]
    )
    args.vae_hidden_dims = [8, 4]
    evidence = tmp_path / "split_evidence.json"
    evidence.write_text(json.dumps({"split_version": "fixture.corrected.v1"}))
    periods = {}
    for role, start, end in (
        ("fit", "2020-01-01", "2020-01-10"),
        ("selection", "2020-01-01", "2020-01-11"),
        ("calibration", "2020-02-01", "2020-02-10"),
        ("final", "2020-03-01", "2020-03-10"),
    ):
        periods[role] = {
            "start": start + "T00:00:00Z",
            "end": end + "T00:00:00Z",
            "source_evidence": descriptor(evidence),
        }
    inputs = frozen.describe_experiment_inputs(args)
    search_space = {
        "slope_window_length": {"type": "int", "low": 1, "high": 10},
        "volatility_window_length": {"type": "int", "low": 1, "high": 10},
        "slope_gamma": {"type": "float", "low": 0.5, "high": 1.0, "log": True},
        "volatility_gamma": {"type": "float", "low": 0.5, "high": 1.0, "log": True},
        "slope_rule_base_threshold": {"type": "float", "low": 0.1, "high": 1.0},
        "volatility_rule_base_threshold": {"type": "float", "low": 0.1, "high": 1.0},
        "hysteresis_exit_ratio": {"type": "float", "low": 0.1, "high": 0.9},
        "expert_switch_lead": {"type": "float", "low": 0.0, "high": 0.5},
        "expert_confirm_steps": {"type": "int", "low": 1, "high": 5},
    }
    document = {
        "schema_version": frozen.SCHEMA_VERSION,
        "experiment_id": "controlled-fixture.v1",
        "strategy_version": "expert_hysteresis.v1",
        "split_version": "fixture.corrected.v1",
        "trading_day_atomic": True,
        "models_retrained_for_split": True,
        "periods": periods,
        "scoring_protocol": {
            "protocol_version": frozen.RAW_SCORING_PROTOCOL_VERSION,
            "seed": 12345,
            "batch_size": 8,
            "column_order": inputs["column_order"],
            "architecture": {
                key: getattr(args, key) for key in ("z_dim", "vae_hidden_dims", "loss_type")
            },
        },
        "search_space": search_space,
        "artifacts": inputs["artifacts"],
    }
    manifest = tmp_path / "experiment.json"
    manifest.write_text(json.dumps(document))
    args.experiment_manifest = str(manifest)
    args.study_name = "expert_hysteresis_fixture"
    args.study_storage_identity = "sha256:fixture-study"
    args.trial_number = 7
    args.trial_state = "COMPLETE"
    args.objective_name = "net_return_with_mdd_guard"
    args.objective_version = "v1"
    args.objective_direction = "maximize"
    args.objective_value = 0.123
    return args, document, frames


def save_manifest(args, document):
    Path(args.experiment_manifest).write_text(json.dumps(document))


def test_public_tuning_preparation_and_final_snapshot_roundtrip(experiment, tmp_path, monkeypatch):
    args, _, _ = experiment
    search = tuning.parser_all.parse_args(
        [
            "--dataset_name",
            "fu",
            "--experiment_name",
            "fixture",
            "--selection_manifest",
            args.selection_manifest,
            "--gating_strategy",
            "expert_hysteresis",
            "--hysteresis_exit_ratio",
            "0.5",
        ]
    )
    prepared = tuning.prepare_base_args(args, search)
    selected = frozen.export_frozen_experiment(prepared, tmp_path / "selected.json")
    assert prepared.slope_hold_threshold == pytest.approx(0.3)
    assert prepared.volatility_hold_threshold == pytest.approx(0.4)
    assert prepared.strategy_version == "expert_hysteresis.v1"
    # Auto-discovered legacy files must never override a frozen selected policy.
    monkeypatch.chdir(tmp_path)
    legacy = Path("result/DiHFT/final_result/fu/fixture/high_level_agent_para.txt")
    legacy.parent.mkdir(parents=True)
    legacy.write_text("strat_hierarchical_ws_999_wv_999\n")
    restored = final.parse_and_prepare_args(["--frozen_config", selected])
    for key in frozen.POLICY_FIELDS + frozen.SCORING_FIELDS + frozen.RUNTIME_FIELDS:
        assert getattr(restored, key) == getattr(prepared, key), key
    assert restored.eval_stage == "test"
    assert restored.frozen_experiment_identity == prepared.frozen_experiment_identity
    assert restored.frozen_config_hash == prepared.frozen_config_hash
    assert restored.frozen_configuration_identity == prepared.frozen_config_hash
    assert restored.frozen_selection["trial_number"] == 7
    assert restored.output_sub_dir == str(
        Path("expert_hysteresis")
        / prepared.frozen_config_hash
        / "test"
        / "final_evaluation"
    )
    assert restored.para_file is None
    base_reloaded = vru.resolve_routing_parameters(
        vru.parser.parse_args(["--frozen_config", selected, "--eval_stage", "test"])
    )
    assert base_reloaded.gating_strategy == "expert_hysteresis"
    assert base_reloaded.expert_confirm_steps == prepared.expert_confirm_steps


def test_best_trial_applies_all_nine_dimensions_before_freeze(experiment):
    args, _, _ = experiment
    frozen.prepare_experiment(args)
    best_params = {
        "slope_window_length": 4,
        "volatility_window_length": 5,
        "slope_gamma": 0.75,
        "volatility_gamma": 0.7,
        "slope_rule_base_threshold": 0.55,
        "volatility_rule_base_threshold": 0.65,
        "hysteresis_exit_ratio": 0.6,
        "expert_switch_lead": 0.2,
        "expert_confirm_steps": 3,
    }
    applied = tuning.apply_best_trial_parameters(
        copy.deepcopy(args),
        best_params,
        Namespace(gating_strategy="expert_hysteresis"),
    )
    assert frozen.effective_trial_configuration(applied) == {
        **best_params,
        "slope_hold_threshold": pytest.approx(0.33),
        "volatility_hold_threshold": pytest.approx(0.39),
    }


def test_raw_quantile_cache_identity_allows_per_trial_smoothing(experiment, tmp_path):
    args, document, _ = experiment
    frozen.prepare_experiment(args)
    directory = tmp_path / "cache"
    directory.mkdir()
    np.save(
        directory / "fu2005.npy",
        np.full((4, 6), 0.5, dtype=np.float32),
    )
    frozen.write_quantile_cache_manifest(args, directory)
    changed = copy.deepcopy(args)
    changed.slope_window_length = 8
    changed.volatility_window_length = 9
    changed.slope_gamma = 0.6
    changed.volatility_gamma = 0.65
    frozen.validate_trial_configuration(changed)
    assert frozen.validate_quantile_cache(changed, directory) == str(directory)
    document["search_space"]["slope_window_length"]["high"] = 12
    save_manifest(changed, document)
    frozen.prepare_experiment(changed)
    assert changed.frozen_experiment_identity != args.frozen_experiment_identity
    assert changed.raw_score_source_identity == args.raw_score_source_identity
    assert frozen.validate_quantile_cache(changed, directory) == str(directory)


def test_old_schema_and_incomplete_selected_trial_are_rejected(experiment, tmp_path):
    args, document, _ = experiment
    document["schema_version"] = 1
    save_manifest(args, document)
    with pytest.raises(ValueError, match="schema_version=2"):
        frozen.prepare_experiment(args)

    document["schema_version"] = frozen.SCHEMA_VERSION
    save_manifest(args, document)
    args.trial_state = "PRUNED"
    with pytest.raises(ValueError, match="COMPLETE"):
        frozen.export_frozen_experiment(args, tmp_path / "selected.json")


@pytest.mark.parametrize(
    "flags",
    [
        ["--expert_confirm_steps", "9"],
        ["--slope_gamma", "0.6"],
        ["--initial_wallet_balance", "1"],
        ["--transcation_cost", "0.1"],
        ["--strategy_version", "future.v2"],
        ["--expert_confirm", "9"],
        ["--slope_gamma=0.6"],
    ],
)
def test_final_rejects_frozen_override(experiment, tmp_path, flags):
    args, _, _ = experiment
    selected = frozen.export_frozen_experiment(args, tmp_path / "selected.json")
    with pytest.raises(ValueError, match="override frozen"):
        final.parse_and_prepare_args(["--frozen_config", selected, *flags])
    with pytest.raises(ValueError, match="override frozen"):
        vru.resolve_routing_parameters(
            vru.parser.parse_args(
                [
                    "--frozen_config",
                    selected,
                    "--eval_stage",
                    "test",
                    *flags,
                ]
            )
        )


def test_final_requires_complete_frozen_config_and_rejects_legacy_mix(experiment, tmp_path):
    args, _, _ = experiment
    with pytest.raises(ValueError, match="requires frozen_config"):
        final.parse_and_prepare_args(["--gating_strategy", "expert_hysteresis"])
    selected = frozen.export_frozen_experiment(args, tmp_path / "selected.json")
    with pytest.raises(ValueError, match="legacy"):
        final.parse_and_prepare_args(["--frozen_config", selected, "--optuna_csv", "old.csv"])
    document = json.loads(Path(selected).read_text())
    document["configuration"].pop("expert_confirm_steps")
    Path(selected).write_text(json.dumps(document))
    with pytest.raises(ValueError, match="hash"):
        final.parse_and_prepare_args(["--frozen_config", selected])


@pytest.mark.parametrize(
    "change,match",
    [
        ("overlap_selection", "overlaps independent"),
        ("overlap_final", "overlaps final"),
        ("atomic", "TradingDay-atomic"),
        ("retrained", "fitted for"),
        ("version", "strategy_version"),
        ("score", "outside search_space"),
        ("model", "hash mismatch"),
        ("missing", "identity set"),
        ("day", "TradingDay crosses"),
        ("outside", "outside declared"),
        ("absent_day", "evidence missing"),
    ],
)
def test_formal_prepare_rejects_missing_or_contaminated_evidence(experiment, change, match):
    args, document, frames = experiment
    if change == "overlap_selection":
        document["periods"]["selection"]["end"] = "2020-02-03T00:00:00Z"
    elif change == "overlap_final":
        document["periods"]["final"]["start"] = "2020-02-05T00:00:00Z"
    elif change == "atomic":
        document["trading_day_atomic"] = False
    elif change == "retrained":
        document["models_retrained_for_split"] = False
    elif change == "version":
        document["strategy_version"] = "expert_hysteresis.v2"
    elif change == "score":
        args.slope_gamma = 0.4
    elif change == "model":
        Path(document["artifacts"]["slope.label_0.model"]["path"]).write_bytes(b"changed")
    elif change == "missing":
        document["artifacts"].pop("slope.label_0.reference")
    else:
        path = frames["calibration"]
        frame = pd.read_feather(path)
        if change == "day":
            frame["trading_day"] = "2020-01-02"
        elif change == "outside":
            frame["timestamp"] = pd.Timestamp("2020-01-02", tz="UTC")
        else:
            frame = frame.drop(columns=["trading_day"])
        frame.to_feather(path)
        document["artifacts"]["data.calibration.fu2005"] = descriptor(path)
    save_manifest(args, document)
    with pytest.raises(ValueError, match=match):
        frozen.prepare_experiment(args)


def test_missing_manifest_and_legacy_parameters_do_not_claim_independence(experiment):
    args, _, _ = experiment
    args.experiment_manifest = None
    with pytest.raises(ValueError, match="experiment_manifest"):
        frozen.prepare_experiment(args)
    args.para_file = "legacy.txt"
    with pytest.raises(ValueError, match="legacy"):
        vru.resolve_routing_parameters(args)


def test_selected_snapshot_detects_source_mutation_and_cannot_be_replaced(experiment, tmp_path):
    args, document, _ = experiment
    selected = frozen.export_frozen_experiment(args, tmp_path / "selected.json")
    args.expert_confirm_steps = 3
    with pytest.raises(ValueError, match="already frozen"):
        frozen.export_frozen_experiment(args, selected)
    document["experiment_id"] = "another-experiment"
    save_manifest(args, document)
    with pytest.raises(ValueError, match="source identity changed"):
        final.parse_and_prepare_args(["--frozen_config", selected])


def test_invalid_policy_fails_at_public_resolver(experiment):
    args, _, _ = experiment
    for field, value in (
        ("expert_confirm_steps", 0),
        ("hysteresis_exit_ratio", 1.0),
        ("expert_switch_lead", float("nan")),
        ("slope_rule_base_threshold", 0),
    ):
        invalid = copy.deepcopy(args)
        setattr(invalid, field, value)
        with pytest.raises(ValueError):
            vru.resolve_routing_parameters(invalid)


def test_quantile_cache_requires_matching_provenance_and_has_no_route_state(experiment, tmp_path):
    args, _, _ = experiment
    frozen.prepare_experiment(args)
    directory = tmp_path / "cache"
    directory.mkdir()
    values = np.array([[0.1, 0.2, 0.6, 0.1, 0.8, 0.2]] * 4, dtype=np.float32)
    np.save(directory / "fu2005.npy", values)
    with pytest.raises(ValueError, match="provenance"):
        frozen.validate_quantile_cache(args, directory)
    frozen.write_quantile_cache_manifest(args, directory)
    assert frozen.validate_quantile_cache(args, directory) == str(directory)
    metadata = json.loads((directory / "quantile_cache_manifest.json").read_text())
    assert set(metadata) == {
        "schema_version",
        "document_type",
        "raw_score_source_identity",
        "raw_scoring_protocol",
        "eval_stage",
        "contracts",
    }
    scores = np.load(directory / "fu2005.npy", mmap_mode="r")
    assert not scores.flags.writeable
    changed = values.copy()
    changed[0, 0] = 0.3
    np.save(directory / "fu2005.npy", changed)
    with pytest.raises(ValueError, match="provenance"):
        frozen.validate_quantile_cache(args, directory)


def test_frozen_identical_scores_via_cache_or_protocol_have_identical_public_routes(
    tmp_path, monkeypatch
):
    values = np.array([[0.1, 0.2, 0.6, 0.1, 0.8, 0.2]] * 4, dtype=np.float32)
    path = tmp_path / "scores.npy"
    np.save(path, values)
    cached = np.load(path, mmap_mode="r")
    first, second = make_router(), make_router()
    for router in (first, second):
        router.gamma = 0.9
        router.window_length = 2
        router.axis_gammas = {"slope": 0.9, "volatility": 0.9}
        router.reset_routing_state()
    first.current_contract_quantiles = cached
    second.current_contract_quantiles = None
    second.vae_models = {axis: list(range(3)) for axis in ("slope", "volatility")}
    second.in_ds_logpx = {
        axis: [np.array([0.0]) for _ in range(3)] for axis in ("slope", "volatility")
    }
    monkeypatch.setattr(vru, "analyze_single_sample", lambda *args: (None, 0.0))
    second.device = "cpu"
    sequence = iter(values.flatten())
    second.find_quantile = lambda value, reference: next(sequence)
    # get_quantiles traverses slope labels then volatility labels, exactly the cache order.
    for step in range(2):
        first.step_idx = second.step_idx = step
        first.get_quantiles(np.array([0.0]), np.array([0.0]))
        second.get_quantiles(np.array([0.0]), np.array([0.0]))
    outputs = []
    for router in (first, second):
        outputs.append(
            [
                router.get_action(
                    {"previous_action": 1, "avaliable_action": [1, 1, 1], "trading_info": [0.0]},
                    np.array([step]),
                    0.0,
                    1,
                )
                for step in range(2)
            ]
        )
    assert outputs[0] == outputs[1] == [1, 2]
    assert first.audit_predictions == second.audit_predictions
    # Interleaving a third run starts a separate K-step streak on the same read-only cache.
    third = make_router()
    third.gamma = 0.9
    third.window_length = 2
    third.axis_gammas = {"slope": 0.9, "volatility": 0.9}
    third.reset_routing_state()
    third.current_contract_quantiles = cached
    third.step_idx = 0
    for step in range(2):
        third.step_idx = step
        third.get_quantiles(np.array([0.0]), np.array([0.0]))
    assert (
        third.get_action(
            {"previous_action": 1, "avaliable_action": [1, 1, 1], "trading_info": [0.0]},
            np.array([0.0]),
            0.0,
            1,
        )
        == 1
    )
    assert third.audit_predictions == []


def test_public_final_replay_preserves_snapshot_and_writes_receipt(
    experiment, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    args, _, _ = experiment
    selected = frozen.export_frozen_experiment(args, tmp_path / "selected.json")
    output = tmp_path / "final-output"
    cache = tmp_path / "final-cache"
    observed = []

    def controlled_scores(final_args):
        cache.mkdir()
        np.save(cache / "fu2005.npy", np.full((4, 6), 0.7, dtype=np.float32))
        frozen.write_quantile_cache_manifest(final_args, cache)
        return str(cache)

    class ControlledReplay:
        def __init__(self, final_args):
            self.args = final_args
            self.test_path = str(output)

        def test(self):
            observed.append({key: getattr(self.args, key) for key in frozen.POLICY_FIELDS})
            return 0.123

    monkeypatch.setattr(final, "ensure_precomputed_vae_quantiles", controlled_scores)
    monkeypatch.setattr(final, "vae_risk_aware_routing", ControlledReplay)
    for module, name in (
        ("analysis.diagnostics.trading_diagnostics", "TradingDiagnosticsCalculator"),
        (
            "analysis.diagnostics.synchronized_portfolio_diagnostics",
            "SynchronizedPortfolioCalculator",
        ),
    ):
        monkeypatch.setattr(f"{module}.{name}", lambda **kwargs: SimpleNamespace(run=lambda: None))
    monkeypatch.setattr("sys.argv", ["replay", "--frozen_config", selected])
    assert final.main() == 0.123
    assert observed == [{key: getattr(args, key) for key in frozen.POLICY_FIELDS}]
    receipt = json.loads((output / "frozen_evaluation_receipt.json").read_text())
    assert receipt["config_hash"] == args.frozen_config_hash
    assert receipt["experiment_identity"] == args.frozen_experiment_identity
    assert receipt["eval_stage"] == "test"
    assert (output / "frozen_config.json").read_bytes() == Path(selected).read_bytes()
    restored = final.parse_and_prepare_args(["--frozen_config", selected])
    restored.precomputed_quantiles_dir = str(cache)
    with pytest.raises(FileExistsError, match="overwrite frozen evaluation"):
        frozen.write_evaluation_receipt(restored, output)
    restored.expert_confirm_steps += 1
    with pytest.raises(ValueError, match="effective configuration changed"):
        frozen.write_evaluation_receipt(restored, output)


def test_frozen_scoring_protocol_generates_reproducible_compatible_cache(
    experiment, tmp_path, monkeypatch
):
    args, _, _ = experiment
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args.precomputed_quantiles_dir = str(tmp_path / "first-scores")
    first = vru.ensure_precomputed_vae_quantiles(args)
    other = copy.deepcopy(args)
    other.precomputed_quantiles_dir = str(tmp_path / "second-scores")
    second = vru.ensure_precomputed_vae_quantiles(other)
    np.testing.assert_array_equal(
        np.load(Path(first) / "fu2005.npy"), np.load(Path(second) / "fu2005.npy")
    )
    assert json.loads((Path(first) / "quantile_cache_manifest.json").read_text()) == json.loads(
        (Path(second) / "quantile_cache_manifest.json").read_text()
    )
    assert vru.ensure_precomputed_vae_quantiles(args) == first
    incompatible = json.loads((Path(first) / "quantile_cache_manifest.json").read_text())
    incompatible["raw_scoring_protocol"]["seed"] += 1
    (Path(first) / "quantile_cache_manifest.json").write_text(json.dumps(incompatible))
    with pytest.raises(ValueError, match="provenance"):
        vru.ensure_precomputed_vae_quantiles(args)


def test_negative_boolean_override_cannot_change_frozen_risk_controls(experiment, tmp_path):
    args, _, _ = experiment
    args.enable_mdd_penalty = True
    selected = frozen.export_frozen_experiment(args, tmp_path / "selected.json")
    with pytest.raises(ValueError, match="override frozen"):
        final.parse_and_prepare_args(["--frozen_config", selected, "--no-enable_mdd_penalty"])


def test_public_joint_search_freezes_and_replays_one_complete_trial(
    experiment, tmp_path, monkeypatch
):
    args, _, _ = experiment
    search = tuning.parser_all.parse_args(
        [
            "--dataset_name",
            "fu",
            "--experiment_name",
            "fixture",
            "--selection_manifest",
            args.selection_manifest,
            "--calibration_run_id",
            "controlled-run",
            "--n_trials",
            "1",
            "--n_workers",
            "1",
        ]
    )
    observed = {"trial": None, "replays": []}

    class ControlledTrial:
        def __init__(self):
            self.params = {}
            self.user_attrs = {}

        def suggest_int(self, name, low, high):
            value = low if name == "slope_window_length" else high
            self.params[name] = value
            return value

        def suggest_float(self, name, low, high, log=False):
            del log
            value = low if name in {"slope_gamma", "expert_switch_lead"} else high
            self.params[name] = value
            return value

        def set_user_attr(self, name, value):
            self.user_attrs[name] = value

    best_trial = types.SimpleNamespace(
        number=0,
        params={},
        user_attrs={},
        value=0.123,
        state=types.SimpleNamespace(name="COMPLETE"),
    )

    class ControlledStudy:
        def __init__(self, selected):
            self._selected = selected
            self.trials = [selected]

        @property
        def best_trial(self):
            return self._selected

        def trials_dataframe(self):
            return pd.DataFrame(
                [{**self._selected.params, "value": self._selected.value}]
            )

    study = ControlledStudy(best_trial)

    class ControlledOptuna:
        @staticmethod
        def create_study(**kwargs):
            assert kwargs["direction"] == "maximize"
            assert kwargs["load_if_exists"] is False
            return study

        @staticmethod
        def load_study(**kwargs):
            del kwargs
            return study

    def controlled_worker(base_args, search_args, study_name, storage_url, n_trials):
        del study_name, storage_url
        assert n_trials == 1
        trial_args = copy.deepcopy(base_args)
        trial_args.save_artifacts = False
        trial = ControlledTrial()
        tuning.suggest_trial_parameters(trial, trial_args, search_args)
        raw_metrics = {
            "objective_value": 0.123,
            "raw_return_rate": 0.15,
            "portfolio_mdd": 0.02,
            "total_trades": 8,
        }
        trial.set_user_attr("raw_metrics", raw_metrics)
        best_trial.params = dict(trial.params)
        best_trial.user_attrs = dict(trial.user_attrs)
        observed["trial"] = trial_args, trial

    class ControlledReplay:
        def __init__(self, replay_args):
            self.args = replay_args
            self.test_path = str(tmp_path / "best-replay")
            self.last_objective_metrics = {
                "objective_value": 0.123,
                "raw_return_rate": 0.15,
                "portfolio_mdd": 0.02,
                "total_trades": 8,
            }

        def test(self):
            observed["replays"].append(copy.deepcopy(self.args))
            return 0.123

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tuning, "optuna", ControlledOptuna)
    monkeypatch.setattr(tuning, "make_rdb_storage", lambda _: None)
    monkeypatch.setattr(
        tuning, "ensure_precomputed_vae_quantiles", lambda _: str(tmp_path / "raw-cache")
    )
    monkeypatch.setattr(tuning, "run_optuna_worker", controlled_worker)
    monkeypatch.setattr(tuning, "vae_risk_aware_routing", ControlledReplay)

    outcome = tuning.tune(args, search)

    trial_args, trial = observed["trial"]
    assert trial_args.save_artifacts is False
    assert set(trial.params) == set(frozen.SEARCH_FIELDS)
    assert trial_args.slope_window_length == 1
    assert trial_args.volatility_window_length == 10
    assert trial_args.slope_gamma == pytest.approx(0.5)
    assert trial_args.volatility_gamma == pytest.approx(1.0)
    assert trial_args.calibration_warmup_steps == 10
    assert trial.user_attrs["effective_configuration"]["slope_hold_threshold"] == pytest.approx(
        0.9
    )
    assert len(observed["replays"]) == 1
    replay = observed["replays"][0]
    assert replay.save_artifacts is True
    assert frozen.effective_trial_configuration(replay) == trial.user_attrs[
        "effective_configuration"
    ]

    frozen_document = json.loads(Path(outcome["frozen_config"]).read_text())
    assert frozen_document["selection"] == {
        "study_name": "expert_hysteresis_calibration.controlled-fixture.v1.controlled-run",
        "study_storage_identity": replay.study_storage_identity,
        "trial_number": 0,
        "trial_state": "COMPLETE",
        "objective_name": tuning.OBJECTIVE_NAME,
        "objective_version": tuning.OBJECTIVE_VERSION,
        "objective_direction": "maximize",
        "objective_value": pytest.approx(0.123),
    }
    evidence = json.loads(Path(outcome["selection_evidence"]).read_text())
    assert evidence["selected_trial"]["parameters"] == trial.params
    assert evidence["selected_trial"]["raw_metrics"]["raw_return_rate"] == pytest.approx(0.15)
    assert evidence["best_replay"]["matches_lean_trial"] is True
    assert evidence["best_replay"]["objective_value"] == pytest.approx(0.123)
    assert "expert_hysteresis" in outcome["frozen_config"]
    assert "vae_risk_aware_routing_optuna" not in outcome["frozen_config"]


def test_step7_rejects_cli_search_overrides_and_existing_study(experiment, tmp_path):
    args, _, _ = experiment
    search = tuning.parser_all.parse_args(
        [
            "--dataset_name",
            "fu",
            "--experiment_name",
            "fixture",
            "--selection_manifest",
            args.selection_manifest,
            "--calibration_run_id",
            "same-run",
            "--window_length_min",
            "2",
        ]
    )
    with pytest.raises(ValueError, match="experiment_manifest"):
        tuning.prepare_base_args(args, search)

    clean = tuning.parser_all.parse_args(
        [
            "--dataset_name",
            "fu",
            "--experiment_name",
            "fixture",
            "--selection_manifest",
            args.selection_manifest,
            "--calibration_run_id",
            "same-run",
        ]
    )
    prepared = tuning.prepare_base_args(args, clean)
    prepared.result_path = str(tmp_path / "results")
    storage_url = tuning.create_study_storage(prepared)
    Path(tuning._storage_path(storage_url)).touch()
    with pytest.raises(FileExistsError, match="refusing to replace"):
        tuning.create_study_storage(prepared)


def test_joint_search_uses_common_warmup_but_trial_specific_smoothing(experiment, tmp_path):
    args, _, _ = experiment
    frozen.prepare_experiment(args)
    first = copy.deepcopy(args)
    second = copy.deepcopy(args)
    first.slope_window_length = 1
    first.slope_gamma = 0.5
    second.slope_window_length = 4
    second.slope_gamma = 1.0
    frozen.validate_trial_configuration(first)
    frozen.validate_trial_configuration(second)
    for candidate in (first, second):
        candidate.save_artifacts = False
        candidate.output_sub_dir = None
        candidate.result_path = str(tmp_path)
    first_router = make_router()
    second_router = make_router()
    for router in (first_router, second_router):
        router.eval_stage = "valid"
        router.dataset_name = "fu"
        router.experiment_name = "fixture"
        router.save_artifacts = False
        router._set_expert_run_identity = lambda _: None
    first_router.reconfigure_routing(first)
    second_router.reconfigure_routing(second)
    first_router.quantiles["slope"][0].extend([0.0, 0.0, 0.0, 1.0])
    second_router.quantiles["slope"][0].extend([0.0, 0.0, 0.0, 1.0])
    assert first_router.initial_rollout_window_length == 10
    assert second_router.initial_rollout_window_length == 10
    assert first_router.calculate_axis_window_result("slope")[0] == pytest.approx(1.0)
    assert second_router.calculate_axis_window_result("slope")[0] == pytest.approx(0.25)


def test_step7_shell_wires_frozen_experiment_and_expert_namespace():
    script = Path("FineFT/script/test/DiHFT/high_level/vae_optuna_fu_10.sh").read_text()
    assert '--experiment_manifest "${EXPERIMENT_MANIFEST}"' in script
    assert '--calibration_run_id "${CALIBRATION_RUN_ID}"' in script
    assert "--gating_strategy expert_hysteresis" in script
    assert "rule_base_threshold_min" not in script
    assert "high_level/expert_hysteresis" in script


@pytest.mark.parametrize(
    ("script_name", "purpose", "stage"),
    [
        ("final_result_fu_10_main_sub.sh", "main_sub", '"$EVAL_STAGE"'),
        ("final_result_fu_10.sh", "final", "test"),
    ],
)
def test_step9_and_step10_shells_only_replay_the_published_frozen_config(
    script_name, purpose, stage
):
    script = Path("FineFT/script/test/DiHFT/high_level") / script_name
    content = script.read_text()
    assert 'FROZEN_CONFIG=${FROZEN_CONFIG:?' in content
    assert f"--evaluation_purpose {purpose}" in content
    assert f'--eval_stage {stage}' in content
    assert "--para_file" not in content
    assert "--optuna_csv" not in content
    assert "--initial_wallet_balance" not in content
    assert "--transcation_cost" not in content
    assert "--stop_loss_return_threshold" not in content


def test_main_sub_frozen_replay_uses_valid_namespace_without_mutating_policy(experiment, tmp_path):
    args, _, _ = experiment
    selected = frozen.export_frozen_experiment(args, tmp_path / "selected.json")

    restored = final.parse_and_prepare_args(
        [
            "--frozen_config",
            selected,
            "--eval_stage",
            "valid",
            "--evaluation_purpose",
            "main_sub",
        ]
    )

    assert restored.eval_stage == "valid"
    assert restored.evaluation_purpose == "main_sub"
    assert restored.initial_wallet_balance == args.initial_wallet_balance
    router = vru.vae_risk_aware_routing.__new__(vru.vae_risk_aware_routing)
    router.dataset_name = restored.dataset_name
    router.experiment_name = restored.experiment_name
    router.eval_stage = restored.eval_stage
    path = router._resolve_test_path(restored)
    assert path.endswith(f"/{args.frozen_config_hash}/valid/main_sub")


def test_final_purpose_rejects_valid_stage(experiment, tmp_path):
    args, _, _ = experiment
    selected = frozen.export_frozen_experiment(args, tmp_path / "selected.json")
    with pytest.raises(ValueError, match="requires eval_stage=test"):
        final.parse_and_prepare_args(
            ["--frozen_config", selected, "--eval_stage", "valid", "--evaluation_purpose", "final"]
        )


@pytest.mark.parametrize(
    "field,low,high,match",
    [
        ("slope_window_length", 0, 10, "positive windows"),
        ("slope_gamma", 0.0, 1.0, "bounds must lie"),
        ("hysteresis_exit_ratio", 0.5, 1.0, "0 < low <= high < 1"),
        ("expert_switch_lead", -0.1, 0.2, "non-negative"),
        ("expert_confirm_steps", 0, 3, "positive integers"),
    ],
)
def test_manifest_search_space_rejects_invalid_expert_bounds(
    experiment, field, low, high, match
):
    args, document, _ = experiment
    document["search_space"][field]["low"] = low
    document["search_space"][field]["high"] = high
    save_manifest(args, document)
    with pytest.raises(ValueError, match=match):
        frozen.prepare_experiment(args, validate_trial=False)
