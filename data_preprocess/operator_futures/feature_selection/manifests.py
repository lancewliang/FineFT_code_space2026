from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import polars as pl


class PersistenceFilterConfig(TypedDict):
    min_half_life_bars: float
    active_feature_pattern: str


class PersistenceDiagnostic(TypedDict):
    feature: str
    lag1_autocorrelation_median: float | None
    half_life_bars_median: float | None
    active_filter: bool


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


@dataclass
class FeatureSelectionContractRecord:
    contract: str
    input_path: str
    metric_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "input_path": self.input_path,
            "metric_path": self.metric_path,
        }


@dataclass
class FilteredOutputRecord:
    contract: str
    output_path: str
    output_row_count: int
    output_column_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "output_path": self.output_path,
            "output_row_count": self.output_row_count,
            "output_column_count": self.output_column_count,
        }



@dataclass
class StreamAuditRecord:
    profile_name: str
    selected_features: list[str]
    selected_feature_count: int
    filter_results: dict[str, list[str]] = field(default_factory=dict)
    candidate_count: int | None = None
    dropped_counts: dict[str, int] | None = None
    tier_breakdown: dict[str, dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "profile_name": self.profile_name,
            "selected_features": list(self.selected_features),
            "selected_feature_count": self.selected_feature_count,
            "filter_results": {
                key: list(values) for key, values in self.filter_results.items()
            },
        }
        if self.candidate_count is not None:
            payload["candidate_count"] = self.candidate_count
        if self.dropped_counts is not None:
            payload["dropped_counts"] = {
                key: int(val) for key, val in self.dropped_counts.items()
            }
        if self.tier_breakdown is not None:
            payload["tier_breakdown"] = self.tier_breakdown
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StreamAuditRecord:
        filter_results = {}
        if "filter_results" in data and data["filter_results"] is not None:
            filter_results = {
                key: list(values) for key, values in data["filter_results"].items()
            }
        return cls(
            profile_name=data["profile_name"],
            selected_features=list(data["selected_features"]),
            selected_feature_count=int(data["selected_feature_count"]),
            filter_results=filter_results,
            candidate_count=data.get("candidate_count"),
            dropped_counts=data.get("dropped_counts"),
            tier_breakdown=data.get("tier_breakdown"),
        )

@dataclass
class ContractOutputShape:
    rows: int
    columns: int

    def to_dict(self) -> dict[str, int]:
        return {
            "rows": self.rows,
            "columns": self.columns,
        }


@dataclass
class FeatureSelectionManifest:
    symbol: str
    target_freq: str
    stage: str
    split_input_dir: str
    windows_list: list[int]
    aggregate_metrics_path: str
    contracts: list[FeatureSelectionContractRecord] = field(default_factory=list)
    rl_feature_file: str | None = None
    vae_slope_feature_file: str | None = None
    vae_volatility_feature_file: str | None = None
    union_selected_feature_count: int | None = None
    union_selected_features: list[str] | None = None
    composite_drop_ratio: float | None = None
    global_feature_blacklist: list[str] | None = None
    feature_ablation_patterns: list[str] | None = None
    rank_ic_mode: str | None = None
    mandatory_state_features: list[str] | None = None
    shared_filter_results: dict[str, list[str]] | None = None
    persistence_filter: PersistenceFilterConfig | None = None
    persistence_diagnostics: list[PersistenceDiagnostic] | None = None
    filtered_outputs: list[FilteredOutputRecord] | None = None
    evaluated_feature_file: str | None = None
    evaluated_feature_count: int | None = None
    evaluated_features: list[str] | None = None
    report_only: bool | None = None
    regime_bins: int | None = None
    target_regime_bins: list[list[int]] | list[tuple[int, int]] | None = None
    regime_quantiles: dict[str, list[float]] | None = None
    regime_audit_path: str | None = None
    distribution_audit_path: str | None = None
    global_max_mean_psi: float | None = None
    global_max_pair_psi: float | None = None
    min_drift_survivors: int | None = None
    global_min_sign_consistency: float | None = None
    conditional_anchors_retained: list[dict[str, Any]] | None = None
    stream_mode: str | None = None
    vae_slope_stream: StreamAuditRecord | None = None
    vae_volatility_stream: StreamAuditRecord | None = None
    rl_stream: StreamAuditRecord | None = None
    process_documentation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "symbol": self.symbol,
            "target_freq": self.target_freq,
            "stage": self.stage,
            "split_input_dir": self.split_input_dir,
        }
        if self.rl_feature_file is not None:
            payload["rl_feature_file"] = self.rl_feature_file
        if self.vae_slope_feature_file is not None:
            payload["vae_slope_feature_file"] = self.vae_slope_feature_file
        if self.vae_volatility_feature_file is not None:
            payload["vae_volatility_feature_file"] = self.vae_volatility_feature_file
        if self.union_selected_feature_count is not None:
            payload["union_selected_feature_count"] = self.union_selected_feature_count
        if self.union_selected_features is not None:
            payload["union_selected_features"] = list(self.union_selected_features)
        if self.evaluated_feature_file is not None:
            payload["evaluated_feature_file"] = self.evaluated_feature_file
        if self.evaluated_feature_count is not None:
            payload["evaluated_feature_count"] = self.evaluated_feature_count
        if self.evaluated_features is not None:
            payload["evaluated_features"] = list(self.evaluated_features)
        payload["windows_list"] = list(self.windows_list)
        if self.composite_drop_ratio is not None:
            payload["composite_drop_ratio"] = self.composite_drop_ratio
        if self.global_feature_blacklist is not None:
            payload["global_feature_blacklist"] = list(self.global_feature_blacklist)
        if self.feature_ablation_patterns is not None:
            payload["feature_ablation_patterns"] = list(self.feature_ablation_patterns)
        if self.rank_ic_mode is not None:
            payload["rank_ic_mode"] = self.rank_ic_mode
        if self.mandatory_state_features is not None:
            payload["mandatory_state_features"] = list(self.mandatory_state_features)
        payload["aggregate_metrics_path"] = self.aggregate_metrics_path
        if self.shared_filter_results is not None:
            payload["shared_filter_results"] = {
                key: list(values) for key, values in self.shared_filter_results.items()
            }
        if self.persistence_filter is not None:
            payload["persistence_filter"] = dict(self.persistence_filter)
        if self.persistence_diagnostics is not None:
            payload["persistence_diagnostics"] = [
                dict(row) for row in self.persistence_diagnostics
            ]
        payload["contracts"] = [contract.to_dict() for contract in self.contracts]
        if self.filtered_outputs is not None:
            payload["filtered_outputs"] = [
                output.to_dict() for output in self.filtered_outputs
            ]
        if self.report_only is not None:
            payload["report_only"] = self.report_only
        if self.regime_bins is not None:
            payload["regime_bins"] = self.regime_bins
        if self.target_regime_bins is not None:
            payload["target_regime_bins"] = [list(b) for b in self.target_regime_bins]
        if self.regime_quantiles is not None:
            payload["regime_quantiles"] = self.regime_quantiles
        if self.regime_audit_path is not None:
            payload["regime_audit_path"] = self.regime_audit_path
        if self.distribution_audit_path is not None:
            payload["distribution_audit_path"] = self.distribution_audit_path
        if self.global_max_mean_psi is not None:
            payload["global_max_mean_psi"] = self.global_max_mean_psi
        if self.global_max_pair_psi is not None:
            payload["global_max_pair_psi"] = self.global_max_pair_psi
        if self.min_drift_survivors is not None:
            payload["min_drift_survivors"] = self.min_drift_survivors
        if self.global_min_sign_consistency is not None:
            payload["global_min_sign_consistency"] = self.global_min_sign_consistency
        if self.conditional_anchors_retained is not None:
            payload["conditional_anchors_retained"] = self.conditional_anchors_retained
        if self.stream_mode is not None:
            payload["stream_mode"] = self.stream_mode
        if self.vae_slope_stream is not None:
            payload["vae_slope_stream"] = self.vae_slope_stream.to_dict()
        if self.vae_volatility_stream is not None:
            payload["vae_volatility_stream"] = self.vae_volatility_stream.to_dict()
        if self.rl_stream is not None:
            payload["rl_stream"] = self.rl_stream.to_dict()
        if self.process_documentation is not None:
            payload["process_documentation"] = self.process_documentation
        return payload

    def write_json(self, path: Path) -> None:
        _write_json(path, self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureSelectionManifest:
        contracts = [
            FeatureSelectionContractRecord(
                contract=c["contract"],
                input_path=c["input_path"],
                metric_path=c["metric_path"],
            )
            for c in data.get("contracts", [])
        ]
        filtered_outputs = None
        if "filtered_outputs" in data and data["filtered_outputs"] is not None:
            filtered_outputs = [
                FilteredOutputRecord(
                    contract=f["contract"],
                    output_path=f["output_path"],
                    output_row_count=f["output_row_count"],
                    output_column_count=f["output_column_count"],
                )
                for f in data["filtered_outputs"]
            ]
        vae_slope_stream = (
            StreamAuditRecord.from_dict(data["vae_slope_stream"])
            if "vae_slope_stream" in data and data["vae_slope_stream"] is not None
            else None
        )
        vae_volatility_stream = (
            StreamAuditRecord.from_dict(data["vae_volatility_stream"])
            if "vae_volatility_stream" in data and data["vae_volatility_stream"] is not None
            else None
        )
        rl_stream = (
            StreamAuditRecord.from_dict(data["rl_stream"])
            if "rl_stream" in data and data["rl_stream"] is not None
            else None
        )
        return cls(
            symbol=data["symbol"],
            target_freq=data["target_freq"],
            stage=data["stage"],
            split_input_dir=data["split_input_dir"],
            windows_list=list(data["windows_list"]),
            aggregate_metrics_path=data["aggregate_metrics_path"],
            contracts=contracts,
            rl_feature_file=data.get("rl_feature_file"),
            vae_slope_feature_file=data.get("vae_slope_feature_file"),
            vae_volatility_feature_file=data.get("vae_volatility_feature_file"),
            union_selected_feature_count=data.get("union_selected_feature_count"),
            union_selected_features=data.get("union_selected_features"),
            composite_drop_ratio=data.get("composite_drop_ratio"),
            global_feature_blacklist=data.get("global_feature_blacklist"),
            feature_ablation_patterns=data.get("feature_ablation_patterns"),
            rank_ic_mode=data.get("rank_ic_mode"),
            mandatory_state_features=data.get("mandatory_state_features"),
            shared_filter_results=data.get("shared_filter_results"),
            persistence_filter=data.get("persistence_filter"),
            persistence_diagnostics=data.get("persistence_diagnostics"),
            filtered_outputs=filtered_outputs,
            evaluated_feature_file=data.get("evaluated_feature_file"),
            evaluated_feature_count=data.get("evaluated_feature_count"),
            evaluated_features=data.get("evaluated_features"),
            report_only=data.get("report_only"),
            regime_bins=data.get("regime_bins"),
            target_regime_bins=data.get("target_regime_bins"),
            regime_quantiles=data.get("regime_quantiles"),
            regime_audit_path=data.get("regime_audit_path"),
            distribution_audit_path=data.get("distribution_audit_path"),
            global_max_mean_psi=data.get("global_max_mean_psi"),
            global_max_pair_psi=data.get("global_max_pair_psi"),
            min_drift_survivors=data.get("min_drift_survivors"),
            global_min_sign_consistency=data.get("global_min_sign_consistency"),
            conditional_anchors_retained=data.get("conditional_anchors_retained"),
            stream_mode=data.get("stream_mode"),
            vae_slope_stream=vae_slope_stream,
            vae_volatility_stream=vae_volatility_stream,
            rl_stream=rl_stream,
            process_documentation=data.get("process_documentation"),
        )

    @classmethod
    def read_json(cls, path: Path) -> FeatureSelectionManifest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)


@dataclass
class FeatureUnionManifest:
    symbol: str
    target_freq: str
    start_date: str
    end_date: str
    summary_path: str
    contracts: list[str]
    contract_state_feature_paths: dict[str, str]
    per_contract_feature_counts: dict[str, int]
    state_feature_count: int
    state_features: list[str]
    candidate_source_path: str | None
    all_feature_path: str
    ic_result_path: str
    finalize_filtered_df: bool
    per_contract_output_paths: dict[str, str] = field(default_factory=dict)
    per_contract_output_shapes: dict[str, ContractOutputShape] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "target_freq": self.target_freq,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "summary_path": self.summary_path,
            "contracts": list(self.contracts),
            "contract_state_feature_paths": dict(self.contract_state_feature_paths),
            "per_contract_feature_counts": dict(self.per_contract_feature_counts),
            "state_feature_count": self.state_feature_count,
            "state_features": list(self.state_features),
            "candidate_source_path": self.candidate_source_path,
            "all_feature_path": self.all_feature_path,
            "ic_result_path": self.ic_result_path,
            "finalize_filtered_df": self.finalize_filtered_df,
            "per_contract_output_paths": dict(self.per_contract_output_paths),
            "per_contract_output_shapes": {
                contract: shape.to_dict()
                for contract, shape in self.per_contract_output_shapes.items()
            },
        }

    def write_json(self, path: Path) -> None:
        _write_json(path, self.to_dict())


@dataclass
class FeatureScoreWindow:
    window_length: int
    scores: dict[str, float]

    def to_dict(self) -> dict[str, float]:
        return {
            str(feature): float(score)
            for feature, score in self.scores.items()
        }

    def write_json(self, path: Path) -> None:
        _write_json(path, self.to_dict())


@dataclass
class FeatureSelectionResult:
    output_dir: Path
    manifest: FeatureSelectionManifest


@dataclass
class FeatureUnionResult:
    output_dir: Path
    manifest: FeatureUnionManifest


@dataclass
class IcCorrelationResult:
    frame: pl.DataFrame
    output_dir: Path
    selected_features: list[str]
    score_windows: list[FeatureScoreWindow]


@dataclass
class RankIcCorrelationResult:
    frame: pl.DataFrame
    output_dir: Path
    selected_features: list[str]
    score_windows: list[FeatureScoreWindow]
