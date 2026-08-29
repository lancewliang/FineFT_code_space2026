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
    selected_feature_file: str | None = None
    selected_feature_count: int | None = None
    selected_features: list[str] | None = None
    composite_drop_ratio: float | None = None
    feature_blacklist: list[str] | None = None
    feature_ablation_patterns: list[str] | None = None
    rank_ic_mode: str | None = None
    mandatory_state_features: list[str] | None = None
    filter_results: dict[str, list[str]] | None = None
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
    conditional_anchors_retained: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "symbol": self.symbol,
            "target_freq": self.target_freq,
            "stage": self.stage,
            "split_input_dir": self.split_input_dir,
        }
        if self.selected_feature_file is not None:
            payload["selected_feature_file"] = self.selected_feature_file
            payload["selected_feature_count"] = self.selected_feature_count
            payload["selected_features"] = list(self.selected_features or [])
        if self.evaluated_feature_file is not None:
            payload["evaluated_feature_file"] = self.evaluated_feature_file
            payload["evaluated_feature_count"] = self.evaluated_feature_count
            payload["evaluated_features"] = list(self.evaluated_features or [])
        payload["windows_list"] = list(self.windows_list)
        if self.composite_drop_ratio is not None:
            payload["composite_drop_ratio"] = self.composite_drop_ratio
        if self.feature_blacklist is not None:
            payload["feature_blacklist"] = list(self.feature_blacklist)
        if self.feature_ablation_patterns is not None:
            payload["feature_ablation_patterns"] = list(self.feature_ablation_patterns)
        if self.rank_ic_mode is not None:
            payload["rank_ic_mode"] = self.rank_ic_mode
        if self.mandatory_state_features is not None:
            payload["mandatory_state_features"] = list(self.mandatory_state_features)
        payload["aggregate_metrics_path"] = self.aggregate_metrics_path
        if self.filter_results is not None:
            payload["filter_results"] = {
                key: list(values) for key, values in self.filter_results.items()
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
        if self.conditional_anchors_retained is not None:
            payload["conditional_anchors_retained"] = self.conditional_anchors_retained
        return payload

    def write_json(self, path: Path) -> None:
        _write_json(path, self.to_dict())


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
