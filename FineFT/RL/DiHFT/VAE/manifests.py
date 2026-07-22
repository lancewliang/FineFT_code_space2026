from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabelArraySource:
    contract: str
    source_file: str


@dataclass(frozen=True)
class TestContractSource:
    __test__ = False

    contract: str
    source_file: str


@dataclass(frozen=True)
class ContractDatasetLoader:
    contract: str
    source_file: str
    loader: object


@dataclass(frozen=True)
class LabelContractSource:
    contract: str
    source_file: str
    sample_count: int

    def to_dict(self) -> dict:
        return {
            "contract": self.contract,
            "source_file": self.source_file,
            "sample_count": int(self.sample_count),
        }


@dataclass(frozen=True)
class LabelTrainingManifest:
    dataset_name: str
    label: str
    merged_path: str
    total_samples: int
    feature_dim: int
    included_contracts: list[LabelContractSource]
    missing_contracts: list[str]

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "label": self.label,
            "merged_path": self.merged_path,
            "total_samples": int(self.total_samples),
            "feature_dim": int(self.feature_dim),
            "included_contracts": [
                item.to_dict() for item in self.included_contracts
            ],
            "missing_contracts": list(self.missing_contracts),
        }


@dataclass(frozen=True)
class ContractLogpxResult:
    contract: str
    source_file: str
    logpx: object
    input_samples: int | None = None


@dataclass(frozen=True)
class TrainBaselineLogpx:
    source_file: str
    logpx: object
    input_samples: int
    analyzed_samples: int


@dataclass(frozen=True)
class SampleIntegrity:
    input_samples: int
    analyzed_samples: int

    @property
    def sample_mismatch(self) -> bool:
        return int(self.input_samples) != int(self.analyzed_samples)

    def to_dict(self) -> dict:
        return {
            "input_samples": int(self.input_samples),
            "analyzed_samples": int(self.analyzed_samples),
            "sample_mismatch": self.sample_mismatch,
        }


@dataclass(frozen=True)
class LogpxStats:
    samples: int
    logpx_mean: float
    logpx_std: float
    logpx_min: float
    logpx_max: float
    quantiles: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "samples": int(self.samples),
            "logpx_mean": float(self.logpx_mean),
            "logpx_std": float(self.logpx_std),
            "logpx_min": float(self.logpx_min),
            "logpx_max": float(self.logpx_max),
            "quantiles": {
                key: float(value) for key, value in self.quantiles.items()
            },
        }


@dataclass(frozen=True)
class AcceptanceStats:
    ge_train_q01_pct: float
    ge_train_q05_pct: float
    ge_train_q50_pct: float

    def to_dict(self) -> dict:
        return {
            "ge_train_q01_pct": float(self.ge_train_q01_pct),
            "ge_train_q05_pct": float(self.ge_train_q05_pct),
            "ge_train_q50_pct": float(self.ge_train_q50_pct),
        }


@dataclass(frozen=True)
class LogpxSummary:
    integrity: SampleIntegrity
    stats: LogpxStats
    acceptance: AcceptanceStats | None = None

    def to_dict(self) -> dict:
        payload = {
            **self.integrity.to_dict(),
            **self.stats.to_dict(),
        }
        if self.acceptance is not None:
            payload["acceptance"] = self.acceptance.to_dict()
        return payload


@dataclass(frozen=True)
class ContractLogpxSummary:
    source_file: str
    summary: LogpxSummary

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            **self.summary.to_dict(),
        }


@dataclass(frozen=True)
class LabelTestSummary:
    contracts: dict[str, ContractLogpxSummary]
    all: LogpxSummary

    def to_dict(self) -> dict:
        return {
            "contracts": {
                contract: summary.to_dict()
                for contract, summary in self.contracts.items()
            },
            "all": self.all.to_dict(),
        }


@dataclass(frozen=True)
class LabelSummary:
    dataset_name: str
    label: str
    test: LabelTestSummary
    train_baseline: ContractLogpxSummary | None = None

    def to_dict(self) -> dict:
        payload = {
            "dataset_name": self.dataset_name,
            "label": self.label,
            "test": self.test.to_dict(),
        }
        if self.train_baseline is not None:
            payload["train_baseline"] = self.train_baseline.to_dict()
        return payload


@dataclass(frozen=True)
class WinnerSummary:
    samples: int
    winner_counts: dict[str, int]
    winner_pct: dict[str, float]
    top1_top2_margin_mean: float
    top1_top2_margin_q25: float
    low_margin_pct: float

    def to_dict(self) -> dict:
        return {
            "samples": int(self.samples),
            "winner_counts": {
                label: int(count) for label, count in self.winner_counts.items()
            },
            "winner_pct": {
                label: float(value) for label, value in self.winner_pct.items()
            },
            "top1_top2_margin_mean": float(self.top1_top2_margin_mean),
            "top1_top2_margin_q25": float(self.top1_top2_margin_q25),
            "low_margin_pct": float(self.low_margin_pct),
        }


@dataclass(frozen=True)
class ContractRoutingSummary:
    winner: WinnerSummary
    input_samples_by_label: dict[str, int]
    sample_mismatch: bool

    def to_dict(self) -> dict:
        return {
            **self.winner.to_dict(),
            "input_samples_by_label": {
                label: int(samples)
                for label, samples in self.input_samples_by_label.items()
            },
            "sample_mismatch": bool(self.sample_mismatch),
        }


@dataclass(frozen=True)
class RoutingSummary:
    dataset_name: str
    labels: list[str]
    score_type: str
    low_margin_threshold: float
    contracts: dict[str, ContractRoutingSummary]
    all: WinnerSummary

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "labels": list(self.labels),
            "score_type": self.score_type,
            "low_margin_threshold": float(self.low_margin_threshold),
            "contracts": {
                contract: summary.to_dict()
                for contract, summary in self.contracts.items()
            },
            "all": self.all.to_dict(),
        }
