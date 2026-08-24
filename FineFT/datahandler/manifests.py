from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


STAGES = ("train", "valid", "test")


def _copy_optional_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    return list(value)


@dataclass
class DatasetSplitContract:
    contract: str
    range: list[str] | None = None
    trading_days: list[str] | None = None
    output_row_count: int | None = None

    @classmethod
    def from_dict(cls, item: dict[str, Any], stage: str) -> "DatasetSplitContract":
        if not isinstance(item, dict) or not isinstance(item.get("contract"), str):
            raise ValueError(
                f"dataset split manifest sets.{stage}.contracts items need contract"
            )
        output_row_count = item.get("output_row_count")
        return cls(
            contract=item["contract"],
            range=_copy_optional_list(item.get("range")),
            trading_days=_copy_optional_list(item.get("trading_days")),
            output_row_count=(
                int(output_row_count) if output_row_count is not None else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"contract": self.contract}
        if self.range is not None:
            payload["range"] = list(self.range)
        if self.trading_days is not None:
            payload["trading_days"] = list(self.trading_days)
        if self.output_row_count is not None:
            payload["output_row_count"] = self.output_row_count
        return payload


@dataclass
class DatasetSkippedContract:
    contract: str
    reason: str | None = None
    extra_fields: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetSkippedContract":
        extra_fields = {
            key: value
            for key, value in payload.items()
            if key not in {"contract", "reason"}
        }
        return cls(
            contract=payload["contract"],
            reason=payload.get("reason"),
            extra_fields=extra_fields,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"contract": self.contract}
        if self.reason is not None:
            payload["reason"] = self.reason
        payload.update(self.extra_fields)
        return payload


@dataclass
class DatasetSplitSet:
    range: list[str] | None = None
    contracts: list[DatasetSplitContract] = field(default_factory=list)
    skipped_contracts: list[DatasetSkippedContract] = field(default_factory=list)

    @classmethod
    def from_dict(cls, stage: str, stage_info: Any) -> "DatasetSplitSet":
        if not isinstance(stage_info, dict):
            raise ValueError(f"dataset split manifest missing sets.{stage}")
        if "contracts" not in stage_info or not isinstance(
            stage_info["contracts"], list
        ):
            raise ValueError(
                f"dataset split manifest sets.{stage}.contracts must be a list"
            )
        return cls(
            range=_copy_optional_list(stage_info.get("range")),
            contracts=[
                DatasetSplitContract.from_dict(item, stage)
                for item in stage_info["contracts"]
            ],
            skipped_contracts=[
                DatasetSkippedContract.from_dict(item)
                for item in stage_info.get("skipped_contracts", [])
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "range": list(self.range) if self.range is not None else None,
            "contracts": [contract.to_dict() for contract in self.contracts],
            "skipped_contracts": [
                skipped_contract.to_dict()
                for skipped_contract in self.skipped_contracts
            ],
        }


@dataclass
class DatasetSplitManifest:
    symbol: str
    target_freq: str
    sets: dict[str, DatasetSplitSet]

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        symbol: str,
        target_freq: str,
    ) -> "DatasetSplitManifest":
        if payload.get("symbol") != symbol:
            raise ValueError(
                "dataset split manifest symbol mismatch: "
                f"expected={symbol} actual={payload.get('symbol')}"
            )
        if payload.get("target_freq") != target_freq:
            raise ValueError(
                "dataset split manifest target_freq mismatch: "
                f"expected={target_freq} actual={payload.get('target_freq')}"
            )
        sets = payload.get("sets")
        if not isinstance(sets, dict):
            raise ValueError("dataset split manifest missing sets")
        return cls(
            symbol=symbol,
            target_freq=target_freq,
            sets={
                stage: DatasetSplitSet.from_dict(stage, sets.get(stage))
                for stage in STAGES
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "target_freq": self.target_freq,
            "sets": {stage: self.sets[stage].to_dict() for stage in STAGES},
        }


@dataclass
class DatasetSliceOutput:
    index: int
    path: str
    row_start: int
    row_end: int
    contract: str | None = None
    source_output: str | None = None
    output_row_count: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetSliceOutput":
        return cls(
            index=int(payload["index"]),
            path=payload["path"],
            row_start=int(payload["row_start"]),
            row_end=int(payload["row_end"]),
            contract=payload.get("contract"),
            source_output=payload.get("source_output"),
            output_row_count=(
                int(payload["output_row_count"])
                if "output_row_count" in payload
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "index": self.index,
        }
        if self.contract is not None:
            payload["contract"] = self.contract
        payload["path"] = self.path
        if self.source_output is not None:
            payload["source_output"] = self.source_output
        payload["row_start"] = self.row_start
        payload["row_end"] = self.row_end
        if self.output_row_count is not None:
            payload["output_row_count"] = self.output_row_count
        return payload


@dataclass
class DatasetContractManifest:
    contract: str
    input_path: str
    output_path: str
    range: list[str] | None = None
    trading_days: list[str] | None = None
    output_row_count: int | None = None
    slice_outputs: list[DatasetSliceOutput] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetContractManifest":
        return cls(
            contract=payload["contract"],
            input_path=payload["input_path"],
            output_path=payload["output_path"],
            range=_copy_optional_list(payload.get("range")),
            trading_days=_copy_optional_list(payload.get("trading_days")),
            output_row_count=(
                int(payload["output_row_count"])
                if "output_row_count" in payload
                else None
            ),
            slice_outputs=[
                DatasetSliceOutput.from_dict(item)
                for item in payload.get("slice_outputs", [])
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contract": self.contract,
            "input_path": self.input_path,
            "output_path": self.output_path,
        }
        if self.range is not None:
            payload["range"] = list(self.range)
        if self.trading_days is not None:
            payload["trading_days"] = list(self.trading_days)
        if self.output_row_count is not None:
            payload["output_row_count"] = self.output_row_count
        if self.slice_outputs:
            payload["slice_outputs"] = [
                slice_output.to_dict() for slice_output in self.slice_outputs
            ]
        return payload


@dataclass
class DatasetSetManifest:
    range: list[str] | None = None
    contracts: list[DatasetContractManifest] = field(default_factory=list)
    skipped_contracts: list[DatasetSkippedContract] = field(default_factory=list)
    contracts_total_count: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetSetManifest":
        return cls(
            range=_copy_optional_list(payload.get("range")),
            contracts=[
                DatasetContractManifest.from_dict(item)
                for item in payload.get("contracts", [])
            ],
            skipped_contracts=[
                DatasetSkippedContract.from_dict(item)
                for item in payload.get("skipped_contracts", [])
            ],
            contracts_total_count=(
                int(payload["contracts_total_count"])
                if "contracts_total_count" in payload
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "range": list(self.range) if self.range is not None else None,
            "contracts": [contract.to_dict() for contract in self.contracts],
            "skipped_contracts": [
                skipped_contract.to_dict()
                for skipped_contract in self.skipped_contracts
            ],
        }
        if self.contracts_total_count is not None:
            payload["contracts_total_count"] = self.contracts_total_count
        return payload


@dataclass
class DatasetManifest:
    symbol: str
    target_freq: str
    dataset_split_manifest_path: str
    state_features_source_path: str
    state_features_path: str
    sets: dict[str, DatasetSetManifest]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetManifest":
        return cls(
            symbol=payload.get("symbol", ""),
            target_freq=payload.get("target_freq", ""),
            dataset_split_manifest_path=payload.get(
                "dataset_split_manifest_path", ""
            ),
            state_features_source_path=payload["state_features_source_path"],
            state_features_path=payload["state_features_path"],
            sets={
                stage: DatasetSetManifest.from_dict(stage_payload)
                for stage, stage_payload in payload.get("sets", {}).items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "target_freq": self.target_freq,
            "dataset_split_manifest_path": self.dataset_split_manifest_path,
            "state_features_source_path": self.state_features_source_path,
            "state_features_path": self.state_features_path,
            "sets": {
                stage: set_info.to_dict() for stage, set_info in self.sets.items()
            },
        }


@dataclass
class SliceFileManifest:
    path: str
    output_row_count: int
    contract: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SliceFileManifest":
        return cls(
            path=payload["path"],
            output_row_count=int(payload["output_row_count"]),
            contract=payload.get("contract"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": self.path,
            "output_row_count": self.output_row_count,
        }
        if self.contract is not None:
            payload["contract"] = self.contract
        return payload


@dataclass
class SliceLabelManifest:
    label: str
    file_count: int = 0
    total_row_count: int = 0
    files: list[SliceFileManifest] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SliceLabelManifest":
        return cls(
            label=payload["label"],
            file_count=int(payload.get("file_count", 0)),
            total_row_count=int(payload.get("total_row_count", 0)),
            files=[
                SliceFileManifest.from_dict(item)
                for item in payload.get("files", [])
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "file_count": self.file_count,
            "total_row_count": self.total_row_count,
            "files": [file_info.to_dict() for file_info in self.files],
        }


@dataclass
class SliceContractManifest:
    contract: str
    processed_path: str
    file_count: int = 0
    total_row_count: int = 0
    input_row_count: int | None = None
    labels: dict[str, SliceLabelManifest] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.labels and not self.file_count:
            self.file_count = sum(label.file_count for label in self.labels.values())
        if self.labels and not self.total_row_count:
            self.total_row_count = sum(
                label.total_row_count for label in self.labels.values()
            )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SliceContractManifest":
        return cls(
            contract=payload["contract"],
            processed_path=payload["processed_path"],
            file_count=int(payload.get("file_count", 0)),
            total_row_count=int(payload.get("total_row_count", 0)),
            input_row_count=(
                int(payload["input_row_count"])
                if payload.get("input_row_count") is not None
                else None
            ),
            labels={
                label: SliceLabelManifest.from_dict(label_info)
                for label, label_info in payload.get("labels", {}).items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "contract": self.contract,
            "processed_path": self.processed_path,
            "file_count": self.file_count,
            "total_row_count": self.total_row_count,
            "labels": {
                label: label_info.to_dict()
                for label, label_info in sorted(self.labels.items())
            },
        }
        if self.input_row_count is not None:
            payload["input_row_count"] = self.input_row_count
        return payload


@dataclass
class SkippedContractManifest:
    contract: str
    processed_path: str
    reason: str
    input_row_count: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SkippedContractManifest":
        return cls(
            contract=payload["contract"],
            processed_path=payload["processed_path"],
            reason=payload["reason"],
            input_row_count=int(payload["input_row_count"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "processed_path": self.processed_path,
            "reason": self.reason,
            "input_row_count": self.input_row_count,
        }


@dataclass
class SliceManifest:
    valid_path: str
    contracts: dict[str, SliceContractManifest] = field(default_factory=dict)
    labels: dict[str, SliceLabelManifest] = field(default_factory=dict)
    skipped_contracts: dict[str, SkippedContractManifest] = field(
        default_factory=dict
    )
    calibration: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SliceManifest":
        manifest = cls(
            valid_path=payload["valid_path"],
            contracts={
                contract: SliceContractManifest.from_dict(contract_info)
                for contract, contract_info in payload.get("contracts", {}).items()
            },
            skipped_contracts={
                contract: SkippedContractManifest.from_dict(skip_info)
                for contract, skip_info in payload.get("skipped_contracts", {}).items()
            },
            calibration=payload.get("calibration"),
        )
        manifest.rebuild_labels()
        manifest.sort()
        return manifest

    @classmethod
    def new(cls, valid_root: object) -> "SliceManifest":
        return cls(valid_path=str(valid_root))

    def replace_contract(self, contract_record: SliceContractManifest) -> None:
        self.skipped_contracts.pop(contract_record.contract, None)
        if contract_record.file_count:
            self.contracts[contract_record.contract] = contract_record
        else:
            self.contracts.pop(contract_record.contract, None)
        self.rebuild_labels()
        self.sort()

    def record_skipped_contract(
        self,
        *,
        contract: str,
        processed_path: str,
        reason: str,
        input_row_count: int,
    ) -> None:
        self.contracts.pop(contract, None)
        self.skipped_contracts[contract] = SkippedContractManifest(
            contract=contract,
            processed_path=processed_path,
            reason=reason,
            input_row_count=int(input_row_count),
        )
        self.rebuild_labels()
        self.sort()

    def rebuild_labels(self) -> None:
        labels: dict[str, SliceLabelManifest] = {}
        for contract_record in self.contracts.values():
            for label, label_info in contract_record.labels.items():
                target = labels.setdefault(label, SliceLabelManifest(label=label))
                target.file_count += label_info.file_count
                target.total_row_count += label_info.total_row_count
                for file_info in label_info.files:
                    target.files.append(
                        SliceFileManifest(
                            contract=contract_record.contract,
                            path=file_info.path,
                            output_row_count=file_info.output_row_count,
                        )
                    )
        self.labels = dict(sorted(labels.items()))

    def sort(self) -> None:
        self.contracts = dict(sorted(self.contracts.items()))
        self.labels = dict(sorted(self.labels.items()))
        self.skipped_contracts = dict(sorted(self.skipped_contracts.items()))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "valid_path": self.valid_path,
            "contracts": {
                contract: contract_info.to_dict()
                for contract, contract_info in sorted(self.contracts.items())
            },
            "labels": {
                label: label_info.to_dict()
                for label, label_info in sorted(self.labels.items())
            },
            "skipped_contracts": {
                contract: skip_info.to_dict()
                for contract, skip_info in sorted(self.skipped_contracts.items())
            },
        }
        if self.calibration is not None:
            payload["calibration"] = self.calibration
        return payload
