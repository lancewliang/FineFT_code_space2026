"""Configuration loading, validation, and fingerprinting."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class ReferencePriceProtocol:
    field: str = "LastPrice"
    observation: str = "latest_valid_snapshot_at_or_before_target"
    same_contract: bool = True
    same_session: bool = True
    max_quote_age_seconds: int = 60
    fallback: str = "none"


@dataclass
class EvidenceProtocol:
    method: str = "joint_trading_day_moving_block_bootstrap"
    confidence_level: float = 0.95
    replicates: int = 2000
    seed: int = 42
    minimum_block_trading_days: int = 20
    respect_max_label_trading_day_span: bool = True
    sensitivity_block_multiplier: int = 2
    minimum_effective_blocks: int = 20
    minimum_contracts_for_sign_consistency: int = 3
    multiple_comparison_adjustment: str = "none_exploratory_only"
    automatic_pass_list: bool = False


@dataclass
class AuditConfig:
    schema_version: str
    protocol_version: str
    symbol: str
    frequency: str
    horizons: List[str]
    calendar_manifest: str
    feature_availability_manifest: str
    price_source_manifest: str
    split_manifest: str
    result_root: str
    hour_clock: str = "active_trading_minutes"
    trading_day_offsets: Dict[str, int] = field(default_factory=lambda: {"today": 0, "next_td": 1, "next_5td": 5, "next_15td": 15})
    contract_policy: str = "same_contract"
    candidate_policy: str = "causal_quality_only"
    targets: List[str] = field(default_factory=lambda: ["return", "abs_return"])
    read_final_test: bool = False
    purge_by_label_available_at: bool = True
    write_selection_outputs: bool = False
    legacy_selection_manifest: Optional[str] = None
    reference_price_protocol: ReferencePriceProtocol = field(default_factory=ReferencePriceProtocol)
    evidence_protocol: EvidenceProtocol = field(default_factory=EvidenceProtocol)
    raw_dict: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditConfig":
        required_keys = [
            "schema_version", "protocol_version", "symbol",
            "calendar_manifest", "feature_availability_manifest",
            "price_source_manifest", "split_manifest", "result_root"
        ]
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise ValueError(f"Missing required configuration fields: {', '.join(missing)}")

        ref_proto_data = data.get("reference_price_protocol", {})
        ref_proto = ReferencePriceProtocol(
            field=ref_proto_data.get("field", "LastPrice"),
            observation=ref_proto_data.get("observation", "latest_valid_snapshot_at_or_before_target"),
            same_contract=ref_proto_data.get("same_contract", True),
            same_session=ref_proto_data.get("same_session", True),
            max_quote_age_seconds=ref_proto_data.get("max_quote_age_seconds", 60),
            fallback=ref_proto_data.get("fallback", "none"),
        )
        ev_proto_data = data.get("evidence_protocol", {})
        ev_proto = EvidenceProtocol(
            method=ev_proto_data.get("method", "joint_trading_day_moving_block_bootstrap"),
            confidence_level=ev_proto_data.get("confidence_level", 0.95),
            replicates=ev_proto_data.get("replicates", 2000),
            seed=ev_proto_data.get("seed", 42),
            minimum_block_trading_days=ev_proto_data.get("minimum_block_trading_days", 20),
            respect_max_label_trading_day_span=ev_proto_data.get("respect_max_label_trading_day_span", True),
            sensitivity_block_multiplier=ev_proto_data.get("sensitivity_block_multiplier", 2),
            minimum_effective_blocks=ev_proto_data.get("minimum_effective_blocks", 20),
            minimum_contracts_for_sign_consistency=ev_proto_data.get("minimum_contracts_for_sign_consistency", 3),
            multiple_comparison_adjustment=ev_proto_data.get("multiple_comparison_adjustment", "none_exploratory_only"),
            automatic_pass_list=ev_proto_data.get("automatic_pass_list", False),
        )

        return cls(
            schema_version=data["schema_version"],
            protocol_version=data["protocol_version"],
            symbol=data["symbol"],
            frequency=data.get("frequency", "10min"),
            horizons=data.get("horizons", ["1h", "today"]),
            calendar_manifest=data["calendar_manifest"],
            feature_availability_manifest=data["feature_availability_manifest"],
            price_source_manifest=data["price_source_manifest"],
            split_manifest=data["split_manifest"],
            result_root=data["result_root"],
            hour_clock=data.get("hour_clock", "active_trading_minutes"),
            trading_day_offsets=data.get("trading_day_offsets", {"today": 0, "next_td": 1, "next_5td": 5, "next_15td": 15}),
            contract_policy=data.get("contract_policy", "same_contract"),
            candidate_policy=data.get("candidate_policy", "causal_quality_only"),
            targets=data.get("targets", ["return", "abs_return"]),
            read_final_test=data.get("read_final_test", False),
            purge_by_label_available_at=data.get("purge_by_label_available_at", True),
            write_selection_outputs=data.get("write_selection_outputs", False),
            legacy_selection_manifest=data.get("legacy_selection_manifest"),
            reference_price_protocol=ref_proto,
            evidence_protocol=ev_proto,
            raw_dict=data,
        )

    @classmethod
    def load(cls, config_path: str | Path) -> "AuditConfig":
        p = Path(config_path)
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found: {p}")
        text = p.read_text(encoding="utf-8")
        if p.suffix in [".yaml", ".yml"]:
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data)

    def compute_fingerprint(self) -> str:
        s = json.dumps(self.raw_dict, sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.schema_version != "multi_horizon_audit.v1":
            errors.append(f"Unsupported schema_version: {self.schema_version}")
        if not self.symbol:
            errors.append("Symbol cannot be empty")
        if self.frequency not in ["10min", "30min"]:
            errors.append(f"Invalid frequency: {self.frequency}. Supported: 10min, 30min")
        
        # Check manifest paths
        for field_name, path_str in [
            ("calendar_manifest", self.calendar_manifest),
            ("feature_availability_manifest", self.feature_availability_manifest),
            ("price_source_manifest", self.price_source_manifest),
            ("split_manifest", self.split_manifest),
        ]:
            if not path_str or not Path(path_str).exists():
                errors.append(f"Manifest file missing or not found for {field_name}: {path_str}")

        if self.write_selection_outputs:
            errors.append("write_selection_outputs must be False to preserve audit independence")

        if self.legacy_selection_manifest and not Path(self.legacy_selection_manifest).is_file():
            errors.append(f"Legacy selection manifest not found: {self.legacy_selection_manifest}")

        return errors
