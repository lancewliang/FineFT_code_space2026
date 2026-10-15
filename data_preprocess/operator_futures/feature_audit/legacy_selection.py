"""Read-only annotations from native three-stream or legacy flat manifests."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

STREAM_FIELDS = (
    ("slope_vae", "vae_slope_stream", "slope_vae_features"),
    ("vol_vae", "vae_volatility_stream", "volatility_vae_features"),
    ("rl_decision", "rl_stream", "rl_decision_features"),
)


@dataclass(frozen=True)
class LegacySelection:
    """Manifest evidence; absent rejection evidence never implies an IC failure."""

    source_path: str
    sha256: str
    selected_streams: dict[str, list[str]]
    drop_reasons: dict[str, list[dict[str, str]]]
    complete_selection: bool

    def membership(self, feature: str) -> str:
        """Preserve every selected stream, even if another stream dropped the feature."""
        streams = self.selected_streams.get(feature, [])
        if streams:
            return ",".join(streams)
        if feature in self.drop_reasons:
            return "rejected"
        return "not_selected" if self.complete_selection else "unassigned"

    def comparison_rows(self, features: list[str]) -> list[dict[str, str | bool | None]]:
        """Return one row per audit candidate, with verbatim scoped drop evidence."""
        return [
            {
                "feature": feature,
                "legacy_stream_membership": self.membership(feature),
                "is_selected": (
                    True if feature in self.selected_streams
                    else False if self.complete_selection else None
                ),
                "selected_streams": json.dumps(self.selected_streams.get(feature, [])),
                "drop_reasons": json.dumps(self.drop_reasons.get(feature, [])),
                "drop_reason_status": "recorded" if feature in self.drop_reasons else "unknown",
                "source_manifest": self.source_path,
                "source_sha256": self.sha256,
            }
            for feature in features
        ]


def _feature_names(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(name, str) for name in value):
        raise ValueError(f"Legacy selection {field} must be a list of feature names")
    return value


def load_legacy_selection(path: str | Path) -> LegacySelection:
    """Load explicit selection evidence; missing files or unknown formats fail loudly."""
    source = Path(path)
    raw = source.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Legacy selection manifest must be a JSON object")
    native = any(nested in data for _, nested, _ in STREAM_FIELDS)
    if not native and not any(flat in data for _, _, flat in STREAM_FIELDS):
        raise ValueError("Unrecognized legacy selection manifest: no supported stream fields")

    selected: dict[str, list[str]] = {}
    reasons: dict[str, list[dict[str, str]]] = {}

    def record_filters(scope: str, filters: object) -> None:
        if not isinstance(filters, dict):
            raise ValueError(f"Legacy selection {scope}.filter_results must be an object")
        for reason, names in filters.items():
            # 'Hard Filter' and 'Composite Score' contain survivors, not rejections.
            if not reason.endswith(" Dropped"):
                continue
            for name in _feature_names(names, f"{scope}.{reason}"):
                reasons.setdefault(name, []).append({"scope": scope, "reason": reason})

    record_filters("global", data.get("filter_results", {}))
    for stream, nested, flat in STREAM_FIELDS:
        if native:
            section = data.get(nested)
            if not isinstance(section, dict) or "selected_features" not in section:
                raise ValueError(f"Legacy selection manifest missing {nested}.selected_features")
            names = _feature_names(section["selected_features"], f"{nested}.selected_features")
            record_filters(stream, section.get("filter_results", {}))
        else:
            names = _feature_names(data.get(flat, []), flat)
        for name in dict.fromkeys(names):
            selected.setdefault(name, []).append(stream)

    rejected = data.get("rejected_features", {})
    if not isinstance(rejected, dict):
        raise ValueError("Legacy selection rejected_features must be an object")
    for name, reason in rejected.items():
        reasons.setdefault(name, []).append({"scope": "legacy", "reason": str(reason)})

    return LegacySelection(
        source_path=str(source.resolve()),
        sha256=hashlib.sha256(raw).hexdigest(),
        selected_streams=selected,
        drop_reasons=reasons,
        complete_selection=native or all(flat in data for _, _, flat in STREAM_FIELDS),
    )
