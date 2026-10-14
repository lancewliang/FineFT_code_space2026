"""Candidate feature qualification and admissibility filtering."""

from typing import List, Set, Tuple
import polars as pl
from .contracts import CandidateFeature


EXCLUDED_EXACT_NAMES: Set[str] = {
    "contract", "timestamp", "trading_day", "date", "time", "datetime",
    "mark_price", "close", "open", "high", "low", "volume", "open_interest",
    "midprice", "LastPrice", "target_return", "reward", "action", "position"
}

EXCLUDED_PREFIXES: Tuple[str, ...] = ("label_", "target_", "future_", "reward_")
EXCLUDED_SUFFIXES: Tuple[str, ...] = ("_right", "_y", "_duplicate")


def qualify_candidates(schema: pl.Schema) -> List[CandidateFeature]:
    """Qualifies candidate features from dataset schema."""
    candidates: List[CandidateFeature] = []

    for col_name, dtype in schema.items():
        # Check type
        is_numeric = dtype in [pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]
        
        if not is_numeric:
            candidates.append(CandidateFeature(
                name=col_name,
                dtype=str(dtype),
                is_qualified=False,
                exclusion_reason="non_numeric_type"
            ))
            continue

        # Check exact exclusions
        if col_name in EXCLUDED_EXACT_NAMES:
            candidates.append(CandidateFeature(
                name=col_name,
                dtype=str(dtype),
                is_qualified=False,
                exclusion_reason="identifier_or_price_or_label"
            ))
            continue

        # Check prefixes/suffixes
        col_lower = col_name.lower()
        if any(col_lower.startswith(p) for p in EXCLUDED_PREFIXES):
            candidates.append(CandidateFeature(
                name=col_name,
                dtype=str(dtype),
                is_qualified=False,
                exclusion_reason="future_label_prefix"
            ))
            continue

        if any(col_lower.endswith(s) for s in EXCLUDED_SUFFIXES):
            candidates.append(CandidateFeature(
                name=col_name,
                dtype=str(dtype),
                is_qualified=False,
                exclusion_reason="duplicate_merge_column"
            ))
            continue

        candidates.append(CandidateFeature(
            name=col_name,
            dtype=str(dtype),
            is_qualified=True
        ))

    return candidates
