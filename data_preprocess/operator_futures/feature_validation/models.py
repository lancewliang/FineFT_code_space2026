from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


StageStatus = Literal["pass", "fail", "partial", "error"]


@dataclass(frozen=True)
class ValidationConfig:
    root_path: Path
    symbol: str
    target_freq: str
    start_date: str
    end_date: str
    report_dir: Path
    tolerance: float = 1e-9
    sample_size: int = 200
    orderbook_depth: int = 5


@dataclass(frozen=True)
class Mismatch:
    stage: str
    column: str
    timestamp: str
    actual: Any
    expected: Any
    abs_diff: float


@dataclass
class StageResult:
    stage: str
    status: StageStatus
    checked_columns: int = 0
    missing_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    unverified_columns: list[str] = field(default_factory=list)
    mismatched_columns: list[str] = field(default_factory=list)
    max_abs_diff: float = 0.0
    sample_failures: list[Mismatch] = field(default_factory=list)
    message: str = ""


@dataclass
class ValidationReport:
    config: ValidationConfig
    stages: list[StageResult]
