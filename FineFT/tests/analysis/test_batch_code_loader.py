from __future__ import annotations

from pathlib import Path
import pytest

from analysis.diagnostics.batch_code_loader import (
    BatchCodeLoader,
    OutlineAstVisitor,
    resolve_profile_files,
)


def test_batch_code_loader_outline_and_targeted(tmp_path: Path):
    file_a = tmp_path / "mock_strategy.py"
    file_a.write_text(
        '''"""Mock strategy module."""

import argparse

class MockQNet:
    """QNet documentation."""
    def __init__(self, state_dim: int, action_dim: int = 3) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, state: list[float]) -> list[float]:
        """Forward pass."""
        return [0.0] * self.action_dim

def execute_stop_loss(price_diff: float, threshold: float = 0.015) -> bool:
    """Trigger hard stop loss."""
    return price_diff > threshold

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop_loss_threshold", type=float, default=0.015, help="Stop loss return threshold")
'''
    )

    loader = BatchCodeLoader(root_dir=tmp_path, file_paths=[str(file_a)])

    # Test outline extraction
    outline = loader.extract_outline(file_a)
    assert outline.module_docstring == "Mock strategy module."
    assert outline.total_lines == 21
    assert len(outline.classes) == 1
    assert outline.classes[0].name == "MockQNet"
    assert len(outline.classes[0].methods) == 2
    assert outline.classes[0].methods[0].name == "__init__"

    assert len(outline.functions) == 2
    func_names = [f.name for f in outline.functions]
    assert "execute_stop_loss" in func_names
    assert "main" in func_names

    assert len(outline.cli_arguments) == 1
    assert outline.cli_arguments[0].flags == ["--stop_loss_threshold"]
    assert outline.cli_arguments[0].default_val == "0.015"

    # Test outline formatted report
    rep_outline = loader.format_outline_report()
    assert "MockQNet" in rep_outline
    assert "execute_stop_loss" in rep_outline
    assert "--stop_loss_threshold" in rep_outline

    # Test targeted keyword search
    rep_targeted = loader.format_targeted_report(keywords=["stop_loss"])
    assert "execute_stop_loss" in rep_targeted

    # Test full bundle format
    rep_full = loader.format_full_bundle()
    assert "File: `mock_strategy.py`" in rep_full
    assert "class MockQNet:" in rep_full


def test_outline_reports_class_schema_fields(tmp_path: Path):
    schema_file = tmp_path / "audit_schema.py"
    schema_file.write_text(
        '''from dataclasses import dataclass
from enum import IntEnum

@dataclass
class AccountRecord:
    snapshot_id: str
    equity: float = 0.0

class DecisionReason(IntEnum):
    POLICY_INFERENCE = 0
    HARD_STOP_LOSS = 1
'''
    )

    loader = BatchCodeLoader(root_dir=tmp_path, file_paths=[str(schema_file)])
    outline = loader.extract_outline(schema_file)
    fields_by_class = {
        item.name: [field.name for field in item.fields] for item in outline.classes
    }

    assert fields_by_class["AccountRecord"] == ["snapshot_id", "equity"]
    assert fields_by_class["DecisionReason"] == [
        "POLICY_INFERENCE",
        "HARD_STOP_LOSS",
    ]
    report = loader.format_outline_report()
    assert "snapshot_id: str" in report
    assert "POLICY_INFERENCE = 0" in report


def test_targeted_mode_returns_matching_method_instead_of_whole_class(tmp_path: Path):
    env_file = tmp_path / "large_env.py"
    env_file.write_text(
        '''class LargeEnvironment:
    def unrelated_calculation(self) -> int:
        return sum(range(100))

    def record_account_snapshot(self, equity: float) -> dict[str, float]:
        account_snapshot = {"equity": equity}
        return account_snapshot
'''
    )

    loader = BatchCodeLoader(root_dir=tmp_path, file_paths=[str(env_file)])
    blocks = loader.extract_targeted(env_file, ["account_snapshot"])

    assert len(blocks) == 1
    assert blocks[0][2] == "def LargeEnvironment.record_account_snapshot"
    assert "record_account_snapshot" in blocks[0][3]
    assert "unrelated_calculation" not in blocks[0][3]


def test_code_profiles_cover_strategy_and_audit_semantics_without_duplicates():
    strategy_files = resolve_profile_files("strategy")
    audit_files = resolve_profile_files("audit")
    all_files = resolve_profile_files("all")

    assert "FineFT/RL/DiHFT/high_level/gating/hierarchical_gating.py" in strategy_files
    assert "FineFT/env/env_class/base_env.py" in audit_files
    assert "FineFT/env/env_class/futures_util.py" in audit_files
    assert len(all_files) == len(set(all_files))
    assert set(strategy_files) | set(audit_files) == set(all_files)


def test_unknown_code_profile_fails_with_available_choices():
    with pytest.raises(ValueError, match="all, audit, strategy"):
        resolve_profile_files("missing")
