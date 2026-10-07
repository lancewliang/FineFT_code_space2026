from __future__ import annotations

from pathlib import Path
import pytest

from analysis.diagnostics.batch_code_loader import (
    BatchCodeLoader,
    OutlineAstVisitor,
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
