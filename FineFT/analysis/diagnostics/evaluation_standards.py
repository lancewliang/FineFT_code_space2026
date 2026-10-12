"""Versioned research thresholds; these grades do not certify deployment readiness."""

from __future__ import annotations

import math
from typing import Mapping

METHODOLOGY_VERSION = "1.1"


def research_status(grades: Mapping[str, str], audit_required: bool = False) -> str:
    if audit_required or any(g in {"NeedsAudit", "Unavailable"} for g in grades.values()):
        return "NEEDS_AUDIT"
    if "Fail" in grades.values():
        return "RESEARCH_FAIL"
    if "Warning" in grades.values():
        return "RESEARCH_WARNING"
    return "RESEARCH_STRONG"


def grade_research_metrics(metrics: Mapping[str, float | None]) -> dict[str, str]:
    """Apply methodology section 2, including exact boundary conventions."""
    rules = {
        "portfolio_return_pct": lambda x: "Fail" if x <= 0 else "Strong" if x > 20 else "Warning",
        "portfolio_gross_return_pct": lambda x: "Fail" if x < 10 else "Strong" if x > 40 else "Warning",
        "friction_consumption_ratio_pct": lambda x: "NeedsAudit" if x < 0 else "Strong" if x < 50 else "Warning" if x <= 75 else "Fail",
        "portfolio_turnover_multiple": lambda x: "Strong" if x < 30 else "Warning" if x <= 80 else "Fail",
        "mean_annual_sr": lambda x: "Strong" if x > 2 else "Warning" if x >= 1 else "Fail",
        "mean_mdd_pct": lambda x: "Strong" if x < 8 else "Warning" if x <= 15 else "Fail",
        "mean_daily_cr": lambda x: "Strong" if x > 2.5 else "Warning" if x >= 1 else "Fail",
        "win_rate_pct": lambda x: "Strong" if x >= 85 else "Warning" if x >= 70 else "Fail",
        "worst_contract_loss_pct": lambda x: "Strong" if x < 5 else "Warning" if x <= 12 else "Fail",
    }
    grades = {}
    for name, rule in rules.items():
        value = metrics.get(name)
        grades[name] = rule(value) if value is not None and math.isfinite(value) else "Unavailable"
    return grades
