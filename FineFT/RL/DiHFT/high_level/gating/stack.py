from __future__ import annotations

from typing import TypeVar


StackT = TypeVar("StackT")


def create_high_level_gating_stack(
    stack_type: str,
    *,
    legacy_stack: StackT | None,
    tri_module_stack: StackT | None = None,
) -> StackT:
    """Select exactly one high-level gating stack without implicit fallback."""

    normalized = str(stack_type).strip().lower()
    if normalized == "legacy":
        if legacy_stack is None:
            raise ValueError("legacy stack is not configured")
        return legacy_stack
    if normalized == "tri_module":
        if tri_module_stack is None:
            raise ValueError("tri_module stack is not configured")
        return tri_module_stack
    raise ValueError(
        f"unknown high-level gating stack {stack_type!r}; "
        "expected one of ['legacy', 'tri_module']"
    )
