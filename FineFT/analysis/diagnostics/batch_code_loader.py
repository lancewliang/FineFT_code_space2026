"""Batch Code Loader and AST Outline Extractor for Strategy Diagnosis.

Extracts module outlines, schema fields, class/method signatures, CLI arguments, and
keyword-targeted code blocks across multiple Python strategy files in one operation
to minimize token consumption and avoid sequential file-by-file inspection.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from dataclasses import dataclass
from pathlib import Path

STRATEGY_DIHFT_PYTHON_FILES: list[str] = [
    "FineFT/analysis/diagnostics/trading_diagnostics.py",
    "FineFT/analysis/pick_agent/DiHFT_high_level_heurstic.py",
    "FineFT/RL/DiHFT/high_level/vae_routing_final_result_macro_action.py",
    "FineFT/RL/DiHFT/high_level/vae_routing_util.py",
    "FineFT/RL/DiHFT/high_level/gating/hierarchical_gating.py",
    "FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py",
    "FineFT/common/artifacts.py",
    "FineFT/common/metric_columns.py",
    "FineFT/common/routing_params.py",
]

AUDIT_DIHFT_PYTHON_FILES: list[str] = [
    "FineFT/analysis/diagnostics/trading_diagnostics.py",
    "FineFT/analysis/diagnostics/batch_diagnostics_loader.py",
    "FineFT/RL/DiHFT/high_level/vae_routing_util.py",
    "FineFT/env/env_class/base_env.py",
    "FineFT/env/env_class/futures_util.py",
    "FineFT/common/artifacts.py",
]


def _deduplicate_paths(paths: list[str]) -> list[str]:
    return list(dict.fromkeys(paths))


CODE_PROFILES: dict[str, list[str]] = {
    "strategy": STRATEGY_DIHFT_PYTHON_FILES,
    "audit": AUDIT_DIHFT_PYTHON_FILES,
    "all": _deduplicate_paths(
        STRATEGY_DIHFT_PYTHON_FILES + AUDIT_DIHFT_PYTHON_FILES
    ),
}

# Backward-compatible import for callers that used the original constant.
DEFAULT_DIHFT_PYTHON_FILES = STRATEGY_DIHFT_PYTHON_FILES

DEFAULT_TARGET_KEYWORDS: list[str] = [
    "stop_loss",
    "persistence",
    "friction",
    "gating",
    "routing",
    "audit",
    "prediction",
    "action_value",
    "trade_event",
    "execution",
    "account_snapshot",
    "trading_day",
    "reconciliation",
    "commission",
    "realized_pnl",
    "slippage",
]


def resolve_profile_files(profile: str) -> list[str]:
    try:
        return list(CODE_PROFILES[profile])
    except KeyError as exc:
        choices = ", ".join(sorted(CODE_PROFILES))
        raise ValueError(
            f"Unknown code profile {profile!r}; choose one of: {choices}"
        ) from exc


@dataclass
class ArgOption:
    flags: list[str]
    default_val: str
    help_text: str
    line_no: int


@dataclass
class FunctionSignature:
    name: str
    signature: str
    return_type: str
    docstring: str
    line_start: int
    line_end: int


@dataclass
class ClassField:
    name: str
    annotation: str
    value: str
    line_no: int


@dataclass
class ClassOutline:
    name: str
    bases: list[str]
    docstring: str
    line_start: int
    line_end: int
    fields: list[ClassField]
    methods: list[FunctionSignature]


@dataclass
class FileCodeOutline:
    file_path: Path
    module_docstring: str
    classes: list[ClassOutline]
    functions: list[FunctionSignature]
    cli_arguments: list[ArgOption]
    total_lines: int


class OutlineAstVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.classes: list[ClassOutline] = []
        self.functions: list[FunctionSignature] = []
        self.cli_arguments: list[ArgOption] = []
        self._current_class: ClassOutline | None = None
        self._function_depth = 0

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        doc = ast.get_docstring(node)
        docstring = doc.split("\n")[0].strip() if doc else ""
        bases = [ast.unparse(b) for b in node.bases]
        end_lineno = node.end_lineno if node.end_lineno is not None else node.lineno
        class_outline = ClassOutline(
            name=node.name,
            bases=bases,
            docstring=docstring,
            line_start=node.lineno,
            line_end=end_lineno,
            fields=[],
            methods=[],
        )
        old_class = self._current_class
        self._current_class = class_outline
        self.generic_visit(node)
        self._current_class = old_class
        self.classes.append(class_outline)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record_function(node)

    def _record_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        doc = ast.get_docstring(node)
        docstring = doc.split("\n")[0].strip() if doc else ""
        args_str = ast.unparse(node.args)
        ret_str = ast.unparse(node.returns) if node.returns is not None else "None"
        end_lineno = node.end_lineno if node.end_lineno is not None else node.lineno
        sig = FunctionSignature(
            name=node.name,
            signature=f"({args_str})",
            return_type=ret_str,
            docstring=docstring,
            line_start=node.lineno,
            line_end=end_lineno,
        )
        if self._function_depth == 0:
            if self._current_class is not None:
                self._current_class.methods.append(sig)
            else:
                self.functions.append(sig)
        self._function_depth += 1
        try:
            self.generic_visit(node)
        finally:
            self._function_depth -= 1

    @staticmethod
    def _expression_text(node: ast.AST | None) -> str:
        if node is None:
            return ""
        text = ast.unparse(node)
        return text if len(text) <= 160 else f"{text[:157]}..."

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if (
            self._current_class is not None
            and self._function_depth == 0
            and isinstance(node.target, ast.Name)
        ):
            self._current_class.fields.append(
                ClassField(
                    name=node.target.id,
                    annotation=self._expression_text(node.annotation),
                    value=self._expression_text(node.value),
                    line_no=node.lineno,
                )
            )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._current_class is not None and self._function_depth == 0:
            value = self._expression_text(node.value)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._current_class.fields.append(
                        ClassField(
                            name=target.id,
                            annotation="",
                            value=value,
                            line_no=node.lineno,
                        )
                    )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func_expr = ast.unparse(node.func)
        if func_expr.endswith("add_argument"):
            flags: list[str] = []
            default_val = "None"
            help_text = ""
            for a in node.args:
                val = ast.unparse(a).strip("\"'")
                if val.startswith("-"):
                    flags.append(val)
            for kw in node.keywords:
                if kw.arg == "default":
                    default_val = ast.unparse(kw.value)
                elif kw.arg == "help":
                    help_text = ast.unparse(kw.value).strip("\"'")
            if flags:
                self.cli_arguments.append(
                    ArgOption(
                        flags=flags,
                        default_val=default_val,
                        help_text=help_text,
                        line_no=node.lineno,
                    )
                )
        self.generic_visit(node)


class BatchCodeLoader:
    def __init__(self, root_dir: Path, file_paths: list[str]) -> None:
        self.root_dir = root_dir
        self.file_paths = [Path(p) for p in file_paths]

    def _read_source(self, target_path: Path) -> tuple[Path, str]:
        resolved = target_path if target_path.is_absolute() else self.root_dir / target_path
        if not resolved.is_file():
            raise FileNotFoundError(f"Python target file not found: {resolved}")
        content = resolved.read_text(encoding="utf-8")
        return resolved, content

    def extract_outline(self, target_path: Path) -> FileCodeOutline:
        resolved, source = self._read_source(target_path)
        tree = ast.parse(source, filename=str(resolved))
        doc = ast.get_docstring(tree)
        module_doc = doc.split("\n")[0].strip() if doc else ""
        total_lines = len(source.splitlines())

        visitor = OutlineAstVisitor()
        visitor.visit(tree)

        rel_path = (
            resolved.relative_to(self.root_dir)
            if resolved.is_relative_to(self.root_dir)
            else resolved
        )
        return FileCodeOutline(
            file_path=rel_path,
            module_docstring=module_doc,
            classes=visitor.classes,
            functions=visitor.functions,
            cli_arguments=visitor.cli_arguments,
            total_lines=total_lines,
        )

    def extract_targeted(
        self, target_path: Path, keywords: list[str]
    ) -> list[tuple[int, int, str, str]]:
        resolved, source = self._read_source(target_path)
        tree = ast.parse(source, filename=str(resolved))
        source_lines = source.splitlines()
        kw_lower = [k.lower() for k in keywords]
        if not kw_lower:
            return []

        parent_by_node: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parent_by_node[child] = parent

        candidates = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        matched_nodes: list[ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef] = []
        code_by_node: dict[ast.AST, str] = {}
        for node in candidates:
            end_lineno = node.end_lineno if node.end_lineno is not None else node.lineno
            block_code = "\n".join(source_lines[node.lineno - 1 : end_lineno])
            if any(keyword in block_code.lower() for keyword in kw_lower):
                matched_nodes.append(node)
                code_by_node[node] = block_code

        # Prefer the smallest matching semantic block. If a method matches, returning
        # its containing multi-thousand-line class would defeat targeted extraction.
        leaf_matches = [
            node
            for node in matched_nodes
            if not any(
                other is not node
                and node.lineno <= other.lineno
                and (other.end_lineno or other.lineno) <= (node.end_lineno or node.lineno)
                for other in matched_nodes
            )
        ]

        def qualified_label(
            node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        ) -> str:
            names = [node.name]
            parent = parent_by_node.get(node)
            while parent is not None:
                if isinstance(parent, ast.ClassDef):
                    names.append(parent.name)
                parent = parent_by_node.get(parent)
            qualified_name = ".".join(reversed(names))
            if isinstance(node, ast.ClassDef):
                return f"class {qualified_name}"
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            return f"{prefix} {qualified_name}"

        return [
            (
                node.lineno,
                node.end_lineno if node.end_lineno is not None else node.lineno,
                qualified_label(node),
                code_by_node[node],
            )
            for node in sorted(leaf_matches, key=lambda item: item.lineno)
        ]

    def format_outline_report(self) -> str:
        lines: list[str] = [
            "# DiHFT 策略核心 Python 代码架构与接口全景大纲 (Batch Code Outline)",
            "",
            "> 自动聚合展示类签名、方法接口、函数原型与 CLI 传参，避免逐个读取源码消耗上下文。",
            "",
        ]

        for p in self.file_paths:
            outline = self.extract_outline(p)
            lines.append(f"## `{outline.file_path}` ({outline.total_lines} 行)")
            if outline.module_docstring:
                lines.append(f"- **模块说明**：{outline.module_docstring}")

            if outline.cli_arguments:
                lines.append("- **可调参数 (CLI Arguments)**：")
                for arg in outline.cli_arguments:
                    flag_str = "/".join(arg.flags)
                    lines.append(
                        f"  - `{flag_str}` (默认: `{arg.default_val}`) L{arg.line_no}: {arg.help_text}"
                    )

            if outline.classes:
                lines.append("- **核心类 (Classes)**：")
                for c in outline.classes:
                    base_str = f"({', '.join(c.bases)})" if c.bases else ""
                    lines.append(
                        f"  - `class {c.name}{base_str}` (L{c.line_start}~L{c.line_end}) {c.docstring}"
                    )
                    for field in c.fields:
                        annotation = f": {field.annotation}" if field.annotation else ""
                        value = f" = {field.value}" if field.value else ""
                        lines.append(
                            f"    - `{field.name}{annotation}{value}` (L{field.line_no})"
                        )
                    for m in c.methods:
                        lines.append(
                            f"    - `def {m.name}{m.signature} -> {m.return_type}` (L{m.line_start})"
                        )

            if outline.functions:
                lines.append("- **顶级函数 (Functions)**：")
                for f in outline.functions:
                    lines.append(
                        f"  - `def {f.name}{f.signature} -> {f.return_type}` (L{f.line_start}~L{f.line_end}) {f.docstring}"
                    )
            lines.append("")

        return "\n".join(lines)

    def format_targeted_report(self, keywords: list[str]) -> str:
        lines: list[str] = [
            f"# DiHFT 策略代码定向检索汇总 (Keywords: {', '.join(keywords)})",
            "",
        ]

        for p in self.file_paths:
            blocks = self.extract_targeted(p, keywords)
            if not blocks:
                continue
            lines.append(f"## 文件：`{p}` (匹配 {len(blocks)} 个代码块)")
            for start, end, header, code in blocks:
                lines.append(f"### {header} (Lines {start}~{end})")
                lines.append("```python")
                lines.append(code)
                lines.append("```")
                lines.append("")

        return "\n".join(lines)

    def format_full_bundle(self) -> str:
        lines: list[str] = [
            "# DiHFT 策略核心 Python 代码全量打包 (Batch Full Code Bundle)",
            "",
        ]

        for p in self.file_paths:
            resolved, source = self._read_source(p)
            rel_path = (
                resolved.relative_to(self.root_dir)
                if resolved.is_relative_to(self.root_dir)
                else resolved
            )
            lines.append(f"## File: `{rel_path}`")
            lines.append("```python")
            for idx, line in enumerate(source.splitlines(), start=1):
                lines.append(f"{idx:4d} | {line}")
            lines.append("```")
            lines.append("")

        return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch load and inspect multiple Python strategy files."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=".",
        help="Workspace root directory (default: current directory).",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(CODE_PROFILES),
        default="strategy",
        help="Built-in code inventory: strategy, audit, or their deduplicated union (all).",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Explicit Python files to inspect. When supplied, overrides --profile.",
    )
    parser.add_argument(
        "--mode",
        choices=["outline", "targeted", "full"],
        default="outline",
        help="Inspection mode: outline (AST classes/methods/args), targeted (keyword search), full (concatenated bundle).",
    )
    parser.add_argument(
        "--keywords",
        nargs="*",
        default=DEFAULT_TARGET_KEYWORDS,
        help="Keywords to search when mode is 'targeted'.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output file path. Defaults to stdout.",
    )
    args = parser.parse_args()

    file_paths = args.files if args.files else resolve_profile_files(args.profile)
    loader = BatchCodeLoader(root_dir=Path(args.root_dir), file_paths=file_paths)

    if args.mode == "outline":
        report = loader.format_outline_report()
    elif args.mode == "targeted":
        report = loader.format_targeted_report(args.keywords)
    else:
        report = loader.format_full_bundle()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"Saved batch code report to {out_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
