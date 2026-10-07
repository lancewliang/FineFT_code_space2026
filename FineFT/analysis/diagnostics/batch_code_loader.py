"""Batch Code Loader and AST Outline Extractor for Strategy Diagnosis.

Extracts module outlines, class/method signatures, CLI arguments, and keyword-targeted
code blocks across multiple Python strategy files in one operation to minimize token
consumption and avoid sequential file-by-file inspection.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DIHFT_PYTHON_FILES: list[str] = [
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
class ClassOutline:
    name: str
    bases: list[str]
    docstring: str
    line_start: int
    line_end: int
    methods: list[FunctionSignature]


@dataclass
class FileCodeOutline:
    file_path: Path
    module_docstring: str
    classes: list[ClassOutline]
    functions: list[FunctionSignature]
    cli_arguments: list[ArgOption]
    total_lines: int


class BlockHeaderVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.label: str = "Block"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.label = f"def {node.name}"

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.label = f"async def {node.name}"

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.label = f"class {node.name}"


class OutlineAstVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.classes: list[ClassOutline] = []
        self.functions: list[FunctionSignature] = []
        self.cli_arguments: list[ArgOption] = []
        self._current_class: ClassOutline | None = None

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
        if self._current_class is not None:
            self._current_class.methods.append(sig)
        else:
            self.functions.append(sig)
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
        matched_blocks: list[tuple[int, int, str, str]] = []

        for node in tree.body:
            end_lineno = node.end_lineno if node.end_lineno is not None else node.lineno
            block_code = "\n".join(source_lines[node.lineno - 1 : end_lineno])
            block_lower = block_code.lower()
            if any(k in block_lower for k in kw_lower):
                header_visitor = BlockHeaderVisitor()
                header_visitor.visit(node)
                matched_blocks.append((node.lineno, end_lineno, header_visitor.label, block_code))

        return matched_blocks

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
        "--files",
        nargs="*",
        default=DEFAULT_DIHFT_PYTHON_FILES,
        help="List of Python files to inspect. If omitted, uses default DiHFT strategy files.",
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
        default=["stop_loss", "persistence", "friction", "gating", "routing"],
        help="Keywords to search when mode is 'targeted'.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output file path. Defaults to stdout.",
    )
    args = parser.parse_args()

    loader = BatchCodeLoader(root_dir=Path(args.root_dir), file_paths=args.files)

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
