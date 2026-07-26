# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Python Code Style

**Readable, typed, and boring Python. Follow existing project style first.**

Formatting and imports:
- Prefer `black`-compatible formatting with an 88 character line width when formatting Python.
- Keep imports grouped as standard library, third-party packages, then local project imports.
- Do not reformat unrelated code or expand diffs just to satisfy style preferences.

Naming:
- Use `snake_case` for functions, variables, modules, and attributes.
- Use `PascalCase` for classes and `UPPER_SNAKE_CASE` for constants.
- Avoid vague names such as `foo`, `data1`, or `tmp` unless the scope is tiny and obvious.

Types and interfaces:
- Add type hints for new or changed functions when practical.
- Public functions, cross-module interfaces, and non-trivial data structures should have explicit types.
- Avoid broad `Any`; if it is necessary, keep it local and explain why through the surrounding code.
- Prefer built-in generics such as `list[str]` and `dict[str, int]` when the supported Python version allows it.

Functions and structure:
- Keep functions focused on one job.
- Avoid hidden side effects; make I/O, mutation, and randomness visible at the call site when possible.
- Do not create abstractions for one-off logic.
- If parameters become hard to understand, prefer a small dataclass, config object, or clearer split.

Errors and logging:
- Catch specific exceptions, not bare `except:`.
- Do not swallow exceptions unless there is a clear fallback or recovery path.
- Include useful context in error messages.
- Use `logging` for library code; reserve `print` for CLI or script user output.

Files and data:
- Prefer `pathlib.Path` for path manipulation.
- Specify file encodings explicitly, usually `encoding="utf-8"`.
- Use context managers for resources.
- Prefer structured types such as `dataclass`, `TypedDict`, `Enum`, or `NamedTuple` when they clarify shared data.

Tests:
- For bug fixes, write or update a test that reproduces the failure before fixing it when feasible.
- For new behavior, cover the main path and important edge cases.
- Keep tests deterministic; avoid real network, wall-clock time, and order dependencies unless explicitly required.
- Name tests by behavior, for example `test_rejects_empty_symbol`.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.


use conda activate finetf run python script

## Agent skills

### Issue tracker

GitHub Issues for this repo. See `docs/agents/issue-tracker.md`.

### Triage labels

Default label vocabulary (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout. See `docs/agents/domain.md`.
