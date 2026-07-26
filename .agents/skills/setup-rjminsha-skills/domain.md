# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`LANGUAGE.md`** at the repo root, or
- **`CONTEXT-MAP.md`** at the repo root if it exists — it points at one `LANGUAGE.md` per context. Read each one relevant to the topic.
- **`MAP.md`** at the repo root — sparse system map of components, UIs, and related systems.
- **`docs/decisions/`** — read decisions that touch the area you're about to work in. In multi-context repos, also check `src/<context>/docs/decisions/` for context-scoped decisions.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The producer skill (`/grill-me-with-docs`) creates them lazily when terms or decisions actually get resolved.

## File structure

Single-context repo (most repos):

```
/
├── LANGUAGE.md
├── MAP.md
├── docs/decisions/
│   ├── 0001-event-sourced-orders.md
│   └── 0002-postgres-for-write-model.md
└── src/
```

Multi-context repo (presence of `CONTEXT-MAP.md` at the root):

```
/
├── CONTEXT-MAP.md
├── MAP.md                             ← always at root, system-wide
├── docs/decisions/                    ← system-wide decisions
└── src/
    ├── ordering/
    │   ├── LANGUAGE.md
    │   └── docs/decisions/            ← context-specific decisions
    └── billing/
        ├── LANGUAGE.md
        └── docs/decisions/
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `LANGUAGE.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/grill-me-with-docs`).

## Flag decision conflicts

If your output contradicts an existing decision, surface it explicitly rather than silently overriding:

> _Contradicts docs/decisions/0007-event-sourced-orders.md — but worth reopening because…_
