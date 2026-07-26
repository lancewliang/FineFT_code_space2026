---
name: grill-me-with-docs
description: Grilling session that challenges a plan against the existing domain model, sharpens terminology, and updates documentation (LANGUAGE.md, MAP.md, decisions) inline as decisions crystallise. Use when user wants to stress-test a plan against their project's language, system map, and documented decisions.
---

<what-to-do>

Use the `grill-me` skill to conduct the interview session, focused on domain terminology, system boundaries, and documentation decisions. If `grill-me` is not available, conduct the interview directly using the fallback instructions in [REFERENCE.md](./REFERENCE.md).

</what-to-do>

## Used skills

| Skill | If not available |
|---|---|
| `grill-me` | Interview the human directly: ask one relentless question at a time, walk each branch of the decision tree, provide a recommended answer for each, and explore the codebase when a question can be answered by it |

<supporting-info>

## Domain awareness

This skill builds and maintains three living documents:

| File | Purpose |
|---|---|
| `LANGUAGE.md` | Canonical domain terms — what things *mean*. Glossary only, no implementation details. |
| `MAP.md` | Sparse system map — where things *live* and how they *connect*. Gives agents and developers with limited context a reference to the whole system. |
| `docs/decisions/` | Hard-to-reverse, surprising, or genuine trade-off decisions. |

### File structure

```
/
├── LANGUAGE.md          ← domain glossary
├── MAP.md               ← system map
├── docs/
│   └── decisions/
│       └── 0001-*.md
└── src/
```

Create files lazily — only when you have something to write. See [REFERENCE.md](./REFERENCE.md) for multi-context structure.

## During the session

Challenge terms against `LANGUAGE.md`, sharpen fuzzy language, discuss concrete scenarios, and cross-reference claims with the code. See [REFERENCE.md](./REFERENCE.md) for detailed session behaviours.

### Update LANGUAGE.md inline

When a term is resolved, update `LANGUAGE.md` right there. Don't batch these up — capture them as they happen. Use the format in [LANGUAGE-FORMAT.md](./LANGUAGE-FORMAT.md).

`LANGUAGE.md` is a glossary and nothing else — no implementation details, no decisions.

### Validate new interfaces

Before any new component, module, UI, API surface, or CLI command is created:

1. Check `MAP.md` for an existing equivalent
2. Check the codebase for an existing equivalent
3. If one exists: "This looks like it already exists as X — is creating a new one intentional?"
4. If confirmed, ask what it does that the existing one does not, then update `MAP.md`

Use the format in [MAP-FORMAT.md](./MAP-FORMAT.md).

### Update MAP.md inline

Update `MAP.md` immediately when any of the following happen:

- A new component, UI, or related system is identified or confirmed
- A validation check reveals an entry in `MAP.md` needs correcting
- End of session — review `MAP.md` for anything discussed but not yet captured

### Offer decisions sparingly

Only offer a decision when it is hard to reverse, surprising without context, and the result of a real trade-off. See [DECISIONS.md](./DECISIONS.md) for the full criteria and format.

## At the end of the session

Print the resulting file structure showing every file created or updated:

```
/
├── LANGUAGE.md                               ← updated
├── MAP.md                                    ← updated
├── docs/
│   └── decisions/
│       └── 0001-example-decision.md          ← created
└── src/
```

</supporting-info>
