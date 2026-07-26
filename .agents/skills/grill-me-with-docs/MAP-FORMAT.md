# MAP.md Format

MAP.md lives at the repo root — always one file, always system-wide. It is sparse documentation at the *seam level*: things that have an interface or boundary another developer needs to know about to avoid duplicating or misusing them. Implementation details belong in the code.

## Instructions preamble

Every MAP.md must open with this preamble:

```md
> Before creating a new component, module, UI, or API surface: check this file and the
> codebase for an existing equivalent. If one exists, use it. If you still intend to
> create something new, confirm with the human what it does that the existing one does not.
>
> LANGUAGE.md owns what things *mean*. This file owns where things *live* and how they
> *connect*. A term can appear in both.
>
> Keep entries sparse — document at the seam level only. Implementation details belong
> in the code.
```

## Template

```md
> [preamble above]

# System Map

## Components

**{Component Name}**: {One sentence — what it owns and its interface boundary.}

**{Component Name}**: {One sentence.}

## User Interfaces

**{UI Name}** (`{type: screen | API | CLI | webhook}`): {One sentence — what it exposes and to whom.}

## Related Systems

**{System Name}** (`{repo or service identifier}`): {One sentence — what it does and how it connects to this system.}
```

## Rules

- **Seam level only.** A component entry describes the boundary, not the internals. "Billing module — owns invoice generation and payment state; exposes a `BillingService` interface consumed by Ordering" is right. A list of its internal classes is wrong.
- **One sentence per entry.** If you need more, the entry is too detailed.
- **Related Systems must include the relationship.** Don't just name the system — say how it connects. "Notifications service (`notifications-repo`) — sends transactional emails; triggered by domain events published from this system."
- **Omit obvious things.** If a component's existence and purpose are self-evident from the repo structure, skip it.
- **No implementation details.** No function signatures, no database schemas, no internal patterns.

## When to update

- A new component, UI, or related system is identified or confirmed during the session
- A validation check reveals an existing entry is wrong or outdated
- End of session — review for anything discussed but not captured
