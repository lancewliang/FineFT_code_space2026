---
name: user-story
description: Create a user story document with BDD scenarios from a git issue, feature description, or existing code. Use when user wants to write, document, or reverse-engineer a user story, acceptance criteria, or BDD scenarios for a feature.
---

# User Story

## Quick start

```markdown
---
status: draft
source: #42
---

# Password reset

## User Story

As a registered user
I want to reset my password via email
So that I can regain access without contacting support

## Scenarios

### Successful reset

Given:
- the user has a verified email address on file

When:
- they request a password reset

Then:
- a reset link is sent to their email
```

## Workflows

### Inputs

Accept any of:

- **Git issue** — read the issue title, body, and labels
- **Feature description** — plain text from the user
- **Code** — reverse-engineer from existing code when explicitly requested

### Process

1. **Identify actors** — check `LANGUAGE.md` first, then context clues (auth roles, route guards, UI entry points), then confirm with the human
2. **Handle multiple stories** — if the input describes multiple related actors or features, default to a single file; ask the human if they want separate files
3. **Draft the user story** — one sentence per story: *As a [ACTOR] / I want to [action] / So that [benefit]*
4. **Review** — if `grill-me` is available use it to stress-test actor, intent, and benefit; otherwise ask: "Does this capture the right actor, action, and benefit?"
5. **Break into BDD scenarios** — one per distinct who/what/when; soft limit 5, hard limit 10
6. **Add example tables** where multiple inputs produce distinct business outcomes
7. **Check `MAP.md`** — if a new UI or component is implied, flag it and update `MAP.md`
8. **Check `LANGUAGE.md`** — if a new actor or concept surfaces, flag it and update `LANGUAGE.md`
9. **Write the file** — only after the human confirms the draft

### File location

Scan the target directory for the highest existing number and increment by one — same pattern as `docs/decisions/`.

- Single context: `docs/user-stories/0001-[story-slug].md`
- Multi-component (`MAP.md` describes multiple components): `docs/user-stories/[component-slug]/0001-[story-slug].md`

If `MAP.md` exists and describes multiple components, ask which component this story belongs to before writing the file.

## Used skills

| Skill | If not available |
|---|---|
| `grill-me` | Ask the human directly to validate the story's actor, intent, and benefit before proceeding |

## Advanced features

See [REFERENCE.md](./REFERENCE.md) for the full document template, BDD guidance, status lifecycle, and background.
