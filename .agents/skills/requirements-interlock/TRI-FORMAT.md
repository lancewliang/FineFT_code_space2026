# TRI.md Format

TRI stands for Technical Requirements and Implementation. It contains only what is **new and specific** to this feature. It references LANGUAGE.md, MAP.md, user-story.md, and decisions — it does not reproduce them.

Together with those documents, a TRI.md should be sufficient for an agent to implement the feature or write a TDD test suite without further context.

## Principle

> Summarise the relevant subset — do not copy the full content. The TRI must be self-contained enough to implement from, without requiring the reader to load every source document. But it must not duplicate information that belongs elsewhere.

## Template

```markdown
---
status: draft
owner: [Team/Person]
---

# [Feature Name] — TRI

## Context

- User story: [link to docs/user-stories/0001-story-slug.md]

### Key domain terms

| Term | Definition |
|------|------------|
| [Term1] | [one-line summary from LANGUAGE.md] |
| [Term2] | [one-line summary from LANGUAGE.md] |

_Full glossary: [LANGUAGE.md](../../LANGUAGE.md)_

### Components involved

| Component | Role in this feature |
|-----------|---------------------|
| [ComponentA] | [what it does in this context] |
| [ComponentB] | [what it does in this context] |

_Full system map: [MAP.md](../../MAP.md)_

### Relevant decisions

| Decision | Summary |
|----------|---------|
| [docs/decisions/0001-*.md] | [one-line summary of the decision and why it matters here] |

_New decisions created during this session are recorded in `docs/decisions/`._

---

## What is being built

[1-2 sentences scoped to this feature only. No background, no restatement of the user story.]

## Functional requirements

[Only requirements not already covered by the user story BDD scenarios.]

1. [Requirement]

## Non-Functional Requirements

- **Performance**: [specific to this feature]
- **Security**: [specific to this feature]
- **Scalability**: [specific to this feature]
- **Reliability**: [specific to this feature]

---

## Design

### Architecture

[Only the parts of the architecture that are new or changed for this feature.]

\`\`\`
[ASCII or Mermaid diagram — new/changed components only]
\`\`\`

### Data Model

[New or changed data structures only.]

\`\`\`yaml
# New fields or entities only
\`\`\`

### API Design

[New or changed endpoints only — method, path, request/response shape.]

### User Interface

[Type: TUI | GUI | API | CLI]
[New or changed UI only.]

---

## Implementation Plan

### Phase 1: [Phase Name]
- [ ] Task 1
- [ ] Task 2

### Phase 2: [Phase Name]
- [ ] Task 1
- [ ] Task 2

---

## New Dependencies

| Dependency | Purpose | Validated |
|------------|---------|-----------|
| [name] | [why needed] | yes/no |

---

## Testing Strategy

### Unit Tests
[What to test and why — scoped to new behaviour only.]

### Integration Tests
[New integration points only.]

### End-to-End Tests
[New user-facing flows only.]

---

## Security Considerations

- [New considerations specific to this feature]

## Performance Considerations

- [New considerations specific to this feature]

---

## Known risks

- [Any risk that doesn't warrant a full decision — one line each]

_Risks with significant impact or surprising mitigations belong in `docs/decisions/`._

## Open Questions

- [ ] [Question]

## References

- [docs/decisions/ entries relevant to this feature]
- [External resources]
```

## Status lifecycle

| Status | Meaning |
|---|---|
| `draft` | Created by skill, not yet reviewed |
| `in-review` | Under review with stakeholders |
| `approved` | Approved for implementation |
| `implemented` | Feature built and tested |
