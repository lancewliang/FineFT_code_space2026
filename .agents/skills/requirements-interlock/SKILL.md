---
name: requirements-interlock
description: Takes user stories, LANGUAGE.md, MAP.md, and decisions as inputs; uses grill-me to produce a concise Technical Requirements and Implementation document (TRI.md) that references but does not duplicate existing context. TRI.md contains only what is new and specific to this feature — sufficient for an agent to implement or write TDD tests without further context. Use when user wants to produce a TRI.md or prepare a feature for agent-driven implementation or TDD.
---

# Design Part 2

## Quick start

Provide one or more user stories and the skill will grill you on the design, then produce a TRI.md alongside the user story:

```
docs/user-stories/0001-password-reset-TRI.md
```

## Workflows

### Inputs

- **User story or set of user stories** — from `docs/user-stories/`
- **`LANGUAGE.md`** — domain glossary (if exists)
- **`MAP.md`** — system map (if exists)
- **`docs/decisions/`** — existing decisions (if exist)

### Process

1. **Read all inputs** — load the user story, LANGUAGE.md, MAP.md, and decisions; extract the subset relevant to this feature; summarise (do not copy in full) key decisions, domain terms, and MAP components in the TRI.md Context section so the TRI is self-contained enough to implement from without requiring the reader to load every source document
2. **Grill on the TRI** — use `grill-me` to interrogate only what is new and specific to this feature: architecture decisions, data model changes, API design, UI, NFRs, testing strategy, security, and performance
3. **Resilience phase** — run after the main interview; see below
4. **Update decisions** — record any new findings that meet the decision criteria in `docs/decisions/`
5. **Write TRI.md** — only after the human confirms the draft

### File location

Co-located with the user story, suffixed with `-TRI`:

- Single context: `docs/user-stories/0001-[story-slug]-TRI.md`
- Multi-component: `docs/user-stories/[component-slug]/0001-[story-slug]-TRI.md`

### Resilience phase

After the main interview, run this check. Use judgment — skip questions where the session content makes them clearly irrelevant. Any finding that meets the decision criteria records as a `docs/decisions/` entry.

This phase runs at **design time**, before any dependency is adopted or any component is built. The goal is to catch structural risks early, when they are cheapest to resolve.

- **Failure points** — for each significant component or flow discussed: what happens when it fails?
- **Duplication** — cross-reference `MAP.md`: does this functionality already exist in this system or a related one?
- **Dependency failure** — for each dependency or function: what happens if it fails? Is a failover or fallback needed?
- **API failure** — for each external API call: what happens on failure? Is a circuit breaker pattern needed?
- **Supply chain** — for each proposed open-source dependency: challenge necessity first ("is this really needed?"), then if confirmed validate trustworthiness (provenance, maintenance status, licence, known vulnerabilities). The `/review` skill re-checks any dependencies that were added during TDD.

## Used skills

| Skill | If not available |
|---|---|
| `grill-me` | Interrogate each TRI section directly: ask one question at a time, provide a recommended answer for each, and explore the codebase when a question can be answered by it |

## Advanced features

See [TRI-FORMAT.md](./TRI-FORMAT.md) for the full TRI.md template.

The intention is that `LANGUAGE.md` + `MAP.md` + `TRI.md` + `user-story.md` + `RULES.md` are sufficient for an agent to produce a full implementation or a TDD test suite without further context.
