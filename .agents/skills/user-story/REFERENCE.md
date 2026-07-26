# User Story — Reference

## Document template

~~~markdown
---
status: draft
source: [#issue-number | path/to/file | omit if plain description]
---

# [Short feature name]

## User Story

As a [ACTOR]
I want to [action or feature]
So that [benefit or value]

## Scenarios

### [Scenario name]

Given:
- [precondition]

When:
- [event]

Then:
- [outcome]

### [Scenario name with examples]

Given:
- [precondition]

When:
- [event]

Then outcomes vary — see examples:

| [variable]  | [variable] | expected outcome |
|-------------|------------|------------------|
| example 1   | ...        | ...              |
| example 2   | ...        | ...              |
~~~

## BDD guidance

- **Purpose is to flush out who/what/when** — not exhaustive test coverage; that belongs in the test suite
- **Soft limit: 5 scenarios** — flag if exceeded and ask whether the story should be split
- **Hard limit: 10 scenarios** — a story with 10+ scenarios is too broad; split it
- **One outcome per `Then`** — multiple outcomes signal the scenario should be split
- **Tables only for distinct business behaviour** — each row is a standalone example readable by a non-technical stakeholder
- **Focus on what, not how** — user intent, not implementation steps
- **Actors are roles, not systems** — "Guest user", "Admin", "Subscribed customer"

## Status lifecycle

| Status | Meaning |
|---|---|
| `draft` | Created by skill, not yet agreed |
| `accepted` | Human has confirmed the story and scenarios |
| `completed` | Implemented and tested |

## Background

The BDD practice in this skill is inspired by content created by Dave Farley — [Modern Software Engineering](https://www.youtube.com/@ModernSoftwareEngineeringYT).
