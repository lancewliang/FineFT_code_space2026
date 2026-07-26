# Grill With Docs — Reference

## Fallback interview instructions

Use these when `grill-me` is not available:

Interview the human relentlessly about every aspect of the plan until reaching a shared understanding. Walk down each branch of the design tree, resolving dependencies between decisions one by one. For each question, provide a recommended answer.

Ask one question at a time, waiting for feedback before continuing. If a question can be answered by exploring the codebase, explore it instead.

## Detailed session behaviours

### Challenge against the glossary

When the user uses a term that conflicts with the existing language in `LANGUAGE.md`, call it out immediately. "Your glossary defines 'cancellation' as X, but you seem to mean Y — which is it?"

### Sharpen fuzzy language

When the user uses vague or overloaded terms, propose a precise canonical term. "You're saying 'account' — do you mean the Customer or the User? Those are different things."

### Discuss concrete scenarios

When domain relationships are being discussed, stress-test them with specific scenarios. Invent scenarios that probe edge cases and force the user to be precise about the boundaries between concepts.

### Cross-reference with code

When the user states how something works, check whether the code agrees. If you find a contradiction, surface it: "Your code cancels entire Orders, but you just said partial cancellation is possible — which is right?"

## Multi-context file structure

If a `CONTEXT-MAP.md` exists at the root, the repo has multiple contexts:

```
/
├── CONTEXT-MAP.md
├── MAP.md                            ← always at root, system-wide
├── docs/
│   └── decisions/                    ← system-wide decisions
├── src/
│   ├── ordering/
│   │   ├── LANGUAGE.md
│   │   └── docs/decisions/           ← context-specific decisions
│   └── billing/
│       ├── LANGUAGE.md
│       └── docs/decisions/
```

When multiple contexts exist, infer which one the current topic relates to. If unclear, ask.

## Background

The `LANGUAGE.md` practice is rooted in Eric Evans' *Domain-Driven Design* — specifically the concept of a Ubiquitous Language: a common, rigorous language shared between developers and domain experts, used consistently in code, conversation, and documentation. This session is a structured way to build and maintain that language.
