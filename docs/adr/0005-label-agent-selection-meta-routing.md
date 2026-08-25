---
status: superseded by ADR-0008
---

# Label Selection Agent Meta-Routing with Archetype Profiles and Semantic Guards

To prevent VAE log-likelihood gating from routing to counter-trend agents (e.g., executing 247 reverse short positions in strong bull markets), FineFT upgrades agent routing from pure likelihood matching to a four-step pipeline: (1) Gating Label Lock, (2) Archetype Profile Pool Retrieval across 12 strategy archetypes, (3) Candidate Generator with Semantic Guard (rejecting actions violating native label direction semantics) and PnL 20% drawdown limit, and (4) Meta Router multi-factor scoring (0.5 VAE + 0.5 PnL memory) with a 15% single-contract circuit breaker falling back to rule-based position closing (`macro_action = 5`).
