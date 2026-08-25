---
status: accepted
---

# Dynamic Labels Are Semantic-Free Identifiers

FineFT treats Valid Dynamic Slice labels as opaque identifiers with no direction, magnitude, or price-limit meaning; the same label may contain market dynamic segments with opposite directions, including across contracts. Limit-price and near-limit-price rows remain in ordinary slicing and share the existing `dynamic_number` labels rather than being filtered or assigned dedicated labels. Agent selection, routing, and action constraints must not infer trading direction from a label number and must instead use observed market, performance, and risk state. This supersedes ADR-0005 because semantic guards make an arbitrary label identifier part of the trading-control contract and prevent scale-aware multi-contract labeling from changing freely.
