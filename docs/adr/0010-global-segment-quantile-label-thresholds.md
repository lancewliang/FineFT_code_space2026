---
status: accepted
---

# Use Global Segment Quantiles For Final Dynamic Label Thresholds

FineFT fits final Semantic-free Dynamic Label thresholds at `i / dynamic_number` quantiles of the pooled signed percentage slopes from all final Market Dynamic Segments in the valid set, then applies that one shared threshold set to every contract. Each final segment has weight one; segment length and contract identity do not change calibration weight, so this is not row-weighted or contract-local calibration. Butterworth filtering, turning-point detection, slice-and-merge, the Contract-local Merge Label, percentage-slope scale, atomic publication, Contract-empty Label handling, and downstream non-semantic Label rules from ADR-0009 remain unchanged; `labeling_method="quantile"` and final DTW labeling remain unsupported because the final score is still slope and only its shared threshold method changes. The 30-minute fu pipeline uses four Labels, while other dataset pipelines must opt in explicitly rather than inheriting this cardinality change.
