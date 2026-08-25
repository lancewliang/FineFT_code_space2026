# ADR-0011: Volatility Labeling and Method-isolated Outputs

Date: 2026-08-25

Status: Accepted

## Decision

The valid-directory production build supports `labeling_method="slope"` and
`labeling_method="volatility"`. Both methods retain the existing independent
per-contract Butterworth, turning-point, and slice-and-merge process. Only the
score used to assign the final Label changes.

Volatility score is the non-annualized population standard deviation of the
log returns inside one final segment, expressed in percent:

```text
100 * std(diff(log(bid1_price)), ddof=0)
```

Volatility thresholds default to pooled global Segment Quantiles at
`i / dynamic_number`, with one vote per final segment. This score is invariant
under multiplicative price-unit changes. A one-row segment has volatility zero;
non-positive prices fail the volatility build.

Published artifacts are isolated by method:

```text
valid/slope/...
valid/volatility/...
```

Each method directory owns its processed files, contract Label directories,
and Slice Manifest. Rebuilding one method atomically replaces only that
method's generation and leaves the other method unchanged. The VAE data
creation entry point can select the method directory and defaults to slope,
with a fallback for legacy datasets whose slope outputs live directly under
`valid/`.

Final `quantile` and `DTW` labeling remain unsupported. This ADR supersedes
ADR-0009 and ADR-0010 only where they state that production final labeling is
slope-only or that official outputs live directly under `valid/`.
