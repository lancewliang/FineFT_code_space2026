# Design: enhance-limit-single-sided-features

## Context

Commodity futures can legally trade with a single-sided order book near price limits. The current downscale path already allows those rows, but snapshot feature generation divides by the empty side's total size. This produces NaN values before time feature generation.

The change also adds `LowerLimitPrice` and `UpperLimitPrice` to commodity reward/execution columns so downstream consumers can inspect the current price-limit boundary.

## Decisions

- Keep single-sided market states valid when exactly one side has zero total size.
- Treat an empty ask side as `ask_side_empty = true` and an empty bid side as `bid_side_empty = true`.
- Use the existing same-side placeholder price for empty-side WAPs: `sell_wap = ask1_price` when ask is empty, and `buy_wap = bid1_price` when bid is empty.
- Set normalized size features for the empty side to `0` rather than fabricating volume.
- Add `ask_side_empty` and `bid_side_empty` as snapshot/state candidate features, not reward/execution columns.
- Add `LowerLimitPrice` and `UpperLimitPrice` to commodity reward/execution columns after depth-aware orderbook columns and before derivative reference columns.
- Preserve fail-fast validation for both-side-empty books and missing required prices.

## Affected Flow

1. Commodity downscale outputs depth-aware orderbook columns plus price-limit columns.
2. Merge keeps those columns in `CONCURRENT_FEATURE`, so they are current-timestamp reward/execution fields and are not shifted by the future-feature branch.
3. Snapshot feature generation emits finite single-sided features and side-empty flags.
4. Feature selection and scale save use the updated commodity reward/execution manifest.
5. Existing illegal-value validators continue to reject NaN/Inf at time feature, feature selection, and scale save boundaries.

## Risks

- Adding reward/execution columns changes the commodity data contract. Tests must update expected column counts and ordering.
- Adding snapshot feature columns changes expected snapshot column counts. Documentation and expected-columns helpers must be updated together.
- Time feature behavior for zero normalized size columns may produce non-finite intermediate log returns; existing time feature cleanup should be covered by focused tests.
