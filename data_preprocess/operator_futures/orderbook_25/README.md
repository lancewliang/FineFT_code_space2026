# Down Scale Orderbook Snapshot
First down scale the orderbook to a proper frequency, then concat and clean it.

During down scale, we should always use the first orderbook snapshot within the group so that and mark the orderbook for each timestamp as the latest timestamp mark.

## Down Scale Multiple Snapshots with one process

## Down Scale one Snapshots with one process