# 01 — Cross-Month Feature Contract And Guards

**What to build:** Establish the Cross-Month Term Structure Feature contract before generation begins: supported pairing modes, canonical output column list, deterministic ordering, No Absolute Price Rule validation, and fail-fast behavior for missing required inputs or illegal feature definitions.

**Blocked by:** None — can start immediately.

**Status:** done

- [x] The system exposes a single contract for Cross-Month Term Structure Feature names, pairing modes, and mandatory-state feature columns.
- [x] The No Absolute Price Rule rejects raw price level columns and raw price-difference columns before outputs are written.
- [x] The contract distinguishes valid post-alignment liquidity gaps from missing whole input artifacts.
- [x] Tests cover accepted relative feature forms, rejected absolute price forms, deterministic ordering, and fail-fast missing-input behavior.
