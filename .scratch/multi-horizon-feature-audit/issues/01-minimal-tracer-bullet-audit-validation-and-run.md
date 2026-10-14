# 01: Minimal Tracer Bullet: Audit Validation & End-to-End Single-Horizon Run

**What to build:** An independent, end-to-end multi-horizon feature audit application slice. Researchers can invoke public CLI operations to validate input manifests and execute a baseline audit run. The application consumes causally verified numerical feature candidates from Stage 6 data without mutating or invoking legacy selection pipelines, aligns observations against active trading time and session schedules for hourly (60 active minutes) and Today target horizons, acquires as-of reference prices from verified same-contract snapshot quotes within freshness limits (maximum 60 seconds, no midprice fallback), calculates unbiased Spearman rank correlations with average tie handling, and publishes a versioned run manifest, row-level labels, feature metrics table, and completion marker.

**Blocked by:** None (can start immediately)

**Status:** completed

- [x] Public CLI provides `validate-inputs` and `run` operations accepting configuration and input manifests without modifying existing selector configs, outputs, or checkpoints.
- [x] Validation blocks execution and reports clear diagnostics when required contracts (calendar, price source, candidate availability, or split manifests) are missing, malformed, or ambiguous.
- [x] Candidate qualification admits all causally available numerical features from Stage 6 inputs, explicitly filtering out identifiers, execution/reward fields, future labels, and non-numeric columns.
- [x] Active trading time calculator accumulates 60 minutes of active trading time across sessions and days while skipping closures and breaks, and resolves Today to the current exchange TradingDay day-session close.
- [x] Reference prices are retrieved strictly as-of the intended target time from valid snapshots of the same contract and session, enforcing maximum 60-second quote age with no future lookahead and no midprice fallback.
- [x] Spearman correlation is computed using average ranks for ties on valid feature-target observation pairs, producing null with explicit reason codes for constant features or insufficient sample pairs.
- [x] Run execution emits an immutable run manifest recording schema version, configuration fingerprint, input fingerprints, random seed, and completion marker `_SUCCESS`.
- [x] Automated end-to-end integration tests pass using fixture data and assert that legacy Stage 7/8 directories, triple-stream blacklists, and shared summary files remain untouched.
