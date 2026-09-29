# 06: Market State Regime Audit Anchor Retention Cross-Regime Variance Bounding

**What to build:** Upgrade `regime_audit.py` conditional anchor retention to enforce cross-regime variance stability ($\sigma^2_{\text{extreme}} / \sigma^2_{\text{neutral}} \le 3.0$) alongside the existing 90% LCB criterion, ensuring conditionally retained market state anchors do not introduce variance explosions into the downstream state space.

**Blocked by:** 05: Contract-Normalized Decentralized Correlation Matrix and OOD-Aware Priority Scoring

**Status:** closed

- [x] In `regime_audit.py:audit_regimes`, calculate feature variance across extreme market regime bins versus neutral bins.
- [x] Disallow conditional anchor retention if $\sigma^2_{\text{extreme}} / \sigma^2_{\text{neutral}} > 3.0$.
- [x] Update `retention_details` in `feature_selection_manifest.json` to record the cross-regime variance ratio for all evaluated anchors.
- [x] Automated tests in `test_regime_audit_and_anchor_retention.py` verifying that anchors with exploding variance in extreme regimes are rejected while stable anchors are retained.
