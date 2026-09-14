# ADR-0014: Phase-Cyclic Parameter Decay in Diverse Training Curriculum

Date: 2026-09-12

Status: Accepted

## Context

Under ADR-0012, Stage I diverse training adopted a 3-phase rotating curriculum (Phase 0: Downtrend [0, 3, 6], Phase 1: Flat [1, 4, 7], Phase 2: Uptrend [2, 5, 8]), running in blocks of $x$ epochs (configured via `curriculum_block_epochs`, default 3). However, the exploration rate $\\epsilon$ and distillation regularization weight $\\alpha$ (`ada`) decayed monotonically across global training epochs via the legacy `--decay_epochs` setting. Consequently, Phase 0 enjoyed high initial exploration and strong prior regularization, while Phase 1 and Phase 2 received partially or almost fully decayed parameters ($\\epsilon \\to \\epsilon_{\\min}$, $\\alpha \\to \\alpha_{\\min}$), severely impeding policy exploration and prior alignment on non-bear market regimes.

## Decision

1. **Phase-Cyclic Decay Topology**: Bind parameter decay directly to `curriculum_block_epochs` ($x$). For the 5-phase curriculum rotation (3 directional regime phases + 1 diagonal matched regime phase + 1 full experience extraction phase, epochs $0 \le e < 5x$):
   - Phase 0 ($0 \le e < x$): Downtrend regime (grids [0, 3, 6]).
   - Phase 1 ($x \le e < 2x$): Flat / range regime (grids [1, 4, 7]).
   - Phase 2 ($2x \le e < 3x$): Uptrend regime (grids [2, 5, 8]).
   - Phase 3 ($3x \le e < 4x$): Diagonal matched regime (slope 0 vol 0, slope 1 vol 1, slope 2 vol 2 -> grids [0, 4, 8]).
   - Phase 4 ($4x \le e < 5x$): Full experience extraction (all 9 grids [0..8]).
   - At intra-phase epoch $k = e \pmod x \in [0, x - 1]$, linearly anneal $\\epsilon$ from $\\epsilon_{\\text{init}}$ to $\\epsilon_{\\text{min}}$ and $\\alpha$ from $\\alpha_{\\text{init}}$ to $\\alpha_{\\text{min}}$ over $x$ epochs (reaching minimum exactly at $k = x - 1$).
   - At phase entry points ($e = x, 2x, 3x, 4x$), reset $\\epsilon$ and $\\alpha$ back to their respective initial maxima.
2. **Post-5x Convergence Clamping**: For all subsequent epochs ($e \ge 5x$, such as epochs 15~17 when $x=3, \\text{num\\_epoch}=18$), clamp $\\epsilon = \\epsilon_{\\text{min}}$ and $\\alpha = \\alpha_{\\text{min}}$ while continuing the 5-phase regime rotation.
3. **Global Learning Rate Monotonicity**: Keep optimizer learning rate `lr` on its existing global monotonic schedule across `num_epoch` (held for the first half, then linearly decayed to `lr_min`) without intra-phase resets, protecting representation stability and avoiding gradient destabilization.
4. **Phase Entry Exploration Recovery**: At phase boundaries $e = x, 2x, 3x, 4x$, reset `consecutive_no_new_experience_epochs = 0` and unmark `skip_exploration` (unless the replay buffer is physically full), ensuring newly activated regime phases have a full window to collect fresh samples.
5. **Validation Invariant**: Require $\\text{num\\_epoch} \ge 5 \times \\text{curriculum\\_block\\_epochs}$ to guarantee that every training session completes all 5 curriculum phases.
