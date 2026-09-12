# ADR-0014: Phase-Cyclic Parameter Decay in Diverse Training Curriculum

Date: 2026-09-12

Status: Accepted

## Context

Under ADR-0012, Stage I diverse training adopted a 3-phase rotating curriculum (Phase 0: Downtrend [0, 3, 6], Phase 1: Flat [1, 4, 7], Phase 2: Uptrend [2, 5, 8]), running in blocks of $x$ epochs (configured via `curriculum_block_epochs`, default 3). However, the exploration rate $\\epsilon$ and distillation regularization weight $\\alpha$ (`ada`) decayed monotonically across global training epochs via the legacy `--decay_epochs` setting. Consequently, Phase 0 enjoyed high initial exploration and strong prior regularization, while Phase 1 and Phase 2 received partially or almost fully decayed parameters ($\\epsilon \\to \\epsilon_{\\min}$, $\\alpha \\to \\alpha_{\\min}$), severely impeding policy exploration and prior alignment on non-bear market regimes.

## Decision

1. **Phase-Cyclic Decay Topology**: Bind parameter decay directly to `curriculum_block_epochs` ($x$). For the first full rotation across the 3 directional regimes (epochs $0 \le e < 3x$):
   - At intra-phase epoch $k = e \pmod x \in [0, x - 1]$, linearly anneal $\\epsilon$ from $\\epsilon_{\\text{init}}$ to $\\epsilon_{\\text{min}}$ and $\\alpha$ from $\\alpha_{\\text{init}}$ to $\\alpha_{\\text{min}}$ over $x$ epochs (i.e. reaching minimum exactly at $k = x - 1$).
   - At phase entry points ($e = x$ and $e = 2x$), reset $\\epsilon$ and $\\alpha$ back to their respective initial maxima.
2. **Post-3x Convergence Clamping**: For all subsequent epochs ($e \ge 3x$, such as epochs 9~17 when $x=3, \\text{num\\_epoch}=18$), clamp $\\epsilon = \\epsilon_{\\text{min}}$ and $\\alpha = \\alpha_{\\text{min}}$ while continuing the 3-phase directional regime rotation.
3. **Global Learning Rate Monotonicity**: Keep optimizer learning rate `lr` on its existing global monotonic schedule across `num_epoch` (held for the first half, then linearly decayed to `lr_min`) without intra-phase resets, protecting representation stability and avoiding gradient destabilization.
4. **Phase Entry Exploration Recovery**: At phase boundaries $e = x$ and $e = 2x$, reset `consecutive_no_new_experience_epochs = 0` and unmark `skip_exploration` (unless the replay buffer is physically full), ensuring newly activated regime phases have a full window to collect fresh samples.
5. **Deprecation of `decay_epochs` and Validation Invariant**: Remove the obsolete `--decay_epochs` parameter from CLI and trainer state. Require $\\text{num\\_epoch} \ge 3 \times \\text{curriculum\\_block\\_epochs}$ to guarantee that every training session completes at least one full three-phase exploration cycle.
