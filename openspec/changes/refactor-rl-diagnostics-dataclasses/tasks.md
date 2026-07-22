## 1. Implementation

- [ ] 1.1 Update focused tests to require dataclass interfaces for loss NaN diagnostics, qtable diagnostics, and parallel rollout contracts while preserving legacy `.to_dict()` and file-format assertions.
- [ ] 1.2 Refactor `loss_nan_diagnostics.py` to return dataclass diagnostics and update logging to use attributes.
- [ ] 1.3 Refactor `pretrain_qtable_diagnostics.py` to use dataclass sample items, manifest, CSV rows, sample diagnostics, worker result, and prepare result while preserving manifest JSON and diagnostics CSV formats.
- [ ] 1.4 Update `weight_advantage_pretrain.py` and the qtable-related portions of `parallel_weight_advantage_pretrain.py` to consume `PretrainQTableDiagnosticsResult` and `SamplePlanItem` via attributes.
- [ ] 1.5 Refactor `parallel_weight_advantage_pretrain.py` rollout task, epoch params, worker messages/results/errors, metrics, transition records, and round summaries to dataclass objects.
- [ ] 1.6 Run focused tests, Python compilation, and OpenSpec strict validation.
