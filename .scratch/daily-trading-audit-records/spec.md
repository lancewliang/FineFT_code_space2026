# 每日交易审计记录规格

## Problem Statement

当前 DiHFT 回测能够生成汇总指标和部分逐步历史，但研究人员无法在每个 TradingDay 上稳定追溯“模型当时输出了什么、策略实际执行了什么、执行后账户发生了什么”。部分账户历史会在仓位结束后重置，旧产物缺少完整模型输出、成交事件、稳定关联键和可靠的 TradingDay 归属，因此无法形成预测、交易和账户之间可对账的证据链，也不能据此判断收益来自模型判断、规则风控、执行差异还是账户记账。

## Solution

为新的历史回测建立审计级事件契约，并新增离线诊断命令。高层回测保存真实模型推理、执行事件和永不重置的独立合约账户快照；诊断命令严格校验这些输入，然后按 TradingDay 原子发布 `prediction_records.csv`、`trade_records.csv`、`account_records.csv` 和根级 `audit_manifest.json`。三类记录使用确定性 ID 相互连接，只表达真实存在的模型输出和执行事实，不把 Q 值解释成盈利概率，也不从旧实验推测缺失事实。

## User Stories

1. As a strategy researcher, I want to inspect every genuine policy inference, so that I can see what the model knew and recommended at the decision time.
2. As a strategy researcher, I want the short, flat, and long action values recorded separately, so that I can inspect the strength and ambiguity of each decision.
3. As a strategy researcher, I want the best-versus-second-best action-value margin recorded, so that I can distinguish decisive outputs from near ties.
4. As a strategy researcher, I want routing slots and available gating evidence recorded with the policy output, so that I can connect high-level regime selection to the low-level action.
5. As a strategy researcher, I want model, feature, and parameter versions recorded, so that I can reproduce the source of a decision.
6. As a strategy researcher, I want action-persistence steps linked to the inference that armed them, so that repeated actions are not mistaken for new predictions.
7. As a strategy researcher, I want warm-up steps to remain visibly prediction-free, so that initialization is not counted as model skill.
8. As a risk researcher, I want defensive closes, hard stops, cooldowns, and circuit breakers identified as rule actions, so that risk interventions are not attributed to the policy network.
9. As an execution researcher, I want requested positions and executed positions recorded separately, so that execution constraints are visible.
10. As an execution researcher, I want every execution leg recorded, so that opens, reductions, closes, reversals, rejections, and partial fills can be audited.
11. As an execution researcher, I want reversals split into closing and opening legs, so that fees, quantities, and realized PnL remain attributable.
12. As an execution researcher, I want rejected or unavailable actions retained as events, so that missing trades are explainable.
13. As an execution researcher, I want reference price, fill price, quantity, commission, realized PnL, and slippage evidence recorded, so that execution costs can be reconciled.
14. As an account auditor, I want an initial account snapshot and a post-step snapshot for every environment step, so that every state transition has explicit boundaries.
15. As an account auditor, I want wallet balance, unrealized PnL, equity, margin, available balance, position, exposure, leverage, peak equity, and drawdown recorded, so that account risk can be reconstructed.
16. As an account auditor, I want account audit history to survive position closure, so that the full run remains available after the final trade.
17. As an account auditor, I want every trade event linked to its pre- and post-execution account snapshots, so that account effects are locally verifiable.
18. As an account auditor, I want equity identities and reward changes validated before publication, so that inconsistent histories cannot look authoritative.
19. As a futures researcher, I want records grouped by TradingDay rather than calendar date, so that night-session events belong to the correct business day.
20. As a futures researcher, I want TradingDay read from explicit market data or an explicit mapping, so that holidays and session exceptions are not guessed.
21. As a multi-contract researcher, I want each independent contract wallet identified explicitly, so that independent accounts are not presented as a shared portfolio.
22. As a data consumer, I want deterministic record IDs, so that repeated diagnostic runs produce stable joins.
23. As a data consumer, I want three event-detail files for each TradingDay, so that I can inspect one day without loading the whole experiment.
24. As a data consumer, I want a run manifest with schema version, source identity, row counts, account scope, TradingDay source, and validation status, so that I can assess provenance before reading the records.
25. As a data consumer, I want all non-null foreign keys validated, so that joins never silently lose events.
26. As an operator, I want the diagnostic publication to be atomic, so that a failed run leaves no partial official output.
27. As an operator, I want missing audit-grade input to stop the diagnostic command, so that legacy proxies cannot enter the new evidence set.
28. As a model reviewer, I want Q values named as action values rather than probabilities, so that they are not misinterpreted as calibrated chances of profit.
29. As a model reviewer, I want the absence of a future-horizon predictor stated explicitly, so that the daily records do not claim to predict today’s or the next TradingDay’s close.
30. As a developer, I want the full feature verified through public run and diagnostic entry points, so that tests describe observable behavior instead of private implementation.

## Implementation Decisions

- The feature applies only to newly generated audit-grade backtests. There is no legacy reconstruction mode and no inference of missing historical facts.
- The account scope is `independent_contract_wallet`. Shared capital allocation, cross-contract margin, and shared-portfolio risk are not synthesized.
- The business date is TradingDay. The generator accepts either an explicit TradingDay market-data column or an explicit event-to-TradingDay mapping. It never derives TradingDay from a fixed clock rule.
- Each TradingDay is a publication partition containing three CSV files. A root JSON manifest describes the complete run and every published partition.
- A Prediction Record exists only for genuine model inference. It contains run and model identity, contract, event time, market index, routing evidence, action values, action-value margin, available-action state, selected action, and prediction provenance.
- Q values are Action Values. They are not profitability probabilities, calibrated confidence, or explicit one-to-two-TradingDay forecasts.
- A Trade Record exists for each execution leg or execution attempt. It records the requested and executed state, quantities, event type, execution status, prices, fees, realized PnL, slippage evidence, decision reason, and account snapshot links.
- A reversal is represented by distinct closing and opening legs sharing one Decision ID. Rejection and partial-fill outcomes remain observable.
- An Account Record contains the initial snapshot and a post-step snapshot for every environment step. Audit histories are append-only for the duration of a run and are never reset when a position becomes flat.
- Account Records include enough information to validate `equity = wallet_balance + unrealized_pnl` and to reconcile equity changes with the frozen reward and fee conventions.
- Prediction, trade-event, execution-leg, account-snapshot, and decision IDs are deterministic functions of run identity, contract, step, event kind, and leg number.
- Action-persistence events reference the Prediction Record that armed persistence. Warm-up and rule-driven risk actions may have no Prediction Record and must carry an explicit Decision Reason.
- The offline command requires explicit result, market-data, output, and run identity inputs. Model and feature identities are read from frozen run metadata and included in provenance.
- The diagnostic command validates all contracts and all partitions in a staging directory. It publishes the final output only after uniqueness, referential integrity, length, TradingDay, account, and reward reconciliation checks pass.
- Any validation failure aborts the whole publication. No partial TradingDay partition is promoted to the official output directory.
- The domain glossary will define Prediction Record, Trade Record, Account Record, Decision ID, and audit-grade backtest without embedding implementation details.
- An ADR will record the hard-to-reverse choice to use an append-only three-record evidence chain, explicit TradingDay assignment, deterministic links, independent contract wallets, and strict atomic publication.

## Testing Decisions

- The primary test seam is the public offline diagnostic command. Given a compact audit-grade multi-contract run, tests assert the published partition structure, CSV contents, deterministic IDs, joins, account reconciliation, manifest, and atomic failure behavior.
- The necessary upstream seam is the public high-level backtest run using existing fake network and fake environment fixtures. Tests assert that genuine action values, execution facts, and full account snapshots reach the saved audit artifacts.
- Tests assert observable records and validation outcomes rather than private helper calls or internal list layouts.
- A minimal open-hold-close path proves the first complete prediction-to-trade-to-account tracer bullet.
- Action-persistence, warm-up, defensive action, hard-stop, cooldown, and circuit-breaker scenarios verify prediction provenance without fabricated records.
- Reversal, partial-fill, and price-limit rejection scenarios verify execution-leg semantics and pre/post account links.
- Night-session and explicit TradingDay mapping scenarios verify business-date partitioning without calendar-date assumptions.
- Repeated runs verify deterministic IDs and stable output ordering.
- Corrupt input scenarios cover missing artifacts, duplicate keys, dangling foreign keys, length mismatches, non-finite values, account identity violations, reward mismatches, and missing TradingDay assignment.
- Atomic-publication tests verify that failed validation leaves no official daily partitions.
- Existing trading-diagnostic end-to-end fixtures, action-persistence router fixtures, multi-step fake environments, position-cost lifecycle tests, and price-limit execution tests are the preferred prior art.

## Out of Scope

- Live, paper-trading, or real-time record publication.
- Backfilling or reconstructing the three records from legacy experiment output.
- Training a new predictor for the current TradingDay close or next TradingDay close.
- Converting Q values or VAE scores into profitability probabilities.
- Shared-wallet portfolio accounting, cross-contract capital allocation, or portfolio margin.
- Broker orders, exchange acknowledgements, or claims about real-world fill reachability.
- A separate one-row-per-day summary dataset; summaries can be derived from the event records.
- Guaranteeing sustained future profitability.

## Further Notes

The three records establish observability and attribution. They make it possible to distinguish model output, policy or risk decisions, simulated execution, and account consequences, but they do not by themselves prove that a model has predictive power. Explicit future-horizon probabilities and outcomes can be added later only when an upstream model produces and calibrates them under a separately frozen prediction protocol.
