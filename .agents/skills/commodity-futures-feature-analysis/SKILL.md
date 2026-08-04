---
name: commodity-futures-feature-analysis
description: Commodity futures feature engineering analysis and evaluation. Use when analyzing feature selection results (aggregate_metrics.csv, feature_selection_manifest.json, state_features.npy), auditing candidate vs selected features, diagnosing feature issues (low IC, multicollinearity, contract instability, train/valid degradation), verifying against code (data_preprocess/operator_futures/commodity/*.py) and spec (openspec/specs/commodity-futures-feature-engineering/spec.md), or generating data science feature reports.
---

# Commodity Futures Feature Analysis Skill

This skill provides procedural workflows, Python analysis scripts, and domain guidance for evaluating commodity futures feature engineering results, diagnosing feature quality issues, cross-referencing python operators and spec definitions, and producing structured data science reports.

## Key Files and Directory Conventions

- **Feature Generation Code**: `data_preprocess/operator_futures/commodity/*.py`
  - `downscale.py`: OHLCV, VWAP/TWAP, Quote Imbalance, Order Flow Imbalance (OFI), Queue Pressure.
  - `base_time_feature.py`: Mandatory time features (`BASE_TIME_FEATURE`).
  - `daily_base_feature.py` / `daily_mixed_frequency_feature.py` / `weekly_*`: Mixed-frequency historical features.
  - `cross_month_feature.py`: Main/sub-main spread, term structure, liquidity rollover.
  - `downscale_continuous_by_trading_day.py`: Rolling risk, Parkinson/Garman-Klass volatility, relative open interest.
- **Specification Document**: `openspec/specs/commodity-futures-feature-engineering/spec.md`
- **Feature Selection Outputs**: `PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/{target_freq}/{symbol}/{stage}/`
  - `feature_selection_manifest.json`: Pipeline run configuration, mandatory features, blacklists, and drop stage statistics.
  - `aggregate_metrics.csv`: IC, Rank IC, Permutation Importance, CatBoost Importance, Sharpe statistics.
  - `state_features.npy`: Array of final selected state feature names.
  - `per_contract/`: Individual contract metric CSVs for cross-contract stability checking.

---

## Workflow Steps

### 1. Run Automated Analysis & Diagnostic Scripts

Execute the bundled Python tools to extract summary statistics and detect potential feature flaws:

```bash
# 1. Summary Analysis
python .agents/skills/commodity-futures-feature-analysis/scripts/analyze_feature_selection.py \
  --dir PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/30min/fu/train \
  --format markdown --output selection_summary.md

# 2. Diagnostic Audit (Protected features, contract stability, train vs valid degradation)
python .agents/skills/commodity-futures-feature-analysis/scripts/diagnose_feature_issues.py \
  --train-dir PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/30min/fu/train \
  --valid-dir PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/30min/fu/valid
```

### 2. Verify Features Against Code & Spec

Cross-reference suspicious or top-performing features against the code and spec:
- Refer to `references/feature_dictionary_quickref.md` for feature prefix mapping.
- Use `codebase-memory-mcp` tools (`search_graph`, `get_code_snippet`) or grep to locate exact Python implementations in `data_preprocess/operator_futures/commodity/*.py`.
- Verify mathematical definitions and fail-fast constraints in `openspec/specs/commodity-futures-feature-engineering/spec.md`.

### 3. Diagnose Data Scientist Issues

Evaluate findings across 4 critical data science dimensions:
1. **Protected / Mandatory Features**: Ensure `BASE_TIME_FEATURE` items (e.g. `session_progress`, `is_night_session`) are present in `state_features.npy`.
2. **Train vs Valid Generalization**: Identify features with >50% IC or Permutation Importance drop on valid split (overfitting / non-stationarity).
3. **Cross-Contract Stability**: Check for sign-flips across contracts in `per_contract/*_metrics.csv`.
4. **Collinearity & Metric Mismatch**: Flag redundant features (e.g. high correlation among rolling windows) or features with strong importance but weak IC.

### 4. Produce Data Science Feature Engineering Report

Structure the report following this template:

```markdown
# Data Science Feature Engineering Analysis Report

## Executive Summary
- Dataset & Symbol: [e.g. FU 30min]
- Candidate Pool Size: [e.g. 865] | Selected Features: [e.g. 136] | Retention Ratio: [15.72%]
- Overall Quality Rating: [EXCELLENT / GOOD / NEEDS_ATTENTION]

## Pipeline Drop Breakdown
| Selection Stage | Dropped Features | Key Reasons |
|---|---|---|
| Hard Filter | N | Zero variance / missing values |
| Composite Drop | N | Low aggregate score |
| Correlation Filter | N | Multicollinearity cutoff |

## Core Findings & Diagnostics
1. **Top Predictive Features**: List top 5-10 by IC and Permutation Importance with quantitative metrics.
2. **Protected Features Audit**: Verification of mandatory state features.
3. **Generalization & Stability**: Highlights of train vs valid degradation or contract sign flips.
4. **Code & Math Trace**: Cross-check with `data_preprocess/operator_futures/commodity/*.py` and `spec.md`.

## Actionable Recommendations
- Parameter adjustments (e.g., window sizes).
- Feature modifications or new operator suggestions.
- Filtering threshold adjustments for next run.
```
