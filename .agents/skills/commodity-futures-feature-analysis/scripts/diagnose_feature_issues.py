#!/usr/bin/env python3
"""
Commodity Futures Feature Engineering Diagnostic Tool

Performs data science checks:
1. Protected / Mandatory feature preservation audit.
2. Train vs Valid metric degradation.
3. Cross-contract metric variance and stability.
4. Metric anomaly detection (IC vs Importance mismatch, negative Sharpe).
"""

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def load_dataset_dir(dir_path: pathlib.Path) -> Dict[str, Any]:
    """Loads feature selection directory data."""
    manifest_file = dir_path / "feature_selection_manifest.json"
    metrics_file = dir_path / "aggregate_metrics.csv"
    state_file = dir_path / "state_features.npy"

    manifest = json.load(open(manifest_file, "r", encoding="utf-8")) if manifest_file.exists() else {}
    metrics_df = pd.read_csv(metrics_file) if metrics_file.exists() else None
    state_features = np.load(state_file, allow_pickle=True).tolist() if state_file.exists() else manifest.get("selected_features", [])

    per_contract = {}
    contract_dir = dir_path / "per_contract"
    if contract_dir.exists():
        for cf in contract_dir.glob("*_metrics.csv"):
            c_name = cf.name.replace("_metrics.csv", "")
            try:
                per_contract[c_name] = pd.read_csv(cf)
            except Exception:
                pass

    return {
        "dir_path": str(dir_path),
        "manifest": manifest,
        "metrics_df": metrics_df,
        "state_features": state_features,
        "per_contract": per_contract,
    }


def diagnose_protected_features(data: Dict[str, Any]) -> Dict[str, Any]:
    """Checks if mandatory/protected features are properly retained in state_features."""
    manifest = data.get("manifest", {})
    state_features = set(data.get("state_features", []))
    mandatory = manifest.get("mandatory_state_features", [])

    missing_mandatory = [f for f in mandatory if f not in state_features]
    blacklist = set(manifest.get("feature_blacklist", []))
    blacklisted_in_selected = [f for f in state_features if f in blacklist]

    return {
        "mandatory_total": len(mandatory),
        "mandatory_retained": len(mandatory) - len(missing_mandatory),
        "missing_mandatory": missing_mandatory,
        "blacklisted_in_selected": blacklisted_in_selected,
        "status": "PASS" if not missing_mandatory and not blacklisted_in_selected else "WARNING",
    }


def diagnose_metric_anomalies(data: Dict[str, Any]) -> Dict[str, Any]:
    """Identifies feature metric anomalies (e.g. strong importance but weak IC or negative Sharpe)."""
    metrics_df = data.get("metrics_df")
    if metrics_df is None or metrics_df.empty:
        return {"status": "SKIPPED", "reason": "No aggregate metrics found"}

    state_features = set(data.get("state_features", []))
    selected_df = metrics_df[metrics_df["feature"].isin(state_features)].copy()

    # 1. High Permutation Importance but low IC (|IC_Mean| < 0.01)
    low_ic_high_imp = []
    if "Permutation Importance_Mean" in selected_df and "IC_Mean" in selected_df:
        imp_threshold = selected_df["Permutation Importance_Mean"].quantile(0.75) if len(selected_df) > 0 else 0.05
        cond = (selected_df["Permutation Importance_Mean"] > imp_threshold) & (selected_df["IC_Mean"].abs() < 0.01)
        low_ic_high_imp = selected_df[cond]["feature"].tolist()

    # 2. Negative Sharpe Mean among selected features
    neg_sharpe = []
    if "Sharpe_Mean" in selected_df:
        neg_sharpe = selected_df[selected_df["Sharpe_Mean"] < -0.1]["feature"].tolist()

    return {
        "high_importance_low_ic_features": low_ic_high_imp,
        "negative_sharpe_features_count": len(neg_sharpe),
        "negative_sharpe_sample": neg_sharpe[:10],
        "status": "PASS" if not low_ic_high_imp else "WARNING",
    }


def diagnose_contract_stability(data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculates cross-contract metric stability."""
    per_contract = data.get("per_contract", {})
    if not per_contract:
        return {"status": "SKIPPED", "reason": "No per-contract metrics found"}

    state_features = data.get("state_features", [])
    contract_names = list(per_contract.keys())

    # Build per-feature IC dataframe across contracts
    feature_ic_across_contracts = {}
    for c_name, c_df in per_contract.items():
        if "feature" in c_df and "IC_Mean" in c_df:
            for _, row in c_df.iterrows():
                feat = row["feature"]
                if feat not in feature_ic_across_contracts:
                    feature_ic_across_contracts[feat] = {}
                feature_ic_across_contracts[feat][c_name] = row["IC_Mean"]

    df_ic = pd.DataFrame.from_dict(feature_ic_across_contracts, orient="index")
    selected_ic = df_ic.loc[df_ic.index.isin(state_features)] if not df_ic.empty else pd.DataFrame()

    unstable_features = []
    if not selected_ic.empty:
        # Features with sign changes across contracts
        pos_counts = (selected_ic > 0).sum(axis=1)
        neg_counts = (selected_ic < 0).sum(axis=1)
        # Sign flips mean both > 20% positive and > 20% negative across contracts
        n_contracts = len(contract_names)
        sign_flip_mask = (pos_counts >= max(1, 0.2 * n_contracts)) & (neg_counts >= max(1, 0.2 * n_contracts))
        unstable_features = selected_ic[sign_flip_mask].index.tolist()

    return {
        "contracts_evaluated": len(contract_names),
        "unstable_sign_flip_features_count": len(unstable_features),
        "unstable_features_sample": unstable_features[:10],
        "status": "PASS" if not unstable_features else "WARNING",
    }


def diagnose_train_valid_degradation(train_data: Dict[str, Any], valid_data: Dict[str, Any]) -> Dict[str, Any]:
    """Compares train vs valid metrics for selected state features."""
    train_metrics = train_data.get("metrics_df")
    valid_metrics = valid_data.get("metrics_df")
    state_features = train_data.get("state_features", [])

    if train_metrics is None or valid_metrics is None:
        return {"status": "SKIPPED", "reason": "Train or Valid aggregate metrics missing"}

    t_df = train_metrics.set_index("feature")
    v_df = valid_metrics.set_index("feature")

    common = [f for f in state_features if f in t_df.index and f in v_df.index]
    if not common:
        return {"status": "SKIPPED", "reason": "No common selected features in valid set"}

    ic_degradation = []
    for feat in common:
        t_ic = abs(t_df.loc[feat, "IC_Mean"]) if "IC_Mean" in t_df.columns else 0.0
        v_ic = abs(v_df.loc[feat, "IC_Mean"]) if "IC_Mean" in v_df.columns else 0.0
        drop_pct = (t_ic - v_ic) / max(1e-5, t_ic)
        if drop_pct > 0.5 and t_ic > 0.02:
            ic_degradation.append({
                "feature": feat,
                "train_ic": float(t_ic),
                "valid_ic": float(v_ic),
                "drop_pct": float(drop_pct),
            })

    return {
        "evaluated_common_features": len(common),
        "severe_degradation_features_count": len(ic_degradation),
        "degradation_list": sorted(ic_degradation, key=lambda x: x["drop_pct"], reverse=True)[:10],
        "status": "PASS" if not ic_degradation else "WARNING",
    }


def main():
    parser = argparse.ArgumentParser(description="Commodity Feature Engineering Diagnostics")
    parser.add_argument("--train-dir", type=str, default="PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/30min/fu/train",
                        help="Path to train selection directory")
    parser.add_argument("--valid-dir", type=str, default="PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/30min/fu/valid",
                        help="Optional path to valid selection directory")
    args = parser.parse_args()

    train_path = pathlib.Path(args.train_dir)
    train_data = load_dataset_dir(train_path)

    print(f"=== Feature Engineering Diagnostic Report: {train_path.name} ===")
    
    # 1. Protected Features Check
    prot_res = diagnose_protected_features(train_data)
    print(f"\n1. Protected / Mandatory Features Audit: [{prot_res['status']}]")
    print(f"   Retained: {prot_res['mandatory_retained']}/{prot_res['mandatory_total']}")
    if prot_res["missing_mandatory"]:
        print(f"   Missing Mandatory: {prot_res['missing_mandatory']}")

    # 2. Metric Anomalies Check
    anom_res = diagnose_metric_anomalies(train_data)
    print(f"\n2. Feature Metric Anomalies: [{anom_res['status']}]")
    if "high_importance_low_ic_features" in anom_res:
        print(f"   High Importance & Low IC Features: {len(anom_res['high_importance_low_ic_features'])}")
    print(f"   Negative Sharpe Features Count: {anom_res.get('negative_sharpe_features_count', 0)}")

    # 3. Contract Stability Check
    stab_res = diagnose_contract_stability(train_data)
    print(f"\n3. Cross-Contract Stability Audit: [{stab_res['status']}]")
    print(f"   Contracts Evaluated: {stab_res.get('contracts_evaluated', 0)}")
    print(f"   Unstable (Sign-flip) Features: {stab_res.get('unstable_sign_flip_features_count', 0)}")
    if stab_res.get("unstable_features_sample"):
        print(f"   Sample Unstable Features: {stab_res['unstable_features_sample']}")

    # 4. Train vs Valid Check
    if args.valid_dir and pathlib.Path(args.valid_dir).exists():
        valid_data = load_dataset_dir(pathlib.Path(args.valid_dir))
        deg_res = diagnose_train_valid_degradation(train_data, valid_data)
        print(f"\n4. Train vs Valid Degradation: [{deg_res['status']}]")
        print(f"   Severe IC Drop (>50% drop): {deg_res.get('severe_degradation_features_count', 0)}")
        for d in deg_res.get("degradation_list", []):
            print(f"   - `{d['feature']}`: Train IC={d['train_ic']:.4f} -> Valid IC={d['valid_ic']:.4f} (Drop: {d['drop_pct']*100:.1f}%)")


if __name__ == "__main__":
    main()
