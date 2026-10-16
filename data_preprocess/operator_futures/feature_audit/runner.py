"""Execution runner for multi-horizon feature audit."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import polars as pl

from .calendar_adapter import CalendarAdapter
from .config import AuditConfig
from .contracts import FeatureMetricResult, InvalidReason
from .eligibility import qualify_candidates
from .legacy_selection import load_legacy_selection
from .metrics import compute_spearman_correlation, compute_spearman_matrix
from .reference_prices import ReferencePriceProvider
from .report import generate_audit_report, generate_matrices
from .split_guard import generate_evaluation_windows, EvaluationWindow
from .uncertainty import MatrixBootstrapResult, joint_moving_block_bootstrap, joint_moving_block_bootstrap_matrix


def categorize_three_way_trajectory(
    ic_1h: Optional[float], ic_1d: Optional[float], ic_5d: Optional[float]
) -> str:
    """Categorize feature trajectory across 1h -> 1day (next_td) -> 5day (next_5td)."""
    if (
        ic_1h is None
        or ic_1d is None
        or ic_5d is None
        or np.isnan(ic_1h)
        or np.isnan(ic_1d)
        or np.isnan(ic_5d)
    ):
        return "INSUFFICIENT_COMMON_DATA"

    a1 = abs(ic_1h)
    ad = abs(ic_1d)
    a5 = abs(ic_5d)

    if max(a1, ad, a5) < 0.03:
        return "NEGLIGIBLE_SIGNAL"

    s1, sd, s5 = np.sign(ic_1h), np.sign(ic_1d), np.sign(ic_5d)
    same_sign = bool(s1 == sd == s5)

    if same_sign and a1 >= ad >= a5 and a1 >= 0.05:
        return "CLEAN_DECAY"
    elif same_sign and a1 <= ad <= a5 and a5 >= 0.05:
        return "CLEAN_EMERGENCE"
    elif ad > a1 and ad > a5 and ad >= 0.05:
        return "DAY_RESONANCE"
    elif (
        same_sign
        and a1 >= 0.05
        and ad >= 0.05
        and a5 >= 0.05
        and (max(a1, ad, a5) / max(min(a1, ad, a5), 1e-6)) < 1.6
    ):
        return "ALL_PERSISTENT"
    elif ad < a1 and ad < a5:
        return "VALLEY_INTERMEDIATE"
    elif not same_sign:
        return "SIGN_FLIP_INSTABILITY"
    else:
        return "MIXED_IRREGULAR"


class AuditRunner:
    """Orchestrates an independent feature audit run."""

    def __init__(self, config: AuditConfig):
        self.config = config

    def run(self) -> Dict[str, Any]:
        # 1. Validation
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

        legacy_selection = (
            load_legacy_selection(self.config.legacy_selection_manifest)
            if self.config.legacy_selection_manifest else None
        )

        result_dir = Path(self.config.result_root)
        result_dir.mkdir(parents=True, exist_ok=True)

        # 2. Split manifest & Test Sealing
        with open(self.config.split_manifest, "r", encoding="utf-8") as f:
            split_manifest = json.load(f)
        raw_test = split_manifest.get("test_contracts")
        if raw_test is None:
            test_val = split_manifest.get("sets", {}).get("test", [])
            if isinstance(test_val, dict):
                raw_test = [c.get("contract") if isinstance(c, dict) else c for c in test_val.get("contracts", [])]
            elif isinstance(test_val, list):
                raw_test = [c.get("contract") if isinstance(c, dict) else c for c in test_val]
            else:
                raw_test = []
        test_contracts = set(raw_test)

        # 3. Adapters
        calendar = CalendarAdapter.from_manifest(self.config.calendar_manifest)
        price_provider = ReferencePriceProvider.from_manifest(self.config.price_source_manifest)

        # 4. Read input feature data (sealing test contracts if read_final_test is False)
        with open(self.config.feature_availability_manifest, "r", encoding="utf-8") as f:
            feat_manifest = json.load(f)
        data_path = Path(feat_manifest.get("data_path", "."))

        feather_files = list(data_path.glob("*.feather"))
        if not feather_files:
            raise FileNotFoundError(f"No feather files found in data path: {data_path}")

        frames: List[pl.DataFrame] = []
        for ff in feather_files:
            if not self.config.read_final_test and ff.stem in test_contracts:
                continue
            df = pl.read_ipc(ff)
            if not self.config.read_final_test and "contract" in df.columns:
                df = df.filter(~pl.col("contract").is_in(list(test_contracts)))
            frames.append(df)

        if not frames:
            raise ValueError("No eligible feature data loaded after applying Test sealing rules")

        all_data = pl.concat(frames)

        # Add row index for fast matrix slicing
        all_data = all_data.with_columns(pl.int_range(0, pl.len()).alias("_row_idx"))

        # 5. Qualify candidate features
        candidates = qualify_candidates(all_data.schema)
        qualified_features = [c.name for c in candidates if c.is_qualified and c.name != "_row_idx"]

        # Extract full feature matrix ONCE
        X_full = all_data.select(qualified_features).to_numpy()

        # 6. Parse legacy selection annotations if provided
        legacy_stream_map = (
            {feature: legacy_selection.membership(feature) for feature in qualified_features}
            if legacy_selection else {}
        )
        if legacy_selection:
            pl.DataFrame(legacy_selection.comparison_rows(qualified_features)).write_csv(
                result_dir / "legacy_selection_comparison.csv"
            )

        # 7. Build timestamps & Evaluation Windows
        if all_data.schema["timestamp"] == pl.Utf8:
            all_data = all_data.with_columns(
                pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
            )

        if "trading_day" not in all_data.columns:
            ts_list = all_data["timestamp"].to_list()
            td_list = [calendar.get_trading_day(t if isinstance(t, datetime) else datetime.fromisoformat(str(t))) for t in ts_list]
            all_data = all_data.with_columns(pl.Series("trading_day", td_list))

        min_dt = all_data["timestamp"].min()
        max_dt = all_data["timestamp"].max()

        eval_windows = generate_evaluation_windows(
            start_date=min_dt,
            train_end_date=max_dt,
            valid_end_date=max_dt,
            warmup_months=6,
            window_months=3,
        )

        row_label_records = []
        contracts = all_data["contract"].unique().to_list()

        contract_max_time = {}
        for c in contracts:
            sub_c = all_data.filter(pl.col("contract") == c)
            contract_max_time[c] = sub_c["timestamp"].max()

        for contract in contracts:
            sub = all_data.filter(pl.col("contract") == contract).sort("timestamp")
            ts_list = sub["timestamp"].to_list()
            td_list = sub["trading_day"].to_list()
            c_max_dt = contract_max_time.get(contract)

            for ts, td in zip(ts_list, td_list):
                dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") if isinstance(ts, str) else ts

                cur_win = eval_windows[0]
                for w in eval_windows:
                    if w.start_time <= dt <= w.end_time:
                        cur_win = w
                        break

                start_res = price_provider.lookup_as_of(contract, dt)

                for horizon in self.config.horizons:
                    target_dt = None
                    target_td = td

                    if horizon == "1h":
                        target_dt, target_td = calendar.add_active_trading_minutes(dt, 60)
                    elif horizon == "2h":
                        target_dt, target_td = calendar.add_active_trading_minutes(dt, 120)
                    elif horizon == "4h":
                        target_dt, target_td = calendar.add_active_trading_minutes(dt, 240)
                    elif horizon == "today":
                        close_dt, target_td = calendar.get_trading_day_close(td, offset=0)
                        target_dt = close_dt
                    elif horizon == "next_td":
                        close_dt, target_td = calendar.get_trading_day_close(td, offset=1)
                        target_dt = close_dt
                    elif horizon == "next_5td":
                        close_dt, target_td = calendar.get_trading_day_close(td, offset=5)
                        target_dt = close_dt
                    elif horizon == "next_15td":
                        close_dt, target_td = calendar.get_trading_day_close(td, offset=15)
                        target_dt = close_dt
                    else:
                        raise ValueError(f"Unsupported horizon: {horizon}")

                    if target_dt is None:
                        reason = InvalidReason.INSUFFICIENT_TAIL.value
                        if c_max_dt and dt >= c_max_dt:
                            reason = InvalidReason.CONTRACT_EXPIRED.value
                        row_label_records.append({
                            "contract": contract,
                            "timestamp": dt,
                            "trading_day": td,
                            "horizon": horizon,
                            "window_id": cur_win.window_id,
                            "is_historical_reference": cur_win.is_historical_reference,
                            "start_price": start_res.price,
                            "end_price": None,
                            "target_return": None,
                            "abs_target_return": None,
                            "valid_mask": False,
                            "invalid_reason": reason,
                        })
                        continue

                    if target_dt <= dt:
                        row_label_records.append({
                            "contract": contract,
                            "timestamp": dt,
                            "trading_day": td,
                            "horizon": horizon,
                            "window_id": cur_win.window_id,
                            "is_historical_reference": cur_win.is_historical_reference,
                            "start_price": start_res.price,
                            "end_price": None,
                            "target_return": None,
                            "abs_target_return": None,
                            "valid_mask": False,
                            "invalid_reason": InvalidReason.ZERO_LENGTH_HORIZON.value,
                        })
                        continue

                    if self.config.purge_by_label_available_at and target_dt > cur_win.end_time and not cur_win.is_partial:
                        row_label_records.append({
                            "contract": contract,
                            "timestamp": dt,
                            "trading_day": td,
                            "horizon": horizon,
                            "window_id": cur_win.window_id,
                            "is_historical_reference": cur_win.is_historical_reference,
                            "start_price": start_res.price,
                            "end_price": None,
                            "target_return": None,
                            "abs_target_return": None,
                            "valid_mask": False,
                            "invalid_reason": "UNMATURED_AT_WINDOW_BOUNDARY",
                        })
                        continue

                    if not start_res.price:
                        row_label_records.append({
                            "contract": contract,
                            "timestamp": dt,
                            "trading_day": td,
                            "horizon": horizon,
                            "window_id": cur_win.window_id,
                            "is_historical_reference": cur_win.is_historical_reference,
                            "start_price": None,
                            "end_price": None,
                            "target_return": None,
                            "abs_target_return": None,
                            "valid_mask": False,
                            "invalid_reason": start_res.reason.value,
                        })
                        continue

                    end_res = price_provider.lookup_as_of(contract, target_dt)
                    if not end_res.price:
                        reason = end_res.reason.value
                        if reason == InvalidReason.MISSING_TARGET_PRICE.value and c_max_dt and target_dt > c_max_dt:
                            reason = InvalidReason.CONTRACT_EXPIRED.value
                        row_label_records.append({
                            "contract": contract,
                            "timestamp": dt,
                            "trading_day": td,
                            "horizon": horizon,
                            "window_id": cur_win.window_id,
                            "is_historical_reference": cur_win.is_historical_reference,
                            "start_price": start_res.price,
                            "end_price": None,
                            "target_return": None,
                            "abs_target_return": None,
                            "valid_mask": False,
                            "invalid_reason": reason,
                        })
                    else:
                        ret = (end_res.price / start_res.price) - 1.0
                        row_label_records.append({
                           "contract": contract,
                           "timestamp": dt,
                           "trading_day": td,
                           "horizon": horizon,
                           "window_id": cur_win.window_id,
                           "is_historical_reference": cur_win.is_historical_reference,
                           "start_price": start_res.price,
                           "end_price": end_res.price,
                           "target_return": ret,
                           "abs_target_return": abs(ret),
                           "valid_mask": True,
                           "invalid_reason": InvalidReason.NONE.value,
                       })

        labels_schema = {
            "contract": pl.String,
            "timestamp": all_data["timestamp"].dtype,
            "trading_day": pl.String,
            "horizon": pl.String,
            "window_id": pl.String,
            "is_historical_reference": pl.Boolean,
            "start_price": pl.Float64,
            "end_price": pl.Float64,
            "target_return": pl.Float64,
            "abs_target_return": pl.Float64,
            "valid_mask": pl.Boolean,
            "invalid_reason": pl.String,
        }
        labels_df = pl.DataFrame(row_label_records, schema=labels_schema)
        labels_df.write_parquet(result_dir / "row_labels.parquet")

        # Summarize invalid reasons
        invalid_summary = (
            labels_df.filter(~pl.col("valid_mask"))
            .group_by(["horizon", "invalid_reason"])
            .len(name="count")
            .sort(["horizon", "count"], descending=[False, True])
        )
        invalid_summary.write_csv(result_dir / "invalid_reasons_summary.csv")

        # 8. Compute window strata, aggregations, and bootstrap uncertainty
        strata_records: List[Dict[str, Any]] = []
        metric_results: List[Dict[str, Any]] = []
        ci_diag_records: List[Dict[str, Any]] = []

        joined_all = all_data.join(labels_df, on=["contract", "timestamp", "trading_day"], how="inner")
        trading_days_sorted = sorted(all_data["trading_day"].unique().to_list())

        ev_proto = self.config.evidence_protocol
        base_block_len = max(ev_proto.minimum_block_trading_days, 1)

        D = len(qualified_features)

        for horizon in self.config.horizons:
            h_joined = joined_all.filter(pl.col("horizon") == horizon)

            h_span = 1
            if horizon == "next_5td":
                h_span = 5
            elif horizon == "next_15td":
                h_span = 15
            effective_block_len = max(base_block_len, h_span)

            for target_kind in ["return", "abs_return"]:
                target_col = "target_return" if target_kind == "return" else "abs_target_return"

                valid_h = h_joined.filter(pl.col("valid_mask"))
                total_p = len(h_joined)
                valid_p = len(valid_h)
                coverage = float(valid_p / total_p) if total_p > 0 else 0.0

                if valid_p < 3:
                    ci_diag_records.append({
                        "horizon": horizon,
                        "target_kind": target_kind,
                        "num_valid_days": 0,
                        "block_length": effective_block_len,
                        "block_length_2x": effective_block_len * ev_proto.sensitivity_block_multiplier,
                        "effective_blocks": 0.0,
                        "effective_blocks_2x": 0.0,
                        "min_effective_blocks": ev_proto.minimum_effective_blocks,
                        "min_effective_blocks_2x": max(1, ev_proto.minimum_effective_blocks // 2),
                        "ci_status": "INSUFFICIENT_PAIRS",
                        "ci_status_2x": "INSUFFICIENT_PAIRS",
                        "null_reason": "INSUFFICIENT_PAIRS",
                        "null_reason_2x": "INSUFFICIENT_PAIRS",
                    })
                    for feat_name in qualified_features:
                        metric_results.append({
                            "feature": feat_name,
                            "horizon": horizon,
                            "target_kind": target_kind,
                            "legacy_stream_membership": legacy_stream_map.get(feat_name, "unassigned"),
                            "spearman_corr": None,
                            "contract_equal_spearman": None,
                            "sample_weighted_spearman": None,
                            "ci_lower": None,
                            "ci_upper": None,
                            "ci_lower_2x": None,
                            "ci_upper_2x": None,
                            "sign_consistency": None,
                            "eligible_contracts_count": 0,
                            "valid_pairs": valid_p,
                            "total_pairs": total_p,
                            "coverage_ratio": coverage,
                            "null_reason": "INSUFFICIENT_PAIRS",
                            "ci_status": "INSUFFICIENT_PAIRS",
                            "ci_status_2x": "INSUFFICIENT_PAIRS",
                            "ci_valid_trading_days": 0,
                            "ci_block_length": effective_block_len,
                            "ci_block_length_2x": effective_block_len * ev_proto.sensitivity_block_multiplier,
                            "ci_effective_blocks": 0.0,
                            "ci_effective_blocks_2x": 0.0,
                            "ci_min_effective_blocks": ev_proto.minimum_effective_blocks,
                            "ci_min_effective_blocks_2x": max(1, ev_proto.minimum_effective_blocks // 2),
                            "ci_null_reason": "INSUFFICIENT_PAIRS",
                            "ci_null_reason_2x": "INSUFFICIENT_PAIRS",
                        })
                    continue

                # Slicing from pre-converted X_full matrix using integer row indices
                row_idxs_h = valid_h["_row_idx"].to_numpy()
                X_mat = X_full[row_idxs_h, :]
                y_vec = valid_h[target_col].to_numpy()

                # Per-contract matrix calculations
                contract_corrs_list = []
                contract_weights_list = []

                for contract in contracts:
                    c_valid = valid_h.filter(pl.col("contract") == contract)
                    if len(c_valid) >= 3:
                        r_c = c_valid["_row_idx"].to_numpy()
                        X_c = X_full[r_c, :]
                        y_c = c_valid[target_col].to_numpy()
                        c_corrs = compute_spearman_matrix(X_c, y_c)
                        contract_corrs_list.append(c_corrs)
                        contract_weights_list.append(len(c_valid))

                    # Window strata
                    for win in eval_windows:
                        w_valid = valid_h.filter(
                            (pl.col("contract") == contract) &
                            (pl.col("window_id") == win.window_id)
                        )
                        if len(w_valid) >= 3:
                            r_w = w_valid["_row_idx"].to_numpy()
                            X_w = X_full[r_w, :]
                            y_w = w_valid[target_col].to_numpy()
                            w_corrs = compute_spearman_matrix(X_w, y_w)
                            for f_idx, feat_name in enumerate(qualified_features):
                                strata_records.append({
                                    "feature": feat_name,
                                    "horizon": horizon,
                                    "target_kind": target_kind,
                                    "contract": contract,
                                    "window_id": win.window_id,
                                    "is_historical_reference": win.is_historical_reference,
                                    "spearman_corr": float(w_corrs[f_idx]) if not np.isnan(w_corrs[f_idx]) else None,
                                    "valid_pairs": len(w_valid),
                                    "null_reason": None if not np.isnan(w_corrs[f_idx]) else "UNDEFINED",
                                })
                        else:
                            for feat_name in qualified_features:
                                strata_records.append({
                                    "feature": feat_name,
                                    "horizon": horizon,
                                    "target_kind": target_kind,
                                    "contract": contract,
                                    "window_id": win.window_id,
                                    "is_historical_reference": win.is_historical_reference,
                                    "spearman_corr": None,
                                    "valid_pairs": len(w_valid),
                                    "null_reason": "INSUFFICIENT_PAIRS",
                                })

                num_eligible_c = len(contract_corrs_list)
                if num_eligible_c > 0:
                    c_corrs_mat = np.array(contract_corrs_list)  # (C, D)
                    weights_arr = np.array(contract_weights_list)

                    valid_mask = ~np.isnan(c_corrs_mat)
                    valid_cnt = np.sum(valid_mask, axis=0)
                    contract_equal_arr = np.where(
                        valid_cnt > 0,
                        np.sum(np.where(valid_mask, c_corrs_mat, 0.0), axis=0) / np.maximum(valid_cnt, 1),
                        np.nan,
                    )

                    w = weights_arr[:, None]
                    masked_w = np.where(valid_mask, w, 0.0)
                    w_sum = np.sum(masked_w, axis=0)
                    weighted_val_sum = np.sum(np.where(valid_mask, c_corrs_mat * w, 0.0), axis=0)
                    sample_weighted_arr = np.where(
                        w_sum > 0,
                        weighted_val_sum / np.maximum(w_sum, 1.0),
                        np.nan,
                    )
                else:
                    c_corrs_mat = np.empty((0, D))
                    contract_equal_arr = compute_spearman_matrix(X_mat, y_vec)
                    sample_weighted_arr = contract_equal_arr

                sign_consistency_arr = [None] * D
                sign_null_reasons = [None] * D
                min_c_for_sign = ev_proto.minimum_contracts_for_sign_consistency
                if num_eligible_c < min_c_for_sign:
                    sign_null_reasons = ["INSUFFICIENT_CONTRACTS_FOR_SIGN_CONSISTENCY"] * D
                else:
                    for f_idx in range(D):
                        if np.isnan(contract_equal_arr[f_idx]):
                            sign_consistency_arr[f_idx] = None
                            sign_null_reasons[f_idx] = "UNDEFINED"
                        else:
                            prim_sign = np.sign(contract_equal_arr[f_idx])
                            valid_contracts_for_feat = c_corrs_mat[~np.isnan(c_corrs_mat[:, f_idx]), f_idx]
                            if len(valid_contracts_for_feat) >= min_c_for_sign:
                                same_cnt = sum(1 for c_c in valid_contracts_for_feat if np.sign(c_c) == prim_sign)
                                sign_consistency_arr[f_idx] = float(same_cnt / len(valid_contracts_for_feat))
                            else:
                                sign_consistency_arr[f_idx] = None
                                sign_null_reasons[f_idx] = "INSUFFICIENT_CONTRACTS_FOR_SIGN_CONSISTENCY"

                # Vectorized joint moving-block bootstrap
                valid_tds = valid_h["trading_day"].to_numpy()
                day_indices: Dict[str, np.ndarray] = {}
                for td in trading_days_sorted:
                    idxs = np.where(valid_tds == td)[0]
                    day_indices[td] = idxs

                res_base = joint_moving_block_bootstrap_matrix(
                    trading_days=trading_days_sorted,
                    day_indices=day_indices,
                    X=X_mat,
                    y=y_vec,
                    block_length=effective_block_len,
                    min_effective_blocks=ev_proto.minimum_effective_blocks,
                    n_replicates=ev_proto.replicates,
                    seed=ev_proto.seed,
                    confidence_level=ev_proto.confidence_level,
                )

                res_2x = joint_moving_block_bootstrap_matrix(
                    trading_days=trading_days_sorted,
                    day_indices=day_indices,
                    X=X_mat,
                    y=y_vec,
                    block_length=effective_block_len * ev_proto.sensitivity_block_multiplier,
                    min_effective_blocks=max(1, ev_proto.minimum_effective_blocks // 2),
                    n_replicates=ev_proto.replicates,
                    seed=ev_proto.seed,
                    confidence_level=ev_proto.confidence_level,
                )

                ci_diag_records.append({
                    "horizon": horizon,
                    "target_kind": target_kind,
                    "num_valid_days": res_base.num_valid_days,
                    "block_length": res_base.block_length,
                    "block_length_2x": res_2x.block_length,
                    "effective_blocks": res_base.effective_blocks,
                    "effective_blocks_2x": res_2x.effective_blocks,
                    "min_effective_blocks": res_base.min_effective_blocks,
                    "min_effective_blocks_2x": res_2x.min_effective_blocks,
                    "ci_status": res_base.ci_status,
                    "ci_status_2x": res_2x.ci_status,
                    "null_reason": res_base.null_reason,
                    "null_reason_2x": res_2x.null_reason,
                })

                for f_idx, feat_name in enumerate(qualified_features):
                    c_eq = float(contract_equal_arr[f_idx]) if not np.isnan(contract_equal_arr[f_idx]) else None
                    s_wt = float(sample_weighted_arr[f_idx]) if not np.isnan(sample_weighted_arr[f_idx]) else None

                    l_base = float(res_base.ci_lower[f_idx]) if not np.isnan(res_base.ci_lower[f_idx]) else None
                    u_base = float(res_base.ci_upper[f_idx]) if not np.isnan(res_base.ci_upper[f_idx]) else None
                    l_2x = float(res_2x.ci_lower[f_idx]) if not np.isnan(res_2x.ci_lower[f_idx]) else None
                    u_2x = float(res_2x.ci_upper[f_idx]) if not np.isnan(res_2x.ci_upper[f_idx]) else None

                    null_reason = sign_null_reasons[f_idx]
                    if c_eq is None and null_reason is None:
                        null_reason = "UNDEFINED"

                    metric_results.append({
                        "feature": feat_name,
                        "horizon": horizon,
                        "target_kind": target_kind,
                        "legacy_stream_membership": legacy_stream_map.get(feat_name, "unassigned"),
                        "spearman_corr": c_eq,
                        "contract_equal_spearman": c_eq,
                        "sample_weighted_spearman": s_wt,
                        "ci_lower": l_base,
                        "ci_upper": u_base,
                        "ci_lower_2x": l_2x,
                        "ci_upper_2x": u_2x,
                        "sign_consistency": sign_consistency_arr[f_idx],
                        "eligible_contracts_count": num_eligible_c,
                        "valid_pairs": valid_p,
                        "total_pairs": total_p,
                        "coverage_ratio": coverage,
                        "null_reason": null_reason,
                        "ci_status": res_base.ci_status,
                        "ci_status_2x": res_2x.ci_status,
                        "ci_valid_trading_days": res_base.num_valid_days,
                        "ci_block_length": res_base.block_length,
                        "ci_block_length_2x": res_2x.block_length,
                        "ci_effective_blocks": res_base.effective_blocks,
                        "ci_effective_blocks_2x": res_2x.effective_blocks,
                        "ci_min_effective_blocks": res_base.min_effective_blocks,
                        "ci_min_effective_blocks_2x": res_2x.min_effective_blocks,
                        "ci_null_reason": res_base.null_reason,
                        "ci_null_reason_2x": res_2x.null_reason,
                    })

        strata_schema = {
            "feature": pl.String,
            "horizon": pl.String,
            "target_kind": pl.String,
            "contract": pl.String,
            "window_id": pl.String,
            "is_historical_reference": pl.Boolean,
            "spearman_corr": pl.Float64,
            "valid_pairs": pl.Int64,
            "null_reason": pl.String,
        }
        strata_df = pl.DataFrame(strata_records, schema=strata_schema)
        strata_df.write_csv(result_dir / "window_strata_metrics.csv")

        ci_diag_schema = {
            "horizon": pl.String,
            "target_kind": pl.String,
            "num_valid_days": pl.Int64,
            "block_length": pl.Int64,
            "block_length_2x": pl.Int64,
            "effective_blocks": pl.Float64,
            "effective_blocks_2x": pl.Float64,
            "min_effective_blocks": pl.Int64,
            "min_effective_blocks_2x": pl.Int64,
            "ci_status": pl.String,
            "ci_status_2x": pl.String,
            "null_reason": pl.String,
            "null_reason_2x": pl.String,
        }
        ci_diag_df = pl.DataFrame(ci_diag_records, schema=ci_diag_schema)
        ci_diag_df.write_csv(result_dir / "ci_diagnostics.csv")

        metric_schema = {
            "feature": pl.String,
            "horizon": pl.String,
            "target_kind": pl.String,
            "legacy_stream_membership": pl.String,
            "spearman_corr": pl.Float64,
            "contract_equal_spearman": pl.Float64,
            "sample_weighted_spearman": pl.Float64,
            "ci_lower": pl.Float64,
            "ci_upper": pl.Float64,
            "ci_lower_2x": pl.Float64,
            "ci_upper_2x": pl.Float64,
            "sign_consistency": pl.Float64,
            "eligible_contracts_count": pl.Int64,
            "valid_pairs": pl.Int64,
            "total_pairs": pl.Int64,
            "coverage_ratio": pl.Float64,
            "null_reason": pl.String,
            "ci_status": pl.String,
            "ci_status_2x": pl.String,
            "ci_valid_trading_days": pl.Int64,
            "ci_block_length": pl.Int64,
            "ci_block_length_2x": pl.Int64,
            "ci_effective_blocks": pl.Float64,
            "ci_effective_blocks_2x": pl.Float64,
            "ci_min_effective_blocks": pl.Int64,
            "ci_min_effective_blocks_2x": pl.Int64,
            "ci_null_reason": pl.String,
            "ci_null_reason_2x": pl.String,
        }
        metrics_df = pl.DataFrame(metric_results, schema=metric_schema)
        metrics_df.write_csv(result_dir / "feature_metrics.csv")

        # 9. Generate Matrices and Markdown Report
        generate_matrices(metrics_df, result_dir)

        manifest = {
            "schema_version": self.config.schema_version,
            "protocol_version": self.config.protocol_version,
            "config_fingerprint": self.config.compute_fingerprint(),
            "symbol": self.config.symbol,
            "frequency": self.config.frequency,
            "horizons": self.config.horizons,
            "status": "completed",
            "qualified_features_count": len(qualified_features),
            "total_rows_evaluated": len(all_data),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        if legacy_selection:
            manifest["legacy_selection_source"] = {
                "path": legacy_selection.source_path,
                "sha256": legacy_selection.sha256,
            }

        # Three-Way Matched Common-Sample Horizon Comparisons (1h <-> next_td <-> next_5td)
        matched_comparisons_df: Optional[pl.DataFrame] = None
        target_3way_horizons = ["1h", "next_td", "next_5td"]
        available_horizons = set(labels_df["horizon"].unique().to_list())
        if set(target_3way_horizons).issubset(available_horizons):
            matched_records: List[Dict[str, Any]] = []
            l_1h = labels_df.filter((pl.col("horizon") == "1h") & pl.col("valid_mask")).select([
                "contract", "timestamp", "trading_day",
                pl.col("target_return").alias("ret_1h"),
                pl.col("abs_target_return").alias("abs_ret_1h"),
            ])
            l_1d = labels_df.filter((pl.col("horizon") == "next_td") & pl.col("valid_mask")).select([
                "contract", "timestamp", "trading_day",
                pl.col("target_return").alias("ret_1d"),
                pl.col("abs_target_return").alias("abs_ret_1d"),
            ])
            l_5d = labels_df.filter((pl.col("horizon") == "next_5td") & pl.col("valid_mask")).select([
                "contract", "timestamp", "trading_day",
                pl.col("target_return").alias("ret_5d"),
                pl.col("abs_target_return").alias("abs_ret_5d"),
            ])

            matched_labels = l_1h.join(
                l_1d, on=["contract", "timestamp", "trading_day"], how="inner"
            ).join(
                l_5d, on=["contract", "timestamp", "trading_day"], how="inner"
            )
            matched_df = all_data.join(
                matched_labels, on=["contract", "timestamp", "trading_day"], how="inner"
            )
            n_matched = len(matched_df)
            n_total = len(all_data)
            retention_ratio = float(n_matched / n_total) if n_total > 0 else 0.0

            for target_kind in ["return", "abs_return"]:
                col_1h = "ret_1h" if target_kind == "return" else "abs_ret_1h"
                col_1d = "ret_1d" if target_kind == "return" else "abs_ret_1d"
                col_5d = "ret_5d" if target_kind == "return" else "abs_ret_5d"

                c_corrs_1h = []
                c_corrs_1d = []
                c_corrs_5d = []

                for contract in contracts:
                    c_valid = matched_df.filter(pl.col("contract") == contract)
                    if len(c_valid) >= 3:
                        r_c = c_valid["_row_idx"].to_numpy()
                        X_c = X_full[r_c, :]
                        y_1h_c = c_valid[col_1h].to_numpy()
                        y_1d_c = c_valid[col_1d].to_numpy()
                        y_5d_c = c_valid[col_5d].to_numpy()
                        c_corrs_1h.append(compute_spearman_matrix(X_c, y_1h_c))
                        c_corrs_1d.append(compute_spearman_matrix(X_c, y_1d_c))
                        c_corrs_5d.append(compute_spearman_matrix(X_c, y_5d_c))

                def _avg_contract_corrs(corr_list: List[np.ndarray]) -> np.ndarray:
                    if not corr_list:
                        return np.full(D, np.nan)
                    arr = np.array(corr_list)
                    valid_mask = ~np.isnan(arr)
                    cnt = np.sum(valid_mask, axis=0)
                    return np.where(
                        cnt > 0,
                        np.sum(np.where(valid_mask, arr, 0.0), axis=0) / np.maximum(cnt, 1),
                        np.nan,
                    )

                if len(c_corrs_1h) > 0:
                    arr_1h = _avg_contract_corrs(c_corrs_1h)
                    arr_1d = _avg_contract_corrs(c_corrs_1d)
                    arr_5d = _avg_contract_corrs(c_corrs_5d)
                elif n_matched >= 3:
                    r_m = matched_df["_row_idx"].to_numpy()
                    X_m = X_full[r_m, :]
                    arr_1h = compute_spearman_matrix(X_m, matched_df[col_1h].to_numpy())
                    arr_1d = compute_spearman_matrix(X_m, matched_df[col_1d].to_numpy())
                    arr_5d = compute_spearman_matrix(X_m, matched_df[col_5d].to_numpy())
                else:
                    arr_1h = np.full(D, np.nan)
                    arr_1d = np.full(D, np.nan)
                    arr_5d = np.full(D, np.nan)

                for f_idx, feat_name in enumerate(qualified_features):
                    v_1h = float(arr_1h[f_idx]) if not np.isnan(arr_1h[f_idx]) else None
                    v_1d = float(arr_1d[f_idx]) if not np.isnan(arr_1d[f_idx]) else None
                    v_5d = float(arr_5d[f_idx]) if not np.isnan(arr_5d[f_idx]) else None

                    d_1h_1d = float(v_1d - v_1h) if (v_1h is not None and v_1d is not None) else None
                    d_1d_5d = float(v_5d - v_1d) if (v_1d is not None and v_5d is not None) else None
                    d_1h_5d = float(v_5d - v_1h) if (v_1h is not None and v_5d is not None) else None

                    traj = categorize_three_way_trajectory(v_1h, v_1d, v_5d)

                    matched_records.append({
                        "feature": feat_name,
                        "target_kind": target_kind,
                        "legacy_stream_membership": legacy_stream_map.get(feat_name, "unassigned"),
                        "matched_samples": n_matched,
                        "sample_retention_ratio": retention_ratio,
                        "ic_common_1h": v_1h,
                        "ic_common_1day": v_1d,
                        "ic_common_5day": v_5d,
                        "delta_1h_to_1day": d_1h_1d,
                        "delta_1day_to_5day": d_1d_5d,
                        "delta_1h_to_5day": d_1h_5d,
                        "trajectory_category": traj,
                    })

            matched_schema = {
                "feature": pl.String,
                "target_kind": pl.String,
                "legacy_stream_membership": pl.String,
                "matched_samples": pl.Int64,
                "sample_retention_ratio": pl.Float64,
                "ic_common_1h": pl.Float64,
                "ic_common_1day": pl.Float64,
                "ic_common_5day": pl.Float64,
                "delta_1h_to_1day": pl.Float64,
                "delta_1day_to_5day": pl.Float64,
                "delta_1h_to_5day": pl.Float64,
                "trajectory_category": pl.String,
            }
            matched_comparisons_df = pl.DataFrame(matched_records, schema=matched_schema)
            matched_comparisons_df.write_csv(result_dir / "matched_horizon_comparisons.csv")

        if matched_comparisons_df is not None:
            manifest["matched_horizon_status"] = "COMPLETED_MATCHED_EVALUATION"
            manifest["matched_samples"] = n_matched
            manifest["sample_retention_ratio"] = retention_ratio
        else:
            manifest["matched_horizon_status"] = "NOT_AVAILABLE"

        manifest["ci_diagnostics_status"] = "completed"

        generate_audit_report(
            manifest=manifest,
            metrics_df=metrics_df,
            invalid_summary_df=invalid_summary,
            output_path=result_dir / "audit_report.md",
            ci_diagnostics_df=ci_diag_df,
            matched_comparisons_df=matched_comparisons_df,
        )

        with open(result_dir / "audit_run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        (result_dir / "_SUCCESS").touch()
        return manifest
