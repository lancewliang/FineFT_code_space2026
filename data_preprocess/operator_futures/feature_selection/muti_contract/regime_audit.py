from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import polars as pl
from scipy.stats import t as student_t

from operator_futures.feature_selection.muti_contract.metrics import calculate_future_return, calculate_ic, calculate_rank_ic

MARKET_STATE_ANCHOR_COLUMNS = [
    "log_price_slope_48",
    "log_price_slope_96",
    "trend_to_noise_48",
    "trend_to_noise_96",
    "signed_efficiency_48",
    "trend_r2_48",
    "log_return_vol_quantile_192",
]

TARGET_REGIME_BINS = [
    (0, 0),  # slope 0, vol 0
    (3, 0),  # slope 3, vol 0
    (0, 1),  # slope 0, vol 1
    (3, 1),  # slope 3, vol 1
]


def extract_slope_and_volatility(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract 48-bar slope and 48-bar return volatility from contract frame."""
    close = df["close"].to_numpy().astype(float)
    row_count = len(close)

    if "log_price_slope_48" in df.columns:
        slope = df["log_price_slope_48"].to_numpy().astype(float)
    else:
        log_prices = np.log(close)
        slope = np.zeros(row_count, dtype=float)
        if row_count >= 48:
            steps = np.arange(48, dtype=float) - 23.5
            sum_sq = float(np.square(steps).sum())
            rolling_log_prices = np.lib.stride_tricks.sliding_window_view(log_prices, 48)
            slopes = rolling_log_prices @ steps / sum_sq
            slope[47:] = slopes

    log_returns = np.diff(np.log(close))
    volatility = np.zeros(row_count, dtype=float)
    if row_count >= 48:
        rolling_returns = np.lib.stride_tricks.sliding_window_view(log_returns, 47)
        volatility[47:] = rolling_returns.std(axis=1, ddof=0)

    return slope, volatility


def compute_regime_quantiles(frames: dict[str, pl.DataFrame]) -> dict[str, list[float]]:
    """Compute 25%, 50%, 75% quartiles for slope and volatility on mature rows (step >= 47)."""
    all_slopes = []
    all_vols = []

    for frame in frames.values():
        if frame.height < 48:
            continue
        slope, vol = extract_slope_and_volatility(frame)
        all_slopes.append(slope[47:])
        all_vols.append(vol[47:])

    if not all_slopes:
        return {
            "slope": [0.0, 0.0, 0.0],
            "volatility": [0.0, 0.0, 0.0],
        }

    cat_slopes = np.concatenate(all_slopes)
    cat_vols = np.concatenate(all_vols)

    q_slope = np.quantile(cat_slopes, [0.25, 0.50, 0.75]).tolist()
    q_vol = np.quantile(cat_vols, [0.25, 0.50, 0.75]).tolist()

    return {
        "slope": [float(x) for x in q_slope],
        "volatility": [float(x) for x in q_vol],
    }


def assign_regime_bin(val: float, thresholds: list[float]) -> int:
    """Map value to bin index 0..3 given 3 quantile thresholds."""
    if val < thresholds[0]:
        return 0
    elif val < thresholds[1]:
        return 1
    elif val < thresholds[2]:
        return 2
    else:
        return 3


def audit_16_regimes(
    frames: dict[str, pl.DataFrame],
    feature_universe: list[str],
    quantiles: dict[str, list[float]],
    windows_list: list[int],
    *,
    min_abs_ic: float = 0.01,
    enable_conditional_anchors: bool = True,
) -> tuple[pl.DataFrame, list[str], list[dict[str, Any]]]:
    """
    Perform 16-bin (slope 0..3 x vol 0..3) market state audit.
    Evaluates conditional retention for market state anchors in target bins.
    Returns (audit_dataframe, conditionally_retained_anchors, retention_details).
    """
    slope_th = quantiles["slope"]
    vol_th = quantiles["volatility"]

    # Pre-segment contracts by mature rows & bin assignment
    contract_bin_masks: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    contract_total_mature_steps: dict[str, int] = {}
    total_mature_steps = 0

    for contract, frame in frames.items():
        if frame.height < 48:
            contract_total_mature_steps[contract] = 0
            contract_bin_masks[contract] = {}
            continue

        slope, vol = extract_slope_and_volatility(frame)
        mature_count = frame.height - 47
        contract_total_mature_steps[contract] = mature_count
        total_mature_steps += mature_count

        masks: dict[tuple[int, int], np.ndarray] = {}
        for s_bin in range(4):
            for v_bin in range(4):
                # Mature row condition
                mask = np.zeros(frame.height, dtype=bool)
                for i in range(47, frame.height):
                    sb = assign_regime_bin(slope[i], slope_th)
                    vb = assign_regime_bin(vol[i], vol_th)
                    if sb == s_bin and vb == v_bin:
                        mask[i] = True
                masks[(s_bin, v_bin)] = mask

        contract_bin_masks[contract] = masks

    audit_rows: list[dict[str, Any]] = []
    # Collect contract-level rank ICs for conditional retention check:
    # key: (feature, window, s_bin, v_bin) -> list of (contract, step_count, rank_ic)
    bin_contract_rank_ics: dict[tuple[str, int, int, int], list[tuple[str, int, float]]] = {}

    for s_bin in range(4):
        for v_bin in range(4):
            bin_key = (s_bin, v_bin)

            # Step count across contracts in this bin
            bin_step_count = 0
            participating_contracts = 0
            contract_step_counts: dict[str, int] = {}

            for contract, masks in contract_bin_masks.items():
                if bin_key in masks:
                    c_steps = int(np.count_nonzero(masks[bin_key]))
                    contract_step_counts[contract] = c_steps
                    if c_steps > 0:
                        bin_step_count += c_steps
                        participating_contracts += 1

            bin_step_ratio = (
                float(bin_step_count) / float(total_mature_steps)
                if total_mature_steps > 0
                else 0.0
            )

            for window in windows_list:
                for feature in feature_universe:
                    # Check warm-up window requirement for 96/192 anchors
                    if feature in ("log_price_slope_96", "trend_to_noise_96") and window < 1:
                        pass

                    contract_ics: list[float] = []
                    contract_rank_ics: list[float] = []

                    for contract, frame in frames.items():
                        mask = contract_bin_masks[contract].get(bin_key)
                        if mask is None or not mask.any():
                            continue

                        future_ret = calculate_future_return(frame, window)
                        min_len = min(len(future_ret), len(mask))
                        if min_len == 0:
                            continue

                        sub_mask = mask[:min_len]
                        if not sub_mask.any():
                            continue

                        feat_vals = frame[feature].slice(0, min_len).to_numpy()[sub_mask]
                        ret_vals = future_ret[sub_mask]

                        valid_pair = ~(np.isnan(feat_vals) | np.isnan(ret_vals))
                        if np.count_nonzero(valid_pair) < 2:
                            continue

                        ic_val = calculate_ic(feat_vals[valid_pair], ret_vals[valid_pair])
                        rank_ic_val = calculate_rank_ic(feat_vals[valid_pair], ret_vals[valid_pair])

                        if np.isfinite(ic_val):
                            contract_ics.append(ic_val)
                        if np.isfinite(rank_ic_val):
                            contract_rank_ics.append(rank_ic_val)
                            valid_step_cnt = int(np.count_nonzero(valid_pair))
                            key = (feature, window, s_bin, v_bin)
                            if key not in bin_contract_rank_ics:
                                bin_contract_rank_ics[key] = []
                            bin_contract_rank_ics[key].append((contract, valid_step_cnt, rank_ic_val))

                    ic_mean = float(np.mean(contract_ics)) if contract_ics else None
                    ic_std = float(np.std(contract_ics, ddof=1)) if len(contract_ics) > 1 else (0.0 if contract_ics else None)
                    ic_median = float(np.median(contract_ics)) if contract_ics else None

                    rank_ic_mean = float(np.mean(contract_rank_ics)) if contract_rank_ics else None
                    rank_ic_std = float(np.std(contract_rank_ics, ddof=1)) if len(contract_rank_ics) > 1 else (0.0 if contract_rank_ics else None)
                    rank_ic_median = float(np.median(contract_rank_ics)) if contract_rank_ics else None

                    # Sign consistency & 90% LCB
                    if contract_rank_ics:
                        non_zero = [r for r in contract_rank_ics if r != 0.0]
                        if non_zero:
                            majority_sign = 1.0 if sum(np.sign(non_zero)) >= 0 else -1.0
                            same_sign_count = sum(1 for r in non_zero if np.sign(r) == majority_sign)
                            sign_consistency = float(same_sign_count) / float(len(non_zero))

                            aligned_rank_ics = [r * majority_sign for r in contract_rank_ics]
                            a_mean = float(np.mean(aligned_rank_ics))
                            a_len = len(aligned_rank_ics)
                            if a_len >= 3:
                                a_std = float(np.std(aligned_rank_ics, ddof=1))
                                se = a_std / np.sqrt(a_len)
                                t_crit = student_t.ppf(0.90, df=a_len - 1)
                                rank_ic_90_lcb = float(a_mean - t_crit * se)
                            else:
                                rank_ic_90_lcb = a_mean
                        else:
                            sign_consistency = 0.0
                            rank_ic_90_lcb = 0.0
                    else:
                        sign_consistency = None
                        rank_ic_90_lcb = None

                    audit_rows.append({
                        "slope_bin": s_bin,
                        "vol_bin": v_bin,
                        "feature": feature,
                        "window": window,
                        "step_count": bin_step_count,
                        "step_ratio": bin_step_ratio,
                        "contract_count": participating_contracts,
                        "IC_Mean": ic_mean,
                        "IC_Std": ic_std,
                        "IC_Median": ic_median,
                        "RankIC_Mean": rank_ic_mean,
                        "RankIC_Std": rank_ic_std,
                        "RankIC_Median": rank_ic_median,
                        "RankIC_Sign_Consistency": sign_consistency,
                        "RankIC_90_LCB": rank_ic_90_lcb,
                    })

    audit_df = pl.DataFrame(audit_rows)

    # Evaluate conditional retention for anchors
    retained_anchors: list[str] = []
    retention_details: list[dict[str, Any]] = []

    if enable_conditional_anchors:
        for anchor in MARKET_STATE_ANCHOR_COLUMNS:
            if anchor not in feature_universe:
                continue

            passed = False
            passing_bins: list[str] = []

            for (s_bin, v_bin) in TARGET_REGIME_BINS:
                # Check for any window length
                for window in windows_list:
                    key = (anchor, window, s_bin, v_bin)
                    records = bin_contract_rank_ics.get(key, [])

                    # Require at least 3 contracts each having at least 30 aligned steps
                    valid_contracts = [
                        (c, cnt, r) for (c, cnt, r) in records if cnt >= 30
                    ]
                    if len(valid_contracts) < 3:
                        continue

                    rank_ics = [r for (_, _, r) in valid_contracts]
                    non_zero = [r for r in rank_ics if r != 0.0]
                    if not non_zero:
                        continue

                    majority_sign = 1.0 if sum(np.sign(non_zero)) >= 0 else -1.0
                    same_sign_count = sum(1 for r in non_zero if np.sign(r) == majority_sign)
                    sign_consistency = float(same_sign_count) / float(len(non_zero))

                    if sign_consistency < 0.60:
                        continue

                    aligned_rank_ics = [r * majority_sign for r in rank_ics]
                    a_mean = float(np.mean(aligned_rank_ics))
                    a_len = len(aligned_rank_ics)
                    a_std = float(np.std(aligned_rank_ics, ddof=1)) if a_len > 1 else 0.0
                    se = a_std / np.sqrt(a_len)
                    t_crit = student_t.ppf(0.90, df=a_len - 1)
                    lcb = float(a_mean - t_crit * se)

                    if lcb >= min_abs_ic:
                        passed = True
                        bin_str = f"slope{s_bin}_vol{v_bin}_w{window}"
                        passing_bins.append(bin_str)
                        retention_details.append({
                            "feature": anchor,
                            "target_bin": bin_str,
                            "rank_ic_90_lcb": lcb,
                            "sign_consistency": sign_consistency,
                            "participating_contracts": len(valid_contracts),
                        })

            if passed and anchor not in retained_anchors:
                retained_anchors.append(anchor)

    return audit_df, retained_anchors, retention_details
