from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import polars as pl
from scipy.stats import t as student_t
import torch

from operator_futures.feature_selection.muti_contract.metrics import calculate_future_return

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


def default_target_regime_bins(
    num_slope_bins: int = 4, num_vol_bins: int = 4
) -> list[tuple[int, int]]:
    """Return default target extreme bins (low volatility, strong directional slopes)."""
    if num_slope_bins == 4 and num_vol_bins == 4:
        return list(TARGET_REGIME_BINS)
    elif num_slope_bins == 3 and num_vol_bins == 3:
        return [(0, 0), (2, 0)]
    else:
        return [(0, 0), (num_slope_bins - 1, 0)]


def extract_slope_and_volatility(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract 48-bar slope and 48-bar return volatility from contract frame."""
    if "close" in df.columns:
        close = df["close"].to_numpy().astype(float)
    elif "mark_price" in df.columns:
        close = df["mark_price"].to_numpy().astype(float)
    else:
        close = np.zeros(df.height, dtype=float)
    row_count = len(close)

    if "log_price_slope_48" in df.columns:
        slope = df["log_price_slope_48"].to_numpy().astype(float)
    else:
        log_prices = np.log(np.maximum(close, 1e-8))
        slope = np.zeros(row_count, dtype=float)
        w = 48 if row_count >= 48 else max(row_count // 2, 2)
        if row_count >= w and w >= 2:
            steps = np.arange(w, dtype=float) - (w - 1) / 2.0
            sum_sq = float(np.square(steps).sum())
            rolling_log_prices = np.lib.stride_tricks.sliding_window_view(log_prices, w)
            slopes = rolling_log_prices @ steps / sum_sq
            slope[w - 1:] = slopes

    log_returns = np.diff(np.log(np.maximum(close, 1e-8)))
    volatility = np.zeros(row_count, dtype=float)
    w_vol = 47 if row_count >= 48 else max(len(log_returns) // 2, 2)
    if len(log_returns) >= w_vol and w_vol >= 2:
        rolling_returns = np.lib.stride_tricks.sliding_window_view(log_returns, w_vol)
        volatility[w_vol:] = rolling_returns.std(axis=1, ddof=0)

    return slope, volatility


def compute_regime_quantiles(
    frames: dict[str, pl.DataFrame],
    num_bins: int = 4,
    quantiles: Sequence[float] | None = None,
) -> dict[str, list[float]]:
    """Compute quantile thresholds for slope and volatility on mature rows (step >= 47)."""
    if num_bins < 2 and quantiles is None:
        raise ValueError(f"num_bins must be >= 2, got {num_bins}")

    if quantiles is None:
        q_points = [float(i / num_bins) for i in range(1, num_bins)]
    else:
        q_points = [float(q) for q in quantiles]

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
            "slope": [0.0] * len(q_points),
            "volatility": [0.0] * len(q_points),
        }

    cat_slopes = np.concatenate(all_slopes)
    cat_vols = np.concatenate(all_vols)

    q_slope = np.quantile(cat_slopes, q_points).tolist()
    q_vol = np.quantile(cat_vols, q_points).tolist()

    return {
        "slope": [float(x) for x in q_slope],
        "volatility": [float(x) for x in q_vol],
    }


def assign_regime_bin(val: float, thresholds: Sequence[float]) -> int:
    """Map value to bin index 0..len(thresholds) given quantile thresholds."""
    for i, th in enumerate(thresholds):
        if val < th:
            return i
    return len(thresholds)


def audit_regimes(
    frames: dict[str, pl.DataFrame],
    feature_universe: list[str],
    quantiles: dict[str, list[float]],
    windows_list: list[int],
    *,
    num_slope_bins: int | None = None,
    num_vol_bins: int | None = None,
    target_regime_bins: Sequence[tuple[int, int]] | None = None,
    min_abs_ic: float = 0.01,
    enable_conditional_anchors: bool = True,
) -> tuple[pl.DataFrame, list[str], list[dict[str, Any]]]:
    """
    Perform multi-bin (slope 0..N-1 x vol 0..M-1) market state audit.
    Evaluates conditional retention for market state anchors in target bins.
    Returns (audit_dataframe, conditionally_retained_anchors, retention_details).
    """
    slope_th = quantiles["slope"]
    vol_th = quantiles["volatility"]

    if num_slope_bins is None:
        num_slope_bins = len(slope_th) + 1
    if num_vol_bins is None:
        num_vol_bins = len(vol_th) + 1

    if target_regime_bins is None:
        target_bins = default_target_regime_bins(num_slope_bins, num_vol_bins)
    else:
        target_bins = list(target_regime_bins)

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
        sb_arr = np.digitize(slope[47:], slope_th, right=False)
        vb_arr = np.digitize(vol[47:], vol_th, right=False)
        for s_bin in range(num_slope_bins):
            for v_bin in range(num_vol_bins):
                mask = np.zeros(frame.height, dtype=bool)
                mask[47:] = (sb_arr == s_bin) & (vb_arr == v_bin)
                masks[(s_bin, v_bin)] = mask

        contract_bin_masks[contract] = masks

    audit_rows: list[dict[str, Any]] = []
    # Collect contract-level rank ICs for conditional retention check:
    # key: (feature, window, s_bin, v_bin) -> list of (contract, step_count, rank_ic)
    bin_contract_rank_ics: dict[tuple[str, int, int, int], list[tuple[str, int, float]]] = {}
    future_returns: dict[tuple[str, int], np.ndarray] = {
        (contract, window): calculate_future_return(frame, window)
        for contract, frame in frames.items()
        for window in windows_list
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_feat = len(feature_universe)
    contract_results: dict[
        tuple[int, int, int],
        list[tuple[str, int, np.ndarray, np.ndarray, np.ndarray]],
    ] = {}

    for contract, frame in frames.items():
        if frame.height < 48 or d_feat == 0 or contract not in contract_bin_masks:
            continue
        c_masks = contract_bin_masks[contract]
        if not c_masks:
            continue

        x_raw = np.ascontiguousarray(
            frame.select(feature_universe).cast(pl.Float64, strict=False).to_numpy()
        )
        x_mat = torch.tensor(x_raw, dtype=torch.float32, device=device)
        x_mat = torch.nan_to_num(x_mat, nan=0.0, posinf=0.0, neginf=0.0)

        for window in windows_list:
            future_ret = future_returns[(contract, window)]
            min_len = min(len(future_ret), frame.height)
            if min_len < 2:
                continue

            x_win = x_mat[:min_len]
            y_win = torch.tensor(future_ret[:min_len], dtype=torch.float32, device=device)

            for s_bin in range(num_slope_bins):
                for v_bin in range(num_vol_bins):
                    bin_key = (s_bin, v_bin)
                    mask = c_masks.get(bin_key)
                    if mask is None:
                        continue
                    sub_mask = mask[:min_len]
                    step_count = int(np.count_nonzero(sub_mask))
                    if step_count < 2:
                        continue

                    idx = torch.tensor(
                        np.nonzero(sub_mask)[0], dtype=torch.long, device=device
                    )
                    x_sub = x_win[idx]
                    y_sub = y_win[idx]

                    y_std = y_sub.std(unbiased=False)
                    if y_std <= 1e-8 or torch.isnan(y_std):
                        continue

                    x_std = x_sub.std(dim=0, unbiased=False, keepdim=True)
                    valid_x = (x_std > 1e-8).squeeze(0)
                    if not valid_x.any():
                        continue

                    x_mean = x_sub.mean(dim=0, keepdim=True)
                    x_norm = torch.where(
                        x_std > 1e-8,
                        (x_sub - x_mean) / x_std,
                        torch.zeros_like(x_sub),
                    )
                    y_norm = (y_sub - y_sub.mean()) / y_std

                    ic_vec = torch.mv(x_norm.T, y_norm) / step_count

                    rx = torch.argsort(torch.argsort(x_sub, dim=0), dim=0).to(torch.float32)
                    rx_mean = rx.mean(dim=0, keepdim=True)
                    rx_std = rx.std(dim=0, unbiased=False, keepdim=True)
                    rx_norm = torch.where(
                        rx_std > 1e-8,
                        (rx - rx_mean) / rx_std,
                        torch.zeros_like(rx),
                    )

                    ry = torch.argsort(torch.argsort(y_sub)).to(torch.float32)
                    ry_std = ry.std(unbiased=False)
                    if ry_std <= 1e-8 or torch.isnan(ry_std):
                        continue
                    ry_norm = (ry - ry.mean()) / ry_std

                    rank_ic_vec = torch.mv(rx_norm.T, ry_norm) / step_count

                    ic_np = ic_vec.cpu().numpy().astype(np.float64)
                    rank_ic_np = rank_ic_vec.cpu().numpy().astype(np.float64)
                    valid_np = valid_x.cpu().numpy()

                    res_key = (s_bin, v_bin, window)
                    if res_key not in contract_results:
                        contract_results[res_key] = []
                    contract_results[res_key].append(
                        (contract, step_count, ic_np, rank_ic_np, valid_np)
                    )

    for s_bin in range(num_slope_bins):
        for v_bin in range(num_vol_bins):
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
                c_data = contract_results.get((s_bin, v_bin, window), [])

                for f_idx, feature in enumerate(feature_universe):
                    contract_ics: list[float] = []
                    contract_rank_ics: list[float] = []

                    for contract, step_cnt, ic_np, rank_ic_np, valid_np in c_data:
                        if valid_np[f_idx]:
                            ic_val = float(ic_np[f_idx])
                            rank_ic_val = float(rank_ic_np[f_idx])
                            if np.isfinite(ic_val):
                                contract_ics.append(ic_val)
                            if np.isfinite(rank_ic_val):
                                contract_rank_ics.append(rank_ic_val)
                                key = (feature, window, s_bin, v_bin)
                                if key not in bin_contract_rank_ics:
                                    bin_contract_rank_ics[key] = []
                                bin_contract_rank_ics[key].append(
                                    (contract, step_cnt, rank_ic_val)
                                )

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

    extreme_bins = set(target_bins)
    if num_slope_bins > 2:
        neutral_bins = {
            (s, v) for s in range(1, num_slope_bins - 1) for v in range(num_vol_bins)
        }
    else:
        neutral_bins = {
            (s, v)
            for s in range(num_slope_bins)
            for v in range(num_vol_bins)
            if (s, v) not in extreme_bins
        }

    if enable_conditional_anchors:
        for anchor in MARKET_STATE_ANCHOR_COLUMNS:
            if anchor not in feature_universe:
                continue

            # Compute cross-regime variance ratio for this anchor (extreme bins vs neutral bins)
            extreme_vals: list[np.ndarray] = []
            neutral_vals: list[np.ndarray] = []

            for contract, frame in frames.items():
                if frame.height < 48 or anchor not in frame.columns:
                    continue
                masks = contract_bin_masks[contract]
                if not masks:
                    continue

                arr = frame[anchor].to_numpy()
                e_mask = np.zeros(frame.height, dtype=bool)
                for eb in extreme_bins:
                    if eb in masks:
                        e_mask |= masks[eb]

                n_mask = np.zeros(frame.height, dtype=bool)
                for nb in neutral_bins:
                    if nb in masks:
                        n_mask |= masks[nb]

                e_sub = arr[e_mask]
                n_sub = arr[n_mask]
                if len(e_sub) > 0:
                    extreme_vals.append(e_sub)
                if len(n_sub) > 0:
                    neutral_vals.append(n_sub)

            cat_extreme = np.concatenate(extreme_vals) if extreme_vals else np.array([])
            cat_neutral = np.concatenate(neutral_vals) if neutral_vals else np.array([])

            var_extreme = float(np.var(cat_extreme, ddof=1)) if len(cat_extreme) > 1 else 0.0
            var_neutral = float(np.var(cat_neutral, ddof=1)) if len(cat_neutral) > 1 else 0.0

            if var_neutral > 0.0:
                variance_ratio = float(var_extreme / var_neutral)
            elif var_extreme == 0.0:
                variance_ratio = 1.0
            else:
                variance_ratio = float("inf")

            passed = False
            passing_bins: list[str] = []

            for (s_bin, v_bin) in target_bins:
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
                        bin_str = f"slope{s_bin}_vol{v_bin}_w{window}"
                        is_variance_stable = variance_ratio <= 3.0
                        if is_variance_stable:
                            passed = True
                            passing_bins.append(bin_str)

                        retention_details.append({
                            "feature": anchor,
                            "target_bin": bin_str,
                            "rank_ic_90_lcb": lcb,
                            "sign_consistency": sign_consistency,
                            "participating_contracts": len(valid_contracts),
                            "variance_ratio": round(variance_ratio, 4),
                            "retained": is_variance_stable,
                        })

            if passed and anchor not in retained_anchors:
                retained_anchors.append(anchor)

    return audit_df, retained_anchors, retention_details


# Alias for backward compatibility
audit_16_regimes = audit_regimes
