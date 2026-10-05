from __future__ import annotations

import numpy as np
import polars as pl
import torch


METRIC_COLUMNS = [
    "Permutation Importance",
    "CatBoost Importance",
    "IC",
    "RankIC",
    "VolIC",
    "VolRankIC",
    "Sharpe",
]
DEFAULT_WINDOWS_LIST = [1, 2, 6, 12, 24, 48]


def calculate_ic(column, target) -> float:
    column = np.asarray(column, dtype=float)
    target = np.asarray(target, dtype=float)
    valid = ~(np.isnan(column) | np.isnan(target))
    column = column[valid]
    target = target[valid]
    if (
        column.size < 2
        or target.size < 2
        or np.std(column) == 0
        or np.std(target) == 0
    ):
        return np.nan
    return float(np.corrcoef(column, target)[0, 1])


def calculate_rank_ic(column, target) -> float:
    column = np.asarray(column, dtype=float)
    target = np.asarray(target, dtype=float)
    if (
        column.size == 0
        or target.size == 0
        or np.nanstd(column) == 0
        or np.nanstd(target) == 0
    ):
        return 0.0
    value = np.corrcoef(
        np.argsort(np.argsort(column)),
        np.argsort(np.argsort(target)),
    )[0, 1]
    return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))


def calculate_future_return(df: pl.DataFrame, window: int = 1) -> np.ndarray:
    if "timestamp" in df.columns:
        df = df.sort("timestamp")
    diff = pl.col("mark_price").shift(-window) - pl.col("mark_price")
    denom = pl.when(pl.col("mark_price").abs() > 1e-8).then(pl.col("mark_price")).otherwise(None)
    target = df.select((diff / denom).alias("target"))["target"]
    return target.slice(0, max(target.len() - window, 0)).to_numpy()


def calculate_sharpe(feature_values, future_return) -> float:
    feature_values = np.asarray(feature_values, dtype=float)
    future_return = np.asarray(future_return, dtype=float)
    size = min(feature_values.size, future_return.size)
    feature_values = feature_values[:size]
    future_return = future_return[:size]
    valid = ~(np.isnan(feature_values) | np.isnan(future_return))
    feature_values = feature_values[valid]
    future_return = future_return[valid]
    if feature_values.size < 2 or np.std(feature_values) == 0:
        return 0.0
    zscore = (feature_values - feature_values.mean()) / feature_values.std(ddof=0)
    pseudo_returns = zscore * future_return
    std = pseudo_returns.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(pseudo_returns.mean() / std)


def _permutation_importance(feature_values, future_return, seed: int = 42) -> float:
    baseline = abs(calculate_ic(feature_values, future_return))
    if np.isnan(baseline):
        baseline = 0.0
    shuffled = np.asarray(feature_values, dtype=float).copy()
    if shuffled.size:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(shuffled)
    shuffled_score = abs(calculate_ic(shuffled, future_return))
    if np.isnan(shuffled_score):
        shuffled_score = 0.0
    return float(max(baseline - shuffled_score, 0.0))


def _catboost_importance(
    df: pl.DataFrame,
    features: list[str],
    future_return: np.ndarray,
    window_length: int = 1,
    iterations: int = 1000,
    early_stopping_rounds: int = 30,
    depth: int = 6,
    random_seed: int = 42,
) -> dict[str, float]:
    if not features:
        return {}
    from catboost import CatBoostRegressor, Pool

    model_df = df.slice(0, future_return.size)
    x = model_df.select([pl.col(f).cast(pl.Float64, strict=False).fill_null(0.0).fill_nan(0.0) for f in features]).to_numpy()
    y = np.asarray(future_return, dtype=float)
    n_samples = len(x)
    if n_samples >= 10:
        split_idx = int(n_samples * 0.8)
        purge_window = max(window_length, 0)
        train_end = split_idx - purge_window
        if train_end < 5:
            train_end = split_idx
        train_pool = Pool(x[:train_end], y[:train_end])
        eval_pool = Pool(x[split_idx:], y[split_idx:])
        fit_pool = train_pool
    else:
        train_pool = Pool(x, y)
        eval_pool = Pool(x, y)
        fit_pool = train_pool
    try:
        model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=0.1,
            depth=depth,
            loss_function="MAE",
            task_type="GPU",
            random_seed=random_seed,
            early_stopping_rounds=early_stopping_rounds,
        )
        model.fit(train_pool, eval_set=eval_pool, verbose=100)
    except Exception:
        model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=0.1,
            depth=depth,
            loss_function="MAE",
            task_type="CPU",
            random_seed=random_seed,
            early_stopping_rounds=early_stopping_rounds,
        )
        model.fit(train_pool, eval_set=eval_pool, verbose=100)
    values = model.get_feature_importance(fit_pool)
    return {feature: float(value) for feature, value in zip(features, values)}


def calculate_metric_frame(
    df: pl.DataFrame,
    features: list[str],
    *,
    window: int | None = None,
    windows_list: list[int] | None = None,
    compute_catboost: bool = True,
) -> pl.DataFrame:
    if windows_list is None:
        windows_list = DEFAULT_WINDOWS_LIST if window is None else [window]
    if not features:
        raise ValueError("features list is empty; cannot calculate feature metrics")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_raw = np.ascontiguousarray(
        df.select(features).cast(pl.Float64, strict=False).to_numpy()
    )
    x_full = torch.tensor(x_raw, dtype=torch.float32, device=device)
    x_full = torch.nan_to_num(x_full, nan=0.0, posinf=0.0, neginf=0.0)
    d_feat = len(features)

    window_frames: list[pl.DataFrame] = []
    for window_length in windows_list:
        future_return = calculate_future_return(df, window_length)
        if future_return.size < 2:
            continue
        n_w = future_return.size
        x_w = x_full[:n_w]
        y_w = torch.tensor(future_return, dtype=torch.float32, device=device)
        y_w = torch.nan_to_num(y_w, nan=0.0, posinf=0.0, neginf=0.0)
        y_vol = torch.abs(y_w)

        # Standardize X
        x_mean = x_w.mean(dim=0, keepdim=True)
        x_std = x_w.std(dim=0, unbiased=False, keepdim=True)
        x_norm = torch.where(x_std > 1e-8, (x_w - x_mean) / x_std, torch.zeros_like(x_w))

        # Standardize Y
        y_mean = y_w.mean()
        y_std = y_w.std(unbiased=False)
        y_norm = torch.where(y_std > 1e-8, (y_w - y_mean) / y_std, torch.zeros_like(y_w))

        # Standardize Y_vol
        y_vol_mean = y_vol.mean()
        y_vol_std = y_vol.std(unbiased=False)
        y_vol_norm = torch.where(y_vol_std > 1e-8, (y_vol - y_vol_mean) / y_vol_std, torch.zeros_like(y_vol))

        ic = torch.mv(x_norm.T, y_norm) / n_w
        vol_ic = torch.mv(x_norm.T, y_vol_norm) / n_w

        # Ranks
        rx = torch.argsort(torch.argsort(x_w, dim=0), dim=0).to(torch.float32)
        rx_mean = rx.mean(dim=0, keepdim=True)
        rx_std = rx.std(dim=0, unbiased=False, keepdim=True)
        rx_norm = torch.where(rx_std > 1e-8, (rx - rx_mean) / rx_std, torch.zeros_like(rx))

        ry = torch.argsort(torch.argsort(y_w)).to(torch.float32)
        ry_mean = ry.mean()
        ry_std = ry.std(unbiased=False)
        ry_norm = torch.where(ry_std > 1e-8, (ry - ry_mean) / ry_std, torch.zeros_like(ry))

        ry_vol = torch.argsort(torch.argsort(y_vol)).to(torch.float32)
        ry_vol_mean = ry_vol.mean()
        ry_vol_std = ry_vol.std(unbiased=False)
        ry_vol_norm = torch.where(ry_vol_std > 1e-8, (ry_vol - ry_vol_mean) / ry_vol_std, torch.zeros_like(ry_vol))

        rank_ic = torch.mv(rx_norm.T, ry_norm) / n_w
        vol_rank_ic = torch.mv(rx_norm.T, ry_vol_norm) / n_w

        pseudo = x_norm * y_w.unsqueeze(1)
        p_mean = pseudo.mean(dim=0)
        p_std = pseudo.std(dim=0, unbiased=True)
        sharpe = torch.where(p_std > 1e-8, p_mean / p_std, torch.zeros_like(p_mean))

        if compute_catboost:
            catboost_values = _catboost_importance(
                df, features, future_return, window_length=window_length
            )
            catboost_col = [catboost_values.get(f, 0.0) for f in features]
        else:
            catboost_col = [0.0] * d_feat

        window_frames.append(
            pl.DataFrame(
                {
                    "feature": features,
                    "window": [window_length] * d_feat,
                    "Permutation Importance": [0.0] * d_feat,
                    "CatBoost Importance": catboost_col,
                    "IC": ic.cpu().numpy().astype(np.float64),
                    "RankIC": rank_ic.cpu().numpy().astype(np.float64),
                    "VolIC": vol_ic.cpu().numpy().astype(np.float64),
                    "VolRankIC": vol_rank_ic.cpu().numpy().astype(np.float64),
                    "Sharpe": sharpe.cpu().numpy().astype(np.float64),
                }
            )
        )

    if not window_frames:
        raise ValueError("future return is empty; cannot calculate feature metrics")
    return pl.concat(window_frames, how="vertical")


def aggregate_metric_frames(frames: list[pl.DataFrame]) -> pl.DataFrame:
    if not frames:
        raise ValueError("cannot aggregate empty metric frame list")
    combined = pl.concat(frames, how="vertical")
    metric_columns = [column for column in METRIC_COLUMNS if column in combined.columns]
    expressions = []
    for metric in metric_columns:
        expressions.extend(
            [
                pl.col(metric).mean().alias(f"{metric}_Mean"),
                pl.col(metric).std().fill_null(0.0).alias(f"{metric}_Std"),
                pl.col(metric).median().alias(f"{metric}_Median"),
            ]
        )
    agg = combined.group_by("feature", maintain_order=True).agg(expressions)
    if "RankIC" in combined.columns and "window" in combined.columns:
        num_contracts = float(len(frames))
        window_sign_df = (
            combined.group_by(["feature", "window"])
            .agg(
                [
                    (pl.col("RankIC") > 0.0).sum().alias("pos_cnt"),
                    (pl.col("RankIC") < 0.0).sum().alias("neg_cnt"),
                ]
            )
            .with_columns(
                (pl.max_horizontal("pos_cnt", "neg_cnt") / num_contracts).alias("SignConsistency")
            )
            .group_by("feature")
            .agg(
                [
                    pl.col("SignConsistency").mean().alias("SignConsistency_Mean"),
                    pl.col("SignConsistency").min().alias("SignConsistency_Min"),
                ]
            )
        )
        agg = agg.join(window_sign_df, on="feature", how="left")
    if "VolRankIC" in combined.columns and "window" in combined.columns:
        num_contracts = float(len(frames))
        vol_window_sign_df = (
            combined.group_by(["feature", "window"])
            .agg(
                [
                    (pl.col("VolRankIC") > 0.0).sum().alias("vol_pos_cnt"),
                    (pl.col("VolRankIC") < 0.0).sum().alias("vol_neg_cnt"),
                ]
            )
            .with_columns(
                (pl.max_horizontal("vol_pos_cnt", "vol_neg_cnt") / num_contracts).alias("VolSignConsistency")
            )
            .group_by("feature")
            .agg(
                [
                    pl.col("VolSignConsistency").mean().alias("VolSignConsistency_Mean"),
                    pl.col("VolSignConsistency").min().alias("VolSignConsistency_Min"),
                ]
            )
        )
        agg = agg.join(vol_window_sign_df, on="feature", how="left")
    return agg
