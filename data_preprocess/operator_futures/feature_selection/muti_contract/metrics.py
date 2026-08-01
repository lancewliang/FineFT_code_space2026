from __future__ import annotations

import numpy as np
import polars as pl


METRIC_COLUMNS = [
    "Permutation Importance",
    "CatBoost Importance",
    "IC",
    "RankIC",
    "Sharpe",
]
DEFAULT_WINDOWS_LIST = [1, 6, 12]


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
    df: pl.DataFrame, features: list[str], future_return: np.ndarray
) -> dict[str, float]:
    if not features:
        return {}
    from catboost import CatBoostRegressor, Pool

    model_df = df.slice(0, future_return.size)
    x = model_df.select(features).to_numpy()
    y = np.asarray(future_return, dtype=float)
    train_pool = Pool(x, y)
    test_pool = Pool(x, y)
    try:
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            loss_function="MAE",
            task_type="GPU",
            random_seed=42,
        )
        model.fit(train_pool, eval_set=test_pool, verbose=100)
    except Exception:
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            loss_function="MAE",
            task_type="CPU",
            random_seed=42,
        )
        model.fit(train_pool, eval_set=test_pool, verbose=100)
    values = model.get_feature_importance(train_pool)
    return {feature: float(value) for feature, value in zip(features, values)}


def calculate_metric_frame(
    df: pl.DataFrame,
    features: list[str],
    *,
    window: int | None = None,
    windows_list: list[int] | None = None,
) -> pl.DataFrame:
    if windows_list is None:
        windows_list = DEFAULT_WINDOWS_LIST if window is None else [window]
    rows = []
    for window_length in windows_list:
        future_return = calculate_future_return(df, window_length)
        if future_return.size == 0:
            continue
        catboost_values = _catboost_importance(df, features, future_return)
        metric_df = df.slice(0, future_return.size)
        for feature in features:
            values = metric_df[feature].to_numpy()
            rows.append(
                {
                    "feature": feature,
                    "window": window_length,
                    "Permutation Importance": _permutation_importance(
                        values, future_return
                    ),
                    "CatBoost Importance": catboost_values.get(feature, 0.0),
                    "IC": calculate_ic(values, future_return),
                    "RankIC": calculate_rank_ic(values, future_return),
                    "Sharpe": calculate_sharpe(values, future_return),
                }
            )
    if not rows:
        raise ValueError("future return is empty; cannot calculate feature metrics")
    return pl.DataFrame(rows)


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
    return combined.group_by("feature", maintain_order=True).agg(expressions)
