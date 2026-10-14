"""Statistical metrics calculation with average tie handling and vectorized matrix support."""

from typing import Optional, Tuple
import numpy as np
from scipy import stats


def compute_spearman_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], Optional[str]]:
    """Calculates Spearman rank correlation with average rank for ties.
    Returns (correlation, null_reason)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x_valid = x[mask]
    y_valid = y[mask]

    n = len(x_valid)
    if n < 3:
        return None, "INSUFFICIENT_PAIRS"

    if np.all(x_valid == x_valid[0]) or np.all(y_valid == y_valid[0]):
        return None, "CONSTANT_SERIES"

    res = stats.spearmanr(x_valid, y_valid)
    corr = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
    if np.isnan(corr):
        return None, "UNDEFINED_CORRELATION"

    return corr, None


def compute_spearman_matrix(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized calculation of Spearman correlation for all columns in X against y.
    X: shape (N, D), y: shape (N,)
    Returns: shape (D,) array of correlations with average rank for ties.
    Columns that are constant or invalid produce np.nan."""
    N, D = X.shape
    if N < 3:
        return np.full(D, np.nan)
    if np.all(y == y[0]):
        return np.full(D, np.nan)

    r_y = stats.rankdata(y, method="average")
    std_y = np.std(r_y)
    if std_y == 0:
        return np.full(D, np.nan)
    u = (r_y - np.mean(r_y)) / std_y

    r_X = stats.rankdata(X, axis=0, method="average")
    mean_X = np.mean(r_X, axis=0)
    std_X = np.std(r_X, axis=0)

    std_X_safe = np.where(std_X == 0, 1.0, std_X)
    V = (r_X - mean_X) / std_X_safe

    corrs = (V.T @ u) / N
    corrs = np.where(std_X == 0, np.nan, corrs)
    return corrs
