"""Joint moving-block bootstrap uncertainty estimation for multi-horizon feature audit."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings
import numpy as np

from .metrics import compute_spearman_correlation, compute_spearman_matrix


@dataclass(frozen=True)
class BootstrapResult:
    status: str  # "OK", "INSUFFICIENT_DATA", "UNDEFINED"
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    effective_blocks: float
    undefined_replicates_count: int
    num_valid_days: int = 0
    block_length: int = 0
    min_effective_blocks: int = 20
    null_reason: Optional[str] = None
    ci_status: str = "AVAILABLE_EXPLORATORY"


@dataclass(frozen=True)
class MatrixBootstrapResult:
    status: str  # "OK", "INSUFFICIENT_DATA", "UNDEFINED"
    ci_status: str  # "AVAILABLE_EXPLORATORY", "INSUFFICIENT_EFFECTIVE_BLOCKS", "INSUFFICIENT_TRADING_DAYS", "UNDEFINED_REPLICATES"
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    effective_blocks: float
    num_valid_days: int
    block_length: int
    min_effective_blocks: int
    null_reason: Optional[str]


def joint_moving_block_bootstrap(
    trading_days: List[str],
    day_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    block_length: int,
    min_effective_blocks: int = 20,
    n_replicates: int = 2000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> BootstrapResult:
    """Executes joint moving-block bootstrap across trading days for single feature."""
    valid_days = [td for td in trading_days if td in day_data and len(day_data[td][0]) > 0]
    num_valid_days = len(valid_days)
    effective_blocks = float(num_valid_days // block_length) if block_length > 0 else 0.0

    if effective_blocks < min_effective_blocks:
        reason = f"effective_blocks({effective_blocks:.1f}) < min_required({min_effective_blocks})"
        return BootstrapResult(
            status="INSUFFICIENT_DATA",
            ci_lower=None,
            ci_upper=None,
            effective_blocks=effective_blocks,
            undefined_replicates_count=0,
            num_valid_days=num_valid_days,
            block_length=block_length,
            min_effective_blocks=min_effective_blocks,
            null_reason=reason,
            ci_status="INSUFFICIENT_EFFECTIVE_BLOCKS",
        )

    n_days = len(trading_days)
    if n_days < block_length:
        reason = f"trading_days({n_days}) < block_length({block_length})"
        return BootstrapResult(
            status="INSUFFICIENT_DATA",
            ci_lower=None,
            ci_upper=None,
            effective_blocks=effective_blocks,
            undefined_replicates_count=0,
            num_valid_days=num_valid_days,
            block_length=block_length,
            min_effective_blocks=min_effective_blocks,
            null_reason=reason,
        )

    num_blocks = n_days - block_length + 1
    blocks = [trading_days[i : i + block_length] for i in range(num_blocks)]
    needed_blocks = int(np.ceil(n_days / block_length))

    rng = np.random.RandomState(seed)
    reps: List[float] = []
    undefined_count = 0
    alpha = (1.0 - confidence_level) / 2.0

    for _ in range(n_replicates):
        chosen_indices = rng.randint(0, num_blocks, size=needed_blocks)
        sampled_days = []
        for b_idx in chosen_indices:
            sampled_days.extend(blocks[b_idx])
        sampled_days = sampled_days[:n_days]

        x_list = []
        y_list = []
        for d in sampled_days:
            if d in day_data:
                dx, dy = day_data[d]
                if len(dx) > 0:
                    x_list.append(dx)
                    y_list.append(dy)

        if not x_list:
            undefined_count += 1
            continue

        boot_x = np.concatenate(x_list)
        boot_y = np.concatenate(y_list)

        corr, null_reason = compute_spearman_correlation(boot_x, boot_y)
        if corr is None or np.isnan(corr):
            undefined_count += 1
        else:
            reps.append(corr)

    if len(reps) < 10:
        reason = f"valid_replicates({len(reps)}) < 10"
        return BootstrapResult(
            status="UNDEFINED",
            ci_lower=None,
            ci_upper=None,
            effective_blocks=effective_blocks,
            undefined_replicates_count=undefined_count,
            num_valid_days=num_valid_days,
            block_length=block_length,
            min_effective_blocks=min_effective_blocks,
            null_reason=reason,
            ci_status="UNDEFINED_REPLICATES",
        )

    ci_lower = float(np.percentile(reps, 100.0 * alpha))
    ci_upper = float(np.percentile(reps, 100.0 * (1.0 - alpha)))

    return BootstrapResult(
        status="OK",
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        effective_blocks=effective_blocks,
        undefined_replicates_count=undefined_count,
        num_valid_days=num_valid_days,
        block_length=block_length,
        min_effective_blocks=min_effective_blocks,
        null_reason=None,
    )


def joint_moving_block_bootstrap_matrix(
    trading_days: List[str],
    day_indices: Dict[str, np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    block_length: int,
    min_effective_blocks: int = 20,
    n_replicates: int = 2000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> MatrixBootstrapResult:
    """Vectorized joint moving block bootstrap for all columns in X simultaneously."""
    N, D = X.shape
    valid_days = [td for td in trading_days if td in day_indices and len(day_indices[td]) > 0]
    num_valid_days = len(valid_days)
    effective_blocks = float(num_valid_days // block_length) if block_length > 0 else 0.0

    n_days = len(trading_days)
    if n_days < block_length:
        reason = f"trading_days({n_days}) < block_length({block_length})"
        return MatrixBootstrapResult(
            status="INSUFFICIENT_DATA",
            ci_status="INSUFFICIENT_TRADING_DAYS",
            ci_lower=np.full(D, np.nan),
            ci_upper=np.full(D, np.nan),
            effective_blocks=effective_blocks,
            num_valid_days=num_valid_days,
            block_length=block_length,
            min_effective_blocks=min_effective_blocks,
            null_reason=reason,
        )

    if effective_blocks < min_effective_blocks:
        reason = f"effective_blocks({effective_blocks:.1f}) < min_required({min_effective_blocks})"
        return MatrixBootstrapResult(
            status="INSUFFICIENT_DATA",
            ci_status="INSUFFICIENT_EFFECTIVE_BLOCKS",
            ci_lower=np.full(D, np.nan),
            ci_upper=np.full(D, np.nan),
            effective_blocks=effective_blocks,
            num_valid_days=num_valid_days,
            block_length=block_length,
            min_effective_blocks=min_effective_blocks,
            null_reason=reason,
        )

    num_blocks = n_days - block_length + 1
    blocks = [trading_days[i : i + block_length] for i in range(num_blocks)]
    needed_blocks = int(np.ceil(n_days / block_length))

    rng = np.random.RandomState(seed)
    reps_list = []
    alpha = (1.0 - confidence_level) / 2.0

    for _ in range(n_replicates):
        chosen_indices = rng.randint(0, num_blocks, size=needed_blocks)
        sampled_days = []
        for b_idx in chosen_indices:
            sampled_days.extend(blocks[b_idx])
        sampled_days = sampled_days[:n_days]

        row_indices_list = [day_indices[d] for d in sampled_days if d in day_indices and len(day_indices[d]) > 0]
        if not row_indices_list:
            continue
        sample_rows = np.concatenate(row_indices_list)

        X_boot = X[sample_rows, :]
        y_boot = y[sample_rows]

        corrs = compute_spearman_matrix(X_boot, y_boot)
        reps_list.append(corrs)

    if len(reps_list) < 10:
        reason = f"valid_replicates({len(reps_list)}) < 10"
        return MatrixBootstrapResult(
            status="UNDEFINED",
            ci_status="UNDEFINED_REPLICATES",
            ci_lower=np.full(D, np.nan),
            ci_upper=np.full(D, np.nan),
            effective_blocks=effective_blocks,
            num_valid_days=num_valid_days,
            block_length=block_length,
            min_effective_blocks=min_effective_blocks,
            null_reason=reason,
        )

    reps_arr = np.array(reps_list)  # (replicates, D)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ci_lower = np.nanpercentile(reps_arr, 100.0 * alpha, axis=0)
        ci_upper = np.nanpercentile(reps_arr, 100.0 * (1.0 - alpha), axis=0)

    return MatrixBootstrapResult(
        status="OK",
        ci_status="AVAILABLE_EXPLORATORY",
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        effective_blocks=effective_blocks,
        num_valid_days=num_valid_days,
        block_length=block_length,
        min_effective_blocks=min_effective_blocks,
        null_reason=None,
    )
