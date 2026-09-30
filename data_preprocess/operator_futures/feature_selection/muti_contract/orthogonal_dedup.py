from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from operator_futures.feature_selection.cor_util import select_feature
from operator_futures.feature_selection.muti_contract.types import (
    OrthogonalDedupConfig,
    PipelineStepResult,
)

logger = logging.getLogger(__name__)


def compute_contract_normalized_spearman_correlation_matrix(
    frames: dict[str, pl.DataFrame],
    features: list[str],
) -> pl.DataFrame:
    if not features:
        return pl.DataFrame(schema={"feature": pl.Utf8})

    num_features = len(features)
    if num_features == 1:
        return pl.DataFrame(
            np.array([[1.0]], dtype=np.float64), schema=features
        ).with_columns(pl.Series("feature", features))

    total_samples = sum(frame.height for frame in frames.values() if frame.height > 0)
    if total_samples == 0:
        raise ValueError("Total sample count across frames must be greater than zero")

    weighted_corr = np.zeros((num_features, num_features), dtype=np.float64)

    for frame in frames.values():
        contract_samples = frame.height
        if contract_samples <= 0:
            continue
        # Convert columns to ranks for Spearman correlation
        ranked_matrix = (
            frame.select(
                [
                    pl.col(f).rank(method="average").cast(pl.Float64).fill_null(0.0)
                    for f in features
                ]
            )
            .to_numpy()
        )
        corr_matrix = np.corrcoef(ranked_matrix, rowvar=False)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        corr_matrix = np.atleast_2d(corr_matrix)
        np.fill_diagonal(corr_matrix, 1.0)
        weight = float(contract_samples) / float(total_samples)
        weighted_corr += weight * corr_matrix

    np.fill_diagonal(weighted_corr, 1.0)
    weighted_corr = np.clip(weighted_corr, -1.0, 1.0)

    return pl.DataFrame(weighted_corr, schema=features).with_columns(
        pl.Series("feature", features)
    )


def prune_by_vif(
    selected: list[str],
    corr_matrix_df: pl.DataFrame,
    max_vif: float = 10.0,
) -> tuple[list[str], list[str]]:
    current = list(selected)
    dropped: list[str] = []
    while len(current) > 1:
        sub_corr = (
            corr_matrix_df.filter(pl.col("feature").is_in(current))
            .select(current)
            .to_numpy()
        )
        try:
            inv_r = np.linalg.inv(sub_corr)
            vifs = np.diag(inv_r)
            worst_idx = int(np.argmax(vifs))
            if vifs[worst_idx] > max_vif:
                drop_f = current.pop(worst_idx)
                dropped.append(drop_f)
            else:
                break
        except np.linalg.LinAlgError:
            # Singular matrix: remove feature with largest off-diagonal sum
            off_diag = np.sum(np.abs(sub_corr), axis=1) - 1.0
            worst_idx = int(np.argmax(off_diag))
            drop_f = current.pop(worst_idx)
            dropped.append(drop_f)
    return current, dropped


def execute_orthogonal_deduplication(
    frames: dict[str, pl.DataFrame],
    features: list[str],
    priority_order: list[str],
    config: OrthogonalDedupConfig,
) -> PipelineStepResult:
    if not features:
        return PipelineStepResult(
            step_name="orthogonal_dedup",
            surviving_features=[],
            dropped_features=[],
            audit_metrics_df=None,
            diagnostics={
                "dedup_method": config.dedup_method,
                "correlation_method": config.correlation_method,
                "vif_dropped": [],
                "cluster_dropped": [],
                "num_clusters": 0,
            },
        )

    # 1. Compute contract-normalized correlation matrix (Spearman rank or Pearson)
    if config.correlation_method == "pearson":
        from operator_futures.feature_selection.cor_util import (
            compute_contract_normalized_correlation_matrix,
        )

        corre_df = compute_contract_normalized_correlation_matrix(frames, features)
    else:
        corre_df = compute_contract_normalized_spearman_correlation_matrix(
            frames, features
        )

    # 2. Greedy Deduplication Fallback Mode
    if config.dedup_method == "greedy" or not config.enable_hierarchical_clustering:
        selected = select_feature(
            features=priority_order, corre_df=corre_df, theshold=config.max_correlation
        )
        all_dropped = [f for f in features if f not in set(selected)]
        return PipelineStepResult(
            step_name="orthogonal_dedup",
            surviving_features=selected,
            dropped_features=all_dropped,
            audit_metrics_df=corre_df,
            diagnostics={
                "dedup_method": "greedy",
                "correlation_method": config.correlation_method,
                "vif_dropped": [],
                "cluster_dropped": all_dropped,
                "num_clusters": len(selected),
            },
        )

    # 3. Ward Hierarchical Clustering Deduplication
    corr_np = corre_df.select(features).to_numpy()
    np.fill_diagonal(corr_np, 1.0)
    n_features = len(features)

    if n_features == 1:
        return PipelineStepResult(
            step_name="orthogonal_dedup",
            surviving_features=list(features),
            dropped_features=[],
            audit_metrics_df=corre_df,
            diagnostics={
                "dedup_method": "cluster",
                "correlation_method": config.correlation_method,
                "vif_dropped": [],
                "cluster_dropped": [],
                "num_clusters": 1,
            },
        )

    # Distance matrix: D = sqrt((1 - R) / 2) in [0, 1]
    dist_matrix = np.sqrt(np.clip((1.0 - corr_np) / 2.0, 0.0, 1.0))
    np.fill_diagonal(dist_matrix, 0.0)
    condensed_dist = squareform(dist_matrix, checks=False)

    z = linkage(condensed_dist, method="ward")

    # Initial cut by distance threshold
    cluster_ids = fcluster(z, t=config.cluster_distance_threshold, criterion="distance")
    num_clusters = len(np.unique(cluster_ids))

    # Dynamic capacity calibration
    if num_clusters > config.max_clusters and n_features > config.max_clusters:
        cluster_ids = fcluster(z, t=config.max_clusters, criterion="maxclust")
        num_clusters = len(np.unique(cluster_ids))

    # Intra-cluster selection according to Priority order
    cluster_selected: list[str] = []
    # If num_clusters < min_clusters and features allow, retain top-2 per cluster
    retain_per_cluster = (
        2
        if (num_clusters < config.min_clusters and n_features >= config.min_clusters)
        else 1
    )

    for cid in sorted(np.unique(cluster_ids)):
        members = [features[idx] for idx, c in enumerate(cluster_ids) if c == cid]
        members_sorted = sorted(
            members,
            key=lambda f: priority_order.index(f) if f in priority_order else 999999,
        )
        cluster_selected.extend(members_sorted[:retain_per_cluster])

    cluster_dropped = [f for f in features if f not in set(cluster_selected)]

    # 4. Variance Inflation Factor (VIF) check on representatives
    vif_surviving, vif_dropped = prune_by_vif(
        cluster_selected, corre_df, max_vif=config.max_vif
    )

    all_dropped = [f for f in features if f not in set(vif_surviving)]

    return PipelineStepResult(
        step_name="orthogonal_dedup",
        surviving_features=vif_surviving,
        dropped_features=all_dropped,
        audit_metrics_df=corre_df,
        diagnostics={
            "dedup_method": "cluster",
            "correlation_method": config.correlation_method,
            "vif_dropped": vif_dropped,
            "cluster_dropped": cluster_dropped,
            "num_clusters": num_clusters,
        },
    )
