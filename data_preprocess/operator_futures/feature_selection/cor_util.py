from __future__ import annotations

import numpy as np
import polars as pl


def calculate_cor(df: pl.DataFrame, feature_1: str, feature_2: str) -> float:
    return float(df.select(pl.corr(feature_1, feature_2)).item())


def _normalise_correlation_matrix(corre_df: pl.DataFrame) -> pl.DataFrame:
    if "feature" in corre_df.columns:
        return corre_df
    if corre_df.columns and corre_df.columns[0] in {"", "index", "column_0"}:
        return corre_df.rename({corre_df.columns[0]: "feature"})
    return corre_df.with_columns(pl.Series("feature", corre_df.columns))


def compute_contract_normalized_correlation_matrix(
    frames: dict[str, pl.DataFrame],
    features: list[str],
) -> pl.DataFrame:
    if not features:
        return pl.DataFrame(schema={"feature": pl.Utf8})

    total_samples = sum(frame.height for frame in frames.values() if frame.height > 0)
    if total_samples == 0:
        raise ValueError("Total sample count across frames must be greater than zero")

    num_features = len(features)
    weighted_corr = np.zeros((num_features, num_features), dtype=np.float64)

    for frame in frames.values():
        contract_sample_count = frame.height
        if contract_sample_count <= 0:
            continue
        corr_matrix = frame.select(features).corr().to_numpy()
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        np.fill_diagonal(corr_matrix, 1.0)
        weight = float(contract_sample_count) / float(total_samples)
        weighted_corr += weight * corr_matrix

    np.fill_diagonal(weighted_corr, 1.0)
    weighted_corr = np.clip(weighted_corr, -1.0, 1.0)

    return pl.DataFrame(weighted_corr, schema=features).with_columns(
        pl.Series("feature", features)
    )


def select_feature(
    features: list[str] | None = None,
    df: pl.DataFrame | None = None,
    corre_df: pl.DataFrame | None = None,
    theshold: float = 0.5,
) -> list[str]:
    if df is None and corre_df is None:
        raise ValueError("df and corre_df cannot be both None")
    if corre_df is not None:
        corre_df = _normalise_correlation_matrix(corre_df)
        all_feature_names = [column for column in corre_df.columns if column != "feature"]
        if not all_feature_names:
            return []

        # Priority order: if features list is provided, order by features list; otherwise matrix columns order
        if features is not None:
            feature_priority = [f for f in features if f in all_feature_names]
        else:
            feature_priority = all_feature_names

        row_feature_names = corre_df["feature"].to_list()
        row_index_by_feature = {
            feature: index for index, feature in enumerate(row_feature_names)
        }
        col_index_by_feature = {
            feature: index for index, feature in enumerate(all_feature_names)
        }
        matrix = corre_df.select(all_feature_names).to_numpy()
        selected_feature_names = []
        remaining_features = list(feature_priority)
        for feature in feature_priority:
            if feature in remaining_features:
                selected_feature_names.append(feature)
                remaining_features.remove(feature)
                row_index = row_index_by_feature.get(feature)
                if row_index is None:
                    continue
                for remain_f in list(remaining_features):
                    col_index = col_index_by_feature[remain_f]
                    value = matrix[row_index, col_index]
                    if np.abs(float(value)) > theshold:
                        remaining_features.remove(remain_f)
        return selected_feature_names
    if features is None or df is None:
        raise ValueError("features and df are required if corre_df is not provided")
    features = list(features)
    corre_df = df.select(features).corr().with_columns(pl.Series("feature", features))
    return select_feature(features=features, corre_df=corre_df, theshold=theshold)
