import numpy as np
import polars as pl


def calculate_cor(df: pl.DataFrame, feature_1: str, feature_2: str):
    return df.select(pl.corr(feature_1, feature_2)).item()


def _normalise_correlation_matrix(corre_df: pl.DataFrame) -> pl.DataFrame:
    if "feature" in corre_df.columns:
        return corre_df
    if corre_df.columns and corre_df.columns[0] in {"", "index", "column_0"}:
        return corre_df.rename({corre_df.columns[0]: "feature"})
    return corre_df.with_columns(pl.Series("feature", corre_df.columns))


def select_feature(features=None, df=None, corre_df: pl.DataFrame = None, theshold=0.5):
    if df is None and corre_df is None:
        raise ValueError("df and corre_df cannot be both None")
    if df is not None and features is None and corre_df is None:
        raise ValueError("features are required if the corre_df is not provided")
    if corre_df is not None:
        corre_df = _normalise_correlation_matrix(corre_df)
        all_feature_names = [column for column in corre_df.columns if column != "feature"]
        if not all_feature_names:
            return []
        row_feature_names = corre_df["feature"].to_list()
        row_index_by_feature = {
            feature: index for index, feature in enumerate(row_feature_names)
        }
        col_index_by_feature = {
            feature: index for index, feature in enumerate(all_feature_names)
        }
        matrix = corre_df.select(all_feature_names).to_numpy()
        selected_feature_names = []
        remaining_features = all_feature_names
        for feature in all_feature_names:
            if feature in remaining_features:
                selected_feature_names.append(feature)
                remaining_features.remove(feature)
                row_index = row_index_by_feature.get(feature)
                if row_index is None:
                    continue
                for remain_f in remaining_features:
                    col_index = col_index_by_feature[remain_f]
                    value = matrix[row_index, col_index]
                    if np.abs(float(value)) > theshold:
                        remaining_features.remove(remain_f)
        return selected_feature_names
    if features is None or df is None:
        raise ValueError("features and df are required if corre_df is not provided")
    features = list(features)
    corre_df = df.select(features).corr().with_columns(pl.Series("feature", features))
    return select_feature(corre_df=corre_df, theshold=theshold)
