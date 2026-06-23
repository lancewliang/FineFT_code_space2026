import pandas as pd
import numpy as np
import os
from multiprocessing import Pool


def calculate_cor(df: pd.DataFrame, feature_1: str, feature_2: str):
    return df[feature_1].corr(df[feature_2])


def select_feature(features=None, df=None, corre_df: pd.DataFrame = None, theshold=0.5):
    max_cpu_num = int(max(os.cpu_count() - 10, os.cpu_count() / 2))
    if df is None and corre_df is None:
        raise ValueError("df and corre_df cannot be both None")
    if df is not None and features is None and corre_df is None:
        raise ValueError("features are required if the corre_df is not provided")
    if corre_df is not None:
        print("Using the provided correlation matrix")
        all_feature_names = corre_df.columns.values.tolist()
        selected_feature_names = []
        discard_feature_names = []
        remaining_features = all_feature_names
        for feature in all_feature_names:
            if feature in remaining_features:
                selected_feature_names.append(feature)
                discard_feature_names.append(feature)
                remaining_features.remove(feature)
                for remain_f in remaining_features:
                    if np.abs(corre_df.loc[feature, remain_f]) > theshold:
                        discard_feature_names.append(remain_f)
                        remaining_features.remove(remain_f)
        return selected_feature_names
    else:
        if features is None or df is None:
            raise ValueError("features and df are required if corre_df is not provided")
        all_feature_names = corre_df.columns.values.tolist()
        selected_feature_names = []
        discard_feature_names = []
        remaining_features = all_feature_names
        for feature in all_feature_names:
            if feature in remaining_features:
                selected_feature_names.append(feature)
                discard_feature_names.append(feature)
                remaining_features.remove(feature)
                with Pool(processes=min(len(remaining_features), max_cpu_num)) as pool:
                    results = [
                        pool.apply_async(calculate_cor, args=(df, feature, remain_f))
                        for remain_f in remaining_features
                    ]
                    # 等待所有结果完成，并获取结果
                    ic_result = [result.get() for result in results]
                for remain_f, cor in zip(remaining_features, ic_result):
                    if np.abs(cor) > theshold:
                        discard_feature_names.append(remain_f)
                        remaining_features.remove(remain_f)
        return selected_feature_names
