from __future__ import annotations


class ArtifactNames:
    DATASET_MANIFEST_JSON: str = "dataset_manifest.json"
    DATASET_SPLIT_MANIFEST_JSON: str = "dataset_split_manifest.json"
    SLICE_MANIFEST_JSON: str = "slice_manifest.json"
    REGIME_THRESHOLDS_JSON: str = "regime_thresholds.json"
    STATE_FEATURES_NPY: str = "state_features.npy"
    MAINTENANCE_MARGIN_RATIO_DICT_NPY: str = "maintenance_margin_ratio_dict.npy"
    TRAINED_MODEL_PKL: str = "trained_model.pkl"
    MODEL_LATEST_PTH: str = "model_latest.pth"
    ANALYSIS_RESULT_CSV: str = "analysis_result.csv"
    ANALYSIS_RESULT_NPY: str = "analysis_result.npy"
    SELECTION_MANIFEST_JSON: str = "selection_manifest.json"
    TWO_DIMENSIONAL_SELECTION_MANIFEST_JSON: str = "two_dimensional_selection_manifest.json"
    TWO_DIMENSIONAL_MARGINAL_METRICS_CSV: str = "two_dimensional_marginal_metrics.csv"
    TWO_DIMENSIONAL_JOINT_METRICS_CSV: str = "two_dimensional_joint_metrics.csv"
    TWO_DIMENSIONAL_CANDIDATE_RANKINGS_CSV: str = "two_dimensional_candidate_rankings.csv"
    TWO_DIMENSIONAL_SELECTION_CSV: str = "two_dimensional_selection.csv"
    OPTUNA_RESULTS_CSV: str = "optuna_results.csv"
    CONTRACT_RESULTS_CSV: str = "contract_results.csv"
    TRADING_INFO_NPY: str = "trading_info.npy"
    RESULT_CSV: str = "result.csv"
    RESULT_ALL_CSV: str = "result_all.csv"
    BEST_RESULT_CSV: str = "best_result.csv"
    HIGH_LEVEL_AGENT_PARA_TXT: str = "high_level_agent_para.txt"
    MODEL_PTH: str = "model.pth"
    BEST_INDEX_INFO_CSV: str = "best_index_info_by_dynamics_with_different_position.csv"
    TEST_FEATHER: str = "test.feather"
    TEST_NPY: str = "test.npy"
    SUMMARY_JSON: str = "summary.json"
    ROUTING_SUMMARY_JSON: str = "routing_summary.json"
    ID_LOGPX_NPY: str = "id_logpx.npy"
    OOD_LOGPX_ALL_NPY: str = "ood_logpx_all.npy"
    OOD_LOGPX_ALL_CSV: str = "ood_logpx_all.csv"


class HistoryArtifactNames:
    REWARD_HISTORY_NPY: str = "reward_history.npy"
    TOTAL_ASSET_HISTORY_NPY: str = "total_asset_history.npy"
    WALLET_BALANCE_HISTORY_NPY: str = "wallet_balance_history.npy"
    UNREALIZED_PNL_HISTORY_NPY: str = "unrealized_pnl_history.npy"
    INITIAL_MARGIN_HISTORY_NPY: str = "initial_margin_history.npy"
    # Preserves historical typo for 100% backward compatibility with on-disk results
    MAINTAIN_MARGIN_HISTORY_NPY: str = "maintain_marigine_history.npy"
    NEW_POSITION_REQUIRED_MONEY_HISTORY_NPY: str = "new_position_required_money_history.npy"
    MICRO_ACTION_HISTORY_NPY: str = "micro_action_history.npy"
    MACRO_ACTION_HISTORY_NPY: str = "macro_action_history.npy"
    MACRO_ACTION_NPY: str = "macro_action.npy"


def get_df_chunk_filename(index: int) -> str:
    return f"df_{index}.feather"


def get_valid_processed_filename(contract: str) -> str:
    return f"valid_processed_{contract}.feather"


def get_trading_detail_csv_filename(epoch: int) -> str:
    return f"trading_action_detail_epoch_{epoch}.csv"


def get_vae_label_filename(label_name: str) -> str:
    return f"{label_name}.npy"


def get_vae_test_filename(contract: str) -> str:
    return f"test_{contract}.npy"


def get_heuristic_plot_filename(contract: str, extension: str = "png") -> str:
    return f"best_result_{contract}.{extension}"


def get_ood_logpx_filename(contract: str, extension: str = "npy") -> str:
    return f"ood_logpx_{contract}.{extension}"
