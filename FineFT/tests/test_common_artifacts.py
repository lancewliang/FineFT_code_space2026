from common.artifacts import (
    ArtifactNames,
    HistoryArtifactNames,
    get_df_chunk_filename,
    get_heuristic_plot_filename,
    get_ood_logpx_filename,
    get_trading_detail_csv_filename,
    get_vae_label_filename,
    get_vae_test_filename,
    get_valid_processed_filename,
)


def test_artifact_names_literals() -> None:
    assert ArtifactNames.DATASET_MANIFEST_JSON == "dataset_manifest.json"
    assert ArtifactNames.SLICE_MANIFEST_JSON == "slice_manifest.json"
    assert ArtifactNames.REGIME_THRESHOLDS_JSON == "regime_thresholds.json"
    assert ArtifactNames.STATE_FEATURES_NPY == "state_features.npy"
    assert (
        ArtifactNames.MAINTENANCE_MARGIN_RATIO_DICT_NPY
        == "maintenance_margin_ratio_dict.npy"
    )
    assert ArtifactNames.TRAINED_MODEL_PKL == "trained_model.pkl"
    assert ArtifactNames.MODEL_LATEST_PTH == "model_latest.pth"
    assert ArtifactNames.ANALYSIS_RESULT_CSV == "analysis_result.csv"
    assert ArtifactNames.ANALYSIS_RESULT_NPY == "analysis_result.npy"
    assert ArtifactNames.SELECTION_MANIFEST_JSON == "selection_manifest.json"
    assert (
        ArtifactNames.TWO_DIMENSIONAL_SELECTION_MANIFEST_JSON
        == "two_dimensional_selection_manifest.json"
    )
    assert ArtifactNames.OPTUNA_RESULTS_CSV == "optuna_results.csv"
    assert ArtifactNames.CONTRACT_RESULTS_CSV == "contract_results.csv"
    assert ArtifactNames.TRADING_INFO_NPY == "trading_info.npy"
    assert ArtifactNames.RESULT_CSV == "result.csv"
    assert ArtifactNames.RESULT_ALL_CSV == "result_all.csv"
    assert ArtifactNames.BEST_RESULT_CSV == "best_result.csv"
    assert ArtifactNames.HIGH_LEVEL_AGENT_PARA_TXT == "high_level_agent_para.txt"
    assert ArtifactNames.SUMMARY_JSON == "summary.json"
    assert ArtifactNames.ROUTING_SUMMARY_JSON == "routing_summary.json"
    assert ArtifactNames.ID_LOGPX_NPY == "id_logpx.npy"
    assert ArtifactNames.OOD_LOGPX_ALL_NPY == "ood_logpx_all.npy"
    assert ArtifactNames.OOD_LOGPX_ALL_CSV == "ood_logpx_all.csv"


def test_history_artifact_names_literals_and_typo_compatibility() -> None:
    assert HistoryArtifactNames.REWARD_HISTORY_NPY == "reward_history.npy"
    assert HistoryArtifactNames.TOTAL_ASSET_HISTORY_NPY == "total_asset_history.npy"
    assert (
        HistoryArtifactNames.WALLET_BALANCE_HISTORY_NPY
        == "wallet_balance_history.npy"
    )
    assert HistoryArtifactNames.UNREALIZED_PNL_HISTORY_NPY == "unrealized_pnl_history.npy"
    assert (
        HistoryArtifactNames.INITIAL_MARGIN_HISTORY_NPY
        == "initial_margin_history.npy"
    )
    # Critical backward-compatibility requirement
    assert (
        HistoryArtifactNames.MAINTAIN_MARGIN_HISTORY_NPY
        == "maintain_marigine_history.npy"
    )
    assert (
        HistoryArtifactNames.NEW_POSITION_REQUIRED_MONEY_HISTORY_NPY
        == "new_position_required_money_history.npy"
    )
    assert HistoryArtifactNames.MICRO_ACTION_HISTORY_NPY == "micro_action_history.npy"
    assert HistoryArtifactNames.MACRO_ACTION_HISTORY_NPY == "macro_action_history.npy"


def test_dynamic_filename_builders() -> None:
    assert get_df_chunk_filename(0) == "df_0.feather"
    assert get_df_chunk_filename(12) == "df_12.feather"
    assert get_valid_processed_filename("fu2401") == "valid_processed_fu2401.feather"
    assert (
        get_trading_detail_csv_filename(45)
        == "trading_action_detail_epoch_45.csv"
    )
    assert get_vae_label_filename("label_0") == "label_0.npy"
    assert get_vae_test_filename("fu2405") == "test_fu2405.npy"
    assert get_heuristic_plot_filename("fu2401", "png") == "best_result_fu2401.png"
    assert get_heuristic_plot_filename("fu2401", "pdf") == "best_result_fu2401.pdf"
    assert get_ood_logpx_filename("fu2401", "npy") == "ood_logpx_fu2401.npy"
    assert get_ood_logpx_filename("fu2401", "csv") == "ood_logpx_fu2401.csv"
