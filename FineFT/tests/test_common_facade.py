import common
from common import (
    AGGREGATE_JSON_COLUMNS,
    CSV_HEADER_LABELS,
    ArtifactNames,
    HistoryArtifactNames,
    MetricColumns,
    TradeColumns,
    get_df_chunk_filename,
    get_heuristic_plot_filename,
    get_ood_logpx_filename,
    get_trading_detail_csv_filename,
    get_vae_label_filename,
    get_vae_test_filename,
    get_valid_processed_filename,
)


def test_common_facade_exports() -> None:
    assert ArtifactNames is not None
    assert HistoryArtifactNames is not None
    assert MetricColumns is not None
    assert TradeColumns is not None
    assert isinstance(CSV_HEADER_LABELS, dict)
    assert isinstance(AGGREGATE_JSON_COLUMNS, list)
    assert callable(get_df_chunk_filename)
    assert callable(get_valid_processed_filename)
    assert callable(get_trading_detail_csv_filename)
    assert callable(get_vae_label_filename)
    assert callable(get_vae_test_filename)
    assert callable(get_heuristic_plot_filename)
    assert callable(get_ood_logpx_filename)
    assert hasattr(common, "ArtifactNames")
    assert hasattr(common, "MetricColumns")
    assert hasattr(common, "TradeColumns")
