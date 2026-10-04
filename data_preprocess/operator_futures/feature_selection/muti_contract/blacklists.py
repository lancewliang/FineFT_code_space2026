from __future__ import annotations

from operator_futures.feature_selection.blacklists import (
    DEFAULT_BLACKLIST_JSON_PATH,
    get_commodity_global_hygiene_blacklist,
    get_commodity_rl_feature_blacklist,
    get_commodity_stream_blacklists,
    get_commodity_vae_slope_feature_blacklist,
    get_commodity_vae_volatility_feature_blacklist,
    load_commodity_feature_blacklists,
    main,
)

if __name__ == "__main__":
    main()
