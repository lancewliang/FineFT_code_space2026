"""As-of reference price lookup with freshness bounds and same-contract constraints."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import polars as pl

from .contracts import InvalidReason


@dataclass(frozen=True)
class PriceLookupResult:
    price: Optional[float]
    observed_time: Optional[datetime]
    quote_age_seconds: Optional[float]
    reason: InvalidReason


class ReferencePriceProvider:
    """Provides as-of snapshot prices for contracts with freshness verification using binary search."""

    def __init__(self, price_df: pl.DataFrame, price_field: str = "LastPrice", max_quote_age_seconds: int = 60):
        self.price_field = price_field
        self.max_quote_age_seconds = max_quote_age_seconds

        if price_df.schema["timestamp"] == pl.Utf8:
            price_df = price_df.with_columns(
                pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
            )

        # Pre-index sorted arrays by contract for O(log N) lookup
        self._contract_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        contracts = price_df["contract"].unique().to_list()
        for c in contracts:
            sub = price_df.filter(pl.col("contract") == c).sort("timestamp")
            ts_arr = sub["timestamp"].to_numpy().astype("datetime64[us]")
            p_arr = sub[self.price_field].to_numpy().astype(float)
            self._contract_data[c] = (ts_arr, p_arr)

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "ReferencePriceProvider":
        p = Path(manifest_path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        price_file = Path(data["price_file"])
        price_field = data.get("price_field", "LastPrice")
        max_age = int(data.get("max_quote_age_seconds", 60))

        if price_file.is_dir():
            feather_files = sorted(list(price_file.glob("*.feather")))
            if feather_files:
                frames = [pl.read_ipc(ff).select(["contract", "timestamp", price_field]) for ff in feather_files]
                df = pl.concat(frames)
            else:
                parquet_files = sorted(list(price_file.glob("*.parquet")))
                if parquet_files:
                    frames = [pl.read_parquet(pf).select(["contract", "timestamp", price_field]) for pf in parquet_files]
                    df = pl.concat(frames)
                else:
                    raise FileNotFoundError(f"No feather or parquet files found in price directory: {price_file}")
        elif price_file.suffix == ".feather":
            df = pl.read_ipc(price_file)
        elif price_file.suffix == ".parquet":
            df = pl.read_parquet(price_file)
        else:
            df = pl.read_csv(price_file)

        return cls(df, price_field=price_field, max_quote_age_seconds=max_age)

    def lookup_as_of(self, contract: str, target_dt: datetime) -> PriceLookupResult:
        """Finds latest valid snapshot at or before target_dt for the given contract using binary search."""
        if contract not in self._contract_data:
            return PriceLookupResult(None, None, None, InvalidReason.MISSING_TARGET_PRICE)

        ts_arr, p_arr = self._contract_data[contract]
        t_target = np.datetime64(target_dt, "us")
        idx = np.searchsorted(ts_arr, t_target, side="right") - 1

        if idx < 0:
            return PriceLookupResult(None, None, None, InvalidReason.MISSING_TARGET_PRICE)

        snap_time_np = ts_arr[idx]
        age_seconds = float((t_target - snap_time_np) / np.timedelta64(1, "s"))

        if age_seconds < 0:
            return PriceLookupResult(None, None, None, InvalidReason.UNVERIFIED_TIMING)
        if age_seconds > self.max_quote_age_seconds:
            return PriceLookupResult(None, None, age_seconds, InvalidReason.STALE_PRICE)

        price = float(p_arr[idx])
        if price is None or price <= 0 or not np.isfinite(price):
            return PriceLookupResult(None, None, age_seconds, InvalidReason.NON_FINITE_PRICE)

        return PriceLookupResult(price, None, age_seconds, InvalidReason.NONE)
