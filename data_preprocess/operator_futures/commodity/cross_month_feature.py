from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import argparse
import re

import math

import polars as pl

from .main_contract import load_main_contract_summary


CROSS_MONTH_PAIRING_MODES: list[str] = [
    "main_sub",
    "delivery_month_sequence",
]

CROSS_MONTH_FEATURE_COLUMNS: list[str] = [
    "cm_contract_role_main",
    "cm_contract_role_sub",
    "cm_contract_role_other",
    "cm_current_main_log_price_ratio",
    "cm_current_main_relative_price_spread",
    "cm_current_main_volume_share_current",
    "cm_current_main_open_interest_share_current",
    "cm_current_sub_log_price_ratio",
    "cm_current_sub_relative_price_spread",
    "cm_current_sub_volume_share_current",
    "cm_current_sub_open_interest_share_current",
    "cm_main_sub_log_price_ratio",
    "cm_main_sub_relative_price_spread",
    "cm_main_sub_volume_share_sub",
    "cm_main_sub_open_interest_share_sub",
    "cm_m1_m2_log_price_ratio",
    "cm_m2_m3_log_price_ratio",
    "cm_m1_m2_relative_price_spread",
    "cm_m2_m3_relative_price_spread",
    "cm_m1_m2_m3_butterfly_ratio",
    "cm_m1_m2_open_interest_share_m2",
    "cm_m2_m3_open_interest_share_m3",
    "cm_main_sub_log_price_spread_velocity_10m",
    "cm_open_interest_shift_speed_10m",
]

_ALLOWED_PRICE_PATTERNS: tuple[str, ...] = (
    "log_price_ratio",
    "relative_price_spread",
    "butterfly_ratio",
    "spread_zscore",
    "spread_velocity",
)


@dataclass(frozen=True)
class MainSubRoleResolution:
    role_trading_day: str
    current_role: str
    main_contract: str
    sub_contract: str


def validate_cross_month_feature_columns(columns: Sequence[str]) -> list[str]:
    feature_columns = [str(column) for column in columns]
    illegal_columns = [
        column for column in feature_columns if _is_illegal_price_feature(column)
    ]
    if illegal_columns:
        raise ValueError(
            "Cross-Month Term Structure Feature violates No Absolute Price Rule: "
            f"{illegal_columns}"
        )
    return feature_columns


def resolve_cross_month_feature_input(path: str | Path, *, required: bool) -> Path | None:
    feature_path = Path(path)
    if feature_path.exists():
        return feature_path
    if required:
        raise ValueError(f"missing required CROSS_MONTH_FEATURE file: {feature_path}")
    return None


def resolve_previous_main_sub_role(
    *,
    main_sub_roles: dict[str, dict[str, str]],
    trading_day: str,
    current_contract: str,
) -> MainSubRoleResolution | None:
    prior_days = sorted(day for day in main_sub_roles if str(day) < str(trading_day))
    if not prior_days:
        return None

    role_trading_day = prior_days[-1]
    roles = main_sub_roles[role_trading_day]
    main_contracts = [contract for contract, role in roles.items() if role == "main"]
    sub_contracts = [contract for contract, role in roles.items() if role == "sub"]
    if len(main_contracts) != 1 or len(sub_contracts) != 1:
        raise ValueError(
            f"invalid main/sub roles for previous trading_day {role_trading_day}"
        )

    return MainSubRoleResolution(
        role_trading_day=role_trading_day,
        current_role=roles.get(current_contract, "other"),
        main_contract=main_contracts[0],
        sub_contract=sub_contracts[0],
    )


def generate_main_sub_cross_month_features(
    *,
    current_contract: str,
    current_bars: pl.DataFrame,
    main_contract: str,
    main_bars: pl.DataFrame,
    sub_contract: str,
    sub_bars: pl.DataFrame,
    current_role: str,
) -> pl.DataFrame:
    if current_role not in {"main", "sub", "other"}:
        raise ValueError(f"current_role must be main, sub, or other: {current_role!r}")
    _validate_bar_columns("current_bars", current_bars)
    _validate_bar_columns("main_bars", main_bars)
    _validate_bar_columns("sub_bars", sub_bars)

    main_by_timestamp = _bars_by_timestamp(main_bars)
    sub_by_timestamp = _bars_by_timestamp(sub_bars)
    rows = []
    for current_row in current_bars.iter_rows(named=True):
        timestamp = current_row["timestamp"]
        main_row = main_by_timestamp.get(timestamp)
        sub_row = sub_by_timestamp.get(timestamp)
        row = {feature: 0.0 for feature in CROSS_MONTH_FEATURE_COLUMNS}
        row["timestamp"] = timestamp
        row[f"cm_contract_role_{current_role}"] = 1.0

        _add_pair_features(
            row,
            prefix="cm_current_main",
            left=current_row,
            right=main_row,
            same_contract=current_contract == main_contract,
            share_name="current",
        )
        _add_pair_features(
            row,
            prefix="cm_current_sub",
            left=current_row,
            right=sub_row,
            same_contract=current_contract == sub_contract,
            share_name="current",
        )
        _add_pair_features(
            row,
            prefix="cm_main_sub",
            left=main_row,
            right=sub_row,
            same_contract=main_contract == sub_contract,
            share_name="sub",
        )
        rows.append(row)

    return pl.DataFrame(rows).select(["timestamp"] + CROSS_MONTH_FEATURE_COLUMNS)


def generate_delivery_month_sequence_features(
    *,
    current_bars: pl.DataFrame,
    contract_bars: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    _validate_bar_columns("current_bars", current_bars)
    if len(contract_bars) < 3:
        raise ValueError("Delivery Month Sequence Pairing requires at least 3 contracts")

    ordered_contracts = sorted(
        contract_bars,
        key=lambda contract: (_extract_delivery_month(contract), contract),
    )
    m1_contract, m2_contract, m3_contract = ordered_contracts[:3]
    for contract in (m1_contract, m2_contract, m3_contract):
        _validate_bar_columns(f"contract_bars[{contract!r}]", contract_bars[contract])

    m1_by_timestamp = _bars_by_timestamp(contract_bars[m1_contract])
    m2_by_timestamp = _bars_by_timestamp(contract_bars[m2_contract])
    m3_by_timestamp = _bars_by_timestamp(contract_bars[m3_contract])

    rows = []
    for current_row in current_bars.iter_rows(named=True):
        timestamp = current_row["timestamp"]
        row = {feature: 0.0 for feature in CROSS_MONTH_FEATURE_COLUMNS}
        row["timestamp"] = timestamp
        m1_row = m1_by_timestamp.get(timestamp)
        m2_row = m2_by_timestamp.get(timestamp)
        m3_row = m3_by_timestamp.get(timestamp)

        _add_pair_features(
            row,
            prefix="cm_m1_m2",
            left=m1_row,
            right=m2_row,
            same_contract=m1_contract == m2_contract,
            share_name="m2",
        )
        _add_pair_features(
            row,
            prefix="cm_m2_m3",
            left=m2_row,
            right=m3_row,
            same_contract=m2_contract == m3_contract,
            share_name="m3",
        )
        _add_butterfly_feature(row, m1_row=m1_row, m2_row=m2_row, m3_row=m3_row)
        rows.append(row)

    return pl.DataFrame(rows).select(["timestamp"] + CROSS_MONTH_FEATURE_COLUMNS)


def generate_empty_cross_month_features(current_bars: pl.DataFrame) -> pl.DataFrame:
    _validate_bar_columns("current_bars", current_bars)
    rows = []
    for current_row in current_bars.iter_rows(named=True):
        row = {feature: 0.0 for feature in CROSS_MONTH_FEATURE_COLUMNS}
        row["timestamp"] = current_row["timestamp"]
        rows.append(row)
    return pl.DataFrame(rows).select(["timestamp"] + CROSS_MONTH_FEATURE_COLUMNS)


def write_cross_month_feature_for_day(
    *,
    root_path: str | Path,
    summary_path: str | Path,
    symbol: str,
    contract: str,
    target_freq: str,
    date: str,
    data_path: str = "PREPROCESS_DATASET/commodity-futures",
    save_path: str = "PREPROCESS_DATASET/commodity-futures/CROSS_MONTH_FEATURE",
) -> Path:
    root = Path(root_path)
    data_root = root / data_path
    summary = load_main_contract_summary(Path(summary_path))
    trading_day = date.replace("-", "")
    resolution = resolve_previous_main_sub_role(
        main_sub_roles=summary.main_sub_roles,
        trading_day=trading_day,
        current_contract=contract,
    )

    current_bars = _read_base_feature(
        data_root=data_root,
        symbol=symbol,
        contract=contract,
        target_freq=target_freq,
        date=date,
    )
    if resolution is None:
        main_sub_features = generate_empty_cross_month_features(current_bars)
    else:
        main_bars = _read_base_feature(
            data_root=data_root,
            symbol=symbol,
            contract=resolution.main_contract,
            target_freq=target_freq,
            date=date,
            required=False,
        )
        sub_bars = _read_base_feature(
            data_root=data_root,
            symbol=symbol,
            contract=resolution.sub_contract,
            target_freq=target_freq,
            date=date,
            required=False,
        )
        main_sub_features = generate_main_sub_cross_month_features(
            current_contract=contract,
            current_bars=current_bars,
            main_contract=resolution.main_contract,
            main_bars=main_bars,
            sub_contract=resolution.sub_contract,
            sub_bars=sub_bars,
            current_role=resolution.current_role,
        )

    contract_bars = _read_base_features_for_date(
        data_root=data_root,
        symbol=symbol,
        target_freq=target_freq,
        date=date,
        required=False,
    )
    if len(contract_bars) < 3:
        delivery_features = generate_empty_cross_month_features(current_bars)
    else:
        delivery_features = generate_delivery_month_sequence_features(
            current_bars=current_bars,
            contract_bars=contract_bars,
        )
    output = _merge_feature_frames(main_sub_features, delivery_features)
    output = output.with_columns(
        pl.col("cm_main_sub_log_price_ratio").diff(10).fill_null(0.0).alias("cm_main_sub_log_price_spread_velocity_10m"),
        pl.col("cm_main_sub_open_interest_share_sub").diff(10).fill_null(0.0).alias("cm_open_interest_shift_speed_10m"),
    )
    validate_cross_month_feature_columns(output.columns)

    output_path = (
        root
        / save_path
        / symbol
        / contract
        / target_freq
        / f"{date}.feather"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_ipc(output_path)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=Path, default=Path("."))
    parser.add_argument("--summary_path", type=Path, required=True)
    parser.add_argument("--symbol", "--symbols", dest="symbol", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--target_freq", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument(
        "--data_path",
        default="PREPROCESS_DATASET/commodity-futures",
    )
    parser.add_argument(
        "--save_path",
        default="PREPROCESS_DATASET/commodity-futures/CROSS_MONTH_FEATURE",
    )
    return parser


def main(args=None) -> Path:
    parsed = build_parser().parse_args(args)
    return write_cross_month_feature_for_day(
        root_path=parsed.root_path,
        summary_path=parsed.summary_path,
        symbol=parsed.symbol,
        contract=parsed.contract,
        target_freq=parsed.target_freq,
        date=parsed.date,
        data_path=parsed.data_path,
        save_path=parsed.save_path,
    )


def _is_illegal_price_feature(column: str) -> bool:
    normalized = column.lower()
    if not normalized.startswith("cm_"):
        return False
    if "price" not in normalized and "spread" not in normalized:
        return False
    return not any(pattern in normalized for pattern in _ALLOWED_PRICE_PATTERNS)


def _validate_bar_columns(name: str, frame: pl.DataFrame) -> None:
    required = {"timestamp", "close", "volume", "open_interest"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def _bars_by_timestamp(frame: pl.DataFrame) -> dict[object, dict[str, object]]:
    return {row["timestamp"]: row for row in frame.iter_rows(named=True)}


def _read_base_feature(
    *,
    data_root: Path,
    symbol: str,
    contract: str,
    target_freq: str,
    date: str,
    required: bool = True,
) -> pl.DataFrame:
    path = data_root / "BASE_FEATURE" / symbol / contract / target_freq / f"{date}.feather"
    if not path.exists():
        if not required:
            return pl.DataFrame(
                schema={
                    "timestamp": pl.Int64,
                    "close": pl.Float64,
                    "volume": pl.Float64,
                    "open_interest": pl.Float64,
                }
            )
        raise ValueError(f"missing required BASE_FEATURE file: {path}")
    frame = pl.read_ipc(path)
    _validate_bar_columns(str(path), frame)
    return frame


def _read_base_features_for_date(
    *,
    data_root: Path,
    symbol: str,
    target_freq: str,
    date: str,
    required: bool = True,
) -> dict[str, pl.DataFrame]:
    root = data_root / "BASE_FEATURE" / symbol
    if not root.exists():
        if not required:
            return {}
        raise ValueError(f"missing required BASE_FEATURE symbol directory: {root}")
    frames = {}
    for path in sorted(root.glob(f"*/{target_freq}/{date}.feather")):
        contract = path.parent.parent.name
        frames[contract] = pl.read_ipc(path)
    if required and len(frames) < 3:
        raise ValueError(
            f"Delivery Month Sequence Pairing requires at least 3 BASE_FEATURE files for {symbol} {date}"
        )
    return frames


def _merge_feature_frames(
    main_sub_features: pl.DataFrame,
    delivery_features: pl.DataFrame,
) -> pl.DataFrame:
    delivery_columns = [
        column
        for column in delivery_features.columns
        if column.startswith("cm_m")
    ]
    return main_sub_features.drop(delivery_columns).join(
        delivery_features.select(["timestamp"] + delivery_columns),
        on="timestamp",
        how="left",
    ).with_columns(pl.col(delivery_columns).fill_null(0.0)).select(
        ["timestamp"] + CROSS_MONTH_FEATURE_COLUMNS
    )


def _extract_delivery_month(contract: str) -> tuple[int, int]:
    match = re.search(r"(\d{4})$", contract)
    if match is None:
        raise ValueError(f"Cannot extract delivery month from contract {contract!r}")
    year_month = match.group(1)
    year = int(year_month[:2])
    month = int(year_month[2:])
    if not 1 <= month <= 12:
        raise ValueError(f"Invalid delivery month in contract {contract!r}")
    return year, month


def _add_pair_features(
    row: dict[str, object],
    *,
    prefix: str,
    left: dict[str, object] | None,
    right: dict[str, object] | None,
    same_contract: bool,
    share_name: str,
) -> None:
    if left is None or right is None:
        return

    left_close = float(left["close"])
    right_close = float(right["close"])
    if not same_contract and left_close > 0.0 and right_close > 0.0:
        row[f"{prefix}_log_price_ratio"] = math.log(left_close / right_close)
        row[f"{prefix}_relative_price_spread"] = (
            left_close - right_close
        ) / left_close

    left_volume = float(left["volume"])
    right_volume = float(right["volume"])
    share_numerator = right_volume if share_name in {"sub", "m2", "m3"} else left_volume
    share_other = left_volume if share_name in {"sub", "m2", "m3"} else right_volume
    row[f"{prefix}_volume_share_{share_name}"] = _share(
        share_numerator,
        share_other,
    )

    left_open_interest = float(left["open_interest"])
    right_open_interest = float(right["open_interest"])
    oi_share_numerator = (
        right_open_interest
        if share_name in {"sub", "m2", "m3"}
        else left_open_interest
    )
    oi_share_other = (
        left_open_interest
        if share_name in {"sub", "m2", "m3"}
        else right_open_interest
    )
    row[f"{prefix}_open_interest_share_{share_name}"] = _share(
        oi_share_numerator,
        oi_share_other,
    )


def _share(numerator: float, other: float) -> float:
    denominator = numerator + other
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _add_butterfly_feature(
    row: dict[str, object],
    *,
    m1_row: dict[str, object] | None,
    m2_row: dict[str, object] | None,
    m3_row: dict[str, object] | None,
) -> None:
    if m1_row is None or m2_row is None or m3_row is None:
        return
    m1_close = float(m1_row["close"])
    m2_close = float(m2_row["close"])
    m3_close = float(m3_row["close"])
    if m2_close <= 0.0:
        return
    row["cm_m1_m2_m3_butterfly_ratio"] = (
        2.0 * m2_close - m1_close - m3_close
    ) / m2_close


if __name__ == "__main__":
    main()
