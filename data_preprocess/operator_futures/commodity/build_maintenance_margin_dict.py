"""Build a single-step maintenance_margin_ratio_dict for a commodity and save it as .npy.

The output dict follows the schema consumed by `get_maintenance_margin` /
`calculate_maintenance_margin` in FineFT/env/env_class/futures_util.py, i.e.
keys are decimal position-value thresholds stored as strings and values are
``[margin_rate, subtract]`` pairs.

For a single-step (flat) dict we use a single very-large threshold so that
every realistic position value maps to the same maintenance margin rate.
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

try:
    from .config import get_commodity_config
except ImportError:  # allow running the script directly, e.g. via a debugger
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import get_commodity_config  # noqa: E402


logger = logging.getLogger(__name__)


DEFAULT_FLAT_THRESHOLD = "1000000000000"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def build_flat_maintenance_margin_dict(
    maintenance_margin_rate: float,
    threshold: str = DEFAULT_FLAT_THRESHOLD,
) -> dict:
    return {threshold: [maintenance_margin_rate, 0]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a single-step maintenance_margin_ratio_dict for a commodity"
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Commodity symbol; the rate defaults to the value in commodity config",
    )
    parser.add_argument(
        "--dataset_name",
        dest="dataset_name",
        default=None,
        help="Alias of --symbol, named after the per-commodity dataset folder",
    )
    parser.add_argument(
        "--maintenance_margin_rate",
        type=float,
        default=None,
        help="Override the maintenance margin rate from the commodity config",
    )
    parser.add_argument(
        "--threshold",
        default=DEFAULT_FLAT_THRESHOLD,
        help="Single-step threshold (string) for the flat maintenance margin dict",
    )
    parser.add_argument(
        "--output_root",
        default="dataset",
        help="Root directory that contains the per-commodity dataset folder",
    )
    parser.add_argument(
        "--filename",
        default="maintenance_margin_ratio_dict.npy",
        help="Output filename inside the per-commodity dataset folder",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    symbol = args.symbol or args.dataset_name or "fu"

    config = get_commodity_config(symbol)
    rate = (
        args.maintenance_margin_rate
        if args.maintenance_margin_rate is not None
        else config.maintenance_margin_rate
    )

    maintenance_margin_dict = build_flat_maintenance_margin_dict(
        maintenance_margin_rate=rate,
        threshold=args.threshold,
    )

    output_dir = Path(args.output_root) / config.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.filename

    np.save(output_path, maintenance_margin_dict)

    logger.info(
        "Wrote maintenance_margin_ratio_dict: symbol=%s rate=%s threshold=%s output=%s",
        config.symbol,
        rate,
        args.threshold,
        output_path,
    )


if __name__ == "__main__":
    main()
