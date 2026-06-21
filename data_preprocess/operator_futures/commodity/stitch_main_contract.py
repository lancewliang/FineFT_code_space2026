import argparse
from pathlib import Path

from .main_contract import build_main_contract_continuous_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch local commodity futures contracts into a main-contract series"
    )
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--commodity_name", required=True)
    parser.add_argument("--year", required=True)
    parser.add_argument("--symbol", default="fu")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stitched = build_main_contract_continuous_frame(
        Path(args.raw_root), args.commodity_name, args.year, args.symbol
    )
    stitched.to_csv(output, index=False)


if __name__ == "__main__":
    main()
