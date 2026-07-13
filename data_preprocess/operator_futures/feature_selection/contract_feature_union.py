import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from operator_futures.commodity.main_contract import load_main_contract_summary


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def build_union_state_features(feature_lists: Iterable[Sequence[str]]) -> list[str]:
    seen: set[str] = set()
    union: list[str] = []
    for features in feature_lists:
        for feature in features:
            name = str(feature)
            if name in seen:
                continue
            seen.add(name)
            union.append(name)
    return union


def _load_state_features(path: Path, contract: str) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing state_features.npy for contract {contract}: {path}"
        )
    return [str(item) for item in np.load(path, allow_pickle=True).tolist()]


def write_contract_feature_union(
    root_path: Path,
    summary_path: Path,
    symbol: str,
    target_freq: str,
    start_date: str,
    end_date: str,
    scale_save_path: str = "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE",
    save_path: str = "PREPROCESS_DATASET/commodity-futures/FEATURE_UNION",
) -> Path:
    summary = load_main_contract_summary(summary_path)
    date_range = f"{start_date}-{end_date}"
    scale_root = root_path / scale_save_path
    output_dir = root_path / save_path / symbol / target_freq / date_range

    contract_features: dict[str, list[str]] = {}
    contract_feature_paths: dict[str, str] = {}
    for contract in summary.contracts:
        feature_path = (
            scale_root
            / symbol
            / contract.contract
            / target_freq
            / date_range
            / "state_features.npy"
        )
        contract_features[contract.contract] = _load_state_features(
            feature_path, contract.contract
        )
        contract_feature_paths[contract.contract] = str(feature_path)

    union = build_union_state_features(contract_features.values())
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "state_features.npy", np.array(union))
    manifest = {
        "symbol": symbol,
        "target_freq": target_freq,
        "start_date": start_date,
        "end_date": end_date,
        "summary_path": str(summary_path),
        "contracts": list(contract_features),
        "contract_state_feature_paths": contract_feature_paths,
        "per_contract_feature_counts": {
            contract: len(features) for contract, features in contract_features.items()
        },
        "state_feature_count": len(union),
        "state_features": union,
    }
    (output_dir / "feature_union_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Wrote contract feature union: symbol=%s contracts=%d state_features=%d output_dir=%s",
        symbol,
        len(contract_features),
        len(union),
        output_dir,
    )
    return output_dir


parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=Path, default=Path("."))
parser.add_argument("--summary", type=Path, required=True)
parser.add_argument("--symbols", type=str, required=True)
parser.add_argument("--target_freq", type=str, required=True)
parser.add_argument("--start_date", type=str, required=True)
parser.add_argument("--end_date", type=str, required=True)
parser.add_argument(
    "--scale_save_path",
    type=str,
    default="PREPROCESS_DATASET/commodity-futures/SCALE_SAVE",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="PREPROCESS_DATASET/commodity-futures/FEATURE_UNION",
)


def main(args: argparse.Namespace) -> None:
    write_contract_feature_union(
        root_path=args.root_path,
        summary_path=args.summary,
        symbol=args.symbols,
        target_freq=args.target_freq,
        start_date=args.start_date,
        end_date=args.end_date,
        scale_save_path=args.scale_save_path,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    configure_logging()
    main(parser.parse_args())
