import argparse
import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import polars as pl

from operator_futures.commodity.main_contract import load_main_contract_summary
from operator_futures.feature_selection.ic_correlation import select_reward_state_features
from operator_futures.feature_selection.manifests import (
    ContractOutputShape,
    FeatureUnionManifest,
    FeatureUnionResult,
)


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


def _feature_path(
    root_path: Path,
    base_path: str,
    symbol: str,
    contract: str,
    target_freq: str,
    date_range: str,
    file_name: str,
) -> Path:
    return root_path / base_path / symbol / contract / target_freq / date_range / file_name


def _all_feature_path(
    root_path: Path,
    all_feature_path: str,
    symbol: str,
    contract: str,
    target_freq: str,
    date_range: str,
) -> Path:
    return root_path / all_feature_path / symbol / contract / target_freq / f"{date_range}.feather"


def _missing_features(df: pl.DataFrame, required: Sequence[str]) -> list[str]:
    available = set(df.columns)
    return [feature for feature in required if feature not in available]


def write_contract_feature_union(
    root_path: Path,
    summary_path: Path,
    symbol: str,
    target_freq: str,
    start_date: str,
    end_date: str,
    scale_save_path: str = "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE",
    save_path: str = "PREPROCESS_DATASET/commodity-futures/FEATURE_UNION",
    candidate_path: str | None = None,
    all_feature_path: str = "PREPROCESS_DATASET/commodity-futures/ALL_FEATURE",
    ic_result_path: str = "PREPROCESS_DATASET/commodity-futures/IC_RESULT",
    finalize_filtered_df: bool = False,
    market_type: str = "commodity_futures",
    orderbook_depth: int = 5,
) -> FeatureUnionResult:
    summary = load_main_contract_summary(summary_path)
    date_range = f"{start_date}-{end_date}"
    output_dir = root_path / save_path / symbol / target_freq / date_range
    feature_base_path = candidate_path if candidate_path is not None else scale_save_path
    feature_file_name = (
        "state_features_candidate.npy" if candidate_path is not None else "state_features.npy"
    )

    contract_features: dict[str, list[str]] = {}
    contract_feature_paths: dict[str, str] = {}
    for contract in summary.contracts:
        feature_path = _feature_path(
            root_path,
            feature_base_path,
            symbol,
            contract.contract,
            target_freq,
            date_range,
            feature_file_name,
        )
        contract_features[contract.contract] = _load_state_features(
            feature_path, contract.contract
        )
        contract_feature_paths[contract.contract] = str(feature_path)

    union = build_union_state_features(contract_features.values())
    if finalize_filtered_df and not union:
        raise ValueError("Feature union is empty; cannot finalize filtered IC_RESULT files")

    per_contract_outputs: dict[str, str] = {}
    per_contract_output_shapes: dict[str, ContractOutputShape] = {}
    finalized_frames: dict[str, tuple[Path, pl.DataFrame]] = {}
    if finalize_filtered_df:
        for contract in summary.contracts:
            input_path = _all_feature_path(
                root_path,
                all_feature_path,
                symbol,
                contract.contract,
                target_freq,
                date_range,
            )
            if not input_path.exists():
                raise FileNotFoundError(
                    f"Missing ALL_FEATURE input for contract {contract.contract}: {input_path}"
                )
            df = pl.read_ipc(input_path)
            reward_features, _ = select_reward_state_features(
                df, market_type=market_type, orderbook_depth=orderbook_depth
            )
            required_columns = [*reward_features, *union]
            missing = _missing_features(df, required_columns)
            if missing:
                raise ValueError(
                    f"Contract {contract.contract} missing union features: {missing}"
                )
            contract_output_dir = (
                root_path
                / ic_result_path
                / symbol
                / contract.contract
                / target_freq
                / date_range
            )
            out = df.select(required_columns)
            finalized_frames[contract.contract] = (contract_output_dir, out)
            per_contract_outputs[contract.contract] = str(contract_output_dir / "df.feather")
            per_contract_output_shapes[contract.contract] = ContractOutputShape(
                rows=out.height,
                columns=len(out.columns),
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "state_features.npy", np.array(union))
    for contract, (contract_output_dir, out) in finalized_frames.items():
        contract_output_dir.mkdir(parents=True, exist_ok=True)
        out.write_ipc(contract_output_dir / "df.feather")
        np.save(contract_output_dir / "state_features.npy", np.array(union))

    manifest = FeatureUnionManifest(
        symbol=symbol,
        target_freq=target_freq,
        start_date=start_date,
        end_date=end_date,
        summary_path=str(summary_path),
        contracts=list(contract_features),
        contract_state_feature_paths=contract_feature_paths,
        per_contract_feature_counts={
            contract: len(features) for contract, features in contract_features.items()
        },
        state_feature_count=len(union),
        state_features=union,
        candidate_source_path=candidate_path,
        all_feature_path=all_feature_path,
        ic_result_path=ic_result_path,
        finalize_filtered_df=finalize_filtered_df,
        per_contract_output_paths=per_contract_outputs,
        per_contract_output_shapes=per_contract_output_shapes,
    )
    manifest.write_json(output_dir / "feature_union_manifest.json")
    logger.info(
        "Wrote contract feature union: symbol=%s contracts=%d state_features=%d output_dir=%s",
        symbol,
        len(contract_features),
        len(union),
        output_dir,
    )
    return FeatureUnionResult(output_dir=output_dir, manifest=manifest)


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
parser.add_argument("--candidate_path", type=str, default=None)
parser.add_argument(
    "--all_feature_path",
    type=str,
    default="PREPROCESS_DATASET/commodity-futures/ALL_FEATURE",
)
parser.add_argument(
    "--ic_result_path",
    type=str,
    default="PREPROCESS_DATASET/commodity-futures/IC_RESULT",
)
parser.add_argument("--finalize_filtered_df", action="store_true")
parser.add_argument(
    "--market_type",
    type=str,
    default="commodity_futures",
    choices=["crypto_futures", "commodity_futures"],
)
parser.add_argument("--orderbook_depth", type=int, default=5)


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
        candidate_path=args.candidate_path,
        all_feature_path=args.all_feature_path,
        ic_result_path=args.ic_result_path,
        finalize_filtered_df=args.finalize_filtered_df,
        market_type=args.market_type,
        orderbook_depth=args.orderbook_depth,
    )


if __name__ == "__main__":
    configure_logging()
    main(parser.parse_args())
