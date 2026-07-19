import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd


STAGES = ("train", "valid", "test")


def _stage_input_path(input_root, symbol, target_freq, stage, contract):
    return Path(input_root) / symbol / target_freq / stage / f"{contract}.feather"


def _stage_output(output_root, symbol, set_name, contract):
    return Path(output_root) / symbol / set_name / f"{contract}.feather"


def load_dataset_split_manifest(path, symbol, target_freq):
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset split manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if manifest.get("symbol") != symbol:
        raise ValueError(
            "dataset split manifest symbol mismatch: "
            f"expected={symbol} actual={manifest.get('symbol')}"
        )
    if manifest.get("target_freq") != target_freq:
        raise ValueError(
            "dataset split manifest target_freq mismatch: "
            f"expected={target_freq} actual={manifest.get('target_freq')}"
        )
    sets = manifest.get("sets")
    if not isinstance(sets, dict):
        raise ValueError("dataset split manifest missing sets")
    for stage in STAGES:
        stage_info = sets.get(stage)
        if not isinstance(stage_info, dict):
            raise ValueError(f"dataset split manifest missing sets.{stage}")
        if "contracts" not in stage_info or not isinstance(
            stage_info["contracts"], list
        ):
            raise ValueError(
                f"dataset split manifest sets.{stage}.contracts must be a list"
            )
        for item in stage_info["contracts"]:
            if not isinstance(item, dict) or not isinstance(item.get("contract"), str):
                raise ValueError(
                    f"dataset split manifest sets.{stage}.contracts items need contract"
                )
    return manifest


def _build_slice_plan(
    row_count, output_root, symbol, contract, start_index, chunk_length, early_stop
):
    outputs = []
    if row_count <= 0:
        return outputs, start_index

    row_start = 0
    index = start_index
    while row_start < row_count:
        row_end = min(row_start + chunk_length + early_stop, row_count)
        outputs.append(
            {
                "index": index,
                "contract": contract,
                "path": str(
                    Path(output_root)
                    / symbol
                    / "train"
                    / "slice"
                    / f"df_{index}.feather"
                ),
                "source_output": str(_stage_output(output_root, symbol, "train", contract)),
                "row_start": row_start,
                "row_end": row_end,
            }
        )
        index += 1
        row_start += chunk_length
    return outputs, index


def build_dataset_manifest(
    split_manifest,
    dataset_split_manifest_path,
    input_root,
    state_features_path,
    output_root,
    symbol,
    target_freq,
    chunk_length,
    early_stop,
):
    manifest = {
        "symbol": symbol,
        "target_freq": target_freq,
        "dataset_split_manifest_path": str(dataset_split_manifest_path),
        "state_features_source_path": str(state_features_path),
        "state_features_path": str(Path(output_root) / symbol / "state_features.npy"),
        "sets": {},
    }

    next_slice = 0
    for set_name in STAGES:
        split_set = split_manifest["sets"].get(set_name, {})
        set_contracts = []
        for item in split_set.get("contracts", []):
            contract_name = item["contract"]
            record = {
                "contract": contract_name,
                "input_path": str(
                    _stage_input_path(input_root, symbol, target_freq, set_name, contract_name)
                ),
                "output_path": str(
                    _stage_output(output_root, symbol, set_name, contract_name)
                ),
            }
            if "range" in item:
                record["range"] = item["range"]
            elif "range" in split_set:
                record["range"] = split_set["range"]
            if "trading_days" in item:
                record["trading_days"] = item["trading_days"]
            if set_name == "train":
                row_count = int(item.get("output_row_count", 0))
                slices, next_slice = _build_slice_plan(
                    row_count,
                    output_root,
                    symbol,
                    contract_name,
                    next_slice,
                    chunk_length,
                    early_stop,
                )
                record["slice_outputs"] = slices
            set_contracts.append(record)
        manifest["sets"][set_name] = {
            "range": split_set.get("range"),
            "contracts": set_contracts,
            "skipped_contracts": split_set.get("skipped_contracts", []),
        }
    return manifest


def write_stage_datasets(manifest):
    state_features_source_path = Path(manifest["state_features_source_path"])
    if not state_features_source_path.exists():
        raise FileNotFoundError(
            f"Missing selected state_features.npy: {state_features_source_path}"
        )
    state_features = np.load(state_features_source_path, allow_pickle=True).tolist()
    if not state_features:
        raise ValueError(f"state feature list is empty: {state_features_source_path}")

    state_features_path = Path(manifest["state_features_path"])
    state_features_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(state_features_source_path, state_features_path)

    for stage, set_info in manifest["sets"].items():
        contracts_total_count = 0
        for contract in set_info["contracts"]:
            input_path = Path(contract["input_path"])
            if not input_path.exists():
                raise FileNotFoundError(
                    "Missing SCALE_SAVE file for "
                    f"stage={stage} contract={contract['contract']}: {input_path}"
                )
            output_path = Path(contract["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(input_path, output_path)
            output_df = pd.read_feather(output_path)
            if output_df.empty:
                raise ValueError(
                    "copied empty stage dataset: "
                    f"stage={stage} contract={contract['contract']} output={output_path}"
                )
            contract["output_row_count"] = int(len(output_df))
            contracts_total_count += contract["output_row_count"]
        set_info["contracts_total_count"] = contracts_total_count


def rebuild_train_slice_plan(manifest, chunk_length, early_stop):
    next_index = 0
    for contract in manifest["sets"]["train"]["contracts"]:
        output_path = Path(contract["output_path"])
        slice_dir = output_path.parent / "slice"
        row_count = int(contract.get("output_row_count", 0))
        row_start = 0
        slice_outputs = []
        while row_start < row_count:
            row_end = min(row_start + chunk_length + early_stop, row_count)
            slice_outputs.append(
                {
                    "index": next_index,
                    "contract": contract["contract"],
                    "path": str(slice_dir / f"df_{next_index}.feather"),
                    "source_output": str(output_path),
                    "row_start": row_start,
                    "row_end": row_end,
                }
            )
            next_index += 1
            row_start += chunk_length
        contract["slice_outputs"] = slice_outputs


def write_train_slices(manifest):
    expected_index = 0
    for contract in manifest["sets"]["train"]["contracts"]:
        df = pd.read_feather(contract["output_path"])
        for slice_info in contract.get("slice_outputs", []):
            if "index" in slice_info and int(slice_info["index"]) != expected_index:
                raise ValueError("train slice indices must be continuous")
            row_start = int(slice_info["row_start"])
            row_end = int(slice_info["row_end"])
            sliced = df.iloc[row_start:row_end].reset_index(drop=True)
            if sliced.empty:
                continue
            output_path = Path(slice_info["path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sliced.to_feather(output_path)
            slice_info["output_row_count"] = int(len(sliced))
            expected_index += 1


def run_dataset_generation(
    dataset_split_manifest_path,
    input_root,
    state_features_path,
    output_root,
    symbol,
    target_freq,
    chunk_length=3200,
    early_stop=320,
):
    split_manifest = load_dataset_split_manifest(
        dataset_split_manifest_path,
        symbol=symbol,
        target_freq=target_freq,
    )
    dataset_root = Path(output_root) / symbol
    manifest = build_dataset_manifest(
        split_manifest=split_manifest,
        dataset_split_manifest_path=dataset_split_manifest_path,
        input_root=input_root,
        state_features_path=state_features_path,
        output_root=output_root,
        symbol=symbol,
        target_freq=target_freq,
        chunk_length=chunk_length,
        early_stop=early_stop,
    )
    dataset_root.mkdir(parents=True, exist_ok=True)
    write_stage_datasets(manifest)
    rebuild_train_slice_plan(manifest, chunk_length=chunk_length, early_stop=early_stop)
    write_train_slices(manifest)
    (dataset_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_split_manifest_path", type=Path, required=True)
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--state_features_path", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--target_freq", type=str, required=True)
    parser.add_argument("--chunk_length", type=int, default=3200)
    parser.add_argument("--early_stop", type=int, default=320)
    return parser


def main(args=None):
    parsed = build_parser().parse_args(args)
    run_dataset_generation(
        dataset_split_manifest_path=parsed.dataset_split_manifest_path,
        input_root=parsed.input_root,
        state_features_path=parsed.state_features_path,
        output_root=parsed.output_root,
        symbol=parsed.symbol,
        target_freq=parsed.target_freq,
        chunk_length=parsed.chunk_length,
        early_stop=parsed.early_stop,
    )


if __name__ == "__main__":
    main()
