import json
from pathlib import Path

import numpy as np

try:
    from .manifests import LabelArraySource, LabelContractSource, LabelTrainingManifest
except ImportError:
    from manifests import LabelArraySource, LabelContractSource, LabelTrainingManifest


RESERVED_VAE_DIRS = {"test", "train", "processed", "__pycache__"}


def label_name_from_index(label_index):
    return "label_{}".format(label_index)


def vae_data_dir(data_base_path, dataset_name):
    return Path(data_base_path) / dataset_name / "VAE_data"


def contract_dirs(root):
    if not root.exists():
        raise FileNotFoundError(f"missing VAE_data path: {root}")
    return [
        path
        for path in sorted(root.iterdir(), key=lambda item: item.name)
        if path.is_dir() and path.name not in RESERVED_VAE_DIRS
    ]


def load_2d_array(path, contract):
    data = np.load(path)
    if data.ndim != 2:
        raise ValueError(
            f"{path} for contract {contract} must be a two-dimensional array"
        )
    if data.shape[0] == 0:
        raise ValueError(f"{path} for contract {contract} has no samples")
    return data


def discover_label_sources(data_base_path, dataset_name, label_index):
    root = vae_data_dir(data_base_path, dataset_name)
    label_name = label_name_from_index(label_index)
    included = []
    missing = []
    for contract_dir in contract_dirs(root):
        source_file = contract_dir / f"{label_name}.npy"
        if source_file.exists():
            included.append(
                LabelArraySource(
                    contract=contract_dir.name,
                    source_file=str(source_file),
                )
            )
        else:
            missing.append(contract_dir.name)
    if not included:
        raise FileNotFoundError(
            f"no arrays found for {label_name} under {root}/<contract>/{label_name}.npy"
        )
    return included, missing


def materialize_label_training_data(data_base_path, dataset_name, label_index):
    root = vae_data_dir(data_base_path, dataset_name)
    label_name = label_name_from_index(label_index)
    included_sources, missing_contracts = discover_label_sources(
        data_base_path, dataset_name, label_index
    )
    arrays = []
    included_contracts = []
    feature_dim = None
    for source in included_sources:
        array = load_2d_array(Path(source.source_file), source.contract)
        if feature_dim is None:
            feature_dim = int(array.shape[1])
        elif int(array.shape[1]) != feature_dim:
            raise ValueError(
                "feature dimension mismatch for "
                f"{source.contract} at {source.source_file}: "
                f"expected {feature_dim}, got {array.shape[1]}"
            )
        arrays.append(array)
        included_contracts.append(
            LabelContractSource(
                contract=source.contract,
                source_file=source.source_file,
                sample_count=int(array.shape[0]),
            )
        )

    merged = np.concatenate(arrays, axis=0)
    if merged.shape[0] == 0:
        raise ValueError(f"merged training set for {label_name} has no samples")

    train_dir = root / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    merged_path = train_dir / f"{label_name}.npy"
    manifest_path = train_dir / f"{label_name}_manifest.json"
    np.save(merged_path, merged)
    manifest = LabelTrainingManifest(
        dataset_name=dataset_name,
        label=label_name,
        merged_path=str(merged_path),
        total_samples=int(merged.shape[0]),
        feature_dim=int(merged.shape[1]),
        included_contracts=included_contracts,
        missing_contracts=missing_contracts,
    )
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest.to_dict(), file, ensure_ascii=False, indent=2)
    return manifest
