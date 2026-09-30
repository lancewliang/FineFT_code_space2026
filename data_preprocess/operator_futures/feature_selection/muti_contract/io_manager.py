from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

from operator_futures.commodity.schema import get_reward_execution_columns
from operator_futures.data_quality import DataQualityValidator
from operator_futures.feature_selection.manifests import (
    FeatureSelectionContractRecord,
    FeatureSelectionManifest,
    FilteredOutputRecord,
)
from operator_futures.feature_selection.muti_contract.types import (
    FeatureSelectionPipelineConfig,
)

logger = logging.getLogger(__name__)

NON_STATE_COLUMNS = {
    "timestamp",
    "trading_day",
    "TradingDay",
    "symbol",
    "contract",
    "open",
    "high",
    "low",
    "open_interest",
}


def extract_state_features(df: pl.DataFrame, *, orderbook_depth: int) -> list[str]:
    reward = set(get_reward_execution_columns(orderbook_depth))
    schema = df.schema
    return [
        column
        for column in df.columns
        if column not in reward
        and column not in NON_STATE_COLUMNS
        and not column.endswith("timestamp")
        and not column.endswith("_right")
        and (schema[column].is_numeric() or schema[column] == pl.Boolean)
    ]


def load_feature_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"feature list file does not exist: {path}")
    values = np.load(path, allow_pickle=True).tolist()
    values = [str(value) for value in values]
    if not values:
        raise ValueError(f"feature list is empty: {path}")
    return values


class PipelineIOManager:
    def __init__(self, config: FeatureSelectionPipelineConfig):
        self.config = config
        self.input_dir = (
            Path(config.root_path)
            / config.split_path
            / config.target_freq
            / config.symbol
            / config.stage
        )
        self.output_dir = (
            Path(config.root_path)
            / config.save_path
            / config.target_freq
            / config.symbol
            / config.stage
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_stage_frames(self) -> dict[str, pl.DataFrame]:
        if not self.input_dir.exists():
            raise FileNotFoundError(
                f"split input directory does not exist: {self.input_dir}"
            )
        paths = sorted(self.input_dir.glob("*.feather"))
        if not paths:
            raise FileNotFoundError(
                f"split input directory contains no contract feather files: {self.input_dir}"
            )
        frames: dict[str, pl.DataFrame] = {}
        for path in paths:
            df = pl.read_ipc(path)
            if "timestamp" in df.columns:
                df = df.sort("timestamp")
            frames[path.stem] = df
        return frames

    def resolve_initial_feature_universe(
        self, frames: dict[str, pl.DataFrame]
    ) -> list[str]:
        if self.config.stage == "train":
            first_frame = next(iter(frames.values()))
            raw_universe = extract_state_features(
                first_frame, orderbook_depth=self.config.orderbook_depth
            )
        else:
            train_dir = (
                Path(self.config.root_path)
                / self.config.save_path
                / self.config.target_freq
                / self.config.symbol
                / "train"
            )
            rl_file = train_dir / "rl_state_features.npy"
            vae_file = train_dir / "vae_state_features.npy"
            raw_universe: list[str] = []
            seen: set[str] = set()
            for path in (rl_file, vae_file):
                if path.exists():
                    for feat in load_feature_list(path):
                        if feat not in seen:
                            seen.add(feat)
                            raw_universe.append(feat)
            if not raw_universe:
                raise FileNotFoundError(
                    f"Missing train feature files in {train_dir}: neither rl_state_features.npy nor vae_state_features.npy found"
                )
        return raw_universe

    def load_validation_outpost_frame(
        self, feature_universe: Sequence[str]
    ) -> pl.DataFrame | None:
        valid_dir = (
            Path(self.config.root_path)
            / self.config.split_path
            / self.config.target_freq
            / self.config.symbol
            / "valid"
        )
        if not valid_dir.exists():
            return None
        paths = sorted(valid_dir.glob("*.feather"))
        if not paths:
            return None
        outpost_path = paths[0]
        schema = pl.read_ipc_schema(outpost_path)
        needed = [feat for feat in feature_universe if feat in schema]
        if not needed:
            return None
        return pl.read_ipc(outpost_path, columns=needed)

    def validate_contract_frame(
        self,
        frame: pl.DataFrame,
        *,
        contract: str,
        feature_universe: list[str],
    ) -> None:
        DataQualityValidator.validate_no_illegal_values(
            frame,
            stage=f"{self.config.stage}_feature_selection_input",
            feature_name="FEATURE_SELECTION",
            contract=contract,
            trading_day="-",
            columns=["mark_price", *feature_universe],
        )

    def write_filtered_outputs(
        self,
        frames: dict[str, pl.DataFrame],
        selected_features: list[str],
    ) -> list[FilteredOutputRecord]:
        reward_columns = get_reward_execution_columns(self.config.orderbook_depth)
        outputs: list[FilteredOutputRecord] = []
        for contract, frame in frames.items():
            missing = [
                feature for feature in selected_features if feature not in frame.columns
            ]
            if missing:
                raise ValueError(
                    f"contract {contract} is missing selected feature columns: {missing}"
                )
            selected_set = set(selected_features)
            reward_present = [
                column
                for column in reward_columns
                if column in frame.columns and column not in selected_set
            ]
            filtered = frame.select([*reward_present, *selected_features]).with_columns(
                pl.lit(self.config.symbol).alias("symbol")
            )
            contract_dir = self.output_dir / contract
            contract_dir.mkdir(parents=True, exist_ok=True)
            output_path = contract_dir / "df.feather"
            filtered.write_ipc(output_path)
            outputs.append(
                FilteredOutputRecord(
                    contract=contract,
                    output_path=str(output_path),
                    output_row_count=filtered.height,
                    output_column_count=len(filtered.columns),
                )
            )
        return outputs

    def save_selected_features(self, selected_features: list[str]) -> Path:
        selected_file = self.output_dir / "rl_state_features.npy"
        np.save(selected_file, np.array(selected_features))
        return selected_file

    def save_dual_stream_features(
        self,
        vae_features: list[str],
        rl_features: list[str],
    ) -> tuple[Path, Path]:
        vae_file = self.output_dir / "vae_state_features.npy"
        rl_file = self.output_dir / "rl_state_features.npy"
        np.save(vae_file, np.array(vae_features))
        np.save(rl_file, np.array(rl_features))
        return vae_file, rl_file

    def save_manifest(self, manifest: FeatureSelectionManifest) -> Path:
        manifest_path = self.output_dir / "feature_selection_manifest.json"
        manifest.write_json(manifest_path)
        return manifest_path
