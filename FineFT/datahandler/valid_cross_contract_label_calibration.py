"""Atomic, directory-level calibration of Valid Dynamic Labels.

验证集跨合约动态标签校准模块 (Valid Dynamic Labels Calibration)。

【模块功能与整体脉络】
1. 目的：针对验证集目录中的所有合约数据，执行统一的全局斜率（slope）动态标签校准与 Feather 切片导出。
2. 代码脉络/执行步骤：
   - 输入校验与准备 (_validate_input, _prepare_raw_data): 校验关键列及数值有效性，生成 key_indicator。
   - 单合约转折点拟合 (_fit_contract): 使用 util.Worker (slice_and_merge 算法) 对单合约价格曲线切割合并，提取拐点与斜率。
   - 跨合约斜率池化与分位数校准 (build_valid_dataset): 汇总所有参与合约的拟合斜率，统一计算全局斜率分类阈值。
   - 标签映射与切片构建 (_build_contract_outputs): 结合全局阈值确定段落标签，按连续相同标签导出 Feather 切片。
   - 原子化发布与容错回滚 (_publish, _backup_published_outputs): 暂存目录生成 -> 移入备份 -> 覆盖发布 -> 清理暂存，确保中途失败可无缝回滚。
"""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from . import label_util as util
    from .manifests import (
        SliceContractManifest,
        SliceFileManifest,
        SliceLabelManifest,
        SliceManifest,
        SkippedContractManifest,
    )
except ImportError:
    import label_util as util
    from manifests import (
        SliceContractManifest,
        SliceFileManifest,
        SliceLabelManifest,
        SliceManifest,
        SkippedContractManifest,
    )


@dataclass
class _ContractFit:
    """保存单个合约转折点拟合与斜率提取的中间结果。"""

    contract: str  # 合约标识名称（如 fu2305）
    source_path: Path  # 原始数据文件路径
    processed_path: Path  # 预处理数据保存路径
    prepared: pd.DataFrame  # 预处理后的基础数据 DataFrame
    groups: list[tuple[Any, list[int], list[float], np.ndarray]]  # 拐点分组信息列表 (tic, points, group_slopes, row_positions)
    slopes: list[float]  # 该合约包含的所有段落斜率集合


def _contract_name(path: Path) -> str:
    """从文件名解析合约标识符（若包含 'df_' 前缀则剥离）。"""
    stem = path.stem
    return stem[3:] if stem.startswith("df_") else stem


def _validate_input(
    raw_data: pd.DataFrame, path: Path, timestamp: str, tic: str
) -> None:
    """校验输入原始数据的完整性与有效性。
    
    检查项：
    1. 必备列是否存在 (bid1_price, tic, timestamp)。
    2. bid1_price 是否包含非有限值 (NaN/Inf) 或零值。
    3. tic 资产标识列是否存在空值。
    4. 时间戳列是否包含空值或非法数值/空字符串。
    """
    required = ["bid1_price", tic]
    if timestamp != "index":
        required.append(timestamp)
    missing = [column for column in required if column not in raw_data.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    prices = pd.to_numeric(raw_data["bid1_price"], errors="coerce").to_numpy()
    if not np.isfinite(prices).all():
        raise ValueError(f"{path} contains non-finite bid1_price values")
    if (prices == 0).any():
        raise ValueError(f"{path} contains zero bid1_price values")
    if raw_data[tic].isna().any():
        raise ValueError(f"{path} contains null {tic} values")
    timestamp_values = raw_data[timestamp] if timestamp != "index" else raw_data.index
    if pd.isna(timestamp_values).any():
        raise ValueError(f"{path} contains null or non-finite {timestamp} values")
    if pd.api.types.is_numeric_dtype(timestamp_values):
        if not np.isfinite(np.asarray(timestamp_values, dtype=float)).all():
            raise ValueError(f"{path} contains null or non-finite {timestamp} values")
    elif pd.api.types.is_object_dtype(timestamp_values):
        if (timestamp_values.astype(str).str.strip() == "").any():
            raise ValueError(f"{path} contains empty {timestamp} values")


def _prepare_raw_data(
    raw_data: pd.DataFrame,
    path: Path,
    *,
    key_indicator: str,
    timestamp: str,
    tic: str,
) -> pd.DataFrame:
    """数据预处理：校验原始数据，确保 timestamp 存在于列中，并设定关键价格指标列 key_indicator。"""
    _validate_input(raw_data, path, timestamp, tic)
    prepared = raw_data.copy()
    if timestamp == "index":
        prepared[timestamp] = prepared.index
    # key_indicator (如 mark_price) 统一从 bid1_price 复制
    prepared[key_indicator] = prepared["bid1_price"]
    return prepared.reset_index(drop=True)


def _point(value: Any) -> int:
    """安全解析转折点位置索引（兼容列表、元组或 numpy 标量）。"""
    if isinstance(value, (list, tuple, np.ndarray)):
        return int(value[0])
    return int(value)


def _finite_slopes(values: Any, path: Path) -> list[float]:
    """提取并校验段落斜率，确保所有斜率为有限数值（无 NaN/Inf）。"""
    slopes = [float(np.asarray(value).reshape(-1)[0]) for value in values]
    if not slopes or not np.isfinite(slopes).all():
        raise ValueError(f"{path} produced no finite final segment slopes")
    return slopes


def _fit_contract(
    source_path: Path,
    prepared: pd.DataFrame,
    processed_path: Path,
    stage_root: Path,
    *,
    key_indicator: str,
    timestamp: str,
    tic: str,
    filter_strength: int,
    min_length_limit: int,
    merging_threshold: float,
    merging_metric: str,
    merging_dynamic_constraint: int,
    max_length_expectation: int,
    dynamic_number: int,
) -> _ContractFit:
    """对单合约进行转折点拟合与斜率提取。
    
    代码逻辑：
    1. 保存归一化前的预处理 Feather 文件到 processed 路径。
    2. 按 tic 分组，将各 tic 的 key_indicator 相对于首个价格进行首日归一化处理（计算相对涨跌）。
    3. 写入 worker 临时数据，并实例化 util.Worker 运行 slice_and_merge 算法拟合转折点。
    4. 提取各 tic 的转折点 (points) 与各段落斜率 (slopes)，并严格校验边界完整性与段落连续性。
    5. 返回包含拟合参数与斜率池的 _ContractFit 对象。
    """
    contract = _contract_name(source_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_feather(processed_path)
    
    # 构建 worker 专用归一化数据：将价格按各自 tic 的起点价格归一化
    worker_data = prepared.copy()
    for _, positions in worker_data.groupby(tic, sort=False).groups.items():
        first_price = worker_data.loc[positions[0], key_indicator]
        worker_data.loc[positions, key_indicator] = (
            worker_data.loc[positions, key_indicator] / first_price
        )
    worker_path = stage_root / "worker" / processed_path.name
    worker_path.parent.mkdir(parents=True, exist_ok=True)
    worker_data.to_feather(worker_path)
    
    # 实例化底层 C++/Python 切片与合并 Worker
    worker = util.Worker(
        str(worker_path),
        "slice_and_merge",
        filter_strength=filter_strength,
        key_indicator=key_indicator,
        timestamp=timestamp,
        tic=tic,
        labeling_method="slope",
        min_length_limit=min_length_limit,
        merging_threshold=merging_threshold,
        merging_metric=merging_metric,
        merging_dynamic_constraint=merging_dynamic_constraint,
    )
    # 执行拟合计算，导出转折点字典与标准化斜率列表
    worker.fit(dynamic_number, max_length_expectation, min_length_limit)

    groups: list[tuple[Any, list[int], list[float], np.ndarray]] = []
    slopes: list[float] = []
    # 遍历每个 tic，校验并整理拟合结果
    for worker_tic in worker.tics:
        points = [_point(value) for value in worker.turning_points_dict[worker_tic]]
        group_slopes = _finite_slopes(
            worker.norm_coef_list_dict[worker_tic], source_path
        )
        # 转折点数量必须恰好等于斜率数量 + 1
        if len(points) != len(group_slopes) + 1:
            raise ValueError(f"{source_path} has inconsistent segment boundaries")
        row_positions = np.flatnonzero(prepared[tic].to_numpy() == worker_tic)
        # 首尾转折点必须完全覆盖该 tic 的起始与终点索引
        if points[0] != 0 or points[-1] != len(row_positions):
            raise ValueError(f"{source_path} has invalid segment boundaries")
        groups.append((worker_tic, points, group_slopes, row_positions))
        slopes.extend(group_slopes)
    return _ContractFit(
        contract=contract,
        source_path=source_path,
        processed_path=processed_path,
        prepared=prepared,
        groups=groups,
        slopes=slopes,
    )


def _label_for_slope(slope: float, thresholds: list[float]) -> int:
    """根据全局斜率阈值列表，将连续斜率映射为离散的标签类别索引 (0 到 N-1)。"""
    for index, threshold in enumerate(thresholds):
        if slope <= threshold:
            return index
    return len(thresholds)


def _slope_statistics(slopes: list[float]) -> dict[str, float | int]:
    """计算斜率样本池的各项描述性统计量（数量、最小值、最大值、均值、标准差、中位数）。"""
    values = np.asarray(slopes, dtype=float)
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
    }


def _build_contract_outputs(
    fit: _ContractFit,
    stage_root: Path,
    valid_root: Path,
    *,
    dynamic_number: int,
    thresholds: list[float],
) -> SliceContractManifest:
    """根据全局斜率阈值，为单合约生成各标签类别的 Feather 切片文件，并构建 SliceContractManifest。
    
    代码逻辑：
    1. 为输入数据的每一行赋予由全局斜率阈值映射得出的离散 label。
    2. 校验所有行均已被成功归类（禁止出现 -1 未标记行）。
    3. 按连续相同 label 将数据拆分为独立的趋势段落切片并写入暂存路径 (df_{index}.feather)。
    4. 汇总各 label 对应的文件数与总行数，返回合约级别的清单对象。
    """
    labels = {
        f"label_{label}": SliceLabelManifest(label=f"label_{label}")
        for label in range(dynamic_number)
    }
    row_labels = np.full(len(fit.prepared), -1, dtype=int)
    # 根据转折点与全局斜率阈值将斜率转换为行级别 label
    for _, points, slopes, row_positions in fit.groups:
        for start, end, slope in zip(points[:-1], points[1:], slopes):
            label = _label_for_slope(slope, thresholds)
            row_labels[row_positions[start:end]] = label
    if (row_labels < 0).any():
        raise ValueError(f"{fit.source_path} has unaccounted input rows")

    contract_root = stage_root / fit.contract
    counters = [0] * dynamic_number
    previous_start = 0
    previous_label = int(row_labels[0])

    def write_segment(start: int, end: int, label: int) -> None:
        """内部辅助函数：将 [start, end) 范围内相同 label 的数据导出为单独的切片 Feather 文件。"""
        if end <= start:
            return
        label_name = f"label_{label}"
        stage_path = contract_root / label_name / f"df_{counters[label]}.feather"
        stage_path.parent.mkdir(parents=True, exist_ok=True)
        fit.prepared.iloc[start:end].reset_index(drop=True).to_feather(stage_path)
        final_path = valid_root / fit.contract / label_name / stage_path.name
        output_rows = int(end - start)
        labels[label_name].file_count += 1
        labels[label_name].total_row_count += output_rows
        labels[label_name].files.append(
            SliceFileManifest(path=str(final_path), output_row_count=output_rows)
        )
        counters[label] += 1

    # 遍历行级别 label，按连续标签段落切分导出
    for index in range(1, len(row_labels)):
        current_label = int(row_labels[index])
        if current_label != previous_label:
            write_segment(previous_start, index, previous_label)
            previous_start = index
            previous_label = current_label
    write_segment(previous_start, len(row_labels), previous_label)
    return SliceContractManifest(
        contract=fit.contract,
        processed_path=str(valid_root / "processed" / fit.processed_path.name),
        input_row_count=int(len(fit.prepared)),
        labels=labels,
    )


def _backup_published_outputs(
    valid_root: Path, staging_path: Path, backup_root: Path
) -> None:
    """原子更新准备：将当前 valid_root 中旧的发布文件/目录移入备份目录 backup_root，并清理残留暂存目录。"""
    backup_root.mkdir()
    for child in valid_root.iterdir():
        if child == staging_path or child == backup_root:
            continue
        # 清理之前可能中断留下的旧暂存目录
        if child.name.startswith(".valid-cross-contract-staging-"):
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
            continue
        # 移动旧清单或旧切片文件夹到备份目录
        if child.name == "slice_manifest.json" or child.is_dir():
            shutil.move(str(child), str(backup_root / child.name))


def build_valid_dataset(
    valid_dir: str | Path,
    *,
    dynamic_number: int = 5,
    labeling_method: str = "slope",
    key_indicator: str = "mark_price",
    timestamp: str = "timestamp",
    tic: str = "symbol",
    filter_strength: int = 1,
    min_length_limit: int = 288,
    merging_threshold: float = 0.0003,
    merging_metric: str = "DTW_distance",
    merging_dynamic_constraint: int = 1,
    max_length_expectation: int = 864,
    filter_padlen: int = 15,
) -> SliceManifest:
    """拟合并原子化发布所有验证集合约的最终斜率校准与动态标签切片。
    
    【主要工作流程】
    1. 校验入参规范，确保目录存在并扫描所有 *.feather 合约文件。
    2. 创建带 UUID 隔离的暂存目录 stage_root。
    3. 逐个读取并预处理合约数据：
       - 数据不足 filter_padlen 的合约，作为 skipped 记录跳过拟合。
       - 正常合约调用 _fit_contract 提取拟合转折点与斜率。
    4. 【全局跨合约校准核心】：池化所有参与合约的拟合斜率 (pooled_slopes)，
       调用 util.calculate_slope_thresholds 统一计算全局跨合约斜率分类分位数阈值 (thresholds)。
    5. 使用计算得到的全局阈值调用 _build_contract_outputs 生成各合约切片与描述 Manifest。
    6. 校验输出行数与输入行数的守恒关系。
    7. 构造完整的 SliceManifest，重建并排序标签，调用 _publish 实施原子化目录替换与清单写出。
    8. 异常防护：任何环节失败则自动清理暂存目录并抛出异常。
    """
    if labeling_method != "slope":
        raise ValueError(
            "production valid cross-contract calibration supports final slope "
            "labeling only"
        )
    if dynamic_number < 1:
        raise ValueError("dynamic_number must be positive")
    valid_root = Path(valid_dir).resolve()
    if not valid_root.is_dir():
        raise FileNotFoundError(f"missing valid directory: {valid_root}")
    source_paths = sorted(valid_root.glob("*.feather"), key=lambda path: path.name)
    if not source_paths:
        raise ValueError(f"valid directory has no contract files: {valid_root}")
    contract_names = [_contract_name(path) for path in source_paths]
    if len(set(contract_names)) != len(contract_names):
        raise ValueError("valid directory contains duplicate contract identities")

    stage_root = valid_root / f".valid-cross-contract-staging-{uuid.uuid4().hex}"
    stage_root.mkdir()
    manifest_path = valid_root / "slice_manifest.json"
    fits: list[_ContractFit] = []
    skipped: dict[str, SkippedContractManifest] = {}
    try:
        # 第一阶段：遍历所有合约数据进行独立拟合与斜率提取
        for source_path in source_paths:
            contract = _contract_name(source_path)
            raw_data = pd.read_feather(source_path)
            prepared = _prepare_raw_data(
                raw_data,
                source_path,
                key_indicator=key_indicator,
                timestamp=timestamp,
                tic=tic,
            )
            processed_path = (
                stage_root / "processed" / f"valid_processed_{contract}.feather"
            )
            # 若数据长度小于等于滤波填充长度，无法稳定分割，记录为跳过合约
            if len(prepared) <= filter_padlen:
                processed_path.parent.mkdir(parents=True, exist_ok=True)
                prepared.to_feather(processed_path)
                skipped[contract] = SkippedContractManifest(
                    contract=contract,
                    processed_path=str(valid_root / "processed" / processed_path.name),
                    reason=(
                        "insufficient rows for dynamic slicing: "
                        f"{len(prepared)} <= filter padlen {filter_padlen}"
                    ),
                    input_row_count=int(len(prepared)),
                )
                continue
            # 执行合约拟合
            fits.append(
                _fit_contract(
                    source_path,
                    prepared,
                    processed_path,
                    stage_root,
                    key_indicator=key_indicator,
                    timestamp=timestamp,
                    tic=tic,
                    filter_strength=filter_strength,
                    min_length_limit=min_length_limit,
                    merging_threshold=merging_threshold,
                    merging_metric=merging_metric,
                    merging_dynamic_constraint=merging_dynamic_constraint,
                    max_length_expectation=max_length_expectation,
                    dynamic_number=dynamic_number,
                )
            )

        # 清理 worker 计算过程中产生的临时文件
        shutil.rmtree(stage_root / "worker", ignore_errors=True)
        
        # 第二阶段：汇总跨合约斜率池，计算全局统一分类阈值
        pooled_slopes = [slope for fit in fits for slope in fit.slopes]
        if not pooled_slopes:
            raise ValueError("valid dataset produced zero final segments")
        thresholds = [
            float(value)
            for value in util.calculate_slope_thresholds(
                pooled_slopes, dynamic_number, risk_bond=0.1
            )
        ]
        
        # 第三阶段：根据全局阈值生成各合约的切片与 Manifest 结构
        contracts = {
            fit.contract: _build_contract_outputs(
                fit,
                stage_root,
                valid_root,
                dynamic_number=dynamic_number,
                thresholds=thresholds,
            )
            for fit in fits
        }
        # 校验各合约切片后的导出总行数是否与原输入完全对应
        for fit in fits:
            contract = contracts[fit.contract]
            if contract.total_row_count != contract.input_row_count:
                raise ValueError(f"{fit.source_path} failed output row accounting")

        # 第四阶段：构建完整的 SliceManifest 对象并规格化
        manifest = SliceManifest(
            valid_path=str(valid_root),
            contracts=contracts,
            skipped_contracts=skipped,
            calibration={
                "fit_scope": "valid_all_contracts",
                "participating_contracts": sorted(contracts),
                "skipped_contracts": sorted(skipped),
                "skipped_contract_details": [
                    skipped[contract].to_dict() for contract in sorted(skipped)
                ],
                "final_segment_count": len(pooled_slopes),
                "dynamic_number": dynamic_number,
                "labeling_method": "slope",
                "shared_thresholds": thresholds,
                "thresholds": thresholds,
                "slope_statistics": _slope_statistics(pooled_slopes),
            },
        )
        manifest.rebuild_labels()
        for label in range(dynamic_number):
            manifest.labels.setdefault(
                f"label_{label}", SliceLabelManifest(label=f"label_{label}")
            )
        manifest.sort()
        
        # 第五阶段：原子化替换与发布结果
        _publish(valid_root, stage_root, manifest_path, manifest)
        return manifest
    except Exception:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise


def _publish(
    valid_root: Path,
    stage_root: Path,
    manifest_path: Path,
    manifest: SliceManifest,
) -> None:
    """原子化发布：备份原目录内容，将暂存目录下的新生成文件移入发布目录，写出 manifest.json。如果出错则自动还原备份。"""
    backup_root = valid_root / f".valid-cross-contract-backup-{uuid.uuid4().hex}"
    manifest_stage_path = valid_root / f".slice-manifest-{uuid.uuid4().hex}.json"
    published_names: list[str] = []
    try:
        # 1. 将现有的发布结果移入备份目录
        _backup_published_outputs(valid_root, stage_root, backup_root)
        # 2. 将暂存目录 stage_root 中的新生成目录与文件移动到 valid_root 根目录下
        for child in sorted(stage_root.iterdir(), key=lambda path: path.name):
            shutil.move(str(child), str(valid_root / child.name))
            published_names.append(child.name)
        # 3. 写入新的 slice_manifest.json 格式化内容并执行原子替换
        manifest_stage_path.write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest_stage_path.replace(manifest_path)
    except Exception:
        # 发生异常时的回滚策略：删除已发布的新文件，恢复备份目录中的旧文件
        if manifest_stage_path.exists():
            manifest_stage_path.unlink()
        for name in published_names:
            published_path = valid_root / name
            if published_path.is_dir():
                shutil.rmtree(published_path)
            elif published_path.exists():
                published_path.unlink()
        if backup_root.exists():
            for child in sorted(backup_root.iterdir(), key=lambda path: path.name):
                shutil.move(str(child), str(valid_root / child.name))
        shutil.rmtree(backup_root, ignore_errors=True)
        raise
    else:
        # 发布成功：清理备份与暂存目录
        shutil.rmtree(backup_root, ignore_errors=True)
        shutil.rmtree(stage_root, ignore_errors=True)


# 函数别名，保持与外部调用的兼容性
build_valid_cross_contract_labels = build_valid_dataset
run_valid_dataset = build_valid_dataset


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--valid_dir", "--data_dir", dest="valid_dir", type=Path, required=True
    )
    parser.add_argument("--dynamic_number", type=int, default=5)
    parser.add_argument("--labeling_method", default="slope")
    parser.add_argument("--timestamp", default="timestamp")
    parser.add_argument("--tic", default="symbol")
    parser.add_argument("--min_length_limit", type=int, default=288)
    parser.add_argument("--merging_threshold", type=float, default=0.0003)
    parser.add_argument("--merging_dynamic_constraint", type=int, default=1)
    parser.add_argument("--max_length_expectation", type=int, default=864)
    return parser


def main(args: list[str] | None = None) -> None:
    """命令行主入口。"""
    parsed = build_parser().parse_args(args)
    build_valid_dataset(
        parsed.valid_dir,
        dynamic_number=parsed.dynamic_number,
        labeling_method=parsed.labeling_method,
        timestamp=parsed.timestamp,
        tic=parsed.tic,
        min_length_limit=parsed.min_length_limit,
        merging_threshold=parsed.merging_threshold,
        merging_dynamic_constraint=parsed.merging_dynamic_constraint,
        max_length_expectation=parsed.max_length_expectation,
    )


if __name__ == "__main__":
    main()
