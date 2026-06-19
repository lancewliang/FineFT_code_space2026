#!/usr/bin/env python3
"""
提取主力合约数据脚本

功能：
1. 扫描 data/原始下载/{品种} 目录下的所有数据文件
2. 根据配置的主力月份和成交量识别每天的主力合约
3. 将主力合约数据复制到新目录结构：data/{品种}/{年}/{合约}-{年}-{月}-{日期}.csv

新目录结构示例：
  data/铝/2023/al2305-2023-01-03.csv

使用方法：
  python scripts/step2_extract_main_contract_data.py
  python scripts/step2_extract_main_contract_data.py --commodity 燃料油
  python scripts/step2_extract_main_contract_data.py --year 2023
  python scripts/step2_extract_main_contract_data.py --commodity 燃料油 --year 2023 --month 01
"""

import polars as pl
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== 配置 ====================
# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_DATA_ROOT = PROJECT_ROOT / "data" / "原始下载"
TARGET_DATA_ROOT = PROJECT_ROOT / "data"

# 各品种的主力合约月份配置
MAIN_CONTRACT_MONTHS = {
    "铝": [1, 3, 5, 7, 9, 11],
    # 上期所燃料油(FU)合约月份为1-12月
    "燃料油": list(range(1, 13))
}

# 品种代码映射
COMMODITY_SYMBOL_MAP = {
    "铝": "al",
    "燃料油": "fu"
}

# ==================== 辅助函数 ====================
def get_available_contracts(date_path: Path,) -> List[str]:
    """获取指定日期和数据类型下可用的合约列表"""
    data_type_path = date_path 

    if not data_type_path.exists():
        return []

    # 获取所有CSV文件
    csv_files = list(data_type_path.glob("*.csv"))

    # 提取合约代码（去掉.csv后缀）
    contracts = [f.stem for f in csv_files]

    return sorted(contracts)



def calculate_contract_volume(csv_path: Path) -> float:
    """计算合约的成交量"""
    try:
        df = pl.read_csv(csv_path)

        if "Volume" in df.columns:
            # 五档行情数据：使用Volume字段的变化量
            total_volume = df["Volume"].max() - df["Volume"].min()
        elif "volume" in df.columns:
            total_volume = df["volume"].max() - df["volume"].min()
        else:
            # 如果没有成交量字段，使用持仓量变化
            if "OpenInterest" in df.columns:
                total_volume = df["OpenInterest"].max() - df["OpenInterest"].min()
            else:
                total_volume = 0

        return float(total_volume)

    except Exception as e:
        logger.warning(f"读取文件失败 {csv_path}: {e}")
        return 0.0


def identify_main_contract(
    date_path: Path,
    commodity: str,
    volume_threshold: float = 0.5
) -> Tuple[str, Dict[str, float]]:
    """
    识别指定日期的主力合约

    Args:
        date_path: 日期目录路径
        commodity: 品种名称
        volume_threshold: 成交量阈值比例

    Returns:
        主力合约代码和所有合约的成交量字典
    """
    # 获取品种的主力月份配置
    main_months = MAIN_CONTRACT_MONTHS.get(commodity, [])
    if not main_months:
        logger.warning(f"品种 {commodity} 没有配置主力月份")
        return None, {}

    commodity_symbol = COMMODITY_SYMBOL_MAP.get(commodity)
    if not commodity_symbol:
        logger.warning(f"品种 {commodity} 没有配置代码映射")
        return None, {}

    # 从期货成交量统计数据识别主力合约
    contracts = get_available_contracts(date_path)

    if not contracts:
        logger.warning(f"日期 {date_path.name} 没有找到期货成交量统计数据")
        return None, {}

    # 筛选符合主力月份的合约
    main_month_contracts = []
    for contract in contracts:
        if len(contract) >= 4:
            if not contract.lower().startswith(commodity_symbol.lower()):
                continue
            try:
                month = int(contract[-2:])
                if month in main_months:
                    main_month_contracts.append(contract)
            except ValueError:
                continue

    if not main_month_contracts:
        logger.warning(f"日期 {date_path.name} 没有找到符合主力月份的合约")
        return None, {}

    # 计算每个合约的成交量
    contract_volumes = {}
    for contract in main_month_contracts:
        csv_path = date_path / f"{contract}.csv"
        if csv_path.exists():
            volume = calculate_contract_volume(csv_path)
            contract_volumes[contract] = volume

    if not contract_volumes:
        logger.warning(f"日期 {date_path.name} 无法读取任何合约数据")
        return None, {}

    # 找到最大成交量
    max_volume = max(contract_volumes.values())

    if max_volume == 0:
        logger.warning(f"日期 {date_path.name} 的所有合约成交量都为0")
        return None, contract_volumes

    # 计算阈值
    threshold_volume = max_volume * volume_threshold

    # 找到成交量最大的合约作为主力合约
    main_contract = max(contract_volumes, key=contract_volumes.get)

    return main_contract, contract_volumes


def copy_main_contract_data(
    source_date_path: Path,
    commodity: str,
    main_contract: str,
    date_str: str
):
    """
    复制主力合约数据到新目录结构

    Args:
        source_date_path: 源日期目录
        commodity: 品种名称
        main_contract: 主力合约代码
        date_str: 日期字符串 YYYYMMDD
    """
    # 解析日期
    try:
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        year = date_obj.strftime("%Y")
        month = date_obj.strftime("%m")
        day = date_obj.strftime("%d")
    except ValueError as e:
        logger.error(f"日期格式错误: {date_str}")
        return

    # 复制每种数据类型的文件
    source_file = source_date_path / f"{main_contract}.csv"

    if not source_file.exists():
        logger.warning(f"文件不存在: {source_file}")
        return

    # 目标目录: data/{品种}/{年}/
    target_dir = TARGET_DATA_ROOT / commodity / year
    target_dir.mkdir(parents=True, exist_ok=True)

    # 目标文件名: {合约}-{年}-{月}-{日}.csv
    target_filename = f"{main_contract}-{year}-{month}-{day}.csv"
    target_file = target_dir / target_filename

    # 复制文件
    try:
        shutil.copy2(source_file, target_file)
        logger.debug(f"  复制: {source_file.name} -> {target_file}")
    except Exception as e:
        logger.error(f"  复制失败: {e}")


def process_commodity(
    commodity: str,
    year: str = None,
    month: str = None,
    volume_threshold: float = 0.5
):
    """
    处理指定品种的数据

    Args:
        commodity: 品种名称
        year: 年份（可选，不指定则处理所有年份）
        month: 月份（可选，不指定则处理所有月份）
        volume_threshold: 成交量阈值比例
    """
    logger.info(f"{'='*60}")
    logger.info(f"开始处理品种: {commodity}")
    logger.info(f"{'='*60}")

    # 品种源目录
    commodity_source_dir = SOURCE_DATA_ROOT / commodity

    if not commodity_source_dir.exists():
        logger.error(f"品种目录不存在: {commodity_source_dir}")
        return

    # 统计信息
    total_dates = 0
    processed_dates = 0
    skipped_dates = 0
    contract_stats = {}  # 统计每个合约作为主力合约的天数

    # prev_date_dir 用于跨月/跨年传递前一个交易日
    prev_date_dir = None

    # 遍历年份
    year_dirs = sorted(commodity_source_dir.glob("*"))
    if year:
        year_dirs = [d for d in year_dirs if d.name == year]

    for year_dir in year_dirs:
        if not year_dir.is_dir():
            continue

        logger.info(f"\n处理年份: {year_dir.name}")

        # 遍历月份
        month_dirs = sorted(year_dir.glob("*"))
        if month:
            month_dirs = [d for d in month_dirs if d.name == month]

        for month_dir in month_dirs:
            if not month_dir.is_dir():
                continue

            logger.info(f"  处理月份: {month_dir.name}")

            # 遍历日期，收集所有日期目录
            date_dirs = sorted([d for d in month_dir.glob("*") if d.is_dir()])

            for i, date_dir in enumerate(date_dirs):
                total_dates += 1
                date_str = date_dir.name

                # 用前一个交易日的成交量来识别今天的主力合约
                if i == 0 and prev_date_dir is None:
                    logger.warning(f"    {date_str}: 无前一交易日数据，跳过")
                    skipped_dates += 1
                    continue

                # 取前一个交易日目录（优先用上个月最后一天，否则用本月前一天）
                ref_date_dir = prev_date_dir if i == 0 else date_dirs[i - 1]

                # 用前一交易日的数据识别主力合约
                main_contract, contract_volumes = identify_main_contract(
                    ref_date_dir, commodity, volume_threshold
                )

                if main_contract is None:
                    logger.warning(f"    {date_str}: 未识别到主力合约（参考日 {ref_date_dir.name}）")
                    skipped_dates += 1
                    continue

                # 统计主力合约
                if main_contract not in contract_stats:
                    contract_stats[main_contract] = 0
                contract_stats[main_contract] += 1

                # 显示识别结果
                volume_info = ", ".join(
                    f"{c}: {v:,.0f}" for c, v in
                    sorted(contract_volumes.items(), key=lambda x: x[1], reverse=True)
                )
                logger.info(f"    {date_str}: 主力合约 {main_contract}（参考日 {ref_date_dir.name}，{volume_info}）")

                # 复制主力合约数据
                copy_main_contract_data(date_dir, commodity, main_contract, date_str)

                processed_dates += 1

            # 记录本月最后一个交易日，供下个月第一天使用
            if date_dirs:
                prev_date_dir = date_dirs[-1]

    # 输出统计信息
    logger.info(f"\n{'='*60}")
    logger.info(f"处理完成")
    logger.info(f"{'='*60}")
    logger.info(f"总日期数: {total_dates}")
    logger.info(f"成功处理: {processed_dates}")
    logger.info(f"跳过: {skipped_dates}")

    if contract_stats:
        logger.info(f"\n主力合约统计:")
        for contract, days in sorted(contract_stats.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {contract}: {days} 天 ({days/processed_dates*100:.1f}%)")

    logger.info(f"\n输出目录: {TARGET_DATA_ROOT / commodity}")


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="提取主力合约数据")
    parser.add_argument(
        "--commodity",
        type=str,
        default="铝",
        help="品种名称（默认：铝，例如：燃料油）"
    )
    parser.add_argument(
        "--year",
        type=str,
        help="年份（可选，例如 2023）"
    )
    parser.add_argument(
        "--month",
        type=str,
        help="月份（可选，例如 01）"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="成交量阈值比例（默认 0.5）"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细日志"
    )

    args = parser.parse_args()

    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.commodity not in MAIN_CONTRACT_MONTHS:
        logger.error(
            f"不支持的品种: {args.commodity}。"
            f"已配置品种: {', '.join(sorted(MAIN_CONTRACT_MONTHS.keys()))}"
        )
        return

    if args.commodity not in COMMODITY_SYMBOL_MAP:
        logger.error(
            f"品种 {args.commodity} 缺少代码映射配置。"
            f"已配置映射: {', '.join(sorted(COMMODITY_SYMBOL_MAP.keys()))}"
        )
        return

    # 处理指定品种数据
    process_commodity(
        commodity=args.commodity,
        year=args.year,
        month=args.month,
        volume_threshold=args.threshold
    )


if __name__ == "__main__":
    main()
