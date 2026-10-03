from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnvTensorPack:
    """单个市场数据切片提纯后的只读纯数值共享张量包。

    将 Pandas DataFrame 的高频行情列与特征列在主进程中提前转换为
    连续的 PyTorch CPU 张量并放入 POSIX 共享内存，子进程直接通过
    .numpy() 获取零拷贝视图，避免在每个子进程中重复反序列化与切片。
    """

    state_tensor: torch.Tensor
    ask_prices_tensor: torch.Tensor
    bid_prices_tensor: torch.Tensor
    ask_qtys_tensor: torch.Tensor
    bid_qtys_tensor: torch.Tensor
    markprice_tensor: torch.Tensor
    timestamp_tensor: torch.Tensor
    funding_rate_tensor: torch.Tensor
    funding_timestamp_tensor: torch.Tensor
    timestamp_dtype: str = "datetime64[ns]"
    funding_timestamp_dtype: str = "datetime64[ns]"
    is_limit_up_tensor: torch.Tensor | None = None
    is_limit_down_tensor: torch.Tensor | None = None
    limit_up_ask_depth_ratio_5_tensor: torch.Tensor | None = None
    limit_down_bid_depth_ratio_5_tensor: torch.Tensor | None = None
    upper_limit_prices_tensor: torch.Tensor | None = None
    lower_limit_prices_tensor: torch.Tensor | None = None
    regime_grid_ids_tensor: torch.Tensor | None = None

    @property
    def state_array(self) -> np.ndarray:
        return self.state_tensor.numpy()

    @property
    def ask_prices_array(self) -> np.ndarray:
        return self.ask_prices_tensor.numpy()

    @property
    def bid_prices_array(self) -> np.ndarray:
        return self.bid_prices_tensor.numpy()

    @property
    def ask_qtys_array(self) -> np.ndarray:
        return self.ask_qtys_tensor.numpy()

    @property
    def bid_qtys_array(self) -> np.ndarray:
        return self.bid_qtys_tensor.numpy()

    @property
    def markprice_array(self) -> np.ndarray:
        return self.markprice_tensor.numpy()

    @property
    def timestamp_array(self) -> np.ndarray:
        if "datetime64" in self.timestamp_dtype:
            return self.timestamp_tensor.numpy().view(self.timestamp_dtype)
        return self.timestamp_tensor.numpy()

    @property
    def funding_rate_array(self) -> np.ndarray:
        return self.funding_rate_tensor.numpy()

    @property
    def funding_timestamp_array(self) -> np.ndarray:
        if "datetime64" in self.funding_timestamp_dtype:
            return self.funding_timestamp_tensor.numpy().view(self.funding_timestamp_dtype)
        return self.funding_timestamp_tensor.numpy()

    @property
    def is_limit_up_array(self) -> np.ndarray | None:
        if self.is_limit_up_tensor is None:
            return None
        return self.is_limit_up_tensor.numpy()

    @property
    def is_limit_down_array(self) -> np.ndarray | None:
        if self.is_limit_down_tensor is None:
            return None
        return self.is_limit_down_tensor.numpy()

    @property
    def limit_up_ask_depth_ratio_5_array(self) -> np.ndarray | None:
        if self.limit_up_ask_depth_ratio_5_tensor is None:
            return None
        return self.limit_up_ask_depth_ratio_5_tensor.numpy()

    @property
    def limit_down_bid_depth_ratio_5_array(self) -> np.ndarray | None:
        if self.limit_down_bid_depth_ratio_5_tensor is None:
            return None
        return self.limit_down_bid_depth_ratio_5_tensor.numpy()

    @property
    def upper_limit_prices_array(self) -> np.ndarray | None:
        if self.upper_limit_prices_tensor is None:
            return None
        return self.upper_limit_prices_tensor.numpy()

    @property
    def lower_limit_prices_array(self) -> np.ndarray | None:
        if self.lower_limit_prices_tensor is None:
            return None
        return self.lower_limit_prices_tensor.numpy()

    @property
    def regime_grid_ids_array(self) -> np.ndarray | None:
        if self.regime_grid_ids_tensor is None:
            return None
        return self.regime_grid_ids_tensor.numpy()

    def to_env_kwargs(self) -> dict[str, Any]:
        """导出交易环境初始化所需的全量零拷贝 NumPy 数组字典。"""
        return {
            "state_array": self.state_array,
            "ask_prices_array": self.ask_prices_array,
            "bid_prices_array": self.bid_prices_array,
            "ask_qtys_array": self.ask_qtys_array,
            "bid_qtys_array": self.bid_qtys_array,
            "markprice_array": self.markprice_array,
            "timestamp_array": self.timestamp_array,
            "funding_rate_array": self.funding_rate_array,
            "funding_timestamp_array": self.funding_timestamp_array,
            "is_limit_up_array": self.is_limit_up_array,
            "is_limit_down_array": self.is_limit_down_array,
            "limit_up_ask_depth_ratio_5_array": self.limit_up_ask_depth_ratio_5_array,
            "limit_down_bid_depth_ratio_5_array": self.limit_down_bid_depth_ratio_5_array,
            "upper_limit_prices_array": self.upper_limit_prices_array,
            "lower_limit_prices_array": self.lower_limit_prices_array,
            "regime_grid_ids_array": self.regime_grid_ids_array,
        }


class SharedMarketDataPack:
    """全局共享市场数据注册表。

    在主进程将全部 DataFrame 切片预转换为只读纯数值共享张量包后统一托管，
    所有子进程通过全局注册表引用以零拷贝方式访问任意 df_index。
    """

    def __init__(self, packs: dict[int, EnvTensorPack]) -> None:
        self.packs = packs

    def __getitem__(self, df_index: int) -> EnvTensorPack:
        return self.packs[df_index]

    def __len__(self) -> int:
        return len(self.packs)

    def __contains__(self, df_index: int) -> bool:
        return df_index in self.packs

    def keys(self):
        return self.packs.keys()

    def values(self):
        return self.packs.values()

    def items(self):
        return self.packs.items()

    @classmethod
    def from_dataframes(
        cls,
        train_df_cache: dict[int, pd.DataFrame],
        env_kwargs: dict[str, Any],
    ) -> SharedMarketDataPack:
        """从 DataFrame 字典统一构建并提纯共享行情张量注册表。"""
        if not train_df_cache:
            return cls({})

        try:
            mp.set_sharing_strategy("file_system")
        except RuntimeError:
            pass

        feature_list = env_kwargs["feature_list"]
        order_book_depth = env_kwargs.get("order_book_depth", 25)
        enable_limit_reward = env_kwargs.get("enable_limit_reward", False)

        bid_prices_names = [f"bid{i}_price" for i in range(1, order_book_depth + 1)]
        ask_prices_names = [f"ask{i}_price" for i in range(1, order_book_depth + 1)]
        bid_sizes_names = [f"bid{i}_size" for i in range(1, order_book_depth + 1)]
        ask_sizes_names = [f"ask{i}_size" for i in range(1, order_book_depth + 1)]

        packs: dict[int, EnvTensorPack] = {}

        for df_index, df in train_df_cache.items():
            if enable_limit_reward:
                missing_limit_cols = [
                    col
                    for col in [
                        "limit_up_single_sided_ratio",
                        "limit_down_single_sided_ratio",
                        "limit_up_ask_depth_ratio_5",
                        "limit_down_bid_depth_ratio_5",
                        "UpperLimitPrice",
                        "LowerLimitPrice",
                    ]
                    if col not in df.columns
                ]
                if missing_limit_cols:
                    raise ValueError(
                        f"enable_limit_reward=True 但 df_index={df_index} 缺少必须的涨跌停列: {missing_limit_cols}"
                    )

            state_tensor = torch.from_numpy(
                np.ascontiguousarray(df[feature_list].values, dtype=np.float32)
            ).share_memory_()
            ask_prices_tensor = torch.from_numpy(
                np.ascontiguousarray(df[ask_prices_names].values, dtype=np.float32)
            ).share_memory_()
            bid_prices_tensor = torch.from_numpy(
                np.ascontiguousarray(df[bid_prices_names].values, dtype=np.float32)
            ).share_memory_()
            ask_qtys_tensor = torch.from_numpy(
                np.ascontiguousarray(df[ask_sizes_names].values, dtype=np.float32)
            ).share_memory_()
            bid_qtys_tensor = torch.from_numpy(
                np.ascontiguousarray(df[bid_sizes_names].values, dtype=np.float32)
            ).share_memory_()
            markprice_tensor = torch.from_numpy(
                np.ascontiguousarray(df["mark_price"].values, dtype=np.float32)
            ).share_memory_()
            funding_rate_tensor = torch.from_numpy(
                np.ascontiguousarray(df["funding_rate"].values, dtype=np.float32)
            ).share_memory_()

            ts_vals = df["timestamp"].values
            ts_dtype_str = str(ts_vals.dtype)
            if np.issubdtype(ts_vals.dtype, np.datetime64):
                ts_raw = ts_vals.view(np.int64)
            else:
                ts_raw = ts_vals.astype(np.int64)
            timestamp_tensor = torch.from_numpy(
                np.ascontiguousarray(ts_raw)
            ).share_memory_()

            fts_vals = df["funding_timestamp"].values
            fts_dtype_str = str(fts_vals.dtype)
            if np.issubdtype(fts_vals.dtype, np.datetime64):
                fts_raw = fts_vals.view(np.int64)
            else:
                fts_raw = fts_vals.astype(np.int64)
            funding_timestamp_tensor = torch.from_numpy(
                np.ascontiguousarray(fts_raw)
            ).share_memory_()

            is_limit_up_tensor = None
            if "limit_up_single_sided_ratio" in df.columns:
                is_limit_up_tensor = torch.from_numpy(
                    np.ascontiguousarray(
                        df["limit_up_single_sided_ratio"].values > 0, dtype=bool
                    )
                ).share_memory_()

            is_limit_down_tensor = None
            if "limit_down_single_sided_ratio" in df.columns:
                is_limit_down_tensor = torch.from_numpy(
                    np.ascontiguousarray(
                        df["limit_down_single_sided_ratio"].values > 0, dtype=bool
                    )
                ).share_memory_()

            limit_up_ask_depth_ratio_5_tensor = None
            if "limit_up_ask_depth_ratio_5" in df.columns:
                limit_up_ask_depth_ratio_5_tensor = torch.from_numpy(
                    np.ascontiguousarray(
                        df["limit_up_ask_depth_ratio_5"].values, dtype=np.float32
                    )
                ).share_memory_()

            limit_down_bid_depth_ratio_5_tensor = None
            if "limit_down_bid_depth_ratio_5" in df.columns:
                limit_down_bid_depth_ratio_5_tensor = torch.from_numpy(
                    np.ascontiguousarray(
                        df["limit_down_bid_depth_ratio_5"].values, dtype=np.float32
                    )
                ).share_memory_()

            upper_limit_prices_tensor = None
            if "UpperLimitPrice" in df.columns:
                upper_limit_prices_tensor = torch.from_numpy(
                    np.ascontiguousarray(
                        df["UpperLimitPrice"].values, dtype=np.float32
                    )
                ).share_memory_()

            lower_limit_prices_tensor = None
            if "LowerLimitPrice" in df.columns:
                lower_limit_prices_tensor = torch.from_numpy(
                    np.ascontiguousarray(
                        df["LowerLimitPrice"].values, dtype=np.float32
                    )
                ).share_memory_()

            regime_grid_ids_tensor = None
            if "regime_grid_id" in df.columns:
                regime_grid_ids_tensor = torch.from_numpy(
                    np.ascontiguousarray(
                        df["regime_grid_id"].values, dtype=np.int64
                    )
                ).share_memory_()

            pack = EnvTensorPack(
                state_tensor=state_tensor,
                ask_prices_tensor=ask_prices_tensor,
                bid_prices_tensor=bid_prices_tensor,
                ask_qtys_tensor=ask_qtys_tensor,
                bid_qtys_tensor=bid_qtys_tensor,
                markprice_tensor=markprice_tensor,
                timestamp_tensor=timestamp_tensor,
                funding_rate_tensor=funding_rate_tensor,
                funding_timestamp_tensor=funding_timestamp_tensor,
                timestamp_dtype=ts_dtype_str,
                funding_timestamp_dtype=fts_dtype_str,
                is_limit_up_tensor=is_limit_up_tensor,
                is_limit_down_tensor=is_limit_down_tensor,
                limit_up_ask_depth_ratio_5_tensor=limit_up_ask_depth_ratio_5_tensor,
                limit_down_bid_depth_ratio_5_tensor=limit_down_bid_depth_ratio_5_tensor,
                upper_limit_prices_tensor=upper_limit_prices_tensor,
                lower_limit_prices_tensor=lower_limit_prices_tensor,
                regime_grid_ids_tensor=regime_grid_ids_tensor,
            )
            packs[df_index] = pack

        logger.info(
            "成功构建只读纯数值共享市场数据注册表 | df_count=%d", len(packs)
        )
        return cls(packs)


def create_demo_env_from_pack(
    pack: EnvTensorPack,
    env_kwargs: dict[str, Any],
    initial_state: tuple[float, float, float, float, float] | None = None,
):
    """直接基于零拷贝共享行情张量包实例化 Demo_Env，彻底规避 DataFrame 反序列化与列切片开销。"""
    from env.env_class.demo_env import Demo_Env

    arrays = pack.to_env_kwargs()
    effective_initial_state = (
        initial_state
        if initial_state is not None
        else (1e5, 0.0, 0.0, 0.0, 1.0)
    )
    leverage_choices = env_kwargs.get(
        "leverage_choices", env_kwargs.get("leverage_choice", [5])
    )
    has_limit_cols = pack.upper_limit_prices_tensor is not None
    enable_limit = env_kwargs.get("enable_limit_reward", False) and has_limit_cols

    return Demo_Env(
        state_array=arrays["state_array"],
        ask_prices_array=arrays["ask_prices_array"],
        bid_prices_array=arrays["bid_prices_array"],
        ask_qtys_array=arrays["ask_qtys_array"],
        bid_qtys_array=arrays["bid_qtys_array"],
        markprice_array=arrays["markprice_array"],
        timestamp_array=arrays["timestamp_array"],
        funding_rate_array=arrays["funding_rate_array"],
        funding_timestamp_array=arrays["funding_timestamp_array"],
        max_holding_number=env_kwargs["max_holding_number"],
        position_choices=env_kwargs["position_choices"],
        leverage_choice=leverage_choices,
        long_estimated_rate=env_kwargs["long_estimated_rate"],
        short_estimated_rate=env_kwargs["short_estimated_rate"],
        commission_rate=env_kwargs["commission_rate"],
        maintenance_margin_ratio_dict=env_kwargs["maintenance_margin_ratio_dict"],
        early_stop=env_kwargs["early_stop"],
        initial_state=effective_initial_state,
        gamma=env_kwargs["gamma"],
        max_punishment=1e10,
        allow_reverse_position=env_kwargs.get("allow_reverse_position", False),
        holding_duration_norm_steps=env_kwargs.get("holding_duration_norm_steps", 180),
        is_limit_up_array=arrays["is_limit_up_array"],
        is_limit_down_array=arrays["is_limit_down_array"],
        limit_up_ask_depth_ratio_5_array=arrays["limit_up_ask_depth_ratio_5_array"],
        limit_down_bid_depth_ratio_5_array=arrays["limit_down_bid_depth_ratio_5_array"],
        upper_limit_prices_array=arrays["upper_limit_prices_array"],
        lower_limit_prices_array=arrays["lower_limit_prices_array"],
        enable_limit_reward=enable_limit,
        limit_hold_bonus=env_kwargs.get("limit_hold_bonus", 1.0),
        limit_stay_bonus=env_kwargs.get("limit_stay_bonus", 0.5),
        limit_reverse_penalty=env_kwargs.get("limit_reverse_penalty", 1.5),
        near_limit_threshold=env_kwargs.get("near_limit_threshold", 0.003),
        regime_grid_ids_array=arrays["regime_grid_ids_array"],
    )
