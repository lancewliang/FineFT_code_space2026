from __future__ import annotations

import atexit
import logging
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    import pandas as pd
    from RL.DiHFT.low_level.parallel_weight_advantage_pretrain import Weighted_Contexts_DQN
    from RL.DiHFT.low_level.shared_model_manager import SharedInferenceManager

logger = logging.getLogger(__name__)


class PersistentRolloutPool:
    """跨 Epoch 常驻探索工作进程池上下文管理器。

    在多样化训练开始时统一创建探索工作进程，跨 Epoch 长期驻留，
    复用内部交易环境与模型引用；在 Epoch 切换时通过原位内存拷贝
    热同步权重；当经验池饱和时提前主动回收。
    """

    def __init__(
        self,
        trainer: Weighted_Contexts_DQN,
        train_df_cache: dict[int, pd.DataFrame],
        env_kwargs: dict[str, Any],
        shared_manager: SharedInferenceManager,
    ) -> None:
        from RL.DiHFT.low_level.parallel_diverse_train import (
            start_parallel_workers,
            shutdown_exploration_workers,
        )

        self.trainer = trainer
        self.shared_manager = shared_manager
        self.is_shutdown = False
        self._shutdown_fn = shutdown_exploration_workers

        logger.info(
            "正在初始化常驻探索工作进程池 | num_workers=%d",
            self.trainer.diverse_num_workers,
        )
        start_parallel_workers(
            trainer=self.trainer,
            train_df_cache=train_df_cache,
            env_kwargs=env_kwargs,
            shared_model=self.shared_manager.get_shared_model(),
        )
        atexit.register(self.shutdown)

    def __enter__(self) -> PersistentRolloutPool:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.shutdown()

    def sync_model_weights(self, gpu_module: nn.Module) -> None:
        """在 Epoch 切换时在主进程原位同步共享权重。"""
        self.shared_manager.sync_weights_from_gpu(gpu_module)

    def shutdown(self, timeout: float = 10.0) -> None:
        """关闭常驻探索进程池并释放资源。"""
        if self.is_shutdown:
            return
        self.is_shutdown = True
        try:
            atexit.unregister(self.shutdown)
        except Exception:
            pass
        logger.info("常驻探索工作进程池开始关闭...")
        self._shutdown_fn(self.trainer)
