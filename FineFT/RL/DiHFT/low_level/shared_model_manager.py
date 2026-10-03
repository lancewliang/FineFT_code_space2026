from __future__ import annotations

import logging
import resource
from typing import Any, Callable

import torch
import torch.multiprocessing as tmp
from torch import nn

logger = logging.getLogger(__name__)


def configure_sharing_strategy_and_limits() -> None:
    """统一设置共享策略为 file_system，并尝试调高进程 open files 限制。"""
    tmp.set_sharing_strategy("file_system")
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = max(soft, 65536)
        if hard < target and hard != resource.RLIM_INFINITY:
            target = hard
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (OSError, ValueError):
        pass


class SharedInferenceManager:
    """管理驻留在 POSIX 共享内存中的单副本推断模型。"""

    def __init__(
        self,
        model_factory: Callable[..., nn.Module],
        model_kwargs: dict[str, Any],
        initial_state_dict: dict[str, Any] | None = None,
    ) -> None:
        configure_sharing_strategy_and_limits()

        self.shared_model: nn.Module = model_factory(**model_kwargs).to("cpu")
        self.shared_model.eval()

        if initial_state_dict is not None:
            self.shared_model.load_state_dict(initial_state_dict)

        self.shared_model.share_memory()

    def get_shared_model(self) -> nn.Module:
        """获取共享模型引用，可直接传给多进程。"""
        return self.shared_model

    def sync_weights_from_gpu(self, gpu_module: nn.Module) -> None:
        """将 GPU/主进程训练得到的最新参数通过 In-place 拷贝同步到共享物理内存页中。

        耗时 < 5ms，90 个子进程下一次前向传播将立即读取到新权重。
        """
        gpu_state = gpu_module.state_dict()
        with torch.no_grad():
            for name, param in self.shared_model.named_parameters():
                param.copy_(gpu_state[name].to("cpu"))
            for name, buf in self.shared_model.named_buffers():
                buf.copy_(gpu_state[name].to("cpu"))
