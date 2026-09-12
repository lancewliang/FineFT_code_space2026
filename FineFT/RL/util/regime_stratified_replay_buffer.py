from __future__ import annotations

import hashlib
import logging
from typing import Any
import numpy as np
import torch

from RL.util.replay_buffer_DQN import Multi_step_ReplayBuffer_multi_info
from RL.DiHFT.low_level.parallel_pretrain import extract_stacked_tensor_dict

logger = logging.getLogger(__name__)

DIRECTIONAL_REGIME_PHASES = {
    0: [0, 3, 6],  # Phase 0: 下跌趋势 (Downtrend / Bear)
    1: [1, 4, 7],  # Phase 1: 横盘震荡 (Range / Flat)
    2: [2, 5, 8],  # Phase 2: 上涨趋势 (Uptrend / Bull)
}


def get_active_grid_ids_for_epoch(epoch_index: int, block_epochs: int = 3) -> list[int]:
    """计算给定 epoch 所属的纯方向阶段激活网格列表。"""
    if block_epochs <= 0:
        raise ValueError(f"block_epochs must be positive, got {block_epochs}")
    phase_index = (epoch_index // block_epochs) % 3
    return list(DIRECTIONAL_REGIME_PHASES[phase_index])



def accumulate_trajectory_n_step(
    transitions: list[tuple[Any, dict[str, Any], int, float, Any, dict[str, Any], bool]],
    gamma: float = 0.99,
    n_step: int = 12,
) -> list[tuple[Any, dict[str, Any], int, float, Any, dict[str, Any], bool]]:
    """在单条时序连续的 episode 轨迹上完成 N 步贴现回报闭环计算。

    尾部不足 N 步的 Transition 进行折现截断并标记 done=True。
    保留决策起始步的 info_t 作为经验属性（包括 regime_grid_id）。
    """
    total_steps = len(transitions)
    if total_steps == 0:
        return []

    accumulated: list[tuple[Any, dict[str, Any], int, float, Any, dict[str, Any], bool]] = []
    gammas = np.array([gamma ** i for i in range(n_step)], dtype=float)

    for t in range(total_steps):
        state_t, info_t, action_t, _, _, _, _ = transitions[t]

        discounted_reward = 0.0
        terminal_idx = -1

        lookahead_len = min(n_step, total_steps - t)
        for k in range(lookahead_len):
            step_k = t + k
            discounted_reward += gammas[k] * float(transitions[step_k][3])
            if transitions[step_k][6]:  # done is True
                terminal_idx = step_k
                break

        if terminal_idx != -1:
            next_state = transitions[terminal_idx][4]
            next_info = transitions[terminal_idx][5]
            done = True
        elif t + n_step < total_steps:
            next_state = transitions[t + n_step][0]
            next_info = transitions[t + n_step][1]
            done = False
        else:
            next_state = transitions[-1][4]
            next_info = transitions[-1][5]
            done = True

        accumulated.append(
            (state_t, info_t, action_t, float(discounted_reward), next_state, next_info, done)
        )

    return accumulated


def _build_semantic_transition_key(
    state: np.ndarray,
    action: int,
    info: dict[str, Any],
) -> tuple[int, int, int, int, int]:
    """提取经验的语义决策键：(state_hash, previous_action, action, pos_dir, dur_bucket)。"""
    state_hash = int.from_bytes(
        hashlib.blake2b(state.tobytes(), digest_size=8).digest(), "big"
    )
    previous_action = int(info["previous_action"])
    trading_info = info["trading_info"]
    pos_dir = int(np.round(float(trading_info[0])))
    raw_steps = int(float(trading_info[3]) * 180)
    if raw_steps <= 2:
        dur_bucket = 0
    elif raw_steps <= 12:
        dur_bucket = 1
    elif raw_steps <= 36:
        dur_bucket = 2
    else:
        dur_bucket = 3

    return (state_hash, previous_action, int(action), pos_dir, dur_bucket)


class RegimeStratifiedReplayBuffer:
    """体制分层多步经验回放池（3x3 = 9 格）。

    将总容量解耦为 9 个独立的环形队列，各队列独立维护容量配额、FIFO 淘汰与去重索引。
    底层的各经验池以 n_step=1 初始化，因为时序多步累加已在轨迹层闭环。
    """

    def __init__(
        self,
        total_buffer_size: int,
        batch_size: int,
        device: str,
        seed: int = 42,
        num_grids: int = 9,
    ) -> None:
        self.num_grids = num_grids
        self.grid_capacity = total_buffer_size // num_grids
        self.batch_size = batch_size
        self.device = device

        self.buffers: dict[int, Multi_step_ReplayBuffer_multi_info] = {
            g: Multi_step_ReplayBuffer_multi_info(
                buffer_size=self.grid_capacity,
                batch_size=batch_size,
                device=device,
                seed=seed + g,
                gamma=1.0,
                n_step=1,
            )
            for g in range(num_grids)
        }
        self.seen_fingerprints: dict[int, dict[Any, tuple[int, float]]] = {
            g: {} for g in range(num_grids)
        }

    def route_transition(self, info: dict[str, Any]) -> int:
        """提取 info 中的体制网格 ID (0..8)。"""
        return int(info["regime_grid_id"])

    def add_transition(
        self,
        transition: tuple[Any, dict[str, Any], int, float, Any, dict[str, Any], bool],
        td_error: float = 0.0,
    ) -> int:
        """按 Transition 起始步的体制将经验路由写入对应网格队列。"""
        info = transition[1]
        grid_id = self.route_transition(info)
        if grid_id < 0 or grid_id >= self.num_grids:
            return -1

        buf = self.buffers[grid_id]
        tracker = self.seen_fingerprints[grid_id]

        state = transition[0]
        action = int(transition[2])
        semantic_key = _build_semantic_transition_key(state, action, info)

        if semantic_key in tracker:
            old_idx, old_td_error = tracker[semantic_key]
            if td_error > old_td_error:
                buf.replace(old_idx, *transition)
                tracker[semantic_key] = (old_idx, td_error)
            else:
                return -2
        else:
            curr_len = len(buf)
            target_idx = curr_len if curr_len < self.grid_capacity else (self.grid_capacity - 1)
            buf.add_transition(transition)
            tracker[semantic_key] = (target_idx, td_error)

        return grid_id

    def get_grid_lengths(self) -> dict[int, int]:
        return {g: len(self.buffers[g]) for g in range(self.num_grids)}

    def total_len(self) -> int:
        return sum(self.get_grid_lengths().values())

    def __len__(self) -> int:
        return self.total_len()

    def create_sampler(
        self,
        epoch_index: int,
        block_epochs: int = 3,
        active_grid_ids: list[int] | None = None,
    ) -> StratifiedStackedSampler:
        """创建针对当前轮次激活网格的分层张量采样器。"""
        if active_grid_ids is None:
            active_grid_ids = get_active_grid_ids_for_epoch(epoch_index, block_epochs=block_epochs)
        tensor_dicts = self.extract_stacked_tensor_dicts()
        return StratifiedStackedSampler(
            tensor_dicts=tensor_dicts,
            active_grid_ids=active_grid_ids,
            batch_size=self.batch_size,
            device=self.device,
        )

    def extract_stacked_tensor_dicts(self) -> dict[int, dict[str, Any]]:
        """分池提取连续 Tensor 字典快照供 GPU 分层采样。"""
        return {
            g: extract_stacked_tensor_dict(self.buffers[g])
            for g in range(self.num_grids)
        }


class StratifiedStackedSampler:
    """分层均衡采样器：从当前轮次激活的网格集合中按批次配额平衡采样。"""

    def __init__(
        self,
        tensor_dicts: dict[int, dict[str, Any]],
        active_grid_ids: list[int],
        batch_size: int,
        device: str,
    ) -> None:
        if not active_grid_ids:
            raise ValueError("active_grid_ids cannot be empty")
        self.tensor_dicts = tensor_dicts
        self.active_grid_ids = list(active_grid_ids)
        self.batch_size = batch_size
        self.device = device

        k = len(self.active_grid_ids)
        base = batch_size // k
        remainder = batch_size % k

        self.quotas: dict[int, int] = {}
        for i, g in enumerate(self.active_grid_ids):
            self.quotas[g] = base + (1 if i < remainder else 0)

    def sample(self) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor,
    ]:
        grid_counts = {
            g: (self.tensor_dicts[g]["buffer_size"] if "buffer_size" in self.tensor_dicts[g] else self.tensor_dicts[g]["states"].shape[0])
            for g in self.tensor_dicts
        }

        available_grids = [g for g in self.active_grid_ids if grid_counts[g] > 0]
        if not available_grids:
            available_grids = [
                g for g in self.tensor_dicts
                if (self.tensor_dicts[g]["buffer_size"] if "buffer_size" in self.tensor_dicts[g] else self.tensor_dicts[g]["states"].shape[0]) > 0
            ]
            if not available_grids:
                raise ValueError("Entire replay buffer is empty; cannot sample")
            logger.warning(
                "All active grids %s are empty; falling back to available grids %s",
                self.active_grid_ids,
                available_grids,
            )

        k = len(available_grids)
        base = self.batch_size // k
        remainder = self.batch_size % k

        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        batch_infos: dict[str, list[torch.Tensor]] = {}
        batch_next_infos: dict[str, list[torch.Tensor]] = {}

        for i, g in enumerate(available_grids):
            quota = base + (1 if i < remainder else 0)
            if quota <= 0:
                continue

            grid_payload = self.tensor_dicts[g]
            n_samples = grid_counts[g]
            replace = n_samples < quota

            idx = np.random.choice(n_samples, size=quota, replace=replace)

            batch_states.append(torch.as_tensor(grid_payload["states"][idx]).float())
            batch_actions.append(torch.as_tensor(grid_payload["actions"][idx]).long())
            batch_rewards.append(torch.as_tensor(grid_payload["rewards"][idx]).float())
            batch_next_states.append(torch.as_tensor(grid_payload["next_states"][idx]).float())
            batch_dones.append(torch.as_tensor(grid_payload["dones"][idx]).float())

            for k_info, v_info in grid_payload["infos"].items():
                if k_info not in batch_infos:
                    batch_infos[k_info] = []
                batch_infos[k_info].append(torch.as_tensor(v_info[idx]).float())

            for k_info, v_info in grid_payload["next_infos"].items():
                if k_info not in batch_next_infos:
                    batch_next_infos[k_info] = []
                batch_next_infos[k_info].append(torch.as_tensor(v_info[idx]).float())

        states = torch.cat(batch_states, dim=0).to(self.device)
        actions = torch.cat(batch_actions, dim=0).to(self.device)
        rewards = torch.cat(batch_rewards, dim=0).to(self.device)
        next_states = torch.cat(batch_next_states, dim=0).to(self.device)
        dones = torch.cat(batch_dones, dim=0).to(self.device)

        infos = {
            k_info: torch.cat(v_list, dim=0).to(self.device)
            for k_info, v_list in batch_infos.items()
        }
        next_infos = {
            k_info: torch.cat(v_list, dim=0).to(self.device)
            for k_info, v_list in batch_next_infos.items()
        }

        return states, infos, actions, rewards, next_states, next_infos, dones
