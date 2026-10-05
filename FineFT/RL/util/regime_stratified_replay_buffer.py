from __future__ import annotations

import hashlib
import logging
from typing import Any
import numpy as np
import torch

from RL.DiHFT.low_level.parallel_pretrain import extract_stacked_tensor_dict

logger = logging.getLogger(__name__)

DIRECTIONAL_REGIME_PHASES = {
    0: [8],        # Phase 0: 极端上涨趋势
    1: [2, 5, 8],  # Phase 1: 上涨趋势 (Uptrend / Bull)
    2: [1, 4, 7],  # Phase 2: 横盘震荡 (Range / Flat)
    3: [0, 3, 6],  # Phase 3: 下跌趋势 (Downtrend / Bear)        
    4: [0, 4, 8],  # Phase 4: 对角匹配体制 (Diagonal / Matched: s0v0, s1v1, s2v2)     
    5: [0, 1, 2, 3, 4, 5, 6, 7, 8],  # Phase 5: 全量经验抽取 (All Regimes / Full Experience)
}


def get_active_grid_ids_for_epoch(epoch_index: int, block_epochs: int = 3) -> list[int]:
    """计算给定 epoch 所属的阶段激活网格列表。"""
    if block_epochs <= 0:
        raise ValueError(f"block_epochs must be positive, got {block_epochs}")
    phase_index = (epoch_index // block_epochs) % len(DIRECTIONAL_REGIME_PHASES)
    return list(DIRECTIONAL_REGIME_PHASES[phase_index])


def accumulate_trajectory_n_step(
    transitions: list[tuple[Any, dict[str, Any], int, float, Any, dict[str, Any], bool]],
    gamma: float = 0.99,
    n_step: int = 12,
) -> list[tuple[Any, dict[str, Any], int, float, Any, dict[str, Any], bool]]:
    """在单条时序连续的 episode 轨迹上完成 N 步贴现回报闭环计算。"""
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
            if transitions[step_k][6]:
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


def build_semantic_transition_key(
    state: np.ndarray,
    action: int,
    info: dict[str, Any],
) -> tuple[int, int, int, int, int]:
    """提取经验的语义决策键：(state_hash, previous_action, action, pos_dir, dur_bucket)。"""
    state_arr = np.asarray(state)
    state_hash = int.from_bytes(
        hashlib.blake2b(state_arr.tobytes(), digest_size=8).digest(), "big"
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


class _GridSlotsProxy:
    def __init__(self, buffer: RegimeStratifiedReplayBuffer, grid_id: int) -> None:
        self._buffer = buffer
        self._grid_id = grid_id

    def __len__(self) -> int:
        return self._buffer.grid_sizes[self._grid_id]

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        if idx >= len(self) or idx < -len(self):
            raise IndexError("Index out of range")
        if idx < 0:
            idx = len(self) + idx
        g = self._grid_id
        b = self._buffer
        s = b.states[g][idx].cpu().numpy()
        a = int(b.actions[g][idx, 0].item())
        r = float(b.rewards[g][idx, 0].item())
        s_ = b.next_states[g][idx].cpu().numpy()
        d = bool(b.dones[g][idx, 0].item() > 0.5)
        info = {
            "previous_action": int(b.previous_actions[g][idx].item()),
            "avaliable_action": b.avaliable_actions[g][idx].cpu().numpy(),
            "funding_count_down_hour": float(b.hours[g][idx].item()),
            "funding_count_down_minute": float(b.minutes[g][idx].item()),
            "trading_info": b.trading_infos[g][idx].cpu().numpy(),
            "q_value": b.q_values[g][idx].cpu().numpy(),
            "regime_grid_id": g,
        }
        next_info = {
            "previous_action": int(b.next_previous_actions[g][idx].item()),
            "avaliable_action": b.next_avaliable_actions[g][idx].cpu().numpy(),
            "funding_count_down_hour": float(b.next_hours[g][idx].item()),
            "funding_count_down_minute": float(b.next_minutes[g][idx].item()),
            "trading_info": b.next_trading_infos[g][idx].cpu().numpy(),
        }
        if idx in b.extra_infos[g]:
            info.update(b.extra_infos[g][idx])
        return (s, info, a, r, s_, d, next_info)


class RegimeStratifiedReplayBuffer:
    """体制分层多步经验回放池（3x3 = 9 格）。

    将总容量解耦为 9 个独立的定长队列，各队列预分配连续数值张量，
    各队列独立维护容量配额、精确 O(1) 循环 FIFO 淘汰与语义去重索引。
    彻底消除动态列表追加、堆碎片与每轮动态堆叠开销。
    """

    def __init__(
        self,
        total_buffer_size: int,
        batch_size: int,
        device: str | torch.device = "cpu",
        seed: int = 42,
        num_grids: int = 9,
        state_dim: int | None = None,
        action_dim: int | None = None,
    ) -> None:
        self.num_grids = num_grids
        self.grid_capacity = total_buffer_size // num_grids
        self.batch_size = batch_size
        self.device = torch.device(device) if isinstance(device, str) else device
        self.seed = seed

        self.grid_sizes: dict[int, int] = {g: 0 for g in range(num_grids)}
        self.write_ptrs: dict[int, int] = {g: 0 for g in range(num_grids)}
        self.keys_by_idx: dict[int, dict[int, Any]] = {g: {} for g in range(num_grids)}
        self.seen_fingerprints: dict[int, dict[Any, tuple[int, float]]] = {
            g: {} for g in range(num_grids)
        }
        self.total_added_count = 0

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.extra_infos: dict[int, dict[int, dict[str, Any]]] = {g: {} for g in range(self.num_grids)}
        self._initialized = False

        self.states: dict[int, torch.Tensor] = {}
        self.actions: dict[int, torch.Tensor] = {}
        self.rewards: dict[int, torch.Tensor] = {}
        self.next_states: dict[int, torch.Tensor] = {}
        self.dones: dict[int, torch.Tensor] = {}

        self.previous_actions: dict[int, torch.Tensor] = {}
        self.avaliable_actions: dict[int, torch.Tensor] = {}
        self.hours: dict[int, torch.Tensor] = {}
        self.minutes: dict[int, torch.Tensor] = {}
        self.trading_infos: dict[int, torch.Tensor] = {}
        self.q_values: dict[int, torch.Tensor] = {}

        self.next_previous_actions: dict[int, torch.Tensor] = {}
        self.next_avaliable_actions: dict[int, torch.Tensor] = {}
        self.next_hours: dict[int, torch.Tensor] = {}
        self.next_minutes: dict[int, torch.Tensor] = {}
        self.next_trading_infos: dict[int, torch.Tensor] = {}

        if state_dim is not None and action_dim is not None:
            self._init_tensors(state_dim, action_dim)

    def _init_tensors(self, state_dim: int, action_dim: int) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.extra_infos: dict[int, dict[int, dict[str, Any]]] = {g: {} for g in range(self.num_grids)}
        for g in range(self.num_grids):
            self.states[g] = torch.zeros((self.grid_capacity, state_dim), dtype=torch.float32, device=self.device)
            self.actions[g] = torch.zeros((self.grid_capacity, 1), dtype=torch.int64, device=self.device)
            self.rewards[g] = torch.zeros((self.grid_capacity, 1), dtype=torch.float32, device=self.device)
            self.next_states[g] = torch.zeros((self.grid_capacity, state_dim), dtype=torch.float32, device=self.device)
            self.dones[g] = torch.zeros((self.grid_capacity, 1), dtype=torch.float32, device=self.device)

            self.previous_actions[g] = torch.zeros((self.grid_capacity,), dtype=torch.float32, device=self.device)
            self.avaliable_actions[g] = torch.zeros((self.grid_capacity, action_dim), dtype=torch.float32, device=self.device)
            self.hours[g] = torch.zeros((self.grid_capacity,), dtype=torch.float32, device=self.device)
            self.minutes[g] = torch.zeros((self.grid_capacity,), dtype=torch.float32, device=self.device)
            self.trading_infos[g] = torch.zeros((self.grid_capacity, 4), dtype=torch.float32, device=self.device)
            self.q_values[g] = torch.zeros((self.grid_capacity, action_dim), dtype=torch.float32, device=self.device)

            self.next_previous_actions[g] = torch.zeros((self.grid_capacity,), dtype=torch.float32, device=self.device)
            self.next_avaliable_actions[g] = torch.zeros((self.grid_capacity, action_dim), dtype=torch.float32, device=self.device)
            self.next_hours[g] = torch.zeros((self.grid_capacity,), dtype=torch.float32, device=self.device)
            self.next_minutes[g] = torch.zeros((self.grid_capacity,), dtype=torch.float32, device=self.device)
            self.next_trading_infos[g] = torch.zeros((self.grid_capacity, 4), dtype=torch.float32, device=self.device)
        self._initialized = True

    def _write_slot(self, grid_id: int, idx: int, transition: tuple[Any, ...]) -> None:
        s, info, a, r, s_, next_info, done = transition
        device = self.device

        self.states[grid_id][idx] = torch.as_tensor(s, dtype=torch.float32, device=device).reshape(-1)
        self.actions[grid_id][idx, 0] = int(a)
        self.rewards[grid_id][idx, 0] = float(r)
        self.next_states[grid_id][idx] = torch.as_tensor(s_, dtype=torch.float32, device=device).reshape(-1)
        self.dones[grid_id][idx, 0] = 1.0 if done else 0.0

        self.previous_actions[grid_id][idx] = float(info["previous_action"])
        self.avaliable_actions[grid_id][idx] = torch.as_tensor(info["avaliable_action"], dtype=torch.float32, device=device)
        self.hours[grid_id][idx] = float(info["funding_count_down_hour"])
        self.minutes[grid_id][idx] = float(info["funding_count_down_minute"])
        self.trading_infos[grid_id][idx] = torch.as_tensor(info["trading_info"], dtype=torch.float32, device=device)
        self.q_values[grid_id][idx] = torch.as_tensor(info["q_value"], dtype=torch.float32, device=device)

        self.next_previous_actions[grid_id][idx] = float(next_info["previous_action"])
        self.next_avaliable_actions[grid_id][idx] = torch.as_tensor(next_info["avaliable_action"], dtype=torch.float32, device=device)
        self.next_hours[grid_id][idx] = float(next_info["funding_count_down_hour"])
        self.next_minutes[grid_id][idx] = float(next_info["funding_count_down_minute"])
        self.next_trading_infos[grid_id][idx] = torch.as_tensor(next_info["trading_info"], dtype=torch.float32, device=device)

        extra = {
            k: v for k, v in info.items()
            if k not in (
                "previous_action", "avaliable_action", "funding_count_down_hour",
                "funding_count_down_minute", "trading_info", "q_value", "regime_grid_id"
            )
        }
        if extra:
            self.extra_infos[grid_id][idx] = extra
        elif idx in self.extra_infos[grid_id]:
            del self.extra_infos[grid_id][idx]

    @property
    def slots(self) -> dict[int, _GridSlotsProxy]:
        return {g: _GridSlotsProxy(self, g) for g in range(self.num_grids)}

    def route_transition(self, info: dict[str, Any]) -> int:
        return int(info["regime_grid_id"])

    def add_transition(
        self,
        transition: tuple[Any, dict[str, Any], int, float, Any, dict[str, Any], bool],
        td_error: float = 0.0,
    ) -> int:
        state = transition[0]
        info = transition[1]
        action = int(transition[2])
        grid_id = int(info["regime_grid_id"])
        if grid_id < 0 or grid_id >= self.num_grids:
            return -1

        if not self._initialized:
            s_dim = int(np.asarray(state).reshape(-1).shape[0])
            a_dim = int(len(info["avaliable_action"]))
            self._init_tensors(s_dim, a_dim)

        tracker = self.seen_fingerprints[grid_id]
        semantic_key = build_semantic_transition_key(state, action, info)

        if semantic_key in tracker:
            old_idx, old_td_error = tracker[semantic_key]
            if td_error > old_td_error:
                self._write_slot(grid_id, old_idx, transition)
                tracker[semantic_key] = (old_idx, td_error)
                self.total_added_count += 1
                return grid_id
            return -2

        if self.grid_sizes[grid_id] < self.grid_capacity:
            idx = self.grid_sizes[grid_id]
            self.grid_sizes[grid_id] += 1
            self._write_slot(grid_id, idx, transition)
            tracker[semantic_key] = (idx, td_error)
            self.keys_by_idx[grid_id][idx] = semantic_key
        else:
            idx = self.write_ptrs[grid_id]
            old_key = self.keys_by_idx[grid_id][idx]
            del tracker[old_key]
            self._write_slot(grid_id, idx, transition)
            tracker[semantic_key] = (idx, td_error)
            self.keys_by_idx[grid_id][idx] = semantic_key
            self.write_ptrs[grid_id] = (idx + 1) % self.grid_capacity

        self.total_added_count += 1
        return grid_id

    def add_round_records_bulk(self, round_records: list[Any]) -> int:
        """批量聚合写入一轮探索记录，在 CPU 端去重后单次 DMA 拷贝写入目标张量。"""
        if not round_records:
            return 0
        from RL.DiHFT.low_level.parallel_diverse_train import sort_round_records
        records = sort_round_records(round_records)

        duplicate_count = 0
        pending_by_grid: dict[int, dict[int, tuple[Any, ...]]] = {
            g: {} for g in range(self.num_grids)
        }

        for df_index, record in records:
            transition = record.transition
            td_error = float(record.td_error)
            state = transition[0]
            info = transition[1]
            action = int(transition[2])
            grid_id = int(info["regime_grid_id"])
            if grid_id < 0 or grid_id >= self.num_grids:
                duplicate_count += 1
                continue

            if not self._initialized:
                s_dim = int(np.asarray(state).reshape(-1).shape[0])
                a_dim = int(len(info["avaliable_action"]))
                self._init_tensors(s_dim, a_dim)

            tracker = self.seen_fingerprints[grid_id]
            semantic_key = build_semantic_transition_key(state, action, info)

            if semantic_key in tracker:
                old_idx, old_td_error = tracker[semantic_key]
                if td_error > old_td_error:
                    tracker[semantic_key] = (old_idx, td_error)
                    pending_by_grid[grid_id][old_idx] = transition
                    self.total_added_count += 1
                else:
                    duplicate_count += 1
                continue

            if self.grid_sizes[grid_id] < self.grid_capacity:
                idx = self.grid_sizes[grid_id]
                self.grid_sizes[grid_id] += 1
                tracker[semantic_key] = (idx, td_error)
                self.keys_by_idx[grid_id][idx] = semantic_key
                pending_by_grid[grid_id][idx] = transition
            else:
                idx = self.write_ptrs[grid_id]
                old_key = self.keys_by_idx[grid_id][idx]
                del tracker[old_key]
                tracker[semantic_key] = (idx, td_error)
                self.keys_by_idx[grid_id][idx] = semantic_key
                pending_by_grid[grid_id][idx] = transition
                self.write_ptrs[grid_id] = (idx + 1) % self.grid_capacity

            self.total_added_count += 1

        for g, pending in pending_by_grid.items():
            if not pending:
                continue
            indices = list(pending.keys())
            for idx in indices:
                info = pending[idx][1]
                extra = {
                    k: v for k, v in info.items()
                    if k not in (
                        "previous_action", "avaliable_action", "funding_count_down_hour",
                        "funding_count_down_minute", "trading_info", "q_value", "regime_grid_id"
                    )
                }
                if extra:
                    self.extra_infos[g][idx] = extra
                elif idx in self.extra_infos[g]:
                    del self.extra_infos[g][idx]

            if len(indices) == 1:
                self._write_slot(g, indices[0], pending[indices[0]])
                continue

            trans_list = [pending[i] for i in indices]
            device = self.device
            idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)

            states_np = np.ascontiguousarray([np.asarray(t[0]).reshape(-1) for t in trans_list], dtype=np.float32)
            actions_np = np.ascontiguousarray([[t[2]] for t in trans_list], dtype=np.int64)
            rewards_np = np.ascontiguousarray([[t[3]] for t in trans_list], dtype=np.float32)
            next_states_np = np.ascontiguousarray([np.asarray(t[4]).reshape(-1) for t in trans_list], dtype=np.float32)
            dones_np = np.ascontiguousarray([[1.0 if t[6] else 0.0] for t in trans_list], dtype=np.float32)

            self.states[g][idx_tensor] = torch.from_numpy(states_np).to(device, non_blocking=True)
            self.actions[g][idx_tensor] = torch.from_numpy(actions_np).to(device, non_blocking=True)
            self.rewards[g][idx_tensor] = torch.from_numpy(rewards_np).to(device, non_blocking=True)
            self.next_states[g][idx_tensor] = torch.from_numpy(next_states_np).to(device, non_blocking=True)
            self.dones[g][idx_tensor] = torch.from_numpy(dones_np).to(device, non_blocking=True)

            prev_actions_np = np.ascontiguousarray([t[1]["previous_action"] for t in trans_list], dtype=np.float32)
            avail_actions_np = np.ascontiguousarray([t[1]["avaliable_action"] for t in trans_list], dtype=np.float32)
            hours_np = np.ascontiguousarray([t[1]["funding_count_down_hour"] for t in trans_list], dtype=np.float32)
            minutes_np = np.ascontiguousarray([t[1]["funding_count_down_minute"] for t in trans_list], dtype=np.float32)
            trading_infos_np = np.ascontiguousarray([t[1]["trading_info"] for t in trans_list], dtype=np.float32)
            q_values_np = np.ascontiguousarray([t[1]["q_value"] for t in trans_list], dtype=np.float32)

            self.previous_actions[g][idx_tensor] = torch.from_numpy(prev_actions_np).to(device, non_blocking=True)
            self.avaliable_actions[g][idx_tensor] = torch.from_numpy(avail_actions_np).to(device, non_blocking=True)
            self.hours[g][idx_tensor] = torch.from_numpy(hours_np).to(device, non_blocking=True)
            self.minutes[g][idx_tensor] = torch.from_numpy(minutes_np).to(device, non_blocking=True)
            self.trading_infos[g][idx_tensor] = torch.from_numpy(trading_infos_np).to(device, non_blocking=True)
            self.q_values[g][idx_tensor] = torch.from_numpy(q_values_np).to(device, non_blocking=True)

            next_prev_actions_np = np.ascontiguousarray([t[5]["previous_action"] for t in trans_list], dtype=np.float32)
            next_avail_actions_np = np.ascontiguousarray([t[5]["avaliable_action"] for t in trans_list], dtype=np.float32)
            next_hours_np = np.ascontiguousarray([t[5]["funding_count_down_hour"] for t in trans_list], dtype=np.float32)
            next_minutes_np = np.ascontiguousarray([t[5]["funding_count_down_minute"] for t in trans_list], dtype=np.float32)
            next_trading_infos_np = np.ascontiguousarray([t[5]["trading_info"] for t in trans_list], dtype=np.float32)

            self.next_previous_actions[g][idx_tensor] = torch.from_numpy(next_prev_actions_np).to(device, non_blocking=True)
            self.next_avaliable_actions[g][idx_tensor] = torch.from_numpy(next_avail_actions_np).to(device, non_blocking=True)
            self.next_hours[g][idx_tensor] = torch.from_numpy(next_hours_np).to(device, non_blocking=True)
            self.next_minutes[g][idx_tensor] = torch.from_numpy(next_minutes_np).to(device, non_blocking=True)
            self.next_trading_infos[g][idx_tensor] = torch.from_numpy(next_trading_infos_np).to(device, non_blocking=True)

        return duplicate_count

    def get_grid_lengths(self) -> dict[int, int]:
        return {g: self.grid_sizes[g] for g in range(self.num_grids)}

    def total_len(self) -> int:
        return sum(self.grid_sizes[g] for g in range(self.num_grids))

    def __len__(self) -> int:
        return self.total_len()

    def create_sampler(
        self,
        epoch_index: int,
        block_epochs: int = 3,
        active_grid_ids: list[int] | None = None,
    ) -> StratifiedContinuousSampler | None:
        """创建针对当前轮次激活网格的分层连续张量采样器。"""
        if active_grid_ids is None:
            active_grid_ids = get_active_grid_ids_for_epoch(epoch_index, block_epochs=block_epochs)
        available_active = [g for g in active_grid_ids if self.grid_sizes[g] > 0]
        if not available_active:
            return None
        return StratifiedContinuousSampler(
            buffer=self,
            active_grid_ids=available_active,
            batch_size=self.batch_size,
            device=self.device,
        )

    def extract_stacked_tensor_dicts(self) -> dict[int, dict[str, Any]]:
        """分池提取连续 Tensor 字典快照。"""
        result = {}
        for g in range(self.num_grids):
            n = self.grid_sizes[g]
            result[g] = {
                "states": self.states[g][:n].cpu() if self._initialized else torch.empty((0,)),
                "actions": self.actions[g][:n].cpu() if self._initialized else torch.empty((0,), dtype=torch.int64),
                "rewards": self.rewards[g][:n].cpu() if self._initialized else torch.empty((0,)),
                "next_states": self.next_states[g][:n].cpu() if self._initialized else torch.empty((0,)),
                "dones": self.dones[g][:n].cpu() if self._initialized else torch.empty((0,)),
                "infos": {
                    "previous_action": self.previous_actions[g][:n].cpu() if self._initialized else torch.empty((0,)),
                    "avaliable_action": self.avaliable_actions[g][:n].cpu() if self._initialized else torch.empty((0,)),
                    "funding_count_down_hour": self.hours[g][:n].cpu() if self._initialized else torch.empty((0,)),
                    "funding_count_down_minute": self.minutes[g][:n].cpu() if self._initialized else torch.empty((0,)),
                    "trading_info": self.trading_infos[g][:n].cpu() if self._initialized else torch.empty((0,)),
                    "q_value": self.q_values[g][:n].cpu() if self._initialized else torch.empty((0,)),
                },
                "next_infos": {
                    "previous_action": self.next_previous_actions[g][:n].cpu() if self._initialized else torch.empty((0,)),
                    "avaliable_action": self.next_avaliable_actions[g][:n].cpu() if self._initialized else torch.empty((0,)),
                    "funding_count_down_hour": self.next_hours[g][:n].cpu() if self._initialized else torch.empty((0,)),
                    "funding_count_down_minute": self.next_minutes[g][:n].cpu() if self._initialized else torch.empty((0,)),
                    "trading_info": self.next_trading_infos[g][:n].cpu() if self._initialized else torch.empty((0,)),
                },
                "buffer_size": n,
            }
        return result


class GPURegimeStratifiedReplayBuffer(RegimeStratifiedReplayBuffer):
    """全 GPU 显存直存的体制分层经验回放池。

    预分配张量直接常驻 GPU 显存，训练迭代采样全程在 CUDA 核心内执行，
    PCIe 数据传输彻底清零。在探索完成时执行单次批量 DMA 上传。
    """

    def __init__(
        self,
        total_buffer_size: int,
        batch_size: int,
        device: str | torch.device = "cuda",
        seed: int = 42,
        num_grids: int = 9,
        state_dim: int | None = None,
        action_dim: int | None = None,
    ) -> None:
        target_device = device
        if isinstance(target_device, str) and "cuda" in target_device and not torch.cuda.is_available():
            target_device = "cpu"
        super().__init__(
            total_buffer_size=total_buffer_size,
            batch_size=batch_size,
            device=target_device,
            seed=seed,
            num_grids=num_grids,
            state_dim=state_dim,
            action_dim=action_dim,
        )


class StratifiedContinuousSampler:
    """基于预分配连续张量的高性能分层采样器，彻底消除动态堆叠与反序列化。"""

    def __init__(
        self,
        buffer: RegimeStratifiedReplayBuffer | None = None,
        active_grid_ids: list[int] | None = None,
        batch_size: int = 64,
        device: str | torch.device = "cpu",
        tensor_dicts: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        if not active_grid_ids:
            raise ValueError("active_grid_ids cannot be empty")
        self.buffer = buffer
        self.tensor_dicts = tensor_dicts
        self.active_grid_ids = list(active_grid_ids)
        self.batch_size = batch_size
        self.device = torch.device(device) if isinstance(device, str) else device

        k = len(self.active_grid_ids)
        base = batch_size // k
        remainder = batch_size % k

        self.quotas: dict[int, int] = {
            g: base + (1 if i < remainder else 0)
            for i, g in enumerate(self.active_grid_ids)
        }
        for g in self.active_grid_ids:
            if buffer is not None:
                n_samples = buffer.grid_sizes[g]
            else:
                n_samples = tensor_dicts[g]["buffer_size"]
            quota = self.quotas[g]
            if n_samples < quota:
                logger.warning(
                    "Active grid %d has %d samples < quota %d; sampling with replacement",
                    g,
                    n_samples,
                    quota,
                )

    def sample(self) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor,
    ]:
        dev = self.device
        b = self.buffer

        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []

        batch_prev_actions = []
        batch_avail_actions = []
        batch_hours = []
        batch_minutes = []
        batch_trading_infos = []
        batch_q_values = []

        batch_next_prev_actions = []
        batch_next_avail_actions = []
        batch_next_hours = []
        batch_next_minutes = []
        batch_next_trading_infos = []

        if b is not None:
            buf_dev = b.device
            for g in self.active_grid_ids:
                quota = self.quotas[g]
                if quota <= 0:
                    continue

                n_samples = b.grid_sizes[g]
                if n_samples == 0:
                    raise ValueError(
                        f"Active grid {g} has 0 transitions; cannot sample quota of {quota}"
                    )

                replace = n_samples < quota
                if replace:
                    idx = torch.randint(0, n_samples, (quota,), device=buf_dev)
                else:
                    idx = torch.randperm(n_samples, device=buf_dev)[:quota]

                batch_states.append(b.states[g][:n_samples][idx])
                batch_actions.append(b.actions[g][:n_samples][idx])
                batch_rewards.append(b.rewards[g][:n_samples][idx])
                batch_next_states.append(b.next_states[g][:n_samples][idx])
                batch_dones.append(b.dones[g][:n_samples][idx])

                batch_prev_actions.append(b.previous_actions[g][:n_samples][idx])
                batch_avail_actions.append(b.avaliable_actions[g][:n_samples][idx])
                batch_hours.append(b.hours[g][:n_samples][idx])
                batch_minutes.append(b.minutes[g][:n_samples][idx])
                batch_trading_infos.append(b.trading_infos[g][:n_samples][idx])
                batch_q_values.append(b.q_values[g][:n_samples][idx])

                batch_next_prev_actions.append(b.next_previous_actions[g][:n_samples][idx])
                batch_next_avail_actions.append(b.next_avaliable_actions[g][:n_samples][idx])
                batch_next_hours.append(b.next_hours[g][:n_samples][idx])
                batch_next_minutes.append(b.next_minutes[g][:n_samples][idx])
                batch_next_trading_infos.append(b.next_trading_infos[g][:n_samples][idx])
        else:
            td = self.tensor_dicts
            for g in self.active_grid_ids:
                quota = self.quotas[g]
                if quota <= 0:
                    continue

                grid_payload = td[g]
                n_samples = grid_payload["buffer_size"]
                if n_samples == 0:
                    raise ValueError(
                        f"Active grid {g} has 0 transitions; cannot sample quota of {quota}"
                    )

                replace = n_samples < quota
                if replace:
                    idx_th = torch.randint(0, n_samples, (quota,))
                else:
                    idx_th = torch.randperm(n_samples)[:quota]

                batch_states.append(grid_payload["states"][idx_th])
                batch_actions.append(grid_payload["actions"][idx_th])
                batch_rewards.append(grid_payload["rewards"][idx_th])
                batch_next_states.append(grid_payload["next_states"][idx_th])
                batch_dones.append(grid_payload["dones"][idx_th])

                batch_prev_actions.append(grid_payload["infos"]["previous_action"][idx_th])
                batch_avail_actions.append(grid_payload["infos"]["avaliable_action"][idx_th])
                batch_hours.append(grid_payload["infos"]["funding_count_down_hour"][idx_th])
                batch_minutes.append(grid_payload["infos"]["funding_count_down_minute"][idx_th])
                batch_trading_infos.append(grid_payload["infos"]["trading_info"][idx_th])
                batch_q_values.append(grid_payload["infos"]["q_value"][idx_th])

                batch_next_prev_actions.append(grid_payload["next_infos"]["previous_action"][idx_th])
                batch_next_avail_actions.append(grid_payload["next_infos"]["avaliable_action"][idx_th])
                batch_next_hours.append(grid_payload["next_infos"]["funding_count_down_hour"][idx_th])
                batch_next_minutes.append(grid_payload["next_infos"]["funding_count_down_minute"][idx_th])
                batch_next_trading_infos.append(grid_payload["next_infos"]["trading_info"][idx_th])

        states = torch.cat(batch_states, dim=0).to(dev)
        actions = torch.cat(batch_actions, dim=0).to(dev)
        rewards = torch.cat(batch_rewards, dim=0).to(dev)
        next_states = torch.cat(batch_next_states, dim=0).to(dev)
        dones = torch.cat(batch_dones, dim=0).to(dev)

        infos = {
            "previous_action": torch.cat(batch_prev_actions, dim=0).to(dev),
            "avaliable_action": torch.cat(batch_avail_actions, dim=0).to(dev),
            "funding_count_down_hour": torch.cat(batch_hours, dim=0).to(dev),
            "funding_count_down_minute": torch.cat(batch_minutes, dim=0).to(dev),
            "trading_info": torch.cat(batch_trading_infos, dim=0).to(dev),
            "q_value": torch.cat(batch_q_values, dim=0).to(dev),
        }
        next_infos = {
            "previous_action": torch.cat(batch_next_prev_actions, dim=0).to(dev),
            "avaliable_action": torch.cat(batch_next_avail_actions, dim=0).to(dev),
            "funding_count_down_hour": torch.cat(batch_next_hours, dim=0).to(dev),
            "funding_count_down_minute": torch.cat(batch_next_minutes, dim=0).to(dev),
            "trading_info": torch.cat(batch_next_trading_infos, dim=0).to(dev),
        }

        return states, infos, actions, rewards, next_states, next_infos, dones


StratifiedStackedSampler = StratifiedContinuousSampler
