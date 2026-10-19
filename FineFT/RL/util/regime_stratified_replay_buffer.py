from __future__ import annotations

import hashlib
import logging
from typing import Any
import numpy as np
import torch

from dataclasses import dataclass, field
from RL.DiHFT.low_level.parallel_pretrain import extract_stacked_tensor_dict

logger = logging.getLogger(__name__)

DIRECTIONAL_REGIME_PHASES = {
    0: [8],        # Phase 0: 极端上涨趋势
    1: [2, 5, 8],  # Phase 1: 上涨趋势 (Uptrend / Bull)
    2: [1, 4, 7],  # Phase 2: 横盘震荡 (Range / Flat)
    3: [0, 3, 6],  # Phase 3: 下跌趋势 (Downtrend / Bear)        
    4: [6],
    5: [6, 7, 8],  # Phase 4: 对角匹配体制 (Diagonal / Matched: s0v0, s1v1, s2v2)     
    6: [0, 1, 2, 3, 4, 5, 6, 7, 8],  # Phase 5: 全量经验抽取 (All Regimes / Full Experience)
    
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


@dataclass
class CompactTrajectoryRecord:
    """低层探索轨迹的紧凑时空切片纯数值流。

    彻底剔除 148 维静态市场状态向量及冗余嵌套 info 字典的 IPC 传输，
    仅通过 Queue 回传由动作、收益、TD-Error、完成标志及时间步切片索引组成的轻量数值连续数组。
    """
    df_index: int
    step_indices: np.ndarray          # (T,) int32
    actions: np.ndarray               # (T,) int64
    rewards: np.ndarray               # (T,) float32 (discounted N-step rewards)
    next_step_indices: np.ndarray     # (T,) int32
    dones: np.ndarray                 # (T,) bool
    td_errors: np.ndarray             # (T,) float32
    previous_actions: np.ndarray      # (T,) float32
    next_previous_actions: np.ndarray # (T,) float32
    trading_infos: np.ndarray         # (T, 4) float32
    next_trading_infos: np.ndarray    # (T, 4) float32
    avaliable_actions: np.ndarray     # (T, action_dim) float32
    next_avaliable_actions: np.ndarray# (T, action_dim) float32
    hours: np.ndarray                 # (T,) float32
    minutes: np.ndarray               # (T,) float32
    next_hours: np.ndarray            # (T,) float32
    next_minutes: np.ndarray          # (T,) float32
    q_values: np.ndarray | None = None          # (T, action_dim) float32
    regime_grid_ids: np.ndarray | None = None   # (T,) int32
    states: np.ndarray | None = None            # (T, state_dim) float32, optional fallback
    next_states: np.ndarray | None = None       # (T, state_dim) float32, optional fallback

    def to_worker_transitions(
        self,
        shared_market_data: Any | None = None,
    ) -> list[Any]:
        """在需要兼容旧版接口时，按需懒加载反序列化生成 WorkerTransitionRecord 列表。"""
        from RL.DiHFT.low_level.parallel_diverse_train import WorkerTransitionRecord
        total_steps = len(self.step_indices)
        records = []
        pack = (
            shared_market_data[self.df_index]
            if shared_market_data is not None and self.df_index in shared_market_data
            else None
        )
        for i in range(total_steps):
            step_idx = int(self.step_indices[i])
            next_step_idx = int(self.next_step_indices[i])
            if self.states is not None and i < len(self.states):
                s = self.states[i]
                s_next = self.next_states[i] if self.next_states is not None and i < len(self.next_states) else s
            elif pack is not None:
                s = pack.state_array[step_idx]
                s_next = pack.state_array[next_step_idx]
            else:
                s = np.zeros(2, dtype=np.float32)
                s_next = np.zeros(2, dtype=np.float32)

            if pack is not None and pack.regime_grid_ids_tensor is not None:
                regime_grid_id = int(pack.regime_grid_ids_array[step_idx])
            elif self.regime_grid_ids is not None and i < len(self.regime_grid_ids):
                regime_grid_id = int(self.regime_grid_ids[i])
            else:
                regime_grid_id = 0

            prev_act = int(self.previous_actions[i])
            if self.q_values is not None and i < len(self.q_values):
                q_val = self.q_values[i]
            elif pack is not None and pack.q_table_tensor is not None:
                q_val = pack.q_table_array[step_idx][prev_act]
            else:
                act_dim = int(self.avaliable_actions.shape[1]) if self.avaliable_actions.ndim > 1 else 9
                q_val = np.zeros(act_dim, dtype=np.float32)

            info = {
                "previous_action": prev_act,
                "avaliable_action": self.avaliable_actions[i],
                "funding_count_down_hour": float(self.hours[i]),
                "funding_count_down_minute": float(self.minutes[i]),
                "trading_info": self.trading_infos[i],
                "q_value": q_val,
                "regime_grid_id": regime_grid_id,
            }
            next_prev_act = int(self.next_previous_actions[i])
            next_info = {
                "previous_action": next_prev_act,
                "avaliable_action": self.next_avaliable_actions[i],
                "funding_count_down_hour": float(self.next_hours[i]),
                "funding_count_down_minute": float(self.next_minutes[i]),
                "trading_info": self.next_trading_infos[i],
                "regime_grid_id": regime_grid_id,
            }
            transition = (
                s,
                info,
                int(self.actions[i]),
                float(self.rewards[i]),
                s_next,
                next_info,
                bool(self.dones[i]),
            )
            records.append(
                WorkerTransitionRecord(
                    step_index=step_idx,
                    transition=transition,
                    td_error=float(self.td_errors[i]),
                )
            )
        return records


def accumulate_compact_trajectory_n_step(
    raw_step_indices: list[int] | np.ndarray,
    raw_actions: list[int] | np.ndarray,
    raw_rewards: list[float] | np.ndarray,
    raw_dones: list[bool] | np.ndarray,
    raw_td_errors: list[float] | np.ndarray,
    raw_previous_actions: list[float] | np.ndarray,
    raw_trading_infos: list[np.ndarray] | np.ndarray,
    raw_avaliable_actions: list[np.ndarray] | np.ndarray,
    raw_hours: list[float] | np.ndarray,
    raw_minutes: list[float] | np.ndarray,
    raw_next_previous_actions: list[float] | np.ndarray,
    raw_next_trading_infos: list[np.ndarray] | np.ndarray,
    raw_next_avaliable_actions: list[np.ndarray] | np.ndarray,
    raw_next_hours: list[float] | np.ndarray,
    raw_next_minutes: list[float] | np.ndarray,
    raw_q_values: list[np.ndarray] | np.ndarray | None = None,
    raw_regime_grid_ids: list[int] | np.ndarray | None = None,
    raw_states: list[np.ndarray] | np.ndarray | None = None,
    raw_next_states: list[np.ndarray] | np.ndarray | None = None,
    gamma: float = 0.99,
    n_step: int = 12,
    df_index: int = 0,
) -> CompactTrajectoryRecord:
    """在紧凑纯数值流上直接完成 N 步贴现回报与截断目标步索引计算。"""
    total_steps = len(raw_actions)
    if total_steps == 0:
        return CompactTrajectoryRecord(
            df_index=df_index,
            step_indices=np.empty(0, dtype=np.int32),
            actions=np.empty(0, dtype=np.int64),
            rewards=np.empty(0, dtype=np.float32),
            next_step_indices=np.empty(0, dtype=np.int32),
            dones=np.empty(0, dtype=bool),
            td_errors=np.empty(0, dtype=np.float32),
            previous_actions=np.empty(0, dtype=np.float32),
            next_previous_actions=np.empty(0, dtype=np.float32),
            trading_infos=np.empty((0, 4), dtype=np.float32),
            next_trading_infos=np.empty((0, 4), dtype=np.float32),
            avaliable_actions=np.empty((0, 0), dtype=np.float32),
            next_avaliable_actions=np.empty((0, 0), dtype=np.float32),
            hours=np.empty(0, dtype=np.float32),
            minutes=np.empty(0, dtype=np.float32),
            next_hours=np.empty(0, dtype=np.float32),
            next_minutes=np.empty(0, dtype=np.float32),
        )

    gammas = np.array([gamma ** i for i in range(n_step)], dtype=np.float32)
    acc_rewards = np.zeros(total_steps, dtype=np.float32)
    next_step_indices = np.zeros(total_steps, dtype=np.int32)
    acc_dones = np.zeros(total_steps, dtype=bool)

    acc_next_prev_actions = np.zeros(total_steps, dtype=np.float32)
    acc_next_hours = np.zeros(total_steps, dtype=np.float32)
    acc_next_minutes = np.zeros(total_steps, dtype=np.float32)

    acc_next_trading_infos = []
    acc_next_avail_actions = []
    acc_next_states = [] if raw_next_states is not None else None

    for t in range(total_steps):
        lookahead_len = min(n_step, total_steps - t)
        discounted_r = 0.0
        terminal_idx = -1
        for k in range(lookahead_len):
            step_k = t + k
            discounted_r += gammas[k] * float(raw_rewards[step_k])
            if raw_dones[step_k]:
                terminal_idx = step_k
                break

        acc_rewards[t] = discounted_r
        if terminal_idx != -1:
            next_step_indices[t] = raw_step_indices[terminal_idx]
            acc_dones[t] = True
            acc_next_prev_actions[t] = raw_next_previous_actions[terminal_idx]
            acc_next_trading_infos.append(raw_next_trading_infos[terminal_idx])
            acc_next_avail_actions.append(raw_next_avaliable_actions[terminal_idx])
            acc_next_hours[t] = raw_next_hours[terminal_idx]
            acc_next_minutes[t] = raw_next_minutes[terminal_idx]
            if acc_next_states is not None:
                acc_next_states.append(raw_next_states[terminal_idx])
        elif t + n_step < total_steps:
            target_step = t + n_step
            next_step_indices[t] = raw_step_indices[target_step]
            acc_dones[t] = False
            acc_next_prev_actions[t] = raw_previous_actions[target_step]
            acc_next_trading_infos.append(raw_trading_infos[target_step])
            acc_next_avail_actions.append(raw_avaliable_actions[target_step])
            acc_next_hours[t] = raw_hours[target_step]
            acc_next_minutes[t] = raw_minutes[target_step]
            if acc_next_states is not None:
                acc_next_states.append(raw_states[target_step])
        else:
            next_step_indices[t] = raw_step_indices[-1]
            acc_dones[t] = True
            acc_next_prev_actions[t] = raw_next_previous_actions[-1]
            acc_next_trading_infos.append(raw_next_trading_infos[-1])
            acc_next_avail_actions.append(raw_next_avaliable_actions[-1])
            acc_next_hours[t] = raw_next_hours[-1]
            acc_next_minutes[t] = raw_next_minutes[-1]
            if acc_next_states is not None:
                acc_next_states.append(raw_next_states[-1])

    q_vals_arr = (
        np.ascontiguousarray(np.stack(raw_q_values), dtype=np.float32)
        if raw_q_values is not None and len(raw_q_values) > 0
        else None
    )
    regime_ids_arr = (
        np.ascontiguousarray(raw_regime_grid_ids, dtype=np.int32)
        if raw_regime_grid_ids is not None and len(raw_regime_grid_ids) > 0
        else None
    )
    states_arr = (
        np.ascontiguousarray(np.stack(raw_states), dtype=np.float32)
        if raw_states is not None and len(raw_states) > 0
        else None
    )
    next_states_arr = (
        np.ascontiguousarray(np.stack(acc_next_states), dtype=np.float32)
        if acc_next_states is not None and len(acc_next_states) > 0
        else None
    )

    return CompactTrajectoryRecord(
        df_index=df_index,
        step_indices=np.ascontiguousarray(raw_step_indices, dtype=np.int32),
        actions=np.ascontiguousarray(raw_actions, dtype=np.int64),
        rewards=acc_rewards,
        next_step_indices=next_step_indices,
        dones=acc_dones,
        td_errors=np.ascontiguousarray(raw_td_errors, dtype=np.float32),
        previous_actions=np.ascontiguousarray(raw_previous_actions, dtype=np.float32),
        next_previous_actions=acc_next_prev_actions,
        trading_infos=np.ascontiguousarray(np.stack(raw_trading_infos), dtype=np.float32),
        next_trading_infos=np.ascontiguousarray(np.stack(acc_next_trading_infos), dtype=np.float32),
        avaliable_actions=np.ascontiguousarray(np.stack(raw_avaliable_actions), dtype=np.float32),
        next_avaliable_actions=np.ascontiguousarray(np.stack(acc_next_avail_actions), dtype=np.float32),
        hours=np.ascontiguousarray(raw_hours, dtype=np.float32),
        minutes=np.ascontiguousarray(raw_minutes, dtype=np.float32),
        next_hours=acc_next_hours,
        next_minutes=acc_next_minutes,
        q_values=q_vals_arr,
        regime_grid_ids=regime_ids_arr,
        states=states_arr,
        next_states=next_states_arr,
    )


def build_spatio_temporal_semantic_key(
    df_index: int,
    step_index: int,
    previous_action: int,
    action: int,
    pos_dir: int,
    dur_bucket: int,
) -> tuple[int, int, int, int, int, int]:
    """提取经验的时空决策去重键：(df_index, step_index, previous_action, action, pos_dir, dur_bucket)。"""
    return (
        int(df_index),
        int(step_index),
        int(previous_action),
        int(action),
        int(pos_dir),
        int(dur_bucket),
    )


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
        shared_market_data: Any | None = None,
    ) -> None:
        self.num_grids = num_grids
        self.grid_capacity = total_buffer_size // num_grids
        self.shared_market_data = shared_market_data
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
        """批量聚合写入探索记录，优先使用紧凑时空去重与批量 GPU 张量写入。"""
        if not round_records:
            return 0

        has_compact = (
            self.shared_market_data is not None
            and any(getattr(r, "compact_trajectory", None) is not None for r in round_records)
        )
        if has_compact:
            return self._add_compact_records_bulk(round_records)
        return self._add_legacy_records_bulk(round_records)

    def _add_compact_records_bulk(self, round_records: list[Any]) -> int:
        """基于紧凑时空数值流与 SharedMarketDataPack 的纳秒级去重与显存直写快速流水线。"""
        duplicate_count = 0
        pending_by_grid: dict[int, dict[int, tuple[Any, int, Any, int]]] = {
            g: {} for g in range(self.num_grids)
        }

        sorted_results = sorted(round_records, key=lambda r: getattr(r, "df_index", 0))

        for result in sorted_results:
            ct = getattr(result, "compact_trajectory", None)
            if ct is None:
                continue
            df_index = result.df_index
            if self.shared_market_data is None or df_index not in self.shared_market_data:
                continue
            pack = self.shared_market_data[df_index]
            total_steps = len(ct.step_indices)
            if total_steps == 0:
                continue

            if not self._initialized:
                s_dim = int(pack.state_tensor.shape[1])
                a_dim = int(ct.avaliable_actions.shape[1])
                self._init_tensors(s_dim, a_dim)

            for i in range(total_steps):
                step_idx = int(ct.step_indices[i])
                grid_id = (
                    int(ct.regime_grid_ids[i])
                    if ct.regime_grid_ids is not None
                    else int(pack.regime_grid_ids_array[step_idx])
                )
                if grid_id < 0 or grid_id >= self.num_grids:
                    duplicate_count += 1
                    continue

                action = int(ct.actions[i])
                prev_action = int(ct.previous_actions[i])
                pos_dir = int(np.round(float(ct.trading_infos[i, 0])))
                raw_steps = int(float(ct.trading_infos[i, 3]) * 180)
                if raw_steps <= 2:
                    dur_bucket = 0
                elif raw_steps <= 12:
                    dur_bucket = 1
                elif raw_steps <= 36:
                    dur_bucket = 2
                else:
                    dur_bucket = 3

                semantic_key = build_spatio_temporal_semantic_key(
                    df_index=df_index,
                    step_index=step_idx,
                    previous_action=prev_action,
                    action=action,
                    pos_dir=pos_dir,
                    dur_bucket=dur_bucket,
                )
                td_error = float(ct.td_errors[i])
                tracker = self.seen_fingerprints[grid_id]

                if semantic_key in tracker:
                    old_idx, old_td_error = tracker[semantic_key]
                    if td_error > old_td_error:
                        tracker[semantic_key] = (old_idx, td_error)
                        pending_by_grid[grid_id][old_idx] = (pack, df_index, ct, i)
                        self.total_added_count += 1
                    else:
                        duplicate_count += 1
                    continue

                if self.grid_sizes[grid_id] < self.grid_capacity:
                    idx = self.grid_sizes[grid_id]
                    self.grid_sizes[grid_id] += 1
                    tracker[semantic_key] = (idx, td_error)
                    self.keys_by_idx[grid_id][idx] = semantic_key
                    pending_by_grid[grid_id][idx] = (pack, df_index, ct, i)
                else:
                    idx = self.write_ptrs[grid_id]
                    old_key = self.keys_by_idx[grid_id][idx]
                    if old_key in tracker:
                        del tracker[old_key]
                    tracker[semantic_key] = (idx, td_error)
                    self.keys_by_idx[grid_id][idx] = semantic_key
                    pending_by_grid[grid_id][idx] = (pack, df_index, ct, i)
                    self.write_ptrs[grid_id] = (idx + 1) % self.grid_capacity

                self.total_added_count += 1

        device = self.device
        for g, pending in pending_by_grid.items():
            if not pending:
                continue
            indices = list(pending.keys())
            items = [pending[idx] for idx in indices]
            idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)

            states_list = []
            next_states_list = []
            q_values_list = []

            for pack, df_idx, ct, i in items:
                step_i = int(ct.step_indices[i])
                next_step_i = int(ct.next_step_indices[i])
                states_list.append(pack.state_array[step_i])
                next_states_list.append(pack.state_array[next_step_i])
                if ct.q_values is not None:
                    q_values_list.append(ct.q_values[i])
                elif pack.q_table_tensor is not None:
                    prev_act = int(ct.previous_actions[i])
                    q_values_list.append(pack.q_table_array[step_i][prev_act])
                else:
                    q_values_list.append(np.zeros(self.action_dim, dtype=np.float32))

            states_np = np.ascontiguousarray(np.stack(states_list), dtype=np.float32)
            next_states_np = np.ascontiguousarray(np.stack(next_states_list), dtype=np.float32)
            q_values_np = np.ascontiguousarray(np.stack(q_values_list), dtype=np.float32)

            actions_np = np.ascontiguousarray([[ct.actions[i]] for _, _, ct, i in items], dtype=np.int64)
            rewards_np = np.ascontiguousarray([[ct.rewards[i]] for _, _, ct, i in items], dtype=np.float32)
            dones_np = np.ascontiguousarray([[1.0 if ct.dones[i] else 0.0] for _, _, ct, i in items], dtype=np.float32)

            prev_actions_np = np.ascontiguousarray([ct.previous_actions[i] for _, _, ct, i in items], dtype=np.float32)
            avail_actions_np = np.ascontiguousarray(np.stack([ct.avaliable_actions[i] for _, _, ct, i in items]), dtype=np.float32)
            hours_np = np.ascontiguousarray([ct.hours[i] for _, _, ct, i in items], dtype=np.float32)
            minutes_np = np.ascontiguousarray([ct.minutes[i] for _, _, ct, i in items], dtype=np.float32)
            trading_infos_np = np.ascontiguousarray(np.stack([ct.trading_infos[i] for _, _, ct, i in items]), dtype=np.float32)

            next_prev_actions_np = np.ascontiguousarray([ct.next_previous_actions[i] for _, _, ct, i in items], dtype=np.float32)
            next_avail_actions_np = np.ascontiguousarray(np.stack([ct.next_avaliable_actions[i] for _, _, ct, i in items]), dtype=np.float32)
            next_hours_np = np.ascontiguousarray([ct.next_hours[i] for _, _, ct, i in items], dtype=np.float32)
            next_minutes_np = np.ascontiguousarray([ct.next_minutes[i] for _, _, ct, i in items], dtype=np.float32)
            next_trading_infos_np = np.ascontiguousarray(np.stack([ct.next_trading_infos[i] for _, _, ct, i in items]), dtype=np.float32)

            self.states[g][idx_tensor] = torch.from_numpy(states_np).to(device, non_blocking=True)
            self.actions[g][idx_tensor] = torch.from_numpy(actions_np).to(device, non_blocking=True)
            self.rewards[g][idx_tensor] = torch.from_numpy(rewards_np).to(device, non_blocking=True)
            self.next_states[g][idx_tensor] = torch.from_numpy(next_states_np).to(device, non_blocking=True)
            self.dones[g][idx_tensor] = torch.from_numpy(dones_np).to(device, non_blocking=True)

            self.previous_actions[g][idx_tensor] = torch.from_numpy(prev_actions_np).to(device, non_blocking=True)
            self.avaliable_actions[g][idx_tensor] = torch.from_numpy(avail_actions_np).to(device, non_blocking=True)
            self.hours[g][idx_tensor] = torch.from_numpy(hours_np).to(device, non_blocking=True)
            self.minutes[g][idx_tensor] = torch.from_numpy(minutes_np).to(device, non_blocking=True)
            self.trading_infos[g][idx_tensor] = torch.from_numpy(trading_infos_np).to(device, non_blocking=True)
            self.q_values[g][idx_tensor] = torch.from_numpy(q_values_np).to(device, non_blocking=True)

            self.next_previous_actions[g][idx_tensor] = torch.from_numpy(next_prev_actions_np).to(device, non_blocking=True)
            self.next_avaliable_actions[g][idx_tensor] = torch.from_numpy(next_avail_actions_np).to(device, non_blocking=True)
            self.next_hours[g][idx_tensor] = torch.from_numpy(next_hours_np).to(device, non_blocking=True)
            self.next_minutes[g][idx_tensor] = torch.from_numpy(next_minutes_np).to(device, non_blocking=True)
            self.next_trading_infos[g][idx_tensor] = torch.from_numpy(next_trading_infos_np).to(device, non_blocking=True)

        return duplicate_count

    def _add_legacy_records_bulk(self, round_records: list[Any]) -> int:
        """传统基于元组与字典记录的入池链路（保留完整兼容性）。"""
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
