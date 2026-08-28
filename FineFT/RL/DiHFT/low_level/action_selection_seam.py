from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import torch


def select_greedy_action(
    q_values: np.ndarray | torch.Tensor,
    current_action: int,
    available_actions: Sequence[bool | int] | np.ndarray | torch.Tensor,
    estimated_costs: np.ndarray | torch.Tensor | Sequence[float] | None = None,
    *,
    cost_multiplier: float = 0.0,
    safety_margin: float = 0.0,
    enabled: bool = False,
    is_risk_emergency: bool = False,
) -> tuple[int, dict[str, Any]]:
    """
    共享 Low-level 贪心动作选择接缝 (Shared Deterministic Greedy Action Selection Seam)。

    主要作用：
    1. **统一动作选择入口**：为 Stage I 训练贪心分支、独立 Low-level 测试以及 High-level 路由后的 Low-level 推理提供单一且行为一致的动作决策接缝。
    2. **成本感知动作迟滞 (Hysteresis)**：当 Q 值的切换优势 `Q(candidate) - Q(current)` 未能严格超过换仓摩擦成本与安全边际门槛 `cost_multiplier * estimated_cost + safety_margin` 时，强制保持当前仓位动作，减少高频过度换仓。
    3. **硬风险/可用性优先**：若当前动作因强平、涨跌停或保证金不足等环境约束不再可用（或触发风控规则），优先执行最佳可用动作，不因迟滞保持不可用动作。
    4. **平滑兼容退化**：当迟滞规则未启用 (`enabled=False` 或门槛参数为 0) 时，退化为标准的掩码 Q 值的 argmax 动作选择，保持现有模型与基线代码完全兼容。

    参数说明:
    ----------
    q_values : 逐动作 Q 值数组或 Tensor，形状 (num_actions,) 或 (1, num_actions)。
    current_action : 当前决策前的持仓动作编号。
    available_actions : 当前决策点可用的动作掩码 (bool 数组) 或可用动作编号列表。
    estimated_costs : 逐可用动作预计换仓摩擦成本（手续费 + 滑点）。
    cost_multiplier : 预计换仓成本倍数。
    safety_margin : 换仓所需的额外 Q 优势安全边际。
    enabled : 是否启用成本感知动作迟滞；False 时退化为标准 argmax 贪心选择。
    is_risk_emergency : 是否触发风控紧急处理，True 时绕过迟滞保持规则。

    返回:
    -------
    (selected_action, diagnostic_dict)
        selected_action : 最终选出的动作编号。
        diagnostic_dict : 包含决策原因 (decision_reason)、Q 优势、预计成本、门槛等逐步诊断信息。
    """
    if cost_multiplier < 0.0:
        raise ValueError(f"cost_multiplier must be non-negative: {cost_multiplier}")
    if safety_margin < 0.0:
        raise ValueError(f"safety_margin must be non-negative: {safety_margin}")

    if isinstance(q_values, torch.Tensor):
        q_arr = q_values.detach().cpu().numpy().reshape(-1).astype(float)
    else:
        q_arr = np.asarray(q_values, dtype=float).reshape(-1)

    if not np.isfinite(q_arr).all():
        raise ValueError(f"q_values contains non-finite values: {q_arr}")

    num_actions = len(q_arr)

    # Process available_actions mask
    if isinstance(available_actions, torch.Tensor):
        avail_arr = available_actions.detach().cpu().numpy().reshape(-1)
    else:
        avail_arr = np.asarray(available_actions)

    if avail_arr.dtype == bool or (avail_arr.ndim == 1 and len(avail_arr) == num_actions and set(avail_arr).issubset({0, 1, True, False})):
        mask = avail_arr.astype(bool)
    else:
        # List of integer action indices
        mask = np.zeros(num_actions, dtype=bool)
        mask[avail_arr.astype(int)] = True

    if not mask.any():
        raise ValueError("available_actions mask is empty or all False")

    # Masked argmax
    masked_q = np.where(mask, q_arr, -np.inf)
    candidate = int(np.argmax(masked_q))
    if masked_q[candidate] == -np.inf:
        raise ValueError("No available action has finite Q-value")

    # Process estimated costs
    if estimated_costs is not None:
        if isinstance(estimated_costs, torch.Tensor):
            cost_arr = estimated_costs.detach().cpu().numpy().reshape(-1).astype(float)
        else:
            cost_arr = np.asarray(estimated_costs, dtype=float).reshape(-1)
        if not np.isfinite(cost_arr).all() or (cost_arr < 0.0).any():
            raise ValueError(f"estimated_costs must be finite and non-negative: {cost_arr}")
    else:
        cost_arr = np.zeros(num_actions, dtype=float)

    current_action = int(current_action)

    diagnostic: dict[str, Any] = {
        "current_action": current_action,
        "candidate_action": candidate,
        "selected_action": candidate,
        "q_advantage": 0.0,
        "estimated_cost": float(cost_arr[candidate]),
        "threshold": 0.0,
        "decision_reason": "q_argmax",
    }

    if not enabled or (cost_multiplier == 0.0 and safety_margin == 0.0):
        return candidate, diagnostic

    # Hysteresis mode is enabled
    current_is_available = bool(0 <= current_action < num_actions and mask[current_action])

    if is_risk_emergency or not current_is_available:
        reason = "risk_bypass" if is_risk_emergency else "current_unavailable"
        diagnostic.update({
            "selected_action": candidate,
            "decision_reason": reason,
        })
        return candidate, diagnostic

    if candidate == current_action:
        diagnostic.update({
            "selected_action": current_action,
            "decision_reason": "q_argmax",
        })
        return current_action, diagnostic

    q_adv = float(q_arr[candidate] - q_arr[current_action])
    candidate_cost = float(cost_arr[candidate])
    threshold = float(cost_multiplier * candidate_cost + safety_margin)

    diagnostic.update({
        "q_advantage": q_adv,
        "estimated_cost": candidate_cost,
        "threshold": threshold,
    })

    if q_adv > threshold:
        diagnostic.update({
            "selected_action": candidate,
            "decision_reason": "hysteresis_switched",
        })
        return candidate, diagnostic
    else:
        diagnostic.update({
            "selected_action": current_action,
            "decision_reason": "hysteresis_maintained",
        })
        return current_action, diagnostic
