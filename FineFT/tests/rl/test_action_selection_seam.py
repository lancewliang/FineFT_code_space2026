import numpy as np
import pytest
import torch

from FineFT.RL.DiHFT.low_level.action_selection_seam import select_greedy_action


def test_select_greedy_action_disabled_returns_argmax():
    q_values = np.array([0.1, 0.5, 0.3, 0.8, 0.2])
    available_actions = np.array([True, True, True, True, True])

    action, diag = select_greedy_action(
        q_values,
        current_action=0,
        available_actions=available_actions,
        enabled=False,
    )

    assert action == 3
    assert diag["decision_reason"] == "q_argmax"


def test_select_greedy_action_respects_available_mask():
    q_values = np.array([0.1, 0.5, 0.3, 0.8, 0.2])
    available_actions = np.array([True, True, True, False, True])  # 3 is unavailable

    action, diag = select_greedy_action(
        q_values,
        current_action=0,
        available_actions=available_actions,
        enabled=False,
    )

    assert action == 1  # 0.5 is max among available [0, 1, 2, 4]
    assert diag["decision_reason"] == "q_argmax"


def test_select_greedy_action_tie_breaking_order():
    q_values = np.array([0.5, 0.5, 0.1, 0.2])
    available_actions = np.array([True, True, True, True])

    action, _ = select_greedy_action(
        q_values,
        current_action=2,
        available_actions=available_actions,
        enabled=False,
    )

    assert action == 0  # first max index on tie


def test_select_greedy_action_hysteresis_maintained_when_advantage_below_threshold():
    q_values = np.array([0.5, 0.52, 0.1])
    available_actions = np.array([True, True, True])
    estimated_costs = np.array([0.0, 0.05, 0.0])

    # candidate is 1, Q(1) - Q(0) = 0.02. threshold = 1.0 * 0.05 + 0.0 = 0.05.
    # 0.02 <= 0.05 -> maintain current_action = 0
    action, diag = select_greedy_action(
        q_values,
        current_action=0,
        available_actions=available_actions,
        estimated_costs=estimated_costs,
        cost_multiplier=1.0,
        safety_margin=0.0,
        enabled=True,
    )

    assert action == 0
    assert diag["decision_reason"] == "hysteresis_maintained"


def test_select_greedy_action_hysteresis_switched_when_advantage_above_threshold():
    q_values = np.array([0.5, 0.60, 0.1])
    available_actions = np.array([True, True, True])
    estimated_costs = np.array([0.0, 0.05, 0.0])

    # candidate is 1, Q(1) - Q(0) = 0.10 > threshold 0.05 -> switch to 1
    action, diag = select_greedy_action(
        q_values,
        current_action=0,
        available_actions=available_actions,
        estimated_costs=estimated_costs,
        cost_multiplier=1.0,
        safety_margin=0.0,
        enabled=True,
    )

    assert action == 1
    assert diag["decision_reason"] == "hysteresis_switched"


def test_select_greedy_action_bypasses_when_current_unavailable():
    q_values = np.array([0.5, 0.52, 0.1])
    available_actions = np.array([False, True, True])  # current action 0 is unavailable

    action, diag = select_greedy_action(
        q_values,
        current_action=0,
        available_actions=available_actions,
        cost_multiplier=1.0,
        enabled=True,
    )

    assert action == 1
    assert diag["decision_reason"] == "current_unavailable"


def test_select_greedy_action_rejects_negative_parameters():
    q_values = np.array([0.1, 0.2])
    available = [True, True]

    with pytest.raises(ValueError, match="cost_multiplier"):
        select_greedy_action(q_values, 0, available, cost_multiplier=-1.0, enabled=True)

    with pytest.raises(ValueError, match="safety_margin"):
        select_greedy_action(q_values, 0, available, safety_margin=-0.5, enabled=True)
