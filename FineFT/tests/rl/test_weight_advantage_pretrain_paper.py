import math

import pytest
import torch


def test_paper_supervisor_kl_weights_each_learner_before_reduction():
    from RL.DiHFT.low_level.weight_advantage_pretrain_paper import (
        calculate_paper_supervisor_kl_loss,
    )

    learner_action_values = torch.tensor(
        [
            [
                [math.log(0.75), math.log(0.25)],
                [math.log(0.50), math.log(0.50)],
            ]
        ],
        requires_grad=True,
    )
    expert_action_values = torch.tensor(
        [[math.log(0.75), math.log(0.25)]]
    )
    learner_weights = torch.tensor([[1.0, 0.5]])

    loss = calculate_paper_supervisor_kl_loss(
        learner_action_values,
        expert_action_values,
        learner_weights,
    )

    assert loss.item() == pytest.approx(0.07192052, abs=1e-7)


def test_paper_weight_matrix_uses_fixed_neighbors_and_paper_decay():
    from RL.DiHFT.low_level.weight_advantage_pretrain_paper import (
        construct_paper_weight_matrix,
    )

    etd_losses = torch.full((1, 5, 5), 10.0)
    etd_losses[0].diagonal().copy_(torch.tensor([4.0, 3.0, 1.0, 2.0, 5.0]))

    learner_weights, weight_matrix = construct_paper_weight_matrix(
        etd_losses,
        neighbor_size=1,
    )

    expected_matrix = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.25, 0.0, 0.0],
            [0.0, 0.25, 1.0, 0.25, 0.0],
            [0.0, 0.0, 0.25, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    torch.testing.assert_close(weight_matrix[0], expected_matrix)
    torch.testing.assert_close(learner_weights[0], expected_matrix.diagonal())


def test_paper_partial_loss_uses_the_paper_weight_matrix():
    from RL.DiHFT.low_level.weight_advantage_pretrain_paper import (
        calculate_paper_partial_loss,
    )

    etd_losses = torch.ones((1, 5, 5), requires_grad=True)
    etd_losses = etd_losses + torch.diag_embed(
        torch.tensor([[3.0, 2.0, 0.0, 1.0, 4.0]])
    )

    learner_weights, loss = calculate_paper_partial_loss(
        etd_losses,
        neighbor_size=1,
    )

    torch.testing.assert_close(
        learner_weights,
        torch.tensor([[0.0, 0.5, 1.0, 0.5, 0.0]]),
    )
    assert loss.item() == pytest.approx(0.9)


def test_paper_training_parser_exposes_fixed_neighbor_size():
    from RL.DiHFT.low_level import weight_advantage_pretrain_paper as paper

    args = paper.parser.parse_args([])

    assert args.neighbor_size == 1
