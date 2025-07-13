from collections import deque
import numpy as np
import torch
import scipy.signal
import os
import random
import torch.nn.functional as F


# * update util
def evaluate_quantile_at_action(s_quantiles, actions):
    assert s_quantiles.shape[0] == actions.shape[0]

    batch_size = s_quantiles.shape[0]
    N = s_quantiles.shape[1]

    # Expand actions into (batch_size, N, 1).
    action_index = actions[..., None].expand(batch_size, N, 1)

    # Calculate quantile values at specified actions.
    sa_quantiles = s_quantiles.gather(dim=2, index=action_index)

    return sa_quantiles


def update_params(optim, loss, network, retain_graph=False, grad_cliping=None):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    # Clip norms of gradients to stebilize training.
    if grad_cliping:
        torch.nn.utils.clip_grad_norm_(network.parameters(), grad_cliping)
    optim.step()


def soft_copy_params(online_net, target_net, tau):
    for target_param, online_param in zip(
        target_net.parameters(), online_net.parameters()
    ):
        target_param.data.copy_(
            tau * online_param.data + (1.0 - tau) * target_param.data
        )


def disable_gradients(network):
    # Disable calculations of gradients.
    for param in network.parameters():
        param.requires_grad = False


# * loss util


def calculate_huber_loss(td_errors, kappa=1.0):
    return torch.where(
        td_errors.abs() <= kappa,
        0.5 * td_errors.pow(2),
        kappa * (td_errors.abs() - 0.5 * kappa),
    )


def find_proper_index(td_errors, outerbond=4, reachout_index=1):
    device = td_errors.device
    batch_size, N, N_dash = td_errors.shape
    # N eval shape, N_dash target shape
    abs_td_errors = td_errors.abs()
    td_errors_diag = torch.diagonal(abs_td_errors, dim1=1, dim2=2)
    min_values, chosen_indices = torch.min(td_errors_diag, dim=1)

    error_bear_threshold = min_values + outerbond
    error_bear_threshold = error_bear_threshold.unsqueeze(1)
    # eval range
    # 针对每一个batch 返回四个值 分别是N的min max以及N_dash的min max
    chosen_indices_expand = chosen_indices.view(batch_size, 1, 1).expand(-1, 1, N)
    eval_target_quantile = torch.gather(
        abs_td_errors, 1, chosen_indices_expand  # 1 reprensnets eval shape
    )
    chosen_indices = chosen_indices.unsqueeze(1)
    eval_target_quantile = eval_target_quantile.squeeze(1)
    threshold_mask = eval_target_quantile <= error_bear_threshold
    range_index = find_batch_bounds_from_index(
        threshold_mask, chosen_indices, reachout_index, device=device
    )
    assert range_index.shape == (batch_size, 2)
    return range_index, chosen_indices


def find_max_index(input, device):
    indices_of_ones = input == 1
    max_indices = (
        torch.where(
            indices_of_ones,
            torch.arange(input.shape[1]).to(device),
            torch.tensor(0).to(device),
        )
        .max(dim=1)
        .values
    )
    return max_indices


def find_batch_bounds_from_index(mask_index, index, reachout_number=1, device="cuda"):
    batch_size, seq_len = mask_index.shape
    range_tensor = (
        torch.arange(seq_len, device=mask_index.device)
        .expand(batch_size, -1)
        .to(device)
    )
    index_expanded = index.expand(batch_size, seq_len)
    # left masks to see the right index, mask using 1
    left_masks = range_tensor < index_expanded
    # right masks to see the left index, mask using 1
    right_masks = range_tensor > index_expanded

    right_mask_index_matrix = mask_index + left_masks

    left_mask_index_matrix = mask_index + right_masks
    right_mask_index_matrix = torch.cumprod(right_mask_index_matrix, dim=1)
    flip_left_mask_index_matrix = torch.flip(left_mask_index_matrix, [1])
    left_mask_index_matrix = torch.cumprod(flip_left_mask_index_matrix, dim=1)
    maxium_index = find_max_index(right_mask_index_matrix, device)
    maxium_index = maxium_index + reachout_number
    minum_index = seq_len - 1 - find_max_index(left_mask_index_matrix, device)
    minum_index = minum_index - reachout_number
    range_index = torch.cat(
        [minum_index.unsqueeze(1), maxium_index.unsqueeze(1)], dim=1
    ).to(device)
    range_index = torch.clamp(range_index, 0, seq_len - 1)
    return range_index


# weights for TD error
def compute_error_weights(range_index, chosen_indices, N):
    assert range_index.shape[1] == 2
    assert range_index.shape[0] == chosen_indices.shape[0]
    assert range_index.device == chosen_indices.device
    device = range_index.device
    batch_size = range_index.shape[0]
    substraction = range_index[:, 1] - range_index[:, 0]
    # initialize_weights
    weights = torch.zeros(batch_size, N, N).to(device)
    # chosen_index
    batch_indices = torch.arange(batch_size).to(device)
    # diganoal_index

    # 使用这些索引来设置对应的对角元素为1
    weights[batch_indices, chosen_indices, chosen_indices] = 1
    diagonal_indices = torch.arange(N).unsqueeze(0).expand(batch_size, -1).to(device)

    # 生成一个mask，标记在range内的对角线元素位置
    diagonal_mask = (diagonal_indices >= range_index[:, 0].unsqueeze(1)) & (
        diagonal_indices <= range_index[:, 1].unsqueeze(1)
    ).to(device)

    # 计算每个对角线元素与chosen_index的差的绝对值
    distances = torch.abs(diagonal_indices - chosen_indices.unsqueeze(1))

    # 计算权重更新值
    updates = 1 - distances.float() / substraction.unsqueeze(1)
    batch_weights = updates * diagonal_mask
    # 应用updates到weights的对角线，但只在diagonal_mask范围内
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, N).to(device)
    weights[batch_indices, diagonal_indices, diagonal_indices] = torch.where(
        diagonal_mask,
        updates,
        weights[batch_indices, diagonal_indices, diagonal_indices],
    )

    # 确保chosen_index对应的对角线元素为1
    weights[torch.arange(batch_size), chosen_indices, chosen_indices] = 1
    diagonal = weights.diagonal(dim1=-2, dim2=-1)
    substraction = diagonal != 0.0  # 非零对角元素的布尔索引

    # 行列索引
    rows, cols = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")
    rows = rows.expand(batch_size, -1, -1).to(device)
    cols = cols.expand(batch_size, -1, -1).to(device)

    # 计算行列差的绝对值
    row_col_diff = torch.abs(rows - cols).float()

    # 计算每个位置的更新值
    min_diag = torch.min(diagonal.unsqueeze(2), diagonal.unsqueeze(1)) #计算一个位置上的所在行和列的对角线的元素更小的那个
    update_values = min_diag * (
        1 - (row_col_diff / (diagonal != 0).sum(dim=1, keepdim=True).unsqueeze(2))
    )

    # 将对角线元素设置为 0，以便不影响原始的对角线元素
    update_values.diagonal(dim1=-2, dim2=-1).zero_()

    # 应用计算出的更新值
    weights += update_values
    batch_weights = batch_weights.detach()
    weights = weights.detach()
    return batch_weights, weights


def calculate_partial_loss(td_errors, outer_bond, reach_out_index):
    assert td_errors.requires_grad == True
    batch_size, N, N_dash = td_errors.shape
    range_index, chosen_indices = find_proper_index(
        td_errors, outer_bond, reach_out_index
    )
    chosen_indices = chosen_indices.squeeze(1)

    batch_weights, weights = compute_error_weights(range_index, chosen_indices, N)
    averaged_error = td_errors * weights
    batch_loss = averaged_error.sum(dim=1).mean(dim=1, keepdim=True)
    assert batch_loss.shape == (batch_size, 1)
    loss = batch_loss.mean()
    return batch_weights, loss


def recalculate_q_demonstration(q_values, avaliable_actions, max_punish=1e12):
    batch_size, num_action = q_values.shape
    assert avaliable_actions.shape == (batch_size, num_action)
    q_values = q_values - (1 - avaliable_actions) * max_punish
    return q_values


# rank related
def get_rank(td_error, device):
    assert td_error.dim() == 2
    assert td_error.shape[0] == 1
    N = td_error.shape[1]
    sorted_indices = torch.argsort(td_error, dim=1)

    # 创建一个与 td_error 相同形状的 rank 矩阵
    rank = torch.zeros_like(td_error).to(device)

    # 使用排序后的索引来填充 rank 矩阵
    rank[0, sorted_indices] = torch.arange(
        1, N + 1, dtype=rank.dtype, device=rank.device
    ).to(device)
    return rank


if __name__ == "__main__":
    td_error = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    rank = get_rank(td_error)
    print(rank)
