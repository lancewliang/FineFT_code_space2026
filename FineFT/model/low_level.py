import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

MAX_PUNISHMENT = 1e12


# Q network for FinFT
# without holding length as input
class Qnet(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS, hidden_nodes, TIME_INFO_DIM):
        super(Qnet, self).__init__()
        self.time_bedding = int(hidden_nodes / 2)
        self.fc1 = nn.Linear(N_STATES, hidden_nodes)
        self.fc2 = nn.Linear(N_ACTIONS + hidden_nodes + self.time_bedding, hidden_nodes)
        self.out = nn.Linear(hidden_nodes, N_ACTIONS)
        self.fc3 = nn.Linear(1, N_ACTIONS)
        self.fc_time = nn.Linear(TIME_INFO_DIM, self.time_bedding)
        self.register_buffer("max_punish", torch.tensor(MAX_PUNISHMENT))

    def forward(
        self,
        state: torch.tensor,
        time: torch.tensor,
        previous_action: torch.tensor,
        avaliable_action: torch.tensor,
    ):
        state_hidden = F.relu(self.fc1(state))
        time = self.fc_time(time)
        previous_action_hidden = F.relu(self.fc3(previous_action))
        information_hidden = torch.cat(
            [state_hidden, previous_action_hidden, time], dim=1
        )
        information_hidden = self.fc2(information_hidden)
        action = self.out(information_hidden)
        masked_action = action + (avaliable_action - 1) * self.max_punish
        return masked_action


class ensemble_Qnet(nn.Module):
    def __init__(
        self, N_STATES, N_ACTIONS, hidden_nodes, TIME_INFO_DIM, ensemble_number
    ):
        super(ensemble_Qnet, self).__init__()
        self.ensemble_number = ensemble_number
        self.N_ACTIONS = N_ACTIONS
        self.qnet_list = nn.ModuleList(
            [
                Qnet(N_STATES, N_ACTIONS, hidden_nodes, TIME_INFO_DIM)
                for _ in range(ensemble_number)
            ]
        )

    def forward(
        self,
        state: torch.tensor,
        time: torch.tensor,
        previous_action: torch.tensor,
        avaliable_action: torch.tensor,
    ):
        q_values = torch.stack(
            [
                qnet(state, time, previous_action, avaliable_action)
                for qnet in self.qnet_list
            ],
            dim=1,
        )
        assert q_values.shape == (state.shape[0], self.ensemble_number, self.N_ACTIONS)
        return q_values

    def get_best_q(self, state, time, previous_action, avaliable_action):
        q_values = self.forward(state, time, previous_action, avaliable_action)
        # 在不同的context下选最好的动作
        best_q, _ = q_values.max(dim=2)
        return best_q


def load_selected_qnets_from_different_save_path(
    saved_model_path_list, selected_indices
):
    selected_state_dict = {}
    assert len(saved_model_path_list) == len(selected_indices)
    for index, save_path, idx in zip(
        range(len(saved_model_path_list)), saved_model_path_list, selected_indices
    ):
        saved_state_dict = torch.load(save_path)
        for key, value in saved_state_dict.items():
            if f"qnet_list.{idx}" in key:
                print("original key:", key)
                print(
                    "new key:",
                    key.replace(
                        f"qnet_list.{idx}",
                        f"qnet_list.{index}",
                    ),
                )
                new_key = key.replace(
                    f"qnet_list.{idx}",
                    f"qnet_list.{index}",
                )
                selected_state_dict[new_key] = value
    return selected_state_dict


def create_new_ensemble_qnet_from_different_save_path(
    N_STATES,
    N_ACTIONS,
    hidden_nodes,
    TIME_INFO_DIM,
    saved_model_path_list,
    selected_indices,
):
    new_ensemble = ensemble_Qnet(
        N_STATES, N_ACTIONS, hidden_nodes, TIME_INFO_DIM, len(selected_indices)
    )
    selected_state_dict = load_selected_qnets_from_different_save_path(
        saved_model_path_list, selected_indices
    )
    new_ensemble.load_state_dict(selected_state_dict, strict=False)
    return new_ensemble


def load_selected_qnets(saved_model_path, selected_indices):
    # 加载训练好的模型
    saved_state_dict = torch.load(saved_model_path)

    # 提取选择的子网络参数
    selected_state_dict = {}
    for index, idx in zip(range(len(selected_indices)), selected_indices):
        for key, value in saved_state_dict.items():
            if f"qnet_list.{idx}" in key:
                print("original key:", key)
                print(
                    "new key:",
                    key.replace(
                        f"qnet_list.{idx}",
                        f"qnet_list.{index}",
                    ),
                )
                new_key = key.replace(
                    f"qnet_list.{idx}",
                    f"qnet_list.{index}",
                )
                selected_state_dict[new_key] = value

    return selected_state_dict


def create_new_ensemble_qnet(
    N_STATES, N_ACTIONS, hidden_nodes, TIME_INFO_DIM, selected_indices, saved_model_path
):
    new_ensemble = ensemble_Qnet(
        N_STATES, N_ACTIONS, hidden_nodes, TIME_INFO_DIM, len(selected_indices)
    )
    selected_state_dict = load_selected_qnets(saved_model_path, selected_indices)
    new_ensemble.load_state_dict(selected_state_dict, strict=False)
    return new_ensemble


# low level agent for MacroHFT
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class subagent(nn.Module):
    def __init__(self, state_dim_1, state_dim_2, action_dim, hidden_dim):
        super(subagent, self).__init__()
        self.fc1 = nn.Linear(state_dim_1, hidden_dim)
        self.fc2 = nn.Linear(state_dim_2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.embedding = nn.Embedding(action_dim, hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * 4, action_dim),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * 4, 1),
        )

    def forward(
        self,
        single_state: torch.tensor,
        trend_state: torch.tensor,
        previous_action: torch.tensor,
    ):
        action_hidden = self.embedding(previous_action)
        single_state_hidden = self.fc1(single_state)
        trend_state_hidden = self.fc2(trend_state)
        c = action_hidden + trend_state_hidden
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(single_state_hidden), shift, scale)
        value = self.value(x)
        advantage = self.advantage(x)

        return value + advantage - advantage.mean()


# CQRDQN


class CQnet(nn.Module):
    def __init__(
        self, N_STATES, N_ACTIONS, hidden_nodes, kernel_size=20, TIME_INFO_DIM=2
    ):
        super(CQnet, self).__init__()
        self.time_bedding = int(hidden_nodes / 2)
        # Convolution layer, input shape should be (B, N, T) for Conv1d
        self.conv1 = nn.Conv1d(
            N_STATES, hidden_nodes, kernel_size=kernel_size, stride=1, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Fully connected layers
        # Note: You need to calculate the output size of the pooling layer
        # and use it as input size of fc1
        # For simplicity, I am assuming it as 'conv_output_size' here.
        conv_output_size = (
            hidden_nodes // 2
        )  # Replace this with actual output size after pooling
        self.fc1 = nn.Linear(
            conv_output_size + N_ACTIONS + self.time_bedding, hidden_nodes
        )
        self.fc2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.out = nn.Linear(hidden_nodes, N_ACTIONS)
        self.fc3 = nn.Linear(1, N_ACTIONS)
        self.fc_time = nn.Linear(TIME_INFO_DIM, self.time_bedding)
        self.register_buffer("max_punish", torch.tensor(MAX_PUNISHMENT))

    def forward(
        self,
        state: torch.tensor,
        time: torch.tensor,
        previous_action: torch.tensor,
        available_action: torch.tensor,
    ):
        # Changing input shape to (B, N, T) for Conv1d
        state = state.permute(0, 2, 1)

        # Convolution and pooling
        state = F.relu(self.conv1(state))
        state = self.pool(state)

        # Flattening the output of the pooling layer
        state = state.view(state.size(0), -1)
        time = self.fc_time(time)
        # Fully connected layers
        previous_action_hidden = F.relu(self.fc3(previous_action))
        information_hidden = torch.cat([state, previous_action_hidden, time], dim=1)
        information_hidden = F.relu(self.fc1(information_hidden))
        information_hidden = F.relu(self.fc2(information_hidden))
        action = self.out(information_hidden)

        # Masking
        masked_action = action + (available_action - 1) * self.max_punish

        return masked_action


# PPO network


def orthogonal_init(layer, gain=np.sqrt(2)):
    for name, param in layer.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.orthogonal_(param, gain=gain)

    return layer


# Actor


class Actor(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS, hidden_nodes, TIME_INFO_DIM=2):
        super(Actor, self).__init__()
        self.time_bedding = int(hidden_nodes / 2)
        self.fc_time = nn.Linear(TIME_INFO_DIM, self.time_bedding)
        self.fc1 = nn.Linear(N_STATES, hidden_nodes)
        self.fc2 = nn.Linear(N_ACTIONS + hidden_nodes + self.time_bedding, hidden_nodes)
        self.fc3 = nn.Linear(1, N_ACTIONS)
        self.out = nn.Linear(hidden_nodes, N_ACTIONS)
        self.activate_func = nn.Tanh()  # Trick10: use tanh
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)
        orthogonal_init(self.fc_time)
        orthogonal_init(self.out, gain=0.01)

        self.register_buffer("max_punish", torch.tensor(MAX_PUNISHMENT))

    def forward(
        self,
        state: torch.tensor,
        time: torch.tensor,
        previous_action: torch.tensor,
        avaliable_action: torch.tensor,
    ):
        state_hidden = self.activate_func(self.fc1(state))
        previous_action_hidden = self.activate_func(self.fc3(previous_action))
        time = self.activate_func(self.fc_time(time))
        information_hidden = torch.cat(
            [state_hidden, previous_action_hidden, time], dim=1
        )
        information_hidden = self.fc2(information_hidden)
        action = self.out(information_hidden)
        logit = action + (avaliable_action - 1) * self.max_punish
        return logit


class Critic(nn.Module):
    def __init__(self, N_STATES, hidden_nodes, TIME_INFO_DIM=2):

        super(Critic, self).__init__()
        self.time_bedding = int(hidden_nodes / 2)
        self.fc_time = nn.Linear(TIME_INFO_DIM, self.time_bedding)
        self.fc1 = nn.Linear(N_STATES, hidden_nodes)
        self.fc2 = nn.Linear(2 * hidden_nodes + self.time_bedding, hidden_nodes)
        self.fc3 = nn.Linear(1, hidden_nodes)
        self.out = nn.Linear(hidden_nodes, 1)
        self.activate_func = nn.Tanh()  # Trick10: use tanh
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)
        orthogonal_init(self.fc_time)
        orthogonal_init(self.out, gain=0.01)

    def forward(
        self,
        state: torch.tensor,
        time: torch.tensor,
        previous_action: torch.tensor,
    ):
        state_hidden = self.activate_func(self.fc1(state))
        previous_action_hidden = self.activate_func(self.fc3(previous_action))
        time = self.activate_func(self.fc_time(time))
        information_hidden = torch.cat(
            [state_hidden, previous_action_hidden, time], dim=1
        )
        information_hidden = self.fc2(information_hidden)
        state_value = self.out(information_hidden)
        return state_value


# PPO lstm code reference:https://github.com/Lizhi-sjtu/DRL-code-pytorch/tree/main


class Actor_Critic_RNN(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, TIME_INFO_DIM=2):
        super(Actor_Critic_RNN, self).__init__()
        self.activate_func = nn.Tanh()  # Trick10: use tanh
        self.time_bedding = int(hidden_dim / 2)

        self.actor_rnn_hidden = None
        self.actor_fc1 = nn.Linear(state_dim, hidden_dim)
        self.actor_fc_pa = nn.Linear(1, hidden_dim)
        self.actor_fc_time = nn.Linear(TIME_INFO_DIM, self.time_bedding)

        self.actor_rnn = nn.LSTM(
            2 * hidden_dim + self.time_bedding, hidden_dim, batch_first=True
        )
        self.actor_fc2 = nn.Linear(hidden_dim, action_dim)

        self.critic_rnn_hidden = None
        self.critic_fc1 = nn.Linear(state_dim, hidden_dim)
        self.critic_fc_pa = nn.Linear(1, hidden_dim)
        self.critic_fc_time = nn.Linear(TIME_INFO_DIM, self.time_bedding)

        self.critic_rnn = nn.LSTM(
            2 * hidden_dim + self.time_bedding, hidden_dim, batch_first=True
        )
        self.critic_fc2 = nn.Linear(hidden_dim, 1)
        # the gain should be smaller for the last layer for both actor and critic
        orthogonal_init(self.actor_fc1)
        orthogonal_init(self.actor_rnn)
        orthogonal_init(self.actor_fc2, gain=0.01)
        orthogonal_init(self.critic_fc1)
        orthogonal_init(self.critic_rnn)
        orthogonal_init(self.critic_fc2, gain=0.01)
        orthogonal_init(self.actor_fc_time)
        orthogonal_init(self.critic_fc_time)
        self.register_buffer("max_punish", torch.tensor(MAX_PUNISHMENT))

    def actor(
        self,
        s: torch.tensor,
        time: torch.tensor,
        previous_action: torch.tensor,
        avaliable_action: torch.tensor,
    ):
        s = self.activate_func(self.actor_fc1(s))
        pa = self.activate_func(self.actor_fc_pa(previous_action))
        time = self.activate_func(self.actor_fc_time(time))
        s = torch.cat([s, pa, time], dim=-1)
        # print(self.actor_rnn_hidden.shape)
        output, self.actor_rnn_hidden = self.actor_rnn(s, self.actor_rnn_hidden)
        # for i in range(len(self.actor_rnn_hidden)):
        # print(self.actor_rnn_hidden[i].shape)
        logit = self.actor_fc2(output) + (avaliable_action - 1) * self.max_punish
        return logit

    def critic(
        self,
        s: torch.tensor,
        time: torch.tensor,
        previous_action: torch.tensor,
    ):
        s = self.activate_func(self.critic_fc1(s))
        pa = self.activate_func(self.critic_fc_pa(previous_action))
        time = self.activate_func(self.critic_fc_time(time))
        s = torch.cat([s, pa, time], dim=-1)

        output, self.critic_rnn_hidden = self.critic_rnn(s, self.critic_rnn_hidden)
        value = self.critic_fc2(output)
        return value


if __name__ == "__main__":
    ACTION_DIM = 5
    BATCH_SIZE = 2
    N_STATES = 6
    TIME_INFO_DIM = 2

    state = torch.randn(BATCH_SIZE, N_STATES)
    time_input = torch.randn(BATCH_SIZE, TIME_INFO_DIM)
    previous_action = torch.randn(BATCH_SIZE, 1)
    avaliable_action = torch.tensor([[0, 1, 1, 1, 0], [0, 1, 1, 1, 0]])
    # start = time()
    # model = ensemble_Qnet(N_STATES, ACTION_DIM, 64, TIME_INFO_DIM, 3)
    # # model = Qnet(N_STATES, ACTION_DIM, 64, TIME_INFO_DIM)
    # for i in range(1000):
    #     q_values = model(state, time_input, previous_action, avaliable_action)
    # end = time()
    # print(q_values.shape)
    # print(end - start)
    # model = CQnet(N_STATES, ACTION_DIM, 128, 20, TIME_INFO_DIM)
    # window_length = 20
    # state = torch.randn(BATCH_SIZE, window_length, N_STATES)
    # q_values = model(state, time_input, previous_action, avaliable_action)
    # print(q_values.shape)
    model = Actor_Critic_RNN(N_STATES, 128, ACTION_DIM, TIME_INFO_DIM)
    logit = model.critic(state, time_input, previous_action)
    print(logit.shape)
