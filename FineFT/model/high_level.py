import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import sys

sys.path.append(".")
from model.low_level import modulate
import torch.nn.functional as F


# previous trail for routing in FineFT (not used in the final version)
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class RankBasedQNetwork(nn.Module):
    def __init__(self, input_dim, num_channels, output_dim, kernel_size=2, dropout=0.2):
        super(RankBasedQNetwork, self).__init__()
        self.tcn = TemporalConvNet(input_dim, num_channels, kernel_size, dropout)
        self.fc = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_length, rank_number)
        x = x.permute(0, 2, 1)  # (batch_size, rank_number, seq_length)
        y = self.tcn(x)  # (batch_size, num_channels[-1], seq_length)
        y = y[:, :, -1]  # 取最后一个时间步的输出 (batch_size, num_channels[-1])
        q_values = self.fc(y)  # (batch_size, output_dim)
        return q_values


# high level agent for MacroHFT


class hyperagent(nn.Module):
    def __init__(self, state_dim_1, state_dim_2, action_dim, hidden_dim):
        super(hyperagent, self).__init__()
        self.fc1 = nn.Linear(state_dim_1 + state_dim_2, hidden_dim)
        self.fc2 = nn.Linear(2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim * 2, elementwise_affine=False, eps=1e-6)
        self.embedding = nn.Embedding(action_dim, hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 4 * hidden_dim, bias=True)
        )
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * 4, 6),
            nn.Softmax(dim=1),
        )
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    def forward(
        self,
        single_state: torch.tensor,
        trend_state: torch.tensor,
        class_state: torch.tensor,
        previous_action: torch.tensor,
    ):
        action_hidden = self.embedding(previous_action)
        state_hidden = self.fc1(torch.cat([single_state, trend_state], dim=1))
        x = torch.cat([action_hidden, state_hidden], dim=1)
        c = self.fc2(class_state)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        weight = self.net(x)

        return weight

    def encode(
        self,
        single_state: torch.tensor,
        trend_state: torch.tensor,
        previous_action: torch.tensor,
    ):
        action_hidden = self.embedding(previous_action)
        state_hidden = self.fc1(torch.cat([single_state, trend_state], dim=1))
        x = torch.cat([action_hidden, state_hidden], dim=1)
        return x


def calculate_q(w, qs):
    q_tensor = torch.stack(qs)
    q_tensor = q_tensor.permute(1, 0, 2)
    weights_reshaped = w.view(-1, 1, 6)
    combined_q = torch.bmm(weights_reshaped, q_tensor).squeeze(1)

    return combined_q


# high level for earnhft
class Qnet_high_level_position(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS, hidden_nodes):
        super(Qnet_high_level_position, self).__init__()
        self.fc1 = nn.Linear(N_STATES, hidden_nodes)
        self.fc3 = nn.Linear(1, N_ACTIONS)
        self.fc2 = nn.Linear(N_ACTIONS + hidden_nodes, hidden_nodes)
        self.out = nn.Linear(hidden_nodes, N_ACTIONS)

    def forward(
        self,
        state: torch.tensor,
        previous_action: torch.tensor,
    ):
        state_hidden = F.relu(self.fc1(state))
        previous_action_hidden = F.relu(self.fc3(previous_action))
        information_hidden = torch.cat([state_hidden, previous_action_hidden], dim=1)
        information_hidden = self.fc2(information_hidden)
        action = self.out(information_hidden)
        masked_action = action
        return masked_action


if __name__ == "__main__":
    device = "cuda"
    # 参数示例
    rank_number = 7
    batch_size = 32
    input_dim = rank_number
    num_channels = [1, 2]  # TCN每层的通道数
    output_dim = 7  # 输出维度，可根据实际动作空间调整
    seq_length = 20  # 序列长度，根据实际情况调整

    model = RankBasedQNetwork(input_dim, num_channels, output_dim).to(device)
    x = torch.randn(batch_size, seq_length, rank_number).to(device)
    q_values = model(x)
    print(q_values.shape)
    print(q_values.requires_grad)
