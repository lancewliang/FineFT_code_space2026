# the frequency of the high level agent is the same as the low level agent
# based on a sequency of high levelimport pandas as pd
import numpy as np
import pandas as pd
import os
import sys
import random
import argparse
import torch
import sys

sys.path.append(".")

from env.env_initiate.base_initiate import initiate_base_env
from collections import deque
from env.env_class.futures_util import (
    map_action_to_position_leverage,
    map_position_leverage_to_action,
    rule_based_close,
)
from RL.DiHFT.VAE.vae import MLP_VAE, analyze_single_sample

import os
from model.low_level import ensemble_Qnet
from model.high_level import RankBasedQNetwork
from RL.util.update import disable_gradients, get_rank
from analysis.calculate_metric.calculate_metric import calculate_differences

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"
parser = argparse.ArgumentParser()
# * Env setting
parser.add_argument(
    "--base_path",
    type=str,
    default="dataset",
    help="the number of action we have in the training and testing env",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default="BTCUSDT",
    help="training data chunk",
)
parser.add_argument(
    "--max_holding_number",
    type=float,
    default=8,
    help="the transcation cost of not holding the same action as before",
)
parser.add_argument(
    "--position_choices",
    type=int,
    default=9,
    help="the transcation cost of not holding the same action as before",
)
parser.add_argument(
    "--leverage_choices",
    action="append",
    type=int,
    default=[5],
    help="the transaction cost of not holding the same action as before",
)
parser.add_argument(
    "--long_estimated_rate",
    type=float,
    default=0.0005,
    help="the transcation cost of not holding the same action as before",
)
parser.add_argument(
    "--short_estimated_rate",
    type=float,
    default=0,
    help="the transcation cost of not holding the same action as before",
)
parser.add_argument(
    "--transcation_cost",
    type=float,
    default=0.0002,
    help="the transcation cost of not holding the same action as before",
)

parser.add_argument(
    "--early_stop",
    type=int,
    default=0,
    help="the transcation cost of not holding the same action as before",
)
parser.add_argument(
    "--initial_wallet_balance",
    type=float,
    default=1e5,
    help="wallet balance",
)
parser.add_argument(
    "--initial_margin",
    type=float,
    default=0,
    help="initial margin",
)
parser.add_argument(
    "--initial_unrealized_pnL",
    type=float,
    default=0,
    help="unrealized pnL",
)
parser.add_argument(
    "--initial_position",
    type=float,
    default=0,
    help="unrealized pnL",
)
parser.add_argument(
    "--initial_leverage",
    type=float,
    default=5,
    help="initial leverage",
)
# low level network setting
parser.add_argument(
    "--hidden_nodes",
    type=int,
    default=128,
    help="the number of the hidden nodes",
)

parser.add_argument(
    "--time_info_dim",
    type=int,
    default=2,
    help="context number",
)
parser.add_argument(
    "--low_level_path",
    type=str,
    default="result/DiHFT/potential_model",
    help="context number",
)

# VAE network path
parser.add_argument(
    "--vae_path",
    type=str,
    default="result/DiHFT/vae_results",
    help="the path for storing the test result",
)
parser.add_argument(
    "--label_number",
    type=int,
    default=5,
    help="the path for storing the test result",
)
# vae related
parser.add_argument(
    "--z_dim",
    type=int,
    default=512,
    help="the sequency length",
)
parser.add_argument(
    "--vae_hidden_dims",
    type=list,
    default=[4096, 2048, 1024, 1024],
    help="the sequency length",
)
parser.add_argument(
    "--loss_type",
    type=str,
    default="NLL",
    help="the sequency length",
)
parser.add_argument(
    "--vae_results",
    type=str,
    default="result/DiHFT/vae_results",
    help="the sequency length",
)
parser.add_argument(
    "--reject_quantile",
    type=float,
    default=0.01,
    help="the sequency length",
)
parser.add_argument(
    "--risk_reject_rate_theshold",
    type=float,
    default=0.2,
    help="the sequency length",
)
# high level network setting
parser.add_argument(
    # surpass the seq length must use repick agent
    "--seq_length",
    type=int,
    default=244,
    help="context number",
)
parser.add_argument(
    "--result_path",
    type=str,
    default="result/DiHFT/high_level",
    help="the path for storing the test result",
)
# rule based setting
parser.add_argument(
    "--close_length",
    type=int,
    default=6,
    help="context number",
)
# loss sequence setting
parser.add_argument(
    "--loss_length",
    type=int,
    default=72,
    help="context number",
)


def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class high_level_trader:
    def __init__(self, args) -> None:
        # device
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.reject_quantile = args.reject_quantile
        self.risk_reject_rate_theshold = args.risk_reject_rate_theshold
        self.model_path = os.path.join(
            args.result_path,
            args.dataset_name,
            "heuristic_1",
        )

        self.test_path = os.path.join(
            self.model_path,
            "test_reject_quantile_{}_reject_rate_{}".format(
                self.reject_quantile, self.risk_reject_rate_theshold
            ),
        )
        if not os.path.exists(self.test_path):
            os.makedirs(self.test_path)
        # trading environment setting
        self.base_path = args.base_path
        self.dataset_name = args.dataset_name
        self.test_data_path = os.path.join(
            self.base_path, self.dataset_name, "test.feather"
        )
        self.tech_indicator_list = np.load(
            os.path.join(self.base_path, self.dataset_name, "state_features.npy")
        )
        self.maintenance_margin_ratio_dict = np.load(
            os.path.join(
                self.base_path, self.dataset_name, "maintenance_margin_ratio_dict.npy"
            ),
            allow_pickle=True,
        ).item()
        self.max_holding_number = args.max_holding_number
        self.position_choices = args.position_choices
        self.single_side_action_num = int((self.position_choices - 1) / 2)
        self.position_list = (
            [
                self.max_holding_number / self.single_side_action_num * i
                for i in range(1, self.single_side_action_num + 1)
            ]
            + [0]
            + [
                self.max_holding_number / self.single_side_action_num * -i
                for i in range(1, self.single_side_action_num + 1)
            ]
        )
        self.position_list.sort()
        self.leverage_choices = args.leverage_choices
        self.long_estimated_rate = args.long_estimated_rate
        self.short_estimated_rate = args.short_estimated_rate
        self.transcation_cost = args.transcation_cost
        self.early_stop = args.early_stop
        self.initial_wallet_balance = args.initial_wallet_balance
        self.initial_margin = args.initial_margin
        self.initial_unrealized_pnL = args.initial_unrealized_pnL
        self.initial_position = args.initial_position
        self.initial_leverage = args.initial_leverage
        self.initial_state = (
            self.initial_wallet_balance,
            self.initial_margin,
            self.initial_unrealized_pnL,
            self.initial_position,
            self.initial_leverage,
        )
        self.initial_action = map_position_leverage_to_action(
            self.initial_position,
            self.initial_leverage,
            self.leverage_choices,
            self.position_list,
        )
        self.zero_position_action = len(self.leverage_choices) * (
            len(self.position_list) // 2
        )

        # low-level network
        self.time_info_dim = args.time_info_dim
        self.hidden_nodes = args.hidden_nodes
        self.N = args.label_number
        self.N_ACTIONS = (self.position_choices - 1) * len(self.leverage_choices) + 1
        self.low_level_network = ensemble_Qnet(
            N_STATES=len(self.tech_indicator_list),
            N_ACTIONS=self.N_ACTIONS,
            hidden_nodes=self.hidden_nodes,
            TIME_INFO_DIM=self.time_info_dim,
            ensemble_number=self.N,
        ).to(self.device)
        self.low_level_path = args.low_level_path
        self.low_level_network.load_state_dict(
            torch.load(
                os.path.join(
                    self.low_level_path,
                    self.dataset_name,
                    "model.pth",
                )
            )
        )
        self.low_level_network.to(self.device)
        self.low_level_network.eval()
        disable_gradients(self.low_level_network)
        # loss deque
        # rule based result
        self.close_length = args.close_length
        self.loss_length = args.loss_length
        self.pick_one_agent_length = 0
        self.loss_seq = deque(maxlen=self.loss_length)
        self.seq_length = args.seq_length
        # label vae
        # VAE network path
        self.label_number = args.label_number
        self.vae_path = os.path.join(args.vae_path, self.dataset_name)
        label_list = ["label_{}".format(i) for i in range(args.label_number)]
        self.vae_model_path_list = [
            os.path.join(self.vae_path, label, "model_latest.pth")
            for label in label_list
        ]
        self.inlogp_path_list = [
            os.path.join(self.vae_path, label, "id_logpx.npy") for label in label_list
        ]
        self.vae_model_list = []
        self.in_ds_logpx_list = []
        for path, id_path in zip(self.vae_model_path_list, self.inlogp_path_list):
            vae_model = MLP_VAE(
                INPUT_DIM=len(self.tech_indicator_list),
                Z_DIM=args.z_dim,
                hidden_dims=args.vae_hidden_dims,
                loss_func=args.loss_type,
            ).to(self.device)
            vae_model.load_state_dict(
                torch.load(path, map_location=torch.device(self.device))
            )
            self.vae_model_list.append(vae_model)
            self.in_ds_logpx_list.append(np.load(id_path))
        

    # huerstic的办法
    # rule 平仓条件 亏钱 考虑换agent
    # rule 换agent条件 VAE的rank发生变化

    # start: VAE rank to pick agent
    # use the agent to do rollout
    # if the agent's draw down is over 5% close the current position and wait for 10 minute to repick
    # if the agent has been used for over 288 steps not necessarily to change the agent, but we might have to repick an agent
    # all the picking relie on the vae result

    def add_loss(self, s):
        self.chosen_vae = self.vae_model_list[self.selected_agent_index]
        mu,loss = analyze_single_sample(self.chosen_vae, s, self.device)
        self.loss_seq.append(loss)

    def test(self):
        self.df = pd.read_feather(self.test_data_path)
        env = initiate_base_env(
            df=self.df,
            feature_list=self.tech_indicator_list,
            max_holding_number=self.max_holding_number,
            position_choices=self.position_choices,  # (must be an odd number, the minum of trading equals to (max_holder_number)/((action_dim-1)/2)s))
            leverage_choice=self.leverage_choices,  # recommend only use one leverage choice, because the leverage does not influence the return directly, the position
            # itself is enough to show the risk preference
            long_estimated_rate=self.long_estimated_rate,
            short_estimated_rate=self.short_estimated_rate,
            commission_rate=self.transcation_cost,
            # maten_mar_ratio_dict varies among different perpertual contracts, need to perform a config file for different perpertual
            # the default is for btcusdt perpetual contract
            maintenance_margin_ratio_dict=self.maintenance_margin_ratio_dict,
            early_stop=self.early_stop,
            # initial_personal_state
            initial_state=self.initial_state,
        )
        s, info = env.reset()
        self.pick_one_agent_length = 0
        episode_reward_sum = 0
        self.heurstic_pick_agent(s)
        self.add_loss(s)
        # 每新整一个state，都算一下当前的选择的vae的loss
        while True:
            done, action = self.act(env, s, info)
            if done:
                break
            s_, r, done, info_ = env.step(action)
            if done:
                break
            self.add_loss(s_)
            s, info = s_, info_
        total_asset_history = env.margine_balance_history
        reward_history = calculate_differences(total_asset_history)
        micro_action_history = env.micro_action_history
        micro_action_history = env.micro_action_history
        trading_info = {
            "return rate": total_asset_history[-1] / self.initial_wallet_balance
        }

        np.save(os.path.join(self.test_path, "reward_history.npy"), reward_history)
        np.save(
            os.path.join(self.test_path, "total_asset_history.npy"), total_asset_history
        )
        np.save(
            os.path.join(self.test_path, "micro_action_history.npy"),
            micro_action_history,
        )
        np.save(os.path.join(self.test_path, "trading_info.npy"), trading_info)
        np.save(
            os.path.join(self.test_path, "initial_margin_history.npy"),
            env.initial_margin_history,
        )
        np.save(
            os.path.join(self.test_path, "wallet_balance_history.npy"),
            env.wallet_balance_history,
        )
        np.save(
            os.path.join(self.test_path, "unrealized_pnl_history.npy"),
            env.unrealized_pnl_history,
        )
        np.save(
            os.path.join(self.test_path, "maintain_marigine_history.npy"),
            env.maintain_marigine_history,
        )
        np.save(
            os.path.join(self.test_path, "new_position_required_money_history.npy"),
            env.new_position_required_money_history,
        )

    def heurstic_pick_agent(self, s):
        loss_list = []
        for vae in self.vae_model_list:
            z_mu,loss = analyze_single_sample(vae, s, self.device)
            loss_list.append(loss)
        # print('loss_list',loss_list)
        # print('min_loss_list',max(loss_list))
        id_min = loss_list.index(max(loss_list))
        self.selected_agent_index = id_min

    def agent_act(self, state, info):
        # low level agent
        state = torch.unsqueeze(torch.FloatTensor(state).reshape(-1), 0).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor([info["previous_action"]]).float().to(self.device), 0
        ).to(self.device)
        avaliable_action = torch.unsqueeze(
            torch.tensor(info["avaliable_action"]).to(self.device), 0
        ).to(self.device)
        hour_count_down = (
            torch.unsqueeze(torch.tensor([info["funding_count_down_hour"]]), 0)
            .to(self.device)
            .float()
        )
        minute_count_down = (
            torch.unsqueeze(torch.tensor([info["funding_count_down_minute"]]), 0)
            .to(self.device)
            .float()
        )
        time_input = torch.cat([hour_count_down, minute_count_down], dim=1).to(
            self.device
        )
        actions_value = self.low_level_network(
            state=state,
            time=time_input,
            previous_action=previous_action,
            avaliable_action=avaliable_action,
        )
        action_value_chosen_index = actions_value[:, self.selected_agent_index, :]
        action = torch.max(action_value_chosen_index, 1)[1].data.cpu().numpy()
        action = action[0]

        return action

    def get_reject(self):
        self.id_logpx = self.in_ds_logpx_list[self.selected_agent_index]
        reject_theshold = np.percentile(self.id_logpx, self.reject_quantile)
        reject_rate = np.sum(np.array(self.loss_seq) < reject_theshold) / len(
            self.loss_seq
        )

        return reject_rate > self.risk_reject_rate_theshold

    def act(self, env, state, info):
        # 回撤出问题 直接强制平仓+重新选
        done = False
        if info["single_holding_max_drawdown"] > 0.05:
            # print("force close position")
            # multi_step close position, be careful about the memory and the output
            trading_action, s, info, done = self.act_close_position(env, state, info)
            self.heurstic_pick_agent(s)
            self.pick_one_agent_length = 0
        # 超过时长 重新选
        elif self.pick_one_agent_length >= self.seq_length:
            # print("over long time, repick agent")
            self.pick_one_agent_length = 1
            self.heurstic_pick_agent(state)
            trading_action = self.agent_act(state, info)
        # reject rate不达标 重新选
        elif self.get_reject():
            # print("reject rate too high, repick agent")
            self.heurstic_pick_agent(state)
            trading_action = self.agent_act(state, info)
            self.pick_one_agent_length = 1
        else:
            # print("previous agent choice action")
            trading_action = self.agent_act(state, info)
            self.pick_one_agent_length += 1
        return done, trading_action

    def act_close_position(self, env, state, info):
        for i in range(self.close_length):
            trading_action = rule_based_close(
                info,
                self.zero_position_action,
                self.leverage_choices,
                self.position_list,
            )
            s_, r, done, info_ = env.step(trading_action)
            self.add_loss(s_)
            s, info = s_, info_
            if done:
                break
        return trading_action, s, info, done


if __name__ == "__main__":
    args = parser.parse_args()
    seed_torch(1024)
    high_level_trader(args).test()
