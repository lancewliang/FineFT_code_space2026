from __future__ import annotations

# Code reference: https://github.com/Lizhi-sjtu/DRL-code-pytorch/tree/main/3.Rainbow_DQN
# Evaluate every sub-agent (each qnet in the ensemble) on all valid dataset files.
# For each sub-model x dataset file x initial action, run one full episode and
# report the final balance (= initial margin balance + cumulative reward),
# the cumulative reward sum, and the final return rate.

import argparse
import logging
import os
import re
import sys
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp

sys.path.append(".")

# model
from model.low_level import ensemble_Qnet

# env
from env.env_initiate.base_initiate import initiate_base_env
from env.env_class.futures_util import map_action_to_position_leverage


os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

logger = logging.getLogger(__name__)
log = logger
if not logger.handlers and not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def configure_logger(logg_file_path: str):
    if not logg_file_path:
        return None
    log_dir = os.path.dirname(logg_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    abs_log_path = os.path.abspath(logg_file_path)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == abs_log_path:
            return abs_log_path

    file_handler = logging.FileHandler(abs_log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    return abs_log_path

   

class weighted_trader:
    def __init__(self, 
        logg_file_path: str,
        data_file_path:str,
        model_path: str,
        tech_indicator_list_path: str,
        maintenance_margin_ratio_dict_path: str,
        transcation_cost: float,
        max_holding_number: int,          
        position_choices: int,   
        N: int,        
        time_info_dim: int = 2,
        hidden_nodes: int=128,
        leverage_choices: List[float]=[1],
        initial_leverage: int = 1,
        initial_position: int = 0,        
        initial_wallet_balance: float = 10000,
        order_book_depth: int=5,
        early_stop:int=2,
        enable_limit_reward: bool=False,
        limit_hold_bonus: float =1.0,
        limit_stay_bonus: float =0.5,
        limit_reverse_penalty: float =1.5,
        near_limit_threshold: float =0.05,
        allow_reverse_position: bool = True,
        
                 ):

        # logger
        self.logg_file_path = logg_file_path
        self.abs_log_path = configure_logger(logg_file_path)
        self.logger = logger
        self.log = logger

        # device
        if torch.cuda.is_available():
            self.device = "cuda"
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            self.device = "cpu"

        # model file to evaluate
        self.model_file = model_path
        self.data_file_path = data_file_path
        self.tech_indicator_list = np.load(
            tech_indicator_list_path
        )
        self.maintenance_margin_ratio_dict = np.load(maintenance_margin_ratio_dict_path, allow_pickle=True).item()
        self.max_holding_number = max_holding_number
        self.order_book_depth = order_book_depth
        self.position_choices = position_choices
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
        self.leverage_choices = leverage_choices
        self.long_estimated_rate = 0
        self.short_estimated_rate = 0
        self.transcation_cost = transcation_cost
        self.allow_reverse_position = allow_reverse_position
        self.enable_limit_reward = enable_limit_reward
        self.limit_hold_bonus = limit_hold_bonus
        self.limit_stay_bonus = limit_stay_bonus
        self.limit_reverse_penalty = limit_reverse_penalty
        self.near_limit_threshold = near_limit_threshold
        self.early_stop = early_stop
        self.initial_wallet_balance = initial_wallet_balance
        self.initial_margin = 0
        self.initial_unrealized_pnL = 0
        self.initial_position = initial_position
        self.initial_leverage = initial_leverage
        # margin balance before any action, cumulative reward accumulates from it
        self.initial_margin_balance = (
            self.initial_wallet_balance + self.initial_unrealized_pnL
        )

        # network: the ensemble size is inferred from the checkpoint so any
        # trained_model.pkl can be evaluated directly
        self.time_info_dim = time_info_dim
        self.hidden_nodes = hidden_nodes        
        self.N_ACTIONS = (self.position_choices - 1) * len(self.leverage_choices) + 1
        state_dict = torch.load(self.model_file, map_location=self.device)
        self.N = N
        self.eval_net = ensemble_Qnet(
            N_STATES=len(self.tech_indicator_list),
            N_ACTIONS=self.N_ACTIONS,
            hidden_nodes=self.hidden_nodes,
            TIME_INFO_DIM=self.time_info_dim,
            ensemble_number=self.N,
        ).to(self.device)
        self.eval_net.load_state_dict(state_dict)
        self.eval_net.eval()
        self.initial_action_list = range(self.N_ACTIONS)

         

    @staticmethod
    def configure_logger(logg_file_path: str):
        return configure_logger(logg_file_path)

    def act_test(self, state: np.ndarray | list[float], info: dict[str, Any], context_index: int) -> int:
        assert context_index in range(self.N)
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
        trading_info = torch.unsqueeze(
            torch.tensor(info["trading_info"]).float().to(self.device), 0
        )
        with torch.inference_mode():
            action_value_chosen_index = self.eval_net.qnet_list[context_index](
                state=state,
                time=time_input,
                previous_action=previous_action,
                avaliable_action=avaliable_action,
                trading_info=trading_info,
            )
            action = torch.max(action_value_chosen_index, 1)[1].data.cpu().numpy()
        action = action[0]

        return action

    def _run_episode(self, test_df: pd.DataFrame, initial_action: int, bin_index: int):
        """Run one full episode with the given sub-model and initial action.

        Returns the cumulative reward, the final balance, the final return
        rate and the maximum drawdown of the episode.
        """
        initial_position, initial_leverage = map_action_to_position_leverage(
            initial_action,
            self.leverage_choices,
            self.position_list,
        )
        current_markprice = test_df["mark_price"].values[0]
        initial_margin = np.abs(
            initial_position * current_markprice / initial_leverage
        )
        initial_state = (
            self.initial_wallet_balance,
            initial_margin,
            self.initial_unrealized_pnL,
            initial_position,
            initial_leverage,
        )
        enable_limit = (
            self.enable_limit_reward
            and "UpperLimitPrice" in test_df.columns
            and "limit_up_single_sided_ratio" in test_df.columns
        )
        test_env = initiate_base_env(
            df=test_df,
            feature_list=self.tech_indicator_list,
            max_holding_number=self.max_holding_number,
            order_book_depth=self.order_book_depth,
            position_choices=self.position_choices,
            leverage_choice=self.leverage_choices,
            long_estimated_rate=self.long_estimated_rate,
            short_estimated_rate=self.short_estimated_rate,
            commission_rate=self.transcation_cost,
            maintenance_margin_ratio_dict=self.maintenance_margin_ratio_dict,
            early_stop=0,
            # initial_personal_state
            initial_state=initial_state,
            allow_reverse_position=self.allow_reverse_position,
            enable_limit_reward=enable_limit,
            limit_hold_bonus=self.limit_hold_bonus,
            limit_stay_bonus=self.limit_stay_bonus,
            limit_reverse_penalty=self.limit_reverse_penalty,
            near_limit_threshold=self.near_limit_threshold,
        )
        s, info = test_env.reset()
        done = False
        reward_sum = 0
        while not done:
            a = self.act_test(s, info, bin_index)
            s_, r, done, info = test_env.step(a)
            s = s_
            reward_sum += r
        final_balance = (
            test_env.wallet_balance_history[-1]
            + test_env.unrealized_pnl_history[-1]
        )
        final_return_rate = (
            final_balance - self.initial_margin_balance
        ) / self.initial_margin_balance
        balance_history = (
            np.array(test_env.wallet_balance_history)
            + np.array(test_env.unrealized_pnl_history)
        )
        peak_history = np.maximum.accumulate(balance_history)
        max_drawdown = float(np.max((peak_history - balance_history) / peak_history))
        return {
            "initial_position": initial_position,
            "initial_leverage": initial_leverage,
            "reward_sum": reward_sum,
            "final_balance": final_balance,
            "final_return_rate": final_return_rate,
            "max_drawdown": max_drawdown,
        }
    
    def test(self):
        logger.info("start")
        self.eval_net.eval()
        overall_result = [] 
        test_df = pd.read_feather(self.data_file_path)
        for initial_action in self.initial_action_list:
            for bin_index in range(self.N):
                single_result = {
                    "sub_model_index": bin_index,
                    "df_path":self.data_file_path,
                    "initial_action": initial_action,
                    "df_length": len(test_df),
                    **self._run_episode(test_df, initial_action, bin_index),
                }
                logger.info(single_result)
                overall_result.append(single_result)
       
        return overall_result
    
def _evaluate_single_file_worker(conn, trader_kwargs: dict):
    try:
        trader = weighted_trader(**trader_kwargs)
        single_file_results = trader.test()
        if conn is not None:
            conn.send((True, single_file_results))
    except Exception as e:
        logger.error(
            "Error evaluating data file %s: %s",
            trader_kwargs.get("data_file_path"),
            traceback.format_exc(),
        )
        if conn is not None:
            conn.send((False, traceback.format_exc()))
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def evaluates( 
        logg_file_path: str,
        data_file_paths:List[str],
        model_path: str,
        tech_indicator_list_path: str,
        maintenance_margin_ratio_dict_path: str,
        transcation_cost: float,
        max_holding_number: int,          
        position_choices: int,   
        N: int,        
        time_info_dim: int = 2,
        hidden_nodes: int=128,
        leverage_choices: List[float]=[1],
        initial_leverage: int = 1,
        initial_position: int = 0,        
        initial_wallet_balance: float = 10000,
        order_book_depth: int=5,
        early_stop:int=2,
        enable_limit_reward: bool=False,
        limit_hold_bonus: float =1.0,
        limit_stay_bonus: float =0.5,
        limit_reverse_penalty: float =1.5,
        near_limit_threshold: float =0.05,
        allow_reverse_position: bool = True,
        **kwargs):
    """Evaluate the sub-agent across multiple data files using one subprocess per file."""
    if logg_file_path:
        configure_logger(logg_file_path)

    if not data_file_paths:
        logger.info("No data files provided for evaluation.")
        return []

    logger.info(
        "Starting multi-process evaluation on %d data files",
        len(data_file_paths),
    )

    base_trader_kwargs = {
        "logg_file_path": logg_file_path,
        "model_path": model_path,
        "tech_indicator_list_path": tech_indicator_list_path,
        "maintenance_margin_ratio_dict_path": maintenance_margin_ratio_dict_path,
        "transcation_cost": transcation_cost,
        "max_holding_number": max_holding_number,
        "position_choices": position_choices,
        "N": N,
        "time_info_dim": time_info_dim,
        "hidden_nodes": hidden_nodes,
        "leverage_choices": leverage_choices,
        "initial_leverage": initial_leverage,
        "initial_position": initial_position,
        "initial_wallet_balance": initial_wallet_balance,
        "order_book_depth": order_book_depth,
        "early_stop": early_stop,
        "enable_limit_reward": enable_limit_reward,
        "limit_hold_bonus": limit_hold_bonus,
        "limit_stay_bonus": limit_stay_bonus,
        "limit_reverse_penalty": limit_reverse_penalty,
        "near_limit_threshold": near_limit_threshold,
        "allow_reverse_position": allow_reverse_position,
        **kwargs,
    }

    ctx = mp.get_context("spawn")
    process_list = []

    for df_path in data_file_paths:
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        trader_kwargs = dict(base_trader_kwargs)
        trader_kwargs["data_file_path"] = df_path

        p = ctx.Process(
            target=_evaluate_single_file_worker,
            args=(child_conn, trader_kwargs),
        )
        p.start()
        child_conn.close()
        process_list.append((p, parent_conn, df_path))

    overall_results = []
    # Collect results from all subprocesses
    for p, parent_conn, df_path in process_list:
        try:
            success, res = parent_conn.recv()
            if success and res:
                overall_results.extend(res)
            elif not success:
                logger.error(
                    "Subprocess evaluation failed for file %s: %s",
                    df_path,
                    res,
                )
        except EOFError:
            logger.error(
                "Subprocess for file %s terminated unexpectedly without returning results",
                df_path,
            )
        finally:
            parent_conn.close()

    # Wait for all subprocesses to complete
    for p, _, df_path in process_list:
        p.join()

    # Per sub-agent averages across all data files and initial actions
    sub_agent_metrics = {}
    for result in overall_results:
        metrics = sub_agent_metrics.setdefault(
            result["sub_model_index"],
            {"final_balance": [], "final_return_rate": [], "max_drawdown": []},
        )
        metrics["final_balance"].append(result["final_balance"])
        metrics["final_return_rate"].append(result["final_return_rate"])
        metrics["max_drawdown"].append(result["max_drawdown"])

    for sub_model_index in sorted(sub_agent_metrics):
        metrics = sub_agent_metrics[sub_model_index]
        logger.info(
            "Sub-agent %d: avg final balance = %.4f, avg return rate = %.2f%%, avg max drawdown = %.2f%%",
            sub_model_index,
            np.mean(metrics["final_balance"]),
            np.mean(metrics["final_return_rate"]) * 100,
            np.mean(metrics["max_drawdown"]) * 100,
        )

    logger.info(
        "All %d evaluation subprocesses finished.",
        len(process_list),
    )
    return overall_results
