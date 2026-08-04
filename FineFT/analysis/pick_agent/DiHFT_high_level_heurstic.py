import pandas as pd
import argparse
import os
import torch
import shutil
import matplotlib.pyplot as plt
import numpy as np
import sys

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.append(".")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from analysis.calculate_metric.calculate_metric import (
    calculate_metric,
    calculate_required_money,
)

parser = argparse.ArgumentParser()
# replay buffer coffient
parser.add_argument(
    "--experiment_name",
    type=str,
    default="default",
    help="experiment name",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default="BNBUSDT",
    # BNB or DOT
    help="the number of transcation we store in one memory",
)

parser.add_argument(
    "--base_path",
    type=str,
    default="dataset/30min",
    help="base path of dataset",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="analysis_result/DiHFT/high_level_heurstic",
    help="the number of initial_position",
)
parser.add_argument(
    "--early_stop",
    type=int,
    default=0,
    help="the number of initial_position",
)


class Picker:
    def __init__(self, args) -> None:
        self.base_path = getattr(args, "base_path", "dataset/30min")
        self.dataset_name = args.dataset_name
        self.experiment_name = getattr(args, "experiment_name", "default")
        self.save_path = os.path.join(args.save_path, args.dataset_name, self.experiment_name)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

        self.early_stop = args.early_stop

    def _get_contract_dirs(self, epoch_path):
        contracts_dir = os.path.join(epoch_path, "contracts")
        if os.path.exists(contracts_dir) and os.path.isdir(contracts_dir):
            subdirs = [
                os.path.join(contracts_dir, d)
                for d in sorted(os.listdir(contracts_dir))
                if os.path.isdir(os.path.join(contracts_dir, d))
            ]
            if subdirs:
                return subdirs
        return [epoch_path]

    def analysis_single_epoch(self, epoch_path):
        contract_dirs = self._get_contract_dirs(epoch_path)
        
        per_contract_metrics = []
        total_rewards = []
        total_req_money = []

        for data_dir in contract_dirs:
            initial_margin_history = np.load(
                os.path.join(data_dir, "initial_margin_history.npy")
            )
            maintain_marigine_history = np.load(
                os.path.join(data_dir, "maintain_marigine_history.npy")
            )
            new_position_required_money_history = np.load(
                os.path.join(data_dir, "new_position_required_money_history.npy")
            )
            micro_action_history = np.load(
                os.path.join(data_dir, "micro_action_history.npy")
            )
            reward_history = np.load(os.path.join(data_dir, "reward_history.npy"))
            total_asset_history = np.load(
                os.path.join(data_dir, "total_asset_history.npy")
            )
            unrealized_pnl_history = np.load(
                os.path.join(data_dir, "unrealized_pnl_history.npy")
            )
            wallet_balance_history = np.load(
                os.path.join(data_dir, "wallet_balance_history.npy")
            )
            requred_money = calculate_required_money(
                initial_margin_history,
                maintain_marigine_history,
                new_position_required_money_history,
                unrealized_pnl_history,
                wallet_balance_history,
            )
            steps = len(reward_history)
            freq_calc = 12 if steps >= 24 else max(1, steps // 2)
            try:
                tr, daily_vol, mdd, downside_deviation_daily, annual_sr, daily_cr, daily_SoR = (
                    calculate_metric(requred_money, reward_history, freq=freq_calc)
                )
            except Exception:
                tr, daily_vol, mdd, downside_deviation_daily, annual_sr, daily_cr, daily_SoR = (
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                )
            
            c_name = os.path.basename(data_dir)
            per_contract_metrics.append({
                "contract": c_name,
                "tr": tr,
                "daily_vol": daily_vol,
                "mdd": mdd,
                "downside_deviation_daily": downside_deviation_daily,
                "annual_sr": annual_sr,
                "daily_cr": daily_cr,
                "daily_SoR": daily_SoR,
                "required_money": requred_money,
                "reward_sum": float(np.sum(reward_history)),
            })
            total_rewards.append(float(np.sum(reward_history)))
            total_req_money.append(float(requred_money))

        df_contracts = pd.DataFrame(per_contract_metrics)
        portfolio_return = np.sum(total_rewards) / (np.sum(total_req_money) + 1e-12)

        result_dict = {}
        result_dict["path"] = epoch_path
        result_dict["num_contracts"] = len(per_contract_metrics)
        result_dict["tr"] = float(df_contracts["tr"].mean())
        result_dict["portfolio_tr"] = float(portfolio_return)
        result_dict["daily_vol"] = float(df_contracts["daily_vol"].mean())
        result_dict["mdd"] = float(df_contracts["mdd"].mean())
        result_dict["downside_deviation_daily"] = float(df_contracts["downside_deviation_daily"].mean())
        result_dict["annual_sr"] = float(df_contracts["annual_sr"].mean())
        result_dict["daily_cr"] = float(df_contracts["daily_cr"].mean())
        result_dict["daily_SoR"] = float(df_contracts["daily_SoR"].mean())
        result_dict["required_money"] = float(np.sum(total_req_money))
        return result_dict

    def analysis_all_epoch(self):
        result_list = []
        model_root_path = os.path.join(
            "result/DiHFT/high_level", self.dataset_name, self.experiment_name, "vae_risk_aware_routing"
        )        
        parameter_list = os.listdir(model_root_path)
        for parameter in parameter_list:
            epoch_path = os.path.join(model_root_path, parameter)
            if not os.path.isdir(epoch_path):
                continue
            if not os.listdir(epoch_path):
                continue
            result = self.analysis_single_epoch(epoch_path)
            result_list.append(result)
        result_df = pd.DataFrame(result_list)
        self.result_df = result_df
        result_df.to_csv(os.path.join(self.save_path, "result.csv"), index=False)

    def analysis_best_epoch(self):
        best_results = []
        df_clean = self.result_df.dropna(subset=["tr"])
        if df_clean.empty:
            df_clean = self.result_df.fillna(0.0)
        for indicator in ["tr", "annual_sr", "daily_cr", "daily_SoR"]:
            idx_series = df_clean[indicator].dropna()
            best_idx = idx_series.idxmax() if not idx_series.empty else df_clean.index[0]
            max_row = df_clean.loc[[best_idx]].copy()
            max_row["indicator"] = indicator
            best_results.append(max_row)
        for indicator in ["daily_vol", "mdd", "downside_deviation_daily"]:
            idx_series = df_clean[indicator].dropna()
            best_idx = idx_series.idxmin() if not idx_series.empty else df_clean.index[0]
            min_row = df_clean.loc[[best_idx]].copy()
            min_row["indicator"] = indicator
            best_results.append(min_row)
        best_results_df = pd.concat(best_results)
        self.best_result_df = best_results_df
        best_results_df.to_csv(os.path.join(self.save_path, "best_result.csv"), index=False)

    def create_best_agent(self):
        path = self.best_result_df.iloc[0]["path"]
        para = os.path.basename(path)
        
        high_level_path = os.path.join("result/DiHFT/final_result", self.dataset_name, self.experiment_name)
        
        if not os.path.exists(high_level_path):
            os.makedirs(high_level_path, exist_ok=True)

        with open(
            os.path.join(high_level_path, "high_level_agent_para.txt"), "w"
        ) as file:
            file.write("%s\n" % para)

        contract_dirs = self._get_contract_dirs(path)
        has_multi_contracts = len(contract_dirs) > 1 or (
            len(contract_dirs) == 1 and os.path.basename(contract_dirs[0]) != os.path.basename(path)
        )

        if has_multi_contracts:
            target_contracts_dir = os.path.join(high_level_path, "contracts")
            os.makedirs(target_contracts_dir, exist_ok=True)
            for c_dir in contract_dirs:
                c_name = os.path.basename(c_dir)
                target_dir = os.path.join(target_contracts_dir, c_name)
                os.makedirs(target_dir, exist_ok=True)
                for f_name in os.listdir(c_dir):
                    if f_name.endswith(".npy") or f_name.endswith(".csv"):
                        shutil.copy2(os.path.join(c_dir, f_name), os.path.join(target_dir, f_name))

            first_dir = contract_dirs[0]
            for f_name in os.listdir(first_dir):
                if f_name.endswith(".npy") or f_name.endswith(".csv"):
                    shutil.copy2(os.path.join(first_dir, f_name), os.path.join(high_level_path, f_name))
        else:
            first_dir = contract_dirs[0]
            for f_name in os.listdir(first_dir):
                if f_name.endswith(".npy") or f_name.endswith(".csv"):
                    shutil.copy2(os.path.join(first_dir, f_name), os.path.join(high_level_path, f_name))

    def _find_valid_contract_files(self):
        candidates = [
            os.path.join(self.base_path, self.dataset_name, "valid"),
        ]
        for cdir in candidates:
            if os.path.exists(cdir) and os.path.isdir(cdir):
                feathers = [
                    (os.path.splitext(f)[0], os.path.join(cdir, f))
                    for f in sorted(os.listdir(cdir))
                    if f.endswith(".feather")
                ]
                if feathers:
                    return feathers
        return []

    def plot(self):
        high_level_path = os.path.join("result/DiHFT/final_result", self.dataset_name, self.experiment_name)
        
        contract_files = self._find_valid_contract_files()
        if not contract_files:
            print("No validation dataset feather files found for plotting.")
            return

        color_list = ["#8ECFC9", "#FFBE7A", "#FA7F6F"]
        all_contract_data = []

        for c_name, feather_path in contract_files:
            c_result_dir = os.path.join(high_level_path, "contracts", c_name)
            if not os.path.exists(c_result_dir):
                c_result_dir = high_level_path

            if not os.path.exists(os.path.join(c_result_dir, "reward_history.npy")):
                continue

            initial_margin_history = np.load(os.path.join(c_result_dir, "initial_margin_history.npy"))
            maintain_marigine_history = np.load(os.path.join(c_result_dir, "maintain_marigine_history.npy"))
            new_position_required_money_history = np.load(os.path.join(c_result_dir, "new_position_required_money_history.npy"))
            reward_history = np.load(os.path.join(c_result_dir, "reward_history.npy"))
            unrealized_pnl_history = np.load(os.path.join(c_result_dir, "unrealized_pnl_history.npy"))
            wallet_balance_history = np.load(os.path.join(c_result_dir, "wallet_balance_history.npy"))

            requred_money = calculate_required_money(
                initial_margin_history,
                maintain_marigine_history,
                new_position_required_money_history,
                unrealized_pnl_history,
                wallet_balance_history,
            )

            df = pd.read_feather(feather_path)
            if self.early_stop > 0:
                df = df.iloc[:-self.early_stop]

            result_dict = {}
            result_dict["Buy & Hold"] = np.array(df.mark_price / df.mark_price.iloc[0]) - 1
            result_dict["Buy & Hold 5 times leverage"] = (np.array(df.mark_price / df.mark_price.iloc[0]) - 1) * 5

            accummulative_reward_sum = [reward_history[0]]
            for i in range(len(reward_history) - 1):
                accummulative_reward_sum.append(accummulative_reward_sum[-1] + reward_history[i + 1])
            result_dict["DiHFT"] = np.array(accummulative_reward_sum) / (requred_money + 1e-12)

            all_contract_data.append((c_name, df, result_dict))

            # Individual plot for each contract
            fig, ax = plt.subplots(figsize=(14, 5))
            for i, key in enumerate(result_dict.keys()):
                if i in (0, 1):
                    plt.plot(
                        df.timestamp[:],
                        result_dict[key] * 100,
                        color=color_list[i],
                        label=key,
                        linewidth=2,
                    )
                else:
                    plt.plot(
                        df.timestamp[:len(result_dict[key])],
                        result_dict[key] * 100,
                        color=color_list[i],
                        label=key,
                        linewidth=2,
                    )
            plt.title(f"Contract: {c_name} (Valid Dataset)", size=16)
            plt.xlabel("Trading Timestamp(s)", size=14)
            plt.ylabel("Total Return(%)", size=14)
            plt.grid(ls="--")
            ax = plt.gca()
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.16),
                ncol=3,
                fontsize=14,
                frameon=True,
            )
            plt.savefig(os.path.join(self.save_path, f"best_result_{c_name}.pdf"), bbox_inches="tight")
            plt.savefig(os.path.join(self.save_path, f"best_result_{c_name}.png"), bbox_inches="tight")
            plt.close()

        # Combined multi-panel figure for all contracts
        num_contracts = len(all_contract_data)
        if num_contracts > 0:
            cols = 3
            rows = (num_contracts + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows), squeeze=False)
            for idx, (c_name, df, result_dict) in enumerate(all_contract_data):
                r_idx, c_idx = divmod(idx, cols)
                ax = axes[r_idx, c_idx]
                for i, key in enumerate(result_dict.keys()):
                    if i in (0, 1):
                        ax.plot(
                            df.timestamp[:],
                            result_dict[key] * 100,
                            color=color_list[i],
                            label=key,
                            linewidth=1.5,
                        )
                    else:
                        ax.plot(
                            df.timestamp[:len(result_dict[key])],
                            result_dict[key] * 100,
                            color=color_list[i],
                            label=key,
                            linewidth=1.5,
                        )
                ax.set_title(f"Contract: {c_name}", fontsize=12)
                ax.set_xlabel("Timestamp", fontsize=10)
                ax.set_ylabel("Return (%)", fontsize=10)
                ax.grid(ls="--")
                if idx == 0:
                    ax.legend(loc="upper left", fontsize=10)
            
            for idx in range(num_contracts, rows * cols):
                r_idx, c_idx = divmod(idx, cols)
                fig.delaxes(axes[r_idx, c_idx])

            plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, "best_result_all_contracts.pdf"), bbox_inches="tight")
            plt.savefig(os.path.join(self.save_path, "best_result.pdf"), bbox_inches="tight")
            plt.close()


if __name__ == "__main__":
    args = parser.parse_args()
    picker = Picker(args)
    picker.analysis_all_epoch()
    picker.analysis_best_epoch()
    picker.create_best_agent()
    picker.plot()
