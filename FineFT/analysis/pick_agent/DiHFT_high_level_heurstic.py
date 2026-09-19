import pandas as pd
import argparse
import os
import re
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
from common import (
    ArtifactNames,
    HistoryArtifactNames,
    MetricColumns,
    RoutingParamColumns,
    get_heuristic_plot_filename,
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
parser.add_argument(
    "--result_path",
    type=str,
    default="result/DiHFT/high_level",
    help="base path of high_level results",
)
parser.add_argument(
    "--optuna_csv",
    type=str,
    default=None,
    help="path to optuna_results.csv",
)
parser.add_argument(
    "--selection_metric",
    type=str,
    default=MetricColumns.TR,
    choices=[
        MetricColumns.TR,
        MetricColumns.PORTFOLIO_TR,
        MetricColumns.ANNUAL_SR,
        MetricColumns.DAILY_CR,
        MetricColumns.DAILY_SOR,
    ],
    help="metric used to pick the best agent",
)


class Picker:
    def __init__(self, args) -> None:
        self.base_path = args.base_path
        self.dataset_name = args.dataset_name
        self.experiment_name = args.experiment_name
        self.save_path = os.path.join(args.save_path, args.dataset_name, self.experiment_name)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

        self.early_stop = args.early_stop
        self.result_path = args.result_path
        self.optuna_csv = args.optuna_csv
        self.selection_metric = args.selection_metric
        self.optuna_param_lookup = self._load_optuna_parameters()

    def _find_optuna_csv(self) -> str | None:
        if self.optuna_csv and os.path.exists(self.optuna_csv):
            return self.optuna_csv
        default_csv = os.path.join(
            self.result_path,
            self.dataset_name,
            self.experiment_name,
            "vae_risk_aware_routing_optuna",
            ArtifactNames.OPTUNA_RESULTS_CSV,
        )
        if os.path.exists(default_csv):
            return default_csv
        return None

    def _load_optuna_parameters(self) -> dict[int, dict[str, float | int]]:
        csv_path = self._find_optuna_csv()
        if not csv_path:
            return {}
        df = pd.read_csv(csv_path)
        lookup = {}
        for _, row in df.iterrows():
            trial_id = int(row[RoutingParamColumns.NUMBER])
            if RoutingParamColumns.PARAMS_SLOPE_WINDOW_LENGTH in df.columns:
                lookup[trial_id] = {
                    RoutingParamColumns.SLOPE_WINDOW_LENGTH: int(
                        row[RoutingParamColumns.PARAMS_SLOPE_WINDOW_LENGTH]
                    ),
                    RoutingParamColumns.VOLATILITY_WINDOW_LENGTH: int(
                        row[RoutingParamColumns.PARAMS_VOLATILITY_WINDOW_LENGTH]
                    ),
                    RoutingParamColumns.SLOPE_GAMMA: float(
                        row[RoutingParamColumns.PARAMS_SLOPE_GAMMA]
                    ),
                    RoutingParamColumns.VOLATILITY_GAMMA: float(
                        row[RoutingParamColumns.PARAMS_VOLATILITY_GAMMA]
                    ),
                    RoutingParamColumns.SLOPE_RULE_BASE_THRESHOLD: float(
                        row[RoutingParamColumns.PARAMS_SLOPE_RULE_BASE_THRESHOLD]
                    ),
                    RoutingParamColumns.VOLATILITY_RULE_BASE_THRESHOLD: float(
                        row[RoutingParamColumns.PARAMS_VOLATILITY_RULE_BASE_THRESHOLD]
                    ),
                }
            elif RoutingParamColumns.PARAMS_WINDOW_LENGTH in df.columns:
                w = int(row[RoutingParamColumns.PARAMS_WINDOW_LENGTH])
                g = float(row[RoutingParamColumns.PARAMS_GAMMA])
                t = float(row[RoutingParamColumns.PARAMS_RULE_BASE_THRESHOLD])
                lookup[trial_id] = {
                    RoutingParamColumns.SLOPE_WINDOW_LENGTH: w,
                    RoutingParamColumns.VOLATILITY_WINDOW_LENGTH: w,
                    RoutingParamColumns.SLOPE_GAMMA: g,
                    RoutingParamColumns.VOLATILITY_GAMMA: g,
                    RoutingParamColumns.SLOPE_RULE_BASE_THRESHOLD: t,
                    RoutingParamColumns.VOLATILITY_RULE_BASE_THRESHOLD: t,
                }
        return lookup

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
                os.path.join(data_dir, HistoryArtifactNames.INITIAL_MARGIN_HISTORY_NPY)
            )
            maintain_marigine_history = np.load(
                os.path.join(data_dir, HistoryArtifactNames.MAINTAIN_MARGIN_HISTORY_NPY)
            )
            new_position_required_money_history = np.load(
                os.path.join(data_dir, HistoryArtifactNames.NEW_POSITION_REQUIRED_MONEY_HISTORY_NPY)
            )
            micro_action_history = np.load(
                os.path.join(data_dir, HistoryArtifactNames.MICRO_ACTION_HISTORY_NPY)
            )
            reward_history = np.load(os.path.join(data_dir, HistoryArtifactNames.REWARD_HISTORY_NPY))
            total_asset_history = np.load(
                os.path.join(data_dir, HistoryArtifactNames.TOTAL_ASSET_HISTORY_NPY)
            )
            unrealized_pnl_history = np.load(
                os.path.join(data_dir, HistoryArtifactNames.UNREALIZED_PNL_HISTORY_NPY)
            )
            wallet_balance_history = np.load(
                os.path.join(data_dir, HistoryArtifactNames.WALLET_BALANCE_HISTORY_NPY)
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
            self.result_path, self.dataset_name, self.experiment_name, "vae_risk_aware_routing"
        )
        if not os.path.exists(model_root_path):
            raise FileNotFoundError(f"Model root path not found: {model_root_path}")
        parameter_list = os.listdir(model_root_path)
        for parameter in parameter_list:
            epoch_path = os.path.join(model_root_path, parameter)
            if not os.path.isdir(epoch_path):
                continue
            if not os.listdir(epoch_path):
                continue
            result = self.analysis_single_epoch(epoch_path)

            trial_match = re.search(r"trial_(\d+)", parameter)
            trial_id = int(trial_match.group(1)) if trial_match else None
            result[MetricColumns.TRIAL_ID] = trial_id

            if trial_id is not None and trial_id in self.optuna_param_lookup:
                result.update(self.optuna_param_lookup[trial_id])
            else:
                ws_m = re.search(r"ws_(\d+)", parameter)
                wv_m = re.search(r"wv_(\d+)", parameter)
                gs_m = re.search(r"gs_([0-9.]+)", parameter)
                gv_m = re.search(r"gv_([0-9.]+)", parameter)
                ts_m = re.search(r"ts_([0-9.]+)", parameter)
                tv_m = re.search(r"tv_([0-9.]+)", parameter)
                gamma_m = re.search(r"gamma_([0-9.]+)", parameter)
                window_m = re.search(r"window_([0-9]+)", parameter)
                thresh_m = re.search(r"threshold_([0-9.]+)", parameter)

                result[RoutingParamColumns.SLOPE_WINDOW_LENGTH] = (
                    int(ws_m.group(1)) if ws_m else (int(window_m.group(1)) if window_m else None)
                )
                result[RoutingParamColumns.VOLATILITY_WINDOW_LENGTH] = (
                    int(wv_m.group(1)) if wv_m else (int(window_m.group(1)) if window_m else None)
                )
                result[RoutingParamColumns.SLOPE_GAMMA] = (
                    float(gs_m.group(1)) if gs_m else (float(gamma_m.group(1)) if gamma_m else None)
                )
                result[RoutingParamColumns.VOLATILITY_GAMMA] = (
                    float(gv_m.group(1)) if gv_m else (float(gamma_m.group(1)) if gamma_m else None)
                )
                result[RoutingParamColumns.SLOPE_RULE_BASE_THRESHOLD] = (
                    float(ts_m.group(1)) if ts_m else (float(thresh_m.group(1)) if thresh_m else None)
                )
                result[RoutingParamColumns.VOLATILITY_RULE_BASE_THRESHOLD] = (
                    float(tv_m.group(1)) if tv_m else (float(thresh_m.group(1)) if thresh_m else None)
                )

            result_list.append(result)
        result_df = pd.DataFrame(result_list)
        self.result_df = result_df
        result_df.to_csv(os.path.join(self.save_path, ArtifactNames.RESULT_CSV), index=False)

    def analysis_best_epoch(self):
        best_results = []
        subset_metric = [self.selection_metric] if self.selection_metric in self.result_df.columns else [MetricColumns.TR]
        df_clean = self.result_df.dropna(subset=subset_metric)
        if df_clean.empty:
            df_clean = self.result_df.fillna(0.0)

        max_candidates = [
            MetricColumns.TR,
            MetricColumns.PORTFOLIO_TR,
            MetricColumns.ANNUAL_SR,
            MetricColumns.DAILY_CR,
            MetricColumns.DAILY_SOR,
        ]
        if self.selection_metric in max_candidates:
            max_indicators = [self.selection_metric] + [
                m for m in max_candidates if m != self.selection_metric
            ]
        else:
            max_indicators = max_candidates

        for indicator in max_indicators:
            if indicator in df_clean.columns:
                idx_series = df_clean[indicator].dropna()
                best_idx = idx_series.idxmax() if not idx_series.empty else df_clean.index[0]
                max_row = df_clean.loc[[best_idx]].copy()
                max_row[MetricColumns.INDICATOR] = indicator
                best_results.append(max_row)
        for indicator in [
            MetricColumns.DAILY_VOL,
            MetricColumns.MDD,
            MetricColumns.DOWNSIDE_DEVIATION_DAILY,
        ]:
            if indicator in df_clean.columns:
                idx_series = df_clean[indicator].dropna()
                best_idx = idx_series.idxmin() if not idx_series.empty else df_clean.index[0]
                min_row = df_clean.loc[[best_idx]].copy()
                min_row[MetricColumns.INDICATOR] = indicator
                best_results.append(min_row)
        best_results_df = pd.concat(best_results)
        self.best_result_df = best_results_df
        best_results_df.to_csv(os.path.join(self.save_path, ArtifactNames.BEST_RESULT_CSV), index=False)

    def _resolve_final_result_path(self) -> str:
        if self.result_path.endswith("final_result"):
            final_root = self.result_path
        elif self.result_path.endswith("high_level"):
            base = self.result_path[: -len("high_level")].rstrip("/" + "\\")
            final_root = os.path.join(base, "final_result") if base else "result/DiHFT/final_result"
        else:
            final_root = os.path.join(self.result_path, "final_result")
        return os.path.join(final_root, self.dataset_name, self.experiment_name)

    def _get_best_row(self) -> pd.Series:
        if "indicator" in self.best_result_df.columns:
            metric_rows = self.best_result_df[self.best_result_df["indicator"] == self.selection_metric]
            if not metric_rows.empty:
                return metric_rows.iloc[0]
        return self.best_result_df.iloc[0]

    def _format_best_para_str(self, best_row: pd.Series) -> str:
        keys = [
            RoutingParamColumns.SLOPE_WINDOW_LENGTH,
            RoutingParamColumns.VOLATILITY_WINDOW_LENGTH,
            RoutingParamColumns.SLOPE_GAMMA,
            RoutingParamColumns.VOLATILITY_GAMMA,
            RoutingParamColumns.SLOPE_RULE_BASE_THRESHOLD,
            RoutingParamColumns.VOLATILITY_RULE_BASE_THRESHOLD,
        ]
        if all(k in best_row and pd.notna(best_row[k]) for k in keys):
            ws = int(best_row[RoutingParamColumns.SLOPE_WINDOW_LENGTH])
            wv = int(best_row[RoutingParamColumns.VOLATILITY_WINDOW_LENGTH])
            gs = best_row[RoutingParamColumns.SLOPE_GAMMA]
            gv = best_row[RoutingParamColumns.VOLATILITY_GAMMA]
            ts = best_row[RoutingParamColumns.SLOPE_RULE_BASE_THRESHOLD]
            tv = best_row[RoutingParamColumns.VOLATILITY_RULE_BASE_THRESHOLD]
            trial_id = (
                best_row[MetricColumns.TRIAL_ID]
                if (MetricColumns.TRIAL_ID in best_row and pd.notna(best_row[MetricColumns.TRIAL_ID]))
                else None
            )
            if trial_id is not None:
                return f"trial_{int(trial_id)}_ws_{ws}_wv_{wv}_gs_{gs}_gv_{gv}_ts_{ts}_tv_{tv}"
            return f"ws_{ws}_wv_{wv}_gs_{gs}_gv_{gv}_ts_{ts}_tv_{tv}"
        return os.path.basename(best_row["path"])

    def create_best_agent(self):
        best_row = self._get_best_row()
        path = best_row["path"]
        para = self._format_best_para_str(best_row)
        
        high_level_path = self._resolve_final_result_path()
        
        if not os.path.exists(high_level_path):
            os.makedirs(high_level_path, exist_ok=True)

        with open(
            os.path.join(high_level_path, ArtifactNames.HIGH_LEVEL_AGENT_PARA_TXT), "w", encoding="utf-8"
        ) as file:
            file.write(f"{para}\n")

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
        high_level_path = self._resolve_final_result_path()
        
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

            if not os.path.exists(os.path.join(c_result_dir, HistoryArtifactNames.REWARD_HISTORY_NPY)):
                continue

            initial_margin_history = np.load(os.path.join(c_result_dir, HistoryArtifactNames.INITIAL_MARGIN_HISTORY_NPY))
            maintain_marigine_history = np.load(os.path.join(c_result_dir, HistoryArtifactNames.MAINTAIN_MARGIN_HISTORY_NPY))
            new_position_required_money_history = np.load(os.path.join(c_result_dir, HistoryArtifactNames.NEW_POSITION_REQUIRED_MONEY_HISTORY_NPY))
            reward_history = np.load(os.path.join(c_result_dir, HistoryArtifactNames.REWARD_HISTORY_NPY))
            unrealized_pnl_history = np.load(os.path.join(c_result_dir, HistoryArtifactNames.UNREALIZED_PNL_HISTORY_NPY))
            wallet_balance_history = np.load(os.path.join(c_result_dir, HistoryArtifactNames.WALLET_BALANCE_HISTORY_NPY))

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
            plt.savefig(os.path.join(self.save_path, get_heuristic_plot_filename(c_name, "pdf")), bbox_inches="tight")
            plt.savefig(os.path.join(self.save_path, get_heuristic_plot_filename(c_name, "png")), bbox_inches="tight")
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
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True
    args = parser.parse_args()
    picker = Picker(args)
    picker.analysis_all_epoch()
    picker.analysis_best_epoch()
    picker.create_best_agent()
    picker.plot()
