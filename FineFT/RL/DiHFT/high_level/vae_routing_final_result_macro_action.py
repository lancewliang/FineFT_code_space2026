import argparse
import os
import sys
import torch

sys.path.append(".")
from RL.DiHFT.high_level.vae_routing_util import (
    parser as base_parser,
    resolve_routing_parameters,
    seed_torch,
    vae_risk_aware_routing,
    logger,
)


def parse_and_prepare_args(args_list=None):
    """Parse and resolve configuration for final result evaluation."""
    parser = argparse.ArgumentParser(parents=[base_parser], add_help=False)
    parser.set_defaults(eval_stage="test")

    if args_list is not None:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    # Automatically find optuna_csv if not specified
    if not args.optuna_csv:
        default_optuna_csv = os.path.join(
            "result",
            "DiHFT",
            "high_level",
            args.dataset_name,
            args.experiment_name,
            "vae_risk_aware_routing_optuna",
            "optuna_results.csv",
        )
        if os.path.exists(default_optuna_csv):
            args.optuna_csv = default_optuna_csv

    # Automatically find para_file if not specified
    if not args.para_file:
        default_para_file = os.path.join(
            "result",
            "DiHFT",
            "final_result",
            args.dataset_name,
            args.experiment_name,
            "high_level_agent_para.txt",
        )
        if os.path.exists(default_para_file):
            args.para_file = default_para_file

    # Resolve dual-axis routing parameters
    args = resolve_routing_parameters(args)

    # Automatically find selection_manifest if not specified
    if not args.selection_manifest:
        default_manifest = os.path.join(
            "analysis_result",
            "DiHFT",
            "low_level",
            args.dataset_name,
            args.experiment_name,
            "two_dimensional_selection",
            "two_dimensional_selection_manifest.json",
        )
        if os.path.exists(default_manifest):
            args.selection_manifest = default_manifest

    # Point result_path to final_result if still on default high_level
    if args.result_path == "result/DiHFT/high_level":
        args.result_path = "result/DiHFT/final_result"

    return args


def main():
    seed_torch(42)

    args = parse_and_prepare_args()
    logger.info("Executing final result dual-dimension VAE routing test with args: %s", args)
    vae_routing = vae_risk_aware_routing(args)
    final_return_rate = vae_routing.test()
    logger.info("Final result test completed with return rate: %.6f", final_return_rate)
    return final_return_rate


if __name__ == "__main__":
    main()
