import argparse
import copy
import os
import shutil
import sys
import torch

sys.path.append(".")
from common import ArtifactNames
from RL.DiHFT.high_level.frozen_experiment import (
    load_frozen_experiment,
    write_evaluation_receipt,
)
from RL.DiHFT.high_level.vae_routing_util import (
    ensure_precomputed_vae_quantiles,
    logger,
    parser as base_parser,
    resolve_routing_parameters,
    seed_torch,
    vae_risk_aware_routing,
)


def _evaluation_phase(args) -> str:
    """Map a declared data stage to the independently verified experiment phase."""
    return "calibration" if args.eval_stage == "valid" else "final"


def _validate_evaluation_purpose(args) -> None:
    """Prevent valid/main-sub and test/final artifacts from sharing a namespace."""
    if args.evaluation_purpose == "final" and args.eval_stage != "test":
        raise ValueError("final evaluation_purpose requires eval_stage=test")
    if args.evaluation_purpose == "main_sub" and args.eval_stage not in {"valid", "test"}:
        raise ValueError("main_sub evaluation_purpose requires eval_stage=valid or test")


def parse_and_prepare_args(args_list=None):
    """Parse and resolve configuration for final result evaluation."""
    parser = argparse.ArgumentParser(parents=[copy.deepcopy(base_parser)], add_help=False)
    parser.set_defaults(
        eval_stage="test",
        non_main_defense_mode="graceful_hold_exit",
        evaluation_purpose="final",
    )

    if args_list is not None:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    tokens = args_list if args_list is not None else sys.argv[1:]
    option_destinations = {
        option: action.dest for action in parser._actions for option in action.option_strings
    }
    args._explicit_routing_options = set()
    for token in tokens:
        if token.startswith("--"):
            option = token.split("=", 1)[0]
            if option in option_destinations:
                args._explicit_routing_options.add(option_destinations[option])
            else:
                # argparse already rejected ambiguous abbreviations. Preserve
                # legacy accepted abbreviations while detecting frozen overrides.
                args._explicit_routing_options.update(
                    destination for name, destination in option_destinations.items()
                    if name.startswith(option)
                )
    if args.frozen_config:
        if args.para_file or args.optuna_csv:
            raise ValueError("frozen_config cannot be combined with legacy para_file/optuna_csv")
        if "output_sub_dir" in args._explicit_routing_options:
            raise ValueError("frozen evaluation rejects output_sub_dir; use evaluation_purpose")
        _validate_evaluation_purpose(args)
        args = load_frozen_experiment(args, args.frozen_config, phase=_evaluation_phase(args))
        if args.result_path == "result/DiHFT/high_level":
            args.result_path = "result/DiHFT/final_result"
        # Keep a readable projection for callers that inspected the historical
        # field, while _resolve_test_path derives the authoritative namespace.
        args.output_sub_dir = os.path.join(
            "expert_hysteresis",
            args.frozen_config_hash,
            args.eval_stage,
            "final_evaluation" if args.evaluation_purpose == "final" else "main_sub",
        )
        return args
    if args.gating_strategy == "expert_hysteresis":
        raise ValueError("final expert_hysteresis evaluation requires frozen_config")

    # Automatically find optuna_csv if not specified
    if not args.optuna_csv:
        default_optuna_csv = os.path.join(
            "result",
            "DiHFT",
            "high_level",
            args.dataset_name,
            args.experiment_name,
            "vae_risk_aware_routing_optuna",
            ArtifactNames.OPTUNA_RESULTS_CSV,
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
            ArtifactNames.HIGH_LEVEL_AGENT_PARA_TXT,
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
            ArtifactNames.TWO_DIMENSIONAL_SELECTION_MANIFEST_JSON,
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
    if args.gating_strategy == "expert_hysteresis":
        args.precomputed_quantiles_dir = ensure_precomputed_vae_quantiles(args)
    logger.info("Executing final result dual-dimension VAE routing test with args: %s", args)
    vae_routing = vae_risk_aware_routing(args)
    final_return_rate = vae_routing.test()
    if args.gating_strategy == "expert_hysteresis" and args.save_artifacts:
        write_evaluation_receipt(args, vae_routing.test_path)
    logger.info("Final result test completed with return rate: %.6f", final_return_rate)

    if args.para_file and os.path.exists(args.para_file) and args.save_artifacts:
        dest_para = os.path.join(vae_routing.test_path, ArtifactNames.HIGH_LEVEL_AGENT_PARA_TXT)
        if os.path.abspath(args.para_file) != os.path.abspath(dest_para):
            shutil.copy(args.para_file, dest_para)

    # Automatically generate comprehensive diagnostics for final result
    from analysis.diagnostics.trading_diagnostics import TradingDiagnosticsCalculator

    data_dir = os.path.join(args.base_path, args.dataset_name, args.eval_stage)
    diag_calculator = TradingDiagnosticsCalculator(
        result_dir=vae_routing.test_path,
        data_dir=data_dir,
        output_dir=os.path.join(vae_routing.test_path, "diagnostics"),
        initial_wallet_balance=float(args.initial_wallet_balance),
        commission_rate=float(args.transcation_cost),
        contract_unit=float(args.diagnostic_contract_unit),
    )
    diag_calculator.run()

    # Automatically generate time-synchronized multi-contract portfolio diagnostics
    from analysis.diagnostics.synchronized_portfolio_diagnostics import (
        SynchronizedPortfolioCalculator,
    )

    port_calculator = SynchronizedPortfolioCalculator(
        result_dir=vae_routing.test_path,
        data_dir=data_dir,
        output_dir=vae_routing.test_path,
        portfolio_initial_capital=float(args.initial_wallet_balance),
        contract_unit=float(args.diagnostic_contract_unit),
        commission_rate=float(args.transcation_cost),
        freq=12,
        eval_stage=args.eval_stage,
    )
    port_calculator.run()

    return final_return_rate


if __name__ == "__main__":
    main()
