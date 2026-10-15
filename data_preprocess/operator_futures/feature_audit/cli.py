"""Public CLI interface for Multi-Horizon Feature Audit."""

import argparse
import traceback
import sys
from pathlib import Path

from .comparisons import ComparisonConfig, ComparisonRunner
from .config import AuditConfig
from .runner import AuditRunner


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="feature_audit",
        description="Independent Multi-Horizon Feature Audit CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # validate-inputs
    val_parser = subparsers.add_parser("validate-inputs", help="Validate audit configuration and manifests")
    val_parser.add_argument("--config", required=True, help="Path to audit config YAML/JSON")

    # run
    run_parser = subparsers.add_parser("run", help="Execute independent feature audit run")
    run_parser.add_argument("--config", required=True, help="Path to audit config YAML/JSON")

    # compare
    cmp_parser = subparsers.add_parser("compare", help="Compare multiple audit runs")
    cmp_parser.add_argument("--manifest", required=True, help="Path to comparison manifest YAML/JSON")

    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.command == "validate-inputs":
        try:
            config = AuditConfig.load(args.config)
            errors = config.validate()
            if errors:
                print(f"Validation failed:\n" + "\n".join(f" - {e}" for e in errors), file=sys.stderr)
                return 1
            print("Input validation passed successfully.")
            return 0
        except Exception as e:
            traceback.print_exc()
            print(f"Error during input validation: {e}", file=sys.stderr)
            return 1

    elif args.command == "run":
        try:
            config = AuditConfig.load(args.config)
            runner = AuditRunner(config)
            manifest = runner.run()
            print(f"Audit run completed successfully. Status: {manifest.get('status')}")
            return 0
        except Exception as e:
            traceback.print_exc()
            print(f"Error during audit run: {e}", file=sys.stderr)
            return 1

    elif args.command == "compare":
        try:
            cmp_config = ComparisonConfig.load(args.manifest)
            runner = ComparisonRunner(cmp_config)
            manifest = runner.run()
            print(f"Comparison completed successfully. Status: {manifest.get('status')}")
            return 0
        except Exception as e:
            traceback.print_exc()
            print(f"Error during comparison: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
