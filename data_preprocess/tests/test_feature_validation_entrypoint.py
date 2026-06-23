from pathlib import Path
import os
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def _copy_commodity_script_tree(tmp_path: Path) -> Path:
    source_dir = (
        REPO_ROOT
        / "data_preprocess"
        / "script_preprocess"
        / "future_upgraded"
        / "commodity"
    )
    target_dir = (
        tmp_path
        / "data_preprocess"
        / "script_preprocess"
        / "future_upgraded"
        / "commodity"
    )
    target_dir.parent.mkdir(parents=True)
    shutil.copytree(source_dir, target_dir)
    return target_dir


def test_validate_features_shell_invokes_cli_without_main_sh(tmp_path):
    script_dir = _copy_commodity_script_tree(tmp_path)
    root_path = tmp_path
    cli_dir = (
        root_path
        / "data_preprocess"
        / "operator_futures"
        / "feature_validation"
    )
    cli_dir.mkdir(parents=True)
    (cli_dir / "__init__.py").write_text("", encoding="utf-8")
    (cli_dir / "validate_features.py").write_text(
        "import argparse\n"
        "def main(argv=None):\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--root_path', required=True)\n"
        "    parser.add_argument('--symbol', required=True)\n"
        "    parser.add_argument('--target_freq', required=True)\n"
        "    parser.add_argument('--start_date', required=True)\n"
        "    parser.add_argument('--end_date', required=True)\n"
        "    parser.add_argument('--report_dir', required=True)\n"
        "    args = parser.parse_args(argv)\n"
        "    print('feature-validation-cli', args.symbol, args.target_freq)\n"
        "    return 0\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(main())\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            str(script_dir / "validate_features.sh"),
            "--root_path",
            str(root_path),
            "--symbol",
            "fu",
            "--target_freq",
            "5min",
            "--start_date",
            "2025-11-03",
            "--end_date",
            "2025-11-08",
        ],
        cwd=root_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "feature-validation-cli fu 5min" in result.stdout
    assert "main.sh" not in result.stdout
    assert "main.sh" not in result.stderr


def test_validate_features_shell_honors_report_dir(tmp_path):
    script_dir = _copy_commodity_script_tree(tmp_path)
    root_path = tmp_path
    report_dir = tmp_path / "custom_reports"
    cli_dir = (
        root_path
        / "data_preprocess"
        / "operator_futures"
        / "feature_validation"
    )
    cli_dir.mkdir(parents=True)
    (cli_dir / "__init__.py").write_text("", encoding="utf-8")
    (cli_dir / "validate_features.py").write_text(
        "import argparse\n"
        "def main(argv=None):\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--root_path', required=True)\n"
        "    parser.add_argument('--symbol', required=True)\n"
        "    parser.add_argument('--target_freq', required=True)\n"
        "    parser.add_argument('--start_date', required=True)\n"
        "    parser.add_argument('--end_date', required=True)\n"
        "    parser.add_argument('--report_dir', required=True)\n"
        "    args = parser.parse_args(argv)\n"
        "    print(args.report_dir)\n"
        "    return 0\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(main())\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            str(script_dir / "validate_features.sh"),
            "--root_path",
            str(root_path),
            "--symbol",
            "fu",
            "--target_freq",
            "5min",
            "--start_date",
            "2025-11-03",
            "--end_date",
            "2025-11-08",
            "--report_dir",
            str(report_dir),
        ],
        cwd=root_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert str(report_dir) in result.stdout


def test_feature_validation_cli_help_imports():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "operator_futures.feature_validation.validate_features",
            "--help",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--symbol" in result.stdout
    assert "--target_freq" in result.stdout


def test_pandas_reference_modules_import_from_validation_namespace():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import operator_futures.feature_validation.pandas_reference.cross_section.create_feature;"
            "import operator_futures.feature_validation.pandas_reference.time_operator.create_feature_multi_processing;"
            "import operator_futures.feature_validation.pandas_reference.scale_describe_save.scale_save",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
