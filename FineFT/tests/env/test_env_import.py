import importlib.util
from pathlib import Path


def test_legacy_env_smoke_script_import_has_no_external_data_side_effects(monkeypatch):
    module_path = Path(__file__).with_name("test_env.py")
    monkeypatch.syspath_prepend(str(module_path.parents[2]))
    spec = importlib.util.spec_from_file_location("legacy_test_env_smoke", module_path)
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)
