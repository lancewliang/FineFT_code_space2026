from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "data_preprocess/script_preprocess/future_upgraded/commodity/commodity_process.sh"
DOC_PATH = REPO_ROOT / "docs/上海商品交易所/commodity_futures_preprocess.md"
README_ZH = REPO_ROOT / "data_preprocess/README.zh_CN.md"
README_EN = REPO_ROOT / "data_preprocess/README.md"


def test_commodity_preprocess_artifacts_exist_and_are_referenced():
    assert SCRIPT_PATH.exists()
    assert DOC_PATH.exists()

    subprocess.run(["bash", "-n", str(SCRIPT_PATH)], check=True)

    zh_text = README_ZH.read_text(encoding="utf-8")
    en_text = README_EN.read_text(encoding="utf-8")
    doc_text = DOC_PATH.read_text(encoding="utf-8")

    assert "商品期货" in zh_text
    assert "Commodity Futures" in en_text
    assert "燃料油" in doc_text
