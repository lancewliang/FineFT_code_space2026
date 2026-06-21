import sys
from pathlib import Path


DATA_PREPROCESS_ROOT = Path(__file__).resolve().parents[1]
if str(DATA_PREPROCESS_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_PREPROCESS_ROOT))
