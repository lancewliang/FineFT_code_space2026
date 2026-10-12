"""Inventory current split files; unknown fit/label provenance stays unresolved."""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
rows = []
times = {}
for stage in ("train", "valid", "test"):
    for path in sorted((ROOT / "dataset/10min/fu" / stage).glob("*.feather")):
        stamps = pd.to_datetime(pd.read_feather(path, columns=["timestamp"])["timestamp"])
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        rows.append({
            "contract": path.stem, "stage": stage,
            "file": str(path.relative_to(ROOT)), "sha256": digest.hexdigest(),
            "start": str(stamps.min()), "end": str(stamps.max()), "rows": len(stamps),
            "declared_directory_use": {"train": "training", "valid": "selection/tuning", "test": "proposed independent time-out test"}[stage],
            "actual_use_provenance": "pending complete pipeline audit",
            "label_maturity_cutoff": "unknown",
            "model_fit_cutoffs": "unknown",
            "feature_selection_and_transform_fit_cutoffs": "unknown",
        })
        times[stage, path.stem] = stamps
pd.DataFrame(rows).to_csv(OUTPUT / "dataset_usage_inventory.csv", index=False)
overlaps = []
for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
    names = sorted({c for stage, c in times if stage == left} & {c for stage, c in times if stage == right})
    for name in names:
        a, b = times[left, name], times[right, name]
        overlaps.append({
            "contract": name, "left_stage": left, "right_stage": right,
            "left_end": str(a.max()), "right_start": str(b.min()),
            "shared_timestamps": len(pd.Index(a).intersection(pd.Index(b))),
            "strictly_later": bool(a.max() < b.min()),
        })
(OUTPUT / "split_overlap_audit.json").write_text(json.dumps({
    "files": len(rows), "overlaps": overlaps,
    "independent_test_verified": False,
    "limitation": "File timestamps alone do not establish label maturity or fit cutoffs.",
}, ensure_ascii=False, indent=2) + "\n")

manifest_path = ROOT / "analysis_result/DiHFT/low_level/fu/10min_parallel/two_dimensional_selection/two_dimensional_selection_manifest.json"
manifest = json.loads(manifest_path.read_text())
artifacts = []
for slot in manifest["slots"]:
    artifacts.append({
        "artifact": f"Q-Net slot {slot['slot_id']} / {slot.get('candidate_id')}",
        "path": slot.get("model_path", "unknown"),
        "path_source": "selection manifest",
        "fit_cutoff": "unknown", "label_maturity_cutoff": "unknown",
        "selection_data": manifest.get("valid_root", "unknown"),
    })
for axis in ("slope", "volatility"):
    for label in range(3):
        artifacts.append({
            "artifact": f"VAE {axis} label_{label}",
            "path": f"result/DiHFT/vae_results/fu/10min_parallel/{axis}/label_{label}/model_latest.pth",
            "path_source": "routing default; exact historical runtime configuration requires audit",
            "fit_cutoff": "unknown", "label_maturity_cutoff": "unknown", "selection_data": "unknown",
        })
for name in ("slope feature selection", "volatility feature selection", "RL feature selection", "scaling/feature transforms", "regime thresholds", "VAE quantile calibration"):
    artifacts.append({
        "artifact": name, "path": "pending pipeline provenance audit", "path_source": "required component",
        "fit_cutoff": "unknown", "label_maturity_cutoff": "unknown", "selection_data": "unknown",
    })
pd.DataFrame(artifacts).to_csv(OUTPUT / "artifact_fit_inventory.csv", index=False)
