import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .merge_vae_train import label_name_from_index, vae_data_dir
except ImportError:
    from merge_vae_train import label_name_from_index, vae_data_dir

try:
    from .manifests import (
        AcceptanceStats,
        ContractLogpxSummary,
        ContractRoutingSummary,
        LabelSummary,
        LabelTestSummary,
        LogpxStats,
        LogpxSummary,
        RoutingSummary,
        SampleIntegrity,
        WinnerSummary,
    )
except ImportError:
    from manifests import (
        AcceptanceStats,
        ContractLogpxSummary,
        ContractRoutingSummary,
        LabelSummary,
        LabelTestSummary,
        LogpxStats,
        LogpxSummary,
        RoutingSummary,
        SampleIntegrity,
        WinnerSummary,
    )


SUMMARY_QUANTILES = (
    ("q01", 0.01),
    ("q05", 0.05),
    ("q25", 0.25),
    ("q50", 0.50),
    ("q75", 0.75),
    ("q95", 0.95),
    ("q99", 0.99),
)


def _logpx_stats(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("logpx array has no samples")
    return LogpxStats(
        samples=int(values.size),
        logpx_mean=float(np.mean(values)),
        logpx_std=float(np.std(values)),
        logpx_min=float(np.min(values)),
        logpx_max=float(np.max(values)),
        quantiles={
            name: float(np.quantile(values, quantile))
            for name, quantile in SUMMARY_QUANTILES
        },
    )


def _acceptance_stats(values, train_quantiles):
    values = np.asarray(values, dtype=float).reshape(-1)
    return AcceptanceStats(
        ge_train_q01_pct=float(np.mean(values >= train_quantiles["q01"]) * 100.0),
        ge_train_q05_pct=float(np.mean(values >= train_quantiles["q05"]) * 100.0),
        ge_train_q50_pct=float(np.mean(values >= train_quantiles["q50"]) * 100.0),
    )


def _sample_integrity(input_samples, analyzed_samples):
    return SampleIntegrity(
        input_samples=int(input_samples),
        analyzed_samples=int(analyzed_samples),
    )


def _logpx_summary(input_samples, logpx, acceptance=None):
    flat_logpx = np.asarray(logpx, dtype=float).reshape(-1)
    return LogpxSummary(
        integrity=_sample_integrity(input_samples, flat_logpx.size),
        stats=_logpx_stats(flat_logpx),
        acceptance=acceptance,
    )


def _logpx_rows(contract, source_file, logpx):
    return [
        {
            "contract": contract,
            "source_file": source_file,
            "row_index": int(index),
            "logpx": float(value),
        }
        for index, value in enumerate(np.asarray(logpx, dtype=float).reshape(-1))
    ]


def write_contract_logpx_outputs(
    contract_results, save_path, dataset_name, label, train_baseline=None
):
    os.makedirs(save_path, exist_ok=True)
    all_logpx = []
    all_rows = []
    contract_summary = {}
    train_summary = None
    if train_baseline is not None:
        train_logpx = np.asarray(train_baseline.logpx, dtype=float).reshape(-1)
        train_summary = ContractLogpxSummary(
            source_file=train_baseline.source_file,
            summary=LogpxSummary(
                integrity=_sample_integrity(
                    train_baseline.input_samples,
                    train_baseline.analyzed_samples,
                ),
                stats=_logpx_stats(train_logpx),
            ),
        )
    for result in sorted(contract_results, key=lambda item: item.contract):
        contract = result.contract
        source_file = result.source_file
        logpx = np.asarray(result.logpx, dtype=float)
        flat_logpx = logpx.reshape(-1)
        input_samples = int(
            result.input_samples
            if result.input_samples is not None
            else flat_logpx.size
        )
        np.save(os.path.join(save_path, f"ood_logpx_{contract}.npy"), logpx)
        rows = _logpx_rows(contract, source_file, logpx)
        pd.DataFrame(
            rows, columns=["contract", "source_file", "row_index", "logpx"]
        ).to_csv(
            os.path.join(save_path, f"ood_logpx_{contract}.csv"),
            index=False,
        )
        all_logpx.append(logpx)
        all_rows.extend(rows)
        acceptance = None
        if train_summary is not None:
            acceptance = _acceptance_stats(
                flat_logpx, train_summary.summary.stats.quantiles
            )
        contract_summary[contract] = ContractLogpxSummary(
            source_file=source_file,
            summary=_logpx_summary(input_samples, flat_logpx, acceptance),
        )
    combined = np.concatenate(all_logpx, axis=0)
    np.save(os.path.join(save_path, "ood_logpx_all.npy"), combined)
    pd.DataFrame(
        all_rows, columns=["contract", "source_file", "row_index", "logpx"]
    ).to_csv(os.path.join(save_path, "ood_logpx_all.csv"), index=False)
    total_input_samples = sum(
        int(
            item.input_samples
            if item.input_samples is not None
            else np.asarray(item.logpx, dtype=float).reshape(-1).size
            )
        for item in contract_results
    )
    combined_flat = combined.reshape(-1)
    all_summary = _logpx_summary(total_input_samples, combined_flat)
    if train_summary is not None:
        all_summary = _logpx_summary(
            total_input_samples,
            combined_flat,
            _acceptance_stats(combined_flat, train_summary.summary.stats.quantiles),
        )
    summary = LabelSummary(
        dataset_name=dataset_name,
        label=label,
        test=LabelTestSummary(contracts=contract_summary, all=all_summary),
        train_baseline=train_summary,
    )
    with open(os.path.join(save_path, "summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary.to_dict(), file, ensure_ascii=False, indent=2)
    return summary


def _winner_summary(scores, labels, low_margin_threshold):
    scores = np.asarray(scores, dtype=float)
    samples = int(scores.shape[0])
    if samples == 0:
        return WinnerSummary(
            samples=0,
            winner_counts={label: 0 for label in labels},
            winner_pct={label: 0.0 for label in labels},
            top1_top2_margin_mean=0.0,
            top1_top2_margin_q25=0.0,
            low_margin_pct=0.0,
        )
    winners = np.argmax(scores, axis=1)
    sorted_scores = np.sort(scores, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2]
    counts = {
        label: int(np.sum(winners == index)) for index, label in enumerate(labels)
    }
    return WinnerSummary(
        samples=samples,
        winner_counts=counts,
        winner_pct={
            label: float(count / samples * 100.0) for label, count in counts.items()
        },
        top1_top2_margin_mean=float(np.mean(margins)),
        top1_top2_margin_q25=float(np.quantile(margins, 0.25)),
        low_margin_pct=float(np.mean(margins <= low_margin_threshold) * 100.0),
    )


def write_routing_summary(
    result_root,
    dataset_name,
    labels,
    low_margin_threshold=1.0,
):
    result_root = Path(result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    contract_names = None
    by_label = {}
    for label in labels:
        label_dir = result_root / label
        files = sorted(label_dir.glob("ood_logpx_*.npy"))
        contract_values = {
            path.stem.replace("ood_logpx_", ""): np.load(path).reshape(-1)
            for path in files
            if path.name != "ood_logpx_all.npy"
        }
        by_label[label] = contract_values
        names = set(contract_values)
        contract_names = names if contract_names is None else contract_names & names

    contract_summaries = {}
    all_scores = []
    for contract in sorted(contract_names or []):
        arrays = [by_label[label][contract] for label in labels]
        input_samples_by_label = {
            label: int(array.size) for label, array in zip(labels, arrays)
        }
        n = min(input_samples_by_label.values())
        scores = np.vstack([array[:n] for array in arrays]).T
        contract_summaries[contract] = ContractRoutingSummary(
            winner=_winner_summary(scores, labels, low_margin_threshold),
            input_samples_by_label=input_samples_by_label,
            sample_mismatch=len(set(input_samples_by_label.values())) != 1,
        )
        all_scores.append(scores)

    combined = (
        np.concatenate(all_scores, axis=0)
        if all_scores
        else np.empty((0, len(labels)))
    )
    summary = RoutingSummary(
        dataset_name=dataset_name,
        labels=list(labels),
        score_type="raw_logpx",
        low_margin_threshold=float(low_margin_threshold),
        contracts=contract_summaries,
        all=_winner_summary(combined, labels, low_margin_threshold),
    )
    with open(result_root / "routing_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary.to_dict(), file, ensure_ascii=False, indent=2)
    return summary


def maybe_write_routing_summary_after_analysis(args):
    labels = [label_name_from_index(index) for index in range(args.total_label_number)]
    test_dir = vae_data_dir(args.data_base_path, args.dataset_name) / "test"
    contracts = [
        path.stem[len("test_") :]
        for path in sorted(test_dir.glob("test_*.npy"), key=lambda item: item.name)
    ]
    result_root = os.path.join(args.base_model_path, "vae_results", args.dataset_name, args.experiment_name)
    for label in labels:
        for contract in contracts:
            logpx_path = os.path.join(result_root, label, f"ood_logpx_{contract}.npy")
            if not os.path.exists(logpx_path):
                return None
    return write_routing_summary(
        result_root=result_root,
        dataset_name=args.dataset_name,
        labels=labels,
    )
