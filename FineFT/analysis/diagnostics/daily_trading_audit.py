from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.append(".")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from common.artifacts import AuditArtifactNames
from common.audit_records import (
    ACCOUNT_RECORD_COLUMNS,
    PREDICTION_RECORD_COLUMNS,
    TRADE_RECORD_COLUMNS,
    AccountRecord,
    AuditManifest,
    PredictionRecord,
    TradeRecord,
)
from common.routing_audit import (
    ROUTING_AUDIT_SCHEMA_VERSION,
    ROUTING_RECORD_COLUMNS,
    RoutingRecord,
    validate_routing_evidence,
)

logger = logging.getLogger("DailyTradingAudit")


class DailyTradingAuditPublisher:
    """Validates audit-grade simulation outputs and publishes daily partitioned CSVs atomically."""

    def __init__(
        self,
        result_dir: str | Path,
        market_data_dir: str | Path,
        output_dir: str | Path,
        run_id: str | None = None,
        trading_day_mapping: str | Path | None = None,
        contracts: list[str] | None = None,
    ):
        self.result_dir = Path(result_dir)
        self.market_data_dir = Path(market_data_dir)
        self.output_dir = Path(output_dir)
        self.specified_run_id = run_id
        self.trading_day_mapping_path = Path(trading_day_mapping) if trading_day_mapping else None
        self.specified_contracts = contracts
        self.trading_day_source = "market_data_column"

    def discover_contracts(self) -> list[str]:
        if self.specified_contracts:
            return sorted(self.specified_contracts)

        contracts_dir = self.result_dir / "contracts"
        if contracts_dir.is_dir():
            found = [d.name for d in contracts_dir.iterdir() if d.is_dir()]
            if found:
                return sorted(found)

        # Check if result_dir itself has audit artifacts
        if (self.result_dir / AuditArtifactNames.AUDIT_TRADES_JSON).is_file():
            meta_path = self.result_dir / AuditArtifactNames.AUDIT_METADATA_JSON
            if meta_path.is_file():
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                    if "contract" in meta:
                        return [meta["contract"]]
            return ["default_contract"]

        raise ValueError(f"No contract result directories found in {self.result_dir}")

    def _get_contract_result_dir(self, contract: str) -> Path:
        p1 = self.result_dir / "contracts" / contract
        if p1.is_dir():
            return p1
        p2 = self.result_dir / contract
        if p2.is_dir():
            return p2
        if (self.result_dir / AuditArtifactNames.AUDIT_TRADES_JSON).is_file():
            return self.result_dir
        raise ValueError(f"Cannot find result directory for contract {contract} in {self.result_dir}")

    def _load_market_data(self, contract: str) -> pd.DataFrame:
        candidates = [
            self.market_data_dir / f"{contract}.feather",
            self.market_data_dir / f"{contract}.csv",
        ]
        for c in candidates:
            if c.is_file():
                if c.suffix == ".feather":
                    return pd.read_feather(c)
                else:
                    return pd.read_csv(c)
        raise ValueError(f"Market data file for contract {contract} not found in {self.market_data_dir}")

    def _resolve_trading_days(self, contract: str, market_df: pd.DataFrame) -> list[str]:
        # 1. Check market data columns
        for col in ["trading_day", "TradingDay", "tradingDay"]:
            if col in market_df.columns:
                self.trading_day_source = "market_data_column"
                tds = [str(x) for x in market_df[col].tolist()]
                if any(x == "" or x.lower() == "nan" or x.lower() == "none" for x in tds):
                    raise ValueError(f"Market data column '{col}' contains empty or invalid TradingDay for {contract}")
                return tds

        # 2. Check explicit mapping file
        if self.trading_day_mapping_path and self.trading_day_mapping_path.is_file():
            self.trading_day_source = "mapping_file"
            if self.trading_day_mapping_path.suffix == ".json":
                with open(self.trading_day_mapping_path, encoding="utf-8") as f:
                    mapping = json.load(f)
            else:
                map_df = pd.read_csv(self.trading_day_mapping_path)
                # If mapping has timestamp and trading_day
                if "trading_day" in map_df.columns and "timestamp" in map_df.columns:
                    mapping = dict(zip(map_df["timestamp"].astype(str), map_df["trading_day"].astype(str)))
                elif "trading_day" in map_df.columns and "step" in map_df.columns:
                    mapping = dict(zip(map_df["step"].astype(str), map_df["trading_day"].astype(str)))
                else:
                    raise ValueError("Mapping CSV must have 'trading_day' and ('timestamp' or 'step') columns")

            tds = []
            if "timestamp" in market_df.columns:
                for ts in market_df["timestamp"].astype(str):
                    if ts in mapping:
                        tds.append(str(mapping[ts]))
                    elif contract in mapping and ts in mapping[contract]:
                        tds.append(str(mapping[contract][ts]))
                    else:
                        raise ValueError(f"Timestamp {ts} for {contract} missing in TradingDay mapping")
            else:
                for step_i in range(len(market_df)):
                    s_key = str(step_i)
                    if s_key in mapping:
                        tds.append(str(mapping[s_key]))
                    else:
                        raise ValueError(f"Step index {step_i} for {contract} missing in TradingDay mapping")
            return tds

        raise ValueError(
            f"Missing explicit TradingDay for contract {contract}: "
            "not found in market data columns ('trading_day') and no trading_day_mapping provided. "
            "Fixed clock or calendar guessing is strictly prohibited."
        )

    def publish(self) -> dict[str, Any]:
        if not self.result_dir.is_dir():
            raise ValueError(f"Result directory does not exist: {self.result_dir}")
        if not self.market_data_dir.is_dir():
            raise ValueError(f"Market data directory does not exist: {self.market_data_dir}")

        contracts = self.discover_contracts()
        all_predictions: list[PredictionRecord] = []
        all_trades: list[TradeRecord] = []
        all_accounts: list[AccountRecord] = []
        all_routing: list[RoutingRecord] = []
        action_positions: dict[str, list[float]] = {}

        global_meta: dict[str, Any] = {}

        for contract in contracts:
            c_dir = self._get_contract_result_dir(contract)
            # Load metadata
            meta_path = c_dir / AuditArtifactNames.AUDIT_METADATA_JSON
            if not meta_path.is_file():
                meta_path_alt = self.result_dir / AuditArtifactNames.AUDIT_METADATA_JSON
                if meta_path_alt.is_file():
                    meta_path = meta_path_alt
                else:
                    raise ValueError(
                        f"Missing audit-grade input: {AuditArtifactNames.AUDIT_METADATA_JSON} not found for contract {contract}."
                    )
            with open(meta_path, encoding="utf-8") as f:
                c_meta = json.load(f)
            if not global_meta:
                global_meta = c_meta

            # Load market data & trading days
            market_df = self._load_market_data(contract)
            market_trading_days = self._resolve_trading_days(contract, market_df)

            # Load account snapshots
            acc_path = c_dir / AuditArtifactNames.AUDIT_ACCOUNT_SNAPSHOTS_JSON
            if not acc_path.is_file():
                raise ValueError(
                    f"Missing audit-grade input: {AuditArtifactNames.AUDIT_ACCOUNT_SNAPSHOTS_JSON} not found for contract {contract}."
                )
            with open(acc_path, encoding="utf-8") as f:
                acc_data = json.load(f)

            # Load predictions
            pred_path = c_dir / AuditArtifactNames.AUDIT_PREDICTIONS_JSON
            if not pred_path.is_file():
                raise ValueError(
                    f"Missing audit-grade input: {AuditArtifactNames.AUDIT_PREDICTIONS_JSON} not found for contract {contract}."
                )
            with open(pred_path, encoding="utf-8") as f:
                pred_data = json.load(f)

            # Load trades
            trade_path = c_dir / AuditArtifactNames.AUDIT_TRADES_JSON
            if not trade_path.is_file():
                raise ValueError(
                    f"Missing audit-grade input: {AuditArtifactNames.AUDIT_TRADES_JSON} not found for contract {contract}."
                )
            with open(trade_path, encoding="utf-8") as f:
                trade_data = json.load(f)

            routing_schema_version = c_meta.get("routing_schema_version")
            if (
                c_meta.get("gating_strategy") == "expert_hysteresis"
                and routing_schema_version is None
            ):
                raise ValueError("Expert routing metadata is missing its schema version")
            if routing_schema_version is not None:
                if routing_schema_version != ROUTING_AUDIT_SCHEMA_VERSION:
                    raise ValueError("Unsupported routing metadata schema version")
                if c_meta.get("gating_strategy") != "expert_hysteresis":
                    raise ValueError("Routing metadata has an incompatible gating strategy")
                routing_path = c_dir / AuditArtifactNames.AUDIT_ROUTING_JSON
                if not routing_path.is_file():
                    raise ValueError(
                        f"Missing audit-grade input: {AuditArtifactNames.AUDIT_ROUTING_JSON} "
                        f"not found for contract {contract}."
                    )
                with open(routing_path, encoding="utf-8") as file:
                    routing_data = json.load(file)
                try:
                    action_positions[contract] = [
                        float(value) for value in c_meta["action_positions"]
                    ]
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid routing action-position metadata for {contract}"
                    ) from exc
                if not action_positions[contract] or not all(
                    math.isfinite(value) for value in action_positions[contract]
                ):
                    raise ValueError(
                        f"Invalid routing action-position metadata for {contract}"
                    )
                for item in routing_data:
                    step_idx = item.get("step_index", 0)
                    if not item.get("trading_day") and step_idx < len(
                        market_trading_days
                    ):
                        item["trading_day"] = market_trading_days[step_idx]
                    record = RoutingRecord.from_dict(item)
                    if (
                        record.run_id != str(c_meta.get("run_id", ""))
                        or record.contract != contract
                    ):
                        raise ValueError(
                            f"Routing metadata identity mismatch for {contract}"
                        )
                    all_routing.append(record)

            # Convert to typed records and assign trading_day if needed
            for item in pred_data:
                step_idx = item.get("step_index", 0)
                if not item.get("trading_day") and step_idx < len(market_trading_days):
                    item["trading_day"] = market_trading_days[step_idx]
                all_predictions.append(PredictionRecord.from_dict(item))

            for item in trade_data:
                step_idx = item.get("step_index", 0)
                if not item.get("trading_day") and step_idx < len(market_trading_days):
                    item["trading_day"] = market_trading_days[step_idx]
                all_trades.append(TradeRecord.from_dict(item))

            for item in acc_data:
                step_idx = item.get("step_index", 0)
                # Initial snapshot is step 0
                day_idx = min(step_idx, len(market_trading_days) - 1)
                if not item.get("trading_day"):
                    item["trading_day"] = market_trading_days[day_idx]
                all_accounts.append(AccountRecord.from_dict(item))

        # Perform strict validations
        self._validate(all_predictions, all_trades, all_accounts)
        if action_positions:
            if set(action_positions) != set(contracts):
                raise ValueError(
                    "Routing audit schema must cover every published contract"
                )
            validate_routing_evidence(
                all_routing,
                all_predictions,
                all_trades,
                action_positions,
            )

        # Atomic publication via staging
        run_id = self.specified_run_id or global_meta.get("run_id", "unspecified_run")
        manifest_dict = self._publish_atomic(
            run_id=run_id,
            model_id=str(global_meta.get("model_id", "unknown_model")),
            feature_version=str(global_meta.get("feature_version", "unknown_features")),
            parameter_version=str(global_meta.get("parameter_version", "unknown_params")),
            contracts=contracts,
            predictions=all_predictions,
            trades=all_trades,
            accounts=all_accounts,
            routing=all_routing if action_positions else None,
        )
        return manifest_dict

    def _validate(
        self,
        predictions: list[PredictionRecord],
        trades: list[TradeRecord],
        accounts: list[AccountRecord],
    ) -> None:
        # 1. Primary key uniqueness
        pred_ids = set()
        for p in predictions:
            if p.prediction_id in pred_ids:
                raise ValueError(f"Duplicate prediction_id detected: {p.prediction_id}")
            pred_ids.add(p.prediction_id)

        trade_ids = set()
        for t in trades:
            if t.trade_id in trade_ids:
                raise ValueError(f"Duplicate trade_id detected: {t.trade_id}")
            trade_ids.add(t.trade_id)

        account_ids = set()
        for a in accounts:
            if a.snapshot_id in account_ids:
                raise ValueError(f"Duplicate snapshot_id detected: {a.snapshot_id}")
            account_ids.add(a.snapshot_id)

        # 2. Foreign key integrity
        for t in trades:
            if t.prediction_id and t.prediction_id not in pred_ids:
                raise ValueError(
                    f"Foreign key violation: trade {t.trade_id} references non-existent prediction_id {t.prediction_id}"
                )
            if t.pre_snapshot_id not in account_ids:
                raise ValueError(
                    f"Foreign key violation: trade {t.trade_id} references non-existent pre_snapshot_id {t.pre_snapshot_id}"
                )
            if t.post_snapshot_id not in account_ids:
                raise ValueError(
                    f"Foreign key violation: trade {t.trade_id} references non-existent post_snapshot_id {t.post_snapshot_id}"
                )

        # 3. Finite numerical values
        for p in predictions:
            for val in [p.action_value_short, p.action_value_flat, p.action_value_long, p.action_value_margin]:
                if not math.isfinite(val):
                    raise ValueError(f"Non-finite action value in prediction {p.prediction_id}: {val}")

        for t in trades:
            for val in [t.requested_position, t.executed_position, t.requested_quantity, t.executed_quantity,
                        t.reference_price, t.fill_price, t.commission, t.realized_pnl, t.slippage]:
                if not math.isfinite(val):
                    raise ValueError(f"Non-finite value in trade {t.trade_id}: {val}")

        for a in accounts:
            for val in [a.wallet_balance, a.unrealized_pnl, a.equity, a.margin, a.available_balance,
                        a.position, a.exposure, a.leverage, a.peak_equity, a.drawdown, a.reward]:
                if not math.isfinite(val):
                    raise ValueError(f"Non-finite value in account snapshot {a.snapshot_id}: {val}")

        # 4. Account scope check
        for a in accounts:
            if a.account_scope != "independent_contract_wallet":
                raise ValueError(f"Account snapshot {a.snapshot_id} has invalid account_scope '{a.account_scope}', expected 'independent_contract_wallet'")

        # 5. Equity identity: equity == wallet_balance + unrealized_pnl
        for a in accounts:
            expected_equity = a.wallet_balance + a.unrealized_pnl
            if abs(a.equity - expected_equity) > 1e-4:
                raise ValueError(
                    f"Equity identity violation in snapshot {a.snapshot_id}: "
                    f"equity={a.equity} != wallet_balance({a.wallet_balance}) + unrealized_pnl({a.unrealized_pnl}) = {expected_equity}"
                )

        # 6. Reward reconciliation across steps per contract
        acc_by_contract: dict[str, list[AccountRecord]] = {}
        for a in accounts:
            acc_by_contract.setdefault(a.contract, []).append(a)

        for contract, c_accs in acc_by_contract.items():
            sorted_accs = sorted(c_accs, key=lambda x: x.step_index)
            # Must start with initial snapshot
            if sorted_accs[0].snapshot_type != "initial":
                raise ValueError(f"Contract {contract} account history must begin with initial snapshot")

            for i in range(1, len(sorted_accs)):
                pre = sorted_accs[i - 1]
                post = sorted_accs[i]
                expected_delta = post.equity - pre.equity
                # Reward should match equity delta (within reasonable numerical tolerance)
                if abs(post.reward - expected_delta) > 1e-4:
                    raise ValueError(
                        f"Reward reconciliation mismatch for contract {contract} at step {post.step_index}: "
                        f"reward={post.reward} vs equity delta={expected_delta} (pre={pre.equity}, post={post.equity})"
                    )

        # 7. Execution leg ordering for reversals
        trades_by_decision: dict[str, list[TradeRecord]] = {}
        for t in trades:
            trades_by_decision.setdefault(t.decision_id, []).append(t)

        for dec_id, dec_trades in trades_by_decision.items():
            if len(dec_trades) > 1:
                # Must have distinct legs starting at 0
                legs = [t.leg_index for t in dec_trades]
                if sorted(legs) != list(range(len(dec_trades))):
                    raise ValueError(f"Invalid execution leg ordering for decision {dec_id}: legs {legs}")

    def _publish_atomic(
        self,
        run_id: str,
        model_id: str,
        feature_version: str,
        parameter_version: str,
        contracts: list[str],
        predictions: list[PredictionRecord],
        trades: list[TradeRecord],
        accounts: list[AccountRecord],
        routing: list[RoutingRecord] | None = None,
    ) -> dict[str, Any]:
        # Create staging directory
        staging_dir = self.output_dir.parent / f".staging_{self.output_dir.name}_{uuid.uuid4().hex[:8]}"
        staging_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Group records by trading day
            days_set = set()
            pred_by_day: dict[str, list[PredictionRecord]] = {}
            trade_by_day: dict[str, list[TradeRecord]] = {}
            acc_by_day: dict[str, list[AccountRecord]] = {}
            routing_by_day: dict[str, list[RoutingRecord]] = {}

            for record in routing or []:
                days_set.add(record.trading_day)
                routing_by_day.setdefault(record.trading_day, []).append(record)

            for p in predictions:
                days_set.add(p.trading_day)
                pred_by_day.setdefault(p.trading_day, []).append(p)

            for t in trades:
                days_set.add(t.trading_day)
                trade_by_day.setdefault(t.trading_day, []).append(t)

            for a in accounts:
                days_set.add(a.trading_day)
                acc_by_day.setdefault(a.trading_day, []).append(a)

            sorted_days = sorted(days_set)
            partitions: dict[str, dict[str, int]] = {}

            for day in sorted_days:
                day_dir = staging_dir / str(day)
                day_dir.mkdir(parents=True, exist_ok=True)

                day_preds = sorted(pred_by_day.get(day, []), key=lambda x: (x.contract, x.step_index))
                day_trades = sorted(trade_by_day.get(day, []), key=lambda x: (x.contract, x.step_index, x.leg_index))
                day_accs = sorted(acc_by_day.get(day, []), key=lambda x: (x.contract, x.step_index))

                # Export CSVs with headers
                pred_dicts = [p.to_dict() for p in day_preds]
                pd.DataFrame(pred_dicts, columns=PREDICTION_RECORD_COLUMNS).to_csv(
                    day_dir / AuditArtifactNames.PREDICTION_RECORDS_CSV, index=False
                )

                trade_dicts = [t.to_dict() for t in day_trades]
                pd.DataFrame(trade_dicts, columns=TRADE_RECORD_COLUMNS).to_csv(
                    day_dir / AuditArtifactNames.TRADE_RECORDS_CSV, index=False
                )

                acc_dicts = [a.to_dict() for a in day_accs]
                pd.DataFrame(acc_dicts, columns=ACCOUNT_RECORD_COLUMNS).to_csv(
                    day_dir / AuditArtifactNames.ACCOUNT_RECORDS_CSV, index=False
                )

                day_routing = sorted(
                    routing_by_day.get(day, []),
                    key=lambda row: (row.contract, row.step_index),
                )
                if routing is not None:
                    route_dicts = [record.to_dict() for record in day_routing]
                    for row in route_dicts:
                        row["volatility_scores"] = json.dumps(row["volatility_scores"])
                        row["slope_scores"] = json.dumps(row["slope_scores"])
                    pd.DataFrame(
                        route_dicts,
                        columns=ROUTING_RECORD_COLUMNS,
                    ).to_csv(
                        day_dir / AuditArtifactNames.ROUTING_RECORDS_CSV,
                        index=False,
                    )

                partitions[str(day)] = {
                    "prediction_count": len(day_preds),
                    "trade_count": len(day_trades),
                    "account_count": len(day_accs),
                    **({"routing_count": len(day_routing)} if routing is not None else {}),
                }

            # Create AuditManifest
            manifest = AuditManifest(
                schema_version="1.1.0" if routing is not None else "1.0.0",
                run_id=run_id,
                model_id=model_id,
                feature_version=feature_version,
                parameter_version=parameter_version,
                source_data=str(self.market_data_dir),
                account_scope="independent_contract_wallet",
                trading_day_source=self.trading_day_source,
                contracts=sorted(contracts),
                trading_days=sorted_days,
                partitions=partitions,
                total_row_counts={
                    "prediction_records": len(predictions),
                    "trade_records": len(trades),
                    "account_records": len(accounts),
                    **({"routing_records": len(routing)} if routing is not None else {}),
                },
                validation_status="passed",
                created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            )
            manifest_dict = manifest.to_dict()
            with open(staging_dir / AuditArtifactNames.AUDIT_MANIFEST_JSON, "w", encoding="utf-8") as f:
                f.write(manifest.to_json(indent=2) + "\n")

            # Promotion to official output directory
            if self.output_dir.exists():
                shutil.rmtree(self.output_dir)
            staging_dir.rename(self.output_dir)
            return manifest_dict

        except Exception:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily Trading Audit Publisher")
    parser.add_argument("--result-dir", "--result_dir", required=True, help="Path to simulation result directory")
    parser.add_argument("--market-data-dir", "--market_data_dir", "--data-dir", "--data_dir", required=True, help="Path to market data directory")
    parser.add_argument("--output-dir", "--output_dir", required=True, help="Official audit publication output directory")
    parser.add_argument("--run-id", "--run_id", default=None, help="Explicit run ID")
    parser.add_argument("--trading-day-mapping", "--trading_day_mapping", default=None, help="Path to explicit TradingDay mapping file")
    parser.add_argument("--contracts", nargs="*", default=None, help="Explicit contracts list")
    parser.add_argument(
        "--skip-missing-input",
        action="store_true",
        help="Skip optional pipeline publication when audit-grade input is incomplete",
    )

    args = parser.parse_args()
    publisher = DailyTradingAuditPublisher(
        result_dir=args.result_dir,
        market_data_dir=args.market_data_dir,
        output_dir=args.output_dir,
        run_id=args.run_id,
        trading_day_mapping=args.trading_day_mapping,
        contracts=args.contracts,
    )
    try:
        manifest = publisher.publish()
    except ValueError as exc:
        if args.skip_missing_input and (
            str(exc).startswith("Missing audit-grade input:")
            or str(exc).startswith("Missing explicit TradingDay")
        ):
            logger.warning("Skipping daily trading audit: %s", exc)
            return
        raise
    logger.info("Successfully published audit manifest: %s", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
