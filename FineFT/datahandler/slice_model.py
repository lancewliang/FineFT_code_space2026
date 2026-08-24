import os
import sys
import shutil
import json
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import warnings
import market_dynamics_modeling_analysis
import label_util as util

try:
    from .manifests import (
        SliceContractManifest,
        SliceFileManifest,
        SliceLabelManifest,
        SliceManifest,
    )
except ImportError:
    datahandler_parent = Path(__file__).resolve().parents[1]
    if str(datahandler_parent) not in sys.path:
        sys.path.insert(0, str(datahandler_parent))
    from datahandler.manifests import (
        SliceContractManifest,
        SliceFileManifest,
        SliceLabelManifest,
        SliceManifest,
    )

parser = argparse.ArgumentParser()
# replay buffer coffient
parser.add_argument(
    "--data_path",
    type=str,
    default="dataset/fu/valid.feather",
    help="the number of transcation we store in one memory",
)
parser.add_argument(
    "--valid_dir",
    "--data_dir",
    dest="valid_dir",
    type=str,
    default=None,
    help="Complete valid directory for atomic cross-contract calibration",
)
parser.add_argument(
    "--key_indicator",
    type=str,
    default="mark_price",
    help="The column name of the feature in the data that will be used for dynamic modeling",
)
parser.add_argument(
    "--timestamp",
    type=str,
    default="index",
    help="The column name of the feature in the data that is the timestamp",
)
parser.add_argument(
    "--tic",
    type=str,
    default="symbol",
    help="The column name of the feature in the data that marks the tic",
)
parser.add_argument(
    "--labeling_method",
    type=str,
    default="slope",
    help="The method that is used for dynamic labeling:quantile/slope/DTW",
)
parser.add_argument(
    "--min_length_limit",
    type=int,
    default=288,
    help="Every slice will have at least this length",
)
parser.add_argument(
    "--merging_metric",
    type=str,
    default="DTW_distance",
    help="The method that is used for slice merging",
)
parser.add_argument(
    "--merging_threshold",
    type=float,
    default=0.0003,
    help="The metric threshold that is used to decide whether a slice will be merged",
)
parser.add_argument(
    "--merging_dynamic_constraint",
    type=int,
    default=1,
    help="Neighbor segment of dynamics spans greater than this number will not be merged(setting this to $-1$ will disable the constraint)",
)
parser.add_argument(
    "--filter_strength",
    type=int,
    default=1,
    help='The strength of the low-pass Butterworth filter, the bigger the lower cutoff frequency, "1" have the cutoff frequency of min_length_limit period',
)
parser.add_argument(
    "--dynamic_number",
    type=int,
    default=5,
    help='The strength of the low-pass Butterworth filter, the bigger the lower cutoff frequency, "1" have the cutoff frequency of min_length_limit period',
)
parser.add_argument(
    "--max_length_expectation",
    type=int,
    default=864,
    help='The strength of the low-pass Butterworth filter, the bigger the lower cutoff frequency, "1" have the cutoff frequency of min_length_limit period',
)


class Linear_Market_Dynamics_Model(object):
    def __init__(self, args):
        super(Linear_Market_Dynamics_Model, self).__init__()
        self.data_path = args.data_path
        self.method = "slice_and_merge"
        self.filter_strength = args.filter_strength
        self.dynamic_number = args.dynamic_number
        self.max_length_expectation = args.max_length_expectation
        self.key_indicator = args.key_indicator
        self.timestamp = args.timestamp
        self.tic = args.tic
        self.labeling_method = args.labeling_method
        self.min_length_limit = args.min_length_limit
        self.merging_metric = args.merging_metric
        self.merging_threshold = args.merging_threshold
        self.merging_dynamic_constraint = args.merging_dynamic_constraint

    def file_extension_selector(self, read):
        if self.data_path.endswith(".csv"):
            if read:
                return pd.read_csv
            else:
                return pd.DataFrame.to_feather
        elif self.data_path.endswith(".feather"):
            if read:
                return pd.read_feather
            else:
                return pd.DataFrame.to_feather
        else:
            raise ValueError("invalid file extension")

    def wirte_data_as_segments(self, data, process_datafile_path):
        # get file name and extension from process_datafile_path
        file_name, file_extension = os.path.splitext(process_datafile_path)

    def prepare_raw_data(self, raw_data):
        print(f"loaded columns: {list(raw_data.columns)}")
        required_columns = ["bid1_price"]
        if self.timestamp != "index":
            required_columns.append(self.timestamp)
        missing_columns = [
            column for column in required_columns if column not in raw_data.columns
        ]
        if missing_columns:
            raise ValueError(
                f"{self.data_path} missing required columns: {missing_columns}"
            )

        raw_data[self.tic] = raw_data["symbol"]
        raw_data[self.key_indicator] = raw_data["bid1_price"]
        if self.timestamp == "index":
            raw_data[self.timestamp] = raw_data.index


        return raw_data

    def _contract_name(self):
        stem = Path(self.data_path).stem
        if stem.startswith("df_"):
            return stem[3:]
        return stem

    def _load_slice_manifest(self, manifest_path, valid_root):
        if manifest_path.exists():
            return SliceManifest.from_dict(
                json.loads(manifest_path.read_text(encoding="utf-8"))
            )
        return SliceManifest.new(valid_root)

    def _write_slice_manifest(
        self,
        manifest_path,
        valid_root,
        contract_name,
        processed_path,
        contract_labels,
    ):
        manifest = self._load_slice_manifest(manifest_path, valid_root)
        contract_file_count = sum(
            label_info.file_count for label_info in contract_labels.values()
        )
        contract_total_rows = sum(
            label_info.total_row_count for label_info in contract_labels.values()
        )
        manifest.replace_contract(
            SliceContractManifest(
                contract=contract_name,
                processed_path=str(processed_path),
                file_count=contract_file_count,
                total_row_count=contract_total_rows,
                labels=contract_labels,
            )
        )
        self._write_manifest(manifest_path, manifest)

    def _write_manifest(self, manifest_path, manifest):
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _write_skip_manifest(
        self,
        manifest_path,
        valid_root,
        contract_name,
        processed_path,
        reason,
        input_row_count,
    ):
        manifest = self._load_slice_manifest(manifest_path, valid_root)
        manifest.record_skipped_contract(
            contract=contract_name,
            processed_path=str(processed_path),
            reason=reason,
            input_row_count=input_row_count,
        )
        self._write_manifest(manifest_path, manifest)

    def _filter_padlen(self):
        return 15

    def run(self):
        print("labeling start")
        input_path = Path(self.data_path).resolve()
        ticker_name_path = input_path.parent
        contract_name = self._contract_name()
        output_path = self.data_path
        raw_data = pd.read_feather(self.data_path)
        raw_data = self.prepare_raw_data(raw_data)

        processed_dir = ticker_name_path / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        process_data_path = processed_dir / f"valid_processed_{contract_name}.feather"
        raw_data.to_feather(process_data_path)
        self.data_path = str(process_data_path)
        output_root = ticker_name_path / contract_name
        filter_padlen = self._filter_padlen()
        if len(raw_data) <= filter_padlen:
            if output_root.exists():
                shutil.rmtree(output_root)
            reason = (
                f"insufficient rows for dynamic slicing: "
                f"{len(raw_data)} <= filter padlen {filter_padlen}"
            )
            print(f"skip {contract_name}: {reason}")
            self._write_skip_manifest(
                ticker_name_path / "slice_manifest.json",
                ticker_name_path,
                contract_name,
                process_data_path,
                reason,
                len(raw_data),
            )
            return

        worker = util.Worker(
            self.data_path,
            "slice_and_merge",
            filter_strength=self.filter_strength,
            key_indicator=self.key_indicator,
            timestamp=self.timestamp,
            tic=self.tic,
            labeling_method=self.labeling_method,
            min_length_limit=self.min_length_limit,
            merging_threshold=self.merging_threshold,
            merging_metric=self.merging_metric,
            merging_dynamic_constraint=self.merging_dynamic_constraint,
        )
        print("start fitting")
        worker.fit(
            self.dynamic_number, self.max_length_expectation, self.min_length_limit
        )
        print("finish fitting")
        worker.label(os.path.dirname(self.data_path))
        labeled_data = pd.concat([v for v in worker.data_dict.values()], axis=0)
        flie_reader = self.file_extension_selector(read=True)
        extension = self.data_path.split(".")[-1]
        data = flie_reader(self.data_path)
        if self.tic in data.columns:
            merge_keys = [self.timestamp, self.tic, self.key_indicator]
        else:
            merge_keys = [self.timestamp, self.key_indicator]
        merged_data = data.merge(
            labeled_data, how="left", on=merge_keys, suffixes=("", "_DROP")
        ).filter(regex="^(?!.*_DROP)")
        if self.labeling_method == "slope":
            self.model_id = f"slice_and_merge_model_{self.dynamic_number}dynamics_minlength{self.min_length_limit}_{self.labeling_method}_labeling_slope"
        else:
            self.model_id = f"slice_and_merge_model_{self.dynamic_number}dynamics_minlength{self.min_length_limit}_{self.labeling_method}_labeling"

        process_datafile_path = (
            os.path.splitext(output_path)[0]
            + "_labeled_"
            + self.model_id
            + "."
            + extension
        )
        # if extension == "csv":
        #     merged_data.to_csv(process_datafile_path, index=False)
        # elif extension == "feather":
        #     merged_data.to_feather(process_datafile_path)
        print("labeling done")
        total_label_count = self.dynamic_number

        if output_root.exists():
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        for i in range(total_label_count):
            (output_root / "label_{}".format(i)).mkdir(parents=True, exist_ok=True)
        previous_label = int(merged_data.label.iloc[0])
        previous_start = 0
        label_counter = [0] * total_label_count
        contract_labels = {}

        def write_segment(label, start, end):
            if end <= start:
                return
            segment = merged_data.iloc[start:end].reset_index(drop=True)
            label_name = "label_{}".format(label)
            output_file = (
                output_root
                / label_name
                / "df_{}.feather".format(label_counter[label])
            )
            segment.to_feather(output_file)
            label_info = contract_labels.setdefault(
                label_name,
                SliceLabelManifest(label=label_name),
            )
            output_row_count = int(len(segment))
            label_info.file_count += 1
            label_info.total_row_count += output_row_count
            label_info.files.append(
                SliceFileManifest(
                    path=str(output_file),
                    output_row_count=output_row_count,
                )
            )
            label_counter[label] += 1

        for i in range(len(merged_data)):
            current_label = int(merged_data.label.iloc[i])
            if current_label != previous_label:
                write_segment(previous_label, previous_start, i)
                previous_start = i
                previous_label = current_label
        write_segment(previous_label, previous_start, len(merged_data))
        self._write_slice_manifest(
            ticker_name_path / "slice_manifest.json",
            ticker_name_path,
            contract_name,
            process_data_path,
            dict(sorted(contract_labels.items())),
        )
                
        # print("plotting start")
        # # a list the path to all the modeling visulizations
        # market_dynamic_labeling_visualization_paths = worker.plot(
        #     worker.tics, self.slope_interval, output_path, self.model_id
        # )
        # print("plotting done")
        # # if self.OE_BTC == True:
        # #     os.remove('./temp/OE_BTC_processed.csv')

        # # MDM analysis
        # MDM_analysis = market_dynamics_modeling_analysis.MarketDynamicsModelingAnalysis(
        #     process_datafile_path, self.key_indicator
        # )
        # MDM_analysis.run_analysis(process_datafile_path)
        # print("Market dynamics modeling analysis done")

        # return (
        #     os.path.abspath(process_datafile_path),
        #     market_dynamic_labeling_visualization_paths,
        # )



if __name__ == "__main__":
    args = parser.parse_args()
    if args.valid_dir is not None:
        try:
            from .valid_cross_contract_label_calibration import build_valid_dataset
        except ImportError:
            from valid_cross_contract_label_calibration import build_valid_dataset

        build_valid_dataset(
            args.valid_dir,
            dynamic_number=args.dynamic_number,
            labeling_method=args.labeling_method,
            timestamp=args.timestamp,
            tic=args.tic,
            filter_strength=args.filter_strength,
            min_length_limit=args.min_length_limit,
            merging_threshold=args.merging_threshold,
            merging_metric=args.merging_metric,
            merging_dynamic_constraint=args.merging_dynamic_constraint,
            max_length_expectation=args.max_length_expectation,
        )
    else:
        raise SystemExit(
            "single-contract --data_path mode is diagnostic-only and cannot publish "
            "official labels; pass --valid_dir for production calibration"
        )
