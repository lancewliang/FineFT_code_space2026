import argparse
import ast
import glob
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process and filter reward analysis CSV files by label_type."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="result/DiHFT/low_level/fu/30min_multi/weights_advantage_pretrain",
        help="Base path containing epoch results",
    )
    parser.add_argument(
        "--label_type",
        type=str,
        choices=["slope", "volatility", "all"],
        default="volatility",
        help="Filter by label_type (slope, volatility, or all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: all_positive_reward_rows_<label_type>.csv)",
    )
    return parser.parse_args()


def get_target_files(base_dir, label_type):
    if label_type == "all":
        nested = sorted(glob.glob(os.path.join(base_dir, "epoch_*", "*", "analysis_result.csv")))
        direct = sorted(glob.glob(os.path.join(base_dir, "epoch_*", "analysis_result.csv")))
        files = sorted(set(nested + direct))
    else:
        files = sorted(glob.glob(os.path.join(base_dir, "epoch_*", label_type, "analysis_result.csv")))
        if not files:
            files = sorted(glob.glob(os.path.join(base_dir, "epoch_*", "analysis_result.csv")))
    return files


def extract_metadata(file_path):
    dir_name = os.path.dirname(file_path)
    parent_name = os.path.basename(dir_name)
    grandparent_name = os.path.basename(os.path.dirname(dir_name))
    if parent_name.startswith("epoch_"):
        return parent_name, ""
    return grandparent_name, parent_name


def main():
    args = parse_args()
    files = get_target_files(args.base_dir, args.label_type)

    frames = []
    total_rows = 0
    matched_rows = 0

    for f in files:
        print(f)
        epoch, label_type_extracted = extract_metadata(f)
        df = pd.read_csv(f)
        total_rows += len(df)

        def all_positive(s):
            try:
                vals = ast.literal_eval(str(s))
            except (ValueError, SyntaxError):
                return False
            if len(vals) == 0:
                return False
            total = sum(vals)
            min_val = min(vals)
            pos_frac = sum(1 for v in vals if v >= 0) / len(vals)
            # 条件: 总和>100 且 单个值>-500 且 正值占比>=70%
            return total > 300 and min_val > -600 and pos_frac >= 0.60

        mask = df["奖励总和"].apply(all_positive)
        matched_rows += int(mask.sum())
        sub = df[mask].copy()
        if not sub.empty:
            if label_type_extracted:
                sub.insert(0, "label_type", label_type_extracted)
            sub.insert(0, "epoch_dir", epoch)
            frames.append(sub)

    if args.output:
        out_path = args.output
    elif args.label_type == "all":
        out_path = os.path.join(args.base_dir, "all_positive_reward_rows.csv")
    else:
        out_path = os.path.join(args.base_dir, f"all_positive_reward_rows_{args.label_type}.csv")

    if frames:
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(out_path, index=False)
    else:
        pd.DataFrame().to_csv(out_path, index=False)
        out = pd.DataFrame()

    print(f"filter label_type: {args.label_type}")
    print(f"files: {len(files)}")
    print(f"total rows: {total_rows}")
    print(f"matched rows (sum>100 & min>-500 & >=70% pos): {matched_rows}")
    print(f"output: {out_path}")
    print(f"output rows: {len(out)}")


if __name__ == "__main__":
    main()
