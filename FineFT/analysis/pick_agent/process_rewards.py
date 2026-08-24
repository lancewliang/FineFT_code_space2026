import glob
import os
import ast

import pandas as pd

BASE = "result/DiHFT/low_level/fu/30min_multi/weights_advantage_pretrain"
files = sorted(glob.glob(os.path.join(BASE, "epoch_*", "analysis_result.csv")))

frames = []
total_rows = 0
matched_rows = 0
for f in files:
    print(f)
    epoch = os.path.basename(os.path.dirname(f))
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
        # 条件: 总和>0 且 单个值>-500 且 正值占比>=50%
        return total > 100 and min_val > -500 and pos_frac >= 0.70

    mask = df["奖励总和"].apply(all_positive)
    matched_rows += int(mask.sum())
    sub = df[mask].copy()
    if not sub.empty:
        sub.insert(0, "epoch_dir", epoch)
        frames.append(sub)

out_path = os.path.join(BASE, "all_positive_reward_rows.csv")
if frames:
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(out_path, index=False)
else:
    # still create an empty file with headers from the last file read
    pd.DataFrame().to_csv(out_path, index=False)
    out = pd.DataFrame()

print(f"files: {len(files)}")
print(f"total rows: {total_rows}")
print(f"matched rows (sum>0 & each>-300 & >=50% pos): {matched_rows}")
print(f"output: {out_path}")
print(f"output rows: {len(out)}")
