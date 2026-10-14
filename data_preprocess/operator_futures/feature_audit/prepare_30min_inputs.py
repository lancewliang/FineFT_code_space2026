"""Isolated data preparation for 30min Stage 1-6 inputs without triggering Stage 7/8 or altering shared summaries."""

import argparse
from datetime import datetime
import json
from typing import Optional
from pathlib import Path
import polars as pl


def is_30min_bar(dt: datetime) -> bool:
    """Checks if a datetime represents a 30-minute bar endpoint in Chinese futures sessions:
    - Standard 30m boundaries: minutes in [0, 30]
    - Session end boundaries: 10:15 (morning break), 11:30, 15:00 (day close), 23:00 (night close)
    """
    if dt.minute % 30 == 0:
        return True
    if dt.hour == 10 and dt.minute == 15:
        return True
    if dt.hour == 15 and dt.minute == 0:
        return True
    if dt.hour == 23 and dt.minute == 0:
        return True
    return False


def prepare_30min_data(source_10min_dir: Path, target_30min_dir: Path, summary_file: Optional[Path] = None) -> None:
    """Prepares isolated 30min dataset by downsampling 10min Stage 6 data to 30min bars."""
    target_30min_dir.mkdir(parents=True, exist_ok=True)

    for partition in ["train", "valid"]:
        src_part_dir = source_10min_dir / partition
        tgt_part_dir = target_30min_dir / partition
        tgt_part_dir.mkdir(parents=True, exist_ok=True)

        part_frames = []
        if src_part_dir.exists():
            for f in sorted(src_part_dir.glob("*.feather")):
                df = pl.read_ipc(f)
                # Parse timestamp if string
                if df.schema["timestamp"] == pl.Utf8:
                    df = df.with_columns(
                        pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
                    )

                # Filter to 30min bars
                ts_list = df["timestamp"].to_list()
                mask = [is_30min_bar(t if isinstance(t, datetime) else datetime.fromisoformat(str(t))) for t in ts_list]
                df_30 = df.filter(pl.Series(mask))

                if len(df_30) > 0:
                    df_30.write_ipc(tgt_part_dir / f.name)
                    part_frames.append(df_30)

        if part_frames:
            combined = pl.concat(part_frames)
            combined.write_ipc(target_30min_dir / f"{partition}.feather")

    # Adapt dataset_split_manifest.json
    manifest_src = source_10min_dir / "dataset_split_manifest.json"
    if manifest_src.exists():
        with open(manifest_src, "r", encoding="utf-8") as f:
            man = json.load(f)
        man["target_freq"] = "30min"
        with open(target_30min_dir / "dataset_split_manifest.json", "w", encoding="utf-8") as f:
            json.dump(man, f, indent=2)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare isolated 30min Stage 1-6 inputs")
    parser.add_argument("--source_10min_dir", required=True, help="Path to 10min split directory")
    parser.add_argument("--target_30min_dir", required=True, help="Path to 30min split target directory")
    parser.add_argument("--summary_file", help="Path to main_contract_summary.json (read-only verification)")
    args = parser.parse_args(argv)

    summary_p = Path(args.summary_file) if args.summary_file else None
    prepare_30min_data(Path(args.source_10min_dir), Path(args.target_30min_dir), summary_p)
    print("30min isolated data preparation complete.")


if __name__ == "__main__":
    main()
