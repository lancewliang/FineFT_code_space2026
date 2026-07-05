#!/usr/bin/env bash
set -euo pipefail

root="${1:-data/原始下载/燃料油}"

if [[ ! -d "$root" ]]; then
    echo "Input directory not found: $root" >&2
    exit 1
fi

find "$root" -mindepth 4 -maxdepth 4 -type f -name '*.csv' -print0 |
while IFS= read -r -d '' csv_file; do
    rel_path="${csv_file#"$root"/}"
    year="${rel_path%%/*}"
    file_name="${csv_file##*/}"
    contract="${file_name%.csv}"
    date_dir="$(basename "$(dirname "$csv_file")")"

    if [[ ! "$date_dir" =~ ^[0-9]{8}$ ]]; then
        echo "Skip unexpected date directory: $csv_file" >&2
        continue
    fi

    source_date="${date_dir:0:4}-${date_dir:4:2}-${date_dir:6:2}"
    target_date="$(date -d "$source_date +1 day" +%F)"
    target_file="$root/$year/$contract-$target_date.csv"

    cp "$csv_file" "$target_file"
    echo "$csv_file -> $target_file"
done
