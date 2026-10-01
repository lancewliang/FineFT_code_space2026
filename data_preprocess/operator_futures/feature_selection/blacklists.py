from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_BLACKLIST_JSON_PATH = (
    Path(__file__).resolve().parent / "commodity_feature_blacklists.json"
)


def _dedup_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def load_commodity_feature_blacklists(
    config_path: Path | None = None,
) -> dict[str, dict]:
    resolved_path = (
        config_path if config_path is not None else DEFAULT_BLACKLIST_JSON_PATH
    )
    content = resolved_path.read_text(encoding="utf-8")
    return json.loads(content)


def get_commodity_global_hygiene_blacklist(
    config_path: Path | None = None,
) -> list[str]:
    data = load_commodity_feature_blacklists(config_path)
    return list(data["scopes"]["global"])


def get_commodity_stream_blacklists(
    target_freq: str = "10min",
    config_path: Path | None = None,
) -> tuple[list[str], list[str]]:
    data = load_commodity_feature_blacklists(config_path)
    global_features = list(data["scopes"]["global"])
    vae_global = list(data["scopes"]["vae"])
    rl_global = list(data["scopes"]["rl_agent"])

    freq_dict = data["frequencies"].get(target_freq, {})
    vae_freq = list(freq_dict.get("vae", []))
    rl_freq = list(freq_dict.get("rl_agent", []))

    vae_blacklist = _dedup_preserve_order(global_features + vae_global + vae_freq)
    rl_blacklist = _dedup_preserve_order(global_features + rl_global + rl_freq)
    return vae_blacklist, rl_blacklist


def get_commodity_vae_feature_blacklist(
    target_freq: str = "10min",
    config_path: Path | None = None,
) -> list[str]:
    vae_blacklist, _ = get_commodity_stream_blacklists(target_freq, config_path)
    return vae_blacklist


def get_commodity_rl_feature_blacklist(
    target_freq: str = "10min",
    config_path: Path | None = None,
) -> list[str]:
    _, rl_blacklist = get_commodity_stream_blacklists(target_freq, config_path)
    return rl_blacklist


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Query commodity feature selection blacklists by stream and frequency."
    )
    parser.add_argument(
        "--stream",
        type=str,
        choices=["vae", "rl", "rl_agent", "global"],
        required=True,
        help="Target stream or scope to query.",
    )
    parser.add_argument(
        "--target_freq",
        "--freq",
        dest="target_freq",
        type=str,
        default="10min",
        help="Target bar sampling frequency (e.g. 1min, 5min, 10min, 30min).",
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        default=None,
        help="Optional custom path to commodity_feature_blacklists.json.",
    )
    args = parser.parse_args(argv)

    if args.stream == "global":
        items = get_commodity_global_hygiene_blacklist(args.config_path)
    elif args.stream == "vae":
        items = get_commodity_vae_feature_blacklist(args.target_freq, args.config_path)
    elif args.stream in ("rl", "rl_agent"):
        items = get_commodity_rl_feature_blacklist(args.target_freq, args.config_path)
    else:
        raise ValueError(f"Unknown stream: {args.stream}")

    for item in items:
        print(item)


if __name__ == "__main__":
    main()
