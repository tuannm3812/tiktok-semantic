"""Extract the supplied TikTok hackathon data archive into the local raw data folder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tiktok_semantic.io import extract_zip, read_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract the TikTok hackathon data zip.")
    parser.add_argument("--config", default="configs/sample.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = read_config(args.config)
    raw_zip = Path(cfg["raw_zip"])
    raw_dir = Path(cfg["raw_dir"])
    extract_zip(raw_zip, raw_dir, overwrite=args.overwrite)
    print(f"Extracted {raw_zip} -> {raw_dir}")


if __name__ == "__main__":
    main()
