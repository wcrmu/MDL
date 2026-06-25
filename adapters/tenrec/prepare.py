from __future__ import annotations

import argparse

from .adapter import prepare_tenrec


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare local Tenrec CSV files for MDL.")
    parser.add_argument("--raw-dir", required=True, help="Directory containing Tenrec CSV files.")
    parser.add_argument("--out-dir", required=True, help="Output directory for encoded splits.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest = prepare_tenrec(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        max_rows=args.max_rows,
        overwrite=args.overwrite,
    )
    print(
        "prepared Tenrec "
        f"rows={manifest['total_rows']} splits={manifest['splits']} "
        f"scenarios={manifest['scenario_names']}"
    )


if __name__ == "__main__":
    main()

