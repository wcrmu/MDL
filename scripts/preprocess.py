from __future__ import annotations

import argparse

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.datasets.preprocess import validate_processed_dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a processed manifest dataset. Dataset-specific raw conversion "
            "belongs in an adapter outside the core model package."
        )
    )
    parser.add_argument("--data-dir", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_processed_dataset(args.data_dir)
    print(f"validated_processed_dataset={args.data_dir}")


if __name__ == "__main__":
    main()
