from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def ensure_processed_layout(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_manifest(output_dir: str | Path, manifest: Mapping[str, Any]) -> Path:
    output_path = ensure_processed_layout(output_dir) / "manifest.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(manifest), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return output_path


def write_csv_split(
    output_dir: str | Path,
    split: str,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: list[str],
) -> Path:
    output_path = ensure_processed_layout(output_dir) / f"{split}.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def validate_processed_dataset(data_dir: str | Path) -> None:
    path = Path(data_dir)
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    for split in manifest.get("splits", []):
        split_path = path / f"{split}.csv"
        if not split_path.exists():
            raise FileNotFoundError(f"missing split csv: {split_path}")
