from __future__ import annotations

import csv
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

TASK_NAMES = ["click", "like", "share", "follow", "read", "favorite"]
CATEGORICAL_COLUMNS = [
    "user_id",
    "item_id",
    "user_gender",
    "user_age",
    "video_category",
    "category_second",
    "category_first",
]
NUMERIC_COLUMNS = [
    "watching_times",
    "click_count",
    "like_count",
    "comment_count",
    "exposure_count",
    "read_percentage",
    "item_score1",
    "item_score2",
    "item_score3",
    "read_time",
]

VIDEO_COLUMNS = [
    "user_id",
    "item_id",
    "click",
    "like",
    "share",
    "follow",
    "video_category",
    "watching_times",
    "user_gender",
    "user_age",
]
ARTICLE_COLUMNS = [
    "user_id",
    "item_id",
    "click",
    "like",
    "share",
    "follow",
    "read",
    "favorite",
    "click_count",
    "like_count",
    "comment_count",
    "exposure_count",
    "read_percentage",
    "category_second",
    "category_first",
    "item_score1",
    "item_score2",
    "item_score3",
    "read_time",
]


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    aliases: tuple[str, ...]
    columns: tuple[str, ...]


SCENARIOS = [
    ScenarioSpec("QK-video", ("qkvideo", "qk-video", "qk_video"), tuple(VIDEO_COLUMNS)),
    ScenarioSpec("QK-article", ("qkarticle", "qk-article", "qk_article"), tuple(ARTICLE_COLUMNS)),
    ScenarioSpec("QB-video", ("qbvideo", "qb-video", "qb_video"), tuple(VIDEO_COLUMNS)),
    ScenarioSpec("QB-article", ("qbarticle", "qb-article", "qb_article"), tuple(ARTICLE_COLUMNS)),
]


def build_tenrec_feature_specs(vocabs: dict[str, dict[str, int]]) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for column in CATEGORICAL_COLUMNS:
        specs.append(
            {
                "name": column,
                "encoder": "categorical_embedding",
                "vocab_size": len(vocabs[column]) + 1,
                "source": {"type": "csv_column", "column": column},
            }
        )
    for column in NUMERIC_COLUMNS:
        specs.append(
            {
                "name": column,
                "encoder": "numeric_value",
                "dim": 1,
                "source": {"type": "csv_column", "column": column},
            }
        )
    return specs


def build_tenrec_token_specs(num_tokens: int = 4) -> list[dict[str, object]]:
    inputs = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
    token_count = max(1, min(num_tokens, len(inputs)))
    chunk_size = math.ceil(len(inputs) / token_count)
    specs: list[dict[str, object]] = []
    for token_id in range(token_count):
        chunk = inputs[token_id * chunk_size : (token_id + 1) * chunk_size]
        if chunk:
            specs.append({"token_id": token_id, "projection": "linear", "inputs": chunk})
    return specs


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def detect_scenario(path: Path) -> ScenarioSpec | None:
    normalized = normalize_name(path.stem)
    for spec in SCENARIOS:
        if any(normalize_name(alias) in normalized for alias in spec.aliases):
            return spec
    return None


def discover_tenrec_files(raw_dir: str | Path) -> list[tuple[ScenarioSpec, Path]]:
    files: list[tuple[ScenarioSpec, Path]] = []
    for path in sorted(Path(raw_dir).rglob("*.csv")):
        spec = detect_scenario(path)
        if spec is not None:
            files.append((spec, path))
    if not files:
        raise FileNotFoundError(
            "no Tenrec CSV files found; expected names containing "
            "QK-video, QK-article, QB-video, or QB-article"
        )
    return files


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def has_header(path: Path, expected_columns: tuple[str, ...]) -> bool:
    with path.open("r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline()
    lowered = [part.strip().lower() for part in first_line.split(",")]
    return "user_id" in lowered or "userid" in lowered or any(
        column in lowered for column in expected_columns[:4]
    )


def iter_rows(path: Path, spec: ScenarioSpec) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        if has_header(path, spec.columns):
            reader = csv.DictReader(handle)
        else:
            reader = csv.DictReader(handle, fieldnames=list(spec.columns))
        for row in reader:
            yield {key.strip(): (value.strip() if value is not None else "") for key, value in row.items()}


def update_numeric_stats(stats: dict[str, dict[str, float]], row: dict[str, str]) -> None:
    for column in NUMERIC_COLUMNS:
        value = parse_float(row.get(column))
        if value is None:
            continue
        stats[column]["count"] += 1.0
        stats[column]["sum"] += value
        stats[column]["sumsq"] += value * value


def add_vocab_values(vocabs: dict[str, dict[str, int]], row: dict[str, str]) -> None:
    for column in CATEGORICAL_COLUMNS:
        value = row.get(column, "")
        if value == "":
            continue
        vocab = vocabs[column]
        if value not in vocab:
            vocab[value] = len(vocab) + 1


def numeric_mean_std(stats: dict[str, float]) -> tuple[float, float]:
    count = stats["count"]
    if count <= 0:
        return 0.0, 1.0
    mean = stats["sum"] / count
    variance = max(stats["sumsq"] / count - mean * mean, 0.0)
    std = math.sqrt(variance)
    return mean, std if std > 1e-12 else 1.0


def split_for_index(index: int, total_rows: int) -> str:
    train_end = int(total_rows * 0.8)
    val_end = int(total_rows * 0.9)
    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def label_value(row: dict[str, str], task: str) -> tuple[int, int]:
    if task not in row or row[task] == "":
        return 0, 0
    value = parse_float(row.get(task))
    if value is None:
        return 0, 0
    return int(value > 0), 1


def prepare_tenrec(
    raw_dir: str | Path,
    out_dir: str | Path,
    max_rows: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    raw_files = discover_tenrec_files(raw_dir)
    out_path = Path(out_dir)
    if out_path.exists():
        has_existing_files = any(out_path.iterdir())
        if has_existing_files:
            if not overwrite:
                raise FileExistsError(
                    f"{out_path} already exists; pass overwrite=True to replace it"
                )
            shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    vocabs = {column: {} for column in CATEGORICAL_COLUMNS}
    numeric_stats = {
        column: {"count": 0.0, "sum": 0.0, "sumsq": 0.0} for column in NUMERIC_COLUMNS
    }
    scenario_names = sorted({spec.name for spec, _ in raw_files})
    scenario_to_id = {name: index for index, name in enumerate(scenario_names)}

    total_rows = 0
    for spec, path in raw_files:
        for row in iter_rows(path, spec):
            add_vocab_values(vocabs, row)
            update_numeric_stats(numeric_stats, row)
            total_rows += 1
            if max_rows is not None and total_rows >= max_rows:
                break
        if max_rows is not None and total_rows >= max_rows:
            break

    numeric_means: dict[str, float] = {}
    numeric_stds: dict[str, float] = {}
    for column, stats in numeric_stats.items():
        mean, std = numeric_mean_std(stats)
        numeric_means[column] = mean
        numeric_stds[column] = std

    label_columns = {task: f"label_{task}" for task in TASK_NAMES}
    label_mask_columns = {task: f"mask_{task}" for task in TASK_NAMES}
    header = (
        ["scenario_id", "group_id"]
        + CATEGORICAL_COLUMNS
        + NUMERIC_COLUMNS
        + [label_columns[task] for task in TASK_NAMES]
        + [label_mask_columns[task] for task in TASK_NAMES]
    )
    split_counts = {"train": 0, "val": 0, "test": 0}
    handles = {
        split: (out_path / f"{split}.csv").open("w", encoding="utf-8", newline="")
        for split in split_counts
    }
    writers = {split: csv.DictWriter(handle, fieldnames=header) for split, handle in handles.items()}
    for writer in writers.values():
        writer.writeheader()

    row_index = 0
    try:
        for spec, path in raw_files:
            for row in iter_rows(path, spec):
                split = split_for_index(row_index, total_rows)
                output: dict[str, int | float | str] = {
                    "scenario_id": scenario_to_id[spec.name],
                    "group_id": str(vocabs["user_id"].get(row.get("user_id", ""), 0)),
                }
                for column in CATEGORICAL_COLUMNS:
                    output[column] = vocabs[column].get(row.get(column, ""), 0)
                for column in NUMERIC_COLUMNS:
                    raw_value = parse_float(row.get(column))
                    if raw_value is None:
                        output[column] = 0.0
                    else:
                        output[column] = (raw_value - numeric_means[column]) / numeric_stds[column]
                for task in TASK_NAMES:
                    label, mask = label_value(row, task)
                    output[label_columns[task]] = label
                    output[label_mask_columns[task]] = mask
                writers[split].writerow(output)
                split_counts[split] += 1
                row_index += 1
                if max_rows is not None and row_index >= max_rows:
                    break
            if max_rows is not None and row_index >= max_rows:
                break
    finally:
        for handle in handles.values():
            handle.close()

    manifest: dict[str, object] = {
        "dataset": "tenrec",
        "raw_files": [{"scenario": spec.name, "path": str(path)} for spec, path in raw_files],
        "scenario_names": scenario_names,
        "task_names": TASK_NAMES,
        "data_columns": {
            "scenario_id": "scenario_id",
            "group_id": "group_id",
            "labels": label_columns,
            "label_masks": label_mask_columns,
        },
        "tokenization": {
            "version": 2,
            "kind": "encoder_registry",
            "features": build_tenrec_feature_specs(vocabs),
            "token_specs": build_tenrec_token_specs(),
        },
        "numeric_means": numeric_means,
        "numeric_stds": numeric_stds,
        "splits": split_counts,
        "total_rows": total_rows,
        "group_id": "encoded user_id",
    }
    with (out_path / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    for column, vocab in vocabs.items():
        with (out_path / f"vocab__{column}.json").open("w", encoding="utf-8") as handle:
            json.dump(vocab, handle, indent=2, sort_keys=True)

    return manifest

