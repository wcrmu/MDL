from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .config import AppConfig
from .data import AggParquetScanner, ParquetScanner, required_columns_for_split


@dataclass(frozen=True)
class SplitBenchmark:
    split: str
    files: int
    record_batches: int
    input_rows: int
    candidate_rows: int | None
    elapsed_seconds: float

    @property
    def rows_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.input_rows / self.elapsed_seconds

    @property
    def candidates_per_second(self) -> float | None:
        if self.candidate_rows is None or self.elapsed_seconds <= 0:
            return None
        return self.candidate_rows / self.elapsed_seconds


def benchmark_split(config: AppConfig, split_name: str, max_batches: int | None = None) -> SplitBenchmark:
    split = config.data.train if split_name == "train" else config.data.test
    if split is None:
        raise ValueError(f"split {split_name!r} is not configured")
    columns = required_columns_for_split(config, split)
    scanner: ParquetScanner
    scanner = AggParquetScanner(split, columns) if split.format == "agg_parquet" else ParquetScanner(split, columns)

    start = perf_counter()
    record_batches = 0
    input_rows = 0
    candidate_rows = 0 if split.format == "agg_parquet" else None

    if isinstance(scanner, AggParquetScanner):
        for table in scanner.iter_tables():
            if max_batches is not None and record_batches >= max_batches:
                break
            record_batches += 1
            input_rows += table.num_rows
            decoded = scanner.decoder.decode(table)
            candidate_rows = int(candidate_rows or 0) + decoded.num_rows
    else:
        for batch in scanner.iter_record_batches():
            if max_batches is not None and record_batches >= max_batches:
                break
            record_batches += 1
            input_rows += batch.num_rows

    elapsed = perf_counter() - start
    return SplitBenchmark(
        split=split_name,
        files=len(scanner.paths),
        record_batches=record_batches,
        input_rows=input_rows,
        candidate_rows=candidate_rows,
        elapsed_seconds=elapsed,
    )
