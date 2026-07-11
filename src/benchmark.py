from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .config import AppConfig
from .dataloader import scan_flat_table_stats
from .train import TrainResult, train_mdl


@dataclass(frozen=True)
class SplitBenchmark:
    split: str
    files: int
    record_batches: int
    input_rows: int
    flat_rows: int
    elapsed_seconds: float

    @property
    def rows_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.flat_rows / self.elapsed_seconds


@dataclass(frozen=True)
class TrainingBenchmark:
    steps: int
    rows: int
    last_loss: float
    elapsed_seconds: float

    @property
    def steps_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.steps / self.elapsed_seconds

    @property
    def rows_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.rows / self.elapsed_seconds


def benchmark_split(config: AppConfig, split_name: str, max_batches: int | None = None) -> SplitBenchmark:
    start = perf_counter()
    stats = scan_flat_table_stats(config, split_name, max_batches=max_batches)
    elapsed = perf_counter() - start
    return SplitBenchmark(
        split=split_name,
        files=stats.files,
        record_batches=stats.raw_record_batches,
        input_rows=stats.raw_rows,
        flat_rows=stats.flat_rows,
        elapsed_seconds=elapsed,
    )


def benchmark_training(config: AppConfig, max_steps: int) -> TrainingBenchmark:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    result: TrainResult = train_mdl(
        config,
        max_steps=max_steps,
        save_checkpoint=False,
        log_steps=False,
    )
    return TrainingBenchmark(
        steps=result.steps,
        rows=result.rows,
        last_loss=result.last_loss,
        elapsed_seconds=result.elapsed_seconds,
    )
