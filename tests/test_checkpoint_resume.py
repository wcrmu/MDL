"""Tests for resumable checkpoints: storage, commit protocol, and data cursors."""

from __future__ import annotations

from dataclasses import replace
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

# This module spawns children, and they inherit this. On a conda numpy
# (mkl-service) plus pip torch (libgomp) mix the child aborts at import with
# "MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1" depending on
# which module imported first.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

from src.checkpoint import (
    CHECKPOINT_MANIFEST,
    COMMIT_MARKER,
    CheckpointUploader,
    CommittedCheckpoint,
    DataCursor,
    LATEST_POINTER,
    StagedCheckpoint,
    fetch_checkpoint_for_rank,
    latest_committed_checkpoint,
    list_committed_checkpoints,
    load_training_checkpoint,
    parse_step_directory,
    prune_run_directory,
    rank_progress_file,
    rank_ready_marker,
    resolve_resume_checkpoint,
    stage_training_checkpoint,
    step_directory_name,
)
from src.checkpoint_store import (
    LocalCheckpointStore,
    is_remote_uri,
    open_checkpoint_store,
)
from src.config import (
    CheckpointConfig,
    ParquetSplitConfig,
    ReaderConfig,
    TrainingConfig,
    load_app_config,
)
from src import dataloader as dataloader_module
from src.dataloader import (
    ParquetScanner,
    ScanCursorChannel,
    ScanPosition,
    ScanResumePlan,
    scan_prefix_digest,
    scan_resume_rewind,
    scan_split_key,
    scan_work_item_key,
    set_scan_cursor_channel,
    set_scan_resume_plan,
)
from src.embeddings import EmbeddingTableSpec, ShardedEmbedding, plan_embedding_shards
from src.main import build_arg_parser
from src.optim import ShardedRowWiseAdagrad
from src.train import (
    DistributedContext,
    _CheckpointCoordinator,
    iter_feature_batches,
)


def _reference_config():
    root = Path(__file__).resolve().parents[1]
    return load_app_config(root / "configs" / "reference" / "default.yaml")


def _publish_from_child(
    storage, work_unit: str, position: int, digest: str | None
) -> None:
    """Publish one position the way a host-prepare reader child does."""

    channel = ScanCursorChannel(storage)
    set_scan_cursor_channel(channel)
    channel.publish(ScanPosition(work_unit, position, digest))


class _DenseToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(3, 2)


class _ShardedToyModel(nn.Module):
    def __init__(self, world_size: int = 1) -> None:
        super().__init__()
        table = EmbeddingTableSpec("item", 12, 4)
        plan = plan_embedding_shards(
            [table],
            world_size=world_size,
            strategy="row_wise",
            table_wise_max_rows=4,
        )
        self.embedding = ShardedEmbedding(
            table.num_embeddings,
            table.embedding_dim,
            table_name=table.name,
            shard_spec=plan.tables[table.name],
        )
        self.dense = nn.Linear(4, 2)


class CheckpointStoreTest(unittest.TestCase):
    def test_local_store_round_trips_bytes_json_and_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LocalCheckpointStore(temporary).child("run-a")
            store.makedirs()
            store.write_json({"step": 7}, "step-000000007", CHECKPOINT_MANIFEST)
            self.assertEqual(
                store.read_json("step-000000007", CHECKPOINT_MANIFEST)["step"], 7
            )
            self.assertTrue(store.exists("step-000000007", CHECKPOINT_MANIFEST))
            self.assertFalse(store.exists("step-000000007", COMMIT_MARKER))

            source = Path(temporary) / "payload.bin"
            source.write_bytes(b"weights")
            store.upload_file(source, "step-000000007", "model", "dense.pt")
            destination = Path(temporary) / "fetched.bin"
            store.download_file(destination, "step-000000007", "model", "dense.pt")
            self.assertEqual(destination.read_bytes(), b"weights")

            names = {entry.name for entry in store.list_entries("step-000000007")}
            self.assertEqual(names, {CHECKPOINT_MANIFEST, "model"})

            store.remove_tree("step-000000007")
            self.assertFalse(store.exists("step-000000007"))

    def test_missing_directory_lists_empty_instead_of_raising(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LocalCheckpointStore(temporary)
            self.assertEqual(store.list_entries("absent"), [])

    def test_uri_scheme_selects_the_store(self) -> None:
        self.assertTrue(is_remote_uri("hdfs://ns/apps/run"))
        self.assertTrue(is_remote_uri("viewfs://cluster/apps/run"))
        self.assertFalse(is_remote_uri("/tmp/run"))
        with tempfile.TemporaryDirectory() as temporary:
            store = open_checkpoint_store(f"file://{temporary}")
            self.assertIsInstance(store, LocalCheckpointStore)
            self.assertFalse(store.is_remote)
        with self.assertRaises(ValueError):
            open_checkpoint_store("s3://bucket/run")


class RunDirectoryTest(unittest.TestCase):
    @staticmethod
    def _write_step(store, step: int, *, committed: bool) -> None:
        directory = step_directory_name(step)
        store.write_json({"step": step, "world_size": 1}, directory, CHECKPOINT_MANIFEST)
        if committed:
            store.write_json({"step": step}, directory, COMMIT_MARKER)

    def test_step_directory_names_round_trip(self) -> None:
        self.assertEqual(step_directory_name(12000), "step-000012000")
        self.assertEqual(parse_step_directory("step-000012000"), 12000)
        self.assertIsNone(parse_step_directory("_latest.json"))
        self.assertIsNone(parse_step_directory("step-latest"))

    def test_only_committed_steps_are_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LocalCheckpointStore(temporary)
            self._write_step(store, 100, committed=True)
            self._write_step(store, 200, committed=True)
            # An interrupted upload leaves files but no commit marker.
            self._write_step(store, 300, committed=False)

            steps = [item.step for item in list_committed_checkpoints(store)]
            self.assertEqual(steps, [100, 200])
            latest = latest_committed_checkpoint(store)
            self.assertIsNotNone(latest)
            self.assertEqual(latest.step, 200)
            self.assertEqual(resolve_resume_checkpoint(store, "auto").step, 200)
            self.assertEqual(resolve_resume_checkpoint(store, "100").step, 100)
            self.assertIsNone(resolve_resume_checkpoint(store, "none"))
            with self.assertRaises(FileNotFoundError):
                resolve_resume_checkpoint(store, "300")

    def test_retention_removes_old_and_stale_partial_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LocalCheckpointStore(temporary)
            for step in (100, 200, 300):
                self._write_step(store, step, committed=True)
            self._write_step(store, 150, committed=False)
            self._write_step(store, 400, committed=False)

            removed = set(prune_run_directory(store, keep_last=2))

            self.assertEqual(
                removed,
                {step_directory_name(100), step_directory_name(150)},
            )
            self.assertEqual(
                [item.step for item in list_committed_checkpoints(store)], [200, 300]
            )
            # A partial newer than the newest commit may still be uploading.
            self.assertTrue(store.exists(step_directory_name(400)))


class MultiRankCommitTest(unittest.TestCase):
    """Rank 0 may only commit a step once every peer's files have landed."""

    @staticmethod
    def _staged(directory: Path, step: int, rank: int) -> StagedCheckpoint:
        directory.mkdir(parents=True, exist_ok=True)
        name = f"rank-{rank:05d}.pt"
        (directory / name).write_bytes(b"shard")
        return StagedCheckpoint(
            step=step,
            staging_dir=directory,
            relative_files=(name,),
            cleanup_staging=False,
        )

    def test_commit_waits_for_every_rank(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LocalCheckpointStore(temporary)
            step_dir = step_directory_name(7)

            peer = CheckpointUploader(store, rank=1, world_size=2, asynchronous=False)
            peer.submit(self._staged(Path(temporary) / "stage-1", 7, 1))
            peer.close()
            # A peer's own upload never commits the step.
            self.assertTrue(store.exists(step_dir, rank_ready_marker(1)))
            self.assertFalse(store.exists(step_dir, COMMIT_MARKER))
            self.assertEqual(list_committed_checkpoints(store), [])

            leader = CheckpointUploader(store, rank=0, world_size=2, asynchronous=False)
            leader.submit(self._staged(Path(temporary) / "stage-0", 7, 0))
            leader.close()

            self.assertTrue(store.exists(step_dir, COMMIT_MARKER))
            self.assertEqual(
                [item.step for item in list_committed_checkpoints(store)], [7]
            )
            self.assertEqual(leader.published_steps, [7])

    def test_a_missing_rank_leaves_the_step_uncommitted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LocalCheckpointStore(temporary)
            leader = CheckpointUploader(
                store,
                rank=0,
                world_size=2,
                asynchronous=False,
                ready_timeout_sec=0.2,
                poll_interval_sec=0.05,
            )
            leader.submit(self._staged(Path(temporary) / "stage-0", 9, 0))
            leader.close()

            self.assertFalse(store.exists(step_directory_name(9), COMMIT_MARKER))
            self.assertEqual(leader.failed_steps, [9])
            self.assertEqual(list_committed_checkpoints(store), [])


class TrainingCheckpointRoundTripTest(unittest.TestCase):
    def _save(self, temporary: str, *, model, optimizer, step: int, rows: int):
        config = _reference_config()
        store = LocalCheckpointStore(temporary).child("run")
        store.makedirs()
        staged = stage_training_checkpoint(
            config,
            model,
            Path(store.uri(step_directory_name(step))),
            step=step,
            rows=rows,
            dense_optimizer=optimizer,
            data_cursor=DataCursor(
                work_unit="file",
                position=42,
                prefix_digest="digest-42",
                split_key="split-key",
            ),
            cleanup_staging=False,
        )
        uploader = CheckpointUploader(store, keep_last=3, asynchronous=False)
        uploader.submit(staged)
        uploader.close()
        return config, store

    def test_step_optimizer_and_cursor_survive_a_restart(self) -> None:
        torch.manual_seed(0)
        model = _DenseToyModel()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.1)
        model.dense.weight.grad = torch.full_like(model.dense.weight, 0.5)
        model.dense.bias.grad = torch.full_like(model.dense.bias, 0.25)
        optimizer.step()
        saved_weight = model.dense.weight.detach().clone()
        saved_square_avg = optimizer.state[model.dense.weight]["square_avg"].clone()

        with tempfile.TemporaryDirectory() as temporary:
            config, store = self._save(
                temporary, model=model, optimizer=optimizer, step=2000, rows=4096
            )

            self.assertTrue(store.exists(step_directory_name(2000), COMMIT_MARKER))
            self.assertEqual(store.read_json(LATEST_POINTER)["step"], 2000)

            restored_model = _DenseToyModel()
            restored_optimizer = torch.optim.RMSprop(
                restored_model.parameters(), lr=0.1
            )
            local_dir = Path(temporary) / "fetched"
            fetch_checkpoint_for_rank(
                store,
                CommittedCheckpoint(
                    step=2000,
                    directory=step_directory_name(2000),
                    uri=store.uri(step_directory_name(2000)),
                ),
                local_dir,
            )
            resumed = load_training_checkpoint(
                config,
                restored_model,
                local_dir,
                device=torch.device("cpu"),
                dense_optimizer=restored_optimizer,
            )

        self.assertEqual(resumed.step, 2000)
        self.assertEqual(resumed.rows, 4096)
        self.assertEqual(resumed.data_cursor.position, 42)
        self.assertEqual(resumed.data_cursor.work_unit, "file")
        self.assertEqual(resumed.data_cursor.prefix_digest, "digest-42")
        torch.testing.assert_close(restored_model.dense.weight, saved_weight)
        torch.testing.assert_close(
            restored_optimizer.state[restored_model.dense.weight]["square_avg"],
            saved_square_avg,
        )

    def test_cursor_is_dropped_when_world_size_changes(self) -> None:
        model = _DenseToyModel()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.1)
        with tempfile.TemporaryDirectory() as temporary:
            config, store = self._save(
                temporary, model=model, optimizer=optimizer, step=10, rows=1
            )
            local_dir = Path(temporary) / "fetched"
            fetch_checkpoint_for_rank(
                store,
                CommittedCheckpoint(
                    step=10,
                    directory=step_directory_name(10),
                    uri=store.uri(step_directory_name(10)),
                ),
                local_dir,
                rank=0,
                world_size=4,
            )
            resumed = load_training_checkpoint(
                config,
                _DenseToyModel(),
                local_dir,
                device=torch.device("cpu"),
                rank=0,
                world_size=4,
            )
        self.assertEqual(resumed.step, 10)
        self.assertIsNone(resumed.data_cursor)

    def test_progress_file_is_human_readable(self) -> None:
        model = _DenseToyModel()
        with tempfile.TemporaryDirectory() as temporary:
            _config, store = self._save(
                temporary, model=model, optimizer=None, step=64, rows=128
            )
            progress = store.read_json(step_directory_name(64), rank_progress_file(0))
        self.assertEqual(progress["step"], 64)
        self.assertEqual(progress["rows"], 128)
        self.assertEqual(progress["data_cursor"]["position"], 42)


class ShardedOptimizerRestoreTest(unittest.TestCase):
    def test_sparse_accumulators_are_restored_with_the_weights(self) -> None:
        config = _reference_config()
        model = _ShardedToyModel()
        optimizer = ShardedRowWiseAdagrad([model.embedding.weight], lr=0.05)
        with torch.no_grad():
            model.embedding.weight.copy_(
                torch.arange(48, dtype=torch.float32).view(12, 4) / 3.0
            )
        accumulator = optimizer.state[model.embedding.weight]["sum"]
        accumulator.copy_(torch.arange(12, dtype=torch.float32) + 1.0)
        saved_weight = model.embedding.weight.detach().clone()
        saved_accumulator = accumulator.clone()

        with tempfile.TemporaryDirectory() as temporary:
            store = LocalCheckpointStore(temporary).child("run")
            store.makedirs()
            staged = stage_training_checkpoint(
                config,
                model,
                Path(store.uri(step_directory_name(5))),
                step=5,
                rows=10,
                sharded_optimizer=optimizer,
                cleanup_staging=False,
            )
            uploader = CheckpointUploader(store, keep_last=1, asynchronous=False)
            uploader.submit(staged)
            uploader.close()

            restored_model = _ShardedToyModel()
            restored_optimizer = ShardedRowWiseAdagrad(
                [restored_model.embedding.weight], lr=0.05
            )
            local_dir = Path(temporary) / "fetched"
            fetch_checkpoint_for_rank(
                store,
                CommittedCheckpoint(
                    step=5,
                    directory=step_directory_name(5),
                    uri=store.uri(step_directory_name(5)),
                ),
                local_dir,
            )
            load_training_checkpoint(
                config,
                restored_model,
                local_dir,
                device=torch.device("cpu"),
                sharded_optimizer=restored_optimizer,
            )

        torch.testing.assert_close(restored_model.embedding.weight, saved_weight)
        torch.testing.assert_close(
            restored_optimizer.state[restored_model.embedding.weight]["sum"],
            saved_accumulator,
        )


class ScanCursorAndResumeTest(unittest.TestCase):
    FILE_COUNT = 6

    def tearDown(self) -> None:
        set_scan_cursor_channel(None)
        set_scan_resume_plan(None)

    def _split(self, directory: str) -> ParquetSplitConfig:
        paths = []
        for index in range(self.FILE_COUNT):
            path = Path(directory) / f"part-{index}.parquet"
            pq.write_table(pa.table({"row_id": [index * 10, index * 10 + 1]}), path)
            paths.append(str(path))
        return ParquetSplitConfig(
            format="flat_parquet",
            inputs=tuple(paths),
            reader=ReaderConfig(
                num_workers=0,
                prefetch_batches=0,
                shard_unit="file",
                scanner_batch_rows=2,
            ),
        )

    @staticmethod
    def _rows(scanner: ParquetScanner) -> list[int]:
        return [
            value
            for batch in scanner.iter_record_batches()
            for value in batch.column("row_id").to_pylist()
        ]

    def test_cursor_tracks_the_file_being_read(self) -> None:
        channel = ScanCursorChannel()
        with tempfile.TemporaryDirectory() as directory:
            split = self._split(directory)
            scanner = ParquetScanner(split, ["row_id"])
            set_scan_cursor_channel(channel, split_key=scanner.split_key)

            positions = []
            for batch in scanner.iter_record_batches():
                del batch
                positions.append(channel.read().position)

        self.assertEqual(positions, list(range(self.FILE_COUNT)))
        self.assertEqual(channel.read().work_unit, "file")

    def test_resume_skips_consumed_files_and_rewinds_one(self) -> None:
        channel = ScanCursorChannel()
        with tempfile.TemporaryDirectory() as directory:
            split = self._split(directory)
            scanner = ParquetScanner(split, ["row_id"])
            set_scan_cursor_channel(channel, split_key=scanner.split_key)
            iterator = scanner.iter_record_batches()
            for _ in range(4):
                next(iterator)
            iterator.close()
            recorded = channel.read()
            self.assertEqual(recorded.position, 3)

            set_scan_cursor_channel(None)
            set_scan_resume_plan(
                ScanResumePlan(
                    work_unit=recorded.work_unit,
                    position=recorded.position,
                    prefix_digest=recorded.prefix_digest,
                    split_key=scanner.split_key,
                    rewind=1,
                )
            )
            resumed_rows = self._rows(ParquetScanner(split, ["row_id"]))

        # Position 3 rewound by one file replays file 2 and continues to the end.
        self.assertEqual(resumed_rows, [20, 21, 30, 31, 40, 41, 50, 51])

    def test_resume_without_rewind_starts_at_the_recorded_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split = self._split(directory)
            scanner = ParquetScanner(split, ["row_id"])
            digest = scan_prefix_digest(
                scan_work_item_key(ref.canonical_uri) for ref in scanner.paths[:3]
            )
            set_scan_resume_plan(
                ScanResumePlan(
                    work_unit="file",
                    position=3,
                    prefix_digest=digest,
                    split_key=scanner.split_key,
                    rewind=0,
                )
            )
            resumed_rows = self._rows(ParquetScanner(split, ["row_id"]))
        self.assertEqual(resumed_rows, [30, 31, 40, 41, 50, 51])

    def test_rewritten_inputs_fall_back_to_a_full_scan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split = self._split(directory)
            scanner = ParquetScanner(split, ["row_id"])
            set_scan_resume_plan(
                ScanResumePlan(
                    work_unit="file",
                    position=3,
                    prefix_digest="digest-of-a-different-input-list",
                    split_key=scanner.split_key,
                    rewind=0,
                )
            )
            rows = self._rows(ParquetScanner(split, ["row_id"]))
        self.assertEqual(rows[:2], [0, 1])
        self.assertEqual(len(rows), 2 * self.FILE_COUNT)

    def test_plan_for_another_split_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split = self._split(directory)
            set_scan_resume_plan(
                ScanResumePlan(
                    work_unit="file",
                    position=3,
                    prefix_digest=None,
                    split_key="a-different-split",
                    rewind=0,
                )
            )
            rows = self._rows(ParquetScanner(split, ["row_id"]))
        self.assertEqual(len(rows), 2 * self.FILE_COUNT)

    def test_split_key_separates_ranks_and_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split = self._split(directory)
            first = scan_split_key(split, shard_rank=0, shard_world_size=2)
            second = scan_split_key(split, shard_rank=1, shard_world_size=2)
        self.assertNotEqual(first, second)

    def test_cursor_channel_survives_a_torn_or_empty_payload(self) -> None:
        channel = ScanCursorChannel()
        self.assertIsNone(channel.read())
        channel.publish(ScanPosition("file", 3, "abc"))
        self.assertEqual(channel.read(), ScanPosition("file", 3, "abc"))
        # An absent digest must not come back as the string "None".
        channel.publish(ScanPosition("row_group", 4, None))
        self.assertEqual(channel.read(), ScanPosition("row_group", 4, None))

    def test_cursor_crosses_the_host_prepare_spawn_boundary(self) -> None:
        """The reader publishes from a spawn child; the trainer reads here."""

        context = mp.get_context("spawn")
        channel = ScanCursorChannel.shared(context)
        process = context.Process(
            target=_publish_from_child,
            args=(channel.storage, "file", 11, "digest-11"),
        )
        process.start()
        process.join(timeout=120)
        self.assertEqual(process.exitcode, 0)
        self.assertEqual(channel.read(), ScanPosition("file", 11, "digest-11"))


class CheckpointCoordinatorTest(unittest.TestCase):
    """The training-loop facade: cadence, staging, publishing, and resume."""

    def tearDown(self) -> None:
        set_scan_cursor_channel(None)
        set_scan_resume_plan(None)

    @staticmethod
    def _config(directory: str, **overrides):
        config = _reference_config()
        checkpoint = CheckpointConfig(
            dir=directory,
            run_name="run",
            every_steps=2,
            keep_last=2,
            async_upload=False,
            **overrides,
        )
        return replace(
            config,
            training=replace(config.training, checkpoint=checkpoint),
        )

    @staticmethod
    def _context():
        return DistributedContext(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device("cpu"),
        )

    def test_cadence_only_fires_on_multiples_of_every_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            coordinator = _CheckpointCoordinator.create(
                self._config(temporary), self._context(), False
            )
            try:
                self.assertFalse(coordinator.due(0))
                self.assertFalse(coordinator.due(1))
                self.assertTrue(coordinator.due(2))
                self.assertTrue(coordinator.due(4))
                self.assertTrue(coordinator.due_on_exit(5))
            finally:
                coordinator.close()

    def test_save_then_resume_restores_step_and_data_position(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(temporary)
            context = self._context()
            model = _DenseToyModel()
            optimizer = torch.optim.RMSprop(model.parameters(), lr=0.1)

            writer = _CheckpointCoordinator.create(config, context, False)
            writer.scan_cursor.publish(ScanPosition("file", 17, "digest-17"))
            writer.save(
                config,
                model,
                context,
                step=4,
                rows=1024,
                elapsed_seconds=12.5,
                dense_optimizer=optimizer,
                replicated_sparse_optimizer=None,
                sharded_optimizer=None,
            )
            writer.close()

            reader = _CheckpointCoordinator.create(config, context, False)
            try:
                resumed = reader.restore(
                    config,
                    _DenseToyModel(),
                    context,
                    torch.device("cpu"),
                    dense_optimizer=torch.optim.RMSprop(
                        _DenseToyModel().parameters(), lr=0.1
                    ),
                    replicated_sparse_optimizer=None,
                    sharded_optimizer=None,
                )
                plan = reader.scan_resume_plan
            finally:
                reader.close()

        self.assertEqual(resumed.step, 4)
        self.assertEqual(resumed.rows, 1024)
        self.assertEqual(plan.position, 17)
        self.assertEqual(plan.work_unit, "file")
        self.assertEqual(plan.prefix_digest, "digest-17")
        self.assertEqual(plan.rewind, 1)
        self.assertEqual(
            plan.split_key,
            scan_split_key(config.data.train, shard_rank=0, shard_world_size=1),
        )

    def test_resume_none_starts_a_fresh_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(temporary)
            context = self._context()
            writer = _CheckpointCoordinator.create(config, context, False)
            writer.save(
                config,
                _DenseToyModel(),
                context,
                step=2,
                rows=8,
                elapsed_seconds=1.0,
                dense_optimizer=None,
                replicated_sparse_optimizer=None,
                sharded_optimizer=None,
            )
            writer.close()

            disabled = self._config(temporary, resume="none")
            reader = _CheckpointCoordinator.create(disabled, context, False)
            try:
                resumed = reader.restore(
                    disabled,
                    _DenseToyModel(),
                    context,
                    torch.device("cpu"),
                    dense_optimizer=None,
                    replicated_sparse_optimizer=None,
                    sharded_optimizer=None,
                )
            finally:
                reader.close()
        self.assertIsNone(resumed)

    def test_retention_keeps_only_the_configured_number_of_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(temporary)
            context = self._context()
            coordinator = _CheckpointCoordinator.create(config, context, False)
            try:
                for step in (2, 4, 6):
                    coordinator.save(
                        config,
                        _DenseToyModel(),
                        context,
                        step=step,
                        rows=step * 10,
                        elapsed_seconds=float(step),
                        dense_optimizer=None,
                        replicated_sparse_optimizer=None,
                        sharded_optimizer=None,
                    )
            finally:
                coordinator.close()
            store = LocalCheckpointStore(temporary).child("run")
            steps = [item.step for item in list_committed_checkpoints(store)]
        self.assertEqual(steps, [4, 6])


class ScanResumeRewindTest(unittest.TestCase):
    """The reader leads the trainer; the rewind must cover exactly that lead."""

    @staticmethod
    def _position(**overrides) -> ScanPosition:
        payload = {
            "work_unit": "file",
            "position": 20,
            "prefix_digest": "d",
            "emitted_rows": 0,
            "min_item_rows": 100,
            "lag_rows": 0,
        }
        payload.update(overrides)
        return ScanPosition(**payload)

    def test_in_flight_rows_become_work_items(self) -> None:
        # The reader emitted 350 rows more than the trainer consumed, and the
        # smallest file held 100, so at most four files can be unfinished.
        rewind = scan_resume_rewind(
            self._position(emitted_rows=1350),
            rows_trained=1000,
            extra_items=0,
        )
        self.assertEqual(rewind, 4)

    def test_scanner_side_buffering_counts_as_lead(self) -> None:
        rewind = scan_resume_rewind(
            self._position(emitted_rows=1000, lag_rows=250),
            rows_trained=1000,
            extra_items=1,
        )
        self.assertEqual(rewind, 4)

    def test_caught_up_reader_only_pays_the_configured_margin(self) -> None:
        rewind = scan_resume_rewind(
            self._position(emitted_rows=1000),
            rows_trained=1200,
            extra_items=2,
        )
        self.assertEqual(rewind, 2)

    def test_unmeasured_item_size_redoes_the_prefix(self) -> None:
        """One reported item leaves no way to size the lead, so redo everything."""

        rewind = scan_resume_rewind(
            self._position(position=6, emitted_rows=900, min_item_rows=0),
            rows_trained=0,
            extra_items=1,
        )
        self.assertEqual(rewind, 7)

    def test_channel_reports_rows_and_the_smallest_item(self) -> None:
        channel = ScanCursorChannel(lag_rows=32)
        channel.publish(ScanPosition("file", 0, "d0"))
        channel.note_emitted_rows(500)
        channel.publish(ScanPosition("file", 1, "d1"))
        channel.note_emitted_rows(100)
        channel.publish(ScanPosition("file", 2, "d2"))
        channel.note_emitted_rows(400)
        channel.publish(ScanPosition("file", 3, "d3"))

        recorded = channel.read()
        self.assertEqual(recorded.position, 3)
        self.assertEqual(recorded.emitted_rows, 1000)
        self.assertEqual(recorded.min_item_rows, 100)
        self.assertEqual(recorded.lag_rows, 32)

    def test_reader_ahead_of_the_trainer_replays_untrained_items(self) -> None:
        """The regression this guards: a prefetching reader losing whole files."""

        channel = ScanCursorChannel()
        channel.note_lag_rows(16)
        for index in range(7):
            channel.publish(ScanPosition("file", index, f"d{index}"))
            channel.note_emitted_rows(16)
        # Six files were read, four were trained.
        recorded = channel.read()
        rewind = scan_resume_rewind(recorded, rows_trained=64, extra_items=1)
        self.assertLessEqual(recorded.position - rewind, 4)


class FeatureBatchIteratorWiringTest(unittest.TestCase):
    """`iter_feature_batches` must hand the cursor to whoever runs the scanner."""

    def tearDown(self) -> None:
        set_scan_cursor_channel(None)
        set_scan_resume_plan(None)

    @staticmethod
    def _config(*, host_prepare_prefetch: int):
        config = _reference_config()
        train = replace(
            config.data.train,
            reader=replace(
                config.data.train.reader,
                host_prepare_prefetch=host_prepare_prefetch,
                device_prefetch_batches=0,
                adapter_workers=0,
            ),
        )
        return replace(config, data=replace(config.data, train=train))

    def test_host_prepare_child_receives_the_cursor_and_plan(self) -> None:
        config = self._config(host_prepare_prefetch=2)
        channel = ScanCursorChannel()
        plan = ScanResumePlan(work_unit="file", position=5, split_key="k")
        with patch("src.train._ProcessHostPrepareIterator") as process_cls:
            process_cls.return_value = iter(())
            list(
                iter_feature_batches(
                    config,
                    "train",
                    vocab_maps={},
                    require_labels=False,
                    scan_cursor=channel,
                    scan_resume_plan=plan,
                )
            )
        kwargs = process_cls.call_args.kwargs
        self.assertIs(kwargs["scan_cursor"], channel)
        self.assertIs(kwargs["scan_resume_plan"], plan)

    def test_in_process_reader_installs_the_cursor_for_this_split(self) -> None:
        config = self._config(host_prepare_prefetch=0)
        channel = ScanCursorChannel()
        plan = ScanResumePlan(work_unit="file", position=5, split_key="k")
        with patch("src.train._iter_batch_tables", return_value=iter(())):
            list(
                iter_feature_batches(
                    config,
                    "train",
                    vocab_maps={},
                    require_labels=False,
                    shard_rank=1,
                    shard_world_size=4,
                    scan_cursor=channel,
                    scan_resume_plan=plan,
                )
            )
        self.assertIs(dataloader_module._SCAN_CURSOR_CHANNEL, channel)
        self.assertIs(dataloader_module._SCAN_RESUME_PLAN, plan)
        self.assertEqual(
            dataloader_module._SCAN_CURSOR_SPLIT_KEY,
            scan_split_key(config.data.train, shard_rank=1, shard_world_size=4),
        )


class CheckpointStoreProbeCommandTest(unittest.TestCase):
    """`check-checkpoint-store` must pass on a good path and fail loudly otherwise."""

    @staticmethod
    def _run(*extra: str) -> int:
        parser_args = [
            "check-checkpoint-store",
            "--probe-mib",
            "1",
            *extra,
        ]
        parser = build_arg_parser()
        args = parser.parse_args(parser_args)
        return int(args.func(args))

    def test_probe_passes_and_leaves_nothing_behind(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            code = self._run("--checkpoint-dir", temporary, "--checkpoint-run-name", "r")
            leftovers = [
                entry.name
                for entry in LocalCheckpointStore(temporary).child("r").list_entries()
            ]
        self.assertEqual(code, 0)
        self.assertEqual(leftovers, [])

    def test_probe_reports_the_step_a_restart_would_use(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LocalCheckpointStore(temporary).child("r")
            for step in (10, 20):
                directory = step_directory_name(step)
                store.write_json({"step": step}, directory, CHECKPOINT_MANIFEST)
                store.write_json({"step": step}, directory, COMMIT_MARKER)
            passed = self._run(
                "--checkpoint-dir", temporary, "--checkpoint-run-name", "r"
            )
            missing = self._run(
                "--checkpoint-dir",
                temporary,
                "--checkpoint-run-name",
                "r",
                "--resume",
                "30",
            )
        self.assertEqual(passed, 0)
        self.assertEqual(missing, 1)

    def test_unusable_directory_fails_without_a_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            blocker = Path(temporary) / "not-a-directory"
            blocker.write_text("", encoding="utf-8")
            self.assertEqual(self._run("--checkpoint-dir", str(blocker)), 1)

    def test_a_directory_is_required(self) -> None:
        with self.assertRaises(ValueError):
            self._run()


class CheckpointConfigTest(unittest.TestCase):
    def test_defaults_are_disabled_until_a_directory_is_set(self) -> None:
        training = TrainingConfig()
        training.validate()
        self.assertFalse(training.checkpoint.enabled)
        self.assertEqual(training.checkpoint.resume, "auto")

    def test_mapping_is_parsed_and_validated(self) -> None:
        training = TrainingConfig.from_mapping(
            {
                "checkpoint": {
                    "dir": "hdfs://temu-data-ns/apps/run",
                    "every_steps": 2000,
                    "keep_last": 3,
                    "resume": "auto",
                }
            }
        )
        training.validate()
        self.assertTrue(training.checkpoint.enabled)
        self.assertEqual(training.checkpoint.every_steps, 2000)

    def test_production_configs_never_share_a_run_directory(self) -> None:
        """Coarse and fine siblings share ``model.name`` but not their vocabulary.

        Two such runs pointed at one run directory would make the second one
        resume from the first's newest step and die on the fingerprint check, so
        every enabled config must resolve to its own run name.
        """

        root = Path(__file__).resolve().parents[1]
        runs: dict[tuple[str, str], list[str]] = {}
        for path in sorted((root / "configs").glob("*.yaml")):
            config = load_app_config(path)
            checkpoint = config.training.checkpoint
            if not checkpoint.enabled:
                continue
            key = (str(checkpoint.dir), checkpoint.run_name or config.model.name)
            runs.setdefault(key, []).append(path.name)
        self.assertTrue(runs, "no production config enables checkpointing")
        collisions = {key: names for key, names in runs.items() if len(names) > 1}
        self.assertEqual(collisions, {})

    def test_invalid_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CheckpointConfig(every_steps=-1).validate()
        with self.assertRaises(ValueError):
            CheckpointConfig(keep_last=-1).validate()
        with self.assertRaises(ValueError):
            CheckpointConfig(resume="").validate()
        with self.assertRaises(ValueError):
            CheckpointConfig(
                dir="hdfs://ns/run", every_steps=0, save_on_exit=False
            ).validate()


if __name__ == "__main__":
    unittest.main()
