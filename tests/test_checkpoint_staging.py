"""Staging-side checkpoint behaviour: chunking, space checks, and streaming.

A full sharded checkpoint is tens of GiB per rank, and every rank on a node
stages into the same filesystem. These tests cover the machinery that keeps that
footprint bounded and turns "the filesystem filled up" into a message that names
the directory instead of a zip writer offset.
"""

from __future__ import annotations

from dataclasses import asdict
import errno
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.checkpoint import (
    CheckpointStagingSpaceError,
    CheckpointUploader,
    DEFAULT_SHARD_CHUNK_BYTES,
    MODEL_SUBDIR,
    SHARDED_CHECKPOINT_CHUNK_FORMAT,
    SHARDED_CHECKPOINT_FORMAT,
    _checkpoint_metadata,
    check_staging_space,
    estimate_staging_space,
    legacy_rank_file,
    load_model_checkpoint,
    plan_shard_chunks,
    rank_ready_marker,
    save_model_checkpoint,
    shard_file_names,
    stage_training_checkpoint,
    step_directory_name,
)
from src.checkpoint_store import LocalCheckpointStore
from src.config import load_app_config
from src.embeddings import EmbeddingTableSpec, ShardedEmbedding, plan_embedding_shards


def _reference_config():
    root = Path(__file__).resolve().parents[1]
    return load_app_config(root / "configs" / "reference" / "default.yaml")


class _MultiTableModel(nn.Module):
    """Three tables of different sizes, so chunk packing has something to do."""

    def __init__(self, world_size: int = 1) -> None:
        super().__init__()
        specs = [
            EmbeddingTableSpec("small", 8, 2),
            EmbeddingTableSpec("medium", 64, 4),
            EmbeddingTableSpec("large", 256, 8),
        ]
        plan = plan_embedding_shards(
            specs,
            world_size=world_size,
            strategy="row_wise",
            table_wise_max_rows=4,
        )
        self.tables = nn.ModuleDict(
            {
                spec.name: ShardedEmbedding(
                    spec.num_embeddings,
                    spec.embedding_dim,
                    table_name=spec.name,
                    shard_spec=plan.tables[spec.name],
                )
                for spec in specs
            }
        )
        self.dense = nn.Linear(8, 2)

    def fill(self) -> dict[str, torch.Tensor]:
        expected: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, module in self.tables.items():
                rows, dim = module.num_embeddings, module.embedding_dim
                weight = torch.arange(rows * dim, dtype=torch.float32).view(rows, dim)
                weight = weight / (rows * dim)
                module.load_full_weight_(weight)
                expected[name] = weight
        return expected


class ShardChunkPlanTest(unittest.TestCase):
    def test_tables_are_packed_up_to_the_budget(self) -> None:
        model = _MultiTableModel()
        modules = [model.tables[name] for name in ("small", "medium", "large")]
        # small=64B, medium=1KiB, large=8KiB at fp32.
        chunks = plan_shard_chunks(modules, world_size=1, chunk_bytes=2048)
        names = [[module.table_name for module in chunk] for chunk in chunks]
        self.assertEqual(names, [["large"], ["medium", "small"]])

    def test_one_oversized_table_still_gets_its_own_file(self) -> None:
        model = _MultiTableModel()
        modules = [model.tables[name] for name in ("small", "large")]
        chunks = plan_shard_chunks(modules, world_size=1, chunk_bytes=16)
        self.assertEqual([len(chunk) for chunk in chunks], [1, 1])

    def test_plan_is_derived_from_the_global_table_shape(self) -> None:
        # Every rank must agree on chunk boundaries, so sizes come from
        # num_embeddings divided by the world size rather than from however many
        # rows this rank happens to own (which differs by the owner mapping).
        model = _MultiTableModel()
        modules = [model.tables[name] for name in ("small", "medium", "large")]

        def _layout(world_size: int) -> list[list[str]]:
            return [
                [module.table_name for module in chunk]
                for chunk in plan_shard_chunks(
                    modules, world_size=world_size, chunk_bytes=4096
                )
            ]

        self.assertEqual(_layout(1), [["large"], ["medium", "small"]])
        self.assertEqual(_layout(4), [["large", "medium", "small"]])


class ChunkedShardFileTest(unittest.TestCase):
    def test_save_writes_several_files_and_loads_back(self) -> None:
        config = _reference_config()
        model = _MultiTableModel()
        expected = model.fill()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model"
            written = save_model_checkpoint(
                config,
                model,
                path,
                chunk_bytes=1024,
            )
            chunk_files = sorted(path.glob("shard-*-rank-00000-of-00001.pt"))
            self.assertGreater(len(chunk_files), 1)
            self.assertIn("dense.pt", written)
            self.assertIn("manifest.json", written)

            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["format"], SHARDED_CHECKPOINT_CHUNK_FORMAT)
            packed = [name for chunk in manifest["chunks"] for name in chunk["tables"]]
            self.assertEqual(sorted(packed), ["large", "medium", "small"])

            restored = _MultiTableModel()
            load_model_checkpoint(
                config, restored, path, device=torch.device("cpu")
            )
        for name, weight in expected.items():
            torch.testing.assert_close(restored.tables[name].weight, weight)

    def test_publish_sees_every_file_as_it_lands(self) -> None:
        config = _reference_config()
        model = _MultiTableModel()
        model.fill()
        seen: list[str] = []
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model"

            def _publish(name: str) -> bool:
                # Files must exist by the time they are offered, otherwise the
                # uploader would race the writer.
                self.assertTrue((path / name).exists(), name)
                seen.append(name)
                return True

            written = save_model_checkpoint(
                config,
                model,
                path,
                chunk_bytes=1024,
                publish=_publish,
            )
        self.assertEqual(seen, written)
        self.assertEqual(seen[-1], "manifest.json")


class LegacyShardLayoutTest(unittest.TestCase):
    """The one-file-per-rank layout stays readable so a resume can cross a deploy."""

    def _write_v1_checkpoint(self, path: Path, config, model) -> None:
        path.mkdir(parents=True, exist_ok=True)
        tables = {}
        for module in model.tables.values():
            tables[module.table_name] = {
                "weight": module.weight.detach().cpu(),
                "num_embeddings": module.num_embeddings,
                "embedding_dim": module.embedding_dim,
                "padding_idx": module.padding_idx,
                "shard_spec": asdict(module.shard_spec),
                "optimizer_state": None,
            }
        torch.save(
            {
                "format": SHARDED_CHECKPOINT_FORMAT,
                "rank": 0,
                "world_size": 1,
                "tables": tables,
            },
            path / legacy_rank_file(0, 1),
        )
        sharded = {f"tables.{name}.weight" for name in model.tables}
        torch.save(
            {
                "model_state_dict": {
                    key: value
                    for key, value in model.state_dict().items()
                    if key not in sharded
                },
                **_checkpoint_metadata(config),
            },
            path / "dense.pt",
        )
        (path / "manifest.json").write_text(
            json.dumps(
                {
                    "format": SHARDED_CHECKPOINT_FORMAT,
                    "version": 1,
                    "world_size": 1,
                    "dense_file": "dense.pt",
                    "rank_files": [legacy_rank_file(0, 1)],
                    "tables": {
                        module.table_name: {
                            "num_embeddings": module.num_embeddings,
                            "embedding_dim": module.embedding_dim,
                            "padding_idx": module.padding_idx,
                        }
                        for module in model.tables.values()
                    },
                    "training_metadata": {
                        "sparse_optimizer": config.training.sparse_optimizer
                    },
                    **_checkpoint_metadata(config),
                }
            ),
            encoding="utf-8",
        )

    def test_a_monolithic_rank_file_still_loads(self) -> None:
        config = _reference_config()
        saved = _MultiTableModel()
        expected = saved.fill()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model"
            self._write_v1_checkpoint(path, config, saved)
            restored = _MultiTableModel()
            load_model_checkpoint(
                config, restored, path, device=torch.device("cpu")
            )
        for name, weight in expected.items():
            torch.testing.assert_close(restored.tables[name].weight, weight)

    def test_shard_file_names_handles_both_layouts(self) -> None:
        legacy = {"world_size": 2, "rank_files": ["a.pt", "b.pt"]}
        self.assertEqual(shard_file_names(legacy, rank=1, world_size=2), ["b.pt"])
        # Resharding needs every saved owner, not just this rank's file.
        self.assertEqual(
            shard_file_names(legacy, rank=0, world_size=1), ["a.pt", "b.pt"]
        )
        chunked = {
            "world_size": 2,
            "chunks": [
                {"index": 0, "rank_files": ["c0r0.pt", "c0r1.pt"]},
                {"index": 1, "rank_files": ["c1r0.pt", "c1r1.pt"]},
            ],
        }
        self.assertEqual(
            shard_file_names(chunked, rank=1, world_size=2), ["c0r1.pt", "c1r1.pt"]
        )
        self.assertEqual(len(shard_file_names(chunked, rank=0, world_size=4)), 4)


class StagingSpaceTest(unittest.TestCase):
    def test_estimate_covers_shards_and_rank_zero_extras(self) -> None:
        model = _MultiTableModel()
        estimate = estimate_staging_space(model, rank=0, world_size=1)
        table_bytes = sum(
            module.weight.numel() * module.weight.element_size()
            for module in model.tables.values()
        )
        dense_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in model.dense.state_dict().values()
        )
        self.assertEqual(estimate.total_bytes, table_bytes + dense_bytes)
        self.assertEqual(estimate.chunk_count, 1)

    def test_smaller_chunks_lower_the_peak_but_not_the_total(self) -> None:
        model = _MultiTableModel()
        whole = estimate_staging_space(model, chunk_bytes=DEFAULT_SHARD_CHUNK_BYTES)
        split = estimate_staging_space(model, chunk_bytes=1024, upload_window=0)
        self.assertEqual(whole.total_bytes, split.total_bytes)
        self.assertLess(split.peak_bytes, whole.peak_bytes)
        self.assertGreater(split.chunk_count, whole.chunk_count)

    def test_too_little_room_names_the_directory_and_the_need(self) -> None:
        model = _MultiTableModel()
        estimate = estimate_staging_space(model)
        with tempfile.TemporaryDirectory() as temporary:
            with patch("src.checkpoint._free_bytes", return_value=8):
                with self.assertRaises(CheckpointStagingSpaceError) as caught:
                    check_staging_space(temporary, estimate, local_ranks=4)
                message = str(caught.exception)
                self.assertIn(temporary, message)
                self.assertIn("staging_dir", message)
                self.assertIn("local_ranks=4", message)

    def test_the_requirement_scales_with_the_ranks_on_the_node(self) -> None:
        model = _MultiTableModel()
        estimate = estimate_staging_space(model)
        with tempfile.TemporaryDirectory() as temporary:
            free = int(estimate.peak_bytes * 2 * 1.5)
            with patch("src.checkpoint._free_bytes", return_value=free):
                check_staging_space(temporary, estimate, local_ranks=2)
                with self.assertRaises(CheckpointStagingSpaceError):
                    check_staging_space(temporary, estimate, local_ranks=8)

    def test_enforce_false_reports_instead_of_raising(self) -> None:
        model = _MultiTableModel()
        estimate = estimate_staging_space(model)
        with tempfile.TemporaryDirectory() as temporary:
            with patch("src.checkpoint._free_bytes", return_value=8):
                summary = check_staging_space(temporary, estimate, enforce=False)
        self.assertIn("needed=", summary)


class OutOfSpaceReportingTest(unittest.TestCase):
    def test_a_full_filesystem_raises_a_named_error_and_leaves_no_partial(self) -> None:
        config = _reference_config()
        model = _MultiTableModel()

        def _fail(payload, path, *args, **kwargs):
            Path(path).write_bytes(b"partial archive")
            raise OSError(errno.ENOSPC, "No space left on device")

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model"
            with patch("src.checkpoint.torch.save", side_effect=_fail):
                with self.assertRaises(CheckpointStagingSpaceError) as caught:
                    save_model_checkpoint(config, model, path)
            self.assertIn("staging_dir", str(caught.exception))
            self.assertEqual(list(path.glob(".*tmp-*")), [])

    def test_an_unrelated_write_error_is_not_relabelled(self) -> None:
        config = _reference_config()
        model = _MultiTableModel()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model"
            with patch(
                "src.checkpoint.torch.save",
                side_effect=RuntimeError("serialization is confused"),
            ):
                with self.assertRaises(RuntimeError) as caught:
                    save_model_checkpoint(config, model, path)
        self.assertNotIsInstance(caught.exception, CheckpointStagingSpaceError)


class StreamingPublishTest(unittest.TestCase):
    def _store(self, root: str) -> LocalCheckpointStore:
        store = LocalCheckpointStore(root).child("run")
        store.makedirs()
        return store

    def test_streamed_files_reach_the_store_and_leave_staging(self) -> None:
        config = _reference_config()
        model = _MultiTableModel()
        model.fill()
        with tempfile.TemporaryDirectory() as temporary:
            store = self._store(str(Path(temporary) / "run-dir"))
            uploader = CheckpointUploader(store, keep_last=3, asynchronous=False)
            staging = Path(temporary) / "staging" / step_directory_name(20)
            staged = stage_training_checkpoint(
                config,
                model,
                staging,
                step=20,
                rows=64,
                chunk_bytes=1024,
                publish=uploader.stream_publisher(staging, 20),
            )
            # Every staged file was handed over and deleted while staging ran.
            self.assertEqual(sorted(p.name for p in staging.rglob("*.pt")), [])
            uploader.submit(staged)
            uploader.close()

            directory = step_directory_name(20)
            for relative in staged.relative_files:
                self.assertTrue(
                    store.exists(directory, *relative.split("/")), relative
                )
            ready = store.read_json(directory, rank_ready_marker(0))
            self.assertEqual(sorted(ready["files"]), sorted(staged.relative_files))

    def test_a_streamed_step_still_resumes(self) -> None:
        config = _reference_config()
        model = _MultiTableModel()
        expected = model.fill()
        with tempfile.TemporaryDirectory() as temporary:
            store = self._store(str(Path(temporary) / "run-dir"))
            uploader = CheckpointUploader(store, keep_last=3, asynchronous=False)
            staging = Path(temporary) / "staging" / step_directory_name(20)
            staged = stage_training_checkpoint(
                config,
                model,
                staging,
                step=20,
                rows=64,
                chunk_bytes=1024,
                publish=uploader.stream_publisher(staging, 20),
            )
            uploader.submit(staged)
            uploader.close()
            restored = _MultiTableModel()
            load_model_checkpoint(
                config,
                restored,
                Path(store.uri(step_directory_name(20))) / MODEL_SUBDIR,
                device=torch.device("cpu"),
            )
        for name, weight in expected.items():
            torch.testing.assert_close(restored.tables[name].weight, weight)

    def test_a_failed_stream_leaves_the_file_for_the_publish_pass(self) -> None:
        config = _reference_config()
        model = _MultiTableModel()
        model.fill()
        with tempfile.TemporaryDirectory() as temporary:
            store = self._store(str(Path(temporary) / "run-dir"))
            uploader = CheckpointUploader(store, keep_last=3, asynchronous=False)
            staging = Path(temporary) / "staging" / step_directory_name(20)
            with patch.object(
                store, "upload_file", side_effect=OSError("run directory is down")
            ):
                staged = stage_training_checkpoint(
                    config,
                    model,
                    staging,
                    step=20,
                    rows=64,
                    chunk_bytes=1024,
                    publish=uploader.stream_publisher(staging, 20),
                )
                # Nothing was published, so nothing may have been deleted.
                for relative in staged.relative_files:
                    self.assertTrue((staging / relative).exists(), relative)
            uploader.submit(staged)
            uploader.close()
            directory = step_directory_name(20)
            for relative in staged.relative_files:
                self.assertTrue(
                    store.exists(directory, *relative.split("/")), relative
                )

    def test_a_local_run_directory_keeps_its_files(self) -> None:
        config = _reference_config()
        model = _MultiTableModel()
        model.fill()
        with tempfile.TemporaryDirectory() as temporary:
            store = self._store(temporary)
            uploader = CheckpointUploader(store, keep_last=3, asynchronous=False)
            staging = Path(store.uri(step_directory_name(20)))
            staged = stage_training_checkpoint(
                config,
                model,
                staging,
                step=20,
                rows=64,
                chunk_bytes=1024,
                cleanup_staging=False,
                # Staging *is* the destination here; deleting after "upload"
                # would delete the checkpoint.
                publish=uploader.stream_publisher(staging, 20, enabled=False),
            )
            uploader.submit(staged)
            uploader.close()
            for relative in staged.relative_files:
                self.assertTrue((staging / relative).exists(), relative)


if __name__ == "__main__":
    unittest.main()
