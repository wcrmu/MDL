"""Data-window scheduling and streaming sparse delta checkpoint tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.checkpoint import (
    CHECKPOINT_MANIFEST,
    COMMIT_MARKER,
    CheckpointDataWindow,
    CheckpointUploader,
    CommittedCheckpoint,
    fetch_checkpoint_for_rank,
    load_training_checkpoint,
    prune_run_directory,
    resolve_resume_checkpoint,
    rank_ready_marker,
    stage_training_checkpoint,
    StagedCheckpoint,
    step_directory_name,
)
from src.checkpoint_store import LocalCheckpointStore
from src.config import CheckpointConfig, load_app_config
from src.embeddings import EmbeddingTableSpec, ShardedEmbedding, plan_embedding_shards
from src.main import _apply_checkpoint_overrides
from src.optim import ShardedRowWiseAdagrad
from src.train import _CheckpointCoordinator, plan_checkpoint_data_windows


def _config():
    return load_app_config(
        Path(__file__).resolve().parents[1] / "configs" / "reference" / "default.yaml"
    )


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        table = EmbeddingTableSpec("item", 24, 4)
        plan = plan_embedding_shards(
            [table],
            world_size=1,
            strategy="row_wise",
            table_wise_max_rows=4,
            optimizer_state_layout="rowwise",
        )
        self.embedding = ShardedEmbedding(
            table.num_embeddings,
            table.embedding_dim,
            table_name=table.name,
            shard_spec=plan.tables[table.name],
        )
        self.dense = nn.Linear(4, 2)


def _mark_committed(root: Path, directory: str) -> None:
    (root / directory / COMMIT_MARKER).write_text("", encoding="utf-8")


def test_dirty_tracker_records_only_owner_local_updated_rows() -> None:
    parameter = nn.Parameter(torch.zeros(16, 4))
    optimizer = ShardedRowWiseAdagrad(
        [parameter], lr=0.1, track_dirty_rows=True
    )
    rows = torch.tensor([[1, 5, 9]], dtype=torch.long)
    values = torch.ones(3, 4)
    parameter.grad = torch.sparse_coo_tensor(
        rows, values, parameter.shape, is_coalesced=True
    )
    optimizer.step()
    assert torch.equal(
        torch.cat(list(optimizer.iter_dirty_rows(parameter, max_rows=2))),
        torch.tensor([1, 5, 9]),
    )
    optimizer.clear_dirty_rows()
    assert list(optimizer.iter_dirty_rows(parameter, max_rows=2)) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_dirty_bitset_handles_word_boundaries() -> None:
    device = torch.device("cuda:0")
    parameter = nn.Parameter(
        torch.zeros(96, 8, device=device, dtype=torch.bfloat16)
    )
    optimizer = ShardedRowWiseAdagrad(
        [parameter], lr=0.1, track_dirty_rows=True
    )
    expected = torch.tensor(
        [0, 1, 30, 31, 32, 33, 62, 63, 64, 95],
        dtype=torch.long,
    )
    rows = expected.to(device).reshape(1, -1)
    values = torch.arange(
        1,
        expected.numel() * parameter.shape[1] + 1,
        device=device,
        dtype=parameter.dtype,
    ).reshape(expected.numel(), parameter.shape[1])
    parameter.grad = torch.sparse_coo_tensor(
        rows,
        values,
        parameter.shape,
        is_coalesced=True,
    )

    optimizer.step()
    torch.cuda.synchronize(device)
    actual = torch.cat(list(optimizer.iter_dirty_rows(parameter, max_rows=3)))
    assert torch.equal(actual, expected)
    assert int((parameter.float().abs().sum(dim=1) > 0).sum()) == len(expected)

    optimizer.clear_dirty_rows()
    torch.cuda.synchronize(device)
    assert list(optimizer.iter_dirty_rows(parameter, max_rows=3)) == []


def test_hour_inputs_are_partitioned_into_contiguous_eight_hour_windows() -> None:
    config = _config()
    inputs = tuple(
        f"hdfs://warehouse/pt=2026-08-06/hr={hour:02d}"
        for hour in range(18)
    )
    checkpoint = replace(
        config.training.checkpoint,
        dir="/tmp/checkpoints",
        every_steps=0,
        data_window_hours=8,
    )
    config = replace(
        config,
        data=replace(
            config.data,
            train=replace(config.data.train, inputs=inputs),
        ),
        training=replace(config.training, checkpoint=checkpoint),
    )
    windows = plan_checkpoint_data_windows(config)
    assert [(item.start, item.end, len(item.inputs)) for item in windows] == [
        ("2026-08-06T00:00:00Z", "2026-08-06T08:00:00Z", 8),
        ("2026-08-06T08:00:00Z", "2026-08-06T16:00:00Z", 8),
        ("2026-08-06T16:00:00Z", "2026-08-06T18:00:00Z", 2),
    ]


def test_data_window_checkpoint_config_rejects_ambiguous_cadence() -> None:
    with pytest.raises(ValueError, match="cannot both be positive"):
        CheckpointConfig(
            dir="/tmp/checkpoints",
            every_steps=100,
            data_window_hours=8,
        ).validate()
    with pytest.raises(ValueError, match="requires data_window_hours"):
        CheckpointConfig(
            dir="/tmp/checkpoints",
            sparse_delta=True,
        ).validate()


def test_cli_overrides_can_switch_production_back_to_step_mode() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_app_config(root / "configs" / "mdl_onetrans.yaml")
    args = SimpleNamespace(
        checkpoint_dir=None,
        checkpoint_run_name=None,
        checkpoint_every_steps=20,
        checkpoint_every_data_hours=0,
        checkpoint_sparse_delta=False,
        checkpoint_sparse_full_every=None,
        checkpoint_lineage_id=None,
        checkpoint_keep_last=None,
        checkpoint_resume=None,
    )
    updated = _apply_checkpoint_overrides(config, args)
    assert updated.training.checkpoint.every_steps == 20
    assert updated.training.checkpoint.data_window_hours == 0
    assert updated.training.checkpoint.sparse_delta is False


def test_streaming_full_plus_dirty_delta_restores_exact_training_state() -> None:
    config = _config()
    with (
        tempfile.TemporaryDirectory() as root_text,
        tempfile.TemporaryDirectory() as local,
    ):
        root = Path(root_text)
        store = LocalCheckpointStore(root)
        model = _ToyModel()
        optimizer = ShardedRowWiseAdagrad(
            [model.embedding.weight], lr=0.1, track_dirty_rows=True
        )

        base = step_directory_name(10)
        stage_training_checkpoint(
            config,
            model,
            root / base,
            step=10,
            rows=100,
            sharded_optimizer=optimizer,
            chunk_bytes=128,
            sparse_stream=True,
            sparse_kind="full",
            generation=1,
            lineage_id="toy",
            trained_through="2026-08-06T08:00:00Z",
        )
        _mark_committed(root, base)
        optimizer.clear_dirty_rows()

        dirty_rows = torch.tensor([[1, 5, 9]], dtype=torch.long)
        model.embedding.weight.grad = torch.sparse_coo_tensor(
            dirty_rows,
            torch.ones(3, 4),
            model.embedding.weight.shape,
            is_coalesced=True,
        )
        optimizer.step()
        with torch.no_grad():
            model.dense.weight.add_(2.0)
        expected_sparse = model.embedding.weight.detach().clone()
        expected_dense = model.dense.weight.detach().clone()

        delta = step_directory_name(20)
        stage_training_checkpoint(
            config,
            model,
            root / delta,
            step=20,
            rows=200,
            sharded_optimizer=optimizer,
            chunk_bytes=128,
            sparse_stream=True,
            sparse_kind="delta",
            parent_checkpoint=base,
            sparse_parent_checkpoint=base,
            generation=2,
            lineage_id="toy",
            trained_through="2026-08-06T16:00:00Z",
        )
        _mark_committed(root, delta)

        fetched = fetch_checkpoint_for_rank(
            store,
            CommittedCheckpoint(20, delta, str(root / delta)),
            local,
        )
        assert not (
            Path(fetched) / "_sparse_chain" / base / "model" / "dense.pt"
        ).exists()
        restored = _ToyModel()
        restored_optimizer = ShardedRowWiseAdagrad(
            [restored.embedding.weight], lr=0.1, track_dirty_rows=True
        )
        state = load_training_checkpoint(
            config,
            restored,
            fetched,
            device=torch.device("cpu"),
            sharded_optimizer=restored_optimizer,
        )
        assert torch.equal(restored.embedding.weight, expected_sparse)
        assert torch.equal(restored.dense.weight, expected_dense)
        assert state.step == 20
        assert state.generation == 2
        assert state.trained_through == "2026-08-06T16:00:00Z"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_streaming_full_plus_delta_restores_exact_state() -> None:
    config = _config()
    device = torch.device("cuda:0")
    with (
        tempfile.TemporaryDirectory() as root_text,
        tempfile.TemporaryDirectory() as local,
    ):
        root = Path(root_text)
        store = LocalCheckpointStore(root)
        model = _ToyModel().to(device)
        optimizer = ShardedRowWiseAdagrad(
            [model.embedding.weight], lr=0.1, track_dirty_rows=True
        )

        base = step_directory_name(10)
        stage_training_checkpoint(
            config,
            model,
            root / base,
            step=10,
            rows=100,
            sharded_optimizer=optimizer,
            chunk_bytes=128,
            sparse_stream=True,
            sparse_kind="full",
            generation=1,
            lineage_id="cuda-toy",
        )
        _mark_committed(root, base)
        optimizer.clear_dirty_rows()

        dirty_rows = torch.tensor([[1, 5, 9]], dtype=torch.long, device=device)
        values = torch.arange(
            1,
            13,
            dtype=model.embedding.weight.dtype,
            device=device,
        ).reshape(3, 4)
        model.embedding.weight.grad = torch.sparse_coo_tensor(
            dirty_rows,
            values,
            model.embedding.weight.shape,
            is_coalesced=True,
        )
        optimizer.step()
        with torch.no_grad():
            model.dense.weight.add_(1.25)
        expected_sparse = model.embedding.weight.detach().clone()
        expected_accumulator = optimizer.state[model.embedding.weight][
            "sum"
        ].detach().clone()
        expected_dense = model.dense.weight.detach().clone()

        delta = step_directory_name(20)
        stage_training_checkpoint(
            config,
            model,
            root / delta,
            step=20,
            rows=200,
            sharded_optimizer=optimizer,
            chunk_bytes=128,
            sparse_stream=True,
            sparse_kind="delta",
            parent_checkpoint=base,
            sparse_parent_checkpoint=base,
            generation=2,
            lineage_id="cuda-toy",
        )
        _mark_committed(root, delta)

        fetched = fetch_checkpoint_for_rank(
            store,
            CommittedCheckpoint(20, delta, str(root / delta)),
            local,
        )
        restored = _ToyModel().to(device)
        restored_optimizer = ShardedRowWiseAdagrad(
            [restored.embedding.weight], lr=0.1, track_dirty_rows=True
        )
        state = load_training_checkpoint(
            config,
            restored,
            fetched,
            device=device,
            sharded_optimizer=restored_optimizer,
        )
        torch.cuda.synchronize(device)
        assert state.step == 20
        assert state.generation == 2
        assert torch.equal(restored.embedding.weight, expected_sparse)
        assert torch.equal(
            restored_optimizer.state[restored.embedding.weight]["sum"],
            expected_accumulator,
        )
        assert torch.equal(restored.dense.weight, expected_dense)


def test_window_selector_and_retention_keep_required_sparse_base() -> None:
    with tempfile.TemporaryDirectory() as root_text:
        root = Path(root_text)
        store = LocalCheckpointStore(root)
        previous: str | None = None
        boundaries = (
            ("2026-08-06T00:00:00Z", "2026-08-06T08:00:00Z"),
            ("2026-08-06T08:00:00Z", "2026-08-06T16:00:00Z"),
            ("2026-08-06T16:00:00Z", "2026-08-07T00:00:00Z"),
        )
        for generation, (start, end) in enumerate(boundaries, start=1):
            directory = step_directory_name(generation * 10)
            store.makedirs(directory)
            store.write_json(
                {
                    "world_size": 1,
                    "saved_at": generation,
                    "lineage_id": "toy",
                    "generation": generation,
                    "sparse_kind": "full" if previous is None else "delta",
                    "sparse_parent_checkpoint": previous,
                    "data_window": {
                        "start": start,
                        "end": end,
                    },
                    "window_complete": True,
                    "trained_through": end,
                },
                directory,
                CHECKPOINT_MANIFEST,
            )
            store.write_bytes(b"", directory, COMMIT_MARKER)
            previous = directory

        windows = (
            CheckpointDataWindow(
                "2026-08-06T16:00:00Z", "2026-08-07T00:00:00Z"
            ),
        )
        selected = resolve_resume_checkpoint(
            store,
            "auto",
            lineage_id="toy",
            requested_windows=windows,
        )
        assert selected is not None and selected.step == 30

        removed = prune_run_directory(store, keep_last=1)
        # The newest delta depends on both older generations, so neither may be
        # collected until a new full sparse base breaks the dependency chain.
        assert removed == []


def test_window_selector_rejects_a_silent_gap_after_known_lineage() -> None:
    with tempfile.TemporaryDirectory() as root_text:
        root = Path(root_text)
        store = LocalCheckpointStore(root)
        directory = step_directory_name(10)
        store.makedirs(directory)
        store.write_json(
            {
                "world_size": 1,
                "lineage_id": "toy",
                "generation": 1,
                "sparse_kind": "full",
                "window_complete": True,
                "trained_through": "2026-08-06T08:00:00Z",
            },
            directory,
            CHECKPOINT_MANIFEST,
        )
        store.write_bytes(b"", directory, COMMIT_MARKER)

        windows = (
            CheckpointDataWindow(
                "2026-08-06T16:00:00Z", "2026-08-07T00:00:00Z"
            ),
        )
        with pytest.raises(ValueError, match="aligns with requested data range"):
            resolve_resume_checkpoint(
                store,
                "auto",
                lineage_id="toy",
                requested_windows=windows,
            )


def test_window_selector_rejects_rewritten_partition_manifest() -> None:
    with tempfile.TemporaryDirectory() as root_text:
        root = Path(root_text)
        store = LocalCheckpointStore(root)
        directory = step_directory_name(10)
        saved_window = CheckpointDataWindow(
            "2026-08-06T00:00:00Z",
            "2026-08-06T08:00:00Z",
            inputs=("hdfs://warehouse/original.parquet",),
        )
        store.write_json(
            {
                "world_size": 1,
                "lineage_id": "toy",
                "generation": 1,
                "sparse_kind": "full",
                "data_window": saved_window.to_payload(),
                "window_complete": True,
                "trained_through": saved_window.end,
            },
            directory,
            CHECKPOINT_MANIFEST,
        )
        store.write_bytes(b"", directory, COMMIT_MARKER)

        requested = (
            CheckpointDataWindow(
                saved_window.start,
                saved_window.end,
                inputs=("hdfs://warehouse/rewritten.parquet",),
            ),
        )
        with pytest.raises(ValueError, match="aligns with requested data range"):
            resolve_resume_checkpoint(
                store,
                "auto",
                lineage_id="toy",
                requested_windows=requested,
            )


def test_explicit_incomplete_resume_still_validates_partition_manifest() -> None:
    saved = CheckpointDataWindow(
        "2026-08-06T00:00:00Z",
        "2026-08-06T08:00:00Z",
        inputs=("hdfs://warehouse/original.parquet",),
    )
    restored_window = CheckpointDataWindow.from_payload(saved.to_payload())
    requested = (
        CheckpointDataWindow(
            saved.start,
            saved.end,
            inputs=("hdfs://warehouse/rewritten.parquet",),
        ),
    )
    resumed = SimpleNamespace(
        data_window=restored_window,
        window_complete=False,
        trained_through=saved.start,
    )
    with pytest.raises(ValueError, match="input manifest changed"):
        _CheckpointCoordinator._window_index_after_resume(resumed, requested)


def test_sparse_delta_without_committed_base_is_not_resumable() -> None:
    config = _config()
    with (
        tempfile.TemporaryDirectory() as root_text,
        tempfile.TemporaryDirectory() as local,
    ):
        root = Path(root_text)
        store = LocalCheckpointStore(root)
        model = _ToyModel()
        optimizer = ShardedRowWiseAdagrad(
            [model.embedding.weight], lr=0.1, track_dirty_rows=True
        )
        directory = step_directory_name(10)
        stage_training_checkpoint(
            config,
            model,
            root / directory,
            step=10,
            rows=100,
            sharded_optimizer=optimizer,
            sparse_stream=True,
            sparse_kind="delta",
            generation=2,
            lineage_id="toy",
        )
        _mark_committed(root, directory)

        with pytest.raises(ValueError, match="sparse delta without a full base"):
            fetch_checkpoint_for_rank(
                store,
                CommittedCheckpoint(10, directory, str(root / directory)),
                local,
            )


def test_stale_rank_ready_marker_cannot_commit_a_new_attempt() -> None:
    with tempfile.TemporaryDirectory() as root_text:
        root = Path(root_text)
        store = LocalCheckpointStore(root)
        directory = step_directory_name(10)
        store.write_json(
            {"step": 10, "attempt_id": "old-attempt"},
            directory,
            rank_ready_marker(1),
        )
        staging = root / "stage-rank0"
        staging.mkdir()
        (staging / "rank0.bin").write_bytes(b"rank0")
        uploader = CheckpointUploader(
            store,
            rank=0,
            world_size=2,
            asynchronous=False,
            ready_timeout_sec=0.05,
            poll_interval_sec=0.01,
        )
        uploader.submit(
            StagedCheckpoint(
                step=10,
                staging_dir=staging,
                relative_files=("rank0.bin",),
                cleanup_staging=False,
                attempt_id="new-attempt",
            )
        )
        uploader.close()
        assert not store.exists(directory, COMMIT_MARKER)
        assert uploader.failed_steps == [10]


@pytest.mark.parametrize(
    "name",
    (
        "rankmixer.yaml",
        "rankmixer_fine.yaml",
        "onetrans.yaml",
        "onetrans_fine.yaml",
        "mdl_rankmixer.yaml",
        "mdl_rankmixer_fine.yaml",
        "mdl_onetrans.yaml",
        "mdl_onetrans_fine.yaml",
        "mixformer.yaml",
        "mixformer_fine.yaml",
        "mdl_mixformer.yaml",
        "mdl_mixformer_fine.yaml",
    ),
)
def test_production_configs_use_eight_hour_sparse_delta_checkpoints(name: str) -> None:
    root = Path(__file__).resolve().parents[1]
    checkpoint = load_app_config(root / "configs" / name).training.checkpoint
    assert checkpoint.every_steps == 0
    assert checkpoint.data_window_hours == 8
    assert checkpoint.sparse_delta is True
    assert checkpoint.sparse_full_every == 8
    assert checkpoint.shard_chunk_bytes == 512 * 1024 * 1024
