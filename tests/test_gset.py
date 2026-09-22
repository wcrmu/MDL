from __future__ import annotations

import copy
from dataclasses import replace
import os
from pathlib import Path
import socket

import pytest
import torch
import torch.distributed as torch_dist
import torch.multiprocessing as torch_mp
from torch import nn

from src.config import GSETConfig, load_app_config
from src.model import FeatureEncoderBank, build_model
from src.modules.gset import (
    GSETEmbeddingView,
    GSETNamespacePolicy,
    GlobalSharedEmbeddingTable,
    gset_batch_outcomes,
    gset_row_owner,
    iter_gset_tables,
    owned_gset_policy_records,
)
from src.optim import ShardedRowWiseAdagrad
from src.train import _classify_model_parameters


def _table(capacity: int = 4, **kwargs: object) -> GlobalSharedEmbeddingTable:
    return GlobalSharedEmbeddingTable(
        capacity,
        3,
        sparse=True,
        dtype=torch.float32,
        **kwargs,
    )


def test_global_mapper_is_collision_free_and_probability_zero_rejects() -> None:
    table = _table(capacity=4)
    GSETEmbeddingView(table, "user")
    GSETEmbeddingView(table, "item")
    GSETEmbeddingView(
        table,
        "rare",
        policy=GSETNamespacePolicy(admission_probability=0.0),
    )

    with torch.no_grad():
        _, user_slot = table.lookup("user", torch.tensor([42]), return_slots=True)
        _, item_slot = table.lookup("item", torch.tensor([42]), return_slots=True)
        rare_output, rare_slot = table.lookup(
            "rare", torch.tensor([42]), return_slots=True
        )

    assert user_slot.item() != item_slot.item()
    assert table.key_for_slot(user_slot.item()) == ("user", 42)
    assert table.key_for_slot(item_slot.item()) == ("item", 42)
    assert rare_slot.item() == 0
    assert not table.contains("rare", 42)
    torch.testing.assert_close(rare_output, torch.zeros_like(rare_output))


def test_two_views_of_an_explicitly_shared_namespace_reuse_one_slot() -> None:
    table = _table(capacity=2)
    first = GSETEmbeddingView(table, "shared")
    second = GSETEmbeddingView(table, "shared")

    with torch.no_grad():
        first(torch.tensor([9]))
        second(torch.tensor([9]))

    assert table.stats().active_entries == 1
    assert first.namespace_index == second.namespace_index
    assert first.weight is second.weight is table.weight


def test_label_context_counts_samples_not_duplicate_tokens() -> None:
    table = _table(
        capacity=4,
        score_decay=1.0,
        positive_weight=5.0,
    )
    table.register_namespace("history")
    labels = torch.tensor([[1.0], [0.0], [1.0]])
    label_mask = torch.tensor([[True], [True], [False]])
    table.set_batch_outcomes(labels, label_mask)
    positive, negative = table.batch_outcome_counts(
        2,
        torch.tensor([0, 0, 1]),
    )
    assert positive is not None and negative is not None
    torch.testing.assert_close(positive, torch.tensor([1, 0]))
    torch.testing.assert_close(negative, torch.tensor([1, 0]))

    # ID 7 is duplicated within each request row. It receives request 0's one
    # positive and one negative exactly once each; masked request 1 adds zero.
    with torch.no_grad():
        table.lookup(
            "history",
            torch.tensor([[7, 7], [7, 8]]),
            row_positive_counts=positive,
            row_negative_counts=negative,
        )
    table.force_score_update()

    assert table.feature_score("history", 7) == pytest.approx(6.0)
    assert table.feature_score("history", 8) == pytest.approx(0.0)


def test_flat_bag_scoring_uses_lengths_and_deduplicates_within_rows() -> None:
    table = _table(capacity=4, score_decay=1.0, positive_weight=3.0)
    table.register_namespace("bag")
    with torch.no_grad():
        table.lookup(
            "bag",
            torch.tensor([5, 5, 6, 5]),
            row_positive_counts=torch.tensor([1, 0]),
            row_negative_counts=torch.tensor([0, 1]),
            row_lengths=torch.tensor([3, 1]),
        )
    table.force_score_update()

    # ID 5 is in one positive and one negative sample: 3*1 + 1.
    assert table.feature_score("bag", 5) == pytest.approx(4.0)
    assert table.feature_score("bag", 6) == pytest.approx(3.0)


def test_logical_id_counts_packed_pairs_match_wide_history() -> None:
    table = _table(capacity=32)
    table.register_namespace("seq")
    ids = torch.tensor(
        [
            [7, 7, 8, 0],
            [8, 9, 9, 7],
            [7, 0, 0, 0],
        ]
    )
    counts, inverse = table._logical_id_counts(
        ids,
        padding_idx=0,
        row_positive_counts=torch.tensor([1, 0, 1]),
        row_negative_counts=torch.tensor([0, 1, 0]),
        valid_lengths=torch.tensor([3, 4, 1]),
    )
    assert counts[7] == [4, 2, 1]
    assert counts[8] == [2, 1, 1]
    assert counts[9] == [2, 0, 1]
    assert inverse.tolist() == [0, 0, 1, -1, 1, 2, 2, 0, 0, -1, -1, -1]


def test_logical_id_counts_match_sample_aware_and_padding_rules() -> None:
    table = _table(capacity=8)
    table.register_namespace("seq")
    ids = torch.tensor([[7, 7, 0], [8, 9, 9], [7, 0, 0]])
    counts, inverse = table._logical_id_counts(
        ids,
        padding_idx=0,
        row_positive_counts=torch.tensor([1, 0, 1]),
        row_negative_counts=torch.tensor([0, 1, 0]),
        valid_lengths=torch.tensor([2, 3, 1]),
    )
    assert list(counts) == [7, 8, 9]
    assert counts[7] == [3, 2, 0]
    assert counts[8] == [1, 0, 1]
    assert counts[9] == [2, 0, 1]
    assert inverse.tolist() == [0, 0, -1, 1, 2, 2, 0, -1, -1]


def test_dense_sequence_lengths_exclude_nonzero_padding_ids() -> None:
    table = _table(capacity=6, score_decay=1.0)
    table.register_namespace("sequence")
    with torch.no_grad():
        output = table.lookup(
            "sequence",
            torch.tensor([[5, 99, 99], [6, 7, 98]]),
            row_positive_counts=torch.tensor([1, 0]),
            row_negative_counts=torch.tensor([0, 1]),
            valid_lengths=torch.tensor([1, 2]),
        )
    table.force_score_update()

    assert table.contains("sequence", 5)
    assert table.contains("sequence", 6)
    assert table.contains("sequence", 7)
    assert not table.contains("sequence", 98)
    assert not table.contains("sequence", 99)
    torch.testing.assert_close(output[0, 1:], torch.zeros_like(output[0, 1:]))
    torch.testing.assert_close(output[1, 2], torch.zeros_like(output[1, 2]))


def test_score_eviction_keeps_the_more_valuable_id() -> None:
    table = _table(capacity=2, score_decay=1.0, positive_weight=4.0)
    table.register_namespace("feature")
    with torch.no_grad():
        table.lookup(
            "feature",
            torch.tensor([1, 2]),
            positive_mask=torch.tensor([True, False]),
        )
    table.force_score_update()

    with torch.no_grad():
        table.lookup("feature", torch.tensor([3]))

    assert table.contains("feature", 1)
    assert not table.contains("feature", 2)
    assert table.contains("feature", 3)
    assert table.stats().score_evictions == 1


def test_duration_runs_before_score_and_can_expire_a_priority_row() -> None:
    table = _table(capacity=2)
    table.register_namespace(
        "expiring_priority",
        GSETNamespacePolicy(ttl_steps=1, high_priority=True),
    )
    table.register_namespace("ordinary")
    with torch.no_grad():
        table.lookup("expiring_priority", torch.tensor([10]))
        table.lookup("ordinary", torch.tensor([20]))
    table.advance_step()

    with torch.no_grad():
        table.lookup("ordinary", torch.tensor([30]))

    assert not table.contains("expiring_priority", 10)
    assert table.contains("ordinary", 20)
    assert table.contains("ordinary", 30)
    assert table.stats().duration_evictions == 1


def test_priority_protects_rows_from_feature_score_eviction() -> None:
    table = _table(capacity=2)
    table.register_namespace(
        "priority",
        GSETNamespacePolicy(high_priority=True),
    )
    table.register_namespace("ordinary")
    with torch.no_grad():
        table.lookup("priority", torch.tensor([10]))
        table.lookup("ordinary", torch.tensor([20]))
        table.lookup("ordinary", torch.tensor([30]))

    assert table.contains("priority", 10)
    assert not table.contains("ordinary", 20)
    assert table.contains("ordinary", 30)


def test_gradient_pin_prevents_slot_aliasing_until_optimizer_boundary() -> None:
    table = _table(capacity=1)
    table.register_namespace("feature")
    _, first_slot = table.lookup("feature", torch.tensor([1]), return_slots=True)
    _, blocked_slot = table.lookup("feature", torch.tensor([2]), return_slots=True)
    assert first_slot.item() == 1
    assert blocked_slot.item() == 0
    assert table.contains("feature", 1)

    table.optimizer_step_completed()
    _, second_slot = table.lookup("feature", torch.tensor([2]), return_slots=True)
    assert second_slot.item() == 1
    assert not table.contains("feature", 1)
    assert table.contains("feature", 2)


def test_rowwise_adagrad_state_is_reset_when_a_slot_is_recycled() -> None:
    table = GlobalSharedEmbeddingTable(1, 2, sparse=True, dtype=torch.float32)
    table.register_namespace("feature")
    optimizer = ShardedRowWiseAdagrad(
        [table.weight],
        lr=0.1,
        initial_accumulator_value=0.1,
    )

    first = table.lookup("feature", torch.tensor([1]))
    first.sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    accumulator = optimizer.state[table.weight]["sum"]
    accumulator[1] = 9.0

    second = table.lookup("feature", torch.tensor([2]))
    second.sum().backward()
    optimizer.step()

    # The recycled row starts from 0.1 and receives mean([1^2, 1^2]) == 1.
    assert float(accumulator[1]) == pytest.approx(1.1)
    assert table.key_for_slot(1) == ("feature", 2)


def test_checkpoint_restores_mapper_policy_counters_rng_and_weights() -> None:
    table = _table(
        capacity=3,
        admission_probability=0.4,
        score_decay=0.5,
        seed=17,
    )
    table.register_namespace("a")
    table.register_namespace("b", GSETNamespacePolicy(ttl_steps=3))
    with torch.no_grad():
        table.lookup("a", torch.tensor([1, 2, 3, 4]))
        table.lookup("b", torch.tensor([7, 8]))
    table.advance_step(2)
    state = copy.deepcopy(table.state_dict())

    restored = _table(
        capacity=3,
        admission_probability=0.4,
        score_decay=0.5,
        seed=17,
    )
    restored.register_namespace("a")
    restored.register_namespace("b", GSETNamespacePolicy(ttl_steps=3))
    restored.load_state_dict(state)

    assert restored.stats() == table.stats()
    assert restored.namespace_names == table.namespace_names
    for slot in range(1, table.capacity + 1):
        assert restored.key_for_slot(slot) == table.key_for_slot(slot)
    torch.testing.assert_close(restored.weight, table.weight)
    assert torch.equal(
        restored.get_extra_state()["rng_state"],
        table.get_extra_state()["rng_state"],
    )


def test_eval_lookup_does_not_change_training_score_or_extend_ttl() -> None:
    table = _table(capacity=2, score_decay=1.0, positive_weight=2.0)
    table.register_namespace("feature", GSETNamespacePolicy(ttl_steps=4))
    with torch.no_grad():
        table.lookup(
            "feature",
            torch.tensor([1]),
            positive_mask=torch.tensor([True]),
        )
    table.force_score_update()
    slot = table.slot_for("feature", 1)
    assert slot is not None
    before = table.get_extra_state()
    table.eval()
    with torch.no_grad():
        table.lookup("feature", torch.tensor([1, 1, 1]))
    after = table.get_extra_state()

    assert table.feature_score("feature", 1) == pytest.approx(2.0)
    assert int(after["slot_positive"][slot]) == int(before["slot_positive"][slot])
    assert int(after["slot_negative"][slot]) == int(before["slot_negative"][slot])
    assert int(after["slot_expires_at"][slot]) == int(before["slot_expires_at"][slot])


def test_model_bank_uses_one_physical_gset_and_shared_alias_namespace() -> None:
    root = Path(__file__).resolve().parents[1]
    base = load_app_config(root / "configs" / "reference" / "default.yaml")
    gset = GSETConfig(
        enabled=True,
        capacity=32,
        score_decay=1.0,
        score_task="click",
    )
    config = replace(
        base,
        runtime=replace(
            base.runtime,
            device="cpu",
            precision="fp32",
            attention_backend="sdpa",
            cuda_graph_backbone=False,
        ),
        training=replace(
            base.training,
            sparse_optimizer="rowwise_adagrad",
            embedding_weight_dtype="fp32",
            gset=gset,
        ),
    )
    config.validate()
    bank = FeatureEncoderBank(
        config,
        {},
        config.model.embedding_dim,
        build_sequence_summaries=False,
        embedding_size_override=8,
    )

    table = bank.gset_table
    assert table is not None
    views = [
        module
        for module in bank.embeddings.values()
        if isinstance(module, GSETEmbeddingView)
    ]
    assert views
    assert all(view.table is table and view.weight is table.weight for view in views)
    history_key = bank.sequence_field_embedding_keys["hist.item_id"]
    assert bank.embeddings[history_key] is bank.embeddings["item_id"]
    assert "hist.item_id" not in table.namespace_names

    with gset_batch_outcomes(bank, torch.tensor([[1.0], [0.0]])):
        bank._lookup_id_embedding(
            bank.embeddings["user_id"],
            torch.tensor([11, 11]),
            source_value=torch.tensor([11, 11]),
        )
    table.force_score_update()
    assert table.feature_score("user_id", 11) == pytest.approx(2.0)


def test_gset_score_task_must_name_a_configured_task() -> None:
    root = Path(__file__).resolve().parents[1]
    base = load_app_config(root / "configs" / "reference" / "default.yaml")
    config = replace(
        base,
        training=replace(
            base.training,
            sparse_optimizer="rowwise_adagrad",
            gset=GSETConfig(
                enabled=True,
                capacity=8,
                score_task="missing_task",
            ),
        ),
    )
    with pytest.raises(ValueError, match="unknown task"):
        config.validate()


def test_full_mdl_forward_backward_uses_the_shared_sparse_table() -> None:
    root = Path(__file__).resolve().parents[1]
    base = load_app_config(root / "configs" / "reference" / "default.yaml")
    config = replace(
        base,
        runtime=replace(
            base.runtime,
            device="cpu",
            precision="fp32",
            attention_backend="sdpa",
            cuda_graph_backbone=False,
        ),
        model=replace(base.model, use_request_cache=True),
        training=replace(
            base.training,
            sparse_optimizer="rowwise_adagrad",
            embedding_weight_dtype="fp32",
            gset=GSETConfig(
                enabled=True,
                capacity=64,
                score_decay=1.0,
                score_task="click",
            ),
        ),
    )
    config.validate()
    model = build_model(config, {}, embedding_size_override=8).train()
    batch_size, history_length = 2, 3
    features = {
        "user_id": torch.tensor([1, 2]),
        "item_id": torch.tensor([3, 4]),
        "scenario_user_id": torch.tensor([5, 6]),
        "scenario_item_id": torch.tensor([7, 1]),
        "task_user_id": torch.tensor([2, 3]),
        "task_item_id": torch.tensor([4, 5]),
        "shop_id": torch.tensor([6, 7]),
        "rankmixer_context_dense": torch.randn(batch_size, 16),
        "hist": {
            "fields": {
                "item_id": torch.tensor([[1, 2, 3], [2, 3, 4]]),
                "shop_id": torch.tensor([[3, 4, 5], [4, 5, 6]]),
                "action": torch.tensor([[1, 2, 1], [2, 1, 2]]),
                "age": torch.randn(batch_size, history_length),
                "time_delta": torch.rand(batch_size, history_length),
            },
            "lengths": torch.tensor([3, 2]),
        },
    }
    labels = torch.tensor([[1.0], [0.0]])

    with gset_batch_outcomes(model, labels):
        output = model(features, torch.tensor([0, 0]))
        output["logits"].sum().backward()

    tables = tuple(iter_gset_tables(model))
    assert len(tables) == 1
    table = tables[0]
    assert output["logits"].shape == (batch_size, 1)
    assert table.stats().active_entries > 0
    assert table.weight.grad is not None and table.weight.grad.is_sparse


def test_apply_policy_records_keeps_cpu_mapper_after_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("cuda required")
    table = _table(capacity=4).cuda()
    table.register_namespace("item")
    table.apply_policy_records(
        torch.tensor([[0, 11, 1, 1, 0], [0, 12, 1, 0, 1]], device="cuda")
    )
    assert table._slot_namespace.device.type == "cpu"
    assert table.slot_for("item", 11) == 1
    assert table.slot_for("item", 12) == 2


def test_mapper_stays_on_cpu_when_default_device_is_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("cuda required")
    with torch.device("cuda"):
        table = _table(capacity=4)
        table.register_namespace("item")
        table.apply_policy_records(torch.tensor([[0, 5, 1, 0, 0]]))
    assert table._slot_namespace.device.type == "cpu"
    assert table.slot_for("item", 5) == 1


def test_merge_policy_records_vectorizes_duplicate_keys() -> None:
    from src.modules.gset import merge_gset_policy_records

    records = torch.tensor(
        [
            [1, 9, 1, 1, 0],
            [0, 4, 2, 0, 1],
            [1, 9, 3, 0, 2],
            [0, 4, 1, 1, 0],
        ],
        dtype=torch.int64,
    )
    merged = merge_gset_policy_records(records)
    torch.testing.assert_close(
        merged,
        torch.tensor(
            [[0, 4, 3, 1, 1], [1, 9, 4, 1, 2]],
            dtype=torch.int64,
        ),
    )


def test_merged_policy_records_keep_two_mappers_in_lockstep() -> None:
    from src.modules.gset import merge_gset_policy_records

    rank0 = torch.tensor([[0, 11, 2, 1, 0], [0, 12, 1, 0, 1]], dtype=torch.int64)
    rank1 = torch.tensor([[0, 12, 1, 1, 0], [0, 13, 3, 0, 1]], dtype=torch.int64)
    merged = merge_gset_policy_records(torch.cat([rank0, rank1], dim=0))
    torch.testing.assert_close(
        merged,
        torch.tensor(
            [[0, 11, 2, 1, 0], [0, 12, 2, 1, 1], [0, 13, 3, 0, 1]],
            dtype=torch.int64,
        ),
    )

    left = _table(capacity=8, score_decay=1.0, seed=7)
    right = _table(capacity=8, score_decay=1.0, seed=7)
    left.register_namespace("item")
    right.register_namespace("item")
    left.apply_policy_records(merged)
    right.apply_policy_records(merged)
    left.force_score_update()
    right.force_score_update()
    assert left.slot_for("item", 11) == right.slot_for("item", 11)
    assert left.slot_for("item", 12) == right.slot_for("item", 12)
    assert left.slot_for("item", 13) == right.slot_for("item", 13)
    assert left.feature_score("item", 11) == right.feature_score("item", 11)
    assert left.get_extra_state()["rng_state"].equal(right.get_extra_state()["rng_state"])


def test_prepared_lookup_does_not_double_count_scores() -> None:
    table = _table(capacity=4, score_decay=1.0, positive_weight=2.0)
    table.register_namespace("item")
    table.train()
    records = torch.tensor([[0, 9, 1, 1, 0]], dtype=torch.int64)
    table.apply_synchronized_policy_records(records)
    table.force_score_update()
    slot = table.slot_for("item", 9)
    assert slot is not None
    before = table.feature_score("item", 9)
    positives_before = int(table.get_extra_state()["slot_positive"][slot])
    with torch.no_grad():
        table.lookup(
            "item",
            torch.tensor([9]),
            row_positive_counts=torch.tensor([1]),
            row_negative_counts=torch.tensor([0]),
        )
    assert int(table.get_extra_state()["slot_positive"][slot]) == positives_before
    assert table.feature_score("item", 9) == pytest.approx(before)


def test_gset_row_owner_matches_python_modulo() -> None:
    ids = torch.tensor([0, 1, 2, 3, -1, -5, 7], dtype=torch.int64)
    owners = gset_row_owner(ids, 4)
    expected = torch.tensor([int(value) % 4 for value in ids.tolist()], dtype=torch.int64)
    torch.testing.assert_close(owners, expected)


def test_owned_policy_records_keep_id_mod_residue() -> None:
    records = torch.tensor(
        [[0, 1, 1, 1, 0], [0, 2, 1, 0, 1], [1, 4, 2, 0, 0]],
        dtype=torch.int64,
    )
    rank0 = owned_gset_policy_records(records, rank=0, world_size=2)
    rank1 = owned_gset_policy_records(records, rank=1, world_size=2)
    torch.testing.assert_close(
        rank0,
        torch.tensor([[0, 2, 1, 0, 1], [1, 4, 2, 0, 0]], dtype=torch.int64),
    )
    torch.testing.assert_close(
        rank1,
        torch.tensor([[0, 1, 1, 1, 0]], dtype=torch.int64),
    )


def test_row_sharded_gset_is_classified_as_sharded_parameter() -> None:
    table = GlobalSharedEmbeddingTable(4, 2, row_sharded=True)

    class _Owner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gset_table = table

    groups = _classify_model_parameters(_Owner())
    assert groups.sharded_optimizer == (table.weight,)
    assert groups.embedding_optimizer == ()
    assert groups.sparse_sync == ()


def test_dense_type_embeddings_use_dense_optimizer_not_rowwise_adagrad() -> None:
    """MixFormer action-type tables index .weight directly (dense grads).

    Production MixFormer enables GSET + rowwise Adagrad. Those type tables
    must not share ShardedRowWiseAdagrad with the GSET COO parameter.
    """

    table = GlobalSharedEmbeddingTable(4, 2, row_sharded=True, sparse=True)

    class _Owner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gset_table = table
            self.sequence_type_embeddings = nn.Embedding(8, 2)

        def forward(self, ids: torch.Tensor, type_id: int) -> torch.Tensor:
            embedded = table.lookup("item", ids)
            type_indicator = self.sequence_type_embeddings.weight[type_id]
            return embedded + type_indicator

    model = _Owner()
    table.register_namespace("item")
    groups = _classify_model_parameters(model)
    assert groups.sharded_optimizer == (table.weight,)
    assert groups.embedding_optimizer == ()
    assert model.sequence_type_embeddings.weight in groups.dense_optimizer

    loss = model(torch.tensor([1, 3]), 2).sum()
    loss.backward()
    assert model.sequence_type_embeddings.weight.grad is not None
    assert not model.sequence_type_embeddings.weight.grad.is_sparse
    assert table.weight.grad is not None
    assert table.weight.grad.is_sparse

    dense_optimizer = torch.optim.RMSprop(list(groups.dense_optimizer), lr=0.01)
    sparse_optimizer = ShardedRowWiseAdagrad(
        list(groups.sharded_optimizer),
        lr=0.1,
        initial_accumulator_value=0.1,
    )
    dense_optimizer.step()
    sparse_optimizer.step()


def test_sharded_bank_marks_gset_row_sharded() -> None:
    root = Path(__file__).resolve().parents[1]
    base = load_app_config(root / "configs" / "reference" / "default.yaml")
    config = replace(
        base,
        runtime=replace(
            base.runtime,
            device="cpu",
            precision="fp32",
            attention_backend="sdpa",
            cuda_graph_backbone=False,
        ),
        training=replace(
            base.training,
            embedding_distribution="sharded",
            sparse_optimizer="rowwise_adagrad",
            embedding_weight_dtype="fp32",
            gset=GSETConfig(enabled=True, capacity=16, score_task="click"),
        ),
    )
    config.validate()
    bank = FeatureEncoderBank(
        config,
        {},
        config.model.embedding_dim,
        build_sequence_summaries=False,
        embedding_size_override=8,
    )
    assert bank.gset_table is not None
    assert bank.gset_table.row_sharded
    assert bank.embedding_sharding_plan is None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _init_gloo(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torch_dist.init_process_group("gloo", rank=rank, world_size=world_size)


def _gset_id_mod_worker(rank: int, world_size: int, port: int) -> None:
    _init_gloo(rank, world_size, port)
    try:
        checker = getattr(torch.sparse, "check_sparse_tensor_invariants", None)
        if checker is not None:
            checker.disable()
        table = GlobalSharedEmbeddingTable(8, 4, row_sharded=True, seed=0)
        table.train()
        table.register_namespace("item")
        ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        table.apply_synchronized_policy_records(
            table.count_policy_records("item", ids)
        )
        for logical_id in (1, 2, 3, 4):
            owned = int(logical_id) % world_size == rank
            assert table.contains("item", logical_id) is owned
        output = table.lookup("item", ids)
        gathered = [torch.empty_like(output) for _ in range(world_size)]
        torch_dist.all_gather(gathered, output.contiguous())
        torch.testing.assert_close(gathered[0], gathered[1])
        output.sum().backward()
        grad = table.weight.grad
        assert grad is not None and grad.is_sparse
        rows = set(grad.coalesce().indices()[0].tolist())
        for logical_id in (1, 2, 3, 4):
            slot = table.slot_for("item", logical_id)
            if int(logical_id) % world_size == rank:
                assert slot is not None and slot in rows
            else:
                assert slot is None
    finally:
        torch_dist.destroy_process_group()


def test_row_sharded_gset_routes_ids_with_modulo_ownership() -> None:
    port = _free_port()
    torch_mp.spawn(
        _gset_id_mod_worker,
        args=(2, port),
        nprocs=2,
        join=True,
    )
