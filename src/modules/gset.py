"""Kraken-style Global Shared Embedding Table (GSET).

This is a correctness-first PyTorch implementation of the mechanisms described
in Kraken:

* a collision-free global mapper keyed by ``(feature namespace, logical ID)``;
* one bounded physical embedding pool shared by all feature namespaces;
* probability-based admission for previously unseen IDs;
* feature-score, duration, and priority-aware eviction; and
* safe slot reuse, including reinitializing both the parameter row and the
  row-wise optimizer accumulator.

The key metadata deliberately stays on CPU.  It is control-plane state (similar
to an in-memory KV-store index), while the embedding weight remains on the
model device.  Lookup output and gradients use ordinary sparse
``torch.nn.functional.embedding`` semantics.

When ``row_sharded=True`` the physical pool is local to each rank and logical
IDs are routed with the online rule ``owner = id % world_size``.  Admission and
eviction then run only on the owner; lookups exchange IDs and embedding rows
with variable-size ``all_to_all``.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
import heapq
import math
import threading
import weakref
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
import torch.distributed as dist

from ..embeddings import (
    _active_id_mask,
    _all_to_all_variable,
    _distributed_rank_world,
    _exchange_and_host_splits,
    _process_group_backend,
)


_GSET_STATE_VERSION = 1
_PARAMETER_OWNERS: dict[int, weakref.ReferenceType["GlobalSharedEmbeddingTable"]] = {}
_DRACARYS_GLOBAL_NAMESPACE = "__dracarys_global__"


def dracarys_feature_hash64(feature_name: str) -> int:
    """Return the stable signed-int64 hash used in Dracarys feature XOR IDs."""

    if not isinstance(feature_name, str) or not feature_name:
        raise ValueError("Dracarys feature name must be a non-empty string")
    unsigned = int.from_bytes(
        sha256(feature_name.encode("utf-8")).digest()[:8],
        "little",
        signed=False,
    )
    return unsigned if unsigned < (1 << 63) else unsigned - (1 << 64)


def dracarys_feature_xor_ids(logical_ids: Tensor, feature_name: str) -> Tensor:
    """Compute ``hash(feature_name) XOR raw_int64_value`` bit-for-bit."""

    if logical_ids.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("Dracarys logical IDs must use an integer tensor dtype")
    values = logical_ids.to(dtype=torch.int64)
    feature_hash = torch.tensor(
        dracarys_feature_hash64(feature_name),
        dtype=torch.int64,
        device=values.device,
    )
    return torch.bitwise_xor(values, feature_hash)


_GSET_POLICY_RECORD_WIDTH = 5
# Hash-capped / mock IDs live in this range.  Vector lookup avoids a Python
# dict insert per admitted key on the 8e7-row production table.
_GSET_DENSE_ID_LIMIT = 1 << 18


def empty_gset_policy_records() -> Tensor:
    return torch.zeros((0, _GSET_POLICY_RECORD_WIDTH), dtype=torch.int64)


def merge_gset_policy_records(records: Tensor) -> Tensor:
    """Sum occupancy/score counts by ``(namespace, logical_id)`` in sorted order."""

    if records.ndim != 2 or int(records.size(1)) != _GSET_POLICY_RECORD_WIDTH:
        raise ValueError("GSET policy records must have shape [N, 5]")
    if records.numel() == 0:
        return empty_gset_policy_records()
    payload = records.detach().to(device="cpu", dtype=torch.int64)
    if payload.size(0) == 1:
        return payload
    # Sort-scan is much cheaper than unique([N, 2]) at MixFormer field-cat size.
    by_id = torch.argsort(payload[:, 1], stable=True)
    by_key = by_id[torch.argsort(payload.index_select(0, by_id)[:, 0], stable=True)]
    ordered = payload.index_select(0, by_key)
    new_group = torch.ones(ordered.size(0), dtype=torch.bool, device="cpu")
    new_group[1:] = (ordered[1:, :2] != ordered[:-1, :2]).any(dim=1)
    group = new_group.cumsum(0) - 1
    n_group = int(group[-1].item()) + 1
    occurrences = torch.zeros(n_group, dtype=torch.int64, device="cpu")
    positives = torch.zeros(n_group, dtype=torch.int64, device="cpu")
    negatives = torch.zeros(n_group, dtype=torch.int64, device="cpu")
    occurrences.scatter_add_(0, group, ordered[:, 2])
    positives.scatter_add_(0, group, ordered[:, 3])
    negatives.scatter_add_(0, group, ordered[:, 4])
    starts = torch.nonzero(new_group, as_tuple=False).flatten()
    keys = ordered.index_select(0, starts)
    return torch.stack(
        (
            keys[:, 0],
            keys[:, 1],
            occurrences,
            positives,
            negatives,
        ),
        dim=1,
    )


def gather_and_merge_gset_policy_records(
    local_records: Tensor,
    *,
    device: torch.device | None = None,
) -> Tensor:
    """All-gather per-rank GSET counts, then merge them in a deterministic order.

    Ranks that see different microbatches still apply the same admission,
    eviction, and score updates when every rank consumes this merged stream.
    """

    merged_local = merge_gset_policy_records(local_records)
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return merged_local
    world_size = dist.get_world_size()
    if device is None or device.type != "cuda":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    count = torch.tensor([int(merged_local.size(0))], dtype=torch.long, device=device)
    gathered_counts = [torch.empty_like(count) for _ in range(world_size)]
    dist.all_gather(gathered_counts, count)
    counts = [int(item.item()) for item in gathered_counts]
    max_rows = max(counts)
    if max_rows == 0:
        return empty_gset_policy_records()
    padded = torch.zeros((max_rows, _GSET_POLICY_RECORD_WIDTH), dtype=torch.int64, device=device)
    if merged_local.size(0):
        padded[: merged_local.size(0)] = merged_local.to(device=device, dtype=torch.int64)
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    parts = [
        rows[:row_count].detach().to(device="cpu")
        for rows, row_count in zip(gathered, counts)
        if row_count
    ]
    if not parts:
        return empty_gset_policy_records()
    return merge_gset_policy_records(torch.cat(parts, dim=0))


def gset_row_owner(logical_ids: Tensor, world_size: int) -> Tensor:
    """Return the online owner rank ``id % world_size`` for each logical ID."""

    if type(world_size) is not int or world_size <= 0:
        raise ValueError("GSET world_size must be a positive integer")
    return torch.remainder(logical_ids, world_size)


def owned_gset_policy_records(
    records: Tensor,
    *,
    rank: int,
    world_size: int,
) -> Tensor:
    """Keep policy rows whose logical ID is owned by ``rank``."""

    if records.numel() == 0:
        return empty_gset_policy_records()
    if records.ndim != 2 or int(records.size(1)) != _GSET_POLICY_RECORD_WIDTH:
        raise ValueError("GSET policy records must have shape [N, 5]")
    owners = gset_row_owner(records[:, 1].to(dtype=torch.int64), world_size)
    mask = owners == int(rank)
    if not bool(mask.any()):
        return empty_gset_policy_records()
    return records[mask]


def _gset_policy_collective_device(
    preferred: torch.device | None,
    process_group: dist.ProcessGroup | None,
) -> torch.device:
    backend = _process_group_backend(process_group)
    if backend == "nccl":
        if preferred is not None and preferred.type == "cuda":
            return preferred
        return torch.device("cuda")
    return torch.device("cpu")


def exchange_owned_gset_policy_records(
    local_records: Tensor,
    *,
    process_group: dist.ProcessGroup | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Send ``(namespace, id)`` counts to the rank that owns ``id % world_size``."""

    merged_local = merge_gset_policy_records(local_records)
    _rank, world_size = _distributed_rank_world(process_group)
    if world_size <= 1:
        return merged_local
    payload_device = _gset_policy_collective_device(device, process_group)
    if merged_local.size(0) == 0:
        payload = torch.zeros(
            (0, _GSET_POLICY_RECORD_WIDTH),
            dtype=torch.int64,
            device=payload_device,
        )
        owners = torch.zeros((0,), dtype=torch.long, device=payload_device)
    else:
        payload = merged_local.to(device=payload_device, dtype=torch.int64)
        owners = gset_row_owner(payload[:, 1], world_size)
        send_order = torch.argsort(owners, stable=True)
        payload = payload.index_select(0, send_order)
        owners = owners.index_select(0, send_order)
    send_splits_tensor = torch.bincount(owners, minlength=world_size)
    send_splits, recv_splits = _exchange_and_host_splits(
        send_splits_tensor,
        process_group,
    )
    received = _all_to_all_variable(
        payload,
        send_splits,
        recv_splits,
        process_group,
    )
    return merge_gset_policy_records(received.detach().cpu())


def _gset_active_id_mask(
    logical_ids: Tensor,
    *,
    padding_idx: int | None,
    allow_negative_ids: bool,
) -> Tensor:
    if allow_negative_ids:
        if padding_idx is None:
            return torch.ones_like(logical_ids, dtype=torch.bool)
        return logical_ids != int(padding_idx)
    return _active_id_mask(logical_ids, padding_idx)


@dataclass(frozen=True)
class GSETNamespacePolicy:
    """Per-feature policy used by Kraken's adaptive replacement layer."""

    admission_probability: float = 1.0
    ttl_steps: int | None = None
    high_priority: bool = False

    def validate(self, namespace: str) -> None:
        if not 0.0 <= float(self.admission_probability) <= 1.0:
            raise ValueError(
                f"GSET namespace {namespace!r} admission_probability must be in [0, 1]"
            )
        if self.ttl_steps is not None and (
            type(self.ttl_steps) is not int or self.ttl_steps <= 0
        ):
            raise ValueError(
                f"GSET namespace {namespace!r} ttl_steps must be a positive integer or null"
            )
        if type(self.high_priority) is not bool:
            raise ValueError(
                f"GSET namespace {namespace!r} high_priority must be boolean"
            )


@dataclass(frozen=True)
class GSETStats:
    capacity: int
    active_entries: int
    free_entries: int
    lookups: int
    hits: int
    misses: int
    admissions: int
    rejected: int
    evictions: int
    duration_evictions: int
    score_evictions: int
    current_step: int


class GlobalSharedEmbeddingTable(nn.Embedding):
    """A bounded, collision-free, globally shared dynamic embedding table.

    Row ``0`` is a permanent zero fallback for missing, filtered, or rejected
    IDs.  Active entries occupy rows ``1..capacity``.  A namespace is part of
    every logical key, so equal numeric IDs from unrelated features never
    collide; explicitly shared embeddings reuse the same namespace view.

    Eviction never reuses a row touched since the last optimizer step.  This
    prevents two logical keys from contributing gradients to one physical row
    during gradient accumulation.  ``ShardedRowWiseAdagrad`` consumes the
    pending-reset rows before applying an update and then calls
    :meth:`optimizer_step_completed`.

    ``row_sharded`` follows the online embedding placement ``id % world_size
    == rank``.  Each rank then owns an independent mapper and physical pool
    for that residue class.
    """

    def __init__(
        self,
        capacity: int,
        embedding_dim: int,
        *,
        init_std: float = 0.02,
        sparse: bool = True,
        dtype: torch.dtype = torch.float32,
        admission_probability: float = 1.0,
        score_decay: float = 0.1,
        positive_weight: float = 1.0,
        score_update_interval: int = 1,
        default_ttl_steps: int | None = None,
        eviction_enabled: bool = True,
        eviction_policy: str = "score",
        seed: int = 2025,
        row_sharded: bool = False,
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("GSET capacity must be a positive integer")
        if type(embedding_dim) is not int or embedding_dim <= 0:
            raise ValueError("GSET embedding_dim must be a positive integer")
        if not math.isfinite(float(init_std)) or init_std <= 0.0:
            raise ValueError("GSET init_std must be positive and finite")
        if not 0.0 <= float(admission_probability) <= 1.0:
            raise ValueError("GSET admission_probability must be in [0, 1]")
        if not 0.0 <= float(score_decay) <= 1.0:
            raise ValueError("GSET score_decay must be in [0, 1]")
        if not math.isfinite(float(positive_weight)) or positive_weight < 0.0:
            raise ValueError("GSET positive_weight must be finite and non-negative")
        if type(score_update_interval) is not int or score_update_interval <= 0:
            raise ValueError("GSET score_update_interval must be positive")
        if default_ttl_steps is not None and (
            type(default_ttl_steps) is not int or default_ttl_steps <= 0
        ):
            raise ValueError("GSET default_ttl_steps must be positive or null")
        if type(eviction_enabled) is not bool:
            raise ValueError("GSET eviction_enabled must be boolean")
        if eviction_policy not in {"score", "lru"}:
            raise ValueError("GSET eviction_policy must be score or lru")
        if type(seed) is not int:
            raise ValueError("GSET seed must be an integer")

        # The physical table has one additional immutable fallback row.
        super().__init__(
            capacity + 1,
            embedding_dim,
            padding_idx=0,
            sparse=sparse,
            dtype=dtype,
        )
        self.capacity = int(capacity)
        self.init_std = float(init_std)
        self.default_admission_probability = float(admission_probability)
        self.score_decay = float(score_decay)
        self.positive_weight = float(positive_weight)
        self.score_update_interval = int(score_update_interval)
        self.default_ttl_steps = default_ttl_steps
        self.eviction_enabled = eviction_enabled
        self.eviction_policy = eviction_policy
        self.seed = seed
        self.row_sharded = bool(row_sharded)
        self.process_group = process_group

        with torch.no_grad():
            self.weight.normal_(mean=0.0, std=self.init_std)
            self.weight[0].zero_()

        # CPU-resident KV metadata.  It is serialized through extra_state so
        # model.to(cuda) does not migrate a cache-management data structure.
        # Pin the factory device: constructing the table after
        # ``torch.set_default_device("cuda")`` would otherwise put the 8e7-row
        # mapper on the GPU and break host-side index writes.
        slots = self.capacity + 1
        host = torch.device("cpu")
        self._slot_namespace = torch.full((slots,), -1, dtype=torch.int64, device=host)
        self._slot_key = torch.zeros((slots,), dtype=torch.int64, device=host)
        self._slot_score = torch.zeros((slots,), dtype=torch.float64, device=host)
        self._slot_positive = torch.zeros((slots,), dtype=torch.int64, device=host)
        self._slot_negative = torch.zeros((slots,), dtype=torch.int64, device=host)
        self._slot_last_access = torch.full((slots,), -1, dtype=torch.int64, device=host)
        self._slot_expires_at = torch.full((slots,), -1, dtype=torch.int64, device=host)
        self._slot_high_priority = torch.zeros((slots,), dtype=torch.bool, device=host)
        self._slot_pinned = torch.zeros((slots,), dtype=torch.bool, device=host)
        self._pending_optimizer_reset = torch.zeros((slots,), dtype=torch.bool, device=host)
        self._slot_generation = torch.zeros((slots,), dtype=torch.int64, device=host)

        self._namespace_names: list[str] = []
        self._namespace_to_id: dict[str, int] = {}
        self._namespace_policies: list[GSETNamespacePolicy] = []
        self._key_to_slot: dict[tuple[int, int], int] = {}
        # Never materialize ``range(1, capacity+1)``. Production Kraken
        # capacities are 8e7 rows; a Python min-heap of that size is the
        # host-side stall that kept 4090s at 0% util. Virgin slots are an
        # integer cursor; only evicted holes (checkpoint restore) go on a heap.
        self._next_virgin_slot = 1
        self._recycled_slots: list[int] = []
        self._dense_id_limit = _GSET_DENSE_ID_LIMIT
        self._dense_slots: Tensor | None = None
        self._current_step = 0
        self._last_score_update_step = 0
        self._rng = torch.Generator(device="cpu")
        rank, _world_size = _distributed_rank_world(process_group)
        self._rng.manual_seed(int(seed) + int(rank) if self.row_sharded else int(seed))
        self._lock = threading.RLock()

        self._lookups = 0
        self._hits = 0
        self._misses = 0
        self._admissions = 0
        self._rejected = 0
        self._evictions = 0
        self._duration_evictions = 0
        self._score_evictions = 0

        # Transient label context for the current training microbatch. It is
        # intentionally absent from checkpoints: activation recomputation uses
        # it within the same forward/backward, then the trainer clears it.
        self._batch_positive: Tensor | None = None
        self._batch_valid: Tensor | None = None
        # Set after a rank-union policy pass so later lookups in the same
        # microbatch (including activation recomputation) map slots without
        # admitting or updating scores a second time.
        self._policy_prepared = False

        # Optimizers discover the control-plane owner without registering a
        # second module/parameter path in the model state_dict.
        owner = weakref.ref(self)
        _PARAMETER_OWNERS[id(self.weight)] = owner
        try:
            self.weight._mdl_gset_owner = owner  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError):
            pass
        self._mdl_id_embedding = True

    def register_namespace(
        self,
        name: str,
        policy: GSETNamespacePolicy | None = None,
    ) -> int:
        """Register one logical feature domain and return its stable ID."""

        if not isinstance(name, str) or not name:
            raise ValueError("GSET namespace name must be a non-empty string")
        resolved = policy or GSETNamespacePolicy(
            admission_probability=self.default_admission_probability,
            ttl_steps=self.default_ttl_steps,
        )
        resolved.validate(name)
        with self._lock:
            existing = self._namespace_to_id.get(name)
            if existing is not None:
                if self._namespace_policies[existing] != resolved:
                    raise ValueError(
                        f"GSET namespace {name!r} was registered with a different policy"
                    )
                return existing
            namespace_id = len(self._namespace_names)
            self._namespace_names.append(name)
            self._namespace_to_id[name] = namespace_id
            self._namespace_policies.append(resolved)
            self._ensure_dense_mapper()
            return namespace_id

    @property
    def namespace_names(self) -> tuple[str, ...]:
        return tuple(self._namespace_names)

    def namespace_id(self, name: str) -> int:
        try:
            return self._namespace_to_id[name]
        except KeyError as error:
            raise KeyError(f"unknown GSET namespace {name!r}") from error

    def namespace_policy(self, namespace: str | int) -> GSETNamespacePolicy:
        namespace_id = (
            self.namespace_id(namespace) if isinstance(namespace, str) else namespace
        )
        if not 0 <= int(namespace_id) < len(self._namespace_policies):
            raise KeyError(f"unknown GSET namespace id {namespace_id!r}")
        return self._namespace_policies[int(namespace_id)]

    def _resolve_namespace(self, namespace: str | int) -> int:
        if isinstance(namespace, str):
            return self.namespace_id(namespace)
        if type(namespace) is not int:
            raise TypeError("GSET namespace must be a string or integer ID")
        if not 0 <= namespace < len(self._namespace_names):
            raise KeyError(f"unknown GSET namespace id {namespace}")
        return namespace

    def _available_free_slots(self) -> int:
        virgin = max(0, self.capacity - self._next_virgin_slot + 1)
        return virgin + len(self._recycled_slots)

    def _ensure_dense_mapper(self) -> None:
        n_ns = len(self._namespace_names)
        limit = int(self._dense_id_limit)
        if n_ns <= 0:
            return
        if self._dense_slots is None:
            self._dense_slots = torch.zeros(
                (n_ns, limit),
                dtype=torch.long,
                device="cpu",
            )
            return
        if self._dense_slots.size(0) < n_ns:
            grown = torch.zeros((n_ns, limit), dtype=torch.long, device="cpu")
            grown[: self._dense_slots.size(0)] = self._dense_slots
            self._dense_slots = grown

    def _lookup_mapped_slots(
        self,
        namespace_ids: Tensor,
        logical_ids: Tensor,
    ) -> Tensor:
        namespace_ids = namespace_ids.detach().to(device="cpu", dtype=torch.int64)
        logical_ids = logical_ids.detach().to(device="cpu", dtype=torch.int64)
        slots = torch.zeros(logical_ids.numel(), dtype=torch.long, device="cpu")
        self._ensure_dense_mapper()
        dense = self._dense_slots
        if dense is not None and logical_ids.numel():
            in_range = (
                (logical_ids >= 0)
                & (logical_ids < dense.size(1))
                & (namespace_ids >= 0)
                & (namespace_ids < dense.size(0))
            )
            if bool(in_range.any()):
                slots[in_range] = dense[namespace_ids[in_range], logical_ids[in_range]]
            missing = ~in_range
        else:
            missing = torch.ones(logical_ids.numel(), dtype=torch.bool, device="cpu")
        if bool(missing.any()) and self._key_to_slot:
            for index, namespace_id, logical_id in zip(
                torch.nonzero(missing, as_tuple=False).flatten().tolist(),
                namespace_ids[missing].tolist(),
                logical_ids[missing].tolist(),
            ):
                slot = self._key_to_slot.get((int(namespace_id), int(logical_id)))
                if slot is not None:
                    slots[index] = int(slot)
        return slots

    def _store_mapped_slots(
        self,
        namespace_ids: Tensor,
        logical_ids: Tensor,
        slots: Tensor,
    ) -> None:
        namespace_ids = namespace_ids.detach().to(device="cpu", dtype=torch.int64)
        logical_ids = logical_ids.detach().to(device="cpu", dtype=torch.int64)
        slots = slots.detach().to(device="cpu", dtype=torch.long)
        self._ensure_dense_mapper()
        dense = self._dense_slots
        if dense is not None and slots.numel():
            in_range = (
                (logical_ids >= 0)
                & (logical_ids < dense.size(1))
                & (namespace_ids >= 0)
                & (namespace_ids < dense.size(0))
            )
            if bool(in_range.any()):
                dense[namespace_ids[in_range], logical_ids[in_range]] = slots[in_range]
            missing = ~in_range
        else:
            missing = torch.ones(slots.numel(), dtype=torch.bool, device="cpu")
        if bool(missing.any()):
            for namespace_id, logical_id, slot in zip(
                namespace_ids[missing].tolist(),
                logical_ids[missing].tolist(),
                slots[missing].tolist(),
            ):
                self._key_to_slot[(int(namespace_id), int(logical_id))] = int(slot)

    def _forget_mapped_slot(
        self,
        namespace_id: int,
        logical_id: int,
        slot: int,
    ) -> None:
        dense_hit = False
        dense = self._dense_slots
        if (
            dense is not None
            and 0 <= namespace_id < int(dense.size(0))
            and 0 <= logical_id < int(dense.size(1))
        ):
            current = int(dense[namespace_id, logical_id])
            if current == int(slot):
                dense[namespace_id, logical_id] = 0
                dense_hit = True
        mapped = self._key_to_slot.pop((namespace_id, logical_id), None)
        if mapped is None and not dense_hit:
            raise RuntimeError("GSET mapper metadata is inconsistent during eviction")
        if mapped is not None and mapped != int(slot):
            raise RuntimeError("GSET mapper metadata is inconsistent during eviction")

    def _take_free_slots(self, count: int) -> Tensor:
        """Pop ``count`` free slots, smallest index first."""

        if type(count) is not int or count < 0:
            raise ValueError("GSET free-slot count must be a non-negative integer")
        if count == 0:
            return torch.zeros((0,), dtype=torch.long, device="cpu")
        if self._available_free_slots() < count:
            raise RuntimeError("GSET requested more free slots than remain")
        if not self._recycled_slots:
            start = self._next_virgin_slot
            self._next_virgin_slot += count
            return torch.arange(
                start, start + count, dtype=torch.long, device="cpu"
            )
        slots = torch.empty((count,), dtype=torch.long, device="cpu")
        for index in range(count):
            take_recycled = bool(self._recycled_slots) and (
                self._next_virgin_slot > self.capacity
                or self._recycled_slots[0] < self._next_virgin_slot
            )
            if take_recycled:
                slots[index] = heapq.heappop(self._recycled_slots)
            else:
                slots[index] = self._next_virgin_slot
                self._next_virgin_slot += 1
        return slots

    def _namespace_admission_probs(self) -> Tensor:
        return torch.tensor(
            [policy.admission_probability for policy in self._namespace_policies],
            dtype=torch.float64,
            device="cpu",
        )

    def _namespace_ttl_steps(self) -> Tensor:
        return torch.tensor(
            [
                -1 if policy.ttl_steps is None else int(policy.ttl_steps)
                for policy in self._namespace_policies
            ],
            dtype=torch.int64,
            device="cpu",
        )

    def _namespace_high_priority(self) -> Tensor:
        return torch.tensor(
            [bool(policy.high_priority) for policy in self._namespace_policies],
            dtype=torch.bool,
            device="cpu",
        )

    def _masked_lexmin_slot(self, mask: Tensor, *keys: Tensor) -> int | None:
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        if idx.numel() == 0:
            return None
        if idx.numel() == 1 or not keys:
            return int(idx[0].item())
        order = torch.arange(idx.numel(), dtype=torch.long)
        for key in reversed(keys):
            values = key.index_select(0, idx).index_select(0, order)
            _sorted, perm = values.sort(stable=True)
            order = order.index_select(0, perm)
        return int(idx[order[0]].item())

    def _refresh_scores(self, *, force: bool = False) -> None:
        elapsed = self._current_step - self._last_score_update_step
        if not force and elapsed < self.score_update_interval:
            return
        active = self._slot_namespace >= 0
        active[0] = False
        if bool(active.any()):
            interval_value = self.positive_weight * self._slot_positive.to(
                torch.float64
            ) + self._slot_negative.to(torch.float64)
            beta = self.score_decay
            self._slot_score[active] = (1.0 - beta) * self._slot_score[
                active
            ] + beta * interval_value[active]
            self._slot_positive[active] = 0
            self._slot_negative[active] = 0
        self._last_score_update_step = self._current_step

    def force_score_update(self) -> None:
        """Apply the Kraken feature-score equation immediately."""

        with self._lock:
            self._refresh_scores(force=True)

    def _evictable_mask(self, excluded_slots: Iterable[int] = ()) -> Tensor:
        mask = self._slot_namespace >= 0
        mask[0] = False
        mask &= ~self._slot_pinned
        excluded = tuple(int(slot) for slot in excluded_slots)
        if excluded:
            mask[torch.tensor(excluded, dtype=torch.long)] = False
        return mask

    def _select_victim(
        self,
        excluded_slots: Iterable[int] = (),
    ) -> tuple[int, str] | None:
        if not self.eviction_enabled:
            return None
        evictable = self._evictable_mask(excluded_slots)
        if not bool(evictable.any()):
            return None

        slot_ids = torch.arange(
            self._slot_namespace.numel(),
            dtype=torch.long,
            device=self._slot_namespace.device,
        )
        if self.eviction_policy == "lru":
            slot = self._masked_lexmin_slot(
                evictable,
                self._slot_last_access,
                slot_ids,
            )
            if slot is None:
                return None
            return slot, "lru"

        self._refresh_scores()

        # Duration-based garbage collection runs before feature-score eviction.
        # Kraken's priority classes constrain score replacement; an explicitly
        # expired row has completed its configured lifecycle regardless of its
        # score-priority class.
        expired = (
            evictable
            & (self._slot_expires_at >= 0)
            & (self._slot_expires_at <= self._current_step)
        )
        if bool(expired.any()):
            slot = self._masked_lexmin_slot(
                expired,
                self._slot_expires_at,
                self._slot_last_access,
                slot_ids,
            )
            if slot is None:
                return None
            return slot, "duration"

        score_evictable = evictable & ~self._slot_high_priority
        if not bool(score_evictable.any()):
            return None
        slot = self._masked_lexmin_slot(
            score_evictable,
            self._slot_score,
            self._slot_last_access,
            slot_ids,
        )
        if slot is None:
            return None
        return slot, "score"

    def _release_slot(self, slot: int, reason: str) -> None:
        namespace_id = int(self._slot_namespace[slot])
        logical_id = int(self._slot_key[slot])
        if namespace_id < 0:
            raise RuntimeError("cannot evict an unoccupied GSET slot")
        self._forget_mapped_slot(namespace_id, logical_id, slot)
        self._evictions += 1
        if reason == "duration":
            self._duration_evictions += 1
        else:
            self._score_evictions += 1

    def _allocate_slot(
        self,
        namespace_id: int,
        logical_id: int,
        policy: GSETNamespacePolicy,
        excluded_slots: Iterable[int] = (),
    ) -> int | None:
        reason = "free"
        if self._available_free_slots() > 0:
            slot = int(self._take_free_slots(1)[0].item())
        else:
            victim = self._select_victim(excluded_slots)
            if victim is None:
                return None
            slot, reason = victim
            self._release_slot(slot, reason)

        self._slot_namespace[slot] = namespace_id
        self._slot_key[slot] = logical_id
        self._slot_score[slot] = 0.0
        self._slot_positive[slot] = 0
        self._slot_negative[slot] = 0
        self._slot_last_access[slot] = self._current_step
        self._slot_expires_at[slot] = (
            -1
            if policy.ttl_steps is None
            else self._current_step + int(policy.ttl_steps)
        )
        self._slot_high_priority[slot] = policy.high_priority
        self._slot_pinned[slot] = False
        self._pending_optimizer_reset[slot] = True
        self._slot_generation[slot] += 1
        self._store_mapped_slots(
            torch.tensor([namespace_id], dtype=torch.int64, device="cpu"),
            torch.tensor([logical_id], dtype=torch.int64, device="cpu"),
            torch.tensor([slot], dtype=torch.long, device="cpu"),
        )
        return slot

    def _initialize_parameter_rows(self, slots: Iterable[int] | Tensor) -> None:
        if isinstance(slots, Tensor):
            unique = torch.unique(slots.detach().to(device="cpu", dtype=torch.long))
            unique = unique[unique > 0]
            if unique.numel() == 0:
                return
            device_slots = unique.to(device=self.weight.device)
        else:
            unique_list = sorted(set(int(slot) for slot in slots))
            if not unique_list:
                return
            device_slots = torch.tensor(
                unique_list,
                dtype=torch.long,
                device=self.weight.device,
            )
        values = torch.empty(
            int(device_slots.numel()),
            self.embedding_dim,
            dtype=self.weight.dtype,
            device=self.weight.device,
        ).normal_(mean=0.0, std=self.init_std)
        with torch.no_grad():
            self.weight.index_copy_(0, device_slots, values)
            self.weight[0].zero_()

    def _admission_draw(self, probability: float, occurrences: int) -> bool:
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        # One or more independent Bernoulli trials have this aggregate
        # probability.  It preserves the paper's geometric waiting time while
        # processing duplicate IDs in one batch without a Python loop per row.
        aggregate = 1.0 - (1.0 - probability) ** max(1, occurrences)
        draw = float(torch.rand((), generator=self._rng).item())
        return draw < aggregate

    def set_batch_outcomes(
        self,
        labels: Tensor,
        label_mask: Tensor | None = None,
        *,
        task_index: int = 0,
        active: bool = True,
    ) -> None:
        """Attach one binary-label stream to subsequent training lookups.

        Kraken's feature score counts *samples containing an ID*, rather than
        raw token occurrences. Lookups combine this candidate-level context
        with request ``row_indices`` and bag ``lengths`` so repeated IDs are
        counted once per sample, including request-deduplicated batches.
        """

        if labels.ndim == 1:
            labels = labels.unsqueeze(1)
        if labels.ndim != 2:
            raise ValueError("GSET labels must have shape [samples, tasks]")
        if type(task_index) is not int or not 0 <= task_index < labels.size(1):
            raise ValueError("GSET task_index is outside the label task axis")
        if label_mask is None:
            valid = torch.ones(
                labels.size(0),
                dtype=torch.bool,
                device=labels.device,
            )
        else:
            if label_mask.ndim == 1:
                label_mask = label_mask.unsqueeze(1)
            if label_mask.shape != labels.shape:
                raise ValueError("GSET label_mask must have the same shape as labels")
            valid = label_mask[:, task_index].bool()
        if not active:
            valid = torch.zeros_like(valid)
        positive = labels[:, task_index] > 0.5
        with self._lock:
            self._batch_positive = positive.detach().to(device="cpu")
            self._batch_valid = valid.detach().to(device="cpu")
            self._policy_prepared = False

    def clear_batch_outcomes(self) -> None:
        """Release the transient label tensors for the completed microbatch."""

        with self._lock:
            self._batch_positive = None
            self._batch_valid = None
            self._policy_prepared = False

    def batch_outcome_counts(
        self,
        row_count: int,
        row_indices: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Return positive/negative sample counts for physical feature rows."""

        if type(row_count) is not int or row_count < 0:
            raise ValueError("GSET row_count must be a non-negative integer")
        with self._lock:
            positive = self._batch_positive
            valid = self._batch_valid
            if positive is None or valid is None:
                return None, None
            positive_values = (positive & valid).to(torch.int64)
            negative_values = ((~positive) & valid).to(torch.int64)
            if row_indices is None:
                if row_count != positive_values.numel():
                    raise ValueError(
                        "GSET request-sized inputs require candidate-to-request "
                        "row_indices while label-aware scoring is active"
                    )
                return positive_values.clone(), negative_values.clone()

            rows = row_indices.detach().to(device="cpu", dtype=torch.int64)
            if rows.ndim != 1 or rows.numel() != positive_values.numel():
                raise ValueError(
                    "GSET row_indices must be rank one and match the label batch"
                )
            if rows.numel() and bool(((rows < 0) | (rows >= row_count)).any()):
                raise ValueError(
                    "GSET row_indices contains an out-of-range request row"
                )
            row_positive = torch.zeros(row_count, dtype=torch.int64)
            row_negative = torch.zeros(row_count, dtype=torch.int64)
            if rows.numel():
                row_positive.index_add_(0, rows, positive_values)
                row_negative.index_add_(0, rows, negative_values)
            return row_positive, row_negative

    def count_policy_records(
        self,
        namespace: str | int,
        logical_ids: Tensor,
        *,
        padding_idx: int | None = None,
        row_positive_counts: Tensor | None = None,
        row_negative_counts: Tensor | None = None,
        row_lengths: Tensor | None = None,
        valid_lengths: Tensor | None = None,
        allow_negative_ids: bool = False,
    ) -> Tensor:
        """Return ``[N, 5]`` records of ``(namespace, id, occ, pos, neg)``."""

        unique_ids, occurrences, positives, negatives, _inverse = (
            self._logical_id_count_tensors(
                logical_ids,
                padding_idx=padding_idx,
                positive_mask=None,
                row_positive_counts=row_positive_counts,
                row_negative_counts=row_negative_counts,
                row_lengths=row_lengths,
                valid_lengths=valid_lengths,
                allow_negative_ids=allow_negative_ids,
            )
        )
        if unique_ids.numel() == 0:
            return empty_gset_policy_records()
        namespace_id = self._resolve_namespace(namespace)
        namespace_col = torch.full_like(unique_ids, int(namespace_id))
        return torch.stack(
            (namespace_col, unique_ids, occurrences, positives, negatives),
            dim=1,
        )

    def _touch_resident_slots(
        self,
        slots: Tensor,
        namespace_ids: Tensor,
        positives: Tensor,
        negatives: Tensor,
    ) -> None:
        if slots.numel() == 0:
            return
        slots = slots.detach().to(device="cpu", dtype=torch.long)
        namespace_ids = namespace_ids.detach().to(device="cpu", dtype=torch.int64)
        positives = positives.detach().to(device="cpu", dtype=torch.int64)
        negatives = negatives.detach().to(device="cpu", dtype=torch.int64)
        ttl = self._namespace_ttl_steps()[namespace_ids]
        self._slot_last_access[slots] = self._current_step
        expires = torch.where(
            ttl >= 0,
            self._current_step + ttl,
            torch.full_like(ttl, -1),
        )
        self._slot_expires_at[slots] = expires
        self._slot_positive[slots] += positives
        self._slot_negative[slots] += negatives
        if self.training:
            self._slot_pinned[slots] = True

    def _bind_new_slots(
        self,
        slots: Tensor,
        namespace_ids: Tensor,
        logical_ids: Tensor,
    ) -> None:
        slots = slots.detach().to(device="cpu", dtype=torch.long)
        namespace_ids = namespace_ids.detach().to(device="cpu", dtype=torch.int64)
        logical_ids = logical_ids.detach().to(device="cpu", dtype=torch.int64)
        ttl = self._namespace_ttl_steps()[namespace_ids]
        self._slot_namespace[slots] = namespace_ids
        self._slot_key[slots] = logical_ids
        self._slot_score[slots] = 0.0
        self._slot_positive[slots] = 0
        self._slot_negative[slots] = 0
        self._slot_last_access[slots] = self._current_step
        self._slot_expires_at[slots] = torch.where(
            ttl >= 0,
            self._current_step + ttl,
            torch.full_like(ttl, -1),
        )
        self._slot_high_priority[slots] = self._namespace_high_priority()[namespace_ids]
        self._slot_pinned[slots] = False
        self._pending_optimizer_reset[slots] = True
        self._slot_generation[slots] += 1
        self._store_mapped_slots(namespace_ids, logical_ids, slots)

    def apply_policy_records(self, records: Tensor) -> None:
        """Admit and score a sorted unique ``(namespace, id)`` stream."""

        merged = merge_gset_policy_records(records)
        if self.row_sharded:
            rank, world_size = _distributed_rank_world(self.process_group)
            if world_size > 1:
                merged = owned_gset_policy_records(
                    merged,
                    rank=rank,
                    world_size=world_size,
                )
        if merged.numel() == 0:
            return
        namespace_ids = merged[:, 0]
        logical_ids = merged[:, 1]
        occurrences = merged[:, 2]
        positives = merged[:, 3]
        negatives = merged[:, 4]
        if bool((namespace_ids < 0).any()) or bool(
            (namespace_ids >= len(self._namespace_policies)).any()
        ):
            raise KeyError("unknown GSET namespace id in policy records")

        n_rows = int(merged.size(0))
        namespace_list = namespace_ids.tolist()
        logical_list = logical_ids.tolist()
        with self._lock:
            existing_slots = self._lookup_mapped_slots(namespace_ids, logical_ids)
            is_new = existing_slots == 0

            resident = ~is_new
            if bool(resident.any()):
                resident_idx = torch.nonzero(resident, as_tuple=False).flatten()
                self._hits += int(occurrences.index_select(0, resident_idx).sum().item())
                self._touch_resident_slots(
                    existing_slots.index_select(0, resident_idx),
                    namespace_ids.index_select(0, resident_idx),
                    positives.index_select(0, resident_idx),
                    negatives.index_select(0, resident_idx),
                )
            if not bool(is_new.any()):
                return

            new_idx = torch.nonzero(is_new, as_tuple=False).flatten()
            self._misses += int(occurrences.index_select(0, new_idx).sum().item())
            admit = torch.zeros(n_rows, dtype=torch.bool, device="cpu")
            admission = self._namespace_admission_probs()
            new_ns = namespace_ids.index_select(0, new_idx)
            if bool((admission.index_select(0, new_ns) >= 1.0).all()):
                admit[new_idx] = True
            else:
                for index in new_idx.tolist():
                    policy = self._namespace_policies[int(namespace_list[index])]
                    if self._admission_draw(
                        policy.admission_probability,
                        int(occurrences[index].item()),
                    ):
                        admit[index] = True
                    else:
                        self._rejected += int(occurrences[index].item())
            if not bool(admit.any()):
                return

            admit_idx = torch.nonzero(admit, as_tuple=False).flatten()
            n_admit = int(admit_idx.numel())
            reserved_slots: set[int] = set(int(slot) for slot in existing_slots[resident].tolist())
            if self._available_free_slots() >= n_admit:
                slots = self._take_free_slots(n_admit)
                self._bind_new_slots(
                    slots,
                    namespace_ids.index_select(0, admit_idx),
                    logical_ids.index_select(0, admit_idx),
                )
                self._touch_resident_slots(
                    slots,
                    namespace_ids.index_select(0, admit_idx),
                    positives.index_select(0, admit_idx),
                    negatives.index_select(0, admit_idx),
                )
                self._admissions += n_admit
                self._initialize_parameter_rows(slots)
                return

            initialized: list[int] = []
            for index in admit_idx.tolist():
                namespace_id = int(namespace_list[index])
                logical_id = int(logical_list[index])
                policy = self._namespace_policies[namespace_id]
                slot = self._allocate_slot(
                    namespace_id,
                    logical_id,
                    policy,
                    reserved_slots,
                )
                if slot is None:
                    self._rejected += int(occurrences[index].item())
                    continue
                initialized.append(slot)
                reserved_slots.add(slot)
                self._admissions += 1
                self._touch_resident_slots(
                    torch.tensor([slot], dtype=torch.long, device="cpu"),
                    torch.tensor([namespace_id], dtype=torch.int64, device="cpu"),
                    positives[index].view(1),
                    negatives[index].view(1),
                )
            self._initialize_parameter_rows(initialized)

    def apply_synchronized_policy_records(self, local_records: Tensor) -> None:
        """Admit this microbatch's keys on the ranks that own them, then freeze.

        Replicated tables all-gather the union so every rank's mapper stays
        identical. Row-sharded tables send each ``(namespace, id)`` record to
        ``id % world_size`` and admit only locally owned keys.
        """

        if self.row_sharded:
            owned = exchange_owned_gset_policy_records(
                local_records,
                process_group=self.process_group,
                device=self.weight.device,
            )
            self.apply_policy_records(owned)
        else:
            merged = gather_and_merge_gset_policy_records(
                local_records,
                device=self.weight.device,
            )
            self.apply_policy_records(merged)
        with self._lock:
            self._policy_prepared = True

    def _logical_id_count_tensors(
        self,
        logical_ids: Tensor,
        *,
        padding_idx: int | None = None,
        positive_mask: Tensor | None = None,
        row_positive_counts: Tensor | None = None,
        row_negative_counts: Tensor | None = None,
        row_lengths: Tensor | None = None,
        valid_lengths: Tensor | None = None,
        allow_negative_ids: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if logical_ids.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("GSET logical IDs must use an integer tensor dtype")
        if positive_mask is not None:
            if positive_mask.shape != logical_ids.shape:
                raise ValueError(
                    "positive_mask must have the same shape as logical_ids"
                )
            positive_mask = positive_mask.to(dtype=torch.bool, device="cpu")
        if positive_mask is not None and (
            row_positive_counts is not None or row_negative_counts is not None
        ):
            raise ValueError(
                "positive_mask cannot be combined with row-level outcome counts"
            )
        if (row_positive_counts is None) != (row_negative_counts is None):
            raise ValueError(
                "row_positive_counts and row_negative_counts must be provided together"
            )

        flat_ids = logical_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        flat_valid: Tensor | None = None
        if valid_lengths is not None:
            if logical_ids.ndim != 2 or valid_lengths.ndim != 1:
                raise ValueError(
                    "GSET valid_lengths requires [rows, length] IDs and rank-one lengths"
                )
            sequence_lengths = valid_lengths.detach().to(
                device="cpu", dtype=torch.int64
            )
            if sequence_lengths.numel() != logical_ids.size(0) or bool(
                (sequence_lengths < 0).any()
            ):
                raise ValueError(
                    "GSET valid_lengths must be non-negative and match ID rows"
                )
            sequence_lengths = sequence_lengths.clamp(max=logical_ids.size(1))
            flat_valid = (
                torch.arange(logical_ids.size(1), dtype=torch.int64).view(1, -1)
                < sequence_lengths.view(-1, 1)
            ).reshape(-1)
        flat_row_indices: Tensor | None = None
        row_positive: Tensor | None = None
        row_negative: Tensor | None = None
        if row_positive_counts is not None and row_negative_counts is not None:
            if row_positive_counts.ndim != 1 or row_negative_counts.ndim != 1:
                raise ValueError("GSET row outcome counts must be rank one")
            row_positive = row_positive_counts.detach().to(
                device="cpu", dtype=torch.int64
            )
            row_negative = row_negative_counts.detach().to(
                device="cpu", dtype=torch.int64
            )
            if bool((row_positive < 0).any()) or bool((row_negative < 0).any()):
                raise ValueError("GSET row outcome counts must be non-negative")
            row_count = int(row_positive.numel())
            if row_negative.numel() != row_count:
                raise ValueError(
                    "GSET row outcome count tensors must have equal length"
                )
            if row_lengths is not None:
                if logical_ids.ndim != 1 or row_lengths.ndim != 1:
                    raise ValueError(
                        "GSET row_lengths is only valid for flat rank-one ID tensors"
                    )
                lengths = row_lengths.detach().to(device="cpu", dtype=torch.int64)
                if lengths.numel() != row_count or bool((lengths < 0).any()):
                    raise ValueError(
                        "GSET row_lengths must be non-negative and match outcome rows"
                    )
                if int(lengths.sum().item()) != flat_ids.numel():
                    raise ValueError("GSET row_lengths does not partition logical_ids")
                flat_row_indices = torch.repeat_interleave(
                    torch.arange(row_count, dtype=torch.int64),
                    lengths,
                )
            else:
                logical_row_count = 1 if logical_ids.ndim == 0 else logical_ids.size(0)
                if logical_row_count != row_count:
                    raise ValueError(
                        "GSET outcome rows must match logical_ids' leading dimension"
                    )
                values_per_row = (
                    1 if logical_ids.ndim <= 1 else math.prod(logical_ids.shape[1:])
                )
                flat_row_indices = torch.arange(
                    row_count, dtype=torch.int64
                ).repeat_interleave(values_per_row)
            if flat_row_indices.numel() != flat_ids.numel():
                raise RuntimeError("GSET failed to align IDs with sample rows")
        elif row_lengths is not None:
            raise ValueError("row_lengths requires row-level outcome counts")

        keep = torch.ones(flat_ids.numel(), dtype=torch.bool, device="cpu")
        if flat_valid is not None:
            keep &= flat_valid
        if not allow_negative_ids:
            keep &= flat_ids >= 0
        if padding_idx is not None:
            keep &= flat_ids != int(padding_idx)
        inverse_full = flat_ids.new_full((flat_ids.numel(),), -1)
        empty = flat_ids.new_empty((0,))
        if not bool(keep.any()):
            return empty, empty, empty, empty, inverse_full

        valid_ids = flat_ids[keep]
        unique_sorted, inv_sorted, occ_sorted = torch.unique(
            valid_ids,
            return_inverse=True,
            return_counts=True,
        )
        first_pos = torch.full(
            (unique_sorted.numel(),),
            valid_ids.numel(),
            dtype=torch.long,
        )
        first_pos.scatter_reduce_(
            0,
            inv_sorted,
            torch.arange(valid_ids.numel(), dtype=torch.long),
            reduce="amin",
            include_self=True,
        )
        order = first_pos.argsort()
        unique_ids = unique_sorted[order]
        occurrences = occ_sorted[order]
        remap = torch.empty_like(order)
        remap[order] = torch.arange(order.numel(), dtype=order.dtype)
        inverse_valid = remap[inv_sorted]
        inverse_full[keep] = inverse_valid

        n_unique = int(unique_ids.numel())
        flat_positive = None if positive_mask is None else positive_mask.reshape(-1)
        if flat_row_indices is None:
            if flat_positive is None:
                positives = torch.zeros(n_unique, dtype=torch.int64, device="cpu")
                negatives = occurrences
            else:
                positives = torch.zeros(n_unique, dtype=torch.int64, device="cpu")
                positives.scatter_add_(
                    0,
                    inverse_valid,
                    flat_positive[keep].to(dtype=torch.int64),
                )
                negatives = occurrences - positives
        else:
            if row_positive is None or row_negative is None:
                raise RuntimeError("GSET row score metadata is missing")
            row_idx = flat_row_indices[keep]
            n_rows = int(row_positive.numel())
            id_min = int(valid_ids.min().item()) if valid_ids.numel() else 0
            id_max = int(valid_ids.max().item()) if valid_ids.numel() else 0
            packed_ok = (
                n_rows > 0
                and id_min >= 0
                and id_max <= (2**63 - 1) // n_rows
            )
            host = torch.device("cpu")
            positives = torch.zeros(n_unique, dtype=torch.int64, device=host)
            negatives = torch.zeros(n_unique, dtype=torch.int64, device=host)
            if packed_ok:
                # 1-D unique of (id * n_rows + row) is much cheaper than
                # unique([N, 2]) and still counts each ID at most once per sample.
                unique_packed = torch.unique(valid_ids * n_rows + row_idx)
                pair_ids = torch.div(unique_packed, n_rows, rounding_mode="floor")
                pair_rows = unique_packed - pair_ids * n_rows
                pair_index = remap[
                    torch.searchsorted(unique_sorted, pair_ids.contiguous())
                ]
            else:
                pairs = torch.stack((valid_ids, row_idx), dim=1)
                unique_pairs = torch.unique(pairs, dim=0)
                pair_index = remap[
                    torch.searchsorted(unique_sorted, unique_pairs[:, 0].contiguous())
                ]
                pair_rows = unique_pairs[:, 1]
            positives.scatter_add_(0, pair_index, row_positive[pair_rows])
            negatives.scatter_add_(0, pair_index, row_negative[pair_rows])

        return unique_ids, occurrences, positives, negatives, inverse_full

    def _logical_id_counts(
        self,
        logical_ids: Tensor,
        *,
        padding_idx: int | None = None,
        positive_mask: Tensor | None = None,
        row_positive_counts: Tensor | None = None,
        row_negative_counts: Tensor | None = None,
        row_lengths: Tensor | None = None,
        valid_lengths: Tensor | None = None,
        allow_negative_ids: bool = False,
    ) -> tuple[OrderedDict[int, list[int]], Tensor]:
        unique_ids, occurrences, positives, negatives, inverse_full = (
            self._logical_id_count_tensors(
                logical_ids,
                padding_idx=padding_idx,
                positive_mask=positive_mask,
                row_positive_counts=row_positive_counts,
                row_negative_counts=row_negative_counts,
                row_lengths=row_lengths,
                valid_lengths=valid_lengths,
                allow_negative_ids=allow_negative_ids,
            )
        )
        counts: OrderedDict[int, list[int]] = OrderedDict()
        for logical_id, occ, pos, neg in zip(
            unique_ids.tolist(),
            occurrences.tolist(),
            positives.tolist(),
            negatives.tolist(),
        ):
            counts[int(logical_id)] = [int(occ), int(pos), int(neg)]
        return counts, inverse_full

    def lookup(
        self,
        namespace: str | int,
        logical_ids: Tensor,
        *,
        padding_idx: int | None = None,
        positive_mask: Tensor | None = None,
        row_positive_counts: Tensor | None = None,
        row_negative_counts: Tensor | None = None,
        row_lengths: Tensor | None = None,
        valid_lengths: Tensor | None = None,
        allow_negative_ids: bool = False,
        admit: bool | None = None,
        track_policy: bool | None = None,
        return_slots: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Map logical IDs to shared slots and perform an embedding lookup.

        ``positive_mask`` is a low-level occurrence-aligned label input.
        Model training instead supplies ``row_positive_counts`` and
        ``row_negative_counts``: one count per sample/request row. Combined
        with optional CSR ``row_lengths``, this counts each ID at most once per
        sample exactly as Kraken specifies, even if a sequence repeats it.
        ``valid_lengths`` masks the padded tail of dense ``[rows, length]``
        sequence IDs before admission and score accounting.
        """

        if self.row_sharded:
            _rank, world_size = _distributed_rank_world(self.process_group)
            if world_size > 1:
                if return_slots:
                    raise RuntimeError(
                        "return_slots is not supported for row-sharded GSET lookups"
                    )
                return _ShardedGSETLookup.apply(
                    self.weight,
                    logical_ids,
                    self,
                    int(self._resolve_namespace(namespace)),
                    padding_idx,
                    bool(allow_negative_ids),
                )
        slot_tensor = self._map_ids_to_slots(
            namespace,
            logical_ids,
            padding_idx=padding_idx,
            positive_mask=positive_mask,
            row_positive_counts=row_positive_counts,
            row_negative_counts=row_negative_counts,
            row_lengths=row_lengths,
            valid_lengths=valid_lengths,
            allow_negative_ids=allow_negative_ids,
            admit=admit,
            track_policy=track_policy,
        )
        output = F.embedding(
            slot_tensor,
            self.weight,
            padding_idx=0,
            sparse=self.sparse,
        )
        if return_slots:
            return output, slot_tensor
        return output

    def _scatter_unique_slots(
        self,
        logical_ids: Tensor,
        inverse: Tensor,
        unique_slots: Tensor,
    ) -> Tensor:
        slots = torch.zeros(int(inverse.numel()), dtype=torch.long, device="cpu")
        valid = inverse >= 0
        if bool(valid.any()) and unique_slots.numel():
            slots[valid] = unique_slots[inverse[valid]]
        return slots.to(device=logical_ids.device).view_as(logical_ids)

    def _map_resident_slots(
        self,
        namespace: str | int,
        logical_ids: Tensor,
        *,
        padding_idx: int | None = None,
        valid_lengths: Tensor | None = None,
        allow_negative_ids: bool = False,
    ) -> Tensor:
        flat_ids = logical_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        keep = torch.ones(flat_ids.numel(), dtype=torch.bool, device="cpu")
        if valid_lengths is not None:
            if logical_ids.ndim != 2 or valid_lengths.ndim != 1:
                raise ValueError(
                    "GSET valid_lengths requires [rows, length] IDs and rank-one lengths"
                )
            sequence_lengths = valid_lengths.detach().to(
                device="cpu", dtype=torch.int64
            ).clamp(max=logical_ids.size(1))
            keep &= (
                torch.arange(
                    logical_ids.size(1), dtype=torch.int64, device="cpu"
                ).view(1, -1)
                < sequence_lengths.view(-1, 1)
            ).reshape(-1)
        if not allow_negative_ids:
            keep &= flat_ids >= 0
        if padding_idx is not None:
            keep &= flat_ids != int(padding_idx)
        namespace_id = self._resolve_namespace(namespace)
        slots = torch.zeros(flat_ids.numel(), dtype=torch.long, device="cpu")
        with self._lock:
            self._lookups += int(flat_ids.numel())
            if bool(keep.any()):
                valid_ids = flat_ids[keep]
                slots[keep] = self._lookup_mapped_slots(
                    torch.full_like(valid_ids, int(namespace_id)),
                    valid_ids,
                )
                if self.training and torch.is_grad_enabled():
                    live = torch.unique(slots[keep])
                    live = live[live > 0]
                    if live.numel():
                        self._slot_pinned[live] = True
        return slots.to(device=logical_ids.device).view_as(logical_ids)

    def _map_ids_to_slots(
        self,
        namespace: str | int,
        logical_ids: Tensor,
        *,
        padding_idx: int | None = None,
        positive_mask: Tensor | None = None,
        row_positive_counts: Tensor | None = None,
        row_negative_counts: Tensor | None = None,
        row_lengths: Tensor | None = None,
        valid_lengths: Tensor | None = None,
        allow_negative_ids: bool = False,
        admit: bool | None = None,
        track_policy: bool | None = None,
    ) -> Tensor:
        prepared = self._policy_prepared
        if prepared and admit is None and track_policy is None:
            return self._map_resident_slots(
                namespace,
                logical_ids,
                padding_idx=padding_idx,
                valid_lengths=valid_lengths,
                allow_negative_ids=allow_negative_ids,
            )
        unique_ids, occurrences, positives, negatives, inverse = (
            self._logical_id_count_tensors(
                logical_ids,
                padding_idx=padding_idx,
                positive_mask=positive_mask,
                row_positive_counts=row_positive_counts,
                row_negative_counts=row_negative_counts,
                row_lengths=row_lengths,
                valid_lengths=valid_lengths,
                allow_negative_ids=allow_negative_ids,
            )
        )
        namespace_id = self._resolve_namespace(namespace)
        policy = self._namespace_policies[namespace_id]
        should_admit = (
            False
            if admit is None and prepared
            else (self.training if admit is None else bool(admit))
        )
        should_track_policy = (
            False
            if track_policy is None and prepared
            else (self.training if track_policy is None else bool(track_policy))
        )
        n_flat = int(inverse.numel())
        unique_slots = torch.zeros(unique_ids.numel(), dtype=torch.long, device="cpu")

        initialized: list[int] = []
        reserved_slots: set[int] = set()
        with self._lock:
            self._lookups += n_flat
            already_mapped = (
                self._lookup_mapped_slots(
                    torch.full_like(unique_ids, int(namespace_id)),
                    unique_ids,
                )
                if unique_ids.numel()
                else unique_slots
            )
            for unique_index, logical_id in enumerate(unique_ids.tolist()):
                occurrence_count = int(occurrences[unique_index].item())
                existing = int(already_mapped[unique_index].item())
                slot = existing if existing > 0 else None
                if slot is not None:
                    if not prepared:
                        self._hits += occurrence_count
                else:
                    if not should_admit or not self._admission_draw(
                        policy.admission_probability,
                        occurrence_count,
                    ):
                        if not prepared:
                            self._misses += occurrence_count
                            self._rejected += occurrence_count
                        continue
                    self._misses += occurrence_count
                    slot = self._allocate_slot(
                        namespace_id,
                        int(logical_id),
                        policy,
                        reserved_slots,
                    )
                    if slot is None:
                        self._rejected += occurrence_count
                        continue
                    initialized.append(slot)
                    self._admissions += 1

                if should_track_policy:
                    self._slot_last_access[slot] = self._current_step
                    if policy.ttl_steps is not None:
                        self._slot_expires_at[slot] = self._current_step + int(
                            policy.ttl_steps
                        )
                    self._slot_positive[slot] += int(positives[unique_index].item())
                    self._slot_negative[slot] += int(negatives[unique_index].item())
                if self.training and torch.is_grad_enabled():
                    self._slot_pinned[slot] = True
                # A single lookup must never recycle a row already referenced
                # by an earlier output position, including no-grad admission.
                reserved_slots.add(slot)
                unique_slots[unique_index] = int(slot)

            self._initialize_parameter_rows(initialized)

        return self._scatter_unique_slots(logical_ids, inverse, unique_slots)

    def forward(self, logical_ids: Tensor) -> Tensor:
        """Default namespace-free calls are intentionally rejected."""

        raise RuntimeError(
            "GlobalSharedEmbeddingTable requires a feature namespace; "
            "use table.lookup(...) or GSETEmbeddingView"
        )

    def contains(self, namespace: str | int, logical_id: int) -> bool:
        return self.slot_for(namespace, logical_id) is not None

    def slot_for(self, namespace: str | int, logical_id: int) -> int | None:
        namespace_id = self._resolve_namespace(namespace)
        with self._lock:
            mapped = self._lookup_mapped_slots(
                torch.tensor([namespace_id], dtype=torch.int64, device="cpu"),
                torch.tensor([int(logical_id)], dtype=torch.int64, device="cpu"),
            )
            slot = int(mapped[0].item())
            return slot if slot > 0 else None

    def feature_score(self, namespace: str | int, logical_id: int) -> float | None:
        """Return the current eviction score for a resident logical key."""

        slot = self.slot_for(namespace, logical_id)
        if slot is None:
            return None
        with self._lock:
            return float(self._slot_score[slot])

    def key_for_slot(self, slot: int) -> tuple[str, int] | None:
        if not 1 <= int(slot) <= self.capacity:
            raise ValueError("GSET slot is outside the physical capacity")
        namespace_id = int(self._slot_namespace[int(slot)])
        if namespace_id < 0:
            return None
        return self._namespace_names[namespace_id], int(self._slot_key[int(slot)])

    def pending_optimizer_reset_rows(self) -> Tensor:
        with self._lock:
            return torch.nonzero(
                self._pending_optimizer_reset,
                as_tuple=False,
            ).flatten()

    def consume_pending_optimizer_reset_rows(self) -> Tensor:
        with self._lock:
            rows = self.pending_optimizer_reset_rows()
            if rows.numel():
                self._pending_optimizer_reset[rows] = False
            return rows

    def advance_step(self, steps: int = 1) -> None:
        if type(steps) is not int or steps <= 0:
            raise ValueError("GSET steps must be a positive integer")
        with self._lock:
            self._current_step += steps
            self._refresh_scores()

    def optimizer_step_completed(self) -> None:
        """Release gradient-accumulation pins and advance the policy clock."""

        with self._lock:
            self._slot_pinned.zero_()
            self.advance_step(1)

    def stats(self) -> GSETStats:
        with self._lock:
            active = int((self._slot_namespace[1:] >= 0).sum().item())
            return GSETStats(
                capacity=self.capacity,
                active_entries=active,
                free_entries=self.capacity - active,
                lookups=self._lookups,
                hits=self._hits,
                misses=self._misses,
                admissions=self._admissions,
                rejected=self._rejected,
                evictions=self._evictions,
                duration_evictions=self._duration_evictions,
                score_evictions=self._score_evictions,
                current_step=self._current_step,
            )

    def _rebuild_indices(self) -> None:
        occupied = self._slot_namespace >= 0
        occupied[0] = False
        if bool((self._slot_namespace[occupied] >= len(self._namespace_names)).any()):
            raise RuntimeError("GSET checkpoint contains an invalid namespace ID")
        idx = torch.nonzero(occupied, as_tuple=False).flatten()
        self._key_to_slot = {}
        self._ensure_dense_mapper()
        if self._dense_slots is not None:
            self._dense_slots.zero_()
        if idx.numel() == 0:
            self._next_virgin_slot = 1
            self._recycled_slots = []
            return
        namespace_ids = self._slot_namespace.index_select(0, idx)
        logical_ids = self._slot_key.index_select(0, idx)
        key_pairs = torch.stack((namespace_ids, logical_ids), dim=1)
        _unique_keys, key_counts = torch.unique(key_pairs, dim=0, return_counts=True)
        if bool((key_counts > 1).any()):
            raise RuntimeError("GSET checkpoint contains duplicate logical keys")
        self._store_mapped_slots(namespace_ids, logical_ids, idx)
        last_occupied = int(idx.max().item())
        self._next_virgin_slot = last_occupied + 1
        hole_mask = ~occupied
        hole_mask[0] = False
        if last_occupied + 1 <= self.capacity:
            hole_mask[last_occupied + 1 :] = False
        holes = torch.nonzero(hole_mask, as_tuple=False).flatten()
        self._recycled_slots = [int(slot) for slot in holes.tolist()]
        heapq.heapify(self._recycled_slots)

    def get_extra_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": _GSET_STATE_VERSION,
                "capacity": self.capacity,
                "embedding_dim": self.embedding_dim,
                "eviction_policy": self.eviction_policy,
                "namespace_names": tuple(self._namespace_names),
                "namespace_policies": tuple(
                    asdict(policy) for policy in self._namespace_policies
                ),
                "slot_namespace": self._slot_namespace.clone(),
                "slot_key": self._slot_key.clone(),
                "slot_score": self._slot_score.clone(),
                "slot_positive": self._slot_positive.clone(),
                "slot_negative": self._slot_negative.clone(),
                "slot_last_access": self._slot_last_access.clone(),
                "slot_expires_at": self._slot_expires_at.clone(),
                "slot_high_priority": self._slot_high_priority.clone(),
                "slot_pinned": self._slot_pinned.clone(),
                "pending_optimizer_reset": self._pending_optimizer_reset.clone(),
                "slot_generation": self._slot_generation.clone(),
                "current_step": self._current_step,
                "last_score_update_step": self._last_score_update_step,
                "rng_state": self._rng.get_state().clone(),
                "row_sharded": self.row_sharded,
                "counters": {
                    "lookups": self._lookups,
                    "hits": self._hits,
                    "misses": self._misses,
                    "admissions": self._admissions,
                    "rejected": self._rejected,
                    "evictions": self._evictions,
                    "duration_evictions": self._duration_evictions,
                    "score_evictions": self._score_evictions,
                },
            }

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        if int(state.get("version", -1)) != _GSET_STATE_VERSION:
            raise RuntimeError("unsupported GSET checkpoint metadata version")
        if int(state.get("capacity", -1)) != self.capacity:
            raise RuntimeError("GSET checkpoint capacity does not match the model")
        if int(state.get("embedding_dim", -1)) != self.embedding_dim:
            raise RuntimeError("GSET checkpoint embedding_dim does not match the model")
        if str(state.get("eviction_policy", "score")) != self.eviction_policy:
            raise RuntimeError("GSET checkpoint eviction_policy does not match the model")
        if "row_sharded" in state and bool(state["row_sharded"]) != self.row_sharded:
            raise RuntimeError("GSET checkpoint row_sharded does not match the model")

        names = [str(name) for name in state.get("namespace_names", ())]
        policies = [
            GSETNamespacePolicy(**dict(payload))
            for payload in state.get("namespace_policies", ())
        ]
        if self._namespace_names and self._namespace_names != names:
            raise RuntimeError(
                "GSET checkpoint namespace order does not match the model"
            )
        if self._namespace_policies and self._namespace_policies != policies:
            raise RuntimeError(
                "GSET checkpoint namespace policies do not match the model"
            )
        for name, policy in zip(names, policies):
            policy.validate(name)

        expected = self.capacity + 1

        def tensor(name: str, dtype: torch.dtype) -> Tensor:
            value = state.get(name)
            if not isinstance(value, Tensor) or value.numel() != expected:
                raise RuntimeError(f"GSET checkpoint field {name!r} has invalid shape")
            return (
                value.detach().to(device="cpu", dtype=dtype).reshape(expected).clone()
            )

        with self._lock:
            self._namespace_names = names
            self._namespace_to_id = {
                name: index for index, name in enumerate(self._namespace_names)
            }
            self._namespace_policies = policies
            self._slot_namespace = tensor("slot_namespace", torch.int64)
            self._slot_key = tensor("slot_key", torch.int64)
            self._slot_score = tensor("slot_score", torch.float64)
            self._slot_positive = tensor("slot_positive", torch.int64)
            self._slot_negative = tensor("slot_negative", torch.int64)
            self._slot_last_access = tensor("slot_last_access", torch.int64)
            self._slot_expires_at = tensor("slot_expires_at", torch.int64)
            self._slot_high_priority = tensor("slot_high_priority", torch.bool)
            # A restored checkpoint starts between optimizer steps.  Never
            # retain transient in-flight pins from a process that crashed.
            self._slot_pinned = torch.zeros(expected, dtype=torch.bool, device="cpu")
            self._pending_optimizer_reset = tensor(
                "pending_optimizer_reset", torch.bool
            )
            self._slot_generation = tensor("slot_generation", torch.int64)
            self._current_step = int(state.get("current_step", 0))
            self._last_score_update_step = int(
                state.get("last_score_update_step", self._current_step)
            )
            rng_state = state.get("rng_state")
            if not isinstance(rng_state, Tensor):
                raise RuntimeError("GSET checkpoint is missing RNG state")
            self._rng.set_state(rng_state.detach().cpu())
            counters = state.get("counters", {})
            self._lookups = int(counters.get("lookups", 0))
            self._hits = int(counters.get("hits", 0))
            self._misses = int(counters.get("misses", 0))
            self._admissions = int(counters.get("admissions", 0))
            self._rejected = int(counters.get("rejected", 0))
            self._evictions = int(counters.get("evictions", 0))
            self._duration_evictions = int(counters.get("duration_evictions", 0))
            self._score_evictions = int(counters.get("score_evictions", 0))
            self._rebuild_indices()
            with torch.no_grad():
                self.weight[0].zero_()


class _ShardedGSETLookup(torch.autograd.Function):
    """Route GSET logical IDs with ``id % world_size`` and return owner rows."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        local_weight: Tensor,
        logical_ids: Tensor,
        table: GlobalSharedEmbeddingTable,
        namespace_id: int,
        padding_idx: int | None,
        allow_negative_ids: bool,
    ) -> Tensor:
        ids = logical_ids if logical_ids.dtype == torch.long else logical_ids.long()
        rank, world_size = _distributed_rank_world(table.process_group)
        flat = ids.reshape(-1)
        active_mask = _gset_active_id_mask(
            flat,
            padding_idx=padding_idx,
            allow_negative_ids=bool(allow_negative_ids),
        )
        active_positions = torch.nonzero(active_mask, as_tuple=False).flatten()
        active_ids = flat.index_select(0, active_positions)
        requester_ids, requester_inverse = torch.unique(
            active_ids,
            sorted=True,
            return_inverse=True,
        )
        owners = (
            gset_row_owner(requester_ids, world_size)
            if requester_ids.numel()
            else requester_ids.new_empty((0,), dtype=torch.long)
        )
        send_order = torch.argsort(owners, stable=True)
        sorted_ids = requester_ids.index_select(0, send_order)
        send_splits_tensor = torch.bincount(owners, minlength=world_size)
        send_splits, recv_splits = _exchange_and_host_splits(
            send_splits_tensor,
            table.process_group,
        )
        received_ids = _all_to_all_variable(
            sorted_ids,
            send_splits,
            recv_splits,
            table.process_group,
        )
        if received_ids.numel():
            expected_owner = gset_row_owner(received_ids, world_size)
            if bool((expected_owner != rank).any()):
                raise RuntimeError("received GSET IDs owned by another rank")
            received_slots = table._map_ids_to_slots(
                int(namespace_id),
                received_ids,
                padding_idx=padding_idx,
                allow_negative_ids=bool(allow_negative_ids),
            )
        else:
            received_slots = received_ids.new_empty((0,), dtype=torch.long)
        owner_unique_slots, owner_inverse = torch.unique(
            received_slots,
            sorted=True,
            return_inverse=True,
        )
        if owner_unique_slots.numel():
            owner_unique_values = local_weight.index_select(0, owner_unique_slots)
        else:
            owner_unique_values = local_weight.new_empty((0, local_weight.size(1)))
        received_values = (
            owner_unique_values.index_select(0, owner_inverse)
            if owner_inverse.numel()
            else owner_unique_values
        )
        returned_values = _all_to_all_variable(
            received_values,
            recv_splits,
            send_splits,
            table.process_group,
        )
        requester_values = local_weight.new_empty(
            (requester_ids.numel(), local_weight.size(1))
        )
        if send_order.numel():
            requester_values.index_copy_(0, send_order, returned_values)
        active_values = requester_values.index_select(0, requester_inverse)
        output = local_weight.new_zeros((flat.numel(), local_weight.size(1)))
        if active_positions.numel():
            output.index_copy_(0, active_positions, active_values)

        ctx.process_group = table.process_group
        ctx.world_size = world_size
        ctx.local_weight_shape = tuple(local_weight.shape)
        ctx.local_weight_dtype = local_weight.dtype
        ctx.send_splits = send_splits
        ctx.recv_splits = recv_splits
        ctx.save_for_backward(
            active_positions,
            requester_inverse,
            send_order,
            owner_unique_slots,
            owner_inverse,
        )
        return output.view(*logical_ids.shape, local_weight.size(1))

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Any, ...]:  # type: ignore[override]
        (
            active_positions,
            requester_inverse,
            send_order,
            owner_unique_slots,
            owner_inverse,
        ) = ctx.saved_tensors
        embedding_dim = int(ctx.local_weight_shape[1])
        flat_grad = grad_output.reshape(-1, embedding_dim)
        active_grad = flat_grad.index_select(0, active_positions)
        requester_count = int(send_order.numel())
        requester_grad = flat_grad.new_zeros((requester_count, embedding_dim))
        if requester_inverse.numel():
            requester_grad.index_add_(0, requester_inverse, active_grad)
        sorted_grad = requester_grad.index_select(0, send_order)
        received_grad = _all_to_all_variable(
            sorted_grad,
            ctx.send_splits,
            ctx.recv_splits,
            ctx.process_group,
        )
        owner_grad = received_grad.new_zeros(
            (owner_unique_slots.numel(), embedding_dim)
        )
        if owner_inverse.numel():
            owner_grad.index_add_(0, owner_inverse, received_grad)
        if owner_unique_slots.numel():
            owner_grad = owner_grad / float(ctx.world_size)
            owner_grad = owner_grad.to(dtype=ctx.local_weight_dtype)
            padding = owner_unique_slots == 0
            if bool(padding.any()):
                owner_grad = owner_grad.clone()
                owner_grad[padding] = 0
        slot_indices = (
            owner_unique_slots.unsqueeze(0)
            if owner_unique_slots.numel()
            else owner_unique_slots.new_empty((1, 0))
        )
        local_weight_grad = torch.sparse_coo_tensor(
            slot_indices,
            owner_grad,
            size=ctx.local_weight_shape,
            dtype=ctx.local_weight_dtype,
            device=grad_output.device,
            is_coalesced=True,
        )
        return (
            local_weight_grad,
            None,
            None,
            None,
            None,
            None,
        )


class GSETEmbeddingView(nn.Module):
    """One feature namespace view over a shared physical GSET parameter."""

    def __init__(
        self,
        table: GlobalSharedEmbeddingTable,
        namespace: str,
        *,
        padding_idx: int | None = None,
        policy: GSETNamespacePolicy | None = None,
        dracarys_feature_xor: bool = False,
    ) -> None:
        super().__init__()
        self.namespace = namespace
        self.dracarys_feature_xor = bool(dracarys_feature_xor)
        registered_namespace = (
            _DRACARYS_GLOBAL_NAMESPACE
            if self.dracarys_feature_xor
            else namespace
        )
        self.namespace_index = table.register_namespace(
            registered_namespace,
            None if self.dracarys_feature_xor else policy,
        )
        self.padding_idx = None if self.dracarys_feature_xor else padding_idx
        self.num_embeddings = table.num_embeddings
        self.embedding_dim = table.embedding_dim
        # A weak reference avoids registering the same large table under every
        # feature view (which would duplicate state_dict paths and checkpoints).
        object.__setattr__(self, "_table_ref", weakref.ref(table))
        self._mdl_id_embedding = True

    @property
    def table(self) -> GlobalSharedEmbeddingTable:
        table = self._table_ref()
        if table is None:
            raise RuntimeError("the GSET table owning this feature view was released")
        return table

    @property
    def weight(self) -> nn.Parameter:
        return self.table.weight

    def forward(self, logical_ids: Tensor) -> Tensor:
        return self.lookup(logical_ids)

    def lookup(
        self,
        logical_ids: Tensor,
        *,
        row_positive_counts: Tensor | None = None,
        row_negative_counts: Tensor | None = None,
        row_lengths: Tensor | None = None,
        valid_lengths: Tensor | None = None,
    ) -> Tensor:
        lookup_ids = (
            dracarys_feature_xor_ids(logical_ids, self.namespace)
            if self.dracarys_feature_xor
            else logical_ids
        )
        output = self.table.lookup(
            self.namespace_index,
            lookup_ids,
            padding_idx=self.padding_idx,
            row_positive_counts=row_positive_counts,
            row_negative_counts=row_negative_counts,
            row_lengths=row_lengths,
            valid_lengths=valid_lengths,
            allow_negative_ids=self.dracarys_feature_xor,
        )
        if not isinstance(output, Tensor):
            raise RuntimeError("GSET lookup returned an invalid result")
        return output

    def count_policy_records(
        self,
        logical_ids: Tensor,
        *,
        row_positive_counts: Tensor | None = None,
        row_negative_counts: Tensor | None = None,
        row_lengths: Tensor | None = None,
        valid_lengths: Tensor | None = None,
    ) -> Tensor:
        lookup_ids = (
            dracarys_feature_xor_ids(logical_ids, self.namespace)
            if self.dracarys_feature_xor
            else logical_ids
        )
        return self.table.count_policy_records(
            self.namespace_index,
            lookup_ids,
            padding_idx=self.padding_idx,
            row_positive_counts=row_positive_counts,
            row_negative_counts=row_negative_counts,
            row_lengths=row_lengths,
            valid_lengths=valid_lengths,
            allow_negative_ids=self.dracarys_feature_xor,
        )


def gset_owner_for_parameter(
    parameter: nn.Parameter,
) -> GlobalSharedEmbeddingTable | None:
    owner_ref = getattr(parameter, "_mdl_gset_owner", None)
    if isinstance(owner_ref, weakref.ReferenceType):
        owner = owner_ref()
        if owner is not None:
            return owner
    fallback = _PARAMETER_OWNERS.get(id(parameter))
    return None if fallback is None else fallback()


def iter_gset_tables(module: nn.Module) -> Iterator[GlobalSharedEmbeddingTable]:
    seen: set[int] = set()
    for child in module.modules():
        if not isinstance(child, GlobalSharedEmbeddingTable):
            continue
        if id(child) in seen:
            continue
        seen.add(id(child))
        yield child


def set_gset_batch_outcomes(
    module: nn.Module,
    labels: Tensor,
    label_mask: Tensor | None = None,
    *,
    task_index: int = 0,
    active: bool = True,
) -> None:
    """Set label-aware score context on every GSET owned by ``module``."""

    for table in iter_gset_tables(module):
        table.set_batch_outcomes(
            labels,
            label_mask,
            task_index=task_index,
            active=active,
        )


def clear_gset_batch_outcomes(module: nn.Module) -> None:
    """Clear label-aware score context after forward/backward completes."""

    for table in iter_gset_tables(module):
        table.clear_batch_outcomes()


@contextmanager
def gset_batch_outcomes(
    module: nn.Module,
    labels: Tensor,
    label_mask: Tensor | None = None,
    *,
    task_index: int = 0,
    active: bool = True,
) -> Iterator[None]:
    """Scope label-aware GSET accounting across forward and recomputation."""

    set_gset_batch_outcomes(
        module,
        labels,
        label_mask,
        task_index=task_index,
        active=active,
    )
    try:
        yield
    finally:
        clear_gset_batch_outcomes(module)
