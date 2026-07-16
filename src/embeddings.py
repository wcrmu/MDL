"""Self-contained distributed embedding tables built on PyTorch collectives.

Row-wise tables use cyclic ownership for hashed IDs; small table-wise shards use
a deterministic LPT plan. Forward requests and backward gradients are routed with
variable-size ``all_to_all_single`` collectives.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from time import perf_counter
from typing import Any, Iterable, Literal

import torch
import torch.distributed as torch_dist
from torch import Tensor, nn


EmbeddingShardStrategy = Literal["row_wise", "table_wise"]


def _invalid_any(invalid: Tensor, message: str) -> bool:
    """Validate CUDA tensors without a per-lookup device-to-host barrier."""

    invalid_result = invalid.any()
    if invalid.device.type == "cuda" and hasattr(torch, "_assert_async"):
        torch._assert_async(~invalid_result, message)
        return False
    return bool(invalid_result.item())


@dataclass(frozen=True)
class EmbeddingTableSpec:
    name: str
    num_embeddings: int
    embedding_dim: int
    element_size: int = 4

    @property
    def weight_bytes(self) -> int:
        return self.num_embeddings * self.embedding_dim * self.element_size


@dataclass(frozen=True)
class EmbeddingShardSpec:
    table_name: str
    strategy: EmbeddingShardStrategy
    world_size: int
    cyclic_offset: int = 0
    table_owner: int | None = None

    def owner(self, ids: Tensor) -> Tensor:
        if self.strategy == "row_wise":
            return torch.remainder(ids + self.cyclic_offset, self.world_size)
        if self.table_owner is None:
            raise RuntimeError("table-wise shard is missing its owner")
        return torch.full_like(ids, self.table_owner)

    def local_rows(self, num_embeddings: int, rank: int) -> int:
        if self.strategy == "table_wise":
            return num_embeddings if rank == self.table_owner else 0
        residue = (rank - self.cyclic_offset) % self.world_size
        if residue >= num_embeddings:
            return 0
        return (num_embeddings - 1 - residue) // self.world_size + 1

    def local_row_ids(self, global_ids: Tensor) -> Tensor:
        if self.strategy == "table_wise":
            return global_ids
        return torch.div(global_ids, self.world_size, rounding_mode="floor")


@dataclass(frozen=True)
class EmbeddingShardingPlan:
    world_size: int
    tables: dict[str, EmbeddingShardSpec]
    fingerprint: str


@dataclass
class EmbeddingCommunicationStats:
    table_name: str
    raw_ids: int = 0
    active_ids: int = 0
    local_unique_ids: int = 0
    owner_unique_ids: int = 0
    sent_ids: int = 0
    received_ids: int = 0
    forward_sent_bytes: int = 0
    forward_received_bytes: int = 0
    backward_sent_bytes: int = 0
    backward_received_bytes: int = 0
    forward_collective_enqueue_seconds: float = 0.0
    backward_collective_enqueue_seconds: float = 0.0

    @property
    def total_communication_bytes(self) -> int:
        return (
            self.forward_sent_bytes
            + self.forward_received_bytes
            + self.backward_sent_bytes
            + self.backward_received_bytes
        )

    def add_(self, other: "EmbeddingCommunicationStats") -> None:
        if self.table_name != other.table_name:
            raise ValueError("cannot combine communication stats for different tables")
        for name in (
            "raw_ids",
            "active_ids",
            "local_unique_ids",
            "owner_unique_ids",
            "sent_ids",
            "received_ids",
            "forward_sent_bytes",
            "forward_received_bytes",
            "backward_sent_bytes",
            "backward_received_bytes",
            "forward_collective_enqueue_seconds",
            "backward_collective_enqueue_seconds",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))


class _EmbeddingStatsSink:
    def __init__(self, table_name: str) -> None:
        self.table_name = table_name
        self._stats = EmbeddingCommunicationStats(table_name=table_name)

    def record_forward(self, stats: EmbeddingCommunicationStats) -> None:
        self._stats.add_(stats)

    def record_backward(
        self,
        *,
        sent_bytes: int,
        received_bytes: int,
        enqueue_seconds: float,
    ) -> None:
        self._stats.backward_sent_bytes += sent_bytes
        self._stats.backward_received_bytes += received_bytes
        self._stats.backward_collective_enqueue_seconds += enqueue_seconds

    def consume(self) -> EmbeddingCommunicationStats:
        result = self._stats
        self._stats = EmbeddingCommunicationStats(table_name=self.table_name)
        return result


def _stable_offset(table_name: str, world_size: int) -> int:
    if world_size <= 1:
        return 0
    digest = sha256(table_name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % world_size


def plan_embedding_shards(
    table_specs: Iterable[EmbeddingTableSpec],
    *,
    world_size: int,
    strategy: Literal["auto", "row_wise", "table_wise"],
    table_wise_max_rows: int,
) -> EmbeddingShardingPlan:
    """Build a deterministic memory-aware plan shared by every rank."""

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if strategy not in {"auto", "row_wise", "table_wise"}:
        raise ValueError("strategy must be auto, row_wise, or table_wise")
    specs = sorted(table_specs, key=lambda item: item.name)
    if len({item.name for item in specs}) != len(specs):
        raise ValueError("embedding table names must be unique")
    loads = [0] * world_size
    planned: dict[str, EmbeddingShardSpec] = {}
    table_wise: list[EmbeddingTableSpec] = []

    for table in specs:
        use_table_wise = strategy == "table_wise" or (
            strategy == "auto" and table.num_embeddings <= table_wise_max_rows
        )
        if use_table_wise:
            table_wise.append(table)
            continue
        offset = _stable_offset(table.name, world_size)
        shard = EmbeddingShardSpec(
            table_name=table.name,
            strategy="row_wise",
            world_size=world_size,
            cyclic_offset=offset,
        )
        planned[table.name] = shard
        # Include weight plus an FP32 Adagrad accumulator in the planning load.
        for rank in range(world_size):
            local_rows = shard.local_rows(table.num_embeddings, rank)
            loads[rank] += local_rows * table.embedding_dim * (
                table.element_size + 4
            )

    for table in sorted(table_wise, key=lambda item: (-item.weight_bytes, item.name)):
        owner = min(range(world_size), key=lambda rank: (loads[rank], rank))
        planned[table.name] = EmbeddingShardSpec(
            table_name=table.name,
            strategy="table_wise",
            world_size=world_size,
            table_owner=owner,
        )
        loads[owner] += table.num_embeddings * table.embedding_dim * (
            table.element_size + 4
        )

    payload = {
        "world_size": world_size,
        "tables": {
            name: asdict(planned[name])
            for name in sorted(planned)
        },
    }
    fingerprint = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return EmbeddingShardingPlan(
        world_size=world_size,
        tables=planned,
        fingerprint=fingerprint,
    )


def _distributed_rank_world(
    process_group: torch_dist.ProcessGroup | None,
) -> tuple[int, int]:
    if not torch_dist.is_available() or not torch_dist.is_initialized():
        return 0, 1
    return (
        torch_dist.get_rank(process_group),
        torch_dist.get_world_size(process_group),
    )


def _exchange_counts(
    send_splits: tuple[int, ...],
    device: torch.device,
    process_group: torch_dist.ProcessGroup | None,
) -> tuple[int, ...]:
    if len(send_splits) == 1:
        return send_splits
    send = torch.tensor(send_splits, dtype=torch.long, device=device)
    received = torch.empty_like(send)
    torch_dist.all_to_all_single(received, send, group=process_group)
    return tuple(int(value) for value in received.cpu().tolist())


def _all_to_all_variable(
    values: Tensor,
    send_splits: tuple[int, ...],
    recv_splits: tuple[int, ...],
    process_group: torch_dist.ProcessGroup | None,
) -> Tensor:
    if len(send_splits) == 1:
        return values.clone()
    output = values.new_empty((sum(recv_splits), *values.shape[1:]))
    torch_dist.all_to_all_single(
        output,
        values.contiguous(),
        output_split_sizes=list(recv_splits),
        input_split_sizes=list(send_splits),
        group=process_group,
    )
    return output


class _ShardedEmbeddingLookup(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        local_weight: Tensor,
        indices: Tensor,
        num_embeddings: int,
        padding_idx: int,
        shard_spec: EmbeddingShardSpec,
        local_dedup: bool,
        process_group: torch_dist.ProcessGroup | None,
        stats_sink: _EmbeddingStatsSink,
        table_name: str,
    ) -> Tensor:
        if indices.dtype != torch.long:
            raise TypeError("sharded embedding indices must be torch.long")
        rank, world_size = _distributed_rank_world(process_group)
        if world_size != shard_spec.world_size:
            raise RuntimeError(
                f"embedding plan world_size={shard_spec.world_size} does not match "
                f"process group world_size={world_size}"
            )
        flat = indices.reshape(-1)
        invalid = (flat < 0) | (flat >= num_embeddings)
        if _invalid_any(
            invalid,
            f"embedding {table_name!r} received an out-of-range ID",
        ):
            examples = flat[invalid][:5].detach().cpu().tolist()
            raise IndexError(
                f"embedding {table_name!r} received IDs outside [0, {num_embeddings}): "
                f"{examples}"
            )
        active_mask = flat != padding_idx
        active_positions = torch.nonzero(active_mask, as_tuple=False).flatten()
        active_ids = flat.index_select(0, active_positions)
        if local_dedup:
            requester_ids, requester_inverse = torch.unique(
                active_ids,
                sorted=True,
                return_inverse=True,
            )
        else:
            requester_ids = active_ids
            requester_inverse = torch.arange(
                active_ids.numel(), dtype=torch.long, device=indices.device
            )

        owners = shard_spec.owner(requester_ids)
        send_order = torch.argsort(owners, stable=True)
        sorted_ids = requester_ids.index_select(0, send_order)
        send_splits_tensor = torch.bincount(owners, minlength=world_size)
        send_splits = tuple(int(value) for value in send_splits_tensor.cpu().tolist())

        communication_started = perf_counter()
        with torch.profiler.record_function(
            f"sharded_embedding::{table_name}::forward_all_to_all"
        ):
            recv_splits = _exchange_counts(
                send_splits, indices.device, process_group
            )
            received_ids = _all_to_all_variable(
                sorted_ids, send_splits, recv_splits, process_group
            )
            received_local_rows = shard_spec.local_row_ids(received_ids)
            if received_ids.numel():
                expected_owner = shard_spec.owner(received_ids)
                if _invalid_any(
                    expected_owner != rank,
                    "received embedding IDs owned by another rank",
                ):
                    raise RuntimeError("received embedding IDs owned by another rank")
            owner_unique_rows, owner_inverse = torch.unique(
                received_local_rows,
                sorted=True,
                return_inverse=True,
            )
            owner_unique_values = local_weight.index_select(0, owner_unique_rows)
            received_values = owner_unique_values.index_select(0, owner_inverse)
            returned_values = _all_to_all_variable(
                received_values,
                recv_splits,
                send_splits,
                process_group,
            )
        communication_seconds = perf_counter() - communication_started

        requester_values = local_weight.new_empty(
            (requester_ids.numel(), local_weight.size(1))
        )
        if send_order.numel():
            requester_values.index_copy_(0, send_order, returned_values)
        active_values = requester_values.index_select(0, requester_inverse)
        output = local_weight.new_zeros((flat.numel(), local_weight.size(1)))
        if active_positions.numel():
            output.index_copy_(0, active_positions, active_values)

        ctx.process_group = process_group
        ctx.stats_sink = stats_sink
        ctx.world_size = world_size
        ctx.local_weight_shape = tuple(local_weight.shape)
        ctx.local_weight_dtype = local_weight.dtype
        ctx.send_splits = send_splits
        ctx.recv_splits = recv_splits
        ctx.save_for_backward(
            active_positions,
            requester_inverse,
            send_order,
            owner_unique_rows,
            owner_inverse,
        )

        id_bytes = indices.element_size()
        value_bytes = local_weight.element_size() * local_weight.size(1)
        stats_sink.record_forward(
            EmbeddingCommunicationStats(
                table_name=table_name,
                raw_ids=flat.numel(),
                active_ids=active_ids.numel(),
                local_unique_ids=requester_ids.numel(),
                owner_unique_ids=owner_unique_rows.numel(),
                sent_ids=sorted_ids.numel(),
                received_ids=received_ids.numel(),
                forward_sent_bytes=(
                    sorted_ids.numel() * id_bytes
                    + received_values.size(0) * value_bytes
                ),
                forward_received_bytes=(
                    received_ids.numel() * id_bytes
                    + returned_values.size(0) * value_bytes
                ),
                forward_collective_enqueue_seconds=communication_seconds,
            )
        )
        return output.view(*indices.shape, local_weight.size(1))

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Any, ...]:  # type: ignore[override]
        (
            active_positions,
            requester_inverse,
            send_order,
            owner_unique_rows,
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

        communication_started = perf_counter()
        with torch.profiler.record_function("sharded_embedding::backward_all_to_all"):
            received_grad = _all_to_all_variable(
                sorted_grad,
                ctx.send_splits,
                ctx.recv_splits,
                ctx.process_group,
            )
        communication_seconds = perf_counter() - communication_started
        owner_grad = received_grad.new_zeros(
            (owner_unique_rows.numel(), embedding_dim)
        )
        if owner_inverse.numel():
            owner_grad.index_add_(0, owner_inverse, received_grad)
        owner_grad.div_(float(ctx.world_size))
        owner_grad = owner_grad.to(dtype=ctx.local_weight_dtype)
        local_weight_grad = torch.sparse_coo_tensor(
            owner_unique_rows.unsqueeze(0),
            owner_grad,
            size=ctx.local_weight_shape,
            dtype=ctx.local_weight_dtype,
            device=grad_output.device,
            is_coalesced=True,
        )
        value_bytes = grad_output.element_size() * embedding_dim
        ctx.stats_sink.record_backward(
            sent_bytes=sorted_grad.size(0) * value_bytes,
            received_bytes=received_grad.size(0) * value_bytes,
            enqueue_seconds=communication_seconds,
        )
        return (
            local_weight_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class ShardedEmbedding(nn.Module):
    """An ``nn.Embedding``-compatible local shard with owner-based routing."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        table_name: str,
        shard_spec: EmbeddingShardSpec,
        padding_idx: int = 0,
        local_dedup: bool = True,
        init_std: float = 0.02,
        process_group: torch_dist.ProcessGroup | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("num_embeddings and embedding_dim must be positive")
        if not 0 <= padding_idx < num_embeddings:
            raise ValueError("padding_idx must be inside the embedding table")
        rank, world_size = _distributed_rank_world(process_group)
        if world_size != shard_spec.world_size:
            raise ValueError("shard spec world size does not match the active process group")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.sparse = True
        self.table_name = table_name
        self.shard_spec = shard_spec
        self.local_dedup = local_dedup
        self.init_std = init_std
        self.process_group = process_group
        self.rank = rank
        self.world_size = world_size
        local_rows = shard_spec.local_rows(num_embeddings, rank)
        self.weight = nn.Parameter(
            torch.empty(local_rows, embedding_dim, dtype=dtype)
        )
        self._stats_sink = _EmbeddingStatsSink(table_name)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        table_seed = int.from_bytes(
            sha256(self.table_name.encode("utf-8")).digest()[:8], "little"
        )
        seed = (torch.initial_seed() + table_seed + self.rank * 104729) % (2**63 - 1)
        generator = torch.Generator(device=self.weight.device)
        generator.manual_seed(seed)
        nn.init.normal_(
            self.weight,
            mean=0.0,
            std=self.init_std,
            generator=generator,
        )
        if self.shard_spec.owner(
            torch.tensor([self.padding_idx], dtype=torch.long)
        ).item() == self.rank:
            local_padding = int(
                self.shard_spec.local_row_ids(
                    torch.tensor([self.padding_idx], dtype=torch.long)
                ).item()
            )
            with torch.no_grad():
                self.weight[local_padding].zero_()

    def forward(self, indices: Tensor) -> Tensor:
        return _ShardedEmbeddingLookup.apply(
            self.weight,
            indices,
            self.num_embeddings,
            self.padding_idx,
            self.shard_spec,
            self.local_dedup,
            self.process_group,
            self._stats_sink,
            self.table_name,
        )

    @torch.no_grad()
    def load_full_weight_(self, full_weight: Tensor) -> None:
        """Load a small reference table slice for tests and migration tools."""

        expected = (self.num_embeddings, self.embedding_dim)
        if tuple(full_weight.shape) != expected:
            raise ValueError(f"full weight must have shape {expected}")
        global_ids = torch.arange(
            self.num_embeddings,
            dtype=torch.long,
            device=full_weight.device,
        )
        owned = self.shard_spec.owner(global_ids) == self.rank
        local_values = full_weight.index_select(0, global_ids[owned])
        if tuple(local_values.shape) != tuple(self.weight.shape):
            raise RuntimeError("full weight slice does not match local shard shape")
        self.weight.copy_(local_values.to(device=self.weight.device, dtype=self.weight.dtype))

    def consume_communication_stats(self) -> EmbeddingCommunicationStats:
        return self._stats_sink.consume()

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
            f"local_rows={self.weight.size(0)}, strategy={self.shard_spec.strategy}, "
            f"rank={self.rank}/{self.world_size}, padding_idx={self.padding_idx}"
        )


@dataclass(frozen=True)
class _GroupedLookupMetadata:
    modules: tuple[ShardedEmbedding, ...]
    request_table_indices: tuple[int, ...]
    request_numels: tuple[int, ...]
    table_offsets: tuple[int, ...]
    local_dedup: bool
    process_group: torch_dist.ProcessGroup | None


def _table_indices_for_keys(keys: Tensor, table_offsets: tuple[int, ...]) -> Tensor:
    if len(table_offsets) == 1:
        return torch.zeros_like(keys)
    boundaries = torch.tensor(
        table_offsets[1:], dtype=keys.dtype, device=keys.device
    )
    return torch.bucketize(keys, boundaries, right=True)


class _GroupedShardedEmbeddingLookup(torch.autograd.Function):
    """Route several compatible embedding tables with one collective group.

    The first two arguments are non-parameter inputs; every remaining argument
    is one distinct local table weight. Keeping weights as explicit autograd
    inputs is important: aliases may issue several requests, but each physical
    table still receives exactly one sparse gradient.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        packed_indices: Tensor,
        metadata: _GroupedLookupMetadata,
        *local_weights: Tensor,
    ) -> Tensor:
        modules = metadata.modules
        if len(modules) != len(local_weights):
            raise RuntimeError("grouped embedding metadata/weight count mismatch")
        if packed_indices.dtype != torch.long:
            raise TypeError("sharded embedding indices must be torch.long")
        if not modules:
            raise ValueError("grouped embedding lookup requires at least one table")
        embedding_dim = modules[0].embedding_dim
        rank, world_size = _distributed_rank_world(metadata.process_group)
        if any(module.world_size != world_size for module in modules):
            raise RuntimeError("embedding plan does not match the active process group")

        active_positions_parts: list[Tensor] = []
        active_keys_parts: list[Tensor] = []
        active_owner_parts: list[Tensor] = []
        raw_offset = 0
        request_offset = 0
        raw_by_table = [0] * len(modules)
        active_by_table = [0] * len(modules)
        for table_index, request_numel in zip(
            metadata.request_table_indices, metadata.request_numels
        ):
            module = modules[table_index]
            request = packed_indices.narrow(0, request_offset, request_numel)
            request_offset += request_numel
            invalid = (request < 0) | (request >= module.num_embeddings)
            if _invalid_any(
                invalid,
                f"embedding {module.table_name!r} received an out-of-range ID",
            ):
                examples = request[invalid][:5].detach().cpu().tolist()
                raise IndexError(
                    f"embedding {module.table_name!r} received IDs outside "
                    f"[0, {module.num_embeddings}): {examples}"
                )
            active_mask = request != module.padding_idx
            positions = torch.nonzero(active_mask, as_tuple=False).flatten()
            active_ids = request.index_select(0, positions)
            active_positions_parts.append(positions + raw_offset)
            active_keys_parts.append(active_ids + metadata.table_offsets[table_index])
            active_owner_parts.append(module.shard_spec.owner(active_ids))
            raw_by_table[table_index] += request_numel
            active_by_table[table_index] += int(active_ids.numel())
            raw_offset += request_numel
        if request_offset != packed_indices.numel():
            raise RuntimeError("grouped embedding request sizes do not cover packed IDs")

        active_positions = torch.cat(active_positions_parts)
        active_keys = torch.cat(active_keys_parts)
        active_owners = torch.cat(active_owner_parts)
        if metadata.local_dedup:
            requester_keys, requester_inverse = torch.unique(
                active_keys, sorted=True, return_inverse=True
            )
            requester_owners = active_owners.new_empty(requester_keys.numel())
            if requester_inverse.numel():
                requester_owners.scatter_(0, requester_inverse, active_owners)
        else:
            requester_keys = active_keys
            requester_inverse = torch.arange(
                active_keys.numel(), dtype=torch.long, device=active_keys.device
            )
            requester_owners = active_owners

        send_order = torch.argsort(requester_owners, stable=True)
        sorted_keys = requester_keys.index_select(0, send_order)
        send_splits_tensor = torch.bincount(requester_owners, minlength=world_size)
        send_splits = tuple(int(value) for value in send_splits_tensor.cpu().tolist())

        communication_started = perf_counter()
        with torch.profiler.record_function(
            f"sharded_embedding_group::{embedding_dim}::forward_all_to_all"
        ):
            recv_splits = _exchange_counts(
                send_splits, packed_indices.device, metadata.process_group
            )
            received_keys = _all_to_all_variable(
                sorted_keys, send_splits, recv_splits, metadata.process_group
            )
            owner_unique_keys, owner_inverse = torch.unique(
                received_keys, sorted=True, return_inverse=True
            )
            owner_unique_values = local_weights[0].new_empty(
                (owner_unique_keys.numel(), embedding_dim)
            )
            owner_table_indices = _table_indices_for_keys(
                owner_unique_keys, metadata.table_offsets
            )
            for table_index, (module, local_weight) in enumerate(
                zip(modules, local_weights)
            ):
                selected = torch.nonzero(
                    owner_table_indices == table_index, as_tuple=False
                ).flatten()
                if not selected.numel():
                    continue
                global_ids = owner_unique_keys.index_select(0, selected)
                global_ids = global_ids - metadata.table_offsets[table_index]
                expected_owner = module.shard_spec.owner(global_ids)
                if _invalid_any(
                    expected_owner != rank,
                    "received embedding IDs owned by another rank",
                ):
                    raise RuntimeError("received embedding IDs owned by another rank")
                local_rows = module.shard_spec.local_row_ids(global_ids)
                values = local_weight.index_select(0, local_rows)
                owner_unique_values.index_copy_(0, selected, values)
            received_values = owner_unique_values.index_select(0, owner_inverse)
            returned_values = _all_to_all_variable(
                received_values,
                recv_splits,
                send_splits,
                metadata.process_group,
            )
        communication_seconds = perf_counter() - communication_started

        requester_values = local_weights[0].new_empty(
            (requester_keys.numel(), embedding_dim)
        )
        if send_order.numel():
            requester_values.index_copy_(0, send_order, returned_values)
        active_values = requester_values.index_select(0, requester_inverse)
        output = local_weights[0].new_zeros(
            (packed_indices.numel(), embedding_dim)
        )
        if active_positions.numel():
            output.index_copy_(0, active_positions, active_values)

        ctx.lookup_metadata = metadata
        ctx.world_size = world_size
        ctx.local_weight_shapes = tuple(tuple(weight.shape) for weight in local_weights)
        ctx.local_weight_dtypes = tuple(weight.dtype for weight in local_weights)
        ctx.send_splits = send_splits
        ctx.recv_splits = recv_splits
        ctx.save_for_backward(
            active_positions,
            requester_inverse,
            send_order,
            sorted_keys,
            owner_unique_keys,
            owner_inverse,
        )

        requester_table_indices = _table_indices_for_keys(
            requester_keys, metadata.table_offsets
        )
        received_table_indices = _table_indices_for_keys(
            received_keys, metadata.table_offsets
        )
        returned_table_indices = _table_indices_for_keys(
            sorted_keys, metadata.table_offsets
        )
        owner_unique_table_indices = _table_indices_for_keys(
            owner_unique_keys, metadata.table_offsets
        )
        id_bytes = packed_indices.element_size()
        value_bytes = local_weights[0].element_size() * embedding_dim
        for table_index, module in enumerate(modules):
            local_unique = int((requester_table_indices == table_index).sum().item())
            received = int((received_table_indices == table_index).sum().item())
            returned = int((returned_table_indices == table_index).sum().item())
            owner_unique = int(
                (owner_unique_table_indices == table_index).sum().item()
            )
            module._stats_sink.record_forward(
                EmbeddingCommunicationStats(
                    table_name=module.table_name,
                    raw_ids=raw_by_table[table_index],
                    active_ids=active_by_table[table_index],
                    local_unique_ids=local_unique,
                    owner_unique_ids=owner_unique,
                    sent_ids=local_unique,
                    received_ids=received,
                    forward_sent_bytes=(local_unique * id_bytes + received * value_bytes),
                    forward_received_bytes=(
                        received * id_bytes + returned * value_bytes
                    ),
                    # One elapsed interval covers the whole fused collective.
                    # Attribute it once so summing table stats does not inflate it.
                    forward_collective_enqueue_seconds=(
                        communication_seconds if table_index == 0 else 0.0
                    ),
                )
            )
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Any, ...]:  # type: ignore[override]
        metadata: _GroupedLookupMetadata = ctx.lookup_metadata
        (
            active_positions,
            requester_inverse,
            send_order,
            sorted_keys,
            owner_unique_keys,
            owner_inverse,
        ) = ctx.saved_tensors
        embedding_dim = metadata.modules[0].embedding_dim
        flat_grad = grad_output.reshape(-1, embedding_dim)
        active_grad = flat_grad.index_select(0, active_positions)
        requester_grad = flat_grad.new_zeros((send_order.numel(), embedding_dim))
        if requester_inverse.numel():
            requester_grad.index_add_(0, requester_inverse, active_grad)
        sorted_grad = requester_grad.index_select(0, send_order)

        communication_started = perf_counter()
        with torch.profiler.record_function(
            f"sharded_embedding_group::{embedding_dim}::backward_all_to_all"
        ):
            received_grad = _all_to_all_variable(
                sorted_grad,
                ctx.send_splits,
                ctx.recv_splits,
                metadata.process_group,
            )
        communication_seconds = perf_counter() - communication_started
        owner_grad = received_grad.new_zeros(
            (owner_unique_keys.numel(), embedding_dim)
        )
        if owner_inverse.numel():
            owner_grad.index_add_(0, owner_inverse, received_grad)
        owner_grad.div_(float(ctx.world_size))

        owner_table_indices = _table_indices_for_keys(
            owner_unique_keys, metadata.table_offsets
        )
        local_weight_grads: list[Tensor] = []
        for table_index, module in enumerate(metadata.modules):
            selected = torch.nonzero(
                owner_table_indices == table_index, as_tuple=False
            ).flatten()
            global_ids = owner_unique_keys.index_select(0, selected)
            global_ids = global_ids - metadata.table_offsets[table_index]
            local_rows = module.shard_spec.local_row_ids(global_ids)
            values = owner_grad.index_select(0, selected).to(
                dtype=ctx.local_weight_dtypes[table_index]
            )
            local_weight_grads.append(
                torch.sparse_coo_tensor(
                    local_rows.unsqueeze(0),
                    values,
                    size=ctx.local_weight_shapes[table_index],
                    dtype=ctx.local_weight_dtypes[table_index],
                    device=grad_output.device,
                    is_coalesced=True,
                )
            )

        sent_table_indices = _table_indices_for_keys(
            sorted_keys, metadata.table_offsets
        )
        value_bytes = grad_output.element_size() * embedding_dim
        received_table_indices = owner_table_indices.index_select(0, owner_inverse)
        for table_index, module in enumerate(metadata.modules):
            sent_count = int((sent_table_indices == table_index).sum().item())
            received_count = int(
                (received_table_indices == table_index).sum().item()
            )
            module._stats_sink.record_backward(
                sent_bytes=sent_count * value_bytes,
                received_bytes=received_count * value_bytes,
                enqueue_seconds=(communication_seconds if table_index == 0 else 0.0),
            )
        return (None, None, *local_weight_grads)


def grouped_sharded_embedding_lookup(
    requests: Iterable[tuple[ShardedEmbedding, Tensor]],
) -> list[Tensor]:
    """Lookup compatible tables in grouped owner-based collectives.

    Requests are partitioned by device, dtype, embedding width, process group,
    and dedup policy. A singleton retains the simpler per-table implementation;
    all other requests in a partition share one count exchange, request route,
    and response route.
    """

    request_list = list(requests)
    if not request_list:
        return []
    outputs: list[Tensor | None] = [None] * len(request_list)
    groups: dict[tuple[Any, ...], list[int]] = {}
    for request_index, (module, indices) in enumerate(request_list):
        if indices.dtype != torch.long:
            raise TypeError("sharded embedding indices must be torch.long")
        key = (
            indices.device,
            module.weight.device,
            module.weight.dtype,
            module.embedding_dim,
            module.world_size,
            module.local_dedup,
            id(module.process_group),
        )
        groups.setdefault(key, []).append(request_index)

    for group_indices in groups.values():
        if len(group_indices) == 1:
            request_index = group_indices[0]
            module, indices = request_list[request_index]
            outputs[request_index] = module(indices)
            continue
        unique_modules: list[ShardedEmbedding] = []
        module_index_by_id: dict[int, int] = {}
        request_table_indices: list[int] = []
        flat_indices: list[Tensor] = []
        request_numels: list[int] = []
        request_shapes: list[torch.Size] = []
        for request_index in group_indices:
            module, indices = request_list[request_index]
            module_id = id(module)
            table_index = module_index_by_id.get(module_id)
            if table_index is None:
                table_index = len(unique_modules)
                module_index_by_id[module_id] = table_index
                unique_modules.append(module)
            request_table_indices.append(table_index)
            flat_indices.append(indices.reshape(-1))
            request_numels.append(indices.numel())
            request_shapes.append(indices.shape)
        table_offsets: list[int] = []
        next_offset = 0
        for module in unique_modules:
            table_offsets.append(next_offset)
            next_offset += module.num_embeddings
        metadata = _GroupedLookupMetadata(
            modules=tuple(unique_modules),
            request_table_indices=tuple(request_table_indices),
            request_numels=tuple(request_numels),
            table_offsets=tuple(table_offsets),
            local_dedup=unique_modules[0].local_dedup,
            process_group=unique_modules[0].process_group,
        )
        packed_indices = torch.cat(flat_indices)
        packed_output = _GroupedShardedEmbeddingLookup.apply(
            packed_indices,
            metadata,
            *(module.weight for module in unique_modules),
        )
        offset = 0
        for request_index, request_numel, request_shape in zip(
            group_indices, request_numels, request_shapes
        ):
            part = packed_output.narrow(0, offset, request_numel)
            outputs[request_index] = part.view(*request_shape, packed_output.size(-1))
            offset += request_numel

    if any(output is None for output in outputs):
        raise RuntimeError("grouped embedding lookup did not produce every output")
    return [output for output in outputs if output is not None]


def sharded_embedding_modules(module: nn.Module) -> list[ShardedEmbedding]:
    return [item for item in module.modules() if isinstance(item, ShardedEmbedding)]


def consume_sharded_embedding_stats(
    module: nn.Module,
) -> list[EmbeddingCommunicationStats]:
    return [item.consume_communication_stats() for item in sharded_embedding_modules(module)]
