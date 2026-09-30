"""One append-only int64 embedding table per rank.

Keys are the hashed ids already stored in the batch. The first time a key is
seen it is inserted; there is no configured capacity and no eviction. Row 0
is a permanent zero vector for absent positions, not a key.
"""

from __future__ import annotations

from collections.abc import Iterable
import weakref
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..features import CATEGORICAL_MISSING_ID


class RankTableStats:
    def __init__(self, *, active_entries: int, current_step: int) -> None:
        self.active_entries = active_entries
        self.current_step = current_step


class RankEmbeddingTable(nn.Module):
    """Rank-local unbounded map from int64 key to an embedding row."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        init_std: float = 0.02,
        sparse: bool = True,
        dtype: torch.dtype = torch.float32,
        row_sharded: bool = False,
        process_group: Any = None,
    ) -> None:
        super().__init__()
        if type(embedding_dim) is not int or embedding_dim <= 0:
            raise ValueError("rank table embedding_dim must be a positive integer")
        if not torch.isfinite(torch.tensor(float(init_std))) or init_std <= 0.0:
            raise ValueError("rank table init_std must be positive and finite")
        self.embedding_dim = int(embedding_dim)
        self.init_std = float(init_std)
        self.sparse = bool(sparse)
        self.row_sharded = bool(row_sharded)
        self.process_group = process_group
        self.weight = nn.Parameter(
            torch.zeros((1, self.embedding_dim), dtype=dtype),
            requires_grad=True,
        )
        self._key_to_row: dict[int, int] = {}
        self._live_rows = 1
        self._namespace_names: list[str] = []
        self._namespace_to_id: dict[str, int] = {}
        self._policy_prepared = False
        self._current_step = 0
        self._optimizers: list[weakref.ReferenceType[Any]] = []
        self._bind_owner(self.weight)
        self._mdl_id_embedding = True

    @property
    def num_embeddings(self) -> int:
        return int(self.weight.shape[0])

    @property
    def namespace_names(self) -> tuple[str, ...]:
        return tuple(self._namespace_names)

    def register_namespace(self, name: str, policy: Any = None) -> int:
        del policy
        if not isinstance(name, str) or not name:
            raise ValueError("rank table namespace name must be a non-empty string")
        existing = self._namespace_to_id.get(name)
        if existing is not None:
            return existing
        namespace_id = len(self._namespace_names)
        self._namespace_names.append(name)
        self._namespace_to_id[name] = namespace_id
        return namespace_id

    def attach_optimizer(self, optimizer: Any) -> None:
        self._optimizers.append(weakref.ref(optimizer))

    def row_for(self, key: int) -> int | None:
        return self._key_to_row.get(int(key))

    def contains(self, key: int) -> bool:
        return int(key) in self._key_to_row

    def stats(self) -> RankTableStats:
        return RankTableStats(
            active_entries=max(0, self._live_rows - 1),
            current_step=self._current_step,
        )

    def set_batch_outcomes(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._policy_prepared = False

    def clear_batch_outcomes(self) -> None:
        return None

    def batch_outcome_counts(
        self,
        row_count: int,
        row_indices: Tensor | None = None,
    ) -> tuple[None, None]:
        del row_count, row_indices
        return None, None

    def consume_pending_optimizer_reset_rows(self) -> Tensor:
        return torch.zeros((0,), dtype=torch.long)

    def optimizer_step_completed(self) -> None:
        self._current_step += 1
        self._policy_prepared = False

    def insert_ids(self, keys: Tensor) -> None:
        """Append one new row for each unseen int64 key."""

        if keys.numel() == 0:
            return
        unique = torch.unique(keys.detach().to(device="cpu", dtype=torch.int64))
        missing = [
            int(value)
            for value in unique.tolist()
            if int(value) != CATEGORICAL_MISSING_ID and int(value) not in self._key_to_row
        ]
        if not missing:
            return
        missing.sort()
        self._ensure_storage(self._live_rows + len(missing))
        rows: list[int] = []
        for key in missing:
            row = self._live_rows
            self._live_rows += 1
            self._key_to_row[key] = row
            rows.append(row)
        self._initialize_rows(rows)

    def apply_policy_records(self, records: Tensor) -> None:
        if records.numel() == 0:
            return
        if records.ndim != 2 or int(records.size(1)) < 2:
            raise ValueError("rank table policy records must have shape [N, >=2]")
        self.insert_ids(records[:, 1])

    def apply_synchronized_policy_records(self, local_records: Tensor) -> None:
        from .gset import (
            exchange_owned_gset_policy_records,
            gather_and_merge_gset_policy_records,
        )

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
        self._policy_prepared = True

    def count_policy_records(
        self,
        namespace: str | int,
        logical_ids: Tensor,
        *,
        padding_idx: int | None = None,
        valid_lengths: Tensor | None = None,
        allow_negative_ids: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        del kwargs
        from .gset import empty_gset_policy_records

        flat = logical_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        keep = torch.ones(flat.numel(), dtype=torch.bool)
        if logical_ids.ndim == 2 and valid_lengths is not None:
            lengths = valid_lengths.detach().to(device="cpu", dtype=torch.int64)
            lengths = lengths.clamp(min=0, max=logical_ids.size(1))
            positions = torch.arange(logical_ids.size(1), dtype=torch.int64).view(1, -1)
            keep = (positions < lengths.view(-1, 1)).reshape(-1)
        if not allow_negative_ids:
            keep = keep & (flat >= 0)
        if padding_idx is not None:
            keep = keep & (flat != int(padding_idx))
        keep = keep & (flat != CATEGORICAL_MISSING_ID)
        if not bool(keep.any()):
            return empty_gset_policy_records()
        unique = torch.unique(flat[keep], sorted=True)
        records = torch.zeros((unique.numel(), 5), dtype=torch.int64)
        namespace_id = namespace if type(namespace) is int else 0
        records[:, 0] = int(namespace_id)
        records[:, 1] = unique
        records[:, 2] = 1
        return records

    def lookup(
        self,
        namespace: str | int,
        logical_ids: Tensor,
        *,
        padding_idx: int | None = None,
        allow_negative_ids: bool = False,
        valid_lengths: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        del namespace, kwargs
        if self.training and not self._policy_prepared and not self.row_sharded:
            self.insert_ids(
                self._active_ids(
                    logical_ids,
                    padding_idx=padding_idx,
                    allow_negative_ids=allow_negative_ids,
                    valid_lengths=valid_lengths,
                )
            )
        from .gset import _ShardedGSETLookup, _distributed_rank_world

        _rank, world_size = _distributed_rank_world(self.process_group)
        if self.row_sharded and world_size > 1:
            output = _ShardedGSETLookup.apply(
                self.weight,
                logical_ids,
                self,
                0,
                padding_idx,
                bool(allow_negative_ids),
            )
            if not isinstance(output, Tensor):
                raise RuntimeError("rank table lookup returned an invalid result")
            return output
        slots = self._map_ids_to_slots(
            0,
            logical_ids,
            padding_idx=padding_idx,
            allow_negative_ids=allow_negative_ids,
            valid_lengths=valid_lengths,
        )
        return F.embedding(
            slots,
            self.weight,
            padding_idx=0,
            sparse=self.sparse,
        )

    def _map_ids_to_slots(
        self,
        namespace: str | int,
        logical_ids: Tensor,
        *,
        padding_idx: int | None = None,
        allow_negative_ids: bool = False,
        valid_lengths: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        del namespace, kwargs
        flat = logical_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        slots = torch.zeros(flat.numel(), dtype=torch.long)
        keep = self._keep_mask(
            flat,
            logical_ids,
            padding_idx=padding_idx,
            allow_negative_ids=allow_negative_ids,
            valid_lengths=valid_lengths,
        )
        if bool(keep.any()):
            mapped = [
                self._key_to_row.get(int(value), 0) for value in flat[keep].tolist()
            ]
            slots[keep] = torch.tensor(mapped, dtype=torch.long)
        return slots.to(device=logical_ids.device).view_as(logical_ids)

    def get_extra_state(self) -> dict[str, Any]:
        keys = torch.zeros((max(0, self._live_rows - 1),), dtype=torch.int64)
        for key, row in self._key_to_row.items():
            keys[row - 1] = int(key)
        return {"keys": keys, "live_rows": self._live_rows,
                "current_step": self._current_step}

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        saved = state_dict.get(prefix + "weight")
        if isinstance(saved, Tensor):
            if saved.ndim != 2 or saved.size(0) < 1 or saved.size(1) != self.embedding_dim:
                error_msgs.append(f"{prefix}weight has invalid rank-table shape {tuple(saved.shape)}")
                return
            if local_metadata.get("assign_to_params_buffers", False):
                error_msgs.append("RankEmbeddingTable requires load_state_dict(assign=False)")
                return
            # The constructor intentionally allocates only the missing-ID row.
            # Restore storage before Module copies weights. Preserve Parameter
            # identity and its owner link; optimizer state is restored afterwards.
            if self.weight.shape != saved.shape:
                with torch.no_grad():
                    self.weight.set_(self.weight.new_empty(saved.shape))
                self.weight.grad = None
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def set_extra_state(self, state: dict[str, Any]) -> None:
        keys = state.get("keys")
        if not isinstance(keys, Tensor):
            raise ValueError("rank table extra state is missing keys")
        keys = keys.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        live_rows = int(state.get("live_rows", keys.numel() + 1))
        if live_rows != keys.numel() + 1 or live_rows > self.num_embeddings:
            raise ValueError("rank table checkpoint live_rows does not match keys/storage")
        if keys.unique().numel() != keys.numel() or bool((keys == CATEGORICAL_MISSING_ID).any()):
            raise ValueError("rank table checkpoint contains duplicate or missing-ID keys")
        self._key_to_row = {int(key): index + 1 for index, key in enumerate(keys.tolist())}
        self._live_rows = live_rows
        self._current_step = int(state.get("current_step", 0))
        self._policy_prepared = False
        with torch.no_grad():
            self.weight[0].zero_()

    def _active_ids(
        self,
        logical_ids: Tensor,
        *,
        padding_idx: int | None,
        allow_negative_ids: bool,
        valid_lengths: Tensor | None,
    ) -> Tensor:
        flat = logical_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        keep = self._keep_mask(
            flat,
            logical_ids,
            padding_idx=padding_idx,
            allow_negative_ids=allow_negative_ids,
            valid_lengths=valid_lengths,
        )
        if not bool(keep.any()):
            return flat.new_empty((0,))
        return flat[keep]

    def _keep_mask(
        self,
        flat: Tensor,
        logical_ids: Tensor,
        *,
        padding_idx: int | None,
        allow_negative_ids: bool,
        valid_lengths: Tensor | None,
    ) -> Tensor:
        keep = torch.ones(flat.numel(), dtype=torch.bool)
        if logical_ids.ndim == 2 and valid_lengths is not None:
            lengths = valid_lengths.detach().to(device="cpu", dtype=torch.int64)
            lengths = lengths.clamp(min=0, max=logical_ids.size(1))
            positions = torch.arange(logical_ids.size(1), dtype=torch.int64).view(1, -1)
            keep = (positions < lengths.view(-1, 1)).reshape(-1)
        if not allow_negative_ids:
            keep = keep & (flat >= 0)
        if padding_idx is not None:
            keep = keep & (flat != int(padding_idx))
        return keep & (flat != CATEGORICAL_MISSING_ID)

    def _ensure_storage(self, need: int) -> None:
        current = int(self.weight.shape[0])
        if need <= current:
            return
        new_rows = current
        while new_rows < need:
            new_rows = max(new_rows * 2, 16)
        old = self.weight
        fresh = old.data.new_empty((new_rows, old.shape[1]))
        with torch.no_grad():
            fresh[:current].copy_(old.data)
            fresh[current:].zero_()
        new_param = nn.Parameter(fresh, requires_grad=old.requires_grad)
        if old.grad is not None:
            grad = old.grad
            if grad.is_sparse:
                coalesced = grad.coalesce()
                new_param.grad = torch.sparse_coo_tensor(
                    coalesced.indices(),
                    coalesced.values(),
                    size=(new_rows, old.shape[1]),
                    dtype=coalesced.dtype,
                    device=coalesced.device,
                    is_coalesced=True,
                )
            else:
                grown = fresh.new_zeros((new_rows, old.shape[1]))
                grown[:current].copy_(grad)
                new_param.grad = grown
        self._parameters["weight"] = new_param
        self._bind_owner(new_param)
        from ..optim import rebind_optimizer_parameter

        alive: list[weakref.ReferenceType[Any]] = []
        for ref in self._optimizers:
            optimizer = ref()
            if optimizer is None:
                continue
            rebind_optimizer_parameter(optimizer, old, new_param)
            alive.append(ref)
        self._optimizers = alive

    def _initialize_rows(self, rows: Iterable[int]) -> None:
        row_list = list(rows)
        if not row_list:
            return
        index = torch.tensor(row_list, dtype=torch.long, device=self.weight.device)
        values = torch.empty(
            (index.numel(), self.embedding_dim),
            dtype=self.weight.dtype,
            device=self.weight.device,
        ).normal_(mean=0.0, std=self.init_std)
        with torch.no_grad():
            self.weight.index_copy_(0, index, values)
            self.weight[0].zero_()

    def _bind_owner(self, parameter: nn.Parameter) -> None:
        owner = weakref.ref(self)
        try:
            parameter._mdl_gset_owner = owner  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError):
            return
