"""Optimizers whose state follows the repository's local embedding shards."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable
import copy

import torch
from torch import Tensor, nn

from .modules.gset import gset_owner_for_parameter


_ROWWISE_ADAGRAD_KERNEL = None
_DIRTY_ROW_KERNEL = None


def _rowwise_adagrad_kernel():
    """Lazily compile the Triton kernel so CPU-only imports stay lightweight."""

    global _ROWWISE_ADAGRAD_KERNEL
    if _ROWWISE_ADAGRAD_KERNEL is not None:
        return _ROWWISE_ADAGRAD_KERNEL

    import triton
    import triton.language as tl

    # Nested @triton.jit kernels resolve annotations (e.g. ``tl.constexpr``)
    # against this module's globals, not the enclosing function locals. Bind
    # ``tl`` here so compile does not raise ``NameError('tl is not defined')``.
    globals()["tl"] = tl

    @triton.jit
    def _rowwise_adagrad_update_kernel(
        param_ptr,
        acc_ptr,
        rows_ptr,
        vals_ptr,
        n_touched,
        dim,
        lr,
        eps,
        stride_param,
        stride_vals,
        dirty_words_ptr,
        TRACK_DIRTY: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """One program per touched row: update accumulator and embedding row."""

        pid = tl.program_id(0)
        if pid >= n_touched:
            return
        row = tl.load(rows_ptr + pid)
        if TRACK_DIRTY:
            word = row // 32
            bit = (1 << (row % 32)).to(tl.int32)
            tl.atomic_or(dirty_words_ptr + word, bit)
        offs = tl.arange(0, BLOCK)
        mask = offs < dim
        vals = tl.load(vals_ptr + pid * stride_vals + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        sq_mean = tl.sum(vals * vals, axis=0) / dim
        acc = tl.load(acc_ptr + row) + sq_mean
        tl.store(acc_ptr + row, acc)
        denom = tl.sqrt(acc) + eps
        upd = vals / denom
        old = tl.load(param_ptr + row * stride_param + offs, mask=mask).to(tl.float32)
        tl.store(param_ptr + row * stride_param + offs, old - lr * upd, mask=mask)

    _ROWWISE_ADAGRAD_KERNEL = _rowwise_adagrad_update_kernel
    return _ROWWISE_ADAGRAD_KERNEL


def fused_rowwise_adagrad_update(
    parameter: torch.Tensor,
    accumulator: torch.Tensor,
    rows: torch.Tensor,
    values: torch.Tensor,
    *,
    lr: float,
    eps: float,
    dirty_words: torch.Tensor | None = None,
) -> None:
    """Fused row-wise Adagrad update for one embedding table (Triton).

    ``rows`` must be unique (coalesced sparse grads). ``values`` may be BF16 or
    FP32; accumulators stay FP32. Parameter may be BF16 or FP32.
    """

    import triton

    n_touched = int(rows.numel())
    if n_touched == 0:
        return
    dim = int(parameter.shape[1])
    rows = rows.contiguous()
    if values.dtype == torch.float32:
        vals = values.contiguous()
    else:
        vals = values.float().contiguous()
    block = triton.next_power_of_2(dim)
    kernel = _rowwise_adagrad_kernel()
    kernel[(n_touched,)](
        parameter,
        accumulator,
        rows,
        vals,
        n_touched,
        dim,
        float(lr),
        float(eps),
        parameter.stride(0),
        vals.stride(0),
        accumulator if dirty_words is None else dirty_words,
        TRACK_DIRTY=dirty_words is not None,
        BLOCK=block,
    )


def _dirty_row_kernel():
    """Lazily compile the packed dirty-bit marker used by plain Adagrad."""

    global _DIRTY_ROW_KERNEL
    if _DIRTY_ROW_KERNEL is not None:
        return _DIRTY_ROW_KERNEL

    import triton
    import triton.language as tl

    globals()["tl"] = tl

    @triton.jit
    def _mark_dirty_rows_kernel(words_ptr, rows_ptr, count, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < count
        rows = tl.load(rows_ptr + offsets, mask=mask, other=0)
        words = rows // 32
        bits = (1 << (rows % 32)).to(tl.int32)
        tl.atomic_or(words_ptr + words, bits, mask=mask)

    _DIRTY_ROW_KERNEL = _mark_dirty_rows_kernel
    return _DIRTY_ROW_KERNEL


class DirtyRowTracker:
    """Packed per-parameter dirty rows owned by this optimizer rank.

    CUDA tensors use one bit per local embedding row.  CPU tensors use a bool
    mask because that path is primarily tests/small models and avoiding a
    platform-specific atomic bit operation keeps it simple.  Rows are marked
    where the owner-routed sparse gradient is consumed; no cross-rank ID gather
    is necessary.
    """

    def __init__(self, parameters: Iterable[nn.Parameter], *, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._masks: dict[nn.Parameter, Tensor] = {}
        if not self.enabled:
            return
        for parameter in parameters:
            rows = int(parameter.shape[0])
            if parameter.is_cuda:
                mask = torch.zeros(
                    ((rows + 31) // 32,),
                    dtype=torch.int32,
                    device=parameter.device,
                )
            else:
                mask = torch.zeros((rows,), dtype=torch.bool, device=parameter.device)
            self._masks[parameter] = mask

    def packed_mask(self, parameter: nn.Parameter) -> Tensor | None:
        mask = self._masks.get(parameter)
        if mask is None or not parameter.is_cuda:
            return None
        return mask

    def mark(self, parameter: nn.Parameter, rows: Tensor) -> None:
        mask = self._masks.get(parameter)
        if mask is None or rows.numel() == 0:
            return
        if parameter.is_cuda:
            import triton

            values = rows.contiguous()
            block = 256
            _dirty_row_kernel()[(triton.cdiv(values.numel(), block),)](
                mask,
                values,
                values.numel(),
                BLOCK=block,
            )
            return
        mask.index_fill_(0, rows.to(device=mask.device, dtype=torch.long), True)

    def iter_rows(
        self,
        parameter: nn.Parameter,
        *,
        max_rows: int,
    ) -> Iterable[Tensor]:
        """Yield sorted local row IDs in bounded CPU tensors."""

        mask = self._masks.get(parameter)
        if mask is None:
            return
        limit = max(1, int(max_rows))
        if not parameter.is_cuda:
            rows = torch.nonzero(mask, as_tuple=False).flatten().to(dtype=torch.int64)
            for start in range(0, int(rows.numel()), limit):
                yield rows.narrow(0, start, min(limit, rows.numel() - start)).cpu()
            return

        # Scan a bounded number of packed words at a time.  Expanding the full
        # 500M-row mask to a bool tensor would erase the memory benefit of the
        # bitset exactly when checkpointing starts.
        bits = torch.arange(32, dtype=torch.int64, device=mask.device)
        bit_values = torch.bitwise_left_shift(
            torch.ones(32, dtype=torch.int64, device=mask.device), bits
        )
        words_per_scan = max(1, min(131072, (limit + 31) // 32))
        total_rows = int(parameter.shape[0])
        pending: Tensor | None = None
        for word_start in range(0, int(mask.numel()), words_per_scan):
            words = mask.narrow(
                0,
                word_start,
                min(words_per_scan, mask.numel() - word_start),
            )
            nonzero = torch.nonzero(words != 0, as_tuple=False).flatten()
            if nonzero.numel() == 0:
                continue
            values = words.index_select(0, nonzero).to(torch.int64) & 0xFFFFFFFF
            present = torch.bitwise_and(values[:, None], bit_values[None, :]) != 0
            expanded = ((nonzero + word_start)[:, None] * 32 + bits[None, :])[
                present
            ]
            expanded = expanded[expanded < total_rows]
            if pending is not None:
                expanded = torch.cat((pending, expanded))
                pending = None
            offset = 0
            while int(expanded.numel()) - offset >= limit:
                yield expanded.narrow(0, offset, limit).cpu()
                offset += limit
            if offset < int(expanded.numel()):
                pending = expanded.narrow(0, offset, expanded.numel() - offset)
        if pending is not None and pending.numel():
            yield pending.cpu()

    def clear(self) -> None:
        for mask in self._masks.values():
            mask.zero_()


class ShardedAdagrad(torch.optim.Optimizer):
    """Exact row-sparse Adagrad over already-local embedding parameters.

    No communication occurs here: owner-based gradient routing is completed by
    ``ShardedEmbedding`` during autograd. Consequently both the parameter and
    accumulator have only local-shard shape.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        eps: float = 1.0e-10,
        *,
        state_dtype: torch.dtype = torch.float32,
        track_dirty_rows: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if lr_decay < 0.0:
            raise ValueError("lr_decay must be non-negative")
        if weight_decay != 0.0:
            raise ValueError(
                "ShardedAdagrad does not support weight decay for sparse gradients"
            )
        if initial_accumulator_value < 0.0:
            raise ValueError("initial_accumulator_value must be non-negative")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        defaults = {
            "lr": lr,
            "lr_decay": lr_decay,
            "weight_decay": weight_decay,
            "initial_accumulator_value": initial_accumulator_value,
            "eps": eps,
            "state_dtype": state_dtype,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            for parameter in group["params"]:
                state = self.state[parameter]
                state["step"] = torch.zeros((), dtype=torch.float64)
                state["sum"] = torch.full(
                    parameter.shape,
                    float(initial_accumulator_value),
                    dtype=state_dtype,
                    device=parameter.device,
                )
        self._dirty_rows = DirtyRowTracker(
            (
                parameter
                for group in self.param_groups
                for parameter in group["params"]
            ),
            enabled=track_dirty_rows,
        )

    @property
    def tracks_dirty_rows(self) -> bool:
        return self._dirty_rows.enabled

    def iter_dirty_rows(
        self, parameter: nn.Parameter, *, max_rows: int
    ) -> Iterable[Tensor]:
        return self._dirty_rows.iter_rows(parameter, max_rows=max_rows)

    def clear_dirty_rows(self) -> None:
        self._dirty_rows.clear()

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], Tensor] | None = None,
    ) -> Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            lr_decay = float(group["lr_decay"])
            eps = float(group["eps"])
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if not grad.is_sparse or grad.layout != torch.sparse_coo:
                    raise RuntimeError(
                        "ShardedAdagrad expects one-dimensional row-sparse COO gradients"
                    )
                grad = grad.coalesce()
                if grad.sparse_dim() != 1 or grad.dense_dim() != 1:
                    raise RuntimeError(
                        "ShardedAdagrad expects one sparse row dimension and one dense dimension"
                    )
                rows = grad.indices()[0]
                values = grad.values()
                state: dict[str, Any] = self.state[parameter]
                state["step"].add_(1.0)
                step = float(state["step"].item())
                clear_lr = lr / (1.0 + (step - 1.0) * lr_decay)
                if rows.numel() == 0:
                    continue
                self._dirty_rows.mark(parameter, rows)

                accumulator: Tensor = state["sum"]
                state_values = values.to(dtype=accumulator.dtype)
                accumulator.index_add_(0, rows, state_values.square())
                denominator = accumulator.index_select(0, rows).sqrt_().add_(eps)
                update = state_values / denominator
                parameter.index_add_(
                    0,
                    rows,
                    update.to(dtype=parameter.dtype),
                    alpha=-clear_lr,
                )
        return loss


class ShardedRowWiseAdagrad(torch.optim.Optimizer):
    """Row-wise Adagrad over already-local embedding parameters.

    Each local row keeps one FP32 accumulator equal to the mean squared
    gradient across the embedding dimension. Weight tensors may be BF16;
    accumulators stay FP32. No communication occurs here.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        eps: float = 1.0e-10,
        *,
        state_dtype: torch.dtype = torch.float32,
        track_dirty_rows: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if lr_decay < 0.0:
            raise ValueError("lr_decay must be non-negative")
        if weight_decay != 0.0:
            raise ValueError(
                "ShardedRowWiseAdagrad does not support weight decay for sparse gradients"
            )
        if initial_accumulator_value < 0.0:
            raise ValueError("initial_accumulator_value must be non-negative")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        defaults = {
            "lr": lr,
            "lr_decay": lr_decay,
            "weight_decay": weight_decay,
            "initial_accumulator_value": initial_accumulator_value,
            "eps": eps,
            "state_dtype": state_dtype,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.ndim != 2:
                    raise ValueError(
                        "ShardedRowWiseAdagrad expects 2D embedding parameters"
                    )
                state = self.state[parameter]
                # Keep step on CPU so .item() never syncs CUDA across hundreds of tables.
                state["step"] = torch.zeros((), dtype=torch.float64)
                state["sum"] = torch.full(
                    (parameter.shape[0],),
                    float(initial_accumulator_value),
                    dtype=state_dtype,
                    device=parameter.device,
                )
        self._dirty_rows = DirtyRowTracker(
            (
                parameter
                for group in self.param_groups
                for parameter in group["params"]
            ),
            enabled=track_dirty_rows,
        )

    @property
    def tracks_dirty_rows(self) -> bool:
        return self._dirty_rows.enabled

    def iter_dirty_rows(
        self, parameter: nn.Parameter, *, max_rows: int
    ) -> Iterable[Tensor]:
        return self._dirty_rows.iter_rows(parameter, max_rows=max_rows)

    def clear_dirty_rows(self) -> None:
        self._dirty_rows.clear()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Deep-copy first: Optimizer.load_state_dict casts state tensors to the
        # parameter dtype (BF16), which would permanently lose FP32 accumulator
        # precision. Restore sum/step from the pre-cast snapshot afterward.
        original = copy.deepcopy(state_dict)
        super().load_state_dict(state_dict)

        saved_items: list[dict[str, Any]] = []
        for group in original.get("param_groups", ()):
            for param_id in group["params"]:
                saved_items.append(original.get("state", {}).get(param_id, {}))

        index = 0
        for group in self.param_groups:
            state_dtype = group.get("state_dtype", torch.float32)
            for parameter in group["params"]:
                saved = saved_items[index] if index < len(saved_items) else {}
                index += 1
                state = self.state.get(parameter)
                if not state:
                    continue
                accumulator = saved.get("sum", state.get("sum"))
                if isinstance(accumulator, Tensor):
                    state["sum"] = accumulator.to(
                        device=parameter.device,
                        dtype=state_dtype,
                    )
                step = saved.get("step", state.get("step"))
                if isinstance(step, Tensor):
                    state["step"] = step.to(device="cpu", dtype=torch.float64)

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], Tensor] | None = None,
    ) -> Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        gset_owners: dict[int, Any] = {}
        for group in self.param_groups:
            lr = float(group["lr"])
            lr_decay = float(group["lr_decay"])
            eps = float(group["eps"])
            for parameter in group["params"]:
                owner = gset_owner_for_parameter(parameter)
                if owner is not None:
                    gset_owners[id(owner)] = owner
                    reset_rows = owner.consume_pending_optimizer_reset_rows()
                    if reset_rows.numel():
                        accumulator = self.state[parameter]["sum"]
                        accumulator.index_fill_(
                            0,
                            reset_rows.to(
                                device=accumulator.device,
                                dtype=torch.long,
                            ),
                            float(group["initial_accumulator_value"]),
                        )
                grad = parameter.grad
                if grad is None:
                    continue
                if not grad.is_sparse or grad.layout != torch.sparse_coo:
                    raise RuntimeError(
                        "ShardedRowWiseAdagrad expects one-dimensional row-sparse COO gradients"
                    )
                # Sparse embedding grads are usually already coalesced by the
                # embedding backward; skip a full rewrite when possible.
                if not grad.is_coalesced():
                    grad = grad.coalesce()
                if grad.sparse_dim() != 1 or grad.dense_dim() != 1:
                    raise RuntimeError(
                        "ShardedRowWiseAdagrad expects one sparse row dimension "
                        "and one dense dimension"
                    )
                rows = grad.indices()[0]
                values = grad.values()
                state: dict[str, Any] = self.state[parameter]
                # step lives on CPU; keep a python float to avoid .item() per table
                # across hundreds of embedding parameters each iteration.
                step_tensor = state["step"]
                step = float(step_tensor) + 1.0
                step_tensor.fill_(step)
                if lr_decay == 0.0:
                    clear_lr = lr
                else:
                    clear_lr = lr / (1.0 + (step - 1.0) * lr_decay)
                if rows.numel() == 0:
                    continue

                accumulator: Tensor = state["sum"]
                # Fused Triton path collapses ~4 tiny launches/table into one.
                # Coalesced grads guarantee unique rows (no accumulator races).
                if parameter.is_cuda and accumulator.is_cuda:
                    fused_rowwise_adagrad_update(
                        parameter,
                        accumulator,
                        rows,
                        values,
                        lr=clear_lr,
                        eps=eps,
                        dirty_words=self._dirty_rows.packed_mask(parameter),
                    )
                    continue

                self._dirty_rows.mark(parameter, rows)

                if values.dtype == torch.float32:
                    state_values = values
                else:
                    state_values = values.float()
                row_squared_mean = state_values.square().mean(dim=1)
                accumulator.index_add_(0, rows, row_squared_mean)
                denominator = (
                    accumulator.index_select(0, rows).sqrt_().add_(eps).unsqueeze(1)
                )
                update = state_values / denominator
                if update.dtype != parameter.dtype:
                    update = update.to(dtype=parameter.dtype)
                parameter.index_add_(
                    0,
                    rows,
                    update,
                    alpha=-clear_lr,
                )
        for owner in gset_owners.values():
            owner.optimizer_step_completed()
        return loss
