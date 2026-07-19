"""Optimizers whose state follows the repository's local embedding shards."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable
import copy

import torch
from torch import Tensor, nn


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
                        "ShardedRowWiseAdagrad expects one-dimensional row-sparse COO gradients"
                    )
                grad = grad.coalesce()
                if grad.sparse_dim() != 1 or grad.dense_dim() != 1:
                    raise RuntimeError(
                        "ShardedRowWiseAdagrad expects one sparse row dimension "
                        "and one dense dimension"
                    )
                rows = grad.indices()[0]
                values = grad.values()
                state: dict[str, Any] = self.state[parameter]
                state["step"].add_(1.0)
                step = float(state["step"].item())
                clear_lr = lr / (1.0 + (step - 1.0) * lr_decay)
                if rows.numel() == 0:
                    continue

                accumulator: Tensor = state["sum"]
                state_values = values.float()
                row_squared_mean = state_values.square().mean(dim=1)
                accumulator.index_add_(0, rows, row_squared_mean)
                denominator = (
                    accumulator.index_select(0, rows).sqrt_().add_(eps).unsqueeze(1)
                )
                update = state_values / denominator
                parameter.index_add_(
                    0,
                    rows,
                    update.to(dtype=parameter.dtype),
                    alpha=-clear_lr,
                )
        return loss
