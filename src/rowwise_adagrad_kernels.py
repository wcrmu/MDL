"""Triton kernels for fused row-wise Adagrad embedding updates."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


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
    BLOCK: tl.constexpr,
):
    """One program per touched row: update accumulator and embedding row."""

    pid = tl.program_id(0)
    if pid >= n_touched:
        return
    row = tl.load(rows_ptr + pid)
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


def fused_rowwise_adagrad_update(
    parameter: torch.Tensor,
    accumulator: torch.Tensor,
    rows: torch.Tensor,
    values: torch.Tensor,
    *,
    lr: float,
    eps: float,
) -> None:
    """Fused row-wise Adagrad update for one embedding table.

    ``rows`` must be unique (coalesced sparse grads). ``values`` may be BF16 or
    FP32; accumulators stay FP32. Parameter may be BF16 or FP32.
    """

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
    _rowwise_adagrad_update_kernel[(n_touched,)](
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
        BLOCK=block,
    )
