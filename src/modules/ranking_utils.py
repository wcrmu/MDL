"""Shared execution primitives for UniFormer and MORE (not shared parameters)."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .attention import _call_varlen_attention, _sdpa_context, varlen_attention_available
from .stca import SwiGLUFFN


class RankingRMSNorm(nn.RMSNorm):
    """Keep autocast activations and affine weights in the same compute dtype."""

    def forward(self, values: Tensor) -> Tensor:
        weight = None if self.weight is None else self.weight.to(values.dtype)
        return F.rms_norm(values, self.normalized_shape, weight, self.eps)


def token_swiglu(ffns: nn.ModuleList, tokens: Tensor) -> Tensor:
    """Independent SwiGLUs as three batched GEMMs; preserve state_dict names."""
    if tokens.ndim != 3 or tokens.size(1) != len(ffns):
        raise ValueError(f"expected {len(ffns)} tokens, got {tuple(tokens.shape)}")
    if not ffns:
        return tokens
    if any(not isinstance(f, SwiGLUFFN) for f in ffns):
        raise TypeError("token_swiglu requires SwiGLUFFN modules")

    def project(x: Tensor, name: str) -> Tensor:
        layers = [getattr(f, name) for f in ffns]
        weight = torch.stack([layer.weight for layer in layers])
        y = torch.bmm(x, weight.transpose(1, 2))
        if layers[0].bias is not None:
            y = y + torch.stack([layer.bias for layer in layers])[:, None].to(y.dtype)
        return y

    x = tokens.transpose(0, 1)
    hidden = project(x, "up_projection") * F.silu(project(x, "gate_projection"))
    return project(hidden, "output_projection").transpose(0, 1)


@dataclass
class RequestLayout:
    """Small candidate permutation; long sequence tensors remain request-major."""

    index: Tensor
    order: Tensor
    inverse: Tensor
    counts: Tensor
    max_candidates: int
    first: Tensor

    @classmethod
    def build(cls, index: Tensor, requests: int) -> "RequestLayout":
        if index.ndim != 1 or index.dtype != torch.long:
            raise ValueError("request_index must be a one-dimensional int64 tensor")
        valid = ((index >= 0) & (index < requests)).all()
        if index.is_cuda:
            torch._assert_async(valid, "request_index out of bounds")
        elif not bool(valid):
            raise ValueError("request_index out of bounds")
        order = index.argsort(stable=True)
        counts = torch.bincount(index, minlength=requests).to(torch.int32)
        first = torch.full((requests,), index.numel(), device=index.device, dtype=torch.long)
        first.scatter_reduce_(0, index, torch.arange(index.numel(), device=index.device),
                              reduce="amin", include_self=True)
        first = first.clamp_max(max(index.numel() - 1, 0))
        return cls(index, order, order.argsort(), counts,
                   int(counts.max().item()) if requests else 0, first)


@dataclass
class SequenceMemory:
    """Per-forward memory and compact KV views reused across attention readers."""

    key: Tensor
    value: Tensor
    mask: Tensor | None = None
    _packed: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if self.key.ndim != 3 or self.key.shape != self.value.shape:
            raise ValueError("sequence K/V must have matching [request,length,dim] shapes")
        if self.mask is not None and (self.mask.shape != self.key.shape[:2]
                                      or self.mask.dtype != torch.bool):
            raise ValueError("sequence mask must be boolean [request,length]")

    def packed(self, heads: int, dtype: torch.dtype):
        cache_key = (heads, dtype)
        if cache_key not in self._packed:
            r, length, dim = self.key.shape
            k = self.key.to(dtype).reshape(r * length, heads, dim // heads)
            v = k if self.value is self.key else self.value.to(dtype).reshape_as(k)
            if self.mask is None:
                counts = torch.full((r,), length, device=k.device, dtype=torch.int32)
            else:
                positions = self.mask.flatten().nonzero().flatten()
                k = k.index_select(0, positions)
                v = k if self.value is self.key else v.index_select(0, positions)
                counts = self.mask.sum(1, dtype=torch.int32)
            cu = F.pad(counts.cumsum(0, dtype=torch.int32), (1, 0))
            self._packed[cache_key] = (k.contiguous(), v.contiguous(), cu)
        return self._packed[cache_key]


def read_sequence(query: Tensor, memory: SequenceMemory, heads: int,
                  backend: str = "auto", layout: RequestLayout | None = None) -> Tensor:
    """Read shared KV without materializing [candidates,history,dim]."""
    if backend not in {"auto", "sdpa", "flash"}:
        raise ValueError("attention backend must be auto, sdpa, or flash")
    b, qlen, dim = query.shape
    r, length, _ = memory.key.shape
    if heads <= 0 or dim % heads or memory.key.size(-1) != dim or (layout is None and b != r):
        raise ValueError("attention dimensions or request mapping do not match")
    if layout is not None and layout.index.numel() != b:
        raise ValueError("one request index is required for each candidate")
    use_flash = (backend != "sdpa" and query.is_cuda
                 and query.dtype in {torch.float16, torch.bfloat16}
                 and varlen_attention_available())
    if backend == "flash" and not use_flash:
        raise RuntimeError("strict flash requires CUDA FP16/BF16 and flash-attn varlen")
    if length == 0 or b == 0 or qlen == 0:
        return query * 0 + (memory.key.sum() + memory.value.sum()) * 0
    if use_flash:
        q = query if layout is None else query.index_select(0, layout.order)
        q = q.reshape(-1, heads, dim // heads)
        k, v, cu_k = memory.packed(heads, query.dtype)
        if layout is None:
            cu_q = torch.arange(r + 1, device=q.device, dtype=torch.int32) * qlen
            max_q = qlen
        else:
            cu_q = F.pad((layout.counts * qlen).cumsum(0, dtype=torch.int32), (1, 0))
            max_q = layout.max_candidates * qlen
        if k.size(0) == 0:
            return query * 0 + (memory.key.sum() + memory.value.sum()) * 0
        output = _call_varlen_attention(q.contiguous(), k, v, cu_q, cu_k,
                                       max_q, length, causal=False, fixed_capacity=False)
        output = output.reshape(b, qlen, dim)
        return output if layout is None else output.index_select(0, layout.inverse)

    def sdpa(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None) -> Tensor:
        def split(x):
            return x.reshape(x.size(0), x.size(1), heads, dim // heads).transpose(1, 2)
        # Empty histories must have zero output/gradient, including older SDPA backends.
        keep = None if mask is None else mask.any(-1)
        if mask is not None:
            mask = mask.clone()
            mask[:, 0] |= ~keep
        out = F.scaled_dot_product_attention(split(q), split(k.to(q.dtype)), split(v.to(q.dtype)),
                    attn_mask=None if mask is None else mask[:, None, None])
        out = out.transpose(1, 2).reshape_as(q)
        return out if keep is None else out * keep[:, None, None].to(out.dtype)

    if layout is None:
        return sdpa(query, memory.key, memory.value, memory.mask)
    # Portable reference path; only Q is grouped, never long K/V.
    outputs = []
    sorted_q = query.index_select(0, layout.order)
    offset = 0
    for row, count in enumerate(layout.counts.tolist()):
        if count:
            q = sorted_q[offset:offset + count].reshape(1, count * qlen, dim)
            m = None if memory.mask is None else memory.mask[row:row + 1]
            outputs.append(sdpa(q, memory.key[row:row + 1], memory.value[row:row + 1], m)
                           .reshape(count, qlen, dim))
        offset += count
    return torch.cat(outputs).index_select(0, layout.inverse)


class ProjectedCrossAttention(nn.Module):
    """Query/output projections with already projected, request-shared K/V."""

    def __init__(self, d_model: int, num_heads: int, attention_backend: str = "auto"):
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads:
            raise ValueError("d_model must be a positive multiple of num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attention_backend = attention_backend
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query: Tensor, key: Tensor | SequenceMemory,
                value: Tensor | None = None, key_mask: Tensor | None = None,
                layout: RequestLayout | None = None) -> Tensor:
        if not isinstance(key, SequenceMemory) and value is None:
            raise ValueError("value is required when key is a tensor")
        memory = key if isinstance(key, SequenceMemory) else SequenceMemory(key, value, key_mask)
        return self.out_proj(read_sequence(self.q_proj(query), memory, self.num_heads,
                                           self.attention_backend, layout))
