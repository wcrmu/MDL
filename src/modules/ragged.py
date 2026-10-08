"""Flat variable-length tokens for one batch.

Every real token in the batch lives in one ``[total, dim]`` array. ``batch_lengths``
is the true token count of each row, ``batch_lengths_added`` is the prefix sum
(``cu_seqlens``), and ``role`` is the per-token role id. Roles are packed
contiguously inside each row, in role order, so one role is itself a varlen
batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .attention import _call_varlen_attention, _sdpa_context, varlen_attention_available
from .ranking_utils import RequestLayout


@dataclass
class RaggedTokens:
    """Real tokens only. Padding is not stored."""

    values: Tensor
    batch_lengths: Tensor
    batch_lengths_added: Tensor
    role: Tensor
    role_lengths: Tensor

    def role_values(self, role_id: int) -> tuple[Tensor, Tensor, int]:
        """Return one role as ``(values, cu_seqlens, max_length)``."""

        lengths = self.role_lengths[:, role_id]
        cumulative = F.pad(lengths.cumsum(0), (1, 0)).to(dtype=torch.int32)
        selected = self.values[self.role == role_id]
        longest = int(lengths.max().item()) if lengths.numel() else 0
        return selected, cumulative, longest

    def index_rows(self, indices: Tensor) -> RaggedTokens:
        """Gather rows, repeating a request when several candidates share it."""

        indices = indices.to(device=self.values.device, dtype=torch.long)
        if indices.ndim != 1:
            raise ValueError("row indices must be a vector")
        lengths = self.batch_lengths.index_select(0, indices)
        added = F.pad(lengths.cumsum(0), (1, 0)).to(dtype=torch.int32)
        total = int(added[-1].item()) if indices.numel() else 0
        if total == 0:
            values = self.values.new_zeros(0, self.values.size(-1)) + self.values.sum() * 0
            role = self.role.new_empty(0)
        else:
            out_rows = torch.repeat_interleave(
                torch.arange(indices.numel(), device=indices.device),
                lengths.to(dtype=torch.long),
            )
            local = torch.arange(total, device=indices.device) - added[out_rows].to(
                dtype=torch.long
            )
            source = (
                self.batch_lengths_added.index_select(0, indices)[out_rows].to(dtype=torch.long)
                + local
            )
            values = self.values.index_select(0, source)
            role = self.role.index_select(0, source)
        return RaggedTokens(
            values,
            lengths,
            added,
            role,
            self.role_lengths.index_select(0, indices),
        )

    def repeat_rows(self, count: int) -> RaggedTokens:
        """Repeat the single stored row ``count`` times."""

        if self.batch_lengths.shape[0] != 1:
            raise ValueError("repeat_rows requires exactly one source row")
        if count < 0:
            raise ValueError("repeat count must be non-negative")
        indices = torch.zeros(count, dtype=torch.long, device=self.values.device)
        return self.index_rows(indices)


def pack_role_streams(values: list[Tensor], lengths: list[Tensor]) -> RaggedTokens:
    """Pack per-role flat tokens into one row-major ragged batch.

    ``values[role]`` is ``[count, dim]`` in row order. ``lengths[role]`` is
    ``[rows]``. Within a row, roles occupy contiguous spans in role order.
    """

    if not values or len(values) != len(lengths):
        raise ValueError("each role needs one value tensor and one length vector")
    rows = int(lengths[0].shape[0])
    dim = int(values[0].size(-1))
    device = values[0].device
    dtype = values[0].dtype
    role_lengths = torch.stack(
        [item.to(device=device, dtype=torch.int32).reshape(rows) for item in lengths],
        dim=1,
    )
    for index, tensor in enumerate(values):
        if tensor.ndim != 2 or tensor.size(-1) != dim:
            raise ValueError(f"role {index} values must have shape [count, {dim}]")
        if int(role_lengths[:, index].sum().item()) != tensor.size(0):
            raise ValueError(f"role {index} lengths do not add up to its token count")
    batch_lengths = role_lengths.sum(dim=1).to(dtype=torch.int32)
    batch_lengths_added = F.pad(batch_lengths.cumsum(0), (1, 0)).to(dtype=torch.int32)
    total = int(batch_lengths_added[-1].item()) if rows else 0
    packed = values[0].new_zeros(total, dim)
    role = torch.empty(total, dtype=torch.int32, device=device)
    role_offset = F.pad(role_lengths.cumsum(dim=1), (1, 0))
    anchor = packed.new_zeros(())
    for index, tensor in enumerate(values):
        count = tensor.size(0)
        if count == 0:
            # Keep an unused stream's projector in the autograd graph.
            anchor = anchor + tensor.sum() * 0
            continue
        cumulative = F.pad(role_lengths[:, index].cumsum(0), (1, 0))
        row_ids = torch.repeat_interleave(
            torch.arange(rows, device=device),
            role_lengths[:, index].to(dtype=torch.long),
        )
        local = torch.arange(count, device=device) - cumulative[row_ids].to(dtype=torch.long)
        destination = (
            batch_lengths_added[row_ids].to(dtype=torch.long)
            + role_offset[row_ids, index].to(dtype=torch.long)
            + local
        )
        packed[destination] = tensor.to(dtype=dtype)
        role[destination] = index
    if total == 0:
        role = torch.empty(0, dtype=torch.int32, device=device)
    return RaggedTokens(
        packed + anchor,
        batch_lengths,
        batch_lengths_added,
        role,
        role_lengths,
    )


def pack_masked_sequences(
    sequences: list[Tensor],
    masks: list[Tensor | None],
) -> RaggedTokens:
    """Drop padded positions and pack each sequence as its own role."""

    if not sequences:
        raise ValueError("at least one sequence is required")
    flats: list[Tensor] = []
    lengths: list[Tensor] = []
    for index, sequence in enumerate(sequences):
        if sequence.ndim != 3:
            raise ValueError(f"sequence {index} must have shape [rows, length, dim]")
        mask = masks[index]
        if mask is None:
            mask = torch.ones(
                sequence.shape[:2],
                dtype=torch.bool,
                device=sequence.device,
            )
        elif mask.shape != sequence.shape[:2] or mask.dtype != torch.bool:
            raise ValueError(f"sequence {index} mask must be bool [rows, length]")
        flats.append(sequence[mask])
        lengths.append(mask.sum(dim=1).to(dtype=torch.int32))
    return pack_role_streams(flats, lengths)


def pack_valid_rows(
    tokens: Tensor,
    mask: Tensor,
    role: Tensor | None = None,
    role_count: int | None = None,
) -> RaggedTokens:
    """Pack ``[rows, length, dim]`` in left-to-right valid order.

    ``role`` is an int id per position. Roles may be mixed inside a row;
    ``role_lengths`` still counts each id. Invalid positions are dropped and
    do not receive a role.
    """

    if tokens.ndim != 3 or mask.shape != tokens.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("tokens must be [rows, length, dim] with a matching bool mask")
    rows, _length, dim = tokens.shape
    selected = tokens[mask]
    if role is None:
        resolved_count = 1
        flat_role = torch.zeros(selected.size(0), dtype=torch.int32, device=tokens.device)
    else:
        if role.shape != mask.shape:
            raise ValueError("role must have shape [rows, length]")
        if role_count is None or role_count <= 0:
            raise ValueError("role_count must be positive when role ids are set")
        resolved_count = role_count
        flat_role = role[mask].to(dtype=torch.int32)
    batch_lengths = mask.sum(dim=1).to(dtype=torch.int32)
    role_lengths = torch.zeros(
        rows,
        resolved_count,
        dtype=torch.int32,
        device=tokens.device,
    )
    if flat_role.numel():
        row_ids = torch.repeat_interleave(
            torch.arange(rows, device=tokens.device),
            batch_lengths.to(dtype=torch.long),
        )
        role_lengths.index_put_(
            (row_ids, flat_role.to(dtype=torch.long)),
            torch.ones_like(flat_role),
            accumulate=True,
        )
    added = F.pad(batch_lengths.cumsum(0), (1, 0)).to(dtype=torch.int32)
    if selected.numel() == 0:
        selected = tokens.new_zeros(0, dim) + tokens.sum() * 0
        flat_role = torch.empty(0, dtype=torch.int32, device=tokens.device)
    return RaggedTokens(selected, batch_lengths, added, flat_role, role_lengths)


def apply_masked_tokenwise(module, tokens: Tensor, mask: Tensor | None) -> Tensor:
    """Run ``module`` on real tokens. Padding stays zero and never enters it.

    An empty mask still runs one dummy token so an unused module stays in the
    autograd graph.
    """

    if tokens.ndim < 2:
        raise ValueError("tokenwise modules expect a trailing feature dimension")
    if mask is None:
        return module(tokens)
    if mask.shape != tokens.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("mask must be bool and match the leading token shape")
    selected = tokens[mask]
    if selected.numel() == 0:
        dummy = module(tokens.new_zeros(1, tokens.size(-1)))
        output = dummy.new_zeros(*tokens.shape[:-1], dummy.size(-1))
        return output + dummy.sum() * 0
    updated = module(selected)
    output = updated.new_zeros(*tokens.shape[:-1], updated.size(-1))
    output[mask] = updated
    return output


def read_ragged_role(
    query: Tensor,
    memory: RaggedTokens,
    role_id: int,
    heads: int,
    backend: str = "auto",
    layout: RequestLayout | None = None,
) -> Tensor:
    """Varlen attention from queries onto one role of a ragged memory."""

    if backend not in {"auto", "sdpa", "flash"}:
        raise ValueError("attention backend must be auto, sdpa, or flash")
    if query.ndim != 3:
        raise ValueError(f"query must have shape [batch, tokens, dim], got {tuple(query.shape)}")
    batch, query_length, dim = query.shape
    if heads <= 0 or dim % heads:
        raise ValueError("query dim must be a positive multiple of heads")
    requests = int(memory.role_lengths.shape[0])
    if layout is None and batch != requests:
        raise ValueError("query batch must match ragged rows unless a request layout is set")
    if layout is not None and layout.index.numel() != batch:
        raise ValueError("one request index is required for each query row")
    if not 0 <= role_id < memory.role_lengths.shape[1]:
        raise ValueError(f"role {role_id} is outside the ragged batch")
    values, cumulative, longest = memory.role_values(role_id)
    if values.numel() and values.size(-1) != dim:
        raise ValueError(f"role values must end in dim {dim}")
    if values.numel() == 0 or batch == 0 or query_length == 0:
        return query * 0 + values.sum() * 0
    use_flash = (
        backend != "sdpa"
        and query.is_cuda
        and query.dtype in {torch.float16, torch.bfloat16}
        and varlen_attention_available()
    )
    if backend == "flash" and not use_flash:
        raise RuntimeError("strict flash requires CUDA FP16/BF16 and flash-attn varlen")
    if use_flash:
        ordered = query if layout is None else query.index_select(0, layout.order)
        flat_query = ordered.reshape(-1, heads, dim // heads)
        flat_key = values.to(dtype=query.dtype).reshape(-1, heads, dim // heads)
        if layout is None:
            query_cumulative = torch.arange(
                requests + 1,
                device=query.device,
                dtype=torch.int32,
            ) * query_length
            max_query = query_length
        else:
            query_cumulative = F.pad(
                (layout.counts * query_length).cumsum(0, dtype=torch.int32),
                (1, 0),
            )
            max_query = layout.max_candidates * query_length
        output = _call_varlen_attention(
            flat_query.contiguous(),
            flat_key.contiguous(),
            flat_key.contiguous(),
            query_cumulative,
            cumulative,
            max_query,
            longest,
            causal=False,
            fixed_capacity=False,
        )
        output = output.reshape(batch, query_length, dim)
        return output if layout is None else output.index_select(0, layout.inverse)

    def sdpa(q: Tensor, key: Tensor, key_mask: Tensor | None = None) -> Tensor:
        if key.size(1) == 0:
            return q * 0 + key.sum() * 0

        def split(token: Tensor) -> Tensor:
            return token.reshape(token.size(0), token.size(1), heads, dim // heads).transpose(1, 2)

        key = key.to(dtype=q.dtype)
        allowed = None if key_mask is None else key_mask[:, None, None, :]
        with _sdpa_context(backend):
            context = F.scaled_dot_product_attention(
                split(q),
                split(key),
                split(key),
                attn_mask=allowed,
            )
        return context.transpose(1, 2).reshape_as(q)

    lengths = memory.role_lengths[:, role_id]
    if layout is None:
        # One SDPA on the batch-max width. The stored memory stays flat; this
        # pad exists only for the portable kernel and is the longest real row.
        padded = values.new_zeros(requests, longest, dim)
        if values.numel():
            row_ids = torch.repeat_interleave(
                torch.arange(requests, device=values.device),
                lengths.to(dtype=torch.long),
            )
            local = torch.arange(values.size(0), device=values.device) - cumulative[row_ids].to(
                dtype=torch.long
            )
            padded[row_ids, local] = values
        positions = torch.arange(longest, device=values.device)
        key_mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
        present = lengths > 0
        if longest:
            key_mask = key_mask.clone()
            key_mask[:, 0] |= ~present
        context = sdpa(query, padded, key_mask)
        return context * present.to(dtype=context.dtype)[:, None, None] + values.sum() * 0
    outputs = []
    ordered = query.index_select(0, layout.order)
    offset = 0
    for row, count in enumerate(layout.counts.tolist()):
        start = int(cumulative[row].item())
        stop = int(cumulative[row + 1].item())
        key = values[start:stop].reshape(1, stop - start, dim)
        if count:
            grouped = ordered[offset : offset + count].reshape(1, count * query_length, dim)
            outputs.append(sdpa(grouped, key).reshape(count, query_length, dim))
        offset += count
    if not outputs:
        return query * 0 + values.sum() * 0
    return torch.cat(outputs, dim=0).index_select(0, layout.inverse)
