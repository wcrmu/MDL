"""Direct agg Arrow → FeatureBatch producer.

``RequestGroupBlock`` holds only axis descriptors before shuffle/bucket/pack.
After packing, ``PreparedAxisBatch`` and ``SequenceSelectionPlan`` feed the
Arrow-free FeatureBatch tensorizer. Controlled by ``reader.agg_direct_mode``
(default ``legacy``).

Historical ranking logs do not place the same ``request_id`` in two scanned
tables at once. Pack-time request plans are therefore identity maps over
unique-per-pack blocks (table-internal multi-candidate sharing stays inside
one block).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class RequestGroupBlock:
    """One request group as a logical view over a shared raw Arrow table.

    Grouping key is ``split.request_id`` (e.g. ``search_id``), not a raw
    ``context_indices`` position. ``representative_request_position`` records
    the first-occurrence request payload source for legacy dedup parity.
    """

    source_id: int
    raw_row_index: int
    request_id: Any
    representative_request_position: int
    candidate_positions: Any
    candidate_offset: int
    candidate_count: int
    pre_compaction_sequence_lengths: Mapping[str, int]
    effective_bucket_length: int
    stable_group_order: int
    slice_ordinal: int = 0

    def slice_candidates(self, offset: int, length: int) -> "RequestGroupBlock":
        """Return a descriptor view; does not take Arrow payload or allocate tensors."""

        if offset < 0 or length < 0:
            raise ValueError("slice_candidates offset/length must be non-negative")
        if offset + length > self.candidate_count:
            raise ValueError(
                f"slice_candidates[{offset}:{offset + length}] exceeds "
                f"candidate_count={self.candidate_count}"
            )
        return replace(
            self,
            candidate_offset=self.candidate_offset + offset,
            candidate_count=length,
            slice_ordinal=self.slice_ordinal + 1,
        )

    def active_candidate_positions(self) -> Any:
        start = self.candidate_offset
        stop = self.candidate_offset + self.candidate_count
        return self.candidate_positions[start:stop]

    @property
    def releases_source_reference(self) -> bool:
        """Whether consuming this descriptor finishes its original group.

        The registry owns one reference per *original* request group.  An
        oversized group may be emitted as several descriptor-only slices, so
        only the slice covering the tail of ``candidate_positions`` is allowed
        to release that reference.
        """

        return (
            self.candidate_offset + self.candidate_count
            == len(self.candidate_positions)
        )


@dataclass(frozen=True)
class PackedRequestPlan:
    """Final-batch request-axis plan after pack.

    One packed block → one request (identity). Multi-candidate rows that share
    a request stay inside one :class:`RequestGroupBlock`.
    """

    blocks: tuple[RequestGroupBlock, ...]
    unique_block_indices: Any
    block_to_request: Any
    candidate_to_request: Any


@dataclass(frozen=True)
class SequenceSelectionPlan:
    """Pack-time sequence selection for one UPS.

    Order is fixed: truncation window first (pre_compaction), then null_anchor
    compaction (compacted). Bucket keys must use pre_compaction lengths only.
    """

    sequence_name: str
    # Per-request local indices into the representative row list after compaction.
    selections: tuple[Any, ...]
    pre_compaction_lengths: Any
    compacted_lengths: Any
    token_to_request: Any


@dataclass(frozen=True)
class PreparedAxisBatch:
    """Pack-boundary, Arrow-free payload for direct FeatureBatch construction.

    Values remain separated on request and candidate axes. Sequence plans are
    built only after shuffle/bucket/pack and are shared by every aligned field
    of the same sequence.
    """

    request_values: Mapping[str, tuple[Any, ...]]
    candidate_values: Mapping[str, tuple[Any, ...]]
    request_row_indices: Any
    sequence_plans: Mapping[str, SequenceSelectionPlan]
    n_requests: int
    n_candidates: int

    @property
    def num_rows(self) -> int:
        return self.n_candidates


def _truncation_window(
    length: int,
    max_length: int | None,
    truncation: str,
) -> tuple[int, int]:
    if max_length is None or length <= max_length:
        return 0, length
    if truncation == "tail":
        return length - max_length, length
    if truncation == "head":
        return 0, max_length
    raise ValueError(f"unsupported sequence truncation {truncation!r}")


def row_sequence_selection_after_truncate_then_compact(
    *,
    list_length: int,
    anchor_is_null: np.ndarray | None,
    max_length: int | None,
    truncation: str,
) -> tuple[np.ndarray, int, int]:
    """Return (kept_local_indices, pre_compaction_length, compacted_length).

    Matches the pack-time contract: clamp/truncate first, then drop null-anchor
    steps. ``pre_compaction_length`` is the bucket input; ``compacted_length``
    is the final FeatureBatch sequence length.
    """

    if list_length < 0:
        raise ValueError("list_length must be non-negative")
    start, end = _truncation_window(list_length, max_length, truncation)
    pre_compaction = end - start
    window = np.arange(start, end, dtype=np.int64)
    if anchor_is_null is None:
        return window, pre_compaction, pre_compaction
    if anchor_is_null.shape[0] != list_length:
        raise ValueError(
            f"anchor_is_null length {anchor_is_null.shape[0]} != list_length {list_length}"
        )
    kept = window[~anchor_is_null[window]]
    return kept.astype(np.int64, copy=False), pre_compaction, int(kept.size)


def _list_value_is_null_flags(array: Any, row_index: int) -> np.ndarray:
    """Boolean null flags for one list row's values (length = list length)."""

    pa, pc = _require_pyarrow()
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    offsets = array.offsets
    start = int(offsets[row_index].as_py())
    stop = int(offsets[row_index + 1].as_py())
    if stop <= start:
        return np.asarray([], dtype=bool)
    flat = array.values.slice(start, stop - start)
    return np.asarray(pc.is_null(flat).to_numpy(zero_copy_only=False), dtype=bool)


def _list_row_length_and_array(column: Any, row_index: int) -> tuple[int, Any]:
    """Return (list_length, combined list array) for one row."""

    pa, pc = _require_pyarrow()
    array = column.combine_chunks() if hasattr(column, "combine_chunks") else column
    if pa.types.is_dictionary(array.type):
        array = pc.take(array.dictionary, array.indices)
    length = int(pc.list_value_length(array[row_index : row_index + 1])[0].as_py() or 0)
    return length, array


def build_sequence_selection_plan(
    sequence: Any,
    *,
    packed: PackedRequestPlan,
    source_tables: Mapping[int, Any],
) -> SequenceSelectionPlan:
    """Build truncate-then-compact selections for unique requests in a pack.

    Reads sequence lists from each block's representative row on its source
    table (adapted candidate-flat transitional layout).
    """

    if not sequence.fields:
        empty = np.asarray([], dtype=np.int64)
        return SequenceSelectionPlan(
            sequence_name=sequence.name,
            selections=tuple(),
            pre_compaction_lengths=empty,
            compacted_lengths=empty,
            token_to_request=empty,
        )

    anchor_field = getattr(sequence, "null_anchor_field", None)
    anchor_source = None
    if anchor_field is not None:
        for field in sequence.fields:
            if field.name == anchor_field:
                anchor_source = field.source
                break
        if anchor_source is None:
            raise ValueError(
                f"sequence {sequence.name!r} null_anchor_field {anchor_field!r} "
                "is not one of its fields"
            )

    primary_source = sequence.fields[0].source
    selections: list[np.ndarray] = []
    pre_lengths: list[int] = []
    compacted: list[int] = []
    token_to_request: list[int] = []

    for request_index, block_index in enumerate(packed.unique_block_indices):
        block = packed.blocks[int(block_index)]
        table = source_tables[block.source_id]
        row = int(block.representative_request_position)
        list_length, _primary = _list_row_length_and_array(table[primary_source], row)

        anchor_flags = None
        if anchor_source is not None:
            _length, anchor_array = _list_row_length_and_array(table[anchor_source], row)
            anchor_flags = _list_value_is_null_flags(anchor_array, row)

        kept, pre_len, compact_len = row_sequence_selection_after_truncate_then_compact(
            list_length=list_length,
            anchor_is_null=anchor_flags,
            max_length=sequence.max_length,
            truncation=sequence.truncation,
        )
        selections.append(kept)
        pre_lengths.append(pre_len)
        compacted.append(compact_len)
        token_to_request.extend([request_index] * compact_len)

    return SequenceSelectionPlan(
        sequence_name=sequence.name,
        selections=tuple(selections),
        pre_compaction_lengths=np.asarray(pre_lengths, dtype=np.int64),
        compacted_lengths=np.asarray(compacted, dtype=np.int64),
        token_to_request=np.asarray(token_to_request, dtype=np.int64),
    )


def build_axis_sequence_selection_plan(
    sequence: Any,
    *,
    packed: PackedRequestPlan,
    bundles: Mapping[int, "AdaptedAxisBundle"],
) -> SequenceSelectionPlan:
    """Build one truncate-then-compact plan over axis-separated payloads.

    The adapter may already apply its configured UPS limit. Reapplying the
    sequence window is idempotent and makes this boundary correct for adapters
    that return the full membership selection.
    """

    if not sequence.fields:
        empty = np.asarray([], dtype=np.int64)
        return SequenceSelectionPlan(
            sequence_name=sequence.name,
            selections=tuple(),
            pre_compaction_lengths=empty,
            compacted_lengths=empty,
            token_to_request=empty,
        )

    anchor_source = None
    if sequence.null_anchor_field is not None:
        for field in sequence.fields:
            if field.name == sequence.null_anchor_field:
                anchor_source = field.source
                break
        if anchor_source is None:
            raise ValueError(
                f"sequence {sequence.name!r} null_anchor_field "
                f"{sequence.null_anchor_field!r} is not one of its fields"
            )

    primary_source = sequence.fields[0].source
    selections: list[np.ndarray] = []
    pre_lengths: list[int] = []
    compacted_lengths: list[int] = []
    token_to_request: list[int] = []

    for request_index, block_index in enumerate(packed.unique_block_indices):
        block = packed.blocks[int(block_index)]
        bundle = bundles[block.source_id]
        request_slot = int(block.representative_request_position)
        try:
            primary_row = bundle.sequence_features[primary_source][request_slot]
        except KeyError as error:
            raise ValueError(
                f"sequence source {primary_source!r} missing from axis bundle"
            ) from error
        list_length = 0 if primary_row is None else len(primary_row)

        for field in sequence.fields[1:]:
            try:
                aligned_row = bundle.sequence_features[field.source][request_slot]
            except KeyError as error:
                raise ValueError(
                    f"sequence source {field.source!r} missing from axis bundle"
                ) from error
            aligned_length = 0 if aligned_row is None else len(aligned_row)
            if aligned_length != list_length:
                raise ValueError(
                    f"sequence {sequence.name!r} field {field.name!r} has length "
                    f"{aligned_length}, expected {list_length} for request "
                    f"{block.request_id!r}"
                )

        anchor_is_null = None
        if anchor_source is not None:
            anchor_row = bundle.sequence_features[anchor_source][request_slot]
            anchor_values = () if anchor_row is None else anchor_row
            anchor_is_null = np.fromiter(
                (value is None for value in anchor_values),
                dtype=bool,
                count=list_length,
            )

        kept, pre_length, compacted_length = (
            row_sequence_selection_after_truncate_then_compact(
                list_length=list_length,
                anchor_is_null=anchor_is_null,
                max_length=sequence.max_length,
                truncation=sequence.truncation,
            )
        )
        expected_pre_length = block.pre_compaction_sequence_lengths.get(
            sequence.name
        )
        if (
            expected_pre_length is not None
            and int(expected_pre_length) != pre_length
        ):
            raise RuntimeError(
                f"sequence {sequence.name!r} pre-compaction length changed "
                f"between bucket and pack for request {block.request_id!r}: "
                f"bucket={expected_pre_length}, pack={pre_length}"
            )
        selections.append(kept)
        pre_lengths.append(pre_length)
        compacted_lengths.append(compacted_length)
        token_to_request.extend([request_index] * compacted_length)

    return SequenceSelectionPlan(
        sequence_name=sequence.name,
        selections=tuple(selections),
        pre_compaction_lengths=np.asarray(pre_lengths, dtype=np.int64),
        compacted_lengths=np.asarray(compacted_lengths, dtype=np.int64),
        token_to_request=np.asarray(token_to_request, dtype=np.int64),
    )


def _require_pyarrow() -> Any:
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
    except ImportError as error:  # pragma: no cover - exercised via runtime env
        raise ImportError("agg_direct requires pyarrow") from error
    return pa, pc


def table_pre_compaction_sequence_lengths(
    sequences: Sequence[Any],
    table: Any,
) -> dict[str, np.ndarray]:
    """Per-row list lengths after ``max_length`` clamp; no null_anchor filter.

    Matches ``train._table_sequence_lengths`` (bucket metric input).
    """

    pa, pc = _require_pyarrow()
    result: dict[str, np.ndarray] = {}
    for sequence in sequences:
        if not sequence.fields:
            continue
        source = sequence.fields[0].source
        array = table[source].combine_chunks()
        if pa.types.is_dictionary(array.type):
            dictionary_lengths = pc.list_value_length(array.dictionary)
            lengths = pc.take(dictionary_lengths, array.indices)
        else:
            lengths = pc.list_value_length(array)
        if lengths.null_count:
            lengths = pc.fill_null(lengths, 0)
        values = lengths.to_numpy(zero_copy_only=False).astype(np.int64, copy=True)
        if sequence.max_length is not None:
            np.minimum(values, int(sequence.max_length), out=values)
        result[sequence.name] = values
    return result


def effective_bucket_length_from_pre_compaction(
    lengths: Mapping[str, int],
    *,
    metric: str = "max",
) -> int:
    """Collapse per-UPS pre-compaction lengths into the configured bucket key."""

    if not lengths:
        return 0
    values = list(lengths.values())
    if metric == "sum":
        return int(sum(values))
    if metric == "max":
        return int(max(values))
    raise ValueError(f"length_bucket_metric must be max or sum, got {metric!r}")


def request_group_blocks_from_adapted_table(
    table: Any,
    *,
    source_id: int,
    request_id_column: str,
    sequences: Sequence[Any] = (),
    length_bucket_metric: str = "max",
) -> tuple[RequestGroupBlock, ...]:
    """Build descriptor blocks grouped by ``request_id`` (first-occurrence order).

    Operates on an adapted candidate-flat table for transitional parity with
    ``_request_group_tables``. Same ``request_id`` with interleaved rows becomes
    one block; ``candidate_positions`` preserve adapter output order and may be
    non-contiguous. Does not take payload columns or allocate feature tensors.
    """

    if request_id_column not in table.column_names:
        raise ValueError(
            f"request_id column {request_id_column!r} missing from adapted table"
        )
    request_ids = table[request_id_column].to_pylist()
    positions_by_request: dict[Any, list[int]] = {}
    for row_index, request_id in enumerate(request_ids):
        if request_id is None:
            raise ValueError(
                f"request_id column {request_id_column!r} contains null at row {row_index}"
            )
        try:
            positions_by_request.setdefault(request_id, []).append(row_index)
        except TypeError as error:
            raise ValueError(
                f"request_id column {request_id_column!r} must contain hashable scalars"
            ) from error

    per_sequence = table_pre_compaction_sequence_lengths(sequences, table)
    blocks: list[RequestGroupBlock] = []
    for stable_group_order, (request_id, positions) in enumerate(
        positions_by_request.items()
    ):
        representative = positions[0]
        pre_compaction = {
            name: int(values[representative]) for name, values in per_sequence.items()
        }
        candidate_positions = np.asarray(positions, dtype=np.int64)
        blocks.append(
            RequestGroupBlock(
                source_id=source_id,
                raw_row_index=representative,
                request_id=request_id,
                representative_request_position=representative,
                candidate_positions=candidate_positions,
                candidate_offset=0,
                candidate_count=len(positions),
                pre_compaction_sequence_lengths=pre_compaction,
                effective_bucket_length=effective_bucket_length_from_pre_compaction(
                    pre_compaction,
                    metric=length_bucket_metric,
                ),
                stable_group_order=stable_group_order,
            )
        )
    return tuple(blocks)


def build_packed_request_plan(
    blocks: Sequence[RequestGroupBlock],
) -> PackedRequestPlan:
    """Identity request plan: one block → one request row in the packed batch."""

    block_tuple = tuple(blocks)
    n_blocks = len(block_tuple)
    unique_block_indices = np.arange(n_blocks, dtype=np.int64)
    block_to_request = np.arange(n_blocks, dtype=np.int64)
    if n_blocks == 0:
        candidate_to_request = np.asarray([], dtype=np.int64)
    else:
        candidate_to_request = np.repeat(
            block_to_request,
            [block.candidate_count for block in block_tuple],
        )
    return PackedRequestPlan(
        blocks=block_tuple,
        unique_block_indices=unique_block_indices,
        block_to_request=block_to_request,
        candidate_to_request=candidate_to_request,
    )


def _shuffle_blocks(
    blocks: list[RequestGroupBlock],
    generator: torch.Generator,
) -> list[RequestGroupBlock]:
    if len(blocks) <= 1:
        return blocks
    permutation = torch.randperm(len(blocks), generator=generator).tolist()
    return [blocks[index] for index in permutation]


def iter_shuffled_request_groups(
    blocks: Iterator[RequestGroupBlock],
    *,
    shuffle_buffer_rows: int,
    shuffle_seed: int,
    shard_rank: int = 0,
) -> Iterator[RequestGroupBlock]:
    """Bounded deterministic shuffle; groups stay intact (candidate-row buffer).

    ``shuffle_buffer_rows == 0`` consumes no RNG and yields source order.
    """

    if shuffle_buffer_rows < 0:
        raise ValueError("shuffle_buffer_rows must be non-negative")
    if shuffle_buffer_rows == 0:
        yield from blocks
        return

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(shuffle_seed) + int(shard_rank))
    buffered: list[RequestGroupBlock] = []
    buffered_rows = 0
    for block in blocks:
        if block.candidate_count > shuffle_buffer_rows:
            yield from _shuffle_blocks(buffered, generator)
            buffered = []
            buffered_rows = 0
            yield block
            continue
        while buffered and buffered_rows + block.candidate_count > shuffle_buffer_rows:
            selected_index = int(
                torch.randint(len(buffered), (), generator=generator).item()
            )
            selected = buffered[selected_index]
            buffered[selected_index] = buffered[-1]
            buffered.pop()
            buffered_rows -= selected.candidate_count
            yield selected
        buffered.append(block)
        buffered_rows += block.candidate_count
    yield from _shuffle_blocks(buffered, generator)


def iter_packed_request_groups(
    blocks: Iterator[RequestGroupBlock],
    *,
    batch_size: int,
) -> Iterator[tuple[RequestGroupBlock, ...]]:
    """Pack request groups without splitting unless one group exceeds capacity."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    buffered: list[RequestGroupBlock] = []
    buffered_rows = 0
    for original in blocks:
        block = original
        while block.candidate_count > batch_size:
            if buffered_rows:
                yield tuple(buffered)
                buffered = []
                buffered_rows = 0
            yield (block.slice_candidates(0, batch_size),)
            block = block.slice_candidates(batch_size, block.candidate_count - batch_size)
        if not block.candidate_count:
            continue
        if buffered_rows and buffered_rows + block.candidate_count > batch_size:
            yield tuple(buffered)
            buffered = []
            buffered_rows = 0
        buffered.append(block)
        buffered_rows += block.candidate_count
        if buffered_rows == batch_size:
            yield tuple(buffered)
            buffered = []
            buffered_rows = 0
    if buffered_rows:
        yield tuple(buffered)


def length_bucket_index(effective_length: int, finite_boundaries: Sequence[int]) -> int:
    """Bisect into ``length_buckets`` using the same rule as train.py."""

    # bisect_left over finite max_length boundaries; catch-all is last.
    lo = 0
    hi = len(finite_boundaries)
    while lo < hi:
        mid = (lo + hi) // 2
        if finite_boundaries[mid] < effective_length:
            lo = mid + 1
        else:
            hi = mid
    return lo


def iter_length_bucketed_packs(
    blocks: Iterator[RequestGroupBlock],
    *,
    buckets: Sequence[Any],
    default_batch_size: int,
    shuffle_buffer_rows: int = 0,
    shuffle_seed: int = 0,
    shard_rank: int = 0,
) -> Iterator[tuple[RequestGroupBlock, ...]]:
    """Shuffle then pack by sequence-length bucket (request groups preserved).

    ``buckets`` entries expose ``max_length`` / ``batch_size`` like
    ``LengthBucketConfig``. Empty ``buckets`` packs with ``default_batch_size``.
    """

    shuffled = iter_shuffled_request_groups(
        blocks,
        shuffle_buffer_rows=shuffle_buffer_rows,
        shuffle_seed=shuffle_seed,
        shard_rank=shard_rank,
    )
    if not buckets:
        yield from iter_packed_request_groups(shuffled, batch_size=default_batch_size)
        return

    finite_boundaries = [
        int(bucket.max_length) for bucket in buckets if bucket.max_length is not None
    ]
    buffered: list[list[RequestGroupBlock]] = [[] for _ in buckets]
    buffered_rows = [0] * len(buckets)

    for block in shuffled:
        bucket_index = length_bucket_index(
            int(block.effective_bucket_length),
            finite_boundaries,
        )
        bucket = buckets[bucket_index]
        capacity = int(bucket.batch_size)
        remaining = block
        while remaining.candidate_count > capacity:
            if buffered_rows[bucket_index]:
                yield tuple(buffered[bucket_index])
                buffered[bucket_index] = []
                buffered_rows[bucket_index] = 0
            yield (remaining.slice_candidates(0, capacity),)
            remaining = remaining.slice_candidates(
                capacity,
                remaining.candidate_count - capacity,
            )
        if not remaining.candidate_count:
            continue
        if (
            buffered_rows[bucket_index]
            and buffered_rows[bucket_index] + remaining.candidate_count > capacity
        ):
            yield tuple(buffered[bucket_index])
            buffered[bucket_index] = []
            buffered_rows[bucket_index] = 0
        buffered[bucket_index].append(remaining)
        buffered_rows[bucket_index] += remaining.candidate_count
        if buffered_rows[bucket_index] == capacity:
            yield tuple(buffered[bucket_index])
            buffered[bucket_index] = []
            buffered_rows[bucket_index] = 0

    for bucket_index in range(len(buckets)):
        if buffered_rows[bucket_index]:
            yield tuple(buffered[bucket_index])


def build_request_deduplication_from_pack(
    packed: PackedRequestPlan,
    source_tables: Mapping[int, Any],
    *,
    columns: Sequence[str],
) -> tuple[Any, Any]:
    """Take one representative request row per packed block (identity dedup)."""

    pa, _pc = _require_pyarrow()
    if not packed.blocks:
        raise ValueError("cannot build request deduplication from an empty pack")
    pieces: list[Any] = []
    for block_index in packed.unique_block_indices:
        block = packed.blocks[int(block_index)]
        table = source_tables[block.source_id]
        available = [name for name in columns if name in table.column_names]
        if not available:
            raise ValueError(
                "request deduplication projected an empty column set for packed blocks"
            )
        row = int(block.representative_request_position)
        pieces.append(table.select(available).slice(row, 1))
    request_table = pieces[0] if len(pieces) == 1 else pa.concat_tables(pieces)
    row_indices = torch.as_tensor(packed.candidate_to_request, dtype=torch.long)
    return request_table, row_indices


@dataclass(frozen=True)
class AdaptedAxisBundle:
    """Adapted scanner payload without candidate-flat Arrow.

    Request/sequence values are stored once per unique ``request_id`` (first
    occurrence wins, matching legacy dedup). Item/label/metadata values are
    stored once per candidate. ``candidate_to_request`` is the FeatureBatch
    ``row_indices`` vector for this source.
    """

    n_candidates: int
    n_requests: int
    request_ids: tuple[Any, ...]
    candidate_to_request: Any
    request_features: Mapping[str, tuple[Any, ...]]
    sequence_features: Mapping[str, tuple[Any, ...]]
    item_features: Mapping[str, tuple[Any, ...]]
    label_features: Mapping[str, tuple[Any, ...]]
    label_mask_features: Mapping[str, tuple[Any, ...]]
    candidate_metadata: Mapping[str, tuple[Any, ...]]
    request_raw_rows: Any
    candidate_raw_rows: Any


class SourceRegistry:
    """Retain axis bundles (or tables) while shuffle/bucket buffers reference them.

    ``acquire`` / ``release`` are counted per retained block. When the last
    reference drops, the payload is deleted so Arrow/Python buffers can GC.
    Peak retained source count is exposed for RSS monitoring (plan phase 8).
    """

    def __init__(self) -> None:
        self._sources: dict[int, Any] = {}
        self._refcount: dict[int, int] = {}
        self._next_id = 0
        self.peak_retained_sources = 0
        self.release_events = 0

    def put(self, payload: Any) -> int:
        source_id = self._next_id
        self._next_id += 1
        self._sources[source_id] = payload
        self._refcount[source_id] = 0
        self.peak_retained_sources = max(
            self.peak_retained_sources, len(self._sources)
        )
        return source_id

    def get(self, source_id: int) -> Any:
        try:
            return self._sources[source_id]
        except KeyError as error:
            raise KeyError(f"source_id {source_id} is not retained") from error

    def acquire(self, source_id: int, count: int = 1) -> None:
        if count < 0:
            raise ValueError("acquire count must be non-negative")
        if source_id not in self._sources:
            raise KeyError(f"source_id {source_id} is not retained")
        self._refcount[source_id] = self._refcount.get(source_id, 0) + count

    def release(self, source_id: int, count: int = 1) -> None:
        if count < 0:
            raise ValueError("release count must be non-negative")
        if source_id not in self._refcount:
            raise KeyError(f"source_id {source_id} has no references")
        remaining = self._refcount[source_id] - count
        if remaining > 0:
            self._refcount[source_id] = remaining
            return
        if remaining < 0:
            raise ValueError(
                f"source_id {source_id} release underflow "
                f"(held={self._refcount[source_id]}, release={count})"
            )
        del self._refcount[source_id]
        del self._sources[source_id]
        self.release_events += 1

    def retained_source_ids(self) -> tuple[int, ...]:
        return tuple(self._sources.keys())

    @property
    def retained_count(self) -> int:
        return len(self._sources)


def request_group_blocks_from_axis_bundle(
    bundle: AdaptedAxisBundle,
    *,
    source_id: int,
    sequences: Sequence[Any] = (),
    length_bucket_metric: str = "max",
) -> tuple[RequestGroupBlock, ...]:
    """Build descriptors from an axis-separated adapted bundle."""

    if bundle.n_requests == 0:
        return ()
    positions_by_slot: list[list[int]] = [[] for _ in range(bundle.n_requests)]
    for candidate_index, slot in enumerate(bundle.candidate_to_request):
        positions_by_slot[int(slot)].append(int(candidate_index))

    blocks: list[RequestGroupBlock] = []
    for stable_group_order, positions in enumerate(positions_by_slot):
        if not positions:
            raise ValueError(
                f"request slot {stable_group_order} has no candidates in axis bundle"
            )
        pre_compaction: dict[str, int] = {}
        for sequence in sequences:
            if not sequence.fields:
                continue
            source = sequence.fields[0].source
            if source not in bundle.sequence_features:
                raise ValueError(
                    f"sequence source {source!r} missing from axis bundle"
                )
            length = len(bundle.sequence_features[source][stable_group_order])
            if sequence.max_length is not None:
                length = min(length, int(sequence.max_length))
            pre_compaction[sequence.name] = int(length)
        blocks.append(
            RequestGroupBlock(
                source_id=source_id,
                raw_row_index=int(bundle.request_raw_rows[stable_group_order]),
                request_id=bundle.request_ids[stable_group_order],
                representative_request_position=stable_group_order,
                candidate_positions=np.asarray(positions, dtype=np.int64),
                candidate_offset=0,
                candidate_count=len(positions),
                pre_compaction_sequence_lengths=pre_compaction,
                effective_bucket_length=effective_bucket_length_from_pre_compaction(
                    pre_compaction,
                    metric=length_bucket_metric,
                ),
                stable_group_order=stable_group_order,
            )
        )
    return tuple(blocks)


def _arrow_array_from_python_values(values: Sequence[Any]) -> Any:
    """Build an Arrow array without collapsing empty lists to null type."""

    pa, _pc = _require_pyarrow()
    array = pa.array(list(values))
    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        value_type = array.type.value_type
        if pa.types.is_null(value_type):
            # All-empty lists infer list<null>; pin a concrete value type.
            sample = next((value for value in values if value), None)
            if sample and isinstance(sample[0], float):
                return pa.array(list(values), type=pa.list_(pa.float32()))
            return pa.array(list(values), type=pa.list_(pa.int64()))
        return array
    if not pa.types.is_null(array.type):
        return array
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool):
            return pa.array(list(values), type=pa.bool_())
        if isinstance(value, int):
            return pa.array(list(values), type=pa.int64())
        if isinstance(value, float):
            return pa.array(list(values), type=pa.float32())
        if isinstance(value, str):
            return pa.array(list(values), type=pa.string())
        if isinstance(value, (list, tuple)):
            if value and isinstance(value[0], float):
                return pa.array(list(values), type=pa.list_(pa.float32()))
            return pa.array(list(values), type=pa.list_(pa.int64()))
        break
    return pa.array(list(values), type=pa.int64())


def prepare_packed_axis_batch(
    bundles: Mapping[int, AdaptedAxisBundle],
    packed: PackedRequestPlan,
    *,
    sequences: Sequence[Any],
    request_id_column: str | None = None,
    candidate_request_columns: Sequence[str] = (),
) -> PreparedAxisBatch:
    """Gather one packed batch without constructing candidate/request Arrow.

    Candidate payload is copied only as references to the already-normalized
    Python scalars/lists owned by each axis bundle. Request and sequence values
    remain unique per request; ``request_row_indices`` performs the only
    candidate-to-request broadcast required by the model.
    """

    if not packed.blocks:
        raise ValueError("cannot prepare an empty packed axis batch")

    source_ids = {block.source_id for block in packed.blocks}
    missing_sources = sorted(source_ids - set(bundles))
    if missing_sources:
        raise KeyError(f"axis bundles missing source IDs {missing_sources}")

    request_names: set[str] = set()
    candidate_names: set[str] = set()
    for source_id in source_ids:
        bundle = bundles[source_id]
        request_names.update(bundle.request_features)
        request_names.update(bundle.sequence_features)
        candidate_names.update(bundle.item_features)
        candidate_names.update(bundle.label_features)
        candidate_names.update(bundle.label_mask_features)
        candidate_names.update(bundle.candidate_metadata)
    if request_id_column is not None:
        request_names.add(request_id_column)

    request_rows: dict[str, list[Any]] = {
        name: [] for name in sorted(request_names)
    }
    candidate_rows: dict[str, list[Any]] = {
        name: [] for name in sorted(candidate_names)
    }
    broadcast_names = tuple(dict.fromkeys(candidate_request_columns))
    for name in broadcast_names:
        candidate_rows.setdefault(name, [])

    def request_value(
        bundle: AdaptedAxisBundle,
        name: str,
        slot: int,
    ) -> Any:
        if name in bundle.request_features:
            return bundle.request_features[name][slot]
        if name in bundle.sequence_features:
            return bundle.sequence_features[name][slot]
        if request_id_column is not None and name == request_id_column:
            return bundle.request_ids[slot]
        raise KeyError(f"request column {name!r} missing from axis bundle")

    for block_index in packed.unique_block_indices:
        block = packed.blocks[int(block_index)]
        bundle = bundles[block.source_id]
        slot = int(block.representative_request_position)
        for name in request_rows:
            request_rows[name].append(request_value(bundle, name, slot))

    for block in packed.blocks:
        bundle = bundles[block.source_id]
        request_slot = int(block.representative_request_position)
        for candidate_position in block.active_candidate_positions():
            candidate_slot = int(candidate_position)
            for name in candidate_rows:
                if name in bundle.item_features:
                    value = bundle.item_features[name][candidate_slot]
                elif name in bundle.label_features:
                    value = bundle.label_features[name][candidate_slot]
                elif name in bundle.label_mask_features:
                    value = bundle.label_mask_features[name][candidate_slot]
                elif name in bundle.candidate_metadata:
                    value = bundle.candidate_metadata[name][candidate_slot]
                elif name in broadcast_names:
                    value = request_value(bundle, name, request_slot)
                else:
                    raise KeyError(
                        f"candidate column {name!r} missing from axis bundle "
                        f"source {block.source_id}"
                    )
                candidate_rows[name].append(value)

    sequence_plans = {
        sequence.name: build_axis_sequence_selection_plan(
            sequence,
            packed=packed,
            bundles=bundles,
        )
        for sequence in sequences
    }
    n_candidates = int(sum(block.candidate_count for block in packed.blocks))
    n_requests = len(packed.unique_block_indices)
    if any(len(values) != n_requests for values in request_rows.values()):
        raise RuntimeError("packed request-axis column lengths are inconsistent")
    if any(len(values) != n_candidates for values in candidate_rows.values()):
        raise RuntimeError("packed candidate-axis column lengths are inconsistent")

    return PreparedAxisBatch(
        request_values={
            name: tuple(values) for name, values in request_rows.items()
        },
        candidate_values={
            name: tuple(values) for name, values in candidate_rows.items()
        },
        request_row_indices=torch.as_tensor(
            packed.candidate_to_request,
            dtype=torch.long,
        ),
        sequence_plans=sequence_plans,
        n_requests=n_requests,
        n_candidates=n_candidates,
    )


def materialize_packed_axis_bundles(
    bundles: Mapping[int, AdaptedAxisBundle],
    packed: PackedRequestPlan,
    *,
    request_columns: Sequence[str],
    sequence_columns: Sequence[str],
    candidate_columns: Sequence[str],
    request_id_column: str | None = None,
) -> tuple[Any, Any, Any]:
    """Build narrow request + candidate Arrow tables for one packed batch.

    This is the only Arrow materialization on the direct path: once per packed
    batch boundary, never a candidate-flat rebuild of an entire scanner table.
    Request-id (and other request scalars listed in ``candidate_columns`` that
    live on the request axis) are broadcast onto candidates only here, for
    group_id / scenario / prediction-key parity with legacy flat tables.
    """

    pa, _pc = _require_pyarrow()
    if not packed.blocks:
        raise ValueError("cannot materialize an empty packed plan")

    request_rows: dict[str, list[Any]] = {
        name: [] for name in (*request_columns, *sequence_columns)
    }
    candidate_rows: dict[str, list[Any]] = {name: [] for name in candidate_columns}
    row_indices: list[int] = []

    for request_index, block_index in enumerate(packed.unique_block_indices):
        block = packed.blocks[int(block_index)]
        bundle = bundles[block.source_id]
        slot = int(block.representative_request_position)
        for name in request_columns:
            request_rows[name].append(bundle.request_features[name][slot])
        for name in sequence_columns:
            request_rows[name].append(bundle.sequence_features[name][slot])

    for block_index, block in enumerate(packed.blocks):
        bundle = bundles[block.source_id]
        request_slot = int(packed.block_to_request[block_index])
        bundle_slot = int(block.representative_request_position)
        for candidate_pos in block.active_candidate_positions():
            cand = int(candidate_pos)
            for name in candidate_columns:
                if name in bundle.item_features:
                    candidate_rows[name].append(bundle.item_features[name][cand])
                elif name in bundle.label_features:
                    candidate_rows[name].append(bundle.label_features[name][cand])
                elif name in bundle.label_mask_features:
                    candidate_rows[name].append(
                        bundle.label_mask_features[name][cand]
                    )
                elif name in bundle.candidate_metadata:
                    candidate_rows[name].append(
                        bundle.candidate_metadata[name][cand]
                    )
                elif name in bundle.request_features:
                    candidate_rows[name].append(
                        bundle.request_features[name][bundle_slot]
                    )
                elif request_id_column is not None and name == request_id_column:
                    candidate_rows[name].append(bundle.request_ids[bundle_slot])
                else:
                    raise KeyError(
                        f"candidate column {name!r} missing from axis bundle"
                    )
            row_indices.append(request_slot)

    request_table = pa.table(
        {
            name: _arrow_array_from_python_values(values)
            for name, values in request_rows.items()
        }
    )
    candidate_table = pa.table(
        {
            name: _arrow_array_from_python_values(values)
            for name, values in candidate_rows.items()
        }
    )
    return (
        candidate_table,
        request_table,
        torch.tensor(row_indices, dtype=torch.long),
    )


@dataclass(frozen=True)
class PreparedBatchTable:
    """Candidate-major Arrow table plus optional precomputed request dedup."""

    table: Any
    request_deduplication: tuple[Any, Any] | None = None

    @property
    def num_rows(self) -> int:
        return int(self.table.num_rows)

    @property
    def nbytes(self) -> int:
        return int(getattr(self.table, "nbytes", 0))

    @property
    def column_names(self) -> Any:
        return self.table.column_names

    def __getitem__(self, key: Any) -> Any:
        return self.table[key]


def materialize_packed_blocks(
    source_tables: Mapping[int, Any],
    blocks: Sequence[RequestGroupBlock],
) -> Any:
    """Take candidate rows for a packed batch (transitional oracle / compare path).

    Does not broadcast or re-encode features; only gathers rows already present
    on adapted source tables. Same ``request_id`` never spans sources. Takes are
    coalesced per ``source_id`` to avoid one ``take``/``concat`` per block.
    """

    pa, _pc = _require_pyarrow()
    if not blocks:
        raise ValueError("cannot materialize an empty packed block list")

    # Preserve pack order while coalescing runs of the same source.
    pieces: list[Any] = []
    run_source: int | None = None
    run_positions: list[np.ndarray] = []

    def flush_run() -> None:
        nonlocal run_source, run_positions
        if run_source is None:
            return
        table = source_tables[run_source]
        positions = (
            run_positions[0]
            if len(run_positions) == 1
            else np.concatenate(run_positions)
        )
        if (
            len(positions) > 0
            and int(positions[-1]) == int(positions[0]) + len(positions) - 1
            and bool(
                np.all(
                    positions
                    == np.arange(int(positions[0]), int(positions[0]) + len(positions))
                )
            )
        ):
            pieces.append(table.slice(int(positions[0]), len(positions)))
        else:
            pieces.append(table.take(pa.array(positions, type=pa.int64())))
        run_source = None
        run_positions = []

    for block in blocks:
        positions = np.asarray(block.active_candidate_positions(), dtype=np.int64)
        if run_source is not None and block.source_id != run_source:
            flush_run()
        run_source = block.source_id
        run_positions.append(positions)
    flush_run()

    if len(pieces) == 1:
        return pieces[0]
    return pa.concat_tables(pieces)
