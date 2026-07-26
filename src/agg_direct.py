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
from dataclasses import dataclass, field, replace
from typing import Any
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DirectPipelineStats:
    """Host-side direct-path counters for benchmark / compare reporting."""

    peak_retained_sources: int = 0
    release_events: int = 0
    packs_observed: int = 0
    cross_source_duplicate_request_id_events: int = 0
    packs_with_cross_source_duplicate_request_ids: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "peak_retained_sources": self.peak_retained_sources,
            "release_events": self.release_events,
            "packs_observed": self.packs_observed,
            "cross_source_duplicate_request_id_events": (
                self.cross_source_duplicate_request_id_events
            ),
            "packs_with_cross_source_duplicate_request_ids": (
                self.packs_with_cross_source_duplicate_request_ids
            ),
        }


_LAST_DIRECT_PIPELINE_STATS = DirectPipelineStats()


def get_direct_pipeline_stats() -> DirectPipelineStats:
    return _LAST_DIRECT_PIPELINE_STATS


def publish_direct_pipeline_stats(stats: DirectPipelineStats) -> None:
    global _LAST_DIRECT_PIPELINE_STATS
    _LAST_DIRECT_PIPELINE_STATS = stats


def reset_direct_pipeline_stats() -> None:
    publish_direct_pipeline_stats(DirectPipelineStats())


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

    Contiguous kept windows are stored as ``(start, end)`` ranges; only
    null-anchor holes materialize an index ndarray. ``selections_are_ranges``
    is True when every request used a contiguous range (fast tensorize path).
    When ranges are used, ``range_starts`` / ``range_ends`` parallel arrays
    avoid per-field tuple unpacking in the gather hot loop.
    """

    sequence_name: str
    # Per-request local indices into the representative row list after compaction.
    selections: tuple[Any, ...]
    pre_compaction_lengths: Any
    compacted_lengths: Any
    token_to_request: Any
    selections_are_ranges: bool = False
    range_starts: Any = None  # np.ndarray[int64] | None
    range_ends: Any = None  # np.ndarray[int64] | None


@dataclass(frozen=True)
class PreparedAxisBatch:
    """Pack-boundary, Arrow-free payload for direct FeatureBatch construction.

    Values remain separated on request and candidate axes. Sequence plans are
    built only after shuffle/bucket/pack and are shared by every aligned field
    of the same sequence.

    Column values may be ``tuple``/``list``, dense ``ndarray``, or
    :class:`SequenceColumnBatch` (lazy CompactListColumn gather).
    """

    request_values: Mapping[str, Any]
    candidate_values: Mapping[str, Any]
    request_row_indices: Any
    sequence_plans: Mapping[str, SequenceSelectionPlan]
    n_requests: int
    n_candidates: int

    @property
    def num_rows(self) -> int:
        return self.n_candidates


@dataclass(frozen=True)
class SequenceColumnRef:
    """Lazy prepare handle: resolve ``column[slot]`` during tensorize gather."""

    column: Any
    slot: int


@dataclass(frozen=True)
class SequenceColumnBatch:
    """Per-request CompactListColumn gather without per-cell ref objects.

    ``columns`` stores unique CompactListColumn objects; ``column_index[i]``
    selects which column request ``i`` reads, and ``slots[i]`` is the row
    inside that column. Tensorize applies the shared selection plan against
    these handles in one pass.
    """

    columns: tuple[Any, ...]
    slots: Any  # np.ndarray[int64], length == n_requests
    column_index: Any = None  # np.ndarray[int16] or None => identity columns[i]

    def __len__(self) -> int:
        return int(np.asarray(self.slots).shape[0])

    def _column_at(self, index: int) -> Any:
        if self.column_index is None:
            return self.columns[index]
        return self.columns[int(self.column_index[index])]

    def __getitem__(self, index: int) -> Any:
        return self._column_at(index)[int(self.slots[index])]

    def __iter__(self):
        slots = self.slots
        for index in range(len(self)):
            yield self._column_at(index)[int(slots[index])]


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
) -> tuple[Any, int, int]:
    """Return (kept_local_indices, pre_compaction_length, compacted_length).

    Matches the pack-time contract: clamp/truncate first, then drop null-anchor
    steps. ``pre_compaction_length`` is the bucket input; ``compacted_length``
    is the final FeatureBatch sequence length.

    When the kept window is contiguous (no null-anchor holes), indices are a
    ``(start, end)`` range instead of a materialized ``arange`` — tensorize
    slices the CompactListColumn buffer directly.
    """

    if list_length < 0:
        raise ValueError("list_length must be non-negative")
    start, end = _truncation_window(list_length, max_length, truncation)
    pre_compaction = end - start
    if anchor_is_null is None:
        return (start, end), pre_compaction, pre_compaction
    if anchor_is_null.shape[0] != list_length:
        raise ValueError(
            f"anchor_is_null length {anchor_is_null.shape[0]} != list_length {list_length}"
        )
    if pre_compaction == 0:
        return (start, end), 0, 0
    window_nulls = anchor_is_null[start:end]
    if not bool(window_nulls.any()):
        return (start, end), pre_compaction, pre_compaction
    kept = np.flatnonzero(~window_nulls).astype(np.int64, copy=False) + start
    return kept, pre_compaction, int(kept.size)


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
    # Field-length alignment is invariant for a whole AdaptedAxisBundle (same
    # membership gather). Validate once per source instead of once per
    # (request × field) — that check dominated prepare_packed_axis_batch.
    alignment_checked_sources: set[int] = set()

    def _sequence_row_length(column: Any, slot: int) -> int:
        row_length = getattr(column, "row_length", None)
        if callable(row_length):
            return int(row_length(slot))
        row = column[slot]
        return 0 if row is None else len(row)

    def _anchor_null_mask(column: Any, slot: int, list_length: int) -> np.ndarray | None:
        if list_length == 0:
            return np.asarray([], dtype=bool)
        values = getattr(column, "values", None)
        if values is not None and getattr(values, "dtype", None) != object:
            # Dense numeric CompactListColumn buffers cannot hold Python None.
            return None
        anchor_row = column[slot]
        if anchor_row is None:
            return np.ones(list_length, dtype=bool)
        if isinstance(anchor_row, np.ndarray):
            if anchor_row.dtype == object:
                return anchor_row == None  # noqa: E711 — element-wise None check
            return None
        return np.fromiter(
            (value is None for value in anchor_row),
            dtype=bool,
            count=list_length,
        )

    def _validate_source_alignment(bundle: "AdaptedAxisBundle", source_id: int) -> None:
        if source_id in alignment_checked_sources:
            return
        try:
            primary_column = bundle.sequence_features[primary_source]
        except KeyError as error:
            raise ValueError(
                f"sequence source {primary_source!r} missing from axis bundle"
            ) from error
        primary_offsets = getattr(primary_column, "offsets", None)
        for field in sequence.fields[1:]:
            try:
                aligned_column = bundle.sequence_features[field.source]
            except KeyError as error:
                raise ValueError(
                    f"sequence source {field.source!r} missing from axis bundle"
                ) from error
            aligned_offsets = getattr(aligned_column, "offsets", None)
            if (
                primary_offsets is not None
                and aligned_offsets is not None
                and (
                    primary_offsets is aligned_offsets
                    or np.array_equal(primary_offsets, aligned_offsets)
                )
            ):
                continue
            for slot in range(bundle.n_requests):
                primary_length = _sequence_row_length(primary_column, slot)
                aligned_length = _sequence_row_length(aligned_column, slot)
                if aligned_length != primary_length:
                    raise ValueError(
                        f"sequence {sequence.name!r} field {field.name!r} has length "
                        f"{aligned_length}, expected {primary_length} for request "
                        f"slot {slot} on source {source_id}"
                    )
        alignment_checked_sources.add(source_id)

    n_plan_requests = len(packed.unique_block_indices)
    range_starts_arr = np.empty(n_plan_requests, dtype=np.int64)
    range_ends_arr = np.empty(n_plan_requests, dtype=np.int64)
    pre_lengths_arr = np.empty(n_plan_requests, dtype=np.int64)
    compacted_lengths_arr = np.empty(n_plan_requests, dtype=np.int64)
    max_length = sequence.max_length
    truncation = sequence.truncation

    source_ids_arr = np.empty(n_plan_requests, dtype=np.int64)
    request_slots_arr = np.empty(n_plan_requests, dtype=np.int64)
    for request_index, block_index in enumerate(packed.unique_block_indices):
        block = packed.blocks[int(block_index)]
        source_ids_arr[request_index] = int(block.source_id)
        request_slots_arr[request_index] = int(block.representative_request_position)
        _validate_source_alignment(bundles[block.source_id], block.source_id)

    # Dense numeric anchors cannot store None → truncate-only, fully vectorizable.
    anchor_dense_numeric = True
    if anchor_source is not None and n_plan_requests:
        for source_id in np.unique(source_ids_arr):
            try:
                anchor_column = bundles[int(source_id)].sequence_features[anchor_source]
            except KeyError as error:
                raise ValueError(
                    f"sequence source {anchor_source!r} missing from axis bundle"
                ) from error
            values = getattr(anchor_column, "values", None)
            if values is None or getattr(values, "dtype", None) == object:
                anchor_dense_numeric = False
                break
    elif anchor_source is None:
        anchor_dense_numeric = True
    else:
        anchor_dense_numeric = True

    if anchor_dense_numeric and n_plan_requests:
        list_lengths = np.empty(n_plan_requests, dtype=np.int64)
        for source_id in np.unique(source_ids_arr):
            reqs = np.flatnonzero(source_ids_arr == source_id)
            try:
                primary_column = bundles[int(source_id)].sequence_features[
                    primary_source
                ]
            except KeyError as error:
                raise ValueError(
                    f"sequence source {primary_source!r} missing from axis bundle"
                ) from error
            offsets = getattr(primary_column, "offsets", None)
            if offsets is None:
                for out_index, slot in zip(reqs, request_slots_arr[reqs]):
                    list_lengths[int(out_index)] = _sequence_row_length(
                        primary_column, int(slot)
                    )
                continue
            row_slots = request_slots_arr[reqs]
            list_lengths[reqs] = offsets[row_slots + 1] - offsets[row_slots]

        if max_length is None:
            range_starts_arr.fill(0)
            range_ends_arr[:] = list_lengths
        elif truncation == "head":
            range_starts_arr.fill(0)
            range_ends_arr[:] = np.minimum(list_lengths, int(max_length))
        elif truncation == "tail":
            range_ends_arr[:] = list_lengths
            range_starts_arr[:] = np.maximum(list_lengths - int(max_length), 0)
        else:
            raise ValueError(f"unsupported sequence truncation {truncation!r}")
        pre_lengths_arr[:] = range_ends_arr - range_starts_arr
        compacted_lengths_arr[:] = pre_lengths_arr

        has_expected = False
        for block_index in packed.unique_block_indices:
            if (
                sequence.name
                in packed.blocks[int(block_index)].pre_compaction_sequence_lengths
            ):
                has_expected = True
                break
        if has_expected:
            for request_index, block_index in enumerate(packed.unique_block_indices):
                block = packed.blocks[int(block_index)]
                expected_pre_length = block.pre_compaction_sequence_lengths.get(
                    sequence.name
                )
                if (
                    expected_pre_length is not None
                    and int(expected_pre_length) != int(pre_lengths_arr[request_index])
                ):
                    raise RuntimeError(
                        f"sequence {sequence.name!r} pre-compaction length changed "
                        f"between bucket and pack for request {block.request_id!r}: "
                        f"bucket={expected_pre_length}, pack={int(pre_lengths_arr[request_index])}"
                    )

        # Direct tensorize uses range_starts/ends; skip materializing per-request
        # selection tuples and token_to_request (only tests assert the latter).
        return SequenceSelectionPlan(
            sequence_name=sequence.name,
            selections=tuple(),
            pre_compaction_lengths=pre_lengths_arr,
            compacted_lengths=compacted_lengths_arr,
            token_to_request=np.empty(0, dtype=np.int64),
            selections_are_ranges=True,
            range_starts=range_starts_arr,
            range_ends=range_ends_arr,
        )

    selections: list[Any] = []
    saw_sparse_selection = False
    for request_index, block_index in enumerate(packed.unique_block_indices):
        block = packed.blocks[int(block_index)]
        bundle = bundles[block.source_id]
        request_slot = int(request_slots_arr[request_index])
        try:
            primary_column = bundle.sequence_features[primary_source]
        except KeyError as error:
            raise ValueError(
                f"sequence source {primary_source!r} missing from axis bundle"
            ) from error
        list_length = _sequence_row_length(primary_column, request_slot)

        anchor_is_null = None
        if anchor_source is not None:
            try:
                anchor_column = bundle.sequence_features[anchor_source]
            except KeyError as error:
                raise ValueError(
                    f"sequence source {anchor_source!r} missing from axis bundle"
                ) from error
            anchor_is_null = _anchor_null_mask(
                anchor_column, request_slot, list_length
            )

        kept, pre_length, compacted_length = (
            row_sequence_selection_after_truncate_then_compact(
                list_length=list_length,
                anchor_is_null=anchor_is_null,
                max_length=max_length,
                truncation=truncation,
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
        if (
            isinstance(kept, tuple)
            and len(kept) == 2
            and not isinstance(kept[0], (list, tuple, np.ndarray))
        ):
            range_starts_arr[request_index] = int(kept[0])
            range_ends_arr[request_index] = int(kept[1])
        else:
            saw_sparse_selection = True
            range_starts_arr[request_index] = 0
            range_ends_arr[request_index] = 0
        selections.append(kept)
        pre_lengths_arr[request_index] = pre_length
        compacted_lengths_arr[request_index] = compacted_length

    selections_tuple = tuple(selections)
    selections_are_ranges = not saw_sparse_selection
    if n_plan_requests and int(compacted_lengths_arr.sum()):
        token_to_request_arr = np.repeat(
            np.arange(n_plan_requests, dtype=np.int64),
            compacted_lengths_arr,
        )
    else:
        token_to_request_arr = np.empty(0, dtype=np.int64)
    return SequenceSelectionPlan(
        sequence_name=sequence.name,
        selections=selections_tuple,
        pre_compaction_lengths=pre_lengths_arr,
        compacted_lengths=compacted_lengths_arr,
        token_to_request=token_to_request_arr,
        selections_are_ranges=selections_are_ranges,
        range_starts=range_starts_arr if selections_are_ranges else None,
        range_ends=range_ends_arr if selections_are_ranges else None,
    )


_PA = None
_PC = None


def _require_pyarrow() -> Any:
    global _PA, _PC
    if _PA is not None:
        return _PA, _PC
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
    except ImportError as error:  # pragma: no cover - exercised via runtime env
        raise ImportError("agg_direct requires pyarrow") from error
    _PA, _PC = pa, pc
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
class CompactListColumn:
    """Columnar ragged lists for one feature across requests/candidates.

    Stores a single flat NumPy values buffer plus offsets so ProcessPool IPC
    pickles ~one array per feature instead of tens of thousands of tiny Python
    lists (sequence features dominate AdaptedAxisBundle pickle cost).
    ``column[i]`` returns a NumPy view into ``values`` (empty when the row is
    empty). Pack/tensorize accept ndarray and avoid a Python list materialization
    on every gather.
    """

    values: Any
    offsets: Any

    def __len__(self) -> int:
        return int(self.offsets.shape[0]) - 1

    def __getitem__(self, index: int) -> Any:
        start = int(self.offsets[index])
        stop = int(self.offsets[index + 1])
        return self.values[start:stop]

    def row_length(self, index: int) -> int:
        return int(self.offsets[index + 1] - self.offsets[index])


def compact_list_column_from_rows(rows: Sequence[Any]) -> CompactListColumn:
    """Pack a sequence of row lists/ndarrays into :class:`CompactListColumn`."""

    n_rows = len(rows)
    offsets = np.empty(n_rows + 1, dtype=np.int64)
    offsets[0] = 0
    total = 0
    sample = None
    saw_none = False
    numeric_nd = True
    sample_dtype: Any = None
    for index, row in enumerate(rows):
        if row is None:
            length = 0
        elif isinstance(row, np.ndarray):
            length = int(row.shape[0])
            if length:
                if row.dtype == object:
                    numeric_nd = False
                    if sample is None:
                        for item in row:
                            if item is None:
                                saw_none = True
                            elif sample is None:
                                sample = item
                elif sample_dtype is None:
                    sample_dtype = row.dtype
                    sample = row.item(0) if length else sample
        else:
            numeric_nd = False
            length = len(row)
            if not saw_none:
                for item in row:
                    if item is None:
                        saw_none = True
                        break
            if sample is None and length:
                for item in row:
                    if item is not None:
                        sample = item
                        break
        total += length
        offsets[index + 1] = total

    if total == 0:
        return CompactListColumn(
            values=np.empty(0, dtype=np.int64),
            offsets=offsets,
        )

    if numeric_nd and sample_dtype is not None and sample_dtype != object:
        if total == 0:
            return CompactListColumn(
                values=np.empty(0, dtype=np.int64),
                offsets=offsets,
            )
        values = np.empty(total, dtype=sample_dtype)
        cursor = 0
        for row in rows:
            if isinstance(row, np.ndarray):
                length = int(row.shape[0])
                if length:
                    values[cursor : cursor + length] = row
                    cursor += length
        return CompactListColumn(values=values, offsets=offsets)

    if saw_none or sample is None:
        dtype: Any = object
    elif isinstance(sample, (bool, np.bool_)):
        dtype = object
    elif isinstance(sample, (int, np.integer)):
        dtype = np.int64
    elif isinstance(sample, (float, np.floating)):
        dtype = np.float64
    else:
        dtype = object
    if dtype is object:
        values = np.empty(total, dtype=object)
    else:
        values = np.empty(total, dtype=dtype)
    for index, row in enumerate(rows):
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        if stop > start:
            values[start:stop] = row
    return CompactListColumn(values=values, offsets=offsets)


def axis_feature_column_from_values(values: Sequence[Any]) -> Any:
    """Pack one axis feature into a NumPy column or :class:`CompactListColumn`.

    Scalar request/item/label columns become dense ``ndarray``; ragged bag
    columns reuse CompactListColumn. Both pickle far cheaper than tuples of
    Python ints/lists across ProcessPool IPC.
    """

    n_rows = len(values)
    if n_rows == 0:
        return np.empty(0, dtype=np.int64)
    sample = None
    for value in values:
        if value is not None:
            sample = value
            break
    if sample is None:
        return np.asarray(values, dtype=object)
    if isinstance(sample, (list, tuple, np.ndarray)):
        return compact_list_column_from_rows(values)
    if isinstance(sample, (bool, np.bool_)):
        return np.asarray(values, dtype=object)
    if isinstance(sample, (int, np.integer)):
        if any(value is None for value in values):
            return np.asarray(values, dtype=object)
        return np.asarray(values, dtype=np.int64)
    if isinstance(sample, (float, np.floating)):
        return np.asarray(values, dtype=np.float64)
    return np.asarray(values, dtype=object)


def share_compact_list_offsets(
    columns: Mapping[str, CompactListColumn],
) -> dict[str, CompactListColumn]:
    """Reuse identical offsets buffers across columns (same UPS membership).

    Adapt builds one CompactListColumn per sequence field; fields from the same
    membership gather share row lengths. Sharing the offsets array makes pack-time
    alignment checks an identity compare instead of ``np.array_equal``.
    """

    unique_offsets: list[Any] = []
    shared: dict[str, CompactListColumn] = {}
    for name, column in columns.items():
        matched = None
        offsets = column.offsets
        for candidate in unique_offsets:
            if candidate.shape == offsets.shape and np.array_equal(
                candidate, offsets
            ):
                matched = candidate
                break
        if matched is None:
            unique_offsets.append(offsets)
            matched = offsets
        if matched is offsets:
            shared[name] = column
        else:
            shared[name] = CompactListColumn(
                values=column.values,
                offsets=matched,
            )
    return shared


@dataclass(frozen=True)
class AdaptedAxisBundle:
    """Adapted scanner payload without candidate-flat Arrow.

    Request/sequence values are stored once per unique ``request_id`` (first
    occurrence wins, matching legacy dedup). Item/label/metadata values are
    stored once per candidate. ``candidate_to_request`` is the FeatureBatch
    ``row_indices`` vector for this source.

    Feature columns may be dense ``ndarray`` or :class:`CompactListColumn` for
    cheaper ProcessPool IPC; ``column[i]`` still returns the per-row value (a
    scalar or a NumPy view for ragged rows).
    """

    n_candidates: int
    n_requests: int
    request_ids: tuple[Any, ...]
    candidate_to_request: Any
    request_features: Mapping[str, Any]
    sequence_features: Mapping[str, Any]
    item_features: Mapping[str, Any]
    label_features: Mapping[str, Any]
    label_mask_features: Mapping[str, Any]
    candidate_metadata: Mapping[str, Any]
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
        self.packs_observed = 0
        self.cross_source_duplicate_request_id_events = 0
        self.packs_with_cross_source_duplicate_request_ids = 0

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

    def observe_pack(self, blocks: Sequence[RequestGroupBlock]) -> int:
        """Count cross-source duplicate ``request_id`` values in one pack.

        Production ranking logs should not repeat a request across scanned
        tables. Direct path keeps one request row per block (no cross-table
        merge); this counter surfaces contract violations without changing
        pack semantics.
        """

        self.packs_observed += 1
        sources_by_request: dict[Any, set[int]] = {}
        for block in blocks:
            try:
                sources_by_request.setdefault(block.request_id, set()).add(
                    int(block.source_id)
                )
            except TypeError:
                # Unhashable request_id is rejected earlier in the adapter.
                continue
        duplicate_events = 0
        for request_id, source_ids in sources_by_request.items():
            if len(source_ids) < 2:
                continue
            duplicate_events += 1
            logger.warning(
                "cross-source duplicate request_id=%r across source_ids=%s "
                "(direct path keeps separate request rows; legacy may dedup)",
                request_id,
                sorted(source_ids),
            )
        if duplicate_events:
            self.packs_with_cross_source_duplicate_request_ids += 1
            self.cross_source_duplicate_request_id_events += duplicate_events
        return duplicate_events

    def snapshot_stats(self) -> DirectPipelineStats:
        return DirectPipelineStats(
            peak_retained_sources=int(self.peak_retained_sources),
            release_events=int(self.release_events),
            packs_observed=int(self.packs_observed),
            cross_source_duplicate_request_id_events=int(
                self.cross_source_duplicate_request_id_events
            ),
            packs_with_cross_source_duplicate_request_ids=int(
                self.packs_with_cross_source_duplicate_request_ids
            ),
        )


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
            seq_column = bundle.sequence_features[source]
            if isinstance(seq_column, CompactListColumn):
                length = seq_column.row_length(stable_group_order)
            else:
                length = len(seq_column[stable_group_order])
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

    broadcast_names = tuple(dict.fromkeys(candidate_request_columns))
    broadcast_set = set(broadcast_names)
    candidate_name_list = sorted(set(candidate_names) | broadcast_set)
    request_name_list = sorted(request_names)

    # Unique-request gather: resolve column once per (source, name), then index.
    request_order: list[tuple[AdaptedAxisBundle, int]] = []
    for block_index in packed.unique_block_indices:
        block = packed.blocks[int(block_index)]
        request_order.append(
            (
                bundles[block.source_id],
                int(block.representative_request_position),
            )
        )
    n_requests = len(request_order)
    shared_slots = np.fromiter(
        (slot for _bundle, slot in request_order),
        dtype=np.int64,
        count=n_requests,
    )
    # Bundle identity → compact index (shared by every sequence field).
    unique_bundles: list[AdaptedAxisBundle] = []
    bundle_id_to_index: dict[int, int] = {}
    shared_column_index = np.empty(n_requests, dtype=np.int16)
    for out_index, (bundle, _slot) in enumerate(request_order):
        key = id(bundle)
        index = bundle_id_to_index.get(key)
        if index is None:
            index = len(unique_bundles)
            bundle_id_to_index[key] = index
            unique_bundles.append(bundle)
        shared_column_index[out_index] = index

    # Contiguous same-bundle runs for vectorized ndarray fancy-index.
    bundle_groups: list[tuple[AdaptedAxisBundle, np.ndarray, np.ndarray]] = []
    if request_order:
        group_bundle = request_order[0][0]
        group_out: list[int] = []
        group_slots: list[int] = []
        for out_index, (bundle, slot) in enumerate(request_order):
            if bundle is not group_bundle:
                bundle_groups.append(
                    (
                        group_bundle,
                        np.asarray(group_out, dtype=np.int64),
                        np.asarray(group_slots, dtype=np.int64),
                    )
                )
                group_bundle = bundle
                group_out = []
                group_slots = []
            group_out.append(out_index)
            group_slots.append(slot)
        bundle_groups.append(
            (
                group_bundle,
                np.asarray(group_out, dtype=np.int64),
                np.asarray(group_slots, dtype=np.int64),
            )
        )

    request_values: dict[str, Any] = {}
    # Classify once from unique bundles instead of re-scanning request_order
    # per feature name (160+ request features × hundreds of requests).
    sequence_name_set: set[str] = set()
    request_feature_name_set: set[str] = set()
    for bundle in unique_bundles:
        sequence_name_set.update(bundle.sequence_features)
        request_feature_name_set.update(bundle.request_features)
    for name in request_name_list:
        if name in sequence_name_set:
            sample_column = None
            for bundle in unique_bundles:
                if name in bundle.sequence_features:
                    sample_column = bundle.sequence_features[name]
                    break
            if isinstance(sample_column, CompactListColumn):
                request_values[name] = SequenceColumnBatch(
                    columns=tuple(
                        bundle.sequence_features[name] for bundle in unique_bundles
                    ),
                    slots=shared_slots,
                    column_index=shared_column_index,
                )
                continue
            rows = [
                bundle.sequence_features[name][slot]
                for bundle, slot in request_order
            ]
            request_values[name] = tuple(rows)
            continue
        if name in request_feature_name_set:
            sample_column = None
            for bundle in unique_bundles:
                if name in bundle.request_features:
                    sample_column = bundle.request_features[name]
                    break
            if isinstance(sample_column, CompactListColumn):
                request_values[name] = SequenceColumnBatch(
                    columns=tuple(
                        bundle.request_features[name] for bundle in unique_bundles
                    ),
                    slots=shared_slots,
                    column_index=shared_column_index,
                )
                continue
            if (
                isinstance(sample_column, np.ndarray)
                and sample_column.ndim == 1
                and sample_column.dtype != object
            ):
                out = np.empty(n_requests, dtype=sample_column.dtype)
                for bundle, out_idx, slots_arr in bundle_groups:
                    out[out_idx] = bundle.request_features[name][slots_arr]
                request_values[name] = out
                continue
            rows = [
                bundle.request_features[name][slot] for bundle, slot in request_order
            ]
            request_values[name] = tuple(rows)
            continue
        if request_id_column is not None and name == request_id_column:
            sample_ids = unique_bundles[0].request_ids
            if (
                isinstance(sample_ids, np.ndarray)
                and sample_ids.ndim == 1
                and sample_ids.dtype != object
            ):
                out = np.empty(n_requests, dtype=sample_ids.dtype)
                for bundle, out_idx, slots_arr in bundle_groups:
                    out[out_idx] = np.asarray(bundle.request_ids)[slots_arr]
                request_values[name] = out
            else:
                request_values[name] = tuple(
                    bundle.request_ids[slot] for bundle, slot in request_order
                )
            continue
        raise KeyError(f"request column {name!r} missing from axis bundle")

    # Candidate gather: shared source index/slots for CompactListColumn bags;
    # dense scalars fancy-index+concat; broadcast/object stay per-row.
    block_positions = [
        np.asarray(block.active_candidate_positions(), dtype=np.int64)
        for block in packed.blocks
    ]
    unique_source_ids: list[int] = []
    source_to_unique: dict[int, int] = {}
    for block in packed.blocks:
        source_id = block.source_id
        if source_id not in source_to_unique:
            source_to_unique[source_id] = len(unique_source_ids)
            unique_source_ids.append(source_id)
    n_candidates = int(sum(block.candidate_count for block in packed.blocks))
    shared_candidate_slots = np.empty(n_candidates, dtype=np.int64)
    shared_candidate_index = np.empty(n_candidates, dtype=np.int16)
    cursor = 0
    for block, positions in zip(packed.blocks, block_positions):
        count = int(positions.size)
        shared_candidate_index[cursor : cursor + count] = source_to_unique[
            block.source_id
        ]
        shared_candidate_slots[cursor : cursor + count] = positions
        cursor += count

    def _candidate_column(bundle: AdaptedAxisBundle, name: str) -> Any:
        if name in bundle.item_features:
            return bundle.item_features[name]
        if name in bundle.label_features:
            return bundle.label_features[name]
        if name in bundle.label_mask_features:
            return bundle.label_mask_features[name]
        if name in bundle.candidate_metadata:
            return bundle.candidate_metadata[name]
        raise KeyError(name)

    # Resolve each candidate name to its owning map once (item/label/...).
    candidate_maps: dict[str, str] = {}
    list_batch_names: list[str] = []
    dense_names: list[str] = []
    object_names: list[str] = []
    for name in candidate_name_list:
        if name in broadcast_set:
            object_names.append(name)
            continue
        sample_column = None
        map_name = None
        for source_id in unique_source_ids:
            bundle = bundles[source_id]
            if name in bundle.item_features:
                sample_column = bundle.item_features[name]
                map_name = "item_features"
            elif name in bundle.label_features:
                sample_column = bundle.label_features[name]
                map_name = "label_features"
            elif name in bundle.label_mask_features:
                sample_column = bundle.label_mask_features[name]
                map_name = "label_mask_features"
            elif name in bundle.candidate_metadata:
                sample_column = bundle.candidate_metadata[name]
                map_name = "candidate_metadata"
            else:
                continue
            break
        if map_name is None or sample_column is None:
            object_names.append(name)
            continue
        candidate_maps[name] = map_name
        if isinstance(sample_column, CompactListColumn):
            list_batch_names.append(name)
        elif (
            isinstance(sample_column, np.ndarray)
            and sample_column.ndim == 1
            and sample_column.dtype != object
        ):
            dense_names.append(name)
        else:
            object_names.append(name)

    candidate_values: dict[str, Any] = {}
    for name in list_batch_names:
        map_name = candidate_maps[name]
        candidate_values[name] = SequenceColumnBatch(
            columns=tuple(
                getattr(bundles[source_id], map_name)[name]
                for source_id in unique_source_ids
            ),
            slots=shared_candidate_slots,
            column_index=shared_candidate_index,
        )

    # Pre-resolve dense columns per source to avoid per-block dict walks.
    # Gather via shared_candidate_slots/index (one fancy-index per source), not
    # per pack block — packs here are often ~1 candidate/request, so the old
    # per-block path did hundreds of size-1 takes + concatenate.
    dense_by_source: list[dict[str, np.ndarray]] = []
    for source_id in unique_source_ids:
        bundle = bundles[source_id]
        dense_by_source.append(
            {
                name: getattr(bundle, candidate_maps[name])[name]
                for name in dense_names
            }
        )
    reqs_by_unique = [
        np.flatnonzero(shared_candidate_index == unique_idx)
        for unique_idx in range(len(dense_by_source))
    ]
    for name in dense_names:
        sample_dtype = None
        for cols in dense_by_source:
            column = cols.get(name)
            if column is not None:
                sample_dtype = column.dtype
                break
        if sample_dtype is None:
            candidate_values[name] = np.empty(0, dtype=np.int64)
            continue
        out = np.empty(n_candidates, dtype=sample_dtype)
        for unique_idx, cols in enumerate(dense_by_source):
            reqs = reqs_by_unique[unique_idx]
            if reqs.size == 0:
                continue
            out[reqs] = cols[name][
                shared_candidate_slots[reqs].astype(np.int64, copy=False)
            ]
        candidate_values[name] = out

    object_rows: dict[str, list[Any]] = {name: [] for name in object_names}
    for block, positions in zip(packed.blocks, block_positions):
        bundle = bundles[block.source_id]
        request_slot = int(block.representative_request_position)
        for name in object_names:
            if name in broadcast_set:
                if name in bundle.request_features:
                    value = bundle.request_features[name][request_slot]
                elif name in bundle.sequence_features:
                    value = bundle.sequence_features[name][request_slot]
                elif request_id_column is not None and name == request_id_column:
                    value = bundle.request_ids[request_slot]
                else:
                    raise KeyError(
                        f"candidate column {name!r} missing from axis bundle "
                        f"source {block.source_id}"
                    )
                object_rows[name].extend([value] * int(positions.size))
                continue
            map_name = candidate_maps.get(name)
            column = (
                getattr(bundle, map_name)[name]
                if map_name is not None
                else _candidate_column(bundle, name)
            )
            if isinstance(column, np.ndarray) and column.ndim == 1:
                object_rows[name].extend(column[positions].tolist())
            else:
                for slot in positions:
                    object_rows[name].append(column[int(slot)])
    for name in object_names:
        candidate_values[name] = tuple(object_rows[name])

    sequence_plans = {
        sequence.name: build_axis_sequence_selection_plan(
            sequence,
            packed=packed,
            bundles=bundles,
        )
        for sequence in sequences
    }
    if any(len(values) != n_requests for values in request_values.values()):
        raise RuntimeError("packed request-axis column lengths are inconsistent")
    if any(len(values) != n_candidates for values in candidate_values.values()):
        raise RuntimeError("packed candidate-axis column lengths are inconsistent")

    return PreparedAxisBatch(
        request_values=request_values,
        candidate_values=candidate_values,
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


@dataclass(frozen=True)
class AggAxisPlan:
    """Control indices only: payload stays in the source Arrow table.

    Built from ``context_indices`` / ``target_indices`` / request-id /
    ``{ups}_x_indices``. Feature values are gathered at pack time.
    """

    n_candidates: int
    n_requests: int
    request_ids: tuple[Any, ...]
    candidate_to_request: Any
    request_raw_rows: Any
    request_local_positions: Any
    candidate_raw_rows: Any
    candidate_locals: Any
    # ups -> per-request token position arrays (already max_length truncated)
    ups_token_positions: Mapping[str, tuple[np.ndarray, ...]]
    # Derived candidate_position ordinals (within request), if configured
    candidate_positions: Any | None = None


@dataclass(frozen=True)
class ArrowAxisSource:
    """Raw Arrow table + compact axis plan (direct_arrow producer payload)."""

    table: Any
    plan: AggAxisPlan
    # Column role maps mirrored from the adapter plan (names are output names).
    request_feature_columns: tuple[str, ...]
    sequence_feature_columns: tuple[str, ...]
    item_feature_columns: tuple[str, ...]
    label_feature_columns: tuple[str, ...]
    label_mask_feature_columns: tuple[str, ...]
    candidate_metadata_columns: tuple[str, ...]
    bag_features: frozenset[str]
    # Physical name resolution: output/canonical -> present table column
    physical_columns: Mapping[str, str]
    # Sequence / time-delta gather hints
    ups_types: tuple[str, ...]
    sequence_columns_by_type: Mapping[str, tuple[str, ...]]
    time_delta_outputs: Mapping[str, str]
    time_delta_transform: str
    request_time_column: str
    request_id_column: str
    # Request-level transforms applied at pack gather
    request_maps: Mapping[str, Mapping[Any, int]]
    coarse_scene: Any | None
    label_masks: Mapping[str, str]
    label_missing_values: Mapping[str, tuple[Any, ...]]
    labels: Mapping[str, str]
    trusted_input: bool = True
    # Mutable once-per-source materialization cache (pack reuses like python direct).
    _cache: dict[str, Any] = field(
        default_factory=dict, repr=False, compare=False, hash=False
    )

    @property
    def n_candidates(self) -> int:
        return self.plan.n_candidates

    @property
    def n_requests(self) -> int:
        return self.plan.n_requests

    @property
    def request_ids(self) -> tuple[Any, ...]:
        return self.plan.request_ids

    @property
    def candidate_to_request(self) -> Any:
        return self.plan.candidate_to_request

    @property
    def request_raw_rows(self) -> Any:
        return self.plan.request_raw_rows


def _combine_array(column: Any) -> Any:
    if hasattr(column, "num_chunks"):
        if column.num_chunks == 1:
            return column.chunk(0)
        return column.combine_chunks()
    return column


def _list_row_values(array: Any, row_index: int) -> Any:
    """Return the values slice for one list cell (may be empty)."""

    pa, _pc = _require_pyarrow()
    if pa.types.is_null(array.type):
        return None
    if not (pa.types.is_list(array.type) or pa.types.is_large_list(array.type)):
        raise TypeError(f"expected list array, got {array.type}")
    if array[row_index].is_valid is False:
        return None
    start = int(array.offsets[row_index].as_py())
    stop = int(array.offsets[row_index + 1].as_py())
    return array.values.slice(start, stop - start)


def _gather_request_axis_cells(
    array: Any,
    raw_rows: Sequence[int],
    locals_: Sequence[int],
    *,
    bag: bool,
) -> list[Any]:
    """Gather request-axis list cells using numpy offsets (batched)."""

    pa, _pc = _require_pyarrow()
    array = _combine_array(array)
    n = len(raw_rows)
    if n == 0:
        return []
    out: list[Any] = [None] * n
    if pa.types.is_null(array.type):
        fill = [] if bag else None
        return [fill for _ in range(n)]

    offsets = array.offsets.to_numpy(zero_copy_only=False)
    values = array.values
    nested = pa.types.is_list(values.type) or pa.types.is_large_list(values.type)
    inner_offsets = (
        values.offsets.to_numpy(zero_copy_only=False) if nested else None
    )
    flat = values.values if nested else None

    for i, (raw_row, local) in enumerate(zip(raw_rows, locals_)):
        row = int(raw_row)
        loc = int(local)
        if not array[row].is_valid:
            out[i] = [] if bag else None
            continue
        start = int(offsets[row])
        stop = int(offsets[row + 1])
        if nested:
            cell = start + loc
            if cell >= stop:
                raise ValueError(
                    f"request local {loc} out of range for length {stop - start}"
                )
            s = int(inner_offsets[cell])
            e = int(inner_offsets[cell + 1])
            if bag:
                out[i] = flat.slice(s, e - s).to_pylist()
            elif e == s:
                out[i] = None
            elif e == s + 1:
                out[i] = flat[s].as_py()
            else:
                raise ValueError(
                    f"single-valued feature has inner length {e - s}"
                )
        else:
            if loc >= (stop - start):
                raise ValueError(
                    f"request local {loc} out of range for length {stop - start}"
                )
            scalar = values[start + loc].as_py()
            if bag:
                if scalar is None:
                    out[i] = []
                elif isinstance(scalar, list):
                    out[i] = scalar
                else:
                    out[i] = [scalar]
            elif isinstance(scalar, list):
                if not scalar:
                    out[i] = None
                elif len(scalar) == 1:
                    out[i] = scalar[0]
                else:
                    raise ValueError(
                        f"single-valued feature has inner length {len(scalar)}"
                    )
            else:
                out[i] = scalar
    return out


def _gather_candidate_axis_cells(
    array: Any,
    raw_rows: Sequence[int],
    locals_: Sequence[int],
    *,
    bag: bool,
) -> list[Any]:
    """Gather candidate-axis cells via Arrow offsets (batched by raw row)."""

    pa, _pc = _require_pyarrow()
    array = _combine_array(array)
    n = len(raw_rows)
    if n == 0:
        return []
    out: list[Any] = [None] * n
    if pa.types.is_null(array.type):
        fill = [] if bag else None
        return [fill for _ in range(n)]

    offsets = array.offsets.to_numpy(zero_copy_only=False)
    values = array.values
    nested = pa.types.is_list(values.type) or pa.types.is_large_list(values.type)

    by_row: dict[int, list[tuple[int, int]]] = {}
    for i, (raw_row, local) in enumerate(zip(raw_rows, locals_)):
        by_row.setdefault(int(raw_row), []).append((i, int(local)))

    if nested:
        inner_offsets = values.offsets.to_numpy(zero_copy_only=False)
        flat = values.values
        for row, items in by_row.items():
            if not array[row].is_valid:
                fill = [] if bag else None
                for out_i, _local in items:
                    out[out_i] = fill
                continue
            start = int(offsets[row])
            stop = int(offsets[row + 1])
            for out_i, loc in items:
                cell = start + loc
                if cell >= stop:
                    raise ValueError(
                        f"candidate local {loc} out of range for length {stop - start}"
                    )
                s = int(inner_offsets[cell])
                e = int(inner_offsets[cell + 1])
                if bag:
                    out[out_i] = flat.slice(s, e - s).to_pylist()
                elif e == s:
                    out[out_i] = None
                elif e == s + 1:
                    out[out_i] = flat[s].as_py()
                else:
                    raise ValueError(
                        f"single-valued feature has inner length {e - s}"
                    )
        return out

    for row, items in by_row.items():
        if not array[row].is_valid:
            fill = [] if bag else None
            for out_i, _local in items:
                out[out_i] = fill
            continue
        start = int(offsets[row])
        stop = int(offsets[row + 1])
        for out_i, loc in items:
            if loc >= (stop - start):
                raise ValueError(
                    f"candidate local {loc} out of range for length {stop - start}"
                )
            scalar = values[start + loc].as_py()
            if bag:
                if scalar is None:
                    out[out_i] = []
                elif isinstance(scalar, list):
                    out[out_i] = scalar
                else:
                    out[out_i] = [scalar]
            else:
                out[out_i] = scalar
    return out


def _take_sequence_tokens(
    array: Any,
    raw_row: int,
    positions: np.ndarray,
) -> list[Any]:
    """Shared UPS selection: one ``pc.take`` per packed request field."""

    pa, pc = _require_pyarrow()
    array = _combine_array(array)
    if pa.types.is_null(array.type) or len(positions) == 0:
        return []
    row_values = _list_row_values(array, int(raw_row))
    if row_values is None:
        return []
    # Flatten validated singleton S-token wrappers: list<list<T>> -> flat T.
    if pa.types.is_list(row_values.type) or pa.types.is_large_list(row_values.type):
        # Prefer list_flatten when every cell is a singleton / null.
        flat = pc.list_flatten(row_values)
        # list_flatten drops nulls' contribution oddly for empty; fall back.
        if len(flat) == len(row_values):
            row_values = flat
        else:
            # Uneven inner lengths: materialize selected positions only.
            selected = [
                row_values[int(pos)].as_py() for pos in positions.tolist()
            ]
            return [
                item[0]
                if isinstance(item, list) and len(item) == 1
                else (None if item == [] else item)
                for item in selected
            ]
    taken = pc.take(row_values, pa.array(positions, type=pa.int64()))
    return taken.to_pylist()


def _vectorized_time_deltas(
    event_times: Sequence[Any],
    request_time: Any,
    *,
    transform: str,
) -> list[float]:
    if not event_times:
        return []
    values = np.asarray(
        [0.0 if t is None else float(t) for t in event_times],
        dtype=np.float64,
    )
    # Null event times pad to 0.0 (match adapter trusted hot path).
    nulls = np.asarray([t is None for t in event_times], dtype=bool)
    deltas = float(request_time) - values
    if transform == "raw_ms":
        out = deltas
    elif transform == "seconds":
        out = deltas / 1000.0
    elif transform == "log1p_seconds":
        out = np.log1p(deltas / 1000.0)
    else:
        raise RuntimeError(f"unsupported time delta transform {transform!r}")
    out = out.astype(np.float64, copy=False)
    out[nulls] = 0.0
    return out.tolist()


def request_group_blocks_from_arrow_source(
    source: ArrowAxisSource,
    *,
    source_id: int,
    sequences: Sequence[Any] = (),
    length_bucket_metric: str = "max",
) -> tuple[RequestGroupBlock, ...]:
    """Build descriptors from an Arrow-backed axis source."""

    plan = source.plan
    if plan.n_requests == 0:
        return ()
    positions_by_slot: list[list[int]] = [[] for _ in range(plan.n_requests)]
    for candidate_index, slot in enumerate(plan.candidate_to_request):
        positions_by_slot[int(slot)].append(int(candidate_index))

    ups_by_sequence: dict[str, str] = {}
    for sequence in sequences:
        if not sequence.fields:
            continue
        if sequence.name in plan.ups_token_positions:
            ups_by_sequence[sequence.name] = sequence.name
            continue
        # Fall back: match ups prefix of the primary sequence source.
        primary = sequence.fields[0].source
        matched = None
        for ups in source.ups_types:
            if primary.startswith(f"{ups}_x_"):
                matched = ups
                break
        if matched is None:
            raise ValueError(
                f"sequence {sequence.name!r} source {primary!r} has no UPS mapping"
            )
        ups_by_sequence[sequence.name] = matched

    blocks: list[RequestGroupBlock] = []
    for stable_group_order, positions in enumerate(positions_by_slot):
        if not positions:
            raise ValueError(
                f"request slot {stable_group_order} has no candidates in arrow source"
            )
        pre_compaction: dict[str, int] = {}
        for sequence in sequences:
            if not sequence.fields:
                continue
            ups = ups_by_sequence[sequence.name]
            length = int(len(plan.ups_token_positions[ups][stable_group_order]))
            if sequence.max_length is not None:
                length = min(length, int(sequence.max_length))
            pre_compaction[sequence.name] = length
        blocks.append(
            RequestGroupBlock(
                source_id=source_id,
                raw_row_index=int(plan.request_raw_rows[stable_group_order]),
                request_id=plan.request_ids[stable_group_order],
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


def _sequence_selection_plan_from_request_values(
    sequence: Any,
    *,
    packed: PackedRequestPlan,
    request_values: Mapping[str, Sequence[Any]],
) -> SequenceSelectionPlan:
    """Truncate-then-compact over already membership-selected request lists."""

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
        primary_row = request_values[primary_source][request_index]
        list_length = 0 if primary_row is None else len(primary_row)

        for field in sequence.fields[1:]:
            aligned_row = request_values[field.source][request_index]
            aligned_length = 0 if aligned_row is None else len(aligned_row)
            if aligned_length != list_length:
                raise ValueError(
                    f"sequence {sequence.name!r} field {field.name!r} has length "
                    f"{aligned_length}, expected {list_length} for request "
                    f"{block.request_id!r}"
                )

        anchor_is_null = None
        if anchor_source is not None:
            anchor_row = request_values[anchor_source][request_index]
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
        expected_pre_length = block.pre_compaction_sequence_lengths.get(sequence.name)
        if expected_pre_length is not None and int(expected_pre_length) != pre_length:
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


def _column_array_cache(
    source: ArrowAxisSource,
    physical_names: set[str],
) -> dict[str, Any]:
    """Combine only the physical columns needed for this pack gather."""

    cache: dict[str, Any] = {}
    table = source.table
    for physical in physical_names:
        if physical in table.column_names:
            cache[physical] = _combine_array(table[physical])
    return cache


def _row_values_cached(
    array: Any,
    raw_row: int,
    row_cache: dict[int, Any],
) -> Any:
    """Cache one list-row values slice (shared by all UPS fields of that row)."""

    if raw_row in row_cache:
        return row_cache[raw_row]
    values = _list_row_values(array, raw_row)
    row_cache[raw_row] = values
    return values


def _numpy_row_cached(
    row_values: Any,
    raw_row: int,
    numpy_cache: dict[int, Any],
) -> Any | None:
    """Cache a numeric numpy view of one UPS row when possible."""

    if raw_row in numpy_cache:
        return numpy_cache[raw_row]
    if row_values is None:
        numpy_cache[raw_row] = None
        return None
    try:
        values = row_values.to_numpy(zero_copy_only=False)
    except (TypeError, ValueError):
        numpy_cache[raw_row] = None
        return None
    if getattr(values.dtype, "kind", "") not in {"i", "u", "f"}:
        numpy_cache[raw_row] = None
        return None
    numpy_cache[raw_row] = values
    return values


def _take_from_row_values(
    row_values: Any,
    positions: np.ndarray,
    *,
    raw_row: int | None = None,
    numpy_cache: dict[int, Any] | None = None,
) -> list[Any]:
    """Select UPS tokens from an already-sliced row values array."""

    pa, pc = _require_pyarrow()
    if row_values is None or len(positions) == 0:
        return []
    if pa.types.is_list(row_values.type) or pa.types.is_large_list(row_values.type):
        flat = pc.list_flatten(row_values)
        if len(flat) == len(row_values):
            row_values = flat
        else:
            selected = [row_values[int(pos)].as_py() for pos in positions.tolist()]
            return [
                item[0]
                if isinstance(item, list) and len(item) == 1
                else (None if item == [] else item)
                for item in selected
            ]
    if raw_row is not None and numpy_cache is not None:
        values = _numpy_row_cached(row_values, raw_row, numpy_cache)
        if values is not None:
            return values[positions].tolist()
    try:
        values = row_values.to_numpy(zero_copy_only=False)
        if getattr(values.dtype, "kind", "") in {"i", "u", "f"}:
            return values[positions].tolist()
    except (TypeError, ValueError):
        pass
    return pc.take(row_values, pa.array(positions, type=pa.int64())).to_pylist()


def _index_request_cell(row_cell: Any, local: int, *, bag: bool) -> Any:
    """Index one raw-row cell for a request-local position."""

    if row_cell is None:
        return [] if bag else None
    if isinstance(row_cell, (list, tuple, np.ndarray)):
        # Nested request-axis: row_cell is the per-request vector.
        if local >= len(row_cell):
            raise ValueError(
                f"request local {local} out of range for length {len(row_cell)}"
            )
        cell = row_cell[local]
    else:
        if local != 0:
            raise ValueError(
                f"scalar request-axis cell cannot index local={local}"
            )
        cell = row_cell
    if bag:
        if cell is None:
            return []
        if isinstance(cell, np.ndarray):
            return cell.tolist()
        if isinstance(cell, list):
            return cell
        return [cell]
    if cell is None:
        return None
    if isinstance(cell, np.ndarray):
        if cell.size == 0:
            return None
        if cell.shape == (1,):
            item = cell[0]
            return item.item() if isinstance(item, np.generic) else item
        return cell.tolist()
    if isinstance(cell, list):
        if not cell:
            return None
        if len(cell) != 1:
            raise ValueError(
                f"single-valued feature has inner length {len(cell)}"
            )
        return cell[0]
    if isinstance(cell, np.generic):
        return cell.item()
    return cell



def _index_candidate_cell(row_cell: Any, local: int, *, bag: bool) -> Any:
    """Index one raw-row cell for a candidate-local position."""

    if row_cell is None:
        return [] if bag else None
    if not isinstance(row_cell, (list, tuple, np.ndarray)):
        raise ValueError("candidate-axis feature must be list-valued")
    if local >= len(row_cell):
        raise ValueError(
            f"candidate local {local} out of range for length {len(row_cell)}"
        )
    cell = row_cell[local]
    if bag:
        if cell is None:
            return []
        if isinstance(cell, np.ndarray):
            return cell.tolist()
        if isinstance(cell, list):
            return cell
        return [cell]
    if cell is None:
        return None
    if isinstance(cell, np.ndarray):
        if cell.size == 0:
            return None
        if cell.shape == (1,):
            item = cell[0]
            return item.item() if isinstance(item, np.generic) else item
        return cell.tolist()
    if isinstance(cell, list):
        if not cell:
            return None
        if len(cell) != 1:
            raise ValueError(
                f"single-valued feature has inner length {len(cell)}"
            )
        return cell[0]
    if isinstance(cell, np.generic):
        return cell.item()
    return cell


def materialize_arrow_axis_source(source: ArrowAxisSource) -> AdaptedAxisBundle:
    """Convert one ArrowAxisSource to AdaptedAxisBundle once (columnar).

    Matches python ``direct`` economics: pay feature materialization once per
    retained source, then pack only copies references. Sequence columns avoid
    full-history ``to_pylist`` and keep membership ``take`` only.
    """

    cached = source._cache.get("bundle")
    if isinstance(cached, AdaptedAxisBundle):
        return cached

    from .dataloader import (
        _arrow_array_to_pylist,
        _map_request_value,
        coarse_scene_ids,
    )

    plan = source.plan
    table = source.table
    n_requests = plan.n_requests
    n_candidates = plan.n_candidates
    pa, _pc = _require_pyarrow()

    def physical(name: str) -> str:
        return source.physical_columns.get(name, name)

    def column_pylist(name: str) -> list[Any] | None:
        phys = physical(name)
        if phys not in table.column_names:
            return None
        return _arrow_array_to_pylist(pa, _combine_array(table[phys]))

    request_raw_rows = [int(v) for v in plan.request_raw_rows.tolist()]
    request_locals = [int(v) for v in plan.request_local_positions.tolist()]
    candidate_raw_rows = [int(v) for v in plan.candidate_raw_rows.tolist()]
    candidate_locals = [int(v) for v in plan.candidate_locals.tolist()]

    # --- Request scalars / bags (whole-column pylist once) ---
    request_features: dict[str, tuple[Any, ...]] = {}
    coarse = source.coarse_scene
    coarse_names = set()
    if coarse is not None:
        coarse_names = {coarse.index_column, coarse.prior_id_column}

    for name in source.request_feature_columns:
        if name in coarse_names:
            continue
        raw = column_pylist(name)
        if raw is None:
            raise KeyError(f"request column {name!r} missing from table")
        bag = name in source.bag_features
        values = [
            _index_request_cell(raw[row], loc, bag=bag)
            for row, loc in zip(request_raw_rows, request_locals)
        ]
        if name in source.request_maps:
            values = [
                _map_request_value(
                    value,
                    column=name,
                    mapping=source.request_maps[name],
                    validate_contract=not source.trusted_input,
                )
                for value in values
            ]
        request_features[name] = tuple(values)

    if coarse is not None:
        raw_scene_col = coarse.raw_scene_column
        if raw_scene_col in request_features:
            scene_values = request_features[raw_scene_col]
        else:
            raw = column_pylist(raw_scene_col)
            if raw is None:
                raise KeyError(f"coarse scene column {raw_scene_col!r} missing")
            scene_values = tuple(
                _index_request_cell(raw[row], loc, bag=False)
                for row, loc in zip(request_raw_rows, request_locals)
            )
        indexes: list[int] = []
        priors: list[int] = []
        for scene in scene_values:
            index, prior = coarse_scene_ids(
                scene,
                coarse.search_scene_ids,
                unlisted_policy=coarse.unlisted_policy,
            )
            indexes.append(index)
            priors.append(prior)
        if coarse.index_column in source.request_feature_columns:
            request_features[coarse.index_column] = tuple(indexes)
        if coarse.prior_id_column in source.request_feature_columns:
            request_features[coarse.prior_id_column] = tuple(priors)

    # --- Sequences: whole-column pylist once, then vectorized membership index ---
    sequence_features: dict[str, tuple[Any, ...]] = {}
    seq_values: dict[str, list[Any]] = {
        name: [None] * n_requests for name in source.sequence_feature_columns
    }

    req_times: list[Any] | None = None
    if source.time_delta_outputs:
        raw_times = column_pylist(source.request_time_column)
        if raw_times is None:
            raise KeyError(
                f"request time column {source.request_time_column!r} missing"
            )
        req_times = [
            _index_request_cell(raw_times[row], loc, bag=False)
            for row, loc in zip(request_raw_rows, request_locals)
        ]

    unique_request_rows = sorted(set(request_raw_rows))

    def _row_as_indexable(row: Any) -> Any:
        if row is None:
            return None
        if isinstance(row, np.ndarray):
            return row
        if not isinstance(row, (list, tuple)):
            raise ValueError("UPS column row must be list-valued")
        if not row:
            return np.empty(0, dtype=np.int64)
        first = row[0]
        if isinstance(first, (list, np.ndarray)):
            # Singleton-wrapped tokens: flatten once.
            flat = []
            for item in row:
                if item is None:
                    flat.append(None)
                elif isinstance(item, np.ndarray):
                    if item.size == 0:
                        flat.append(None)
                    elif item.shape == (1,):
                        value = item[0]
                        flat.append(
                            value.item() if isinstance(value, np.generic) else value
                        )
                    else:
                        flat.append(item)
                elif isinstance(item, list):
                    if not item:
                        flat.append(None)
                    elif len(item) == 1:
                        flat.append(item[0])
                    else:
                        flat.append(item)
                else:
                    flat.append(item)
            row = flat
            if not row or row[0] is None or isinstance(row[0], bool):
                return row
            first = row[0]
        if isinstance(first, bool) or first is None:
            return row
        if isinstance(first, (int, float, np.integer, np.floating)):
            try:
                return np.asarray(row)
            except (TypeError, ValueError):
                return row
        return row

    def _index_positions(row_obj: Any, positions: np.ndarray) -> list[Any]:
        if row_obj is None or len(positions) == 0:
            return []
        if isinstance(row_obj, np.ndarray):
            return row_obj[positions].tolist()
        return [row_obj[int(pos)] for pos in positions.tolist()]

    for ups, columns in source.sequence_columns_by_type.items():
        positions_list = plan.ups_token_positions[ups]
        for column in columns:
            if column not in seq_values:
                continue
            raw = column_pylist(column)
            if raw is None:
                for slot in range(n_requests):
                    seq_values[column][slot] = []
                continue
            row_cache = {
                row: _row_as_indexable(raw[row]) for row in unique_request_rows
            }
            for slot in range(n_requests):
                seq_values[column][slot] = _index_positions(
                    row_cache[request_raw_rows[slot]],
                    positions_list[slot],
                )

        out_name = source.time_delta_outputs.get(ups)
        if out_name is None or out_name not in seq_values:
            continue
        time_raw = column_pylist(f"{ups}_x_time")
        assert req_times is not None
        if time_raw is None:
            for slot in range(n_requests):
                seq_values[out_name][slot] = []
            continue
        row_cache = {
            row: _row_as_indexable(time_raw[row]) for row in unique_request_rows
        }
        for slot in range(n_requests):
            event_times = _index_positions(
                row_cache[request_raw_rows[slot]],
                positions_list[slot],
            )
            seq_values[out_name][slot] = _vectorized_time_deltas(
                event_times,
                req_times[slot],
                transform=source.time_delta_transform,
            )

    for name, values in seq_values.items():
        sequence_features[name] = compact_list_column_from_rows(values)
    sequence_features = share_compact_list_offsets(sequence_features)

    # --- Candidate item / label / metadata ---
    item_features: dict[str, tuple[Any, ...]] = {}
    for name in source.item_feature_columns:
        raw = column_pylist(name)
        if raw is None:
            raise KeyError(f"item column {name!r} missing from table")
        bag = name in source.bag_features
        item_features[name] = tuple(
            _index_candidate_cell(raw[row], loc, bag=bag)
            for row, loc in zip(candidate_raw_rows, candidate_locals)
        )

    label_features: dict[str, tuple[Any, ...]] = {}
    task_by_column = {column: task for task, column in source.labels.items()}
    for name in source.label_feature_columns:
        raw = column_pylist(name)
        if raw is None:
            raise KeyError(f"label column {name!r} missing from table")
        values = [
            _index_candidate_cell(raw[row], loc, bag=False)
            for row, loc in zip(candidate_raw_rows, candidate_locals)
        ]
        task = task_by_column.get(name)
        if task is not None:
            mask_column = source.label_masks.get(task)
            missing_vals = source.label_missing_values.get(task, ())
            complete = mask_column is None and not missing_vals
            if complete:
                normalized: list[Any] = []
                for value in values:
                    valid_binary = (
                        not isinstance(value, bool)
                        and isinstance(value, (int, float))
                        and float(value) in {0.0, 1.0}
                        and (not isinstance(value, float) or value == value)
                    )
                    if not valid_binary:
                        raise ValueError(
                            f"label {name!r} must be numeric 0/1; got {value!r}"
                        )
                    normalized.append(int(value))
                values = normalized
        label_features[name] = tuple(values)

    if source.label_mask_feature_columns:
        raise KeyError(
            "label mask columns are not yet materialized on direct_arrow "
            "(configure complete labels)"
        )

    candidate_metadata: dict[str, tuple[Any, ...]] = {}
    for name in source.candidate_metadata_columns:
        if (
            name == "candidate_position"
            and plan.candidate_positions is not None
        ):
            candidate_metadata[name] = tuple(
                int(v) for v in plan.candidate_positions.tolist()
            )
            continue
        raw = column_pylist(name)
        if raw is None:
            if plan.candidate_positions is not None:
                candidate_metadata[name] = tuple(
                    int(v) for v in plan.candidate_positions.tolist()
                )
            else:
                candidate_metadata[name] = tuple(None for _ in range(n_candidates))
            continue
        candidate_metadata[name] = tuple(
            _index_candidate_cell(raw[row], loc, bag=False)
            for row, loc in zip(candidate_raw_rows, candidate_locals)
        )

    bundle = AdaptedAxisBundle(
        n_candidates=n_candidates,
        n_requests=n_requests,
        request_ids=plan.request_ids,
        candidate_to_request=plan.candidate_to_request,
        request_features=request_features,
        sequence_features=sequence_features,
        item_features=item_features,
        label_features=label_features,
        label_mask_features={},
        candidate_metadata=candidate_metadata,
        request_raw_rows=plan.request_raw_rows,
        candidate_raw_rows=plan.candidate_raw_rows,
    )
    source._cache["bundle"] = bundle
    # Drop the raw table after materialization so retained sources don't keep
    # both Arrow and Python payloads for the rest of the shuffle window.
    source._cache["table_released"] = True
    object.__setattr__(source, "table", None)
    return bundle


def prepare_packed_arrow_axis_batch(
    sources: Mapping[int, ArrowAxisSource],
    packed: PackedRequestPlan,
    *,
    sequences: Sequence[Any],
    request_id_column: str | None = None,
    candidate_request_columns: Sequence[str] = (),
) -> PreparedAxisBatch:
    """Materialize each Arrow source once, then reuse python-direct pack gather.

    The previous pack-time cell/column gather re-converted Arrow→Python on every
    pack and was slower than ``direct``. Caching an ``AdaptedAxisBundle`` per
    retained source restores the same "pay once, pack by reference" model while
    still building control indices without ``_adapter_table_to_python``.
    """

    if not packed.blocks:
        raise ValueError("cannot prepare an empty packed arrow axis batch")

    bundles = {
        source_id: materialize_arrow_axis_source(source)
        for source_id, source in sources.items()
    }
    return prepare_packed_axis_batch(
        bundles,
        packed,
        sequences=sequences,
        request_id_column=request_id_column,
        candidate_request_columns=candidate_request_columns,
    )


