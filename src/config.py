"""Configuration schema, derived state, and validation for MDL.

The YAML surface should stay stable. Internal helpers build the model-facing view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
import importlib
from pathlib import Path
from typing import Any, Literal

import yaml


# Public string choices used by YAML and type hints.
EncodingType = Literal["vocab", "hash", "identity", "shared_vocab"]
EmbeddingScope = Literal["feature", "scenario", "task", "shared"]
ModelName = Literal["rankmixer", "mdl_rankmixer", "onetrans", "mdl_onetrans", "longer"]
SequenceFieldKind = Literal["categorical", "dense"]
SequenceEncoderType = Literal["raw", "attention_pool", "mean_pool", "longer"]
LRScheduleType = Literal["constant", "cosine"]
ActivationType = Literal["gelu", "relu"]
SequenceFusionType = Literal["timestamp_aware", "intent_ordered"]
RankMixerFFNType = Literal["dense", "sparse_moe"]
SequenceOrderType = Literal["oldest_to_newest", "newest_to_oldest"]
MDLFeatureInteractionType = Literal["paper", "rankmixer_full"]
DTSITrainingOutputType = Literal["dense_router", "mean"]
LossReductionType = Literal["sum", "mean_per_task"]
IdentityOutOfRangeType = Literal["error", "padding"]


def _validate_identity_bounds(
    *,
    num_buckets: int | None,
    max_id: int | None,
    padding_id: int,
    out_of_range: str,
    path: str,
) -> None:
    if num_buckets is not None and max_id is not None:
        raise ValueError(f"{path} must set num_buckets or legacy max_id, not both")
    if num_buckets is None and max_id is None:
        raise ValueError(f"{path}.num_buckets is required for identity encoding")
    resolved_buckets = num_buckets if num_buckets is not None else int(max_id) + 1
    if resolved_buckets <= 0:
        raise ValueError(f"{path}.num_buckets must be positive")
    if padding_id != 0:
        raise ValueError(f"{path}.padding_id must be 0")
    if not 0 <= padding_id < resolved_buckets:
        raise ValueError(f"{path}.padding_id must be inside [0, num_buckets)")
    if out_of_range not in {"error", "padding"}:
        raise ValueError(f"{path}.out_of_range must be error or padding")


# Raw YAML schema. These dataclasses mirror configs/default.yaml field names.
@dataclass(frozen=True)
class CategoricalEncodingConfig:
    """Inline categorical encoding config under a logical feature declaration."""

    # Accept YAML key "type" in from_mapping, but keep the internal name explicit.
    encoding: EncodingType
    min_count: int | None = None
    max_size: int | None = None
    artifact: str | None = None
    num_buckets: int | None = None
    salt: str | None = None
    # ``max_id`` is accepted only as a compatibility alias for
    # ``num_buckets=max_id+1``. New identity configurations use num_buckets.
    max_id: int | None = None
    padding_id: int = 0
    out_of_range: IdentityOutOfRangeType = "error"
    share_with: str | None = None
    share_embedding: bool = False

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "CategoricalEncodingConfig | None":
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ValueError("inline categorical encoding must be an object")
        allowed = {
            "type",
            "encoding",
            "min_count",
            "max_size",
            "artifact",
            "num_buckets",
            "salt",
            "max_id",
            "padding_id",
            "out_of_range",
            "share_with",
            "share_embedding",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError("inline categorical encoding contains unknown keys: " + ", ".join(unknown))
        explicit_type = payload.get("type")
        explicit_encoding = payload.get("encoding")
        if explicit_type is not None and explicit_encoding is not None and explicit_type != explicit_encoding:
            raise ValueError("inline categorical encoding type and encoding must match when both are set")
        encoding = explicit_type or explicit_encoding
        if encoding is None:
            raise ValueError("inline categorical encoding requires type")
        values = dict(payload)
        values.pop("type", None)
        values["encoding"] = encoding
        return cls(**values)

    def validate(self, path: str) -> None:
        if self.encoding not in {"vocab", "hash", "identity", "shared_vocab"}:
            raise ValueError(f"{path}.type is invalid")
        if self.encoding == "vocab":
            if not self.artifact:
                raise ValueError(f"{path}.artifact is required for vocab encoding")
            if self.min_count is not None and self.min_count <= 0:
                raise ValueError(f"{path}.min_count must be positive")
            if self.max_size is not None and self.max_size <= 0:
                raise ValueError(f"{path}.max_size must be positive")
        if self.encoding == "hash":
            if self.num_buckets is None or self.num_buckets <= 0:
                raise ValueError(f"{path}.num_buckets must be positive for hash encoding")
        if self.encoding == "identity":
            _validate_identity_bounds(
                num_buckets=self.num_buckets,
                max_id=self.max_id,
                padding_id=self.padding_id,
                out_of_range=self.out_of_range,
                path=path,
            )
            if self.share_embedding and not self.share_with:
                raise ValueError(
                    f"{path}.share_with is required when identity share_embedding=true"
                )
        if self.encoding == "shared_vocab" and not self.share_with:
            raise ValueError(f"{path}.share_with is required for shared_vocab encoding")


@dataclass(frozen=True)
class ParquetAdapterConfig:
    """External preprocessing hook for non-flat Parquet layouts.

    The callable is imported during config validation but is only executed by
    the data pipeline after raw Parquet has been read into Arrow tables.
    """

    callable: str
    input_columns: list[str] | None = None
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "ParquetAdapterConfig | None":
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ValueError("data split adapter must be an object")
        input_columns = payload.get("input_columns")
        if input_columns is not None:
            if isinstance(input_columns, str):
                raise ValueError("data split adapter.input_columns must be a list of column names")
            input_columns = list(input_columns)
        return cls(
            callable=payload.get("callable", ""),
            input_columns=input_columns,
            options=dict(payload.get("options", {})),
        )

    def validate(self, path: str) -> None:
        if not self.callable:
            raise ValueError(
                f"{path}.callable is required; use 'package.module:function'. "
                "The adapter must accept (pyarrow.Table, *, context=ParquetAdapterContext) "
                "and return a flat pyarrow.Table or an iterable of flat tables."
            )
        if ":" not in self.callable:
            raise ValueError(f"{path}.callable must use 'package.module:function' format")
        module_name, attribute_name = self.callable.split(":", 1)
        if not module_name or not attribute_name:
            raise ValueError(f"{path}.callable must use 'package.module:function' format")
        if self.input_columns is not None:
            if not all(isinstance(column, str) and column for column in self.input_columns):
                raise ValueError(f"{path}.input_columns must contain non-empty column names")
        if not isinstance(self.options, dict):
            raise ValueError(f"{path}.options must be an object")
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            raise ValueError(f"{path}.callable could not import module {module_name!r}: {error}") from error
        target: Any = module
        for part in attribute_name.split("."):
            if not part:
                raise ValueError(f"{path}.callable must not contain empty attribute path segments")
            try:
                target = getattr(target, part)
            except AttributeError as error:
                raise ValueError(
                    f"{path}.callable could not find attribute {attribute_name!r} "
                    f"in module {module_name!r}"
                ) from error
        if not callable(target):
            raise ValueError(f"{path}.callable target {self.callable!r} is not callable")


@dataclass(frozen=True)
class LengthBucketConfig:
    """One upper-bounded sequence-length bucket and its per-rank batch size."""

    max_length: int | None
    batch_size: int

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "LengthBucketConfig":
        if not isinstance(payload, dict):
            raise ValueError("reader.length_buckets entries must be objects")
        return cls(**payload)

    def validate(self) -> None:
        if self.max_length is not None and self.max_length <= 0:
            raise ValueError("reader.length_buckets.max_length must be positive or null")
        if self.batch_size <= 0:
            raise ValueError("reader.length_buckets.batch_size must be positive")


@dataclass(frozen=True)
class ReaderConfig:
    """Parquet reader options for one data split.

    The scanner currently has one supported backend. The fields below still stay
    configurable because secure environments may need different threading,
    batching, or sharding behavior without changing data code.
    """

    # Keep the backend explicit so adding another reader later is a schema change.
    engine: str = "pyarrow_dataset"
    # When true, the scanner reads only columns required by features and labels.
    columns_pruning: bool = True
    # PyArrow uses this to size CPU and IO thread pools. Zero means default behavior.
    num_workers: int = 0
    # Dataset scanner readahead. Larger values can help throughput but use more memory.
    prefetch_batches: int = 2
    # Hard queue budget per rank. Count and bytes are both enforced.
    max_prefetch_bytes: int = 512 * 1024 * 1024
    # DataLoader pinning is only useful when batches are later moved to CUDA.
    pin_memory: bool = False
    # DDP can shard by files, row groups, or record batches depending on data layout.
    shard_unit: Literal["file", "row_group", "record_batch"] = "row_group"
    # Optional Arrow scanner batch size. Training batch size is independent.
    scanner_batch_rows: int | None = None
    # Optional vectorized length buckets. Null max_length is the final catch-all.
    length_buckets: tuple[LengthBucketConfig, ...] = ()

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "ReaderConfig":
        if payload is None:
            return cls()
        values = dict(payload)
        # Keep removed YAML keys explicit so old configs fail with a useful message.
        if "batch_size_candidates" in values:
            raise ValueError("reader.batch_size_candidates was removed; use training.batch_size")
        if "batch_size_rows" in values:
            if "scanner_batch_rows" in values:
                raise ValueError(
                    "reader must set scanner_batch_rows, not both scanner_batch_rows "
                    "and legacy batch_size_rows"
                )
            values["scanner_batch_rows"] = values.pop("batch_size_rows")
        values["length_buckets"] = tuple(
            LengthBucketConfig.from_mapping(item)
            for item in values.get("length_buckets", ())
        )
        return cls(**values)

    def validate(self) -> None:
        if self.engine != "pyarrow_dataset":
            raise ValueError("reader.engine must be 'pyarrow_dataset'")
        if self.num_workers < 0:
            raise ValueError("reader.num_workers must be non-negative")
        if self.prefetch_batches < 0:
            raise ValueError("reader.prefetch_batches must be non-negative")
        if self.max_prefetch_bytes <= 0:
            raise ValueError("reader.max_prefetch_bytes must be positive")
        if self.shard_unit not in {"file", "row_group", "record_batch"}:
            raise ValueError("reader.shard_unit must be file, row_group, or record_batch")
        if self.scanner_batch_rows is not None and self.scanner_batch_rows <= 0:
            raise ValueError("reader.scanner_batch_rows must be positive")
        previous = 0
        saw_catch_all = False
        for index, bucket in enumerate(self.length_buckets):
            bucket.validate()
            if saw_catch_all:
                raise ValueError(
                    "reader.length_buckets catch-all entry must be last"
                )
            if bucket.max_length is None:
                saw_catch_all = True
            elif bucket.max_length <= previous:
                raise ValueError(
                    "reader.length_buckets max_length values must be strictly increasing"
                )
            else:
                previous = bucket.max_length
            if index == len(self.length_buckets) - 1 and not saw_catch_all:
                raise ValueError(
                    "reader.length_buckets requires a final max_length: null catch-all"
                )


@dataclass(frozen=True)
class SchemaPolicy:
    """Rules for schema compatibility across parquet inputs.

    These options describe the intended contract, but only the strict path is
    implemented today. Unsupported relaxed modes fail fast during validation.
    """

    # All files in a split must expose the same Arrow schema.
    require_same_schema: bool = True
    # Kept for future schema evolution support; not implemented in the scanner.
    allow_missing_nullable_columns: bool = False
    # Validate schemas before training so data issues fail before model startup.
    validate_before_train: bool = True

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "SchemaPolicy":
        if payload is None:
            return cls()
        return cls(**payload)

    def validate(self) -> None:
        if self.require_same_schema is not True:
            raise ValueError(
                "data.schema_policy.require_same_schema=false is not supported by the parquet scanner"
            )
        if self.allow_missing_nullable_columns:
            raise ValueError(
                "data.schema_policy.allow_missing_nullable_columns is not implemented; "
                "use a dataset-specific adapter for schema evolution"
            )
        if self.validate_before_train is not True:
            raise ValueError(
                "data.schema_policy.validate_before_train=false is not supported by the parquet scanner"
            )


@dataclass(frozen=True)
class ParquetSplitConfig:
    """Input paths and label columns for a train or test split.

    A split is intentionally file/path based. It does not know about feature
    semantics; required columns are computed later from AppConfig.
    """

    # flat_parquet is the identity/default contract; adapter_parquet calls an
    # external preprocessing function before feature encoding.
    format: Literal["flat_parquet", "adapter_parquet"]
    # Paths can be files, directories, or glob patterns. A single string is accepted.
    inputs: list[str]
    # Reader options can differ per split, for example smaller test batches.
    reader: ReaderConfig = field(default_factory=ReaderConfig)
    # External preprocessing hook used only when format=adapter_parquet.
    adapter: ParquetAdapterConfig | None = None
    # Request id is used for request-level caching and optional grouping.
    request_id: str | None = None
    # Group id is used by evaluation/prediction code when preserving request groups.
    group_id: str | None = None
    # Mapping from task name to label column. Train must declare at least one task.
    labels: dict[str, str] = field(default_factory=dict)
    # Optional per-task mask columns. If present, masks must match labels exactly.
    label_masks: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ParquetSplitConfig":
        if not isinstance(payload, dict):
            raise ValueError("data split config must be an object")
        if "agg_layout" in payload:
            raise ValueError(
                "data split agg_layout was removed; implement a dataset-specific agg parquet adapter "
                "after the real schema is available"
            )
        reader = ReaderConfig.from_mapping(payload.get("reader"))
        inputs = payload.get("inputs")
        # YAML authors often use one path during smoke tests; normalize to a list.
        if isinstance(inputs, str):
            inputs = [inputs]
        return cls(
            format=payload["format"],
            inputs=list(inputs or []),
            reader=reader,
            adapter=ParquetAdapterConfig.from_mapping(payload.get("adapter")),
            request_id=payload.get("request_id"),
            group_id=payload.get("group_id"),
            labels=dict(payload.get("labels", {})),
            label_masks=dict(payload.get("label_masks", {})),
        )

    def validate(self, name: str) -> None:
        if self.format not in {"flat_parquet", "adapter_parquet"}:
            raise ValueError(
                f"data.{name}.format must be flat_parquet or adapter_parquet; "
                "use adapter_parquet with data.{name}.adapter.callable for non-flat Parquet layouts"
            )
        if self.format == "flat_parquet" and self.adapter is not None:
            raise ValueError(
                f"data.{name}.adapter is only valid when data.{name}.format=adapter_parquet"
            )
        if self.format == "adapter_parquet":
            if self.adapter is None:
                raise ValueError(
                    f"data.{name}.format=adapter_parquet requires data.{name}.adapter.callable. "
                    "Configure an external adapter such as 'package.module:function' that converts "
                    "raw Parquet Arrow tables to the flat_parquet contract."
                )
            self.adapter.validate(f"data.{name}.adapter")
        if not self.inputs:
            raise ValueError(f"data.{name}.inputs must contain at least one path, glob, or directory")
        self.reader.validate()
        if name == "train" and not self.labels:
            raise ValueError("data.train.labels must declare at least one task label")
        if self.label_masks:
            if not self.labels:
                raise ValueError(f"data.{name}.label_masks requires data.{name}.labels")
            label_names = set(self.labels)
            mask_names = set(self.label_masks)
            missing = sorted(label_names - mask_names)
            unknown = sorted(mask_names - label_names)
            details = []
            if missing:
                details.append("missing masks for labels: " + ", ".join(missing))
            if unknown:
                details.append("unknown label masks: " + ", ".join(unknown))
            if details:
                raise ValueError(f"data.{name}.label_masks must match labels exactly; " + "; ".join(details))


@dataclass(frozen=True)
class DataConfig:
    """Dataset splits and schema policy for parquet loading."""

    train: ParquetSplitConfig
    test: ParquetSplitConfig | None = None
    schema_policy: SchemaPolicy = field(default_factory=SchemaPolicy)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "DataConfig":
        if not isinstance(payload, dict):
            raise ValueError("data must be an object")
        return cls(
            train=ParquetSplitConfig.from_mapping(payload["train"]),
            test=ParquetSplitConfig.from_mapping(payload["test"]) if "test" in payload else None,
            schema_policy=SchemaPolicy.from_mapping(payload.get("schema_policy")),
        )

    def validate(self) -> None:
        self.train.validate("train")
        if self.test is not None:
            self.test.validate("test")
        self.schema_policy.validate()


@dataclass(frozen=True)
class FeatureConfig:
    """Logical model input from one parquet column.

    Sequence inputs are not represented here anymore. New behavior histories use
    top-level SequenceConfig so multi-field steps are explicit and validated.
    """

    # Stable logical name used by tokenization, vocab strategy, and model code.
    name: str
    # Categorical values are encoded through vocab/hash/identity; dense values are floats.
    kind: Literal["categorical", "dense"]
    # Physical parquet column name. It may differ from the logical feature name.
    source: str
    # Optional source dtype hint. Tensorization currently infers the concrete torch dtype.
    dtype: str | None = None
    # Controls which token families can consume this input.
    embedding_scope: EmbeddingScope = "feature"
    # Dense vector width. Categorical scalar features must keep dimension 1.
    dimension: int = 1
    # Optional per-feature embedding width; defaults to model.embedding_dim.
    embedding_dim: int | None = None
    # Optional inline categorical encoding. Mutually exclusive with vocab_strategy.features[name].
    encoding: CategoricalEncodingConfig | None = None
    # Legacy scalar-sequence fields remain rejected; keep these only for YAML compatibility.
    max_length: int | None = None
    truncation: Literal["head", "tail"] = "tail"

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "FeatureConfig":
        values = dict(payload)
        values["encoding"] = CategoricalEncodingConfig.from_mapping(values.get("encoding"))
        return cls(**values)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("feature.name is required")
        if self.kind not in {"categorical", "dense"}:
            raise ValueError(
                f"feature {self.name!r} kind must be categorical or dense; "
                "use top-level sequences for sequence inputs"
            )
        if not self.source:
            raise ValueError(f"feature {self.name!r} source is required")
        if self.embedding_scope not in {"feature", "scenario", "task", "shared"}:
            raise ValueError(
                f"feature {self.name!r} embedding_scope must be feature, scenario, task, or shared"
            )
        if self.dimension <= 0:
            raise ValueError(f"feature {self.name!r} dimension must be positive")
        if self.kind != "dense" and self.dimension != 1:
            raise ValueError(f"feature {self.name!r} dimension is only supported for dense features")
        if self.embedding_dim is not None and self.embedding_dim <= 0:
            raise ValueError(f"feature {self.name!r} embedding_dim must be positive")
        if self.kind == "dense" and self.embedding_dim is not None:
            raise ValueError(f"feature {self.name!r} embedding_dim is only supported for categorical features")
        if self.kind != "categorical" and self.encoding is not None:
            raise ValueError(f"feature {self.name!r} encoding is only supported for categorical features")
        if self.encoding is not None:
            self.encoding.validate(f"features.{self.name}.encoding")
        if self.max_length is not None and self.max_length <= 0:
            raise ValueError(f"feature {self.name!r} max_length must be positive")
        if self.truncation not in {"head", "tail"}:
            raise ValueError(f"feature {self.name!r} truncation must be head or tail")


@dataclass(frozen=True)
class SequenceFieldConfig:
    """One field inside a multi-field behavior sequence.

    All fields in a SequenceConfig must be aligned list columns with the same
    row-level length. Tensorization checks that alignment before model use.
    """

    # Field name is local to the sequence; qualified name is sequence.field.
    name: str
    # Each step field is either categorical id-like data or dense side information.
    kind: SequenceFieldKind
    # Physical list column containing this field for every request row.
    source: str
    dtype: str | None = None
    # Dense field width at each sequence step. Categorical fields must be scalar.
    dimension: int = 1
    # Optional categorical embedding width for this sequence field.
    embedding_dim: int | None = None
    # Optional inline categorical encoding. Mutually exclusive with vocab_strategy.features[sequence.field].
    encoding: CategoricalEncodingConfig | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "SequenceFieldConfig":
        values = dict(payload)
        values["encoding"] = CategoricalEncodingConfig.from_mapping(values.get("encoding"))
        return cls(**values)

    def qualified_name(self, sequence_name: str) -> str:
        # Vocab strategy keys use this fully qualified name for sequence fields.
        return f"{sequence_name}.{self.name}"

    def validate(self, sequence_name: str) -> None:
        if not self.name:
            raise ValueError(f"sequence {sequence_name!r} field.name is required")
        if "." in self.name:
            raise ValueError(f"sequence {sequence_name!r} field name must not contain '.'")
        if self.kind not in {"categorical", "dense"}:
            raise ValueError(
                f"sequence {sequence_name!r} field {self.name!r} kind must be categorical or dense"
            )
        if not self.source:
            raise ValueError(f"sequence {sequence_name!r} field {self.name!r} source is required")
        if self.dimension <= 0:
            raise ValueError(f"sequence {sequence_name!r} field {self.name!r} dimension must be positive")
        if self.kind == "categorical" and self.dimension != 1:
            raise ValueError(
                f"sequence {sequence_name!r} categorical field {self.name!r} must have dimension 1"
            )
        if self.embedding_dim is not None and self.embedding_dim <= 0:
            raise ValueError(f"sequence {sequence_name!r} field {self.name!r} embedding_dim must be positive")
        if self.kind == "dense" and self.embedding_dim is not None:
            raise ValueError(
                f"sequence {sequence_name!r} dense field {self.name!r} must not set embedding_dim"
            )
        if self.kind != "categorical" and self.encoding is not None:
            raise ValueError(
                f"sequence {sequence_name!r} field {self.name!r} encoding is only supported for categorical fields"
            )
        if self.encoding is not None:
            self.encoding.validate(f"sequences.{sequence_name}.fields.{self.name}.encoding")


@dataclass(frozen=True)
class SequenceConfig:
    """A top-level behavior sequence made from aligned list columns.

    The sequence encoder converts variable-length behavior steps into one or
    more fixed-width summary tokens that downstream tokenizers can consume.
    """

    # Logical sequence name. Also used as the encoded input name after pooling.
    name: str
    # Step fields. Every row must have equal list lengths across all fields.
    fields: list[SequenceFieldConfig]
    # Feature/shared sequences can become model feature tokens. Other scopes are reserved.
    embedding_scope: EmbeddingScope = "feature"
    # Optional maximum and physical head/tail window kept during tensorization.
    max_length: int | None = None
    truncation: Literal["head", "tail"] = "tail"
    # Physical order of valid events in every configured list column. Model code
    # canonicalizes both choices to oldest_to_newest before causal attention.
    sequence_order: SequenceOrderType = "oldest_to_newest"
    # raw leaves event-level modeling to OneTrans. attention_pool and mean_pool
    # are summary baselines; longer is the paper-aligned standalone encoder path.
    encoder: SequenceEncoderType = "attention_pool"
    # Scalar features used as target context for LONGER query construction.
    target_inputs: list[str] = field(default_factory=list)
    # Number of fixed summary slices exposed to RankMixer token packing.
    rankmixer_summary_tokens: int = 1
    # LONGER-specific query/self-attention parameters. Ignored by simpler encoders.
    longer_query_tokens: int = 32
    longer_self_layers: int = 1
    longer_token_merge: int = 1
    longer_inner_layers: int = 0
    # Cacheable user/CLS globals are kept separate from candidate globals so
    # candidate information cannot leak into reusable sequence-side states.
    longer_user_global_inputs: list[str] = field(default_factory=list)
    longer_user_global_tokens: int = 0
    longer_cls_tokens: int = 0
    # None assigns all remaining rankmixer_summary_tokens to target_inputs.
    longer_candidate_global_tokens: int | None = None
    # Explicit temporal semantics used by OneTrans and LONGER.
    timestamp_field: str | None = None
    time_delta_field: str | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "SequenceConfig":
        # Sequences now use field-level list columns only. Reject old layout keys early.
        removed = sorted(set(payload) & {"layout", "source"})
        if removed:
            raise ValueError(
                "sequence layout/source handling was removed; expose sequence fields "
                "as list columns and configure them with fields[].source instead: "
                + ", ".join(removed)
            )
        return cls(
            name=payload["name"],
            fields=[
                SequenceFieldConfig.from_mapping(item)
                for item in payload.get("fields", [])
            ],
            embedding_scope=payload.get("embedding_scope", "feature"),
            max_length=payload.get("max_length"),
            truncation=payload.get("truncation", "tail"),
            sequence_order=payload.get("sequence_order", "oldest_to_newest"),
            encoder=payload.get("encoder", "attention_pool"),
            target_inputs=list(payload.get("target_inputs", [])),
            rankmixer_summary_tokens=payload.get("rankmixer_summary_tokens", 1),
            longer_query_tokens=payload.get("longer_query_tokens", 32),
            longer_self_layers=payload.get("longer_self_layers", 1),
            longer_token_merge=payload.get("longer_token_merge", 1),
            longer_inner_layers=payload.get("longer_inner_layers", 0),
            longer_user_global_inputs=list(payload.get("longer_user_global_inputs", [])),
            longer_user_global_tokens=payload.get("longer_user_global_tokens", 0),
            longer_cls_tokens=payload.get("longer_cls_tokens", 0),
            longer_candidate_global_tokens=payload.get("longer_candidate_global_tokens"),
            timestamp_field=payload.get("timestamp_field"),
            time_delta_field=payload.get("time_delta_field"),
        )

    def resolved_longer_candidate_global_tokens(self) -> int:
        if self.longer_candidate_global_tokens is not None:
            return self.longer_candidate_global_tokens
        return self.rankmixer_summary_tokens - self.longer_user_global_tokens - self.longer_cls_tokens

    def validate(self, scalar_feature_names: set[str]) -> None:
        if not self.name:
            raise ValueError("sequence.name is required")
        if "." in self.name:
            raise ValueError(f"sequence name {self.name!r} must not contain '.'")
        if self.embedding_scope not in {"feature", "scenario", "task", "shared"}:
            raise ValueError(
                f"sequence {self.name!r} embedding_scope must be feature, scenario, task, or shared"
            )
        if self.max_length is not None and self.max_length <= 0:
            raise ValueError(f"sequence {self.name!r} max_length must be positive")
        if self.truncation not in {"head", "tail"}:
            raise ValueError(f"sequence {self.name!r} truncation must be head or tail")
        if self.sequence_order not in {"oldest_to_newest", "newest_to_oldest"}:
            raise ValueError(
                f"sequence {self.name!r} sequence_order must be oldest_to_newest or newest_to_oldest"
            )
        if self.encoder not in {"raw", "attention_pool", "mean_pool", "longer"}:
            raise ValueError(
                f"sequence {self.name!r} encoder must be raw, attention_pool, "
                "mean_pool, or longer"
            )
        if self.rankmixer_summary_tokens <= 0:
            raise ValueError(f"sequence {self.name!r} rankmixer_summary_tokens must be positive")
        if self.encoder != "longer" and self.rankmixer_summary_tokens != 1:
            raise ValueError(
                f"sequence {self.name!r} rankmixer_summary_tokens > 1 requires encoder=longer"
            )
        if self.longer_query_tokens <= 0:
            raise ValueError(f"sequence {self.name!r} longer_query_tokens must be positive")
        if self.longer_self_layers < 0:
            raise ValueError(f"sequence {self.name!r} longer_self_layers must be non-negative")
        if self.longer_token_merge <= 0:
            raise ValueError(f"sequence {self.name!r} longer_token_merge must be positive")
        if self.longer_inner_layers < 0:
            raise ValueError(f"sequence {self.name!r} longer_inner_layers must be non-negative")
        if self.longer_user_global_tokens < 0:
            raise ValueError(f"sequence {self.name!r} longer_user_global_tokens must be non-negative")
        if self.longer_cls_tokens < 0:
            raise ValueError(f"sequence {self.name!r} longer_cls_tokens must be non-negative")
        candidate_global_tokens = self.resolved_longer_candidate_global_tokens()
        if candidate_global_tokens < 0:
            raise ValueError(
                f"sequence {self.name!r} longer_candidate_global_tokens must be non-negative"
            )
        if self.encoder == "longer":
            total_global_tokens = (
                self.longer_user_global_tokens
                + self.longer_cls_tokens
                + candidate_global_tokens
            )
            if total_global_tokens != self.rankmixer_summary_tokens:
                raise ValueError(
                    f"sequence {self.name!r} LONGER global token counts must sum to "
                    f"rankmixer_summary_tokens={self.rankmixer_summary_tokens}, got {total_global_tokens}"
                )
            if bool(self.longer_user_global_inputs) != (self.longer_user_global_tokens > 0):
                raise ValueError(
                    f"sequence {self.name!r} longer_user_global_inputs and "
                    "longer_user_global_tokens must either both be configured or both be empty"
                )
            if bool(self.target_inputs) != (candidate_global_tokens > 0):
                raise ValueError(
                    f"sequence {self.name!r} target_inputs and resolved candidate global token count "
                    "must either both be configured or both be empty"
                )
        if not self.fields:
            raise ValueError(f"sequence {self.name!r} must declare at least one field")
        field_names: set[str] = set()
        for field_config in self.fields:
            field_config.validate(self.name)
            if field_config.name in field_names:
                raise ValueError(f"duplicate field {field_config.name!r} in sequence {self.name!r}")
            field_names.add(field_config.name)
        fields_by_name = {item.name: item for item in self.fields}
        if (
            self.encoder == "longer"
            and self.time_delta_field is not None
            and len(self.fields) == 1
        ):
            raise ValueError(
                f"sequence {self.name!r} LONGER input requires at least one item/side field "
                "in addition to time_delta_field"
            )
        for option_name, field_name in (
            ("timestamp_field", self.timestamp_field),
            ("time_delta_field", self.time_delta_field),
        ):
            if field_name is None:
                continue
            temporal_field = fields_by_name.get(field_name)
            if temporal_field is None:
                raise ValueError(
                    f"sequence {self.name!r} {option_name} references unknown field {field_name!r}"
                )
            if temporal_field.kind != "dense" or temporal_field.dimension != 1:
                raise ValueError(
                    f"sequence {self.name!r} {option_name} must reference a scalar dense field"
                )
        missing_targets = [name for name in self.target_inputs if name not in scalar_feature_names]
        if missing_targets:
            raise ValueError(
                f"sequence {self.name!r} target_inputs references unknown scalar features: "
                + ", ".join(missing_targets)
            )
        missing_user_globals = [
            name for name in self.longer_user_global_inputs if name not in scalar_feature_names
        ]
        if missing_user_globals:
            raise ValueError(
                f"sequence {self.name!r} longer_user_global_inputs references unknown scalar features: "
                + ", ".join(missing_user_globals)
            )


@dataclass(frozen=True)
class ScenarioConfig:
    """Scenario ids used by MDL scenario-aware modules.

    A single default scenario does not need a source column. Multiple scenarios
    require a column so tensorization can build scenario masks.
    """

    # Ordered scenario names define model output token order.
    names: list[str] = field(default_factory=lambda: ["default"])
    # Optional parquet column carrying scenario id or scenario mask information.
    source: str | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "ScenarioConfig":
        if payload is None:
            return cls()
        return cls(**payload)

    def validate(self) -> None:
        if not self.names:
            raise ValueError("scenarios.names must contain at least one scenario")
        if any(not isinstance(name, str) for name in self.names):
            raise ValueError("scenarios.names must contain strings")
        if any(not name for name in self.names):
            raise ValueError("scenarios.names must not contain empty names")
        if len(set(self.names)) != len(self.names):
            raise ValueError("scenarios.names must not contain duplicates")
        if "global" in self.names:
            raise ValueError("scenarios.names must not contain reserved scenario name 'global'")
        if self.source is not None and not self.source:
            raise ValueError("scenarios.source must be null or a non-empty column name")
        if len(self.names) > 1 and self.source is None:
            raise ValueError("scenarios.source is required when multiple scenarios are configured")


@dataclass(frozen=True)
class TokenGroupConfig:
    """A named token built from one or more encoded inputs."""

    # Token name is model-visible and must be unique within its token section.
    name: str
    # Input references point to FeatureConfig.name or SequenceConfig.name, not columns.
    inputs: list[str]

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "TokenGroupConfig":
        return cls(**payload)

    def validate(self, feature_names: set[str], section: str) -> None:
        if not self.name:
            raise ValueError(f"tokenization.{section} token name is required")
        if not self.inputs:
            raise ValueError(f"tokenization.{section}.{self.name} inputs must not be empty")
        missing = [name for name in self.inputs if name not in feature_names]
        if missing:
            raise ValueError(
                f"tokenization.{section}.{self.name} references unknown features: "
                + ", ".join(missing)
            )


@dataclass(frozen=True)
class DomainTokenConfig:
    """A scenario or task token spec with ordered input groups.

    inputs, important_inputs, and prior_inputs are kept separate in YAML because
    they mirror the MDL paper language. Model code consumes the flattened order.
    """

    # Scenario name or task name. Scenario tokens may also include reserved global.
    name: str
    # Generic inputs, followed by paper-specific important/prior groups.
    inputs: list[str] = field(default_factory=list)
    important_inputs: list[str] = field(default_factory=list)
    prior_inputs: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "DomainTokenConfig":
        return cls(
            name=payload["name"],
            inputs=list(payload.get("inputs", [])),
            important_inputs=list(payload.get("important_inputs", [])),
            prior_inputs=list(payload.get("prior_inputs", [])),
        )

    def resolved_inputs(self) -> list[str]:
        ordered = [*self.inputs, *self.important_inputs, *self.prior_inputs]
        return list(dict.fromkeys(ordered))

    def validate(self, feature_names: set[str], section: str) -> None:
        if not self.name:
            raise ValueError(f"tokenization.{section} token name is required")
        inputs = self.resolved_inputs()
        if not inputs:
            raise ValueError(f"tokenization.{section}.{self.name} inputs must not be empty")
        missing = [name for name in inputs if name not in feature_names]
        if missing:
            raise ValueError(
                f"tokenization.{section}.{self.name} references unknown features: "
                + ", ".join(missing)
            )



@dataclass(frozen=True)
class TokenizationConfig:
    """Raw tokenization settings from YAML.

    Public resolved_* methods stay for compatibility. They delegate to the
    resolved config helpers below so the derivation logic has one source.
    """

    # groupwise uses explicit TokenGroupConfig entries; auto_split and rankmixer
    # project or reshape a flat ordered input list into fixed-width tokens.
    feature_tokenizer: Literal["groupwise", "rankmixer", "auto_split"] = "groupwise"
    # Required for auto_split/rankmixer so output token count is explicit.
    num_feature_tokens: int | None = None
    # Ordered input list for auto_split/rankmixer. Empty means all tokenizable inputs.
    feature_token_inputs: list[str] = field(default_factory=list)
    # Optional explicit feature token groups for groupwise tokenization.
    feature_tokens: list[TokenGroupConfig] = field(default_factory=list)
    # S-token groups for OneTrans-style sequence tokenization.
    sequence_tokens: list[TokenGroupConfig] = field(default_factory=list)
    # Non-sequence token groups. Kept as ns_tokens to preserve the YAML surface.
    ns_tokens: list[TokenGroupConfig] = field(default_factory=list)
    # MDL domain tokens. Scenario tokens also get a global token during resolution.
    scenario_tokens: list[DomainTokenConfig] = field(default_factory=list)
    # Task token names must match train label task names.
    task_tokens: list[DomainTokenConfig] = field(default_factory=list)
    # Fallback input sets used when scenario/task tokens are omitted.
    scenario_token_inputs: list[str] = field(default_factory=list)
    task_token_inputs: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "TokenizationConfig":
        if payload is None:
            return cls()
        return cls(
            feature_tokenizer=payload.get("feature_tokenizer", "groupwise"),
            num_feature_tokens=payload.get("num_feature_tokens"),
            feature_token_inputs=list(payload.get("feature_token_inputs", [])),
            feature_tokens=[
                TokenGroupConfig.from_mapping(item)
                for item in payload.get("feature_tokens", [])
            ],
            sequence_tokens=[
                TokenGroupConfig.from_mapping(item)
                for item in payload.get("sequence_tokens", [])
            ],
            ns_tokens=[
                TokenGroupConfig.from_mapping(item)
                for item in payload.get("ns_tokens", [])
            ],
            scenario_tokens=[
                DomainTokenConfig.from_mapping(item)
                for item in payload.get("scenario_tokens", [])
            ],
            task_tokens=[
                DomainTokenConfig.from_mapping(item)
                for item in payload.get("task_tokens", [])
            ],
            scenario_token_inputs=list(payload.get("scenario_token_inputs", [])),
            task_token_inputs=list(payload.get("task_token_inputs", [])),
        )

    def _sequences(self, sequences: list[SequenceConfig] | None) -> list[SequenceConfig]:
        return [] if sequences is None else sequences

    def _tokenizable_input_names(
        self,
        features: list[FeatureConfig],
        sequences: list[SequenceConfig] | None = None,
    ) -> list[str]:
        return tokenizable_input_names(features, self._sequences(sequences))

    def _sequence_input_names(
        self,
        features: list[FeatureConfig],
        sequences: list[SequenceConfig] | None = None,
    ) -> set[str]:
        return sequence_input_names(self._sequences(sequences))

    def resolved_feature_token_inputs(
        self,
        features: list[FeatureConfig],
        sequences: list[SequenceConfig] | None = None,
    ) -> list[str]:
        resolved = resolve_tokenization(self, features, self._sequences(sequences), [], [])
        return list(resolved.feature_token_inputs)

    def resolved_feature_token_count(
        self,
        features: list[FeatureConfig],
        sequences: list[SequenceConfig] | None = None,
    ) -> int:
        resolved = resolve_tokenization(self, features, self._sequences(sequences), [], [])
        return resolved.feature_token_count

    def resolved_feature_tokens(
        self,
        features: list[FeatureConfig],
        sequences: list[SequenceConfig] | None = None,
    ) -> list[TokenGroupConfig]:
        resolved = resolve_tokenization(self, features, self._sequences(sequences), [], [])
        return [group.as_token_group() for group in resolved.feature_token_groups]

    def resolved_sequence_tokens(
        self,
        features: list[FeatureConfig],
        sequences: list[SequenceConfig] | None = None,
    ) -> list[TokenGroupConfig]:
        resolved = resolve_tokenization(self, features, self._sequences(sequences), [], [])
        return [group.as_token_group() for group in resolved.sequence_token_groups]

    def resolved_ns_tokens(
        self,
        features: list[FeatureConfig],
        sequences: list[SequenceConfig] | None = None,
    ) -> list[TokenGroupConfig]:
        resolved = resolve_tokenization(self, features, self._sequences(sequences), [], [])
        return [group.as_token_group() for group in resolved.scalar_token_groups]

    def resolved_scenario_inputs(
        self,
        features: list[FeatureConfig],
        sequences: list[SequenceConfig] | None = None,
    ) -> list[str]:
        resolved = resolve_tokenization(self, features, self._sequences(sequences), [], [])
        return list(resolved.scenario_token_inputs)

    def resolved_task_inputs(
        self,
        features: list[FeatureConfig],
        sequences: list[SequenceConfig] | None = None,
    ) -> list[str]:
        resolved = resolve_tokenization(self, features, self._sequences(sequences), [], [])
        return list(resolved.task_token_inputs)

    def resolved_scenario_tokens(
        self,
        features: list[FeatureConfig],
        scenario_names: list[str],
        sequences: list[SequenceConfig] | None = None,
    ) -> list[DomainTokenConfig]:
        resolved = resolve_tokenization(self, features, self._sequences(sequences), scenario_names, [])
        return [token.as_domain_token() for token in resolved.scenario_token_specs]

    def resolved_task_tokens(
        self,
        features: list[FeatureConfig],
        task_names: list[str],
        sequences: list[SequenceConfig] | None = None,
    ) -> list[DomainTokenConfig]:
        resolved = resolve_tokenization(self, features, self._sequences(sequences), [], task_names)
        return [token.as_domain_token() for token in resolved.task_token_specs]

    def _validate_unique_domain_token_names(
        self,
        tokens: list[DomainTokenConfig],
        section: str,
    ) -> None:
        names: set[str] = set()
        duplicates: set[str] = set()
        for token in tokens:
            if token.name in names:
                duplicates.add(token.name)
            names.add(token.name)
        if duplicates:
            raise ValueError(
                f"tokenization.{section} contains duplicate token names: "
                + ", ".join(sorted(duplicates))
            )

    def validate(
        self,
        features: list[FeatureConfig],
        sequences: list[SequenceConfig],
        scenario_names: list[str],
        task_names: list[str],
    ) -> None:
        validate_tokenization_config(self, features, sequences, scenario_names, task_names)


@dataclass(frozen=True)
class VocabDefaults:
    """Default behavior for vocab fitting and lookup."""

    # Only train vocab fitting is supported so artifacts are reproducible.
    fit_split: str = "train"
    # Id 0 is reserved everywhere so padding and OOV are safe for embeddings.
    oov_id: int = 0
    padding_id: int = 0
    # In production, oov is safer; error is useful for strict debugging.
    unseen_policy: Literal["oov", "error"] = "oov"
    # Relative or absolute directory for fitted vocab artifacts.
    artifact_dir: str = "artifacts/vocab"

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "VocabDefaults":
        if payload is None:
            return cls()
        return cls(**payload)


@dataclass(frozen=True)
class VocabFeatureStrategy:
    """Encoding strategy for one categorical input.

    The dict key is the logical input name. source is the physical column used
    when fitting or reading the categorical values.
    """

    # vocab uses fitted artifacts, hash uses stable buckets, identity trusts ids.
    encoding: EncodingType
    # Physical column used to fit/load this categorical input.
    source: str
    # Vocab fitting filters and caps. Only used when encoding == "vocab".
    min_count: int | None = None
    max_size: int | None = None
    artifact: str | None = None
    # Hash bucket count and salt. Only used when encoding == "hash".
    num_buckets: int | None = None
    salt: str | None = None
    # Maximum accepted id. Only used when encoding == "identity".
    max_id: int | None = None
    padding_id: int = 0
    out_of_range: IdentityOutOfRangeType = "error"
    # shared_vocab reuses another feature's fitted vocab, optionally its embedding.
    share_with: str | None = None
    share_embedding: bool = False

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "VocabFeatureStrategy":
        return cls(**payload)

    def validate(self, feature_name: str) -> None:
        if self.encoding not in {"vocab", "hash", "identity", "shared_vocab"}:
            raise ValueError(f"vocab_strategy.features.{feature_name}.encoding is invalid")
        if not self.source:
            raise ValueError(f"vocab_strategy.features.{feature_name}.source is required")
        if self.encoding == "vocab":
            if not self.artifact:
                raise ValueError(f"vocab feature {feature_name!r} requires artifact")
            if self.min_count is not None and self.min_count <= 0:
                raise ValueError(f"vocab feature {feature_name!r} min_count must be positive")
            if self.max_size is not None and self.max_size <= 0:
                raise ValueError(f"vocab feature {feature_name!r} max_size must be positive")
        if self.encoding == "hash":
            if self.num_buckets is None or self.num_buckets <= 0:
                raise ValueError(f"hash feature {feature_name!r} requires positive num_buckets")
        if self.encoding == "identity":
            _validate_identity_bounds(
                num_buckets=self.num_buckets,
                max_id=self.max_id,
                padding_id=self.padding_id,
                out_of_range=self.out_of_range,
                path=f"vocab_strategy.features.{feature_name}",
            )
            if self.share_embedding and not self.share_with:
                raise ValueError(
                    f"identity feature {feature_name!r} requires share_with when "
                    "share_embedding=true"
                )
        if self.encoding == "shared_vocab" and not self.share_with:
            raise ValueError(f"shared_vocab feature {feature_name!r} requires share_with")


@dataclass(frozen=True)
class VocabStrategy:
    """All categorical encoding strategies keyed by input name."""

    defaults: VocabDefaults = field(default_factory=VocabDefaults)
    features: dict[str, VocabFeatureStrategy] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "VocabStrategy":
        if payload is None:
            return cls()
        features = {
            name: VocabFeatureStrategy.from_mapping(strategy)
            for name, strategy in payload.get("features", {}).items()
        }
        return cls(
            defaults=VocabDefaults.from_mapping(payload.get("defaults")),
            features=features,
        )

    def validate(self) -> None:
        if self.defaults.oov_id != 0 or self.defaults.padding_id != 0:
            raise ValueError("vocab_strategy defaults must reserve id 0 for OOV and padding")
        if self.defaults.fit_split != "train":
            raise ValueError("vocab_strategy.defaults.fit_split must be train")
        if self.defaults.unseen_policy not in {"oov", "error"}:
            raise ValueError("vocab_strategy.defaults.unseen_policy must be oov or error")
        for name, strategy in self.features.items():
            strategy.validate(name)


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime options for device, precision, and distributed launch."""

    # Device string is passed to torch.device by training code.
    device: str = "cpu"
    # Mixed precision mode. fp32 is the conservative default.
    precision: Literal["fp32", "bf16", "fp16"] = "fp32"
    # torch.compile toggle. Keep false for easier debugging and wider compatibility.
    compile: bool = False
    # TensorFloat-32 accelerates FP32 matrix multiplications on supported GPUs.
    allow_tf32: bool = True
    # none avoids recompute; selective checkpoints large blocks; full also
    # checkpoints model-specific preprocessing/merge stages where supported.
    activation_checkpoint: Literal["none", "selective", "full"] = "none"
    # Attention backend selection is resolved by model modules at runtime.
    attention_backend: Literal["auto", "sdpa", "flash"] = "auto"
    # DDP launch options. none means single process.
    distributed: Literal["none", "ddp"] = "none"
    nproc_per_node: int | None = None
    master_addr: str = "127.0.0.1"
    master_port: int = 29500

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "RuntimeConfig":
        if payload is None:
            return cls()
        values = dict(payload)
        legacy_checkpoint = values.get("activation_checkpoint")
        if isinstance(legacy_checkpoint, bool):
            values["activation_checkpoint"] = (
                "full" if legacy_checkpoint else "none"
            )
        return cls(**values)

    def validate(self) -> None:
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("runtime.precision must be fp32, bf16, or fp16")
        if self.attention_backend not in {"auto", "sdpa", "flash"}:
            raise ValueError("runtime.attention_backend must be auto, sdpa, or flash")
        if self.activation_checkpoint not in {"none", "selective", "full"}:
            raise ValueError(
                "runtime.activation_checkpoint must be none, selective, or full"
            )
        if self.attention_backend == "flash":
            if not self.device.startswith("cuda"):
                raise ValueError("runtime.attention_backend=flash requires a CUDA device")
            if self.precision not in {"bf16", "fp16"}:
                raise ValueError(
                    "runtime.attention_backend=flash requires BF16 or FP16 precision"
                )
        if self.distributed not in {"none", "ddp"}:
            raise ValueError("runtime.distributed must be none or ddp")
        if self.nproc_per_node is not None and self.nproc_per_node <= 0:
            raise ValueError("runtime.nproc_per_node must be positive")
        if not 1 <= self.master_port <= 65535:
            raise ValueError("runtime.master_port must be in [1, 65535]")


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture and paper-alignment switches."""

    # Selects the concrete model class in model.build_model.
    name: ModelName
    # Default embedding width for categorical scalar features and sequence fields.
    embedding_dim: int = 32
    # Transformer/token width. RankMixer reshape rules depend on this exactly.
    token_dim: int = 768
    # Shared backbone depth and attention/FFN dimensions.
    num_layers: int = 6
    num_heads: int = 12
    hidden_dim: int = 1536
    # Initialization and activation choices for trainable modules.
    init_std: float = 0.02
    ffn_activation: ActivationType = "gelu"
    # Optional task head MLP override. None means model.hidden_dim.
    task_head_hidden_dim: int | None = None
    task_head_dropout: float = 0.0
    task_head_activation: ActivationType = "gelu"
    # MDL token ablations. False removes the corresponding token projectors and
    # block modules, then uses the repository's explicit task/scenario towers.
    use_task_tokens: bool = True
    use_scenario_tokens: bool = True
    # False removes the global token itself (including projector/FFN parameters).
    use_global_scenario_token: bool = True
    # MDL interaction ablations. False replaces DomainAwareAttention with
    # RankMixer mixing over concatenated [feature; domain] tokens; it never
    # means a zero update.
    use_task_feature_interaction: bool = True
    use_scenario_feature_interaction: bool = True
    # MDL Eq. (6) has no second residual/LayerNorm. rankmixer_full is retained
    # only as an explicit ablation/compatibility path.
    mdl_feature_interaction: MDLFeatureInteractionType = "paper"
    # Local request-cache support for sequence encoders.
    use_request_cache: bool = False
    # OneTrans pyramid controls for reducing S tokens over layers.
    use_pyramid: bool = True
    pyramid_round_to: int = 32
    # OneTrans NS tokenizer mode and optional token count override.
    ns_tokenizer: Literal["auto_split", "groupwise"] = "auto_split"
    num_ns_tokens: int | None = None
    # Capacity of the learned absolute embedding added to the unified [S; NS]
    # sequence. None infers the exact configured maximum when every behavior
    # sequence declares max_length.
    max_position_embeddings: int | None = None
    # Separator and final S-token controls for OneTrans sequence handling.
    use_sep_tokens: bool = True
    final_s_tokens: int | None = None
    sequence_fusion: SequenceFusionType = "intent_ordered"
    rankmixer_ffn_type: RankMixerFFNType = "dense"
    sparse_moe_num_experts: int = 4
    sparse_moe_use_dtsi: bool = True
    sparse_moe_inference_threshold: float = 0.0
    sparse_moe_target_active_ratio: float = 0.25
    sparse_moe_regularization_initial: float = 1.0e-8
    sparse_moe_regularization_multiplier: float = 1.2
    sparse_moe_loss_weight: float = 1.0
    # RankMixer does not publish the DTSI training-output fusion equation. A
    # sparse DTSI run must therefore acknowledge an implementation choice.
    sparse_moe_dtsi_training_output: DTSITrainingOutputType | None = None
    # mdl_onetrans is an experimental composition, not a published model.
    experimental_model_acknowledged: bool = False

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ModelConfig":
        if not isinstance(payload, dict):
            raise ValueError("model must be an object")
        return cls(**payload)

    def validate(self) -> None:
        if self.name not in {"rankmixer", "mdl_rankmixer", "onetrans", "mdl_onetrans", "longer"}:
            raise ValueError(
                "model.name must be rankmixer, mdl_rankmixer, onetrans, mdl_onetrans, or longer"
            )
        if self.token_dim <= 0:
            raise ValueError("model.token_dim must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("model.embedding_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("model.num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("model.num_heads must be positive")
        if self.token_dim % self.num_heads != 0:
            raise ValueError("model.token_dim must be divisible by num_heads")
        if self.hidden_dim <= 0:
            raise ValueError("model.hidden_dim must be positive")
        if self.init_std <= 0:
            raise ValueError("model.init_std must be positive")
        if self.ffn_activation not in {"gelu", "relu"}:
            raise ValueError("model.ffn_activation must be gelu or relu")
        if self.task_head_hidden_dim is not None and self.task_head_hidden_dim <= 0:
            raise ValueError("model.task_head_hidden_dim must be positive")
        if self.task_head_dropout < 0.0 or self.task_head_dropout >= 1.0:
            raise ValueError("model.task_head_dropout must be in [0, 1)")
        if self.task_head_activation not in {"gelu", "relu"}:
            raise ValueError("model.task_head_activation must be gelu or relu")
        if self.mdl_feature_interaction not in {"paper", "rankmixer_full"}:
            raise ValueError("model.mdl_feature_interaction must be paper or rankmixer_full")
        if self.pyramid_round_to <= 0:
            raise ValueError("model.pyramid_round_to must be positive")
        if self.ns_tokenizer not in {"auto_split", "groupwise"}:
            raise ValueError("model.ns_tokenizer must be auto_split or groupwise")
        if self.num_ns_tokens is not None and self.num_ns_tokens <= 0:
            raise ValueError("model.num_ns_tokens must be positive")
        if self.max_position_embeddings is not None and self.max_position_embeddings <= 0:
            raise ValueError("model.max_position_embeddings must be positive")
        if self.final_s_tokens is not None and self.final_s_tokens < 0:
            raise ValueError("model.final_s_tokens must be non-negative")
        if self.sequence_fusion not in {"timestamp_aware", "intent_ordered"}:
            raise ValueError("model.sequence_fusion must be timestamp_aware or intent_ordered")
        if self.rankmixer_ffn_type not in {"dense", "sparse_moe"}:
            raise ValueError("model.rankmixer_ffn_type must be dense or sparse_moe")
        if self.sparse_moe_num_experts <= 0:
            raise ValueError("model.sparse_moe_num_experts must be positive")
        if self.sparse_moe_inference_threshold < 0.0:
            raise ValueError("model.sparse_moe_inference_threshold must be non-negative")
        if not 0.0 < self.sparse_moe_target_active_ratio <= 1.0:
            raise ValueError("model.sparse_moe_target_active_ratio must be in (0, 1]")
        if self.sparse_moe_regularization_initial <= 0.0:
            raise ValueError("model.sparse_moe_regularization_initial must be positive")
        if self.sparse_moe_regularization_multiplier <= 1.0:
            raise ValueError("model.sparse_moe_regularization_multiplier must be greater than 1")
        if self.sparse_moe_loss_weight < 0.0:
            raise ValueError("model.sparse_moe_loss_weight must be non-negative")
        if self.sparse_moe_dtsi_training_output not in {
            None,
            "dense_router",
            "mean",
        }:
            raise ValueError(
                "model.sparse_moe_dtsi_training_output must be dense_router, mean, or null"
            )
        if (
            self.rankmixer_ffn_type == "sparse_moe"
            and self.sparse_moe_use_dtsi
            and self.sparse_moe_dtsi_training_output is None
        ):
            raise ValueError(
                "RankMixer does not publish the DTSI training-output fusion equation; set "
                "model.sparse_moe_dtsi_training_output explicitly to acknowledge the "
                "implementation choice"
            )
        if self.name == "mdl_onetrans" and not self.experimental_model_acknowledged:
            raise ValueError(
                "model.name=mdl_onetrans is experimental and is not defined by the MDL or OneTrans paper; "
                "set model.experimental_model_acknowledged=true to opt in"
            )


@dataclass(frozen=True)
class EmbeddingShardingConfig:
    """Model-independent ownership policy for industrial ID embeddings."""

    strategy: Literal["auto", "row_wise", "table_wise"] = "auto"
    local_dedup: bool = True
    table_wise_max_rows: int = 65536

    @classmethod
    def from_mapping(
        cls, payload: dict[str, Any] | None
    ) -> "EmbeddingShardingConfig":
        if payload is None:
            return cls()
        return cls(**payload)

    def validate(self) -> None:
        if self.strategy not in {"auto", "row_wise", "table_wise"}:
            raise ValueError(
                "training.embedding_sharding.strategy must be auto, row_wise, or table_wise"
            )
        if self.table_wise_max_rows <= 0:
            raise ValueError(
                "training.embedding_sharding.table_wise_max_rows must be positive"
            )


@dataclass(frozen=True)
class DDPConfig:
    """Dense DDP reducer settings with explicit validation evidence gates."""

    static_graph: bool = False
    find_unused_parameters: bool = True
    gradient_as_bucket_view: bool = True
    bucket_cap_mb: float = 25.0
    audit_steps: int = 10
    validated_no_unused_parameters: bool = False
    validated_static_graph: bool = False

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "DDPConfig":
        if payload is None:
            return cls()
        return cls(**payload)

    def validate(self) -> None:
        if self.bucket_cap_mb <= 0.0:
            raise ValueError("training.ddp.bucket_cap_mb must be positive")
        if self.audit_steps < 0:
            raise ValueError("training.ddp.audit_steps must be non-negative")
        if self.static_graph and self.find_unused_parameters:
            raise ValueError(
                "training.ddp.static_graph=true requires find_unused_parameters=false"
            )
        if self.static_graph and not self.validated_static_graph:
            raise ValueError(
                "training.ddp.static_graph=true requires validated_static_graph=true "
                "after a representative audit"
            )
        if (
            not self.static_graph
            and not self.find_unused_parameters
            and not self.validated_no_unused_parameters
        ):
            raise ValueError(
                "training.ddp.find_unused_parameters=false requires "
                "validated_no_unused_parameters=true when static_graph is false"
            )


@dataclass(frozen=True)
class TrainingConfig:
    """Optimizer, batch, schedule, and checkpoint settings."""

    # Batch size is per process when DDP is enabled.
    batch_size: int = 2048
    # Dense optimizer learning rate. Sparse lr falls back to this when None.
    lr_dense: float = 0.005
    lr_sparse: float | None = None
    # Learning-rate schedule parameters used by train.py.
    lr_schedule: LRScheduleType = "constant"
    lr_warmup_steps: int = 0
    lr_decay_steps: int | None = None
    lr_min_ratio: float = 0.0
    # Optimizer names are constrained for paper alignment and implementation scope.
    dense_optimizer: Literal["rmsprop"] = "rmsprop"
    fused_dense_optimizer: bool = True
    rmsprop_alpha: float = 0.99999
    rmsprop_momentum: float = 0.0
    sparse_optimizer: Literal["adagrad"] = "adagrad"
    adagrad_lr_decay: float = 0.0
    adagrad_weight_decay: float = 0.0
    adagrad_initial_accumulator_value: float = 0.1
    adagrad_eps: float = 1.0e-10
    # Replicated is the small-table correctness baseline. Sharded stores only
    # the locally owned rows and optimizer state on each rank.
    embedding_distribution: Literal["replicated", "sharded"] = "replicated"
    dense_distribution: Literal["ddp"] = "ddp"
    embedding_sharding: EmbeddingShardingConfig = field(
        default_factory=EmbeddingShardingConfig
    )
    ddp: DDPConfig = field(default_factory=DDPConfig)
    # Both built-in embedding distributions use sparse gradients. Replicated
    # exchanges touched rows; sharded routes IDs/gradients only to row owners.
    embedding_sparse_gradients: bool = True
    # Compatibility selector for the replicated path or an out-of-scope
    # external adapter. embedding_distribution selects built-in owner sharding.
    sparse_update_mode: Literal["ddp_synced_adagrad", "external_parameter_server"] = "ddp_synced_adagrad"
    sparse_parameter_server_adapter: str | None = None
    # Optional gradient clipping for dense and sparse parameter groups.
    dense_clip_norm: float | None = None
    sparse_clip_norm: float | None = None
    # MDL Eq. (1) uses a literal sum. mean_per_task is an explicit engineering
    # alternative for datasets that need task-balanced normalization.
    loss_reduction: LossReductionType = "sum"
    # Default checkpoint path used by train/predict when CLI does not override it.
    checkpoint_path: str | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "TrainingConfig":
        if payload is None:
            return cls()
        values = dict(payload)
        values["embedding_sharding"] = EmbeddingShardingConfig.from_mapping(
            values.get("embedding_sharding")
        )
        values["ddp"] = DDPConfig.from_mapping(values.get("ddp"))
        return cls(**values)

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("training.batch_size must be positive")
        if self.lr_dense <= 0:
            raise ValueError("training.lr_dense must be positive")
        if self.lr_sparse is not None and self.lr_sparse <= 0:
            raise ValueError("training.lr_sparse must be positive")
        if self.lr_schedule not in {"constant", "cosine"}:
            raise ValueError("training.lr_schedule must be constant or cosine")
        if self.lr_warmup_steps < 0:
            raise ValueError("training.lr_warmup_steps must be non-negative")
        if self.lr_decay_steps is not None and self.lr_decay_steps <= 0:
            raise ValueError("training.lr_decay_steps must be positive")
        if self.lr_schedule == "cosine" and self.lr_decay_steps is not None:
            if self.lr_decay_steps <= self.lr_warmup_steps:
                raise ValueError("training.lr_decay_steps must be greater than training.lr_warmup_steps")
        if not 0.0 <= self.lr_min_ratio <= 1.0:
            raise ValueError("training.lr_min_ratio must be in [0, 1]")
        if self.dense_optimizer != "rmsprop":
            raise ValueError("training.dense_optimizer must be rmsprop for paper alignment")
        if not 0.0 <= self.rmsprop_alpha < 1.0:
            raise ValueError("training.rmsprop_alpha must be in [0, 1)")
        if self.rmsprop_momentum < 0.0:
            raise ValueError("training.rmsprop_momentum must be non-negative")
        if self.sparse_optimizer != "adagrad":
            raise ValueError("training.sparse_optimizer must be adagrad for paper alignment")
        if self.adagrad_lr_decay < 0.0:
            raise ValueError("training.adagrad_lr_decay must be non-negative")
        if self.adagrad_weight_decay < 0.0:
            raise ValueError("training.adagrad_weight_decay must be non-negative")
        if self.adagrad_initial_accumulator_value < 0.0:
            raise ValueError("training.adagrad_initial_accumulator_value must be non-negative")
        if self.adagrad_eps <= 0.0:
            raise ValueError("training.adagrad_eps must be positive")
        if self.embedding_distribution not in {"replicated", "sharded"}:
            raise ValueError(
                "training.embedding_distribution must be replicated or sharded"
            )
        if self.dense_distribution != "ddp":
            raise ValueError("training.dense_distribution must be ddp")
        self.embedding_sharding.validate()
        self.ddp.validate()
        if self.embedding_distribution == "sharded" and not self.embedding_sparse_gradients:
            raise ValueError(
                "training.embedding_sparse_gradients must be true for sharded embeddings"
            )
        if (
            self.embedding_distribution == "sharded"
            and self.sparse_update_mode != "ddp_synced_adagrad"
        ):
            raise ValueError(
                "training.embedding_distribution=sharded uses the built-in owner-based "
                "implementation and is incompatible with external_parameter_server"
            )
        if self.sparse_update_mode not in {"ddp_synced_adagrad", "external_parameter_server"}:
            raise ValueError(
                "training.sparse_update_mode must be ddp_synced_adagrad or external_parameter_server"
            )
        if self.sparse_update_mode == "external_parameter_server" and not self.sparse_parameter_server_adapter:
            raise ValueError(
                "training.sparse_parameter_server_adapter is required when sparse_update_mode is external_parameter_server"
            )
        if self.dense_clip_norm is not None and self.dense_clip_norm <= 0:
            raise ValueError("training.dense_clip_norm must be positive")
        if self.sparse_clip_norm is not None and self.sparse_clip_norm <= 0:
            raise ValueError("training.sparse_clip_norm must be positive")
        if self.loss_reduction not in {"sum", "mean_per_task"}:
            raise ValueError("training.loss_reduction must be sum or mean_per_task")


@dataclass(frozen=True)
class AppConfig:
    """Top-level config object used by CLI, data, model, and training code."""

    data: DataConfig
    features: list[FeatureConfig]
    sequences: list[SequenceConfig]
    vocab_strategy: VocabStrategy
    model: ModelConfig
    scenarios: ScenarioConfig = field(default_factory=ScenarioConfig)
    tokenization: TokenizationConfig = field(default_factory=TokenizationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "AppConfig":
        if not isinstance(payload, dict):
            raise ValueError("config must be an object")
        features = [FeatureConfig.from_mapping(item) for item in payload.get("features", [])]
        sequences = [SequenceConfig.from_mapping(item) for item in payload.get("sequences", [])]
        return cls(
            data=DataConfig.from_mapping(payload["data"]),
            features=features,
            sequences=sequences,
            vocab_strategy=VocabStrategy.from_mapping(payload.get("vocab_strategy")),
            model=ModelConfig.from_mapping(payload["model"]),
            scenarios=ScenarioConfig.from_mapping(payload.get("scenarios")),
            tokenization=TokenizationConfig.from_mapping(payload.get("tokenization")),
            runtime=RuntimeConfig.from_mapping(payload.get("runtime")),
            training=TrainingConfig.from_mapping(payload.get("training")),
        )

    @property
    def task_names(self) -> list[str]:
        return list(self.data.train.labels.keys())

    @cached_property
    def resolved(self):
        # Derived config is immutable for this AppConfig instance, so cache it.
        return resolve_app_config(self)

    def _encoded_input_dims(self) -> dict[str, int]:
        return dict(self.resolved.encoded_input_dims)

    def _categorical_embedding_dims(self) -> dict[str, int]:
        return dict(self.resolved.categorical_embedding_dims)

    def _validate_vocab_strategy_references(self) -> None:
        validate_vocab_strategy_references(self)

    def validate(self) -> None:
        validate_app_config(self)


# Resolved configuration helpers keep derived model-facing state separate from raw YAML fields.
@dataclass(frozen=True)
class ResolvedTokenGroup:
    """Model-facing token group with immutable input references."""

    name: str
    input_refs: tuple[str, ...]

    @property
    def inputs(self) -> list[str]:
        return list(self.input_refs)

    def as_token_group(self) -> TokenGroupConfig:
        return TokenGroupConfig(name=self.name, inputs=list(self.input_refs))


@dataclass(frozen=True)
class ResolvedDomainToken:
    """Model-facing scenario or task token with flattened inputs."""

    name: str
    input_refs: tuple[str, ...]
    direct_input_refs: tuple[str, ...] = ()
    important_input_refs: tuple[str, ...] = ()
    prior_input_refs: tuple[str, ...] = ()

    @property
    def inputs(self) -> list[str]:
        return list(self.direct_input_refs)

    @property
    def important_inputs(self) -> list[str]:
        return list(self.important_input_refs)

    @property
    def prior_inputs(self) -> list[str]:
        return list(self.prior_input_refs)

    def resolved_inputs(self) -> list[str]:
        return list(self.input_refs)

    def as_domain_token(self) -> DomainTokenConfig:
        # Preserve the public DomainTokenConfig shape for legacy callers.
        return DomainTokenConfig(
            name=self.name,
            inputs=list(self.direct_input_refs),
            important_inputs=list(self.important_input_refs),
            prior_inputs=list(self.prior_input_refs),
        )


@dataclass(frozen=True)
class ResolvedTokenization:
    """Fully derived token layout used by model builders and validators."""

    feature_tokenizer: str
    feature_token_inputs: tuple[str, ...]
    feature_token_count: int
    feature_token_groups: tuple[ResolvedTokenGroup, ...]
    sequence_token_groups: tuple[ResolvedTokenGroup, ...]
    scalar_token_groups: tuple[ResolvedTokenGroup, ...]
    scenario_token_inputs: tuple[str, ...]
    task_token_inputs: tuple[str, ...]
    scenario_token_specs: tuple[ResolvedDomainToken, ...]
    task_token_specs: tuple[ResolvedDomainToken, ...]
    sequence_names: tuple[str, ...]

    @property
    def ns_token_groups(self) -> tuple[ResolvedTokenGroup, ...]:
        return self.scalar_token_groups


@dataclass(frozen=True)
class ResolvedVocabEncoding:
    encoding: Literal["vocab"] = "vocab"
    artifact: str = ""
    min_count: int | None = None
    max_size: int | None = None


@dataclass(frozen=True)
class ResolvedHashEncoding:
    encoding: Literal["hash"] = "hash"
    num_buckets: int = 0
    salt: str | None = None


@dataclass(frozen=True)
class ResolvedIdentityEncoding:
    encoding: Literal["identity"] = "identity"
    num_buckets: int = 0
    padding_id: int = 0
    out_of_range: IdentityOutOfRangeType = "error"
    share_with: str | None = None
    share_embedding: bool = False

    @property
    def max_id(self) -> int:
        """Compatibility view; new code must use the exclusive num_buckets."""

        return self.num_buckets - 1


@dataclass(frozen=True)
class ResolvedSharedVocabEncoding:
    encoding: Literal["shared_vocab"] = "shared_vocab"
    share_with: str = ""
    share_embedding: bool = False


ResolvedEncoding = (
    ResolvedVocabEncoding
    | ResolvedHashEncoding
    | ResolvedIdentityEncoding
    | ResolvedSharedVocabEncoding
)


@dataclass(frozen=True)
class ResolvedCategoricalInput:
    """Model-facing categorical input and its resolved integer-id encoding."""

    name: str
    source: str
    location: Literal["feature", "sequence_field"]
    sequence_name: str | None
    field_name: str | None
    encoding: ResolvedEncoding


def resolve_categorical_base_input(
    categorical_input_by_name: dict[str, ResolvedCategoricalInput],
    name: str,
) -> ResolvedCategoricalInput:
    """Resolve the non-alias ID namespace behind a shared categorical input.

    ``shared_vocab`` historically meant that two inputs shared a fitted string
    vocabulary. Industrial direct-ID configs also need the same aliasing
    contract: multiple physical columns can already contain integers from one
    bounded ID namespace while retaining independent embedding tables. The
    returned input owns the effective bounds/lookup strategy; callers must
    continue using the original input's physical source column.
    """

    current = name
    seen: set[str] = set()
    while True:
        if current in seen:
            raise ValueError(f"shared_vocab cycle detected at {name!r}")
        seen.add(current)
        try:
            categorical_input = categorical_input_by_name[current]
        except KeyError as error:
            raise ValueError(
                f"shared_vocab feature {name!r} references unknown categorical input "
                f"{current!r}"
            ) from error
        encoding = categorical_input.encoding
        if not isinstance(encoding, ResolvedSharedVocabEncoding):
            return categorical_input
        current = encoding.share_with


@dataclass(frozen=True)
class ResolvedConfig:
    """Cached derived state for cross-section validation and model setup."""

    # Token layout after defaults and implicit groups have been expanded.
    tokenization: ResolvedTokenization
    # Embedding widths after per-feature overrides and shared vocab chains resolve.
    categorical_embedding_dims: dict[str, int]
    # Encoded width for every model input after scalar and sequence encoding.
    encoded_input_dims: dict[str, int]
    # All logical categorical inputs that need vocab_strategy coverage.
    categorical_input_names: set[str]
    # Ordered categorical inputs resolved from inline encoding or vocab_strategy.
    categorical_inputs: tuple[ResolvedCategoricalInput, ...]
    # Name lookup for hot paths; values are resolved and do not expose raw YAML strategy objects.
    categorical_input_by_name: dict[str, ResolvedCategoricalInput]
    # Scalar feature names available to OneTrans NS auto_split.
    scalar_feature_names: tuple[str, ...]


def tokenizable_input_names(features: list[FeatureConfig], sequences: list[SequenceConfig]) -> list[str]:
    """Inputs that can become feature tokens by default."""

    return [
        feature.name
        for feature in features
        if feature.embedding_scope in {"feature", "shared"}
    ] + [
        sequence.name
        for sequence in sequences
        if sequence.embedding_scope in {"feature", "shared"}
    ]


def sequence_input_names(sequences: list[SequenceConfig]) -> set[str]:
    """Sequence names that are active in feature/shared token scopes."""

    return {
        sequence.name
        for sequence in sequences
        if sequence.embedding_scope in {"feature", "shared"}
    }


def categorical_input_names(config: AppConfig) -> set[str]:
    """Categorical inputs that must have vocab_strategy entries."""

    return {
        feature.name
        for feature in config.features
        if feature.kind == "categorical"
    } | {
        field.qualified_name(sequence.name)
        for sequence in config.sequences
        for field in sequence.fields
        if field.kind == "categorical"
    }


def _resolved_encoding_from_config(
    encoding: CategoricalEncodingConfig | VocabFeatureStrategy,
    path: str,
) -> ResolvedEncoding:
    if encoding.encoding == "vocab":
        if not encoding.artifact:
            raise ValueError(f"{path}.artifact is required for vocab encoding")
        if encoding.min_count is not None and encoding.min_count <= 0:
            raise ValueError(f"{path}.min_count must be positive")
        if encoding.max_size is not None and encoding.max_size <= 0:
            raise ValueError(f"{path}.max_size must be positive")
        return ResolvedVocabEncoding(
            artifact=encoding.artifact,
            min_count=encoding.min_count,
            max_size=encoding.max_size,
        )
    if encoding.encoding == "hash":
        if encoding.num_buckets is None or encoding.num_buckets <= 0:
            raise ValueError(f"{path}.num_buckets must be positive for hash encoding")
        return ResolvedHashEncoding(
            num_buckets=encoding.num_buckets,
            salt=encoding.salt,
        )
    if encoding.encoding == "identity":
        _validate_identity_bounds(
            num_buckets=encoding.num_buckets,
            max_id=encoding.max_id,
            padding_id=encoding.padding_id,
            out_of_range=encoding.out_of_range,
            path=path,
        )
        num_buckets = (
            encoding.num_buckets
            if encoding.num_buckets is not None
            else int(encoding.max_id) + 1
        )
        return ResolvedIdentityEncoding(
            num_buckets=num_buckets,
            padding_id=encoding.padding_id,
            out_of_range=encoding.out_of_range,
            share_with=encoding.share_with,
            share_embedding=encoding.share_embedding,
        )
    if encoding.encoding == "shared_vocab":
        if not encoding.share_with:
            raise ValueError(f"{path}.share_with is required for shared_vocab encoding")
        return ResolvedSharedVocabEncoding(
            share_with=encoding.share_with,
            share_embedding=encoding.share_embedding,
        )
    raise ValueError(f"{path}.type is invalid")


def _resolved_categorical_input(
    config: AppConfig,
    name: str,
    source: str,
    location: Literal["feature", "sequence_field"],
    sequence_name: str | None,
    field_name: str | None,
    inline_encoding: CategoricalEncodingConfig | None,
) -> ResolvedCategoricalInput:
    legacy = config.vocab_strategy.features.get(name)
    if inline_encoding is not None and legacy is not None:
        raise ValueError(
            f"categorical input {name!r} declares encoding both inline and in vocab_strategy.features"
        )
    if inline_encoding is not None:
        encoding = _resolved_encoding_from_config(inline_encoding, f"{name}.encoding")
    elif legacy is not None:
        if legacy.source != source:
            raise ValueError(
                f"vocab_strategy.features.{name}.source {legacy.source!r} does not match "
                f"logical source {source!r}"
            )
        encoding = _resolved_encoding_from_config(legacy, f"vocab_strategy.features.{name}")
    else:
        raise ValueError(f"missing encoding for categorical input {name!r}")
    return ResolvedCategoricalInput(
        name=name,
        source=source,
        location=location,
        sequence_name=sequence_name,
        field_name=field_name,
        encoding=encoding,
    )


def resolve_encoding_strategies(config: AppConfig) -> tuple[ResolvedCategoricalInput, ...]:
    """Resolve categorical encodings from exactly one config source per input."""

    inputs: list[ResolvedCategoricalInput] = []
    for feature in config.features:
        if feature.kind != "categorical":
            continue
        inputs.append(
            _resolved_categorical_input(
                config,
                name=feature.name,
                source=feature.source,
                location="feature",
                sequence_name=None,
                field_name=None,
                inline_encoding=feature.encoding,
            )
        )
    for sequence in config.sequences:
        for field_config in sequence.fields:
            if field_config.kind != "categorical":
                continue
            qualified = field_config.qualified_name(sequence.name)
            inputs.append(
                _resolved_categorical_input(
                    config,
                    name=qualified,
                    source=field_config.source,
                    location="sequence_field",
                    sequence_name=sequence.name,
                    field_name=field_config.name,
                    inline_encoding=field_config.encoding,
                )
            )

    by_name = {item.name: item for item in inputs}
    if len(by_name) != len(inputs):
        raise ValueError("duplicate categorical input names are not allowed")
    unknown = sorted(name for name in config.vocab_strategy.features if name not in by_name)
    if unknown:
        raise ValueError("vocab_strategy contains unknown categorical inputs: " + ", ".join(unknown))

    resolved_base: dict[str, str] = {}

    def base_encoding(name: str, stack: set[str]) -> str:
        if name in resolved_base:
            return resolved_base[name]
        if name in stack:
            raise ValueError(f"shared_vocab cycle detected at {name!r}")
        item = by_name[name]
        encoding = item.encoding
        if encoding.encoding != "shared_vocab":
            resolved_base[name] = encoding.encoding
            return encoding.encoding
        if encoding.share_with not in by_name:
            raise ValueError(
                f"shared_vocab feature {name!r} references unknown categorical input "
                f"{encoding.share_with!r}"
            )
        if encoding.share_with == name:
            raise ValueError(f"shared_vocab feature {name!r} cannot share with itself")
        base = base_encoding(encoding.share_with, stack | {name})
        resolved_base[name] = base
        return base

    for item in inputs:
        if item.encoding.encoding != "shared_vocab":
            continue
        base = base_encoding(item.name, set())
        if base not in {"vocab", "identity"}:
            raise ValueError(
                f"shared_vocab feature {item.name!r} must ultimately share with a "
                "vocab- or identity-encoded feature"
            )

    for item in inputs:
        encoding = item.encoding
        if not isinstance(encoding, ResolvedIdentityEncoding):
            continue
        if not encoding.share_embedding:
            continue
        if encoding.share_with not in by_name:
            raise ValueError(
                f"identity feature {item.name!r} references unknown embedding base "
                f"{encoding.share_with!r}"
            )
        if encoding.share_with == item.name:
            raise ValueError(
                f"identity feature {item.name!r} cannot share embedding with itself"
            )
        base_input = resolve_categorical_base_input(by_name, encoding.share_with)
        if not isinstance(base_input.encoding, ResolvedIdentityEncoding):
            raise ValueError(
                f"identity feature {item.name!r} can share an embedding only with "
                "an identity-encoded feature"
            )
        if base_input.encoding.num_buckets != encoding.num_buckets:
            raise ValueError(
                f"identity feature {item.name!r} num_buckets={encoding.num_buckets} "
                f"does not match embedding base {encoding.share_with!r} "
                f"num_buckets={base_input.encoding.num_buckets}"
            )
    return tuple(inputs)


def _resolved_group(group: TokenGroupConfig) -> ResolvedTokenGroup:
    return ResolvedTokenGroup(name=group.name, input_refs=tuple(group.inputs))


def _resolved_domain_token(token: DomainTokenConfig) -> ResolvedDomainToken:
    return ResolvedDomainToken(
        name=token.name,
        input_refs=tuple(token.resolved_inputs()),
        direct_input_refs=tuple(token.inputs),
        important_input_refs=tuple(token.important_inputs),
        prior_input_refs=tuple(token.prior_inputs),
    )


def _default_domain_inputs(
    explicit_inputs: list[str],
    features: list[FeatureConfig],
    sequences: list[SequenceConfig],
) -> tuple[str, ...]:
    # This keeps the old fallback behavior: first scalar feature, then first sequence.
    if explicit_inputs:
        return tuple(explicit_inputs)
    if features:
        return (features[0].name,)
    if sequences:
        return (sequences[0].name,)
    return ()


def _feature_token_groups(
    tokenization: TokenizationConfig,
    features: list[FeatureConfig],
    sequences: list[SequenceConfig],
) -> tuple[ResolvedTokenGroup, ...]:
    # Explicit groups win. Otherwise each tokenizable input becomes its own group.
    if tokenization.feature_tokens:
        return tuple(_resolved_group(group) for group in tokenization.feature_tokens)
    return tuple(
        ResolvedTokenGroup(name=name, input_refs=(name,))
        for name in tokenizable_input_names(features, sequences)
    )


def _sequence_token_groups(
    tokenization: TokenizationConfig,
    sequences: list[SequenceConfig],
) -> tuple[ResolvedTokenGroup, ...]:
    # OneTrans S tokens default to one token group per active behavior sequence.
    if tokenization.sequence_tokens:
        return tuple(_resolved_group(group) for group in tokenization.sequence_tokens)
    return tuple(
        ResolvedTokenGroup(name=sequence.name, input_refs=(sequence.name,))
        for sequence in sequences
        if sequence.embedding_scope in {"feature", "shared"}
    )


def _scalar_token_groups(
    tokenization: TokenizationConfig,
    features: list[FeatureConfig],
    sequences: list[SequenceConfig],
) -> tuple[ResolvedTokenGroup, ...]:
    # ns_tokens means non-sequence tokens. The YAML name follows OneTrans wording.
    if tokenization.ns_tokens:
        return tuple(_resolved_group(group) for group in tokenization.ns_tokens)
    by_name = {feature.name: feature for feature in features}
    sequence_names = sequence_input_names(sequences)
    if tokenization.feature_tokens:
        # Reuse feature_tokens for NS groups only when every input is scalar.
        return tuple(
            _resolved_group(group)
            for group in tokenization.feature_tokens
            if all(name in by_name and name not in sequence_names for name in group.inputs)
        )
    return tuple(
        ResolvedTokenGroup(name=feature.name, input_refs=(feature.name,))
        for feature in features
        if feature.embedding_scope in {"feature", "shared"}
    )


def _scenario_tokens(
    tokenization: TokenizationConfig,
    features: list[FeatureConfig],
    sequences: list[SequenceConfig],
    scenario_names: list[str],
    scenario_inputs: tuple[str, ...],
) -> tuple[ResolvedDomainToken, ...]:
    # Scenario tokens always include a global token for MDL domain fusion.
    if tokenization.scenario_tokens:
        if not scenario_names:
            # Compatibility path for legacy callers that ask for tokens without AppConfig.
            return tuple(_resolved_domain_token(token) for token in tokenization.scenario_tokens)
        by_name = {token.name: token for token in tokenization.scenario_tokens}
        missing = [name for name in scenario_names if name not in by_name]
        if missing:
            raise ValueError("tokenization.scenario_tokens missing scenarios: " + ", ".join(missing))
        extras = sorted(set(by_name) - set(scenario_names) - {"global"})
        if extras:
            raise ValueError("tokenization.scenario_tokens contains unknown scenarios: " + ", ".join(extras))
        tokens = [_resolved_domain_token(by_name[name]) for name in scenario_names]
        if "global" in by_name:
            tokens.append(_resolved_domain_token(by_name["global"]))
        else:
            tokens.append(
                ResolvedDomainToken(
                    name="global",
                    input_refs=scenario_inputs,
                    direct_input_refs=scenario_inputs,
                )
            )
        return tuple(tokens)

    return tuple(
        ResolvedDomainToken(name=name, input_refs=scenario_inputs, direct_input_refs=scenario_inputs)
        for name in [*scenario_names, "global"]
    )


def _task_tokens(
    tokenization: TokenizationConfig,
    task_names: list[str],
    task_inputs: tuple[str, ...],
) -> tuple[ResolvedDomainToken, ...]:
    # Task tokens map one-to-one with train labels.
    if tokenization.task_tokens:
        if not task_names:
            # Compatibility path for legacy callers that ask for tokens without AppConfig.
            return tuple(_resolved_domain_token(token) for token in tokenization.task_tokens)
        by_name = {token.name: token for token in tokenization.task_tokens}
        missing = [name for name in task_names if name not in by_name]
        if missing:
            raise ValueError("tokenization.task_tokens missing tasks: " + ", ".join(missing))
        extras = sorted(set(by_name) - set(task_names))
        if extras:
            raise ValueError("tokenization.task_tokens contains unknown tasks: " + ", ".join(extras))
        return tuple(_resolved_domain_token(by_name[name]) for name in task_names)

    return tuple(
        ResolvedDomainToken(name=name, input_refs=task_inputs, direct_input_refs=task_inputs)
        for name in task_names
    )


def resolve_tokenization(
    tokenization: TokenizationConfig,
    features: list[FeatureConfig],
    sequences: list[SequenceConfig],
    scenario_names: list[str],
    task_names: list[str],
) -> ResolvedTokenization:
    # Build the complete token layout without changing the raw YAML object.
    feature_token_inputs = tuple(tokenization.feature_token_inputs or tokenizable_input_names(features, sequences))
    feature_groups = _feature_token_groups(tokenization, features, sequences)
    if tokenization.feature_tokenizer in {"auto_split", "rankmixer"}:
        feature_token_count = tokenization.num_feature_tokens or len(feature_token_inputs)
    else:
        feature_token_count = len(feature_groups)
    scenario_inputs = _default_domain_inputs(tokenization.scenario_token_inputs, features, sequences)
    task_inputs = _default_domain_inputs(tokenization.task_token_inputs, features, sequences)
    return ResolvedTokenization(
        feature_tokenizer=tokenization.feature_tokenizer,
        feature_token_inputs=feature_token_inputs,
        feature_token_count=feature_token_count,
        feature_token_groups=feature_groups,
        sequence_token_groups=_sequence_token_groups(tokenization, sequences),
        scalar_token_groups=_scalar_token_groups(tokenization, features, sequences),
        scenario_token_inputs=scenario_inputs,
        task_token_inputs=task_inputs,
        scenario_token_specs=_scenario_tokens(
            tokenization,
            features,
            sequences,
            scenario_names,
            scenario_inputs,
        ),
        task_token_specs=_task_tokens(tokenization, task_names, task_inputs),
        sequence_names=tuple(sequence.name for sequence in sequences),
    )


def resolve_categorical_embedding_dims(
    config: AppConfig,
    categorical_input_by_name: dict[str, ResolvedCategoricalInput] | None = None,
) -> dict[str, int]:
    # Shared vocab entries can also share embedding size; resolve chains once here.
    if categorical_input_by_name is None:
        categorical_inputs = resolve_encoding_strategies(config)
        categorical_input_by_name = {item.name: item for item in categorical_inputs}
    feature_by_name = {feature.name: feature for feature in config.features}
    sequence_fields = {
        field.qualified_name(sequence.name): field
        for sequence in config.sequences
        for field in sequence.fields
        if field.kind == "categorical"
    }
    resolved: dict[str, int] = {}
    resolving: set[str] = set()

    def resolve(name: str) -> int:
        if name in resolved:
            return resolved[name]
        if name in resolving:
            raise ValueError(f"shared_vocab cycle detected at {name!r}")
        resolving.add(name)
        encoding = categorical_input_by_name[name].encoding
        if getattr(encoding, "share_embedding", False):
            share_with = getattr(encoding, "share_with", None)
            if not share_with:
                raise ValueError(
                    f"categorical input {name!r} shares an embedding without share_with"
                )
            dim = resolve(share_with)
        elif name in feature_by_name:
            feature = feature_by_name[name]
            if feature.kind != "categorical":
                raise ValueError(f"vocab strategy {name!r} references non-categorical feature")
            dim = feature.embedding_dim or config.model.embedding_dim
        elif name in sequence_fields:
            field_config = sequence_fields[name]
            dim = field_config.embedding_dim or config.model.embedding_dim
        else:
            raise ValueError(f"vocab strategy references unknown categorical input {name!r}")
        resolving.remove(name)
        resolved[name] = dim
        return dim

    for name in categorical_input_by_name:
        resolve(name)
    return resolved


def resolve_encoded_input_dims(config: AppConfig, categorical_dims: dict[str, int]) -> dict[str, int]:
    # RankMixer validates token packing from these encoded widths.
    dims: dict[str, int] = {}
    for feature in config.features:
        if feature.kind == "dense":
            dims[feature.name] = feature.dimension
        else:
            dims[feature.name] = categorical_dims[feature.name]
    for sequence in config.sequences:
        if sequence.encoder == "longer":
            merged_dim = config.model.token_dim * sequence.longer_token_merge
            dims[sequence.name] = (
                sequence.rankmixer_summary_tokens + sequence.longer_query_tokens
            ) * merged_dim
        else:
            dims[sequence.name] = config.model.token_dim * sequence.rankmixer_summary_tokens
    return dims


def resolve_app_config(config: AppConfig) -> ResolvedConfig:
    # Build all derived values in one place so validation and model setup agree.
    categorical_inputs = resolve_encoding_strategies(config)
    categorical_input_by_name = {item.name: item for item in categorical_inputs}
    categorical_dims = resolve_categorical_embedding_dims(config, categorical_input_by_name)
    return ResolvedConfig(
        tokenization=resolve_tokenization(
            config.tokenization,
            config.features,
            config.sequences,
            config.scenarios.names,
            config.task_names,
        ),
        categorical_embedding_dims=categorical_dims,
        encoded_input_dims=resolve_encoded_input_dims(config, categorical_dims),
        categorical_input_names=set(categorical_input_by_name),
        categorical_inputs=categorical_inputs,
        categorical_input_by_name=categorical_input_by_name,
        scalar_feature_names=tuple(
            feature.name
            for feature in config.features
            if feature.embedding_scope in {"feature", "shared"}
        ),
    )


# Cross-section validation stays outside the raw dataclass declarations.
def _validate_token_group(group: ResolvedTokenGroup, input_names: set[str], section: str) -> None:
    # Token groups may be raw YAML groups or generated defaults; validate both.
    if not group.name:
        raise ValueError(f"tokenization.{section} token name is required")
    if not group.input_refs:
        raise ValueError(f"tokenization.{section}.{group.name} inputs must not be empty")
    missing = [name for name in group.input_refs if name not in input_names]
    if missing:
        raise ValueError(
            f"tokenization.{section}.{group.name} references unknown features: "
            + ", ".join(missing)
        )


def _validate_domain_token(token: ResolvedDomainToken, input_names: set[str], section: str) -> None:
    if not token.name:
        raise ValueError(f"tokenization.{section} token name is required")
    if not token.input_refs:
        raise ValueError(f"tokenization.{section}.{token.name} inputs must not be empty")
    missing = [name for name in token.input_refs if name not in input_names]
    if missing:
        raise ValueError(
            f"tokenization.{section}.{token.name} references unknown features: "
            + ", ".join(missing)
        )


def _validate_unique_group_names(groups: tuple[ResolvedTokenGroup, ...], section: str) -> None:
    names: set[str] = set()
    for group in groups:
        if group.name in names:
            raise ValueError(f"duplicate {section} token name {group.name!r}")
        names.add(group.name)


def _validate_unique_domain_names(tokens: tuple[ResolvedDomainToken, ...], section: str) -> None:
    names: set[str] = set()
    for token in tokens:
        if token.name in names:
            raise ValueError(f"duplicate {section} token name {token.name!r}")
        names.add(token.name)


def validate_tokenization_config(
    tokenization: TokenizationConfig,
    features: list[FeatureConfig],
    sequences: list[SequenceConfig],
    scenario_names: list[str],
    task_names: list[str],
) -> None:
    # Validate both raw token declarations and their resolved defaults.
    input_names = {feature.name for feature in features} | {sequence.name for sequence in sequences}
    sequence_names = sequence_input_names(sequences)
    tokenization._validate_unique_domain_token_names(tokenization.scenario_tokens, "scenario_tokens")
    tokenization._validate_unique_domain_token_names(tokenization.task_tokens, "task_tokens")
    if tokenization.feature_tokenizer not in {"groupwise", "rankmixer", "auto_split"}:
        raise ValueError("tokenization.feature_tokenizer must be groupwise, rankmixer, or auto_split")
    if tokenization.num_feature_tokens is not None and tokenization.num_feature_tokens <= 0:
        raise ValueError("tokenization.num_feature_tokens must be positive")

    resolved = resolve_tokenization(tokenization, features, sequences, scenario_names, task_names)
    if tokenization.feature_tokenizer in {"auto_split", "rankmixer"}:
        if tokenization.feature_tokens:
            raise ValueError("tokenization.feature_tokens cannot be used when feature_tokenizer is auto_split or rankmixer")
        if tokenization.num_feature_tokens is None:
            raise ValueError("tokenization.num_feature_tokens is required when feature_tokenizer is auto_split or rankmixer")
        if not resolved.feature_token_inputs:
            raise ValueError("tokenization.feature_token_inputs must not be empty")
        missing = [name for name in resolved.feature_token_inputs if name not in input_names]
        if missing:
            raise ValueError("tokenization.feature_token_inputs references unknown inputs: " + ", ".join(missing))

    for section, groups in (
        ("feature_tokens", resolved.feature_token_groups),
        ("sequence_tokens", resolved.sequence_token_groups),
        ("ns_tokens", resolved.scalar_token_groups),
    ):
        # Check generated defaults too, not only tokens explicitly written in YAML.
        _validate_unique_group_names(groups, section)
        for group in groups:
            _validate_token_group(group, input_names, section)
            if section == "sequence_tokens" and not any(name in sequence_names for name in group.input_refs):
                raise ValueError(
                    f"tokenization.sequence_tokens.{group.name} must include at least one sequence input"
                )
            if section == "ns_tokens" and any(name in sequence_names for name in group.input_refs):
                raise ValueError(f"tokenization.ns_tokens.{group.name} must not include sequence inputs")

    for section, inputs in (
        ("scenario_token_inputs", resolved.scenario_token_inputs),
        ("task_token_inputs", resolved.task_token_inputs),
    ):
        if not inputs:
            raise ValueError(f"tokenization.{section} must not be empty")
        missing = [name for name in inputs if name not in input_names]
        if missing:
            raise ValueError(f"tokenization.{section} references unknown inputs: " + ", ".join(missing))

    for section, tokens in (
        ("scenario_tokens", resolved.scenario_token_specs),
        ("task_tokens", resolved.task_token_specs),
    ):
        _validate_unique_domain_names(tokens, section)
        for token in tokens:
            _validate_domain_token(token, input_names, section)


def _validate_mdl_extra_embeddings(config: AppConfig, resolved: ResolvedConfig) -> None:
    if config.model.name not in {"mdl_rankmixer", "mdl_onetrans"}:
        return
    feature_by_name = {feature.name: feature for feature in config.features}
    for section, tokens, expected_scope in (
        ("scenario_tokens", resolved.tokenization.scenario_token_specs, "scenario"),
        ("task_tokens", resolved.tokenization.task_token_specs, "task"),
    ):
        for token in tokens:
            for input_name in token.important_input_refs:
                feature = feature_by_name.get(input_name)
                if feature is None or feature.kind != "categorical":
                    raise ValueError(
                        f"tokenization.{section}.{token.name}.important_inputs must reference "
                        "dedicated categorical extra-embedding features"
                    )
                if feature.embedding_scope != expected_scope:
                    raise ValueError(
                        f"important input {input_name!r} for {section}.{token.name} must use "
                        f"embedding_scope={expected_scope!r}, not {feature.embedding_scope!r}"
                    )
                encoding = resolved.categorical_input_by_name[input_name].encoding
                if getattr(encoding, "share_embedding", False):
                    raise ValueError(
                        f"important input {input_name!r} must set share_embedding=false so its "
                        "embedding table is independent from the feature-token embedding"
                    )


def _validate_mdl_domain_priors(config: AppConfig, resolved: ResolvedConfig) -> None:
    """Keep multi-domain MDL priors specific to the token they initialize.

    The public smoke profile has one scenario and one task, where a generic
    history remains a useful compact fixture.  Once a family contains multiple
    tokens, however, accepting only the same generic prior for every token
    weakens the paper's scenario/task tokenization into token identity plus a
    per-token FFN.  Require an explicitly scoped, token-unique prior in that
    case while still allowing additional common inputs.
    """

    if config.model.name != "mdl_rankmixer":
        # MDL-OneTrans is an explicitly experimental composition. Its sequence
        # context is carried by OneTrans S/NS interaction instead of a second
        # set of MDL sequence priors.
        return

    input_scopes = {
        feature.name: feature.embedding_scope for feature in config.features
    }
    input_scopes.update(
        {
            sequence.name: sequence.embedding_scope
            for sequence in config.sequences
        }
    )

    def validate_family(
        section: str,
        tokens: tuple[ResolvedDomainToken, ...],
        names: list[str],
        expected_scope: Literal["scenario", "task"],
        enabled: bool,
    ) -> None:
        if not enabled or len(names) <= 1:
            return

        token_by_name = {token.name: token for token in tokens}
        scoped_priors: dict[str, tuple[str, ...]] = {}
        for name in names:
            token = token_by_name[name]
            if not token.prior_input_refs:
                raise ValueError(
                    f"tokenization.{section}.{name}.prior_inputs must declare a "
                    f"{expected_scope}-related prior when multiple {expected_scope}s are configured"
                )

            wrong_domain_priors = [
                input_name
                for input_name in token.prior_input_refs
                if input_scopes[input_name] in {"scenario", "task"}
                and input_scopes[input_name] != expected_scope
            ]
            if wrong_domain_priors:
                raise ValueError(
                    f"tokenization.{section}.{name}.prior_inputs contains inputs scoped for "
                    f"the other domain family: " + ", ".join(wrong_domain_priors)
                )

            scoped = tuple(
                input_name
                for input_name in token.prior_input_refs
                if input_scopes[input_name] == expected_scope
            )
            if not scoped:
                raise ValueError(
                    f"tokenization.{section}.{name}.prior_inputs must include at least one "
                    f"input with embedding_scope={expected_scope!r}; a generic shared history "
                    "alone is not a paper-aligned domain prior"
                )
            scoped_priors[name] = scoped

        use_counts: dict[str, int] = {}
        for priors in scoped_priors.values():
            for input_name in set(priors):
                use_counts[input_name] = use_counts.get(input_name, 0) + 1
        tokens_without_unique_prior = [
            name
            for name, priors in scoped_priors.items()
            if not any(use_counts[input_name] == 1 for input_name in priors)
        ]
        if tokens_without_unique_prior:
            raise ValueError(
                f"tokenization.{section} must give every {expected_scope} token at least one "
                f"{expected_scope}-scoped prior_input not reused by another token; shared-only "
                "tokens: " + ", ".join(tokens_without_unique_prior)
            )

    validate_family(
        "scenario_tokens",
        resolved.tokenization.scenario_token_specs,
        config.scenarios.names,
        "scenario",
        config.model.use_scenario_tokens,
    )
    validate_family(
        "task_tokens",
        resolved.tokenization.task_token_specs,
        config.task_names,
        "task",
        config.model.use_task_tokens,
    )


def validate_vocab_strategy_references(config: AppConfig) -> None:
    # Every categorical scalar or sequence field needs exactly one encoding source:
    # inline under the logical input, or legacy vocab_strategy.features[name].
    categorical_inputs = resolve_encoding_strategies(config)
    resolve_categorical_embedding_dims(
        config,
        {item.name: item for item in categorical_inputs},
    )


def resolve_onetrans_max_position_embeddings(
    config: AppConfig,
    resolved: ResolvedConfig | None = None,
) -> int:
    """Resolve capacity for OneTrans's unified learned position table."""

    resolved = config.resolved if resolved is None else resolved
    sequence_by_name = {sequence.name: sequence for sequence in config.sequences}
    inferred_s_tokens = 0
    has_dynamic_length = False
    for group in resolved.tokenization.sequence_token_groups:
        group_lengths: list[int] = []
        for input_name in group.input_refs:
            sequence = sequence_by_name.get(input_name)
            if sequence is None:
                continue
            if sequence.max_length is None:
                has_dynamic_length = True
                continue
            group_lengths.append(sequence.max_length)
        if group_lengths:
            inferred_s_tokens += max(group_lengths)

    if (
        config.model.sequence_fusion == "intent_ordered"
        and config.model.use_sep_tokens
    ):
        inferred_s_tokens += max(
            len(resolved.tokenization.sequence_token_groups) - 1,
            0,
        )

    if config.model.ns_tokenizer == "auto_split":
        ns_tokens = config.model.num_ns_tokens or max(
            len(resolved.scalar_feature_names),
            1,
        )
    else:
        ns_tokens = len(resolved.tokenization.scalar_token_groups)
    inferred_total = None if has_dynamic_length else inferred_s_tokens + ns_tokens

    configured = config.model.max_position_embeddings
    if configured is None:
        if inferred_total is None:
            raise ValueError(
                "OneTrans requires model.max_position_embeddings when any S-token "
                "sequence omits max_length"
            )
        return inferred_total
    if inferred_total is not None and configured < inferred_total:
        raise ValueError(
            "model.max_position_embeddings is smaller than the configured OneTrans "
            f"[S; NS] token maximum: {configured} < {inferred_total}"
        )
    return configured


def validate_app_config(config: AppConfig) -> None:
    # Validate local sections first, then cross-section references and model rules.
    config.data.validate()
    if not config.features and not config.sequences:
        raise ValueError("features or sequences must contain at least one model input")
    feature_names: set[str] = set()
    for feature in config.features:
        feature.validate()
        if feature.name in feature_names:
            raise ValueError(f"duplicate feature name {feature.name!r}")
        feature_names.add(feature.name)
    sequence_names: set[str] = set()
    scalar_feature_names = {feature.name for feature in config.features}
    for sequence in config.sequences:
        sequence.validate(scalar_feature_names)
        if sequence.name in sequence_names:
            raise ValueError(f"duplicate sequence name {sequence.name!r}")
        if sequence.name in feature_names:
            raise ValueError(f"sequence name {sequence.name!r} conflicts with a feature name")
        sequence_names.add(sequence.name)
    config.scenarios.validate()
    config.model.validate()
    config.runtime.validate()
    config.training.validate()
    validate_tokenization_config(
        config.tokenization,
        config.features,
        config.sequences,
        config.scenarios.names,
        config.task_names,
    )
    config.vocab_strategy.validate()
    validate_vocab_strategy_references(config)

    resolved = resolve_app_config(config)
    _validate_mdl_extra_embeddings(config, resolved)
    _validate_mdl_domain_priors(config, resolved)
    sequence_by_name = {sequence.name: sequence for sequence in config.sequences}
    if config.model.name in {"onetrans", "mdl_onetrans"} and config.model.sequence_fusion == "timestamp_aware":
        for group in resolved.tokenization.sequence_token_groups:
            for input_name in group.input_refs:
                sequence = sequence_by_name.get(input_name)
                if sequence is not None and sequence.timestamp_field is None:
                    raise ValueError(
                        f"timestamp-aware OneTrans requires sequences.{sequence.name}.timestamp_field; "
                        "use model.sequence_fusion=intent_ordered when timestamps are unavailable"
                    )
    if config.model.name in {"onetrans", "mdl_onetrans"}:
        encoded_sequences = [
            sequence.name for sequence in config.sequences if sequence.encoder != "raw"
        ]
        if encoded_sequences:
            raise ValueError(
                f"model.name={config.model.name!r} requires encoder=raw for every behavior "
                "sequence because OneTrans performs event-level sequence modeling itself; "
                "pre-encoding would recreate an encode-then-interaction path: "
                + ", ".join(encoded_sequences)
            )
    else:
        raw_sequences = [
            sequence.name for sequence in config.sequences if sequence.encoder == "raw"
        ]
        if raw_sequences:
            raise ValueError(
                "encoder=raw delegates sequence modeling to OneTrans and is only valid for "
                "model.name=onetrans or mdl_onetrans: "
                + ", ".join(raw_sequences)
            )
    if config.model.name == "mdl_onetrans":
        s_sequence_names = {
            input_name
            for group in resolved.tokenization.sequence_token_groups
            for input_name in group.input_refs
            if input_name in sequence_by_name
        }
        duplicated_domain_inputs: list[str] = []
        for section, tokens in (
            ("scenario_tokens", resolved.tokenization.scenario_token_specs),
            ("task_tokens", resolved.tokenization.task_token_specs),
        ):
            for token in tokens:
                duplicated = sorted(set(token.input_refs) & s_sequence_names)
                if duplicated:
                    duplicated_domain_inputs.append(
                        f"{section}.{token.name}=" + ",".join(duplicated)
                    )
        if duplicated_domain_inputs:
            raise ValueError(
                "model.name='mdl_onetrans' must model each behavior sequence exactly once as "
                "OneTrans S-tokens; remove those sequences from MDL scenario/task prior_inputs. "
                "Sequence context reaches MDL through the OneTrans-updated NS tokens: "
                + "; ".join(duplicated_domain_inputs)
            )
    for sequence in config.sequences:
        if sequence.encoder == "longer" and sequence.max_length is None:
            raise ValueError(
                f"paper-aligned LONGER sequence {sequence.name!r} requires max_length so recent-k "
                "compression and downstream dimensions are independent of batch composition"
            )
        if sequence.encoder == "longer" and sequence.time_delta_field is None:
            raise ValueError(
                f"paper-aligned LONGER sequence {sequence.name!r} requires time_delta_field; "
                "declare a scalar dense sequence field containing absolute time difference"
            )
    # Model-specific checks use resolved values because defaults affect token counts.
    feature_token_count = resolved.tokenization.feature_token_count
    if feature_token_count <= 0:
        raise ValueError("tokenization must produce at least one feature token")
    if config.model.name == "longer":
        if len(config.sequences) != 1 or config.sequences[0].encoder != "longer":
            raise ValueError(
                "model.name=longer requires exactly one sequence configured with encoder=longer; "
                "put all event side information in that sequence's fields"
            )
        expected_layers = config.sequences[0].longer_self_layers + 1
        if config.model.num_layers != expected_layers:
            raise ValueError(
                "model.name=longer requires model.num_layers to count the cross layer plus "
                "sequences[0].longer_self_layers: "
                f"{config.model.num_layers} != {expected_layers}"
            )
    if config.model.name in {"rankmixer", "mdl_rankmixer"} and config.tokenization.feature_tokenizer == "rankmixer":
        input_dim = sum(resolved.encoded_input_dims[name] for name in resolved.tokenization.feature_token_inputs)
        if input_dim % feature_token_count != 0:
            raise ValueError(
                "rankmixer tokenization requires equal-width input slices: "
                f"sum(feature_token_inputs)={input_dim}, "
                f"num_feature_tokens={feature_token_count}. "
                "The input width must be divisible by the token count; each slice is then "
                "projected independently to model.token_dim."
            )
    if config.model.name in {"rankmixer", "mdl_rankmixer"} and config.model.token_dim % feature_token_count != 0:
        raise ValueError(
            "model.token_dim must be divisible by the resolved feature token count "
            f"for {config.model.name}: {config.model.token_dim} % {feature_token_count} != 0"
        )
    if config.model.name in {"onetrans", "mdl_onetrans"}:
        if not resolved.tokenization.sequence_token_groups:
            raise ValueError(f"model.name={config.model.name!r} requires at least one sequence token")
        if config.model.ns_tokenizer == "auto_split" and not resolved.scalar_feature_names:
            raise ValueError(
                f"model.name={config.model.name!r} with model.ns_tokenizer=auto_split "
                "requires at least one scalar feature with embedding_scope feature or shared"
            )
        if config.model.ns_tokenizer == "groupwise" and not resolved.tokenization.scalar_token_groups:
            raise ValueError(
                f"model.name={config.model.name!r} with model.ns_tokenizer=groupwise "
                "requires tokenization.ns_tokens or scalar feature inputs"
            )
        resolve_onetrans_max_position_embeddings(config, resolved)


def _merge_config_mappings(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _merge_config_mappings(base_value, value)
        else:
            merged[key] = value
    return merged


def _load_config_mapping(
    config_path: Path,
    parents: tuple[Path, ...] = (),
) -> dict[str, Any]:
    resolved_path = config_path.expanduser().resolve()
    if resolved_path in parents:
        cycle = " -> ".join(str(path) for path in (*parents, resolved_path))
        raise ValueError(f"config extends cycle detected: {cycle}")
    with resolved_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"config {resolved_path} must contain a YAML object")
    payload = dict(payload)
    parent_ref = payload.pop("extends", None)
    if parent_ref is None:
        return payload
    if not isinstance(parent_ref, str) or not parent_ref:
        raise ValueError(f"config {resolved_path} extends must be a non-empty path string")
    parent_path = Path(parent_ref)
    if not parent_path.is_absolute():
        parent_path = resolved_path.parent / parent_path
    base = _load_config_mapping(parent_path, (*parents, resolved_path))
    return _merge_config_mappings(base, payload)


def load_app_config(path: str | Path) -> AppConfig:
    """Load YAML, build AppConfig, and validate it before use."""

    config_path = Path(path)
    payload = _load_config_mapping(config_path)
    config = AppConfig.from_mapping(payload)
    config.validate()
    return config
