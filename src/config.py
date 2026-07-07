from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


AggLayoutType = Literal["parallel_lists", "list_struct", "custom"]
EncodingType = Literal["vocab", "hash", "identity", "shared_vocab"]
EmbeddingScope = Literal["feature", "scenario", "task", "shared"]
ModelName = Literal["mdl_rankmixer", "onetrans", "mdl_onetrans"]
SequenceFieldKind = Literal["categorical", "dense"]
SequenceLayoutType = Literal["parallel_lists", "list_struct"]
SequenceEncoderType = Literal["attention_pool", "mean_pool", "longer"]


@dataclass(frozen=True)
class ReaderConfig:
    engine: str = "pyarrow_dataset"
    columns_pruning: bool = True
    num_workers: int = 0
    prefetch_batches: int = 2
    pin_memory: bool = False
    shard_unit: Literal["file", "row_group", "record_batch"] = "row_group"
    batch_size_rows: int | None = None
    batch_size_candidates: int | None = 2048

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "ReaderConfig":
        if payload is None:
            return cls()
        return cls(**payload)

    def validate(self) -> None:
        if self.engine != "pyarrow_dataset":
            raise ValueError("reader.engine must be 'pyarrow_dataset'")
        if self.num_workers < 0:
            raise ValueError("reader.num_workers must be non-negative")
        if self.prefetch_batches < 0:
            raise ValueError("reader.prefetch_batches must be non-negative")
        if self.batch_size_rows is not None and self.batch_size_rows <= 0:
            raise ValueError("reader.batch_size_rows must be positive")
        if self.batch_size_candidates is not None and self.batch_size_candidates <= 0:
            raise ValueError("reader.batch_size_candidates must be positive")


@dataclass(frozen=True)
class SchemaPolicy:
    require_same_schema: bool = True
    allow_missing_nullable_columns: bool = False
    validate_before_train: bool = True

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "SchemaPolicy":
        if payload is None:
            return cls()
        return cls(**payload)


@dataclass(frozen=True)
class AggLayout:
    type: AggLayoutType
    request_id: str
    shared_columns: list[str] = field(default_factory=list)
    candidate_columns: list[str] = field(default_factory=list)
    candidate_struct_column: str | None = None
    labels: dict[str, str] = field(default_factory=dict)
    label_masks: dict[str, str] = field(default_factory=dict)
    custom_decoder: str | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "AggLayout":
        if not isinstance(payload, dict):
            raise ValueError("data.train.agg_layout must be an object")
        return cls(**payload)

    def validate(self) -> None:
        if self.type not in {"parallel_lists", "list_struct", "custom"}:
            raise ValueError("agg_layout.type must be parallel_lists, list_struct, or custom")
        if not self.request_id:
            raise ValueError("agg_layout.request_id is required")
        if self.type == "parallel_lists" and not self.candidate_columns:
            raise ValueError("parallel_lists agg_layout requires candidate_columns")
        if self.type == "list_struct" and not self.candidate_struct_column:
            raise ValueError("list_struct agg_layout requires candidate_struct_column")
        if self.type == "custom" and not self.custom_decoder:
            raise ValueError("custom agg_layout requires custom_decoder")
        if not self.labels:
            raise ValueError("agg_layout.labels must declare at least one task label")


@dataclass(frozen=True)
class ParquetSplitConfig:
    format: Literal["agg_parquet", "flat_parquet"]
    inputs: list[str]
    reader: ReaderConfig = field(default_factory=ReaderConfig)
    agg_layout: AggLayout | None = None
    request_id: str | None = None
    group_id: str | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ParquetSplitConfig":
        if not isinstance(payload, dict):
            raise ValueError("data split config must be an object")
        reader = ReaderConfig.from_mapping(payload.get("reader"))
        agg_payload = payload.get("agg_layout")
        agg_layout = AggLayout.from_mapping(agg_payload) if agg_payload is not None else None
        inputs = payload.get("inputs")
        if isinstance(inputs, str):
            inputs = [inputs]
        return cls(
            format=payload["format"],
            inputs=list(inputs or []),
            reader=reader,
            agg_layout=agg_layout,
            request_id=payload.get("request_id"),
            group_id=payload.get("group_id"),
        )

    def validate(self, name: str) -> None:
        if self.format not in {"agg_parquet", "flat_parquet"}:
            raise ValueError(f"data.{name}.format must be agg_parquet or flat_parquet")
        if not self.inputs:
            raise ValueError(f"data.{name}.inputs must contain at least one path, glob, or directory")
        self.reader.validate()
        if self.format == "agg_parquet":
            if self.agg_layout is None:
                raise ValueError(f"data.{name}.agg_layout is required for agg_parquet")
            self.agg_layout.validate()
        elif self.agg_layout is not None:
            raise ValueError(f"data.{name}.agg_layout is only valid for agg_parquet")


@dataclass(frozen=True)
class DataConfig:
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


@dataclass(frozen=True)
class FeatureConfig:
    name: str
    # kind="sequence" is the legacy single-column sequence path. New configs
    # should use top-level `sequences` so multi-field behavior steps are explicit.
    kind: Literal["categorical", "dense", "sequence"]
    source: str
    dtype: str | None = None
    embedding_scope: EmbeddingScope = "feature"
    dimension: int = 1
    embedding_dim: int | None = None
    max_length: int | None = None
    truncation: Literal["head", "tail"] = "tail"

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "FeatureConfig":
        return cls(**payload)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("feature.name is required")
        if self.kind not in {"categorical", "dense", "sequence"}:
            raise ValueError(f"feature {self.name!r} kind must be categorical, dense, or sequence")
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
            raise ValueError(f"feature {self.name!r} embedding_dim is only supported for categorical/sequence features")
        if self.max_length is not None and self.max_length <= 0:
            raise ValueError(f"feature {self.name!r} max_length must be positive")
        if self.truncation not in {"head", "tail"}:
            raise ValueError(f"feature {self.name!r} truncation must be head or tail")


@dataclass(frozen=True)
class SequenceFieldConfig:
    name: str
    kind: SequenceFieldKind
    source: str
    dtype: str | None = None
    dimension: int = 1
    embedding_dim: int | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "SequenceFieldConfig":
        return cls(**payload)

    def qualified_name(self, sequence_name: str) -> str:
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


@dataclass(frozen=True)
class SequenceConfig:
    name: str
    fields: list[SequenceFieldConfig]
    source: str | None = None
    layout: SequenceLayoutType = "parallel_lists"
    embedding_scope: EmbeddingScope = "feature"
    max_length: int | None = None
    truncation: Literal["head", "tail"] = "tail"
    encoder: SequenceEncoderType = "attention_pool"
    target_inputs: list[str] = field(default_factory=list)
    rankmixer_summary_tokens: int = 1
    longer_query_tokens: int = 32
    longer_self_layers: int = 1
    longer_token_merge: int = 1
    longer_inner_layers: int = 0
    time_delta_field: str | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "SequenceConfig":
        return cls(
            name=payload["name"],
            fields=[
                SequenceFieldConfig.from_mapping(item)
                for item in payload.get("fields", [])
            ],
            source=payload.get("source"),
            layout=payload.get("layout", "parallel_lists"),
            embedding_scope=payload.get("embedding_scope", "feature"),
            max_length=payload.get("max_length"),
            truncation=payload.get("truncation", "tail"),
            encoder=payload.get("encoder", "attention_pool"),
            target_inputs=list(payload.get("target_inputs", [])),
            rankmixer_summary_tokens=payload.get("rankmixer_summary_tokens", 1),
            longer_query_tokens=payload.get("longer_query_tokens", 32),
            longer_self_layers=payload.get("longer_self_layers", 1),
            longer_token_merge=payload.get("longer_token_merge", 1),
            longer_inner_layers=payload.get("longer_inner_layers", 0),
            time_delta_field=payload.get("time_delta_field"),
        )

    def validate(self, scalar_feature_names: set[str]) -> None:
        if not self.name:
            raise ValueError("sequence.name is required")
        if "." in self.name:
            raise ValueError(f"sequence name {self.name!r} must not contain '.'")
        if self.layout not in {"parallel_lists", "list_struct"}:
            raise ValueError(f"sequence {self.name!r} layout must be parallel_lists or list_struct")
        if self.layout == "list_struct" and not self.source:
            raise ValueError(f"sequence {self.name!r} source is required for list_struct layout")
        if self.embedding_scope not in {"feature", "scenario", "task", "shared"}:
            raise ValueError(
                f"sequence {self.name!r} embedding_scope must be feature, scenario, task, or shared"
            )
        if self.max_length is not None and self.max_length <= 0:
            raise ValueError(f"sequence {self.name!r} max_length must be positive")
        if self.truncation not in {"head", "tail"}:
            raise ValueError(f"sequence {self.name!r} truncation must be head or tail")
        if self.encoder not in {"attention_pool", "mean_pool", "longer"}:
            raise ValueError(f"sequence {self.name!r} encoder must be attention_pool, mean_pool, or longer")
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
        if not self.fields:
            raise ValueError(f"sequence {self.name!r} must declare at least one field")
        field_names: set[str] = set()
        for field_config in self.fields:
            field_config.validate(self.name)
            if field_config.name in field_names:
                raise ValueError(f"duplicate field {field_config.name!r} in sequence {self.name!r}")
            field_names.add(field_config.name)
        if self.time_delta_field is not None:
            by_field = {field.name: field for field in self.fields}
            if self.time_delta_field not in by_field:
                raise ValueError(f"sequence {self.name!r} time_delta_field references unknown field")
            if by_field[self.time_delta_field].kind != "dense":
                raise ValueError(f"sequence {self.name!r} time_delta_field must reference a dense field")
        missing_targets = [name for name in self.target_inputs if name not in scalar_feature_names]
        if missing_targets:
            raise ValueError(
                f"sequence {self.name!r} target_inputs references unknown scalar features: "
                + ", ".join(missing_targets)
            )


@dataclass(frozen=True)
class ScenarioConfig:
    names: list[str] = field(default_factory=lambda: ["default"])
    source: str | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "ScenarioConfig":
        if payload is None:
            return cls()
        return cls(**payload)

    def validate(self) -> None:
        if not self.names:
            raise ValueError("scenarios.names must contain at least one scenario")
        if any(not name for name in self.names):
            raise ValueError("scenarios.names must not contain empty names")
        if len(set(self.names)) != len(self.names):
            raise ValueError("scenarios.names must not contain duplicates")


@dataclass(frozen=True)
class TokenGroupConfig:
    name: str
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
    name: str
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
    feature_tokenizer: Literal["groupwise", "rankmixer", "auto_split"] = "groupwise"
    num_feature_tokens: int | None = None
    feature_token_inputs: list[str] = field(default_factory=list)
    feature_tokens: list[TokenGroupConfig] = field(default_factory=list)
    sequence_tokens: list[TokenGroupConfig] = field(default_factory=list)
    ns_tokens: list[TokenGroupConfig] = field(default_factory=list)
    scenario_tokens: list[DomainTokenConfig] = field(default_factory=list)
    task_tokens: list[DomainTokenConfig] = field(default_factory=list)
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
        resolved_sequences = self._sequences(sequences)
        return [
            feature.name
            for feature in features
            if feature.embedding_scope in {"feature", "shared"}
        ] + [
            sequence.name
            for sequence in resolved_sequences
            if sequence.embedding_scope in {"feature", "shared"}
        ]

    def _sequence_input_names(
        self,
        features: list[FeatureConfig],
        sequences: list[SequenceConfig] | None = None,
    ) -> set[str]:
        resolved_sequences = self._sequences(sequences)
        return {
            feature.name
            for feature in features
            if feature.kind == "sequence" and feature.embedding_scope in {"feature", "shared"}
        } | {
            sequence.name
            for sequence in resolved_sequences
            if sequence.embedding_scope in {"feature", "shared"}
        }

    def resolved_feature_token_inputs(
        self,
        features: list[FeatureConfig],
        sequences: list[SequenceConfig] | None = None,
    ) -> list[str]:
        if self.feature_token_inputs:
            return self.feature_token_inputs
        return self._tokenizable_input_names(features, sequences)

    def resolved_feature_token_count(
        self,
        features: list[FeatureConfig],
        sequences: list[SequenceConfig] | None = None,
    ) -> int:
        if self.feature_tokenizer in {"auto_split", "rankmixer"}:
            if self.num_feature_tokens is None:
                return len(self.resolved_feature_token_inputs(features, sequences))
            return self.num_feature_tokens
        return len(self.resolved_feature_tokens(features, sequences))

    def resolved_feature_tokens(
        self,
        features: list[FeatureConfig],
        sequences: list[SequenceConfig] | None = None,
    ) -> list[TokenGroupConfig]:
        if self.feature_tokens:
            return self.feature_tokens
        return [
            TokenGroupConfig(name=name, inputs=[name])
            for name in self._tokenizable_input_names(features, sequences)
        ]

    def resolved_sequence_tokens(
        self,
        features: list[FeatureConfig],
        sequences: list[SequenceConfig] | None = None,
    ) -> list[TokenGroupConfig]:
        if self.sequence_tokens:
            return self.sequence_tokens
        return [
            TokenGroupConfig(name=feature.name, inputs=[feature.name])
            for feature in features
            if feature.kind == "sequence" and feature.embedding_scope in {"feature", "shared"}
        ] + [
            TokenGroupConfig(name=sequence.name, inputs=[sequence.name])
            for sequence in self._sequences(sequences)
            if sequence.embedding_scope in {"feature", "shared"}
        ]

    def resolved_ns_tokens(
        self,
        features: list[FeatureConfig],
        sequences: list[SequenceConfig] | None = None,
    ) -> list[TokenGroupConfig]:
        if self.ns_tokens:
            return self.ns_tokens
        by_name = {feature.name: feature for feature in features}
        sequence_names = self._sequence_input_names(features, sequences)
        if self.feature_tokens:
            return [
                token
                for token in self.feature_tokens
                if all(
                    name in by_name
                    and by_name[name].kind != "sequence"
                    and name not in sequence_names
                    for name in token.inputs
                )
            ]
        return [
            TokenGroupConfig(name=feature.name, inputs=[feature.name])
            for feature in features
            if feature.kind != "sequence" and feature.embedding_scope in {"feature", "shared"}
        ]

    def resolved_scenario_inputs(
        self,
        features: list[FeatureConfig],
        sequences: list[SequenceConfig] | None = None,
    ) -> list[str]:
        if self.scenario_token_inputs:
            return self.scenario_token_inputs
        if features:
            return [features[0].name]
        resolved_sequences = self._sequences(sequences)
        return [resolved_sequences[0].name] if resolved_sequences else []

    def resolved_task_inputs(
        self,
        features: list[FeatureConfig],
        sequences: list[SequenceConfig] | None = None,
    ) -> list[str]:
        if self.task_token_inputs:
            return self.task_token_inputs
        if features:
            return [features[0].name]
        resolved_sequences = self._sequences(sequences)
        return [resolved_sequences[0].name] if resolved_sequences else []

    def resolved_scenario_tokens(
        self,
        features: list[FeatureConfig],
        scenario_names: list[str],
        sequences: list[SequenceConfig] | None = None,
    ) -> list[DomainTokenConfig]:
        if self.scenario_tokens:
            by_name = {token.name: token for token in self.scenario_tokens}
            tokens = [by_name[name] for name in scenario_names if name in by_name]
            missing = [name for name in scenario_names if name not in by_name]
            if missing:
                raise ValueError("tokenization.scenario_tokens missing scenarios: " + ", ".join(missing))
            extras = sorted(set(by_name) - set(scenario_names) - {"global"})
            if extras:
                raise ValueError("tokenization.scenario_tokens contains unknown scenarios: " + ", ".join(extras))
            if "global" in by_name:
                tokens.append(by_name["global"])
            else:
                tokens.append(
                    DomainTokenConfig(
                        name="global",
                        inputs=self.resolved_scenario_inputs(features, sequences),
                    )
                )
            return tokens

        inputs = self.resolved_scenario_inputs(features, sequences)
        return [
            DomainTokenConfig(name=name, inputs=inputs)
            for name in scenario_names
        ] + [DomainTokenConfig(name="global", inputs=inputs)]

    def resolved_task_tokens(
        self,
        features: list[FeatureConfig],
        task_names: list[str],
        sequences: list[SequenceConfig] | None = None,
    ) -> list[DomainTokenConfig]:
        if self.task_tokens:
            by_name = {token.name: token for token in self.task_tokens}
            missing = [name for name in task_names if name not in by_name]
            if missing:
                raise ValueError("tokenization.task_tokens missing tasks: " + ", ".join(missing))
            extras = sorted(set(by_name) - set(task_names))
            if extras:
                raise ValueError("tokenization.task_tokens contains unknown tasks: " + ", ".join(extras))
            return [by_name[name] for name in task_names]

        inputs = self.resolved_task_inputs(features, sequences)
        return [DomainTokenConfig(name=name, inputs=inputs) for name in task_names]

    def validate(
        self,
        features: list[FeatureConfig],
        sequences: list[SequenceConfig],
        scenario_names: list[str],
        task_names: list[str],
    ) -> None:
        input_names = {feature.name for feature in features} | {sequence.name for sequence in sequences}
        sequence_input_names = self._sequence_input_names(features, sequences)
        if self.feature_tokenizer not in {"groupwise", "rankmixer", "auto_split"}:
            raise ValueError("tokenization.feature_tokenizer must be groupwise, rankmixer, or auto_split")
        if self.num_feature_tokens is not None and self.num_feature_tokens <= 0:
            raise ValueError("tokenization.num_feature_tokens must be positive")
        if self.feature_tokenizer in {"auto_split", "rankmixer"}:
            if self.feature_tokens:
                raise ValueError("tokenization.feature_tokens cannot be used when feature_tokenizer is auto_split or rankmixer")
            if self.num_feature_tokens is None:
                raise ValueError("tokenization.num_feature_tokens is required when feature_tokenizer is auto_split or rankmixer")
            inputs = self.resolved_feature_token_inputs(features, sequences)
            if not inputs:
                raise ValueError("tokenization.feature_token_inputs must not be empty")
            missing = [name for name in inputs if name not in input_names]
            if missing:
                raise ValueError("tokenization.feature_token_inputs references unknown inputs: " + ", ".join(missing))
        for section, tokens in (
            ("feature_tokens", self.resolved_feature_tokens(features, sequences)),
            ("sequence_tokens", self.resolved_sequence_tokens(features, sequences)),
            ("ns_tokens", self.resolved_ns_tokens(features, sequences)),
        ):
            token_names: set[str] = set()
            for token in tokens:
                token.validate(input_names, section)
                if token.name in token_names:
                    raise ValueError(f"duplicate {section} token name {token.name!r}")
                token_names.add(token.name)
                if section == "sequence_tokens" and not any(name in sequence_input_names for name in token.inputs):
                    raise ValueError(
                        f"tokenization.sequence_tokens.{token.name} must include at least one sequence input"
                    )
                if section == "ns_tokens" and any(name in sequence_input_names for name in token.inputs):
                    raise ValueError(
                        f"tokenization.ns_tokens.{token.name} must not include sequence inputs"
                    )
        for section, inputs in (
            ("scenario_token_inputs", self.resolved_scenario_inputs(features, sequences)),
            ("task_token_inputs", self.resolved_task_inputs(features, sequences)),
        ):
            if not inputs:
                raise ValueError(f"tokenization.{section} must not be empty")
            missing = [name for name in inputs if name not in input_names]
            if missing:
                raise ValueError(f"tokenization.{section} references unknown inputs: " + ", ".join(missing))
        for section, tokens in (
            ("scenario_tokens", self.resolved_scenario_tokens(features, scenario_names, sequences)),
            ("task_tokens", self.resolved_task_tokens(features, task_names, sequences)),
        ):
            token_names: set[str] = set()
            for token in tokens:
                token.validate(input_names, section)
                if token.name in token_names:
                    raise ValueError(f"duplicate {section} token name {token.name!r}")
                token_names.add(token.name)


@dataclass(frozen=True)
class VocabDefaults:
    fit_split: str = "train"
    oov_id: int = 0
    padding_id: int = 0
    unseen_policy: Literal["oov", "error"] = "oov"
    artifact_dir: str = "artifacts/vocab"

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "VocabDefaults":
        if payload is None:
            return cls()
        return cls(**payload)


@dataclass(frozen=True)
class VocabFeatureStrategy:
    encoding: EncodingType
    source: str
    min_count: int | None = None
    max_size: int | None = None
    artifact: str | None = None
    num_buckets: int | None = None
    salt: str | None = None
    max_id: int | None = None
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
            if self.max_id is None or self.max_id <= 0:
                raise ValueError(f"identity feature {feature_name!r} requires positive max_id")
        if self.encoding == "shared_vocab" and not self.share_with:
            raise ValueError(f"shared_vocab feature {feature_name!r} requires share_with")


@dataclass(frozen=True)
class VocabStrategy:
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
        for name, strategy in self.features.items():
            if strategy.encoding == "shared_vocab" and strategy.share_with not in self.features:
                raise ValueError(
                    f"shared_vocab feature {name!r} references unknown feature {strategy.share_with!r}"
                )


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cpu"
    precision: Literal["fp32", "bf16", "fp16"] = "fp32"
    compile: bool = False
    activation_checkpoint: bool = False
    attention_backend: Literal["auto", "sdpa", "flash"] = "auto"
    distributed: Literal["none", "ddp"] = "none"
    nproc_per_node: int | None = None
    master_addr: str = "127.0.0.1"
    master_port: int = 29500

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "RuntimeConfig":
        if payload is None:
            return cls()
        return cls(**payload)

    def validate(self) -> None:
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("runtime.precision must be fp32, bf16, or fp16")
        if self.attention_backend not in {"auto", "sdpa", "flash"}:
            raise ValueError("runtime.attention_backend must be auto, sdpa, or flash")
        if self.distributed not in {"none", "ddp"}:
            raise ValueError("runtime.distributed must be none or ddp")
        if self.nproc_per_node is not None and self.nproc_per_node <= 0:
            raise ValueError("runtime.nproc_per_node must be positive")
        if not 1 <= self.master_port <= 65535:
            raise ValueError("runtime.master_port must be in [1, 65535]")


@dataclass(frozen=True)
class ModelConfig:
    name: ModelName
    embedding_dim: int = 32
    token_dim: int = 768
    num_layers: int = 6
    num_heads: int = 12
    hidden_dim: int = 1536
    use_task_tokens: bool = True
    use_scenario_tokens: bool = True
    use_global_scenario_token: bool = True
    use_task_feature_interaction: bool = True
    use_scenario_feature_interaction: bool = True
    use_request_cache: bool = False
    use_pyramid: bool = True
    pyramid_round_to: int = 32
    ns_tokenizer: Literal["auto_split", "groupwise"] = "auto_split"
    num_ns_tokens: int | None = None
    use_sep_tokens: bool = True
    final_s_tokens: int | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ModelConfig":
        if not isinstance(payload, dict):
            raise ValueError("model must be an object")
        return cls(**payload)

    def validate(self) -> None:
        if self.name not in {"mdl_rankmixer", "onetrans", "mdl_onetrans"}:
            raise ValueError("model.name must be mdl_rankmixer, onetrans, or mdl_onetrans")
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
        if self.pyramid_round_to <= 0:
            raise ValueError("model.pyramid_round_to must be positive")
        if self.ns_tokenizer not in {"auto_split", "groupwise"}:
            raise ValueError("model.ns_tokenizer must be auto_split or groupwise")
        if self.num_ns_tokens is not None and self.num_ns_tokens <= 0:
            raise ValueError("model.num_ns_tokens must be positive")
        if self.final_s_tokens is not None and self.final_s_tokens < 0:
            raise ValueError("model.final_s_tokens must be non-negative")


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 2048
    lr_dense: float = 0.005
    lr_sparse: float | None = None
    dense_optimizer: Literal["rmsprop"] = "rmsprop"
    sparse_optimizer: Literal["adagrad"] = "adagrad"
    embedding_sparse_gradients: bool = True
    sparse_update_mode: Literal["ddp_synced_adagrad", "external_parameter_server"] = "ddp_synced_adagrad"
    sparse_parameter_server_adapter: str | None = None
    dense_clip_norm: float | None = None
    sparse_clip_norm: float | None = None
    checkpoint_path: str | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "TrainingConfig":
        if payload is None:
            return cls()
        return cls(**payload)

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("training.batch_size must be positive")
        if self.lr_dense <= 0:
            raise ValueError("training.lr_dense must be positive")
        if self.lr_sparse is not None and self.lr_sparse <= 0:
            raise ValueError("training.lr_sparse must be positive")
        if self.dense_optimizer != "rmsprop":
            raise ValueError("training.dense_optimizer must be rmsprop for paper alignment")
        if self.sparse_optimizer != "adagrad":
            raise ValueError("training.sparse_optimizer must be adagrad for paper alignment")
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


@dataclass(frozen=True)
class AppConfig:
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
        labels = self.data.train.agg_layout.labels if self.data.train.agg_layout is not None else {}
        return list(labels.keys()) or ["default"]

    def _encoded_input_dims(self) -> dict[str, int]:
        dims: dict[str, int] = {}
        for feature in self.features:
            if feature.kind == "dense":
                dims[feature.name] = feature.dimension
            elif feature.kind == "sequence":
                dims[feature.name] = feature.embedding_dim or self.model.embedding_dim
            else:
                dims[feature.name] = feature.embedding_dim or self.model.embedding_dim
        for sequence in self.sequences:
            dims[sequence.name] = self.model.token_dim * sequence.rankmixer_summary_tokens
        return dims

    def validate(self) -> None:
        self.data.validate()
        if not self.features and not self.sequences:
            raise ValueError("features or sequences must contain at least one model input")
        feature_names: set[str] = set()
        for feature in self.features:
            feature.validate()
            if feature.name in feature_names:
                raise ValueError(f"duplicate feature name {feature.name!r}")
            feature_names.add(feature.name)
        sequence_names: set[str] = set()
        scalar_feature_names = {feature.name for feature in self.features if feature.kind != "sequence"}
        for sequence in self.sequences:
            sequence.validate(scalar_feature_names)
            if sequence.name in sequence_names:
                raise ValueError(f"duplicate sequence name {sequence.name!r}")
            if sequence.name in feature_names:
                raise ValueError(f"sequence name {sequence.name!r} conflicts with a feature name")
            sequence_names.add(sequence.name)
        self.scenarios.validate()
        self.tokenization.validate(self.features, self.sequences, self.scenarios.names, self.task_names)
        self.vocab_strategy.validate()
        self.model.validate()
        feature_token_count = self.tokenization.resolved_feature_token_count(self.features, self.sequences)
        if feature_token_count <= 0:
            raise ValueError("tokenization must produce at least one feature token")
        if self.tokenization.feature_tokenizer == "rankmixer":
            input_dims = self._encoded_input_dims()
            feature_token_inputs = self.tokenization.resolved_feature_token_inputs(self.features, self.sequences)
            input_dim = sum(input_dims[name] for name in feature_token_inputs)
            target_dim = feature_token_count * self.model.token_dim
            if input_dim != target_dim:
                raise ValueError(
                    "rankmixer tokenization requires exact encoded input dimension: "
                    f"sum(feature_token_inputs)={input_dim}, "
                    f"num_feature_tokens * model.token_dim={feature_token_count} * "
                    f"{self.model.token_dim} = {target_dim}. "
                    "Implicit zero padding is disabled; adjust secure-environment "
                    "feature dimensions, token packing, or tokenization dimensions."
                )
        if self.model.name == "mdl_rankmixer" and self.model.token_dim % feature_token_count != 0:
            raise ValueError(
                "model.token_dim must be divisible by the resolved feature token count "
                f"for mdl_rankmixer: {self.model.token_dim} % {feature_token_count} != 0"
            )
        self.runtime.validate()
        self.training.validate()


def load_app_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    config = AppConfig.from_mapping(payload)
    config.validate()
    return config
