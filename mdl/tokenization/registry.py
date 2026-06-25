from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class EncoderBuildContext:
    default_embedding_dim: int


class EncoderRegistry:
    def __init__(self) -> None:
        self._builders: dict[str, type] = {}

    def register(self, name: str) -> Callable[[type], type]:
        def decorator(cls: type) -> type:
            if name in self._builders:
                raise ValueError(f"encoder {name!r} is already registered")
            self._builders[name] = cls
            return cls

        return decorator

    def build(self, spec: dict[str, Any], context: EncoderBuildContext) -> Any:
        encoder_name = spec.get("encoder")
        if not encoder_name:
            raise ValueError(f"feature {spec.get('name')!r} is missing an encoder")
        if encoder_name not in self._builders:
            raise ValueError(f"unknown feature encoder {encoder_name!r}")
        return self._builders[encoder_name](spec, context)


DEFAULT_ENCODER_REGISTRY = EncoderRegistry()


def register_encoder(name: str) -> Callable[[type], type]:
    return DEFAULT_ENCODER_REGISTRY.register(name)
