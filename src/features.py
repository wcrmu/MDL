from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Iterable

from .config import VocabStrategy


@dataclass(frozen=True)
class VocabArtifactRef:
    feature_name: str
    encoding: str
    artifact_path: str | None
    size_hint: int | None = None


def vocab_strategy_fingerprint(strategy: VocabStrategy) -> str:
    payload = {
        "defaults": {
            "fit_split": strategy.defaults.fit_split,
            "oov_id": strategy.defaults.oov_id,
            "padding_id": strategy.defaults.padding_id,
            "unseen_policy": strategy.defaults.unseen_policy,
            "artifact_dir": strategy.defaults.artifact_dir,
        },
        "features": {
            name: {
                "encoding": feature.encoding,
                "source": feature.source,
                "min_count": feature.min_count,
                "max_size": feature.max_size,
                "artifact": feature.artifact,
                "num_buckets": feature.num_buckets,
                "salt": feature.salt,
                "max_id": feature.max_id,
                "share_with": feature.share_with,
            }
            for name, feature in sorted(strategy.features.items())
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def stable_hash_bucket(value: Any, num_buckets: int, salt: str | None = None) -> int:
    if num_buckets <= 0:
        raise ValueError("num_buckets must be positive")
    prefix = "" if salt is None else f"{salt}:"
    digest = sha256((prefix + str(value)).encode("utf-8")).digest()
    # Bucket ids start at 1 because 0 is reserved for OOV/padding.
    return int.from_bytes(digest[:8], "little") % num_buckets + 1


def stable_hash_buckets(values: Iterable[Any], num_buckets: int, salt: str | None = None) -> list[int]:
    return [stable_hash_bucket(value, num_buckets, salt) for value in values]


def vocab_artifacts(strategy: VocabStrategy) -> list[VocabArtifactRef]:
    refs: list[VocabArtifactRef] = []
    artifact_dir = strategy.defaults.artifact_dir.rstrip("/")
    for feature_name, feature_strategy in sorted(strategy.features.items()):
        artifact_path = None
        if feature_strategy.artifact:
            artifact_path = f"{artifact_dir}/{feature_strategy.artifact}"
        size_hint = feature_strategy.max_size or feature_strategy.num_buckets or feature_strategy.max_id
        refs.append(
            VocabArtifactRef(
                feature_name=feature_name,
                encoding=feature_strategy.encoding,
                artifact_path=artifact_path,
                size_hint=size_hint,
            )
        )
    return refs
