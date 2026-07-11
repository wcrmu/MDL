"""Minimal Parquet preprocessing adapter example.

This adapter is intentionally an identity transform. It demonstrates the
callable signature expected by ``format: adapter_parquet`` without encoding any
dataset-specific layout rules in the core repository.
"""

from __future__ import annotations

from typing import Any


def adapt(table: Any, *, context: Any) -> Any:
    """Return the input Arrow table unchanged.

    Real adapters should use ``context.options`` to interpret their environment
    schema and return one or more flat Arrow tables matching ``flat_parquet``:
    one row is one training sample and column names match the YAML sources.
    """

    return table
