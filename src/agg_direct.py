"""Backward-compatible re-exports for the direct agg path.

Implementation lives in ``src.dataloader``. Prefer importing from there for
new code; this module remains so existing ``from src.agg_direct import ...``
call sites keep working.
"""

from __future__ import annotations

from . import dataloader as _dataloader

__all__ = [name for name in dir(_dataloader) if not name.startswith("__")]


def __getattr__(name: str):
    return getattr(_dataloader, name)


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
