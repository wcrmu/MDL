#!/usr/bin/env python3
"""Compatibility shim — use ``scripts/build_production_configs.py`` instead.

Kept so older docs, imports, and muscle-memory commands keep working while
callers migrate to the name that matches the four-model surface this tool
actually builds.
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

warnings.warn(
    "scripts.build_mdl_rankmixer_config is renamed to "
    "scripts.build_production_configs; update imports and CLI invocations",
    DeprecationWarning,
    stacklevel=2,
)

from scripts.build_production_configs import *  # noqa: E402,F403
from scripts.build_production_configs import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
