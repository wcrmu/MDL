"""Start a forkserver before this process initializes CUDA.

Linux ``spawn`` is ``vfork`` plus exec. ``vfork`` freezes every thread in the
caller until the child execs, so doing it from the training process after
NCCL and the step watchdog are up sticks the whole job on
``Process.start()``. The server started here is a fresh interpreter with an
empty preload: it never imports the training program and never touches CUDA.
Later children are ordinary forks inside that server.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Any

_CONTEXT: Any | None = None


def _noop() -> None:
    """Picklable target used only to finish forkserver startup."""


def _inheriting() -> bool:
    return bool(getattr(mp.current_process(), "_inheriting", False))


def warm_forkserver() -> Any:
    """Return the forkserver context, starting the server on the first call."""

    global _CONTEXT
    if _CONTEXT is not None:
        return _CONTEXT
    ctx = mp.get_context("forkserver")
    # A host-prepare child re-imports this module while it is still being
    # spawned. Starting another server from there deadlocks the spawn pipe.
    if _inheriting() or os.environ.get("MDL_FORKSERVER_WARMED") == "1":
        _CONTEXT = ctx
        return _CONTEXT
    if "forkserver" not in mp.get_all_start_methods():
        raise RuntimeError("host-prepare requires the forkserver start method")
    # Inherited by the server and by every child forked from it.
    os.environ["MDL_FORKSERVER_WARMED"] = "1"
    # Conda mkl-service aborts a forkserver child after libgomp is loaded.
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
    # Default preload is __main__, which pulls torch into the server. Keep the
    # server empty so its fork cannot copy a CUDA context or an OpenMP pool.
    ctx.set_forkserver_preload([])
    rank = os.environ.get("RANK", "0")
    if rank in {"0", ""}:
        print("host-prepare start method=forkserver warming", flush=True)
    proc = ctx.Process(target=_noop, name="mdl-forkserver-warmup")
    proc.start()
    proc.join(timeout=120)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
        os.environ.pop("MDL_FORKSERVER_WARMED", None)
        raise RuntimeError("forkserver warmup did not finish within 120s")
    if proc.exitcode != 0:
        os.environ.pop("MDL_FORKSERVER_WARMED", None)
        raise RuntimeError(f"forkserver warmup exited {proc.exitcode}")
    _CONTEXT = ctx
    if rank in {"0", ""}:
        print("host-prepare start method=forkserver ready", flush=True)
    return _CONTEXT


def warm_forkserver_before_imports() -> None:
    """Warm from ``src/main.py`` before that module imports torch.

    Torchrun workers already have ``LOCAL_RANK``. The outer launcher does not,
    and it only re-execs torchrun, so it must not start a server of its own.
    """

    if "LOCAL_RANK" not in os.environ and os.environ.get("MDL_DDP_LAUNCHED") != "1":
        return
    warm_forkserver()
