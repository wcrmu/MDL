#!/usr/bin/env python3
"""Compare sequential versus per-dtype parallel packed-buffer copies."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_app_config
from src.dataloader import _coalesce_feature_batch, _map_feature_value
from src.features import load_vocab_maps
from src.train import _iter_batch_tables, _prepare_feature_batch


def _leaves(batch):
    result = []
    seen = set()

    def collect(tensor):
        if id(tensor) not in seen:
            seen.add(id(tensor))
            result.append(tensor)
        return tensor

    for value in batch.features.values():
        _map_feature_value(value, collect)
    for value in (batch.labels, batch.label_mask, batch.scenario_id):
        if isinstance(value, torch.Tensor):
            collect(value)
    return result


def _parallel_copy(batch):
    grouped = defaultdict(list)
    for tensor in _leaves(batch):
        grouped[tensor.dtype].append(tensor)

    def copy_group(item):
        dtype, tensors = item
        total = sum(tensor.numel() for tensor in tensors)
        element_size = torch.empty((), dtype=dtype).element_size()
        storage = torch.UntypedStorage._new_shared(total * element_size, device="cpu")
        buffer = torch.empty(0, dtype=dtype).set_(storage, 0, (total,), (1,))
        target = buffer.numpy()
        offset = 0
        for tensor in tensors:
            source = tensor.detach().numpy().reshape(-1)
            count = source.size
            target[offset : offset + count] = source
            offset += count
        return buffer

    with ThreadPoolExecutor(max_workers=len(grouped)) as executor:
        return list(executor.map(copy_group, grouped.items()))


def _concatenate_copy(batch):
    grouped = defaultdict(list)
    for tensor in _leaves(batch):
        grouped[tensor.dtype].append(tensor)
    buffers = []
    for dtype, tensors in grouped.items():
        total = sum(tensor.numel() for tensor in tensors)
        element_size = torch.empty((), dtype=dtype).element_size()
        storage = torch.UntypedStorage._new_shared(total * element_size, device="cpu")
        buffer = torch.empty(0, dtype=dtype).set_(storage, 0, (total,), (1,))
        sources = [tensor.detach().numpy().reshape(-1) for tensor in tensors]
        np.concatenate(sources, out=buffer.numpy())
        buffers.append(buffer)
    return buffers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    config = load_app_config(args.config)
    split = config.data.train
    assert split is not None
    vocab = load_vocab_maps(config)
    iterator = _iter_batch_tables(
        config, "train", shard_rank=0, shard_world_size=1, require_labels=True
    )
    try:
        table = None
        for _ in range(args.warmup + 1):
            table = next(iterator)
        batch = _prepare_feature_batch(
            config, split, table, vocab, True, False, False, False
        )
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
    grouped = defaultdict(int)
    for tensor in _leaves(batch):
        grouped[str(tensor.dtype)] += tensor.numel() * tensor.element_size()
    print("bytes_by_dtype", dict(grouped))
    modes = {
        "sequential": lambda: _coalesce_feature_batch(
            batch, pin_memory=False, shared_memory=True
        ),
        "parallel": lambda: _parallel_copy(batch),
        "concatenate": lambda: _concatenate_copy(batch),
    }
    for function in modes.values():
        function()
    for name, function in modes.items():
        started = perf_counter()
        for _ in range(args.repeats):
            function()
        print(f"{name}_ms={(perf_counter() - started) / args.repeats * 1000:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
