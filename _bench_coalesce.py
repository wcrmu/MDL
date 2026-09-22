"""Profile coalesce leaf count and numpy vs torch copy."""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import torch

from src.config import load_app_config
import src.dataloader as dl
import src.train as T


def main() -> None:
    cfg = load_app_config("artifacts/bench_b512_direct.yaml")
    reader = replace(
        cfg.data.train.reader, adapter_workers=3, pin_memory=False
    )
    c = replace(
        cfg, data=replace(cfg.data, train=replace(cfg.data.train, reader=reader))
    )
    it = T.iter_feature_batches(
        c, "train", {}, require_labels=True, pin_memory=False
    )
    for _ in range(3):
        next(it)
    batch = next(it)

    leaves = []
    seen = set()

    def collect(t):
        if id(t) not in seen:
            seen.add(id(t))
            leaves.append(t)
        return t

    for value in batch.features.values():
        dl._map_feature_value(value, collect)
    for value in (batch.labels, batch.label_mask, batch.scenario_id):
        if isinstance(value, torch.Tensor):
            collect(value)
    print("n_leaves", len(leaves))
    by = {}
    for t in leaves:
        by.setdefault(t.dtype, []).append(t)
    for dt, ts in by.items():
        print(dt, len(ts), "numel", sum(t.numel() for t in ts))

    # time coalesce variants
    def torch_copy():
        return dl._coalesce_feature_batch(batch, pin_memory=True)

    def numpy_copy():
        from collections import defaultdict

        leaves2 = []
        seen2 = set()

        def collect2(t):
            if id(t) not in seen2:
                seen2.add(id(t))
                leaves2.append(t)
            return t

        for value in batch.features.values():
            dl._map_feature_value(value, collect2)
        for value in (batch.labels, batch.label_mask, batch.scenario_id):
            if isinstance(value, torch.Tensor):
                collect2(value)
        by_dtype = defaultdict(list)
        for tensor in leaves2:
            by_dtype[tensor.dtype].append(tensor)
        replacements = {}
        buffers = []
        for dtype, tensors in by_dtype.items():
            total = sum(tensor.numel() for tensor in tensors)
            buffer = torch.empty(total, dtype=dtype, pin_memory=True)
            buffers.append(buffer)
            buf_np = buffer.numpy()
            offset = 0
            for tensor in tensors:
                count = tensor.numel()
                src = tensor.detach().numpy().reshape(-1)
                if src.dtype != buf_np.dtype:
                    src = src.astype(buf_np.dtype, copy=False)
                buf_np[offset : offset + count] = src
                replacements[id(tensor)] = buffer.narrow(0, offset, count).view(
                    tensor.shape
                )
                offset += count

        def replace_tensor(tensor):
            return replacements[id(tensor)]

        return dl.FeatureBatch(
            features={
                k: dl._map_feature_value(v, replace_tensor)
                for k, v in batch.features.items()
            },
            labels=None if batch.labels is None else replace_tensor(batch.labels),
            label_mask=(
                None
                if batch.label_mask is None
                else replace_tensor(batch.label_mask)
            ),
            scenario_id=replace_tensor(batch.scenario_id),
            group_id=batch.group_id,
            prediction_keys=batch.prediction_keys,
            _packed_buffers=tuple(buffers),
        )

    for name, fn in [("torch", torch_copy), ("numpy", numpy_copy)]:
        for _ in range(2):
            fn()
        t0 = time.perf_counter()
        for _ in range(20):
            fn()
        print(name, f"{(time.perf_counter()-t0)/20*1000:.2f}ms")


if __name__ == "__main__":
    main()
