"""Sweep adapter_workers for data-only mean throughput."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def main() -> None:
    base = Path("artifacts/bench_b512_direct.yaml").read_text()
    out_dir = Path("artifacts/agg_direct_bench/aw_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    for aw in (1, 2, 3):
        text = base.replace("adapter_workers: 2", f"adapter_workers: {aw}")
        # Only replace the train one - both train and maybe only one has comment.
        # Safer: replace the last occurrence near agg_direct_mode.
        cfg = Path(f"/tmp/bench_aw{aw}.yaml")
        cfg.write_text(text)
        rates = []
        for i in range(1, 5):
            out = out_dir / f"aw{aw}_r{i}.json"
            subprocess.run(
                [
                    "/root/anaconda3/bin/python",
                    "-m",
                    "src.main",
                    "benchmark",
                    "--config",
                    str(cfg),
                    "--mode",
                    "data",
                    "--warmup-steps",
                    "4",
                    "--steps",
                    "16",
                    "--profile-steps",
                    "0",
                    "--output",
                    str(out),
                ],
                check=True,
                env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "CUDA_VISIBLE_DEVICES": "0"},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            rate = json.loads(out.read_text())["samples_per_second"]
            rates.append(rate)
            print(f"aw={aw} r{i} {rate:.1f}", flush=True)
        print(
            f"aw={aw} mean={sum(rates)/len(rates):.1f} best={max(rates):.1f} worst={min(rates):.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
