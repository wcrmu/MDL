# Agg Parquet direct pipeline: implementation and benchmark report

Date: 2026-07-23

## Scope and data semantics

The benchmark dataset is the two-file ZSTD Parquet fixture under
`artifacts/mock_parquet_full_2x2500_zstd/` (2,500 raw rows per file). It was
generated from `sample_row_mock_json`.

All physical slots in the anonymized row are data. In particular, zero-valued
mock fields are privacy substitutions, not padding sentinels. The generated
fixture preserves 7 candidates and 161 `impr` events per raw row and replaces
categorical zero placeholders with deterministic non-zero int64 values.

## Implemented production path

For `adapter_parquet` with
`reader.deduplicate_request_features=true` and
`reader.agg_direct_mode=direct`, the path is now:

```text
Parquet/Arrow scanner batch
  -> axis-separated adapter output
  -> RequestGroupBlock descriptors
  -> request shuffle + sequence-length bucket + candidate pack
  -> PreparedAxisBatch
  -> direct Torch tensorization
  -> FeatureBatch
```

The following release blockers were addressed:

- Oversized request groups share the source payload across all slices. Only the
  final slice releases the original source reference.
- Bucket lengths are computed before null-anchor compaction. Pack-time
  `SequenceSelectionPlan` then applies truncate-before-null-anchor compaction
  once and shares the selection across every aligned field.
- The production `adapter_parquet` direct path no longer rebuilds a
  pack-boundary candidate/request Arrow table.
- Identity and pre-hashed int64 encoding are vectorized with NumPy instead of
  per-element Python conversion.
- `adapter_workers` now applies to the axis-separated path through an ordered,
  bounded process pool.
- `agg_direct_mode=compare` runs independent legacy and direct iterators and
  compares every `FeatureBatch` field, tensor dtype, shape, and value. It
  reports the first differing path/index and yields the legacy batch.
- `deduplicate_request_features=false` explicitly uses the legacy path.

`flat_parquet` keeps the narrow-Arrow transitional implementation because it
does not use the agg adapter's request/candidate/sequence axes.

## Correctness gates

- Complete test suite: **379 passed, 3 skipped**.
- Direct module tests: **27 passed**.
- Full mock first-batch oracle: legacy and direct matched exactly for 511
  candidates and all 178 feature/sequence entries, labels, masks, scenario IDs,
  group IDs, and prediction keys.
- The null-anchor fixture simultaneously verifies pre-compaction bucket length
  3 and final sequence length 2.
- Oversized request integration tests verify 5 candidates split as 2/2/1
  without an early `SourceRegistry` release.
- `compileall`, focused lint, and `git diff --check` pass.

## Data-only results

The paired comparison used the same fixture, batch size 512, 4 Parquet workers,
4 adapter workers, 4 host-prefetch batches, no device prefetch, and two runs per
mode.

| Metric | Legacy | Direct | Change |
|---|---:|---:|---:|
| Throughput | 361.28 samples/s | 475.85 samples/s | **+31.71%** |
| Mean data wait | 1.4174 s | 1.0362 s | **-26.90%** |
| Peak host RSS | 1.603 GB | 2.588 GB | +61.47% |

With the tuned reader (8 Parquet workers, 4 adapter workers, 8 host-prefetch
batches, 1 device-prefetch batch), the one-pair data-only result was:

| Metric | Legacy | Direct | Change |
|---|---:|---:|---:|
| Throughput | 376.39 samples/s | 505.47 samples/s | **+34.29%** |
| Mean data wait | 1.3555 s | 0.8903 s | **-34.32%** |
| Peak host RSS | 1.760 GB | 3.095 GB | +75.80% |

The additional host memory is the principal trade-off: axis-separated Python
payloads stay live across the shuffle/prefetch runway. Reducing host prefetch
from 8 to 4 is the appropriate first tuning step on memory-constrained hosts.

Cold data-only time to the first batch was 6.851 s for legacy and 7.076 s for
direct (+0.225 s, +3.29%). The direct path improves steady state, not process
pool cold start.

## GPU results and validity boundary

All eight GPUs were occupied by unrelated containers during this run. The only
repeatable experiment available was a sequential legacy/direct pair on GPU 7;
therefore its absolute utilization and step throughput are contaminated and
must not be treated as an isolated A/B result.

Across two steady-state runs per mode:

| Metric | Legacy | Direct | Observed change |
|---|---:|---:|---:|
| Dataloader wait/step | 0.13081 s | 0.01132 s | **-91.35%** |
| Dataloader wait ratio | 9.181% | 0.712% | **-8.469 pp** |
| Sampled GPU utilization | 58.318% | 62.652% | +4.334 pp / +7.43% relative |
| Allocated HBM | 10.763 GB | 10.763 GB | unchanged |
| Reserved HBM | 15.487 GB | 15.487 GB | unchanged |

The wait reduction is attributable to the reader because it is measured inside
the process. The utilization delta is only an observation: foreign GPU kernels
also contribute to the sampler.

The earlier uncontended legacy baseline was 7.544% GPU utilization at a 7.955%
dataloader wait ratio. Because the direct and legacy `FeatureBatch` values and
shapes match exactly, their model kernel graph is unchanged. Normalizing only
the removable wait fraction gives:

```text
7.544% * (1 - 0.712%) / (1 - 7.955%) = 8.138%
```

This is an **estimate**, not a clean direct measurement: about **+0.594
percentage points** or **+7.87% relative**. A causal utilization number still
requires an exclusive GPU rerun.

The direct data path itself does not increase HBM: exact batch parity implies
the same device tensors. An exploratory direct batch-size-768 run allocated
14.706 GB versus 10.763 GB at batch size 512 (+36.64%) and reached 381.42
samples/s in the contaminated environment. This is optional capacity tuning
and requires learning-rate/global-batch validation.

## First-step timing

The benchmark's timed first training step (warmup 0, one measured step) was:

| Metric | Legacy | Direct |
|---|---:|---:|
| Timed training step | 11.357 s | 19.948 s |
| Data wait inside step | 6.259 s | 8.539 s |
| Forward | 3.209 s | 10.903 s |

This pair is not suitable for comparing implementations: foreign work and CUDA
lazy initialization made the direct forward 7.7 seconds slower even though the
model inputs are identical. The clean CPU data-only first-batch values above
are the reliable cold-reader measurement. End-to-end startup-to-first-step
wall time should be rerun on an exclusive GPU.

## Remaining performance limit

The pack/tensor boundary is now direct, but the agg adapter still converts
Arrow rows and nested values to Python objects. A no-worker two-batch profile
improved from 6.986 s (legacy) to 5.218 s (direct, -25.3%); direct tensorization
itself fell to 0.674 s, leaving adapter normalization at 3.329 s as the largest
reader-side CPU target. Further gains require an Arrow-native or compiled
adapter, not another FeatureBatch rewrite.

Raw reports are under `artifacts/agg_direct_bench/`. The principal files are:

- `legacy_data_direct_rework_r1.json`, `r2.json`
- `direct_data_noarrow_mp_r1.json`, `r2.json`
- `legacy_data_tuned_reader_r1.json`
- `direct_data_tuned_reader_r1.json`
- `legacy_e2e_tuned_gpu7_contended_r1.json`, `r2.json`
- `direct_e2e_tuned_gpu7_contended_r1.json`, `r2.json`
- `direct_b768_e2e_gpu7_contended.json`
