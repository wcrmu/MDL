# Resumable checkpoints on HDFS

## Scope

A training job that dies after consuming several data hours should restart at
the latest durable data boundary, not at step 0. This document describes the
resumable checkpoint written under
`training.checkpoint.dir`, what one checkpoint contains, how a restart decides
where to re-enter both the optimizer schedule and the input stream, and how to
operate the whole thing from the command line.

This is separate from `training.save_checkpoint` / `training.checkpoint_path`,
which writes a single final artifact for serving and evaluation. That path has
no step number, no optimizer state, and no data position, so it cannot resume a
run. Both can be enabled at the same time.

## Configuration

```yaml
training:
  checkpoint:
    dir: hdfs://temu-data-ns/apps/nothive/warehouse/searchrec/searchrec_dracarys_cvr_comm_us_8k/intern_train/aiden.fan
    every_steps: 0
    data_window_hours: 8
    sparse_delta: true
    sparse_full_every: 8
    shard_chunk_bytes: 536870912
    keep_last: 3
    resume: auto
```

| Field | Default | Meaning |
| --- | --- | --- |
| `dir` | `null` | Run root. A local path, `hdfs://…`, or `viewfs://…`. Empty disables checkpointing. |
| `run_name` | model name | Subdirectory of `dir`. Several models can share one root without colliding. |
| `every_steps` | `0` | Legacy step cadence. It must remain `0` when `data_window_hours` is enabled. |
| `data_window_hours` | `0` | Group contiguous `pt=YYYY-MM-DD/hr=HH` inputs into this many data hours and commit at each boundary. Production uses `8`. |
| `sparse_delta` | `false` | Save rank-owned sparse rows touched since the previous durable boundary; requires data-window mode. |
| `sparse_full_every` | `8` | Write a fresh full sparse base every N generations to bound restore-chain length. |
| `lineage_id` | run name | Stable history identity used by automatic time-range selection. |
| `keep_last` | `3` | Committed heads to retain. Sparse ancestors needed by retained deltas are protected automatically. |
| `save_on_exit` | `true` | Save once more when the loop ends normally, if that step was not just saved. |
| `resume` | `auto` | With data windows, select the newest committed checkpoint compatible with the requested input range; `none` starts fresh, or name one step explicitly. |
| `async_upload` | `true` | Pipeline local segment creation with remote upload. A data-window boundary still waits for the global commit. |
| `staging_dir` | system temp | Local scratch for staged files. Production configs set this explicitly; see [Staging space](#staging-space). |
| `shard_chunk_bytes` | `2GiB` | Maximum sparse stream segment size. Production uses `512MiB`; even one very large table is split. |
| `preflight_staging` | `true` | Refuse to start when staging cannot hold the local ranks' checkpoints. |
| `ready_timeout_sec` | `1800` | How long rank 0 waits for every peer's files before giving up on the commit. |
| `upload_stall_timeout_sec` | `300` | Fall back from streaming only after this many seconds with no completed remote write chunk; byte progress resets the deadline. |
| `data_resume` | `true` | Skip input files the previous run already consumed. Needs `reader.shard_unit: file` or `row_group`. |
| `data_resume_rewind` | `1` | Extra work items to re-read on top of the measured in-flight window. |

The directory does not need to exist; the first save creates
`dir/run_name/`. `configs/mdl_rankmixer.yaml`, `configs/mdl_onetrans.yaml`,
and the standalone `configs/mdl_mixformer.yaml` already point at the HDFS root
above, so those three runs land in `…/aiden.fan/mdl_rankmixer/`,
`…/mdl_onetrans/`, and `…/mdl_mixformer/`.

Everything is also overridable per launch, which is the easiest way to test the
plumbing against a local directory before trusting HDFS:

```bash
python -m src.main train --config configs/mdl_rankmixer.yaml \
  --checkpoint-dir /tmp/ckpt-smoke --checkpoint-every-data-hours 8 \
  --checkpoint-keep-last 2 --max-steps 45
```

To exercise the legacy step cadence against a production config, disable its
data cadence explicitly: `--checkpoint-every-data-hours 0
--no-checkpoint-sparse-delta --checkpoint-every-steps 20`.

## Checking the directory before a long run

A checkpoint that cannot be written is only discovered thousands of steps in, so
verify the run directory first. This writes a probe file, reads it back and
compares bytes, lists it, deletes it, and reports what the next launch would
resume from:

```bash
python -m src.main check-checkpoint-store --config configs/mdl_rankmixer.yaml
```

```text
Checkpoint store | run_dir=hdfs://…/aiden.fan/mdl_rankmixer remote=True
Checkpoint store | probe=hdfs://…/mdl_rankmixer/_probe-<host>-<pid>
  create_dir             ok    0.31s
  write_json             ok    0.08s
  read_json              ok    0.05s content verified
  upload 8MiB            ok    0.42s
  download 8MiB          ok    0.21s bytes verified
  list_entries           ok    0.04s 2 entries
  remove_tree            ok    0.12s
Committed checkpoints | count=3 latest=step-000038000
Next launch | resume=auto -> step 38000
Checkpoint store: OK (writable, readable, prunable)
```

It exits non-zero with a one-line reason on failure, so it works as a gate in a
launch script. The delete is part of the check on purpose: retention prunes
superseded steps, and a directory that cannot be pruned fills up until the job
dies. Run it without a config to check a bare path, and raise `--probe-mib` to
get a rough throughput reading for sizing sparse segments and timeouts:

```bash
python -m src.main check-checkpoint-store \
  --checkpoint-dir hdfs://temu-data-ns/apps/…/aiden.fan --probe-mib 256
```

## Staging space

A remote run directory cannot be written by `torch.save`, so every rank first
writes its files locally and then uploads them. That local scratch is the part
that is easy to get wrong: at `world_size: 4` one rank's embedding shard for the
production configs is roughly **30GiB** (BF16 weights plus a Row-Wise Adagrad
accumulator per row), and all four ranks on a node stage into the same
filesystem.

Each local rank owns a separate `step-…/rank-xxxxx/` staging subdirectory. A
fast rank can therefore clean its files without deleting a slower peer's queued
fallback segments.

Two properties of the trainjob image make the default unusable:

- `/tmp` is a **tmpfs fixed at 48GiB**. It does not grow with the memory the
  launch requests, so no amount of extra memory makes it fit.
- `tempfile.gettempdir()` follows `TMPDIR`, and host-prepare repoints `TMPDIR`
  at `/dev/shm` when its shared-memory IPC starts. Whether staging landed on
  `/tmp` or in RAM therefore depended on initialization order.

So production configs name the directory explicitly:

```yaml
training:
  checkpoint:
    staging_dir: /dev/shm/mdl-checkpoint-staging
```

`/dev/shm` is sized by the launch request rather than fixed, which is why it is
the choice here; it is still RAM, so it counts against the job's memory.

Three things keep the footprint down and make a shortfall legible:

1. **Bounded raw sparse segments.** Each rank scans only its local dirty-bit
   masks and pulls selected weight/Adagrad rows from GPU to CPU in bounded
   blocks. One table is split at `shard_chunk_bytes`; neither pickle nor a
   complete CPU copy of the rank shard is needed.
2. **Publish while staging.** Each file is handed to the uploader as it lands
   and deleted once it reaches the run directory, so staging holds a chunk or two
   instead of the whole step. Backpressure is measured from completed remote
   write chunks, not merely from queue occupancy, so a slow multi-GiB upload does
   not look stuck while bytes are still moving. A true no-progress interval opens
   one circuit for the rest of that step: later files stay in staging without
   each waiting through the timeout again. The ordered step finalizer is never
   dropped, because it retries those files and writes the rank-ready marker.
3. **A preflight and a named error.** Startup compares free space against the
   estimated need times the local rank count and refuses to run when it does not
   fit. A write that still fails raises `CheckpointStagingSpaceError`, which
   names the directory and the shortfall, rather than
   `PytorchStreamWriter failed writing file data/303`.

Staging failures abort the job on purpose. `save_model_checkpoint` synchronizes
ranks with a barrier, so a rank that logged the error and carried on would leave
its peers waiting there forever; a clear crash is better than a hang.

```text
Checkpointing | ready run_dir=hdfs://…/mdl_rankmixer staging_dir=/dev/shm/mdl-checkpoint-staging staging_default=False …
Checkpointing | staging staging_dir=/dev/shm/mdl-checkpoint-staging free=598.4GiB needed=… peak=… chunks=…
```

## Layout on disk

```text
hdfs://…/aiden.fan/mdl_rankmixer/
  _latest.json                      newest committed step (a hint, not the source of truth)
  step-000038000/
    checkpoint.json                 lineage, generation, data window/watermark, fingerprints
    _COMMIT                         written last; only committed steps are resumable
    train_state.pt                  dense + replicated-sparse optimizer state (rank 0)
    model/
      manifest.json                 sparse kind (full/delta), parent and rank manifests
      dense.pt                      all non-sharded weights (rank 0)
      sparse-manifest-rank-00000.json
      sparse-t0000-s00000-rank-00000-of-00004.bin
      sparse-t0001-s00000-rank-00000-of-00004.bin
      …
    progress-rank-00000.json        rows, window and optional in-window cursor
    …
    _READY-rank-00000               per-rank "my files all landed" marker
    …
```

Step directories are zero-padded to nine digits so a lexicographic listing is
also a numeric ordering.

## Why `_COMMIT` matters

A checkpoint spans many files across many ranks, and HDFS gives no
cross-file atomicity. A job killed halfway through an upload therefore leaves a
step directory that looks complete but is not. The protocol is:

1. Every rank writes its own files into a local staging directory, then uploads
   them into `step-<n>/` and writes `_READY-rank-<r>`.
2. Rank 0 waits for all `world_size` ready markers, then writes `_COMMIT` and
   `_latest.json`.
3. All ranks synchronize on rank 0's result. Dirty bits are cleared and the
   next data window opens only after that global success.
4. Discovery ignores any step directory without `_COMMIT`.

So a torn checkpoint is invisible to a resume rather than being loaded as
truth. Retention reclaims those partials: `prune_run_directory` deletes
uncommitted directories older than the newest commit, and keeps uncommitted
directories newer than it, because those may be an upload still in flight.

`_latest.json` exists to make `hdfs dfs -cat` cheap for humans and dashboards.
Resume never trusts it; it lists step directories and picks the newest one with
a commit marker, so a stale or missing pointer cannot mislead a restart.

## What a resume restores

- **Weights.** Replicated modules come from `model/dense.pt`. Sparse restore
  starts from the nearest full base and replays dirty-row deltas in generation
  order. At the same world size each rank downloads only its own manifests and
  segments; a world-size change remaps saved global IDs to the new owners.
  Legacy chunked/full checkpoints still load.
- **Optimizer state.** The dense and replicated-sparse `state_dict`s come from
  `train_state.pt`. Sharded embedding optimizers (`ShardedAdagrad`,
  `ShardedRowWiseAdagrad`) keep their accumulator next to the rows it belongs
  to, in the per-rank shard file, and are restored row-wise. Dropping this is
  what makes a naive "reload the weights" resume regress: Adagrad restarts with
  a zero accumulator and takes effectively enormous steps on frequent ids.
- **Step and row counts.** `steps` resumes at the checkpointed value, so the
  learning-rate schedule, warmup, and step-based cadences (`log_every_steps`,
  `fixed_test_eval.every_steps`) continue rather than replaying warmup.
- **Input position.** See below.

A resume refuses to load a checkpoint whose model name, task names, vocabulary
strategy fingerprint, or sparse optimizer disagrees with the current config, and
drops the data cursor when the world size changed, since a cursor is defined
relative to one rank's slice of the shard.

## Eight-hour data windows and incremental selection

`data_window_hours: 8` is based on data time, not wall-clock runtime and not the
number of optimizer steps. Every configured train input must contain an hourly
partition `pt=YYYY-MM-DD/hr=HH`. The trainer sorts and validates those hours,
rejects gaps, and opens only one window at a time. The final window may contain
fewer than eight hours and is still committed when exhausted.

At a boundary the reader and prefetch workers are closed first, all ranks finish
the same optimizer window, and checkpointing runs synchronously with training.
Rank 0 saves dense model and dense/replicated optimizer state with `torch.save`.
Each rank independently streams its own sparse rows and row-wise Adagrad state;
there is no cross-rank ID `all_gather`. Remote transfer overlaps segment
creation, but training cannot enter the next data window until `_COMMIT` exists.

For `resume: auto`, the requested train inputs define the permitted time range.
Discovery scans committed manifests for the same lineage and picks the greatest
`(trained_through, generation, step)` compatible with that range:

- a completed boundary resumes at the next contiguous window;
- an incomplete graceful-exit checkpoint resumes its matching window with the
  recorded cursor and rewind;
- if the whole requested range is already complete, no data is replayed;
- a time gap, an in-window watermark, a changed input-manifest digest, a missing
  sparse base, or a cross-lineage dependency fails before training starts.

Use `resume: none` (and normally a new lineage/run name) when a genuinely fresh
model is intended. This explicit opt-out prevents a discontinuity from silently
becoming a random initialization.

Sparse checkpoints form `full -> delta -> …`. A delta contains each locally
owned row at most once, even if the ID was updated many times in the window.
After `sparse_full_every` generations a new full base breaks the old dependency
chain; retention then collects ancestors no retained head needs.

## Knowing where the data stopped

Each rank turns its discovered inputs into one deterministic, ordered work
list — whole files under `reader.shard_unit: file`, row groups under
`row_group`. A position in that list is the entire data cursor: work unit,
index, and a rolling digest of the already-consumed prefix.

The reader usually lives in a separate `host_prepare` process, so it publishes
its position into a small shared-memory channel (`ScanCursorChannel`) that the
trainer reads when it takes a checkpoint. The channel is keyed by a split
identity (`scan_split_key`), which is derived from the input list plus rank and
world size, so a periodic held-out evaluation scan in the same process cannot
overwrite the training cursor.

On restart the recorded position becomes a `ScanResumePlan`, and the scanner
skips the consumed prefix:

- **Legacy step mode permits append-only growth.** The prefix digest covers
  only the consumed part of the list, so "the same inputs plus new hours" can
  resume. Data-window mode instead requires a fixed manifest per window and
  rejects a change before loading.
- **Rewritten legacy inputs fall back safely.** If the digest at the recorded
  position does not match, the old index means nothing, and the scanner logs a
  warning and rescans instead of resuming into an unrelated file.

### The reader is not where the trainer is

The cursor names the file the *reader* opened, and the reader runs ahead of the
trainer by everything sitting in the prefetch queues, the host-prepare IPC
queue, and the device prefetch. Resuming at the recorded position would
therefore skip files that were read but never trained on — silent data loss,
and the failure this design exists to prevent.

So the reader also publishes how many rows it has handed downstream, counted the
same way the training loop counts rows. The difference between those two
counters is the reader's lead in rows, and dividing it by the *smallest* work
item observed converts it into a number of work items to replay. The scanner
adds what the row counters cannot see — the shuffle buffer plus one batch — and
`data_resume_rewind` adds a final constant margin. The result is stored with the
cursor as `rewind`:

```text
Checkpoint | step=38000 data_position=file[431] rewind=6 reader_rows=1104384 trained_rows=1094208 …
Checkpoint resume | step=38000 rows=… data_position=file[425] (reader_stopped_at=431 rewind=6) …
```

Using the smallest item as the divisor and rounding up means the overlap is
never too small, so resume is at-least-once: a handful of files at the boundary
are trained on twice, and none is dropped. When the reader has already been
drained the lead is zero and only the constant margin applies.

`progress-rank-<r>.json` records the same cursor in plain JSON, which is the
fastest way to answer "where did rank 3 stop?" without loading any tensors:

```bash
hdfs dfs -cat hdfs://…/mdl_rankmixer/step-000038000/progress-rank-00003.json
```

Under `reader.agg_direct_mode: compare` two readers walk the same rows, so
neither may own the cursor. Checkpoints still resume the step and the weights
there; only the input scan restarts from the beginning of the shard.

## Operating notes

- **Restarting is the normal case.** With `resume: auto` the same command
  resumes from the latest compatible data watermark and starts fresh only when
  the lineage has no checkpoint. Platform retry needs no step calculation.
- **Cost of a save.** Sparse segment construction and upload are pipelined, but
  each eight-hour boundary is deliberately blocking. Timeout or any rank's
  failure fails the run; success is never returned without a global `_COMMIT`.
- **Crash exposure.** A hard crash replays at most the uncommitted current data
  window. A normal early stop writes an incomplete recovery checkpoint with an
  at-least-once cursor rewind.
- **Rolling back.** `--checkpoint-resume 36000` can read an older committed
  step. A retraining branch must also use a new `--checkpoint-run-name` (or
  directory); overwriting an existing committed step is refused.

## Tests

`tests/test_checkpoint_resume.py` covers the store/commit protocol and legacy
round trips. `tests/test_checkpoint_delta.py` covers dirty-row tracking,
eight-hour planning, full-plus-delta exact restore, dependency-aware retention,
time/digest selection, and missing-base rejection.
