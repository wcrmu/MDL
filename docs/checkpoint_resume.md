# Resumable checkpoints on HDFS

## Scope

A training job that dies at step 37,000 should not start over at step 0. This
document describes the periodic checkpoint written under
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
    every_steps: 2000
    keep_last: 3
    resume: auto
```

| Field | Default | Meaning |
| --- | --- | --- |
| `dir` | `null` | Run root. A local path, `hdfs://…`, or `viewfs://…`. Empty disables checkpointing. |
| `run_name` | model name | Subdirectory of `dir`. Several models can share one root without colliding. |
| `every_steps` | `0` | Steps between saves. `0` disables periodic saves. |
| `keep_last` | `3` | Committed steps to retain. Older steps are deleted after each new commit. |
| `save_on_exit` | `true` | Save once more when the loop ends normally, if that step was not just saved. |
| `resume` | `auto` | `auto` / `latest` take the newest committed step, `none` starts fresh, or name one step (`38000` or `step-000038000`). |
| `async_upload` | `true` | Stage locally, then upload on a background thread so the step loop is not blocked on HDFS. |
| `staging_dir` | system temp | Local scratch for staged files. Production configs set this explicitly; see [Staging space](#staging-space). |
| `shard_chunk_bytes` | `2GiB` | Tables are packed into staged files of at most this size. Smaller lowers the host-memory and staging peak. |
| `preflight_staging` | `true` | Refuse to start when staging cannot hold the local ranks' checkpoints. |
| `ready_timeout_sec` | `1800` | How long rank 0 waits for every peer's files before giving up on the commit. |
| `data_resume` | `true` | Skip input files the previous run already consumed. Needs `reader.shard_unit: file` or `row_group`. |
| `data_resume_rewind` | `1` | Extra work items to re-read on top of the measured in-flight window. |

The directory does not need to exist; the first save creates
`dir/run_name/`. `configs/mdl_rankmixer.yaml` and `configs/mdl_onetrans.yaml`
(inherited by `configs/mdl_mixformer.yaml`) already point at the HDFS root
above, so those three runs land in `…/aiden.fan/mdl_rankmixer/`,
`…/mdl_onetrans/`, and `…/mdl_mixformer/`.

Everything is also overridable per launch, which is the easiest way to test the
plumbing against a local directory before trusting HDFS:

```bash
python -m src.main train --config configs/mdl_rankmixer.yaml \
  --checkpoint-dir /tmp/ckpt-smoke --checkpoint-every-steps 20 \
  --checkpoint-keep-last 2 --max-steps 45
```

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
get a rough throughput reading for sizing `every_steps`:

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

1. **Chunked writes.** Tables are packed into files of at most
   `shard_chunk_bytes`, and each group's CPU copy is dropped once written, so
   neither host memory nor staging ever holds a whole shard. A table larger than
   the budget still gets its own file.
2. **Publish while staging.** Each file is handed to the uploader as it lands
   and deleted once it reaches the run directory, so staging holds a chunk or two
   instead of the whole step. If the run directory stops draining, the sink
   declines and files simply wait, which degrades to the old behaviour instead of
   blocking the step loop.
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
Checkpointing | staging staging_dir=/dev/shm/mdl-checkpoint-staging free=598.4GiB needed=27.1GiB (local_ranks=4 total=30.2GiB peak=6.2GiB chunks=16)
```

## Layout on disk

```text
hdfs://…/aiden.fan/mdl_rankmixer/
  _latest.json                      newest committed step (a hint, not the source of truth)
  step-000038000/
    checkpoint.json                 manifest: step, world size, model layout, config fingerprints
    _COMMIT                         written last; only committed steps are resumable
    train_state.pt                  dense + replicated-sparse optimizer state (rank 0)
    model/
      manifest.json                 embedding shard plan and chunk-to-table map
      dense.pt                      all non-sharded weights (rank 0)
      shard-0000-rank-00000-of-00008.pt   some of this rank's tables, with their accumulators
      shard-0001-rank-00000-of-00008.pt
      …
    progress-rank-00000.json        step, rows, and the input scan cursor for that rank
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
3. Discovery ignores any step directory without `_COMMIT`.

So a torn checkpoint is invisible to a resume rather than being loaded as
truth. Retention reclaims those partials: `prune_run_directory` deletes
uncommitted directories older than the newest commit, and keeps uncommitted
directories newer than it, because those may be an upload still in flight.

`_latest.json` exists to make `hdfs dfs -cat` cheap for humans and dashboards.
Resume never trusts it; it lists step directories and picks the newest one with
a commit marker, so a stale or missing pointer cannot mislead a restart.

## What a resume restores

- **Weights.** Replicated modules come from `model/dense.pt`; each rank reads
  its own embedding rows from its `model/shard-<n>-rank-<r>-of-<w>.pt` files, so
  an eight-rank restart never materializes the full table on one host. The older
  one-file-per-rank layout (`rank-<r>-of-<w>.pt`) still loads, so a resume can
  cross the deploy that introduced chunking.
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

- **Growing inputs are fine.** The prefix digest covers only the consumed part
  of the list, so "the same inputs plus new hours" still resumes.
- **Rewritten inputs fall back safely.** If the digest at the recorded position
  does not match, the files behind that prefix changed, the old index means
  nothing, and the scanner logs a warning and rescans from the beginning
  instead of resuming into an unrelated file.

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
  resumes if a committed checkpoint exists and starts fresh otherwise, so a
  platform job that retries on failure needs no change.
- **Cost of a save.** With `async_upload: true` the step loop only pays the
  local write; the upload and the commit happen on a background thread, and an
  upload still in flight is drained at shutdown. Upload failures are retried
  and then logged — a failed checkpoint never fails the run.
- **Pick `every_steps` from the save cost, not the crash rate.** At
  2000 steps the recomputation exposure is bounded by 2000 steps of work,
  while the amortized upload cost stays under a percent of step time for the
  reference 8k-token configs.
- **Rolling back.** `--checkpoint-resume 36000` re-enters at an older committed step; the
  newer directories are left alone until retention removes them.

## Tests

`tests/test_checkpoint_resume.py` covers the store abstraction, the commit and
retention protocol, the optimizer and cursor round trip (dense and sharded),
scan resume behavior including digest mismatch and cross-split isolation, the
rewind arithmetic that converts the reader's row lead into work items, and the
training-loop coordinator's cadence, save, and restore path.
