# MixFormer and MDL-MixFormer implementation

## Scope

The implementation is based on MixFormer's published architecture. The
standalone `mixformer` path follows that architecture, including UI-MixFormer
when `model.mixformer_user_item_decouple` is true. `mdl_mixformer` is an
explicitly experimental composition of the MixFormer backbone and the
repository's MDL scenario/task-token semantics.

The production MixFormer YAMLs are standalone files. They reuse the current
adapter contract, eight UPS behavior streams, request-level feature
deduplication, three task labels, and coarse/fine scenario variants.

## Paper-to-code mapping

| Paper component | Implementation |
| --- | --- |
| Concatenate `e_ns`, split it into `N` contiguous slices, project each slice independently | `MixFormerFeatureHeadProjector` |
| `P = HeadMixing(RMSNorm(X)) + X` | `MixFormerQueryMixer` |
| Per-head `q_i = SwiGLUFFN_i(RMSNorm(p_i)) + p_i` | `StackedPerHeadSwiGLUFFN` |
| Per-layer `h_t = SwiGLUFFN^(l)(RMSNorm(s_t)) + s_t` | `MixFormerCrossAttention.sequence_ffn` (one instance per block) |
| One full-dimensional sequence query per feature head | `MixFormerCrossAttention` |
| `z_i = Attention(q_i, h^i) + q_i` | `MixFormerCrossAttention.forward` |
| Per-head output fusion | `MixFormerOutputFusion` |
| `L` stacked blocks and task-specific networks | `MixFormerModel` |
| UI-MixFormer user/item split, one-way HeadMixing mask, user-side request reuse | `model.mixformer_user_item_decouple` |

The cross attention uses the algebraically equivalent single-query reordering
shown in the manuscript's commented efficiency derivation:

```text
softmax(q (H W_k)^T / sqrt(D)) H W_v
= softmax((q W_k) H^T / sqrt(D)) H W_v
```

This avoids materializing sequence-length-sized projected K/V tensors.
Scores use the printed MixFormer form `q^T k / sqrt(D)` with no extra
query RMSNorm; Query Mixer and the sequence SwiGLU already own the paper's
pre-RMSNorm residuals.

## Request-level reuse vs UI-MixFormer

Two different request-level optimizations exist, and they are not the same
thing:

1. **History / sequence RLB** (already present for vanilla MixFormer). The
   adapter supplies candidate-to-request `row_indices`. Candidate queries are
   packed onto a small target axis and attend one request-major history tensor.
   The long sequence is transformed once per unique request and is not copied
   per candidate.
2. **UI-MixFormer user-side RLB** (paper UI-MixFormer). Non-sequential features
   are split into user-side `N_U` heads and item-side `N_G` heads. HeadMixing
   is masked so user outputs cannot contain item chunks. Because user heads do
   not depend on candidate features, the user Query Mixer and user-head
   attention can run once per request and be reused across candidates.

Vanilla MixFormer with history RLB still ran Query Mixer and all `N` attention
heads on the candidate axis: user and item chunks were mixed, so user queries
were candidate-specific. Enabling UI-MixFormer is what makes the user-side
modules request-major.

On `MixFormerModel` both (1) and (2) are used together. On `MDLMixFormerModel`
the split, mask, and history RLB are used, but user Query Mixer / user-head
attention stay on the candidate axis: scenario/task tokens are candidate-sized
and the scenario query router can make otherwise-shared user queries
candidate-specific.

## UI-MixFormer layout

Paper formulas:

```text
N_G = floor(D_ns^G * N / D_ns)
N_U = N - N_G
```

The manuscript also says a practical 1:1 split is used. The resolver tries the
formula first, then 1:1 when `N` is even, then the packed split closest to the
formula.

The user/item axis comes from the train adapter: `context_features` are
request-axis (user), `item_features` are candidate-axis (item), plus coarse-scene
derived request columns. Each side is concatenated and even-split independently,
then projected to `D`.

On the current production pack:

- user width **624** (request-axis features)
- item width **1312** (candidate-axis features)
- `N=8`
- formula would pick `N_U=3`, `N_G=5`, but `1312 % 5 != 0`
- 1:1 packs: `N_U=N_G=4` (`624/4=156`, `1312/4=328`)

Configs leave `mixformer_user_head_count: null` so this auto-split is used.
Set it only to force a specific even split.

## Current-data choices

- The 144 active non-sequential inputs have a packed embedding width of 1936.
  It divides exactly into `N=8` contiguous slices (242 values per head), so no
  padding or learned global pre-projection is used.
- The eight main behavior streams remain raw event streams and retain their
  configured truncation/order/null semantics. Timestamp-aware fusion merges
  them into one global 8000-event window.
- All physical streams derive time from the same request clock. MixFormer
  profiles globally interleave valid actions by `time` and add a learned
  stream/type embedding to each action. This realizes the paper's single
  temporally ordered sequence without inventing absolute timestamps.
  Separator tokens are disabled because the paper defines only real actions in
  `S`, and action type is already represented explicitly.
- Raw action field widths differ by behavior family and do not naturally equal
  `N*D`. A bias-free per-family linear alignment maps the concatenated action
  embedding into `N*D` before the paper's per-layer sequence SwiGLU. This is
  the minimal data-shape adaptation; all MixFormer block equations remain
  unchanged.
- The manuscript reports `D=386` for MixFormer-small, but HeadMixing requires
  `D/N` with `N=16`. Production configs use a reduced `N=8`, `D=128`
  (`128/8=16`), not the paper-small width `N=16`, `D=384`.
- The SwiGLU intermediate width is not disclosed. Production uses `H=512`.
  Sequence tokens are aligned to `N·D=1024`. With `N=8`, `D=128`, `L=4`,
  and `task_head_hidden_dim=1024` left unchanged, the three-task MixFormer
  has about **24.80M dense parameters**. That is far below the paper's
  reported 282M MixFormer-small budget.
- The coarse `mdl_mixformer` composition has **46.41M dense parameters** with
  the current scenario/task domain modules. Domain MHA uses `num_heads=8`
  so `token_dim` divides evenly (`128/8=16`). Sparse embedding tables are not
  included in either count.

This is not a bit-identical Douyin UI-MixFormer: the industrial feature
contract, reduced `D=128`, and undisclosed SwiGLU `H=512` differ. The semantic
target is the paper's UI equations plus request-level user sharing.

## MDL-MixFormer innovation

Each `MDLMixFormerBlock` performs:

1. the published MixFormer Query Mixer (with the UI mask when decoupling is on);
2. active-scenario query routing;
3. the published sequence cross attention and Output Fusion;
4. MDL scenario/task domain interaction over the newly fused heads.

The scenario router pools the active scenario state (and the global state when
enabled), creates head-specific gates, and adds a shared scenario delta to each
query before sequence attention. Its delta projection is zero-initialized, so
the initial function is exactly the standalone MixFormer query path. Training
then learns which semantic heads should retrieve scenario-specific behavior.
Task tokens remain separate readers, preventing task labels from being mixed
into one shared sequence query.

This gives a layerwise feedback loop:

```text
scenario state -> query routing -> sequence retrieval -> fused feature heads
               -> scenario/task token update -> next-layer query routing
```

The model is guarded by `experimental_model_acknowledged: true`; it is not
presented as a published MDL or MixFormer result.

## Configurations

- `configs/mixformer.yaml`: standalone coarse search/recommendation production profile.
- `configs/mdl_mixformer.yaml`: standalone coarse-scene, three-task MDL profile.
- `configs/mixformer_fine.yaml`: standalone fine-scene sibling of `mixformer.yaml`.
- `configs/mdl_mixformer_fine.yaml`: standalone fine-scene MDL sibling.

All four are self-contained and do not `extends` OneTrans. All four enable
`mixformer_user_item_decouple: true`.

Validate or run them through the existing CLI:

```bash
python -m src.main validate-config --config configs/mixformer.yaml
python -m src.main validate-config --config configs/mdl_mixformer.yaml
python -m src.main train --config configs/mixformer.yaml --max-steps 100 \
  --train-start-hour 2026-07-22-22 --train-end-hour 2026-07-29-22
python -m src.main benchmark --config configs/mixformer.yaml \
  --mode compute --batch-size 8 --sequence-length 128
```

When test hours are omitted, training uses the full calendar day after
`--train-end-hour` (the example above evaluates 2026-07-30 00:00–24:00).
Explicit test hours remain half-open. Rank 0 freezes a deterministic manifest
sampled across that window (25 Parquet files per rank by default, 50 total on
the production two-rank launch), and the same held-out rows are evaluated every
5000 steps. Override the cost with
`--test-files-per-rank` and `--eval-every-steps`.

The supplied batch sizes are conservative starting points for 2xH100. They
must be tuned on the deployment driver/runtime because this environment cannot
execute a representative H100 benchmark.
