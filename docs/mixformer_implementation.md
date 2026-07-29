# MixFormer and MDL-MixFormer implementation

## Scope

The implementation is based on `paper/mixformer/main.tex`. The standalone
`mixformer` path follows the published architecture. `mdl_mixformer` is an
explicitly experimental composition of the published MixFormer backbone and
the repository's MDL scenario/task-token semantics.

The production overlays reuse the current 147-field adapter contract, the nine
main UPS behavior streams, request-level feature deduplication, three task
labels, and coarse/fine scenario variants.

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
| UI-MixFormer one-way HeadMixing mask | `model.mixformer_user_head_count` |

The cross attention uses the algebraically equivalent single-query reordering
shown in the manuscript's commented efficiency derivation:

```text
softmax(q (H W_k)^T / sqrt(D)) H W_v
= softmax((q W_k) H^T / sqrt(D)) H W_v
```

This avoids materializing sequence-length-sized projected K/V tensors. When
the adapter supplies request-to-candidate row indices, candidate queries are
packed on a small target axis and attend to one request-major history tensor.
The long history is not copied per candidate.

## Current-data choices

- The 144 active non-sequential inputs have a packed embedding width of 3216.
  It divides exactly into the paper's `N=16` contiguous slices (201 values per
  head), so no padding or learned global pre-projection is used.
- The nine main behavior streams remain raw event streams and retain their
  configured truncation/order/null semantics. Their total configured capacity
  is 2048 events.
- The source has meaningful time deltas but no common absolute timestamp
  across all nine streams. Therefore the production configs use the paper's
  single sequence interface with the project's existing intent-ordered fusion
  and learned separators instead of inventing a cross-stream chronology.
- Raw action field widths differ by behavior family and do not naturally equal
  `N*D`. A bias-free per-family linear alignment maps the concatenated action
  embedding into `N*D` before the paper's per-layer sequence SwiGLU. This is
  the minimal data-shape adaptation; all MixFormer block equations remain
  unchanged.
- The manuscript reports `D=386` for MixFormer-small, but its HeadMixing
  definition requires `D/N` with `N=16`. The implementation uses the internally
  consistent `D=384`.
- The SwiGLU intermediate width is not disclosed. `H=1024` is used. With
  `N=16`, `D=384`, `L=4`, the current three-task production model has about
  **278.79M dense parameters**, close to the paper's reported 282M
  MixFormer-small budget.
- The coarse `mdl_mixformer` composition has **504.73M dense parameters** with
  the current scenario/task domain modules. Sparse embedding tables are not
  included in either count.
- UI-MixFormer's mask is implemented but disabled in the supplied configs.
  The current feature contract does not yet declare a verified user/item
  boundary in the ordered 3216-wide pack. Enabling it prematurely would make
  the mask syntactically valid but semantically wrong.

## MDL-MixFormer innovation

Each `MDLMixFormerBlock` performs:

1. the published MixFormer Query Mixer;
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

- `configs/mixformer.yaml`: coarse search/recommendation production profile.
- `configs/mdl_mixformer.yaml`: coarse-scene, three-task MDL profile.
- `configs/mixformer_fine.yaml`: fine-scene discovery sibling.
- `configs/mdl_mixformer_fine.yaml`: fine-scene MDL sibling.

Validate or run them through the existing CLI:

```bash
python -m src.main validate-config --config configs/mixformer.yaml
python -m src.main validate-config --config configs/mdl_mixformer.yaml
python -m src.main train --config configs/mixformer.yaml --max-steps 100
python -m src.main benchmark --config configs/mixformer.yaml \
  --mode compute --batch-size 8 --sequence-length 128
```

The supplied batch sizes are conservative starting points for 2xH100. They
must be tuned on the deployment driver/runtime because this environment cannot
execute a representative H100 benchmark.
