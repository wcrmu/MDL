# STCA sequence encoder

`rankmixer` and `mdl_rankmixer` can use `stca` as a per-sequence encoder while
leaving the existing `longer`, `mean_pool`, and `attention_pool` paths intact.
The implementation follows `paper/STCA/main.tex`.

## Configuration

Set the encoder on any feature/shared sequence and provide the scalar candidate
features used to construct the target token:

```yaml
sequences:
  - name: hist
    encoder: stca
    max_length: 10000
    sequence_order: newest_to_oldest
    truncation: head
    target_inputs:
      - goods_id_hn
      - cat1_id_hn
      - price_hn
    rankmixer_summary_tokens: 1
    stca_layers: 4
    stca_num_heads: 16
    stca_expansion_ratio: 4
    time_delta_field: time_delta_log1p_seconds
    fields:
      - name: item_id
        kind: categorical
        source: hist_item_id
        # encoding: ...
      - name: time_delta_log1p_seconds
        kind: dense
        source: hist_time_delta_log1p_seconds
        dimension: 1
```

The options are:

| Option | Default | Meaning |
| --- | ---: | --- |
| `stca_dim` | `256` | STCA working width, paper symbol `d`. Independent of RankMixer `token_dim`. |
| `stca_layers` | `4` | Number of stacked target-to-history layers, paper symbol `M`. |
| `stca_num_heads` | `null` | Attention heads, paper symbol `h`; `null` inherits `model.num_heads`. |
| `stca_expansion_ratio` | `4` | SwiGLU hidden expansion, paper symbol `r`. |
| `stca_parameter_group` | `null` | Optional shared STCA weight group for schemas that split action families into separate streams. |
| `stca_history_group` | `null` | Optional chronological merge group; the first configured member owns the group's single paper `z`. |
| `target_inputs` | required | Ordered scalar candidate inputs fused into the target token `x_t`. |
| `rankmixer_summary_tokens` | `1` | STCA produces exactly the single final target-aware token `z`. |

The resolved head count must divide `stca_dim` (paper example `d=256`). STCA is intentionally
accepted only by `rankmixer` and `mdl_rankmixer`. A paper-aligned STCA config
must also declare `max_length`, a scalar `time_delta_field`, and truncation that
keeps the recent temporal suffix (`newest_to_oldest + head` or
`oldest_to_newest + tail`).

## Architecture fidelity

For every layer `i`, the encoder:

1. applies a layer-specific dimension-preserving SwiGLU FFN and LayerNorm to
   each history token;
2. applies that same layer transform to the current target/fused query;
3. computes exact multi-head single-query target-to-history attention;
4. constructs the next query from all preceding layer outputs and the original
   target.

After the last layer, all layer outputs and the original target are projected
to `stca_dim` and passed through the paper's final SwiGLU FFN.
The resulting `z` occupies one dedicated RankMixer feature-token position and
is projected from `stca_dim` to `model.token_dim` as one indivisible token.
Ordinary feature inputs are equal-width sliced only across the remaining token
positions, so `z` is never split or mixed with an adjacent feature slice.

When physical storage splits one logical history into action-family columns,
give those sequences the same `stca_history_group`. Their valid events are
merged and stably sorted from oldest to newest using the configured scalar
`time_delta_field` (larger request-time-minus-event-time first). Every physical
stream retains its own learned action-type embedding, while one learned
position embedding is applied after the global merge. Padding is compacted
behind all valid events. Only the first member is exposed to downstream
tokenization, so the group produces exactly one `z`, matching the paper's
single chronological `(video, action_type)` history. Timestamp sorting remains
FP32 even when model activations use BF16.

The attention implementation uses the paper's algebraic reordering:

```text
softmax((((q W_Q) W_K^T) X^T) / sqrt(d_h)) X W_V
```

It does not materialize length-dependent projected K/V tensors. Padding is
strictly masked, an all-invalid history contributes an exact zero attention
summary, and an empty history remains a valid target-only input rather than
producing NaNs. The token-wise history SwiGLU can be evaluated in bounded
`runtime.sequence_projection_chunk_tokens` chunks; this changes neither values
nor gradients and limits the forward-time activation peak for long histories.

## Request-level batches

When a sequence payload includes `row_indices`, history embeddings and every
history-side SwiGLU transform are computed once per request. Candidate queries
are grouped by request for attention. The implementation pads only the small
target axis and does not `index_select` or save a
`[candidate_count, history_length, dim]` history copy. It produces the same
values and gradients as explicit history replication.

STCA is target-dependent, so its final `z` token cannot be cached as a
candidate-independent request summary. Only history-side work is shared.

The generated STCA production profile enables request-deduplicated train/test
payloads and uses `training.loss_reduction: mean_per_request_per_task`. This is
the paper's RLB objective: mean BCE over targets inside each request, then mean
over requests (independently for each configured task).

Production configs can also be generated directly:

```bash
python scripts/build_mdl_rankmixer_config.py \
  --model mdl_rankmixer \
  --sequence-encoder stca \
  --report /path/to/profile.json
```

Omitting `--sequence-encoder` preserves the existing `longer` default.

## Scope of paper alignment

The STCA encoder equations are implemented literally: dimension-preserving
bias-free SwiGLU, the layer-shared history/query transform, LayerNorm placement,
multi-head single-query attention, all-prior-layer query fusion, and the final
unnormalized `z` SwiGLU. The optimized path is tested against materialized K/V
for both values and every gradient. Configured per-source recent-suffix
truncation, time-delta input, unified `(video, action_type, position)` history,
one `z`, and RLB reuse/loss are also implemented.

The production generator puts all nine physical action-family streams in one
`stca_history_group`; therefore they are globally reordered into one history,
run through one STCA stack and one target projector, and emit one `z`.
Per-stream event projectors and action-type embeddings remain necessary because
the source schema stores heterogeneous event fields. A hand-written config
without `stca_history_group` intentionally retains independent per-sequence
plugin behavior. `stca_parameter_group` remains available when independent
histories should share weights without being chronologically merged.

The paper's U-shaped Beta stochastic-length curriculum, batch token-budget
redistribution, and flattened ragged custom kernel are data/training-pipeline
features rather than encoder equations. They are not enabled merely by setting
`encoder: stca`; static `max_length` truncation and the existing padded batch
layout remain in effect. The paper's staged 512-to-2048 pretraining recipe is
also an experiment/training schedule rather than part of the encoder module.

## Paper-to-code audit

The following mapping uses line numbers from `paper/STCA/main.tex`:

| Paper | Requirement | Implementation status |
| --- | --- | --- |
| 240–249 | One chronological history of `(video, action type)` tokens and one candidate target | A history group merges physical action streams chronologically and constructs one target token. Exact at the logical-token boundary. |
| 280–293, Eq. (1)–(3) | Dimension-preserving bias-free SwiGLU; row-wise history transform; LayerNorm; the layer-1 transform also constructs `q^(1)` | Implemented literally. Each layer owns one shared SwiGLU/LayerNorm transform used by its history and query path. |
| 295–313, Eq. (4)–(6) | `h` independent target-to-history heads, `1/sqrt(d_h)` scaling, head concatenation and `W_O` | Implemented literally, including strict padding masks. |
| 315–324, Eq. (7) | Query `i+1` consumes every earlier `o^(1..i)` plus the original `x_t`, then `W_C`, layer `i+1` SwiGLU and LayerNorm | Implemented literally; no residual, dropout, or omitted earlier summary is inserted. |
| 326–345 | Final `z` consumes every layer output plus original target, then `W_Z` and a final SwiGLU; `z` enters RankMixer; BCE objective | Implemented. The final equation has no LayerNorm, so the code intentionally has none. `z` occupies one dedicated RankMixer token and BCE is evaluated with the equivalent numerically stable logits form. |
| 348–360 and 709–745 | Reordered exact single-query attention without materialized `XW_K`/`XW_V` | Implemented literally. Optimized and materialized paths are checked for outputs and all gradients. |
| 364–383 and 761–770 | Request-level history reuse and request-first target averaging | Implemented for request-deduplicated batches: history transforms are request-major, targets carry a candidate-to-request map, and loss averages targets before requests. Even all-unique request batches retain an explicit identity map. |
| 388–431 and 772–826 | U-shaped Beta stochastic lengths, global token-budget balancing and custom flattened ragged kernel | Not implemented by the encoder option; these remain separate data/training-system work. |
| 690–697 | 512→2048 staging and sequence-subnetwork pretraining before joint finetuning | Not automated; this remains an experiment schedule. |

The overview figure labels its final fusion block “SwiGLUFFN & LayerNorm”, while
the explicit final-token equation at lines 327–330 specifies only
`SwiGLUFFN(... W_Z)` and omits LayerNorm. The implementation treats the explicit
equation as authoritative. Input fusion details—such as whether video,
action-type, position and time-delta embeddings are added or concatenated—are
not specified mathematically in the paper; the implementation therefore makes
that boundary explicit instead of claiming an unrecoverable private-code
replica.
