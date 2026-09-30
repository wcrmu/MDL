# UniFormer and MORE on the current feature contract

Both models use the same coarse pack as `configs/mixformer.yaml`: the rankmixer
non-sequential inputs, the eight raw behavior streams, and the three labels
`fst_cart`, `upid_pay`, and `cateid_filter`. Fine configs extend
`mixformer_fine.yaml` the same way.

## What is aligned

| Current field | UniFormer | MORE |
| --- | --- | --- |
| Request-axis columns | User query tokens, placed first | The user half of `F` |
| Candidate-axis columns | Item query tokens | The item half of `F` |
| Head split | Eight explicit whole-field semantic groups in `tokenization.ns_tokens`, four request groups and four candidate groups; independent SwiGLU projections | Eight even-split tokens, `M=8` |
| Eight raw streams: `impr`, `clk_long`, `view_long`, `cart_long`, `buy_long`, `flatten_query_hash`, `srch_q2i`, `ups_clk_sku` | One lazy KV and one sequence SwiGLU each. The eight outputs are mixed with a softmax, because the paper's two-stream `alpha` does not cover eight families. `srch_q2i` stays its own stream: this contract gives it no target-attention inputs | Timestamp-fused into one `S`, with learned stream type and valid-event position embeddings |
| Three train labels | Three TIM queries and three heads | Three Private Anchors and three heads |

UniFormer preserves whole fields in declared semantic groups; MORE retains the
MixFormer concatenate-and-slice projector. Behavior events use the existing
per-stream projection into `token_dim`. Long K/V remain request-major: candidate
queries are grouped by `row_indices`, and no candidate-expanded history is made.
The same mapping is honored even when the request and candidate counts coincide.
MORE valid-event positions start at 1 after timestamp sorting; padding stays 0.

## What the papers do not supply for this pack

* UniFormer does not say how many of its "3 layers" are FIM versus TIM.
  Here FIM depth is `model.num_layers` (2) and TIM depth is
  `model.uniformer_tim_layers` (1).
* There is no task-id column. UniFormer task queries are a learned task vector
  plus the mean of the user tokens.
* MORE's published default is `M=15`, `K=8`, `T=9`, `d=256`. This pack has
  three tasks and eight feature tokens. `K=5` makes
  `M+K+T=16`, and `token_dim=128` is divisible by 16, which the masked token
  mix requires.
* MORE uses a stream-type proxy, **not** the paper's Cartesian action id. These streams are
  already separated by behavior family; they are not a per-event multi-label
  action combination.
* Kuaishou and Momo GAUC numbers are not comparable to this data.

`runtime.compile` defaults off. Both models now support dense-only compilation:
UniFormer feature/task FFNs, MORE gated mixing and task enhancement. Sparse
tables, dynamic packing and attention metadata stay eager. CPU AOT-eager graph
validation is not evidence of GPU/Inductor throughput improvement.

Strict `attention_backend: flash` now requires CUDA BF16/FP16 plus flash-attn
varlen; UniFormer also checks padded SDPA Flash capability. `sdpa` is the portable
reference. Unsupported activation checkpoint/CUDA graph/cross-forward cache
options are rejected rather than silently ignored. Within-forward KV reuse is
supported; user-side FIM reuse across candidates remains an optimization target.

Production configs preserve RMSprop + row-wise Adagrad. Optional
`uniformer_adamw.yaml` and `more_adam.yaml` isolate dense-optimizer ablations;
they do not claim original-data or all-parameter optimizer reproduction.

The four production variants use separate `*_v2` checkpoint namespaces. Semantic
projectors and position embeddings change the state dict; pre-fix checkpoints
must not be automatically resumed into these architectures.

The current workspace uses an append-only `RankEmbeddingTable` behind the legacy
`gset_table` attribute. Its capacity is driven by observed IDs, **not** bounded by
`training.gset.capacity`, and it does not implement GSET eviction. This pre-existing
design is preserved; capacity/eviction must be evaluated separately before a long
production run. Restore now allocates saved storage and retains keys/step count;
optimizer binding and clipping follow grown parameters.

For exact validation scope and remaining requirements, see
[the repair report](uniformer_more_fix_20260930.md).
