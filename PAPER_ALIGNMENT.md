# Paper Alignment

This repository keeps three model surfaces aligned with the local paper sources:

- `mdl_rankmixer`: MDL scenario/task tokens with a RankMixer-style feature-token backbone.
- `onetrans`: OneTrans S/NS tokenizer, mixed causal attention, mixed FFN, and pyramid stack.
- `mdl_onetrans`: Hybrid MDL + OneTrans, using OneTrans-derived feature tokens with MDL domain-aware attention and task/scenario tokens.

## Open Deviations

- The default YAML is a secure-environment template. Field names, vocab strategy, and token inputs must be adapted locally.
- `mdl_rankmixer` defaults to `tokenization.feature_tokenizer: rankmixer` with 32 feature tokens of width 768, excluding scenario/task tokens. Encoded inputs are concatenated in YAML order and must exactly equal `num_feature_tokens * token_dim`; implicit zero padding is disabled before the direct reshape. The default template now uses per-sequence multi-slice summaries plus a small explicit context dense slice rather than a large packed placeholder.
- `auto_split` remains available only as an explicit OneTrans-style engineering fallback.
- Multi-field behavior sequences are first fused per step. `mdl_rankmixer` consumes a LONGER-style sequence encoder with target/global tokens, recent query compression, Token Merge, InnerTrans, time-delta side projection, cross-causal attention, self-causal refinement, and configurable multi-slice RankMixer summaries.
- `FeatureConfig.kind: sequence` is a legacy single-column compatibility path; new behavior sequences should use top-level `sequences`.
- `model.use_request_cache` is implemented for local prediction through model-side precomputed sequence caches. It is still not a production cross-request KV cache service.
