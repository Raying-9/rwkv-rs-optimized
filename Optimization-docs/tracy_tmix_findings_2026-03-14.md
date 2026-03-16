# Tracy TMix Findings

Date: 2026-03-14 22:45:05 +0800
Repo: `/home/asturian/rwkv-rs`
Target: `examples/rwkv-lm`, default `rwkv-lm-0.1b` inference path
Profiler: Tracy

## Scope

This note records the current profiling findings from the RWKV inference path after adding Tracy markers to:

- `rwkv.infer.execution_loop.*`
- `rwkv.infer.executor.*`
- `rwkv.infer.model.weight_prepare.*`
- `rwkv.infer.model.gated_readout.*`

The main focus here is TMix internals rather than prefill/decode scheduling policy.

## High-Level Conclusion

For the current profiling run, the main hotspot inside TMix is not `wkv7` itself.
The dominant cost is concentrated in the two sides around it:

1. `weight_prepare`
2. `gated_readout`

This suggests the main bottleneck is currently in tensor preparation, LoRA-related paths, gating, and normalization, rather than the WKV7 recurrent core kernel.

## Most Expensive Sub-Stages

The following zones were observed as very expensive:

- `rwkv.infer.model.weight_prepare.token_shift_diff`
- `rwkv.infer.model.weight_prepare.learning_rate_lora`
- `rwkv.infer.model.weight_prepare.value_residual_lora`
- `rwkv.infer.model.gated_readout.gate`
- `rwkv.infer.model.gated_readout.group_norm`

Interpretation:

- `token_shift` is a major cost in input preparation.
- LoRA paths in `weight_prepare` are a major cost.
- output gate computation in `gated_readout` is also expensive.
- `group_norm` is a meaningful hotspot by itself.

## Secondary Hotspots

The following zones were observed as moderately expensive:

- `rwkv.infer.model.weight_prepare.projection_receptance`
- `rwkv.infer.model.weight_prepare.weight_decay_lora`
- `rwkv.infer.model.gated_readout.apply_gate`
- `rwkv.infer.model.gated_readout.projection_output`

Interpretation:

- projection and post-gate output processing are non-trivial but are not the first-order bottlenecks compared with the top group above.

## Current Reading of the Bottleneck

The current evidence points to the following:

1. The bottleneck is likely dominated by many medium/small tensor operations rather than a single dominant recurrent kernel.
2. LoRA-related forward paths are more expensive than initially expected.
3. `token_shift` and `group_norm` deserve direct implementation review.
4. The core `wkv7` path is not currently the main optimization priority for this run.

## Candidate Root Causes

Based on the current profile, likely causes include:

- too many small kernel launches
- insufficient fusion around LoRA paths
- expensive materialization or copying in `token_shift`
- costly normalization under the current tensor shapes
- extra reshape/broadcast/mask overhead around gate and readout paths

## Prioritized Next Steps

Recommended optimization order:

1. Inspect `token_shift` implementation for avoidable copies, reshapes, and masking overhead.
2. Inspect LoRA forward implementation used by:
   - `learning_rate_lora`
   - `value_residual_lora`
   - `gated_readout.gate`
3. Inspect `group_norm` cost and surrounding reshape/layout transitions.
4. Only after the above, revisit `projection_output` and secondary projections.

## Practical Takeaway

At the current stage, optimization effort should not start from `wkv7`.
It should start from:

- `token_shift`
- LoRA forward paths
- `group_norm`

These are the best current candidates for reducing TMix wall time.

## Related Code Areas

- `/home/asturian/rwkv-rs/crates/rwkv-nn/src/modules/time_mixer/weight_prepare.rs`
- `/home/asturian/rwkv-rs/crates/rwkv-nn/src/modules/time_mixer/gated_readout.rs`
- `/home/asturian/rwkv-rs/crates/rwkv-nn/src/modules/time_mixer/mod.rs`
