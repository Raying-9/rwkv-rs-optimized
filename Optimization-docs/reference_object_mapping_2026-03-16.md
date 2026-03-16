# Reference Object Mapping

## Purpose

This document maps the reference implementations under
[`reference object`](/home/asturian/rwkv-rs/reference%20object)
to the current RWKV Rust inference hotspots observed in Tracy.

The goal is not to copy the Python/Triton code directly. The goal is to identify:

- which ideas are structurally relevant,
- which current Rust hotspots they correspond to,
- and which directions are likely worth engineering effort.

## Current Hotspots We Care About

These are the steady-state inference hotspots that remain relevant after separating prefill from token generation:

### `weight_prepare`

- `rwkv.infer.model.weight_prepare.projection_receptance`
- `rwkv.infer.model.weight_prepare.learning_rate_lora`
- `rwkv.infer.model.weight_prepare.value_residual_lora`
- `rwkv.infer.model.weight_prepare.weight_decay_lora`

### `gated_readout`

- `rwkv.infer.model.gated_readout.gate`
- `rwkv.infer.model.gated_readout.group_norm`
- `rwkv.infer.model.gated_readout.apply_gate`
- `rwkv.infer.model.gated_readout.projection_output`

## Reference Objects

### 1. `fused_addcmul.py`

File:
[`fused_addcmul.py`](/home/asturian/rwkv-rs/reference%20object/fused_addcmul.py)

Core idea:

- Read one `hidden` tensor and one `delta` tensor.
- Produce multiple output branches in one fused pass:
  - `hidden + delta * r`
  - `hidden + delta * w`
  - `hidden + delta * k`
  - `hidden + delta * v`
  - `hidden + delta * a`
  - optional `hidden + delta * g`

This is the most relevant reference object for the current Rust bottlenecks.

### 2. `fused_k_update.py`

File:
[`fused_k_update.py`](/home/asturian/rwkv-rs/reference%20object/fused_k_update.py)

Core idea:

- Fuse one short elementwise update chain:
  - `k * (1 + (a - 1) * ka)`

This is structurally useful as an example of "small post-processing chain fused into one pass",
but it is not the best first target for the current hotspot profile.

### 3. `fused_recurrent.py`

File:
[`fused_recurrent.py`](/home/asturian/rwkv-rs/reference%20object/fused_recurrent.py)

Core idea:

- Fuse the recurrent RWKV7 core itself.

This is relevant to `wkv7`, but `wkv7` is not the primary bottleneck in the current profile.
So this file is informative, but not the immediate next step.

## Mapping Table

| Reference object | What it really optimizes | Closest Rust location | Relevance |
| --- | --- | --- | --- |
| `fused_addcmul.py` | Multi-branch `hidden + delta * coeff` generation in one pass | [`weight_prepare.rs`](/home/asturian/rwkv-rs/crates/rwkv-nn/src/modules/time_mixer/weight_prepare.rs) | High |
| `fused_addcmul.py` | Same pattern could apply to output-side gate input preparation | [`gated_readout.rs`](/home/asturian/rwkv-rs/crates/rwkv-nn/src/modules/time_mixer/gated_readout.rs) | Medium |
| `fused_k_update.py` | Short elementwise chain fusion | [`gated_readout.rs`](/home/asturian/rwkv-rs/crates/rwkv-nn/src/modules/time_mixer/gated_readout.rs) | Medium-low |
| `fused_recurrent.py` | Recurrent core fusion | `wkv7` path | Low for current priority |

## Direct Mapping to Current Rust Code

### A. `fused_addcmul.py` vs `weight_prepare`

Current Rust file:
[`weight_prepare.rs`](/home/asturian/rwkv-rs/crates/rwkv-nn/src/modules/time_mixer/weight_prepare.rs)

Current structure in Rust is roughly:

1. Build `token_shifted_diff`
2. Build multiple branch inputs from:
   - `embedded_context`
   - `token_shifted_diff`
   - per-branch learned scale tensors
3. Run per-branch projection / LoRA

That is conceptually very close to the fused-addcmul pattern:

```text
shared base:  embedded_context
shared delta: token_shifted_diff

branch 1: embedded_context + token_shifted_diff * param_receptance
branch 2: embedded_context + token_shifted_diff * param_learning_rate
branch 3: embedded_context + token_shifted_diff * param_weight_decay
branch 4: embedded_context + token_shifted_diff * param_key
branch 5: embedded_context + token_shifted_diff * param_value
```

This is the most important structural analogy in the entire reference set.

### Why this matters

The current profile says the expensive part is not only the LoRA itself, but the whole
`weight_prepare` stage before `wkv7`. That makes the multi-branch input generation cost relevant.

Even if we do not write a custom kernel yet, this reference strongly suggests that the right mental model is:

- do not treat each branch as an isolated tiny pipeline,
- treat them as one shared-input fanout stage.

### Practical implication

The next optimization direction should likely be:

- first reduce duplicated branch construction,
- then consider whether branch generation can be grouped or fused,
- only then revisit LoRA or projection internals.

## `weight_prepare` Branch Map

This table is the most useful working view for future optimization.

| Rust output path | Current building pattern | Main expensive step after build | Candidate idea from reference |
| --- | --- | --- | --- |
| `projection_receptance` | `embedded_context + token_shifted_diff * param_receptance` | linear projection | Batch/fuse branch preparation |
| `learning_rate_lora` | `embedded_context + token_shifted_diff * param_learning_rate` | LoRA forward | Batch/fuse branch preparation |
| `weight_decay_lora` | `embedded_context + token_shifted_diff * param_weight_decay` | LoRA forward | Batch/fuse branch preparation |
| `projection_key` | `embedded_context + token_shifted_diff * param_key` | linear projection | Batch/fuse branch preparation |
| `projection_value` | `embedded_context + token_shifted_diff * param_value` | linear projection | Batch/fuse branch preparation |
| `value_residual_lora` | built from value-related mixed input | LoRA forward | Shared input-prep cleanup first |

## B. `fused_addcmul.py` vs `gated_readout`

Current Rust file:
[`gated_readout.rs`](/home/asturian/rwkv-rs/crates/rwkv-nn/src/modules/time_mixer/gated_readout.rs)

Relevant pieces:

- `gate_input = embedded_context + token_shifted_diff * param_gate`
- `gate = output_gate_lora.forward(gate_input)`
- `out_gated = (group_norm(wkv7_out) + bonus) * gate`
- `projection_output.forward(out_gated)`

The first line has the same shape as fused addcmul:

```text
gate_input = hidden + delta * coeff
```

But unlike `weight_prepare`, this is only one branch, not a multi-branch fanout. So the reference value here is smaller.

What is still useful:

- it confirms that the pattern is fusion-friendly,
- and it suggests `gate_input` should be viewed as a fused pre-processing candidate rather than a separate tiny op chain.

## C. `fused_k_update.py` vs `gated_readout.apply_gate`

`fused_k_update.py` is a reminder that short elementwise algebra chains can be worth fusing when they are stable hotspots.

The closest current Rust expression is:

```text
(wkv7_forward_output_normalized + bonus) * gate
```

This is simpler than the reference update, but it lives inside a hot path:

- `group_norm`
- `apply_gate`
- `projection_output`

If we later decide to attack `gated_readout`, this chain is a reasonable candidate for a small fusion pass.
It is not the first thing to optimize now because:

- `group_norm` itself is already expensive,
- and `projection_output` is a larger downstream cost.

## D. `fused_recurrent.py` vs `wkv7`

This mapping is straightforward:

- reference object optimizes the recurrent core,
- Rust currently profiles `wkv7` as not being the primary problem.

Therefore:

- this file is architecturally useful,
- but it is not the file to imitate first if the goal is immediate latency reduction.

## LoRA-Specific Conclusion

Current Rust LoRA implementation:
[`lora.rs`](/home/asturian/rwkv-rs/crates/rwkv-nn/src/layers/lora.rs)

Important observation:

- the reference object does not contain a direct LoRA fusion equivalent,
- so there is no one-to-one "copy this fused LoRA kernel" candidate in `reference object`.

Current LoRA forward is:

1. `x.matmul(w_a)`
2. activation
3. `x.matmul(w_b)`
4. optional bias add

This means the reference object is more useful for:

- branch input generation,
- elementwise mix fusion,

than for LoRA itself.

So the correct interpretation is:

- the reference object gives us a strong idea for optimizing the inputs feeding the hot LoRA paths,
- not a finished answer for the LoRA hotspot itself.

## Recommended Priority Order

### Priority 1

Study whether `weight_prepare` branch construction can be restructured as a shared fanout stage.

This is the strongest takeaway from `fused_addcmul.py`.

### Priority 2

Only after that, revisit whether the remaining cost is dominated by:

- LoRA matmul A,
- LoRA activation,
- LoRA matmul B,
- projection matmuls.

### Priority 3

If `gated_readout` remains dominant after `weight_prepare` cleanup:

- investigate `gate_input`,
- then `apply_gate`,
- then `projection_output`.

### Deferred

Do not prioritize `wkv7` kernel work based on the current profile.

## Suggested Engineering Strategy

### Low-risk

- keep current Rust decomposition,
- add more measurements inside LoRA if needed.

### Medium-risk, high-value

- restructure `weight_prepare` so multiple `embedded_context + token_shifted_diff * coeff` branches are produced in a more shared way,
- reduce repeated materialization before projections and LoRA.

### High-risk

- custom fused backend/kernel for branch generation,
- custom fused gate/readout path,
- grouped LoRA execution.

## Working Hypothesis

If we apply the reference-object lesson correctly, the next meaningful gain is more likely to come from:

- reducing branch-preparation overhead in `weight_prepare`,

than from:

- immediately rewriting `wkv7`,
- or trying to guess at LoRA internals without first cleaning the surrounding dataflow.

## One-Line Summary

The most useful reusable idea in `reference object` is not the recurrent kernel. It is the
`fused_addcmul` pattern: one shared base tensor, one shared delta tensor, many branch outputs in one pass.
That idea maps directly onto the current `weight_prepare` hotspot structure.
