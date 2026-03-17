# Optimization Execution Path

Date: 2026-03-17
Repo: `/home/asturian/rwkv-rs-optimized`
Inputs:

- Tracy traces:
  - `/home/asturian/rwkv-rs-optimized/tracy/0p1b.tracy`
  - `/home/asturian/rwkv-rs-optimized/tracy/0p1b 加速结果.tracy`
- Reference objects:
  - `/home/asturian/rwkv-rs-optimized/reference object/fused_addcmul.py`
  - `/home/asturian/rwkv-rs-optimized/reference object/fused_k_update.py`
  - `/home/asturian/rwkv-rs-optimized/reference object/fused_recurrent.py`
- Existing notes:
  - `/home/asturian/rwkv-rs-optimized/Optimization-docs/tracy_tmix_findings_2026-03-14.md`
  - `/home/asturian/rwkv-rs-optimized/Optimization-docs/token_shift_analysis_2026-03-15.md`
  - `/home/asturian/rwkv-rs-optimized/Optimization-docs/reference_object_mapping_2026-03-16.md`

## Current Reading

The latest accelerated trace confirms that `wkv7` is still not the first optimization target.
The main remaining opportunities are around:

1. `weight_prepare` grouped fanout cost
2. `value_residual_lora`
3. `gated_readout.gate`
4. `normalize_removal_key` and `group_norm`
5. executor-side `scatter_state`

Important update versus the older notes:

- `weight_prepare` has already been partially regrouped.
- The bottleneck has shifted from several separate branch ops to:
  - grouped packing work,
  - remaining ungrouped branches,
  - and decode-heavy small-op overhead.

## Execution Path

### Phase 1: inference-only prepacking for `weight_prepare`

Status: completed

Goal:

- eliminate per-forward `Tensor::stack(...)` cost for:
  - `mix_params`
  - grouped projection weights
  - grouped LoRA weights and bias

Reason:

- the accelerated trace still shows non-trivial self time in:
  - `weight_prepare.scale_diff`
  - `weight_prepare.grouped_projection`
  - `weight_prepare.grouped_lora`
  - `weight_prepare.mix_inputs`

Implementation rule:

- inference only
- do not change training semantics
- prepack after loading model weights

Implemented:

- added inference cache preparation hook after weight load
- prepacked:
  - `mix_params`
  - grouped projection weights
  - grouped LoRA weights and bias
- `weight_prepare.forward()` now reuses the packed tensors when available

Validation:

- `cargo check -p rwkv-lm --example rwkv-lm-infer --no-default-features --features inferring,wgpu`
- `source /home/asturian/use_cuda_12_8.sh >/dev/null && cargo check -p rwkv-lm --example rwkv-lm-infer --no-default-features --features inferring,cuda`

## Phase 2: decode-specialized fast path

Status: in progress

Goal:

- bypass generic `token_shift_diff()` materialization for `context_length == 1`
- compute decode-time mixed inputs directly from:
  - current embedding
  - external shift
  - per-branch coefficients

Reason:

- decode dominates call counts
- Tracy shows cumulative cost in:
  - `token_shift_diff.external_diff`
  - `token_shift_diff.decode_fastpath`
  - `token_shift_diff.final_cat`
  - `token_shift_diff.tail_blend`

Implemented so far:

- added a decode-specialized `token_shift_diff_decode()` helper
- `weight_prepare` now routes `context_length == 1` through the decode helper
- `channel_mixer` now routes `context_length == 1` through the decode helper
- `weight_prepare` now has a decode-specialized inner execution path using 2D grouped inputs
- `channel_mixer` now has a decode-specialized 2D execution path
- `gated_readout` now routes `context_length == 1` through a decode-specialized 2D path
- the decode path now avoids generic 3D gate input, gate LoRA, reshape-heavy bonus assembly, and
  output projection flow
- `TimeMixer` now avoids cloning the full `Wkv7ForwardInput` bundle and uses a direct decode
  `embedded_token_shift` squeeze path when `context_length == 1`

Remaining:

- push the decode branch further down so mixed inputs themselves can be built directly without
  the generic fanout-shaped tensor path
- evaluate whether `TimeMixer` can bypass some outer reshape/mask work for decode as well

## Phase 3: extend fanout fusion to remaining branches

Status: in progress

Goal:

- bring `value_residual_lora` and `gate_input/gate` into the same shared-input mental model

Reason:

- both still depend on the same `embedded_context + token_shifted_diff * coeff` structure
- this matches the shape of `fused_addcmul.py`

Implemented so far:

- `weight_prepare` now folds `value_residual_lora` into the grouped LoRA path when present
- because LoRA ranks differ across branches, grouped packing now pads to the max rank with zeros
  during inference cache prep and fallback packing
- `gated_readout` now prepares inference-only packed tensors for:
  - `param_gate`
  - output gate LoRA weights
  - receptance-key bonus scale
  - output projection weight
- `gated_readout.forward()` and its decode fast path now reuse those packed tensors when available

## Phase 4: specialized normalization and gated readout cleanup

Status: in progress

Goal:

- replace generic L2 normalization path with a specialized implementation
- investigate fusion-friendly handling for:
  - `(group_norm(out) + bonus) * gate`

Reason:

- `normalize_removal_key` and `group_norm` are still material hotspots

Implemented so far:

- `weight_prepare` now uses a dedicated L2 normalization helper for `removal_key`
  instead of the generic `abs -> pow -> sum -> pow` path

## Phase 5: executor-side state movement

Status: pending

Goal:

- reduce `gather_state` / `scatter_state` overhead

Reason:

- trace still shows visible executor-side overhead outside the model kernel path

## Immediate Success Criteria

Phase 1 is successful if:

1. inference code precomputes grouped tensors once after weight load
2. forward path reuses those packed tensors
3. training path remains unchanged unless the cache is explicitly enabled
4. the project still builds cleanly
