use burn::{Tensor, prelude::Backend};

#[cfg(feature = "trace")]
macro_rules! trace_lite_scope {
    ($name:literal) => {
        let _rwkv_trace_scope = tracing::trace_span!($name).entered();
    };
}

#[cfg(not(feature = "trace"))]
macro_rules! trace_lite_scope {
    ($name:literal) => {};
}

pub fn token_shift<B: Backend>(
    embedded_context: Tensor<B, 3>,
    embedded_token_shift: Option<Tensor<B, 2>>,
    context_mask: Option<Tensor<B, 2>>,
) -> Tensor<B, 3> {
    let [batch_size, context_length, embedded_dim] = embedded_context.dims();

    // Empty contexts are rare but possible in edge cases; avoid usize underflow.
    if context_length == 0 {
        return embedded_context;
    }

    let embedded_token_shift = embedded_token_shift.unwrap_or(Tensor::zeros(
        [batch_size, embedded_dim],
        &embedded_context.device(),
    ));

    // Decode path is typically T=1; skip generic concat/mask plumbing in this case.
    if context_length == 1 {
        trace_lite_scope!("rwkv.infer.model.token_shift.decode_fastpath");
        return embedded_token_shift.unsqueeze_dim(1);
    }

    // Standard previous-token shift: [shift, x[:, 0..T-1]].
    let shifted = {
        trace_lite_scope!("rwkv.infer.model.token_shift.shifted_cat");
        Tensor::cat(
            vec![
                embedded_token_shift.clone().unsqueeze_dim(1),
                embedded_context
                    .clone()
                    .slice([0..batch_size, 0..(context_length - 1)]),
            ],
            1,
        )
    };

    let Some(context_mask) = context_mask else {
        return shifted;
    };

    let [mask_batch_size, mask_context_length] = context_mask.dims();
    debug_assert_eq!(
        (batch_size, context_length),
        (mask_batch_size, mask_context_length),
        "context_mask shape mismatch with embedded_context"
    );

    // `prev_valid_mask[t] = context_mask[t-1]` (t=0 treated as 0).
    let prev_valid_mask = {
        trace_lite_scope!("rwkv.infer.model.token_shift.prev_valid_mask");
        Tensor::cat(
            vec![
                Tensor::zeros([batch_size, 1], &embedded_context.device()),
                context_mask
                    .clone()
                    .slice([0..batch_size, 0..(context_length - 1)]),
            ],
            1,
        )
    };

    // If the previous timestep is masked (padding), keep using the external shift state.
    // This handles left padding where the prefix padding should not advance the previous token.
    let use_external_shift = {
        trace_lite_scope!("rwkv.infer.model.token_shift.use_external_shift");
        Tensor::ones([batch_size, context_length], &embedded_context.device()) - prev_valid_mask
    };

    let external_shift = {
        trace_lite_scope!("rwkv.infer.model.token_shift.external_shift");
        embedded_token_shift.unsqueeze_dim(1)
    };
    {
        trace_lite_scope!("rwkv.infer.model.token_shift.masked_blend");
        shifted.clone() + use_external_shift.unsqueeze_dim(2) * (external_shift - shifted)
    }
}

/// Return the token-shifted difference directly.
///
/// This is the main quantity consumed by TMix/CMix:
/// `prev_token_embedding - current_embedding`, with left-padding-aware handling.
pub fn token_shift_diff<B: Backend>(
    embedded_context: Tensor<B, 3>,
    embedded_token_shift: Option<Tensor<B, 2>>,
    context_mask: Option<Tensor<B, 2>>,
) -> Tensor<B, 3> {
    let [batch_size, context_length, embedded_dim] = embedded_context.dims();

    if context_length == 0 {
        return embedded_context;
    }

    let embedded_token_shift = embedded_token_shift.unwrap_or(Tensor::zeros(
        [batch_size, embedded_dim],
        &embedded_context.device(),
    ));

    let external_shift = {
        trace_lite_scope!("rwkv.infer.model.token_shift_diff.external_shift");
        embedded_token_shift.unsqueeze_dim(1)
    };
    let external_diff = {
        trace_lite_scope!("rwkv.infer.model.token_shift_diff.external_diff");
        external_shift.clone() - embedded_context.clone()
    };

    if context_length == 1 {
        trace_lite_scope!("rwkv.infer.model.token_shift_diff.decode_fastpath");
        return match context_mask {
            Some(mask) => external_diff * mask.unsqueeze_dim(2),
            None => external_diff,
        };
    }

    let first_token_diff = {
        trace_lite_scope!("rwkv.infer.model.token_shift_diff.first_token_diff");
        external_diff.clone().slice([0..batch_size, 0..1])
    };
    let prev_context_diff = {
        trace_lite_scope!("rwkv.infer.model.token_shift_diff.prev_context_diff");
        embedded_context
            .clone()
            .slice([0..batch_size, 0..(context_length - 1)])
            - embedded_context
                .clone()
                .slice([0..batch_size, 1..context_length])
    };

    let shifted_diff = match context_mask {
        Some(context_mask) => {
            let [mask_batch_size, mask_context_length] = context_mask.dims();
            debug_assert_eq!(
                (batch_size, context_length),
                (mask_batch_size, mask_context_length),
                "context_mask shape mismatch with embedded_context"
            );

            let prev_valid_mask = {
                trace_lite_scope!("rwkv.infer.model.token_shift_diff.prev_valid_mask");
                context_mask
                    .clone()
                    .slice([0..batch_size, 0..(context_length - 1)])
            };
            let external_tail_diff = {
                trace_lite_scope!("rwkv.infer.model.token_shift_diff.external_tail_diff");
                external_diff
                    .clone()
                    .slice([0..batch_size, 1..context_length])
            };
            let use_prev_context = {
                trace_lite_scope!("rwkv.infer.model.token_shift_diff.use_prev_context");
                prev_valid_mask.clone().unsqueeze_dim(2)
            };
            let use_external_shift = {
                trace_lite_scope!("rwkv.infer.model.token_shift_diff.use_external_shift");
                (Tensor::ones([batch_size, context_length - 1], &embedded_context.device())
                    - prev_valid_mask)
                    .unsqueeze_dim(2)
            };

            let tail_diff = {
                trace_lite_scope!("rwkv.infer.model.token_shift_diff.tail_blend");
                use_prev_context * prev_context_diff + use_external_shift * external_tail_diff
            };
            let shifted_diff = {
                trace_lite_scope!("rwkv.infer.model.token_shift_diff.final_cat");
                Tensor::cat(vec![first_token_diff, tail_diff], 1)
            };

            {
                trace_lite_scope!("rwkv.infer.model.token_shift_diff.final_mask");
                shifted_diff * context_mask.unsqueeze_dim(2)
            }
        }
        None => {
            trace_lite_scope!("rwkv.infer.model.token_shift_diff.final_cat_nomask");
            Tensor::cat(vec![first_token_diff, prev_context_diff], 1)
        }
    };

    shifted_diff
}

pub fn get_embedded_token_shift<B: Backend>(embedded_context: Tensor<B, 3>) -> Tensor<B, 2> {
    let [batch_size, context_length, embedded_dim] = embedded_context.dims();
    if context_length == 0 {
        Tensor::zeros([batch_size, embedded_dim], &embedded_context.device())
    } else {
        embedded_context
            .clone()
            .slice([0..batch_size, (context_length - 1)..context_length])
            .squeeze_dim(1)
    }
}
