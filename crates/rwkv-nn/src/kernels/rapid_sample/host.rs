use burn::tensor::{
    DType, Shape,
    ops::{FloatTensor, IntTensor},
};
use burn_cubecl::{
    BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement,
    cubecl::prelude::ScalarArg,
    cubecl::{CubeCount, CubeDim},
    kernel::{cast, into_contiguous},
    ops::numeric::empty_device,
};

use crate::kernels::rapid_sample::{
    RapidSampleOutput,
    kernel::{
        RapidSampleConfig, RapidSamplePenaltyParamsLaunch, RapidSampleRepetitionInputsLaunch,
        RapidSampleRepetitionOutputsLaunch, RapidSampleSamplingParamsLaunch,
        RapidSampleTemperatureInputsLaunch, RapidSampleTemperatureOutputsLaunch,
        rapid_sample_repetition_temperature_topk_topp_kernel,
        rapid_sample_temperature_topk_topp_kernel,
    },
};

#[allow(clippy::too_many_arguments)]
pub(crate) fn rapid_sample_topk_topp_impl<
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
>(
    logits: FloatTensor<CubeBackend<R, F, I, BT>>,
    states: IntTensor<CubeBackend<R, F, I, BT>>,
    inv_temperatures: FloatTensor<CubeBackend<R, F, I, BT>>,
    top_ks: IntTensor<CubeBackend<R, F, I, BT>>,
    top_ps: FloatTensor<CubeBackend<R, F, I, BT>>,
    penalties: Option<(
        FloatTensor<CubeBackend<R, F, I, BT>>,
        FloatTensor<CubeBackend<R, F, I, BT>>,
        FloatTensor<CubeBackend<R, F, I, BT>>,
        FloatTensor<CubeBackend<R, F, I, BT>>,
    )>,
) -> RapidSampleOutput<FloatTensor<CubeBackend<R, F, I, BT>>, IntTensor<CubeBackend<R, F, I, BT>>> {
    let logits = into_contiguous(logits);
    let logits = cast::<R>(logits, DType::F32);

    let states = into_contiguous(states);
    let states = cast::<R>(states, DType::U32);

    let inv_temperatures = into_contiguous(inv_temperatures);
    let inv_temperatures = cast::<R>(inv_temperatures, DType::F32);

    let top_ks = into_contiguous(top_ks);
    let top_ks = cast::<R>(top_ks, DType::U32);

    let top_ps = into_contiguous(top_ps);
    let top_ps = cast::<R>(top_ps, DType::F32);

    let client = logits.client.clone();
    let device = logits.device.clone();
    let shape = logits.shape.clone();

    debug_assert_eq!(
        shape.num_dims(),
        2,
        "logits must have shape [batch_size, vocab_size]"
    );

    let batch_size = shape.dims[0];
    let vocab_size = shape.dims[1];

    debug_assert!(batch_size > 0, "batch_size must be > 0");
    debug_assert!(vocab_size > 0, "vocab_size must be > 0");
    debug_assert_eq!(
        vocab_size % 4,
        0,
        "vocab_size must be a multiple of 4 for rapid_sample, got {vocab_size}"
    );

    let expected_1d = Shape::new([batch_size]);
    debug_assert_eq!(
        states.shape, expected_1d,
        "states must have shape [batch_size]"
    );
    debug_assert_eq!(
        inv_temperatures.shape, expected_1d,
        "inv_temperatures must have shape [batch_size]"
    );
    debug_assert_eq!(
        top_ks.shape, expected_1d,
        "top_ks must have shape [batch_size]"
    );
    debug_assert_eq!(
        top_ps.shape, expected_1d,
        "top_ps must have shape [batch_size]"
    );

    let max_units_per_cube = client.properties().hardware.max_units_per_cube as usize;
    let block_size = select_block_size(vocab_size, max_units_per_cube);
    debug_assert!(
        vocab_size <= block_size * block_size,
        "vocab_size too large for block_size={block_size} (max={}): vocab_size={vocab_size}",
        block_size * block_size
    );

    let config = RapidSampleConfig {
        block_size,
        num_warps: block_size / 32,
    };

    let token_ids =
        empty_device::<R, i32>(client.clone(), device.clone(), Shape::new([batch_size]));
    let probs = empty_device::<R, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch_size, vocab_size]),
    );

    let cube_count = CubeCount::Static(batch_size as u32, 1, 1);
    let cube_dim = CubeDim::new_1d(block_size as u32);

    match penalties {
        None => {
            rapid_sample_temperature_topk_topp_kernel::launch(
                &client,
                cube_count,
                cube_dim,
                RapidSampleTemperatureInputsLaunch::new(logits.as_tensor_arg(4)),
                RapidSampleTemperatureOutputsLaunch::new(
                    token_ids.as_tensor_arg(1),
                    states.as_tensor_arg(1),
                    probs.as_tensor_arg(4),
                ),
                RapidSampleSamplingParamsLaunch::new(
                    inv_temperatures.as_tensor_arg(1),
                    top_ks.as_tensor_arg(1),
                    top_ps.as_tensor_arg(1),
                ),
                ScalarArg::new(vocab_size as u32),
                config,
            )
            .expect("rapid_sample_temperature_topk_topp_kernel should never fail");

            RapidSampleOutput {
                token_ids,
                states,
                probs,
                penalties: None,
            }
        }
        Some((penalties, presence_penalty, repetition_penalty, penalty_decay)) => {
            let penalties = into_contiguous(penalties);
            debug_assert_eq!(
                penalties.dtype,
                DType::F32,
                "rapid_sample: penalties must be F32, got {:?}",
                penalties.dtype
            );
            debug_assert_eq!(
                penalties.shape, shape,
                "penalties must have the same shape as logits"
            );

            let presence_penalty = into_contiguous(presence_penalty);
            let presence_penalty = cast::<R>(presence_penalty, DType::F32);
            let repetition_penalty = into_contiguous(repetition_penalty);
            let repetition_penalty = cast::<R>(repetition_penalty, DType::F32);
            let penalty_decay = into_contiguous(penalty_decay);
            let penalty_decay = cast::<R>(penalty_decay, DType::F32);

            rapid_sample_repetition_temperature_topk_topp_kernel::launch(
                &client,
                cube_count,
                cube_dim,
                RapidSampleRepetitionInputsLaunch::new(logits.as_tensor_arg(4)),
                RapidSampleRepetitionOutputsLaunch::new(
                    token_ids.as_tensor_arg(1),
                    penalties.as_tensor_arg(4),
                    states.as_tensor_arg(1),
                    probs.as_tensor_arg(4),
                ),
                RapidSamplePenaltyParamsLaunch::new(
                    inv_temperatures.as_tensor_arg(1),
                    top_ks.as_tensor_arg(1),
                    top_ps.as_tensor_arg(1),
                    presence_penalty.as_tensor_arg(1),
                    repetition_penalty.as_tensor_arg(1),
                    penalty_decay.as_tensor_arg(1),
                ),
                ScalarArg::new(vocab_size as u32),
                config,
            )
            .expect("rapid_sample_repetition_temperature_topk_topp_kernel should never fail");

            RapidSampleOutput {
                token_ids,
                states,
                probs,
                penalties: Some(penalties),
            }
        }
    }
}

fn desired_block_size(vocab_size: usize) -> usize {
    // The rapid-sampling 2-phase mapping supports vocab sizes up to `block_size^2`.
    if vocab_size <= 256 * 256 { 256 } else { 1024 }
}

fn select_block_size(vocab_size: usize, max_units_per_cube: usize) -> usize {
    let desired = desired_block_size(vocab_size);
    if desired <= max_units_per_cube {
        return desired;
    }

    // Try to fallback to 256 if the vocab size allows it.
    if 256 <= max_units_per_cube && vocab_size <= 256 * 256 {
        return 256;
    }

    panic!(
        "rapid_sample: required block_size={desired} exceeds device \
         max_units_per_cube={max_units_per_cube} (vocab_size={vocab_size})"
    );
}
