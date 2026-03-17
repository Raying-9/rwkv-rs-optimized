use burn::tensor::{Shape, ops::FloatTensor};
use burn_cubecl::{
    BoolElement, CubeBackend, CubeElement, CubeRuntime, FloatElement, IntElement,
    cubecl::{CubeCount, CubeDim},
    kernel::into_contiguous,
    ops::numeric::empty_device,
};

use crate::kernels::wkv7_infer::{
    Wkv7InferForwardOutput,
    kernel::{
        Wkv7InferConfig, Wkv7InferForwardInputsLaunch, Wkv7InferForwardOutputsLaunch,
        wkv7_infer_forward_kernel,
    },
};

pub(crate) fn wkv7_infer_forward_impl<
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
>(
    weight_decay: FloatTensor<CubeBackend<R, F, I, BT>>,
    receptance: FloatTensor<CubeBackend<R, F, I, BT>>,
    key: FloatTensor<CubeBackend<R, F, I, BT>>,
    value: FloatTensor<CubeBackend<R, F, I, BT>>,
    removal: FloatTensor<CubeBackend<R, F, I, BT>>,
    replacement: FloatTensor<CubeBackend<R, F, I, BT>>,
    initial_state: FloatTensor<CubeBackend<R, F, I, BT>>,
    context_mask: FloatTensor<CubeBackend<R, F, I, BT>>,
) -> Wkv7InferForwardOutput<FloatTensor<CubeBackend<R, F, I, BT>>> {
    wkv7_infer_forward_impl_inner::<R, F, I, BT>(
        weight_decay,
        receptance,
        key,
        value,
        removal,
        replacement,
        initial_state,
        context_mask,
    )
}

fn wkv7_infer_forward_impl_inner<
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
>(
    weight_decay: FloatTensor<CubeBackend<R, F, I, BT>>,
    receptance: FloatTensor<CubeBackend<R, F, I, BT>>,
    key: FloatTensor<CubeBackend<R, F, I, BT>>,
    value: FloatTensor<CubeBackend<R, F, I, BT>>,
    removal: FloatTensor<CubeBackend<R, F, I, BT>>,
    replacement: FloatTensor<CubeBackend<R, F, I, BT>>,
    initial_state: FloatTensor<CubeBackend<R, F, I, BT>>,
    context_mask: FloatTensor<CubeBackend<R, F, I, BT>>,
) -> Wkv7InferForwardOutput<FloatTensor<CubeBackend<R, F, I, BT>>> {
    let weight_decay = into_contiguous(weight_decay);
    let receptance = into_contiguous(receptance);
    let key = into_contiguous(key);
    let value = into_contiguous(value);
    let removal = into_contiguous(removal);
    let replacement = into_contiguous(replacement);
    let initial_state = into_contiguous(initial_state);
    let context_mask = into_contiguous(context_mask);

    let client = weight_decay.client.clone();
    let device = weight_decay.device.clone();
    let shape = weight_decay.shape.clone();

    let batch_size = shape.dims[0];
    let context_length = shape.dims[1];
    let num_heads = shape.dims[2];
    let dim = shape.dims[3];

    debug_assert!(batch_size > 0, "batch size must be > 0");
    debug_assert!(context_length > 0, "context length must be > 0");
    debug_assert!(num_heads > 0, "num_heads must be > 0");
    debug_assert!(dim > 0, "head size must be > 0");

    let expected_initial_state_shape = Shape::new([batch_size, num_heads, dim, dim]);
    debug_assert_eq!(
        initial_state.shape, expected_initial_state_shape,
        "initial_state shape must be [batch_size, num_heads, head_size, head_size]"
    );
    debug_assert_eq!(
        receptance.shape, shape,
        "receptance shape mismatch with weight_decay"
    );
    debug_assert_eq!(key.shape, shape, "key shape mismatch with weight_decay");
    debug_assert_eq!(value.shape, shape, "value shape mismatch with weight_decay");
    debug_assert_eq!(
        removal.shape, shape,
        "removal shape mismatch with weight_decay"
    );
    debug_assert_eq!(
        replacement.shape, shape,
        "replacement shape mismatch with weight_decay"
    );

    let expected_context_mask_shape = Shape::new([batch_size, context_length]);
    debug_assert_eq!(
        context_mask.shape, expected_context_mask_shape,
        "context_mask shape must be [batch_size, context_length]"
    );

    let output = empty_device::<R, F>(client.clone(), device.clone(), shape);
    let final_state =
        empty_device::<R, F>(client.clone(), device.clone(), expected_initial_state_shape);

    let config = Wkv7InferConfig {
        context_length,
        num_heads,
        head_size: dim,
    };

    let cube_count = CubeCount::Static(num_heads as u32, batch_size as u32, 1);
    let cube_dim = CubeDim::new_1d(dim as u32);

    wkv7_infer_forward_kernel::launch::<F, R>(
        &client,
        cube_count,
        cube_dim,
        Wkv7InferForwardInputsLaunch::new(
            weight_decay.as_tensor_arg(1),
            receptance.as_tensor_arg(1),
            key.as_tensor_arg(1),
            value.as_tensor_arg(1),
            removal.as_tensor_arg(1),
            replacement.as_tensor_arg(1),
            initial_state.as_tensor_arg(1),
            context_mask.as_tensor_arg(1),
        ),
        Wkv7InferForwardOutputsLaunch::new(output.as_tensor_arg(1), final_state.as_tensor_arg(1)),
        config,
    )
    .unwrap();

    Wkv7InferForwardOutput {
        output,
        final_state,
    }
}
