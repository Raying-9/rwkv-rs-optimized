use std::mem::size_of;

use burn::{
    tensor::{DType, Shape, ops::FloatTensor},
};
use burn_cubecl::cubecl::{CubeCount, CubeDim, tensor_line_size_parallel};
use burn_cubecl::{
    BoolElement, CubeBackend, CubeElement, CubeRuntime, FloatElement, IntElement,
    kernel::{cast, into_contiguous},
    ops::numeric::{empty_device, zeros_client},
};

use crate::kernels::wkv7_common::{
    Wkv7StateBackwardOutput, Wkv7StatePassForwardOutput,
    kernel::{
        Wkv7BackwardInputsLaunch, Wkv7BackwardOutputsLaunch, Wkv7Config, Wkv7ForwardInputsLaunch,
        Wkv7ForwardOutputsLaunch, wkv7_backward_kernel, wkv7_forward_kernel,
    },
};

pub(crate) fn wkv7_forward_impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement>(
    weight_decay: FloatTensor<CubeBackend<R, F, I, BT>>,
    receptance: FloatTensor<CubeBackend<R, F, I, BT>>,
    key: FloatTensor<CubeBackend<R, F, I, BT>>,
    value: FloatTensor<CubeBackend<R, F, I, BT>>,
    removal: FloatTensor<CubeBackend<R, F, I, BT>>,
    replacement: FloatTensor<CubeBackend<R, F, I, BT>>,
    initial_state: Option<FloatTensor<CubeBackend<R, F, I, BT>>>,
    chunk_len: usize,
    use_initial_state: bool,
    return_final_state: bool,
) -> Wkv7StatePassForwardOutput<FloatTensor<CubeBackend<R, F, I, BT>>>
where
    f32: CubeElement,
    F: CubeElement,
{
    if size_of::<F>() != size_of::<f32>() {
        let result_f32 = wkv7_forward_impl_inner::<R, f32, I, BT>(
            cast::<R>(weight_decay, DType::F32),
            cast::<R>(receptance, DType::F32),
            cast::<R>(key, DType::F32),
            cast::<R>(value, DType::F32),
            cast::<R>(removal, DType::F32),
            cast::<R>(replacement, DType::F32),
            initial_state.map(|s| cast::<R>(s, DType::F32)),
            chunk_len,
            use_initial_state,
            return_final_state,
        );

        Wkv7StatePassForwardOutput {
            state: result_f32.state,
            removal_state: result_f32.removal_state,
            output: cast::<R>(result_f32.output, F::dtype()),
            final_state: result_f32.final_state,
        }
    } else {
        wkv7_forward_impl_inner::<R, F, I, BT>(
            weight_decay,
            receptance,
            key,
            value,
            removal,
            replacement,
            initial_state,
            chunk_len,
            use_initial_state,
            return_final_state,
        )
    }
}

fn wkv7_forward_impl_inner<R: CubeRuntime, FE: FloatElement, I: IntElement, BT: BoolElement>(
    weight_decay: FloatTensor<CubeBackend<R, FE, I, BT>>,
    receptance: FloatTensor<CubeBackend<R, FE, I, BT>>,
    key: FloatTensor<CubeBackend<R, FE, I, BT>>,
    value: FloatTensor<CubeBackend<R, FE, I, BT>>,
    removal: FloatTensor<CubeBackend<R, FE, I, BT>>,
    replacement: FloatTensor<CubeBackend<R, FE, I, BT>>,
    initial_state: Option<FloatTensor<CubeBackend<R, FE, I, BT>>>,
    chunk_len: usize,
    use_initial_state: bool,
    return_final_state: bool,
) -> Wkv7StatePassForwardOutput<FloatTensor<CubeBackend<R, FE, I, BT>>> {
    let weight_decay = into_contiguous(weight_decay);
    let receptance = into_contiguous(receptance);
    let key = into_contiguous(key);
    let value = into_contiguous(value);
    let removal = into_contiguous(removal);
    let replacement = into_contiguous(replacement);

    let client = weight_decay.client.clone();
    let device = weight_decay.device.clone();
    let shape = weight_decay.shape.clone();

    let batch_size = shape.dims[0];
    let seq_len = shape.dims[1];
    let num_heads = shape.dims[2];
    let dim = shape.dims[3];

    assert!(
        seq_len % chunk_len == 0,
        "chunk_len must divide sequence length"
    );

    let output = empty_device::<R, FE>(client.clone(), device.clone(), shape.clone());
    let removal_state = empty_device::<R, FE>(client.clone(), device.clone(), shape.clone());

    let num_chunks = seq_len / chunk_len;
    let state_shape = Shape::new([batch_size, num_heads, num_chunks, dim, dim]);
    let state = empty_device::<R, FE>(client.clone(), device.clone(), state_shape);

    let initial_state_shape = Shape::new([batch_size, num_heads, dim, dim]);
    let initial_state = if use_initial_state {
        into_contiguous(initial_state.expect("initial_state required"))
    } else {
        zeros_client::<R>(
            client.clone(),
            device.clone(),
            initial_state_shape.clone(),
            FE::dtype(),
        )
    };

    let final_state = if return_final_state {
        empty_device::<R, FE>(client.clone(), device.clone(), initial_state_shape)
    } else {
        empty_device::<R, FE>(client.clone(), device.clone(), Shape::new([1]))
    };

    let config = Wkv7Config {
        context_len: seq_len,
        num_heads,
        head_size: dim,
        chunk_length: chunk_len,
        state_line_size: 1,
        use_initial_state: u32::from(use_initial_state),
        return_final_state: u32::from(return_final_state),
        use_final_state_grad: 0,
        write_initial_state_grad: 0,
    };

    let cube_count = CubeCount::Static(num_heads as u32, batch_size as u32, 1);
    let cube_dim = CubeDim::new_1d(dim as u32);

    wkv7_forward_kernel::launch::<FE, R>(
        &client,
        cube_count,
        cube_dim,
        Wkv7ForwardInputsLaunch::new(
            weight_decay.as_tensor_arg(1),
            receptance.as_tensor_arg(1),
            key.as_tensor_arg(1),
            value.as_tensor_arg(1),
            removal.as_tensor_arg(1),
            replacement.as_tensor_arg(1),
            initial_state.as_tensor_arg(1),
        ),
        Wkv7ForwardOutputsLaunch::new(
            state.as_tensor_arg(1),
            removal_state.as_tensor_arg(1),
            output.as_tensor_arg(1),
            final_state.as_tensor_arg(1),
        ),
        config,
    )
    .unwrap();

    Wkv7StatePassForwardOutput {
        state,
        removal_state,
        output,
        final_state,
    }
}

pub(crate) fn wkv7_backward_impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement>(
    weight_decay: FloatTensor<CubeBackend<R, F, I, BT>>,
    receptance: FloatTensor<CubeBackend<R, F, I, BT>>,
    key: FloatTensor<CubeBackend<R, F, I, BT>>,
    value: FloatTensor<CubeBackend<R, F, I, BT>>,
    removal: FloatTensor<CubeBackend<R, F, I, BT>>,
    replacement: FloatTensor<CubeBackend<R, F, I, BT>>,
    state: FloatTensor<CubeBackend<R, F, I, BT>>,
    removal_state: FloatTensor<CubeBackend<R, F, I, BT>>,
    output_grad: FloatTensor<CubeBackend<R, F, I, BT>>,
    final_state_grad: Option<FloatTensor<CubeBackend<R, F, I, BT>>>,
    chunk_len: usize,
    use_final_state_grad: bool,
    write_initial_state_grad: bool,
) -> Wkv7StateBackwardOutput<FloatTensor<CubeBackend<R, F, I, BT>>>
where
    f32: CubeElement,
    F: CubeElement,
{
    if size_of::<F>() != size_of::<f32>() {
        let result_f32 = wkv7_backward_impl_inner::<R, f32, I, BT>(
            cast::<R>(weight_decay, DType::F32),
            cast::<R>(receptance, DType::F32),
            cast::<R>(key, DType::F32),
            cast::<R>(value, DType::F32),
            cast::<R>(removal, DType::F32),
            cast::<R>(replacement, DType::F32),
            cast::<R>(state, DType::F32),
            cast::<R>(removal_state, DType::F32),
            cast::<R>(output_grad, DType::F32),
            final_state_grad.map(|s| cast::<R>(s, DType::F32)),
            chunk_len,
            use_final_state_grad,
            write_initial_state_grad,
        );

        Wkv7StateBackwardOutput {
            weight_decay_grad: cast::<R>(result_f32.weight_decay_grad, F::dtype()),
            receptance_grad: cast::<R>(result_f32.receptance_grad, F::dtype()),
            key_grad: cast::<R>(result_f32.key_grad, F::dtype()),
            value_grad: cast::<R>(result_f32.value_grad, F::dtype()),
            removal_grad: cast::<R>(result_f32.removal_grad, F::dtype()),
            replacement_grad: cast::<R>(result_f32.replacement_grad, F::dtype()),
            initial_state_grad: result_f32.initial_state_grad,
        }
    } else {
        wkv7_backward_impl_inner::<R, F, I, BT>(
            weight_decay,
            receptance,
            key,
            value,
            removal,
            replacement,
            state,
            removal_state,
            output_grad,
            final_state_grad,
            chunk_len,
            use_final_state_grad,
            write_initial_state_grad,
        )
    }
}

fn wkv7_backward_impl_inner<R: CubeRuntime, FE: FloatElement, I: IntElement, BT: BoolElement>(
    weight_decay: FloatTensor<CubeBackend<R, FE, I, BT>>,
    receptance: FloatTensor<CubeBackend<R, FE, I, BT>>,
    key: FloatTensor<CubeBackend<R, FE, I, BT>>,
    value: FloatTensor<CubeBackend<R, FE, I, BT>>,
    removal: FloatTensor<CubeBackend<R, FE, I, BT>>,
    replacement: FloatTensor<CubeBackend<R, FE, I, BT>>,
    state: FloatTensor<CubeBackend<R, FE, I, BT>>,
    removal_state: FloatTensor<CubeBackend<R, FE, I, BT>>,
    output_grad: FloatTensor<CubeBackend<R, FE, I, BT>>,
    final_state_grad: Option<FloatTensor<CubeBackend<R, FE, I, BT>>>,
    chunk_len: usize,
    use_final_state_grad: bool,
    write_initial_state_grad: bool,
) -> Wkv7StateBackwardOutput<FloatTensor<CubeBackend<R, FE, I, BT>>> {
    let weight_decay = into_contiguous(weight_decay);
    let receptance = into_contiguous(receptance);
    let key = into_contiguous(key);
    let value = into_contiguous(value);
    let removal = into_contiguous(removal);
    let replacement = into_contiguous(replacement);
    let state = into_contiguous(state);
    let removal_state = into_contiguous(removal_state);
    let output_grad = into_contiguous(output_grad);

    let client = weight_decay.client.clone();
    let device = weight_decay.device.clone();
    let shape = weight_decay.shape.clone();

    let batch_size = shape.dims[0];
    let seq_len = shape.dims[1];
    let num_heads = shape.dims[2];
    let dim = shape.dims[3];

    assert!(
        seq_len % chunk_len == 0,
        "chunk_len must divide sequence length"
    );

    let weight_decay_grad = empty_device::<R, FE>(client.clone(), device.clone(), shape.clone());
    let receptance_grad = empty_device::<R, FE>(client.clone(), device.clone(), shape.clone());
    let key_grad = empty_device::<R, FE>(client.clone(), device.clone(), shape.clone());
    let value_grad = empty_device::<R, FE>(client.clone(), device.clone(), shape.clone());
    let removal_grad = empty_device::<R, FE>(client.clone(), device.clone(), shape.clone());
    let replacement_grad = empty_device::<R, FE>(client.clone(), device.clone(), shape.clone());

    let initial_state_shape = Shape::new([batch_size, num_heads, dim, dim]);
    let initial_state_grad = if write_initial_state_grad {
        empty_device::<R, FE>(client.clone(), device.clone(), initial_state_shape.clone())
    } else {
        empty_device::<R, FE>(client.clone(), device.clone(), Shape::new([1]))
    };

    let final_state_grad = if use_final_state_grad {
        into_contiguous(final_state_grad.expect("final_state_grad required"))
    } else {
        empty_device::<R, FE>(client.clone(), device.clone(), Shape::new([1]))
    };

    let state_line_size = tensor_line_size_parallel(
        state.client.io_optimized_line_sizes(&state.dtype.into()),
        &state.shape,
        &state.strides,
        state.shape.num_dims() - 1,
    );

    let config = Wkv7Config {
        context_len: seq_len,
        num_heads,
        head_size: dim,
        chunk_length: chunk_len,
        state_line_size,
        use_initial_state: 0,
        return_final_state: 0,
        use_final_state_grad: u32::from(use_final_state_grad),
        write_initial_state_grad: u32::from(write_initial_state_grad),
    };

    let cube_count = CubeCount::Static(num_heads as u32, batch_size as u32, 1);
    let cube_dim = CubeDim::new_1d(dim as u32);

    wkv7_backward_kernel::launch::<FE, R>(
        &client,
        cube_count,
        cube_dim,
        Wkv7BackwardInputsLaunch::new(
            weight_decay.as_tensor_arg(1),
            receptance.as_tensor_arg(1),
            key.as_tensor_arg(1),
            value.as_tensor_arg(1),
            removal.as_tensor_arg(1),
            replacement.as_tensor_arg(1),
            state.as_tensor_arg(state_line_size),
            removal_state.as_tensor_arg(1),
            output_grad.as_tensor_arg(1),
            final_state_grad.as_tensor_arg(1),
        ),
        Wkv7BackwardOutputsLaunch::new(
            weight_decay_grad.as_tensor_arg(1),
            receptance_grad.as_tensor_arg(1),
            key_grad.as_tensor_arg(1),
            value_grad.as_tensor_arg(1),
            removal_grad.as_tensor_arg(1),
            replacement_grad.as_tensor_arg(1),
            initial_state_grad.as_tensor_arg(1),
        ),
        config,
    )
    .expect("wkv7_backward_kernel should never fail");

    Wkv7StateBackwardOutput {
        weight_decay_grad,
        receptance_grad,
        key_grad,
        value_grad,
        removal_grad,
        replacement_grad,
        initial_state_grad,
    }
}
