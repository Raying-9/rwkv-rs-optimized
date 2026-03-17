use burn_cubecl::{BoolElement, CubeBackend, CubeElement, CubeRuntime, FloatElement, IntElement};

use crate::kernels::wkv7_common::{
    Wkv7StateBackwardOutput, Wkv7StatePassForwardOutput,
    host::{wkv7_backward_impl, wkv7_forward_impl},
};
use crate::kernels::wkv7_statepass::Wkv7StatePassBackend;

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> Wkv7StatePassBackend
    for CubeBackend<R, F, I, BT>
where
    f32: CubeElement,
    F: CubeElement,
{
    fn wkv7_statepass_forward(
        weight_decay: burn::tensor::ops::FloatTensor<Self>,
        receptance: burn::tensor::ops::FloatTensor<Self>,
        key: burn::tensor::ops::FloatTensor<Self>,
        value: burn::tensor::ops::FloatTensor<Self>,
        removal: burn::tensor::ops::FloatTensor<Self>,
        replacement: burn::tensor::ops::FloatTensor<Self>,
        initial_state: burn::tensor::ops::FloatTensor<Self>,
        chunk_len: usize,
    ) -> Wkv7StatePassForwardOutput<burn::tensor::ops::FloatTensor<Self>> {
        wkv7_forward_impl::<R, F, I, BT>(
            weight_decay,
            receptance,
            key,
            value,
            removal,
            replacement,
            Some(initial_state),
            chunk_len,
            true,
            true,
        )
    }

    fn wkv7_statepass_backward(
        weight_decay: burn::tensor::ops::FloatTensor<Self>,
        receptance: burn::tensor::ops::FloatTensor<Self>,
        key: burn::tensor::ops::FloatTensor<Self>,
        value: burn::tensor::ops::FloatTensor<Self>,
        removal: burn::tensor::ops::FloatTensor<Self>,
        replacement: burn::tensor::ops::FloatTensor<Self>,
        state: burn::tensor::ops::FloatTensor<Self>,
        removal_state: burn::tensor::ops::FloatTensor<Self>,
        output_grad: burn::tensor::ops::FloatTensor<Self>,
        final_state_grad: burn::tensor::ops::FloatTensor<Self>,
        chunk_len: usize,
    ) -> Wkv7StateBackwardOutput<burn::tensor::ops::FloatTensor<Self>> {
        wkv7_backward_impl::<R, F, I, BT>(
            weight_decay,
            receptance,
            key,
            value,
            removal,
            replacement,
            state,
            removal_state,
            output_grad,
            Some(final_state_grad),
            chunk_len,
            true,
            true,
        )
    }
}

#[cfg(feature = "fusion")]
mod fusion_impl {
    use burn::tensor::{DType, Element, Shape, ops::FloatTensor};
    use burn_fusion::{
        Fusion, FusionBackend, FusionRuntime,
        stream::{Operation, OperationStreams},
    };
    use burn_ir::{CustomOpIr, HandleContainer, OperationIr, TensorIr};

    use crate::kernels::wkv7_common::{Wkv7StateBackwardOutput, Wkv7StatePassForwardOutput};
    use crate::kernels::wkv7_statepass::Wkv7StatePassBackend;

    impl<B: FusionBackend + Wkv7StatePassBackend> Wkv7StatePassBackend for Fusion<B> {
        fn wkv7_statepass_forward(
            weight_decay: FloatTensor<Self>,
            receptance: FloatTensor<Self>,
            key: FloatTensor<Self>,
            value: FloatTensor<Self>,
            removal: FloatTensor<Self>,
            replacement: FloatTensor<Self>,
            initial_state: FloatTensor<Self>,
            chunk_len: usize,
        ) -> Wkv7StatePassForwardOutput<FloatTensor<Self>> {
            let client = weight_decay.client.clone();
            let batch_size = weight_decay.shape[0];
            let seq_len = weight_decay.shape[1];
            let num_heads = weight_decay.shape[2];
            let dim = weight_decay.shape[3];
            let num_chunks = seq_len.div_ceil(chunk_len);

            #[derive(Clone, Debug)]
            struct Wkv7StatePassForwardOp<B1> {
                desc: CustomOpIr,
                chunk_len: usize,
                _b: core::marker::PhantomData<B1>,
            }

            impl<B1: FusionBackend + Wkv7StatePassBackend> Operation<B1::FusionRuntime>
                for Wkv7StatePassForwardOp<B1>
            {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let (
                        [
                            weight_decay,
                            receptance,
                            key,
                            value,
                            removal,
                            replacement,
                            initial_state,
                        ],
                        [state_out, removal_state_out, output_out, final_state_out],
                    ) = self.desc.as_fixed();

                    let weight_decay_tensor = handles.get_float_tensor::<B1>(weight_decay);
                    let receptance_tensor = handles.get_float_tensor::<B1>(receptance);
                    let key_tensor = handles.get_float_tensor::<B1>(key);
                    let value_tensor = handles.get_float_tensor::<B1>(value);
                    let removal_tensor = handles.get_float_tensor::<B1>(removal);
                    let replacement_tensor = handles.get_float_tensor::<B1>(replacement);
                    let initial_state_tensor = handles.get_float_tensor::<B1>(initial_state);

                    let output = B1::wkv7_statepass_forward(
                        weight_decay_tensor,
                        receptance_tensor,
                        key_tensor,
                        value_tensor,
                        removal_tensor,
                        replacement_tensor,
                        initial_state_tensor,
                        self.chunk_len,
                    );

                    handles.register_float_tensor::<B1>(&state_out.id, output.state);
                    handles
                        .register_float_tensor::<B1>(&removal_state_out.id, output.removal_state);
                    handles.register_float_tensor::<B1>(&output_out.id, output.output);
                    handles.register_float_tensor::<B1>(&final_state_out.id, output.final_state);
                }
            }

            let mut streams = OperationStreams::default();
            streams.tensor(&weight_decay);
            streams.tensor(&receptance);
            streams.tensor(&key);
            streams.tensor(&value);
            streams.tensor(&removal);
            streams.tensor(&replacement);
            streams.tensor(&initial_state);

            let output_desc = [
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, num_heads, num_chunks, dim, dim]),
                    DType::F32,
                ),
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, seq_len, num_heads, dim]),
                    DType::F32,
                ),
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, seq_len, num_heads, dim]),
                    B::FloatElem::dtype(),
                ),
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, num_heads, dim, dim]),
                    DType::F32,
                ),
            ];

            let desc = CustomOpIr::new(
                "wkv7_statepass_forward",
                &[
                    weight_decay.into_ir(),
                    receptance.into_ir(),
                    key.into_ir(),
                    value.into_ir(),
                    removal.into_ir(),
                    replacement.into_ir(),
                    initial_state.into_ir(),
                ],
                &output_desc,
            );

            let op = Wkv7StatePassForwardOp::<B> {
                desc,
                chunk_len,
                _b: core::marker::PhantomData,
            };

            let mut outputs = client.register(streams, OperationIr::Custom(op.desc.clone()), op);

            let final_state = outputs.pop().expect("missing final_state");
            let output = outputs.pop().expect("missing output");
            let removal_state = outputs.pop().expect("missing removal_state");
            let state = outputs.pop().expect("missing state");

            Wkv7StatePassForwardOutput {
                state,
                removal_state,
                output,
                final_state,
            }
        }

        fn wkv7_statepass_backward(
            weight_decay: FloatTensor<Self>,
            receptance: FloatTensor<Self>,
            key: FloatTensor<Self>,
            value: FloatTensor<Self>,
            removal: FloatTensor<Self>,
            replacement: FloatTensor<Self>,
            state: FloatTensor<Self>,
            removal_state: FloatTensor<Self>,
            output_grad: FloatTensor<Self>,
            final_state_grad: FloatTensor<Self>,
            chunk_len: usize,
        ) -> Wkv7StateBackwardOutput<FloatTensor<Self>> {
            let client = weight_decay.client.clone();
            let batch_size = weight_decay.shape[0];
            let seq_len = weight_decay.shape[1];
            let num_heads = weight_decay.shape[2];
            let dim = weight_decay.shape[3];

            #[derive(Clone, Debug)]
            struct Wkv7StatePassBackwardOp<B1> {
                desc: CustomOpIr,
                chunk_len: usize,
                _b: core::marker::PhantomData<B1>,
            }

            impl<B1: FusionBackend + Wkv7StatePassBackend> Operation<B1::FusionRuntime>
                for Wkv7StatePassBackwardOp<B1>
            {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let (
                        [
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
                        ],
                        [
                            weight_decay_grad,
                            receptance_grad,
                            key_grad,
                            value_grad,
                            removal_grad,
                            replacement_grad,
                            initial_state_grad,
                        ],
                    ) = self.desc.as_fixed();

                    let weight_decay_tensor = handles.get_float_tensor::<B1>(weight_decay);
                    let receptance_tensor = handles.get_float_tensor::<B1>(receptance);
                    let key_tensor = handles.get_float_tensor::<B1>(key);
                    let value_tensor = handles.get_float_tensor::<B1>(value);
                    let removal_tensor = handles.get_float_tensor::<B1>(removal);
                    let replacement_tensor = handles.get_float_tensor::<B1>(replacement);
                    let state_tensor = handles.get_float_tensor::<B1>(state);
                    let removal_state_tensor = handles.get_float_tensor::<B1>(removal_state);
                    let output_grad_tensor = handles.get_float_tensor::<B1>(output_grad);
                    let final_state_grad_tensor = handles.get_float_tensor::<B1>(final_state_grad);

                    let grads = B1::wkv7_statepass_backward(
                        weight_decay_tensor,
                        receptance_tensor,
                        key_tensor,
                        value_tensor,
                        removal_tensor,
                        replacement_tensor,
                        state_tensor,
                        removal_state_tensor,
                        output_grad_tensor,
                        final_state_grad_tensor,
                        self.chunk_len,
                    );

                    handles.register_float_tensor::<B1>(
                        &weight_decay_grad.id,
                        grads.weight_decay_grad,
                    );
                    handles.register_float_tensor::<B1>(&receptance_grad.id, grads.receptance_grad);
                    handles.register_float_tensor::<B1>(&key_grad.id, grads.key_grad);
                    handles.register_float_tensor::<B1>(&value_grad.id, grads.value_grad);
                    handles.register_float_tensor::<B1>(&removal_grad.id, grads.removal_grad);
                    handles
                        .register_float_tensor::<B1>(&replacement_grad.id, grads.replacement_grad);
                    handles.register_float_tensor::<B1>(
                        &initial_state_grad.id,
                        grads.initial_state_grad,
                    );
                }
            }

            let mut streams = OperationStreams::default();
            streams.tensor(&weight_decay);
            streams.tensor(&receptance);
            streams.tensor(&key);
            streams.tensor(&value);
            streams.tensor(&removal);
            streams.tensor(&replacement);
            streams.tensor(&state);
            streams.tensor(&removal_state);
            streams.tensor(&output_grad);
            streams.tensor(&final_state_grad);

            let output_desc = [
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, seq_len, num_heads, dim]),
                    B::FloatElem::dtype(),
                ),
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, seq_len, num_heads, dim]),
                    B::FloatElem::dtype(),
                ),
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, seq_len, num_heads, dim]),
                    B::FloatElem::dtype(),
                ),
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, seq_len, num_heads, dim]),
                    B::FloatElem::dtype(),
                ),
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, seq_len, num_heads, dim]),
                    B::FloatElem::dtype(),
                ),
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, seq_len, num_heads, dim]),
                    B::FloatElem::dtype(),
                ),
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, num_heads, dim, dim]),
                    DType::F32,
                ),
            ];

            let desc = CustomOpIr::new(
                "wkv7_statepass_backward",
                &[
                    weight_decay.into_ir(),
                    receptance.into_ir(),
                    key.into_ir(),
                    value.into_ir(),
                    removal.into_ir(),
                    replacement.into_ir(),
                    state.into_ir(),
                    removal_state.into_ir(),
                    output_grad.into_ir(),
                    final_state_grad.into_ir(),
                ],
                &output_desc,
            );

            let op = Wkv7StatePassBackwardOp::<B> {
                desc,
                chunk_len,
                _b: core::marker::PhantomData,
            };

            let mut outputs = client.register(streams, OperationIr::Custom(op.desc.clone()), op);

            let initial_state_grad = outputs.pop().expect("missing initial_state_grad");
            let replacement_grad = outputs.pop().expect("missing replacement_grad");
            let removal_grad = outputs.pop().expect("missing removal_grad");
            let value_grad = outputs.pop().expect("missing value_grad");
            let key_grad = outputs.pop().expect("missing key_grad");
            let receptance_grad = outputs.pop().expect("missing receptance_grad");
            let weight_decay_grad = outputs.pop().expect("missing weight_decay_grad");

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
    }
}
