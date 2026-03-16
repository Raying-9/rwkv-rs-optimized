use burn::tensor::ops::FloatTensor;
use burn_cubecl::{BoolElement, CubeBackend, CubeElement, CubeRuntime, FloatElement, IntElement};

use crate::kernels::wkv7_infer::{
    Wkv7InferBackend, Wkv7InferForwardOutput, host::wkv7_infer_forward_impl,
};

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> Wkv7InferBackend
    for CubeBackend<R, F, I, BT>
where
    F: CubeElement,
{
    fn wkv7_infer_forward(
        weight_decay: FloatTensor<Self>,
        receptance: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        removal: FloatTensor<Self>,
        replacement: FloatTensor<Self>,
        initial_state: FloatTensor<Self>,
        context_mask: FloatTensor<Self>,
    ) -> Wkv7InferForwardOutput<FloatTensor<Self>> {
        wkv7_infer_forward_impl::<R, F, I, BT>(
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
}

#[cfg(feature = "fusion")]
mod fusion_impl {
    use burn::tensor::{Element, Shape};
    use burn_fusion::{
        Fusion, FusionBackend, FusionRuntime,
        stream::{Operation, OperationStreams},
    };
    use burn_ir::{CustomOpIr, HandleContainer, OperationIr, TensorIr};

    use super::*;
    use crate::kernels::wkv7_infer::{Wkv7InferBackend, Wkv7InferForwardOutput};

    impl<B: FusionBackend + Wkv7InferBackend> Wkv7InferBackend for Fusion<B> {
        fn wkv7_infer_forward(
            weight_decay: FloatTensor<Self>,
            receptance: FloatTensor<Self>,
            key: FloatTensor<Self>,
            value: FloatTensor<Self>,
            removal: FloatTensor<Self>,
            replacement: FloatTensor<Self>,
            initial_state: FloatTensor<Self>,
            context_mask: FloatTensor<Self>,
        ) -> Wkv7InferForwardOutput<FloatTensor<Self>> {
            let client = weight_decay.client.clone();
            let batch_size = weight_decay.shape[0];
            let context_length = weight_decay.shape[1];
            let num_heads = weight_decay.shape[2];
            let dim = weight_decay.shape[3];

            #[derive(Clone, Debug)]
            struct Wkv7InferForwardOp<B1> {
                desc: CustomOpIr,
                _b: core::marker::PhantomData<B1>,
            }

            impl<B1: FusionBackend + Wkv7InferBackend> Operation<B1::FusionRuntime> for Wkv7InferForwardOp<B1> {
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
                            context_mask,
                        ],
                        [output_out, final_state_out],
                    ) = self.desc.as_fixed();

                    let weight_decay_tensor = handles.get_float_tensor::<B1>(weight_decay);
                    let receptance_tensor = handles.get_float_tensor::<B1>(receptance);
                    let key_tensor = handles.get_float_tensor::<B1>(key);
                    let value_tensor = handles.get_float_tensor::<B1>(value);
                    let removal_tensor = handles.get_float_tensor::<B1>(removal);
                    let replacement_tensor = handles.get_float_tensor::<B1>(replacement);
                    let initial_state_tensor = handles.get_float_tensor::<B1>(initial_state);
                    let context_mask_tensor = handles.get_float_tensor::<B1>(context_mask);

                    let output = B1::wkv7_infer_forward(
                        weight_decay_tensor,
                        receptance_tensor,
                        key_tensor,
                        value_tensor,
                        removal_tensor,
                        replacement_tensor,
                        initial_state_tensor,
                        context_mask_tensor,
                    );

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
            streams.tensor(&context_mask);

            let output_desc = [
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, context_length, num_heads, dim]),
                    B::FloatElem::dtype(),
                ),
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, num_heads, dim, dim]),
                    B::FloatElem::dtype(),
                ),
            ];

            let desc = CustomOpIr::new(
                "wkv7_infer_forward",
                &[
                    weight_decay.into_ir(),
                    receptance.into_ir(),
                    key.into_ir(),
                    value.into_ir(),
                    removal.into_ir(),
                    replacement.into_ir(),
                    initial_state.into_ir(),
                    context_mask.into_ir(),
                ],
                &output_desc,
            );

            let op = Wkv7InferForwardOp::<B> {
                desc,
                _b: core::marker::PhantomData,
            };

            let mut outputs = client.register(streams, OperationIr::Custom(op.desc.clone()), op);

            let final_state = outputs.pop().expect("missing final_state");
            let output = outputs.pop().expect("missing output");

            Wkv7InferForwardOutput {
                output,
                final_state,
            }
        }
    }
}
