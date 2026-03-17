use burn::{
    config::Config,
    module::{Ignored, Module, Param},
    nn::{Linear, LinearConfig},
    prelude::*,
    tensor::activation::{sigmoid, softplus},
};
use std::{any::Any, sync::Arc};

use crate::kernels::wkv7_common::Wkv7ForwardInput;
use crate::{
    functions::{
        init_weights::{
            calculate_token_shift_with_offset, constant_init, get_token_shift_diff_scale,
            uniform_init,
        },
        lerp::lerp,
        normalize::normalize,
        token_shift::token_shift_diff,
    },
    layers::lora::{ActivationFn, LoRA, LoRAConfig, LoRAType},
};

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

#[derive(Config, Debug)]
pub struct WeightPrepareConfig {
    num_cells: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,

    weight_decay_lora_rank: usize,
    learning_rate_lora_rank: usize,
    value_residual_lora_rank: usize,
}

#[derive(Clone, Debug)]
struct PackedWeightPrepare<B: Backend> {
    mix_params: Tensor<B, 4>,
    projection_weights: Tensor<B, 4>,
    lora_w_a: Tensor<B, 4>,
    lora_w_b: Tensor<B, 4>,
    lora_bias: Tensor<B, 4>,
}

#[derive(Clone)]
struct PackedWeightPrepareAny(Arc<dyn Any + Send + Sync>);

impl core::fmt::Debug for PackedWeightPrepareAny {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("PackedWeightPrepareAny(..)")
    }
}

impl PackedWeightPrepareAny {
    fn new<B: Backend>(packed: PackedWeightPrepare<B>) -> Self
    where
        PackedWeightPrepare<B>: 'static + Send + Sync,
    {
        Self(Arc::new(packed))
    }

    fn downcast_ref<B: Backend>(&self) -> Option<&PackedWeightPrepare<B>>
    where
        PackedWeightPrepare<B>: 'static,
    {
        self.0.downcast_ref::<PackedWeightPrepare<B>>()
    }
}

impl WeightPrepareConfig {
    pub fn init<B: Backend>(&self, cell_id: usize, device: &B::Device) -> WeightPrepare<B> {
        let empty_param = Param::from_tensor(Tensor::empty([1, 1, self.embedded_dim], device));

        let projection = LinearConfig::new(self.embedded_dim, self.embedded_dim)
            .with_bias(false)
            .init(device);

        WeightPrepare {
            param_receptance: empty_param.clone(),
            param_weight_decay: empty_param.clone(),
            param_key: empty_param.clone(),
            param_value: empty_param.clone(),
            param_learning_rate: empty_param.clone(),

            projection_receptance: projection.clone(),
            projection_key: projection.clone(),
            projection_value: projection.clone(),

            param_weight_decay_lora: LoRAConfig::new(
                self.num_cells,
                self.embedded_dim,
                self.weight_decay_lora_rank,
                self.head_size,
                true,
                ActivationFn::Tanh,
            )
            .init(device),
            param_learning_rate_lora: LoRAConfig::new(
                self.num_cells,
                self.embedded_dim,
                self.learning_rate_lora_rank,
                self.head_size,
                true,
                ActivationFn::NoOP,
            )
            .init(device),
            param_value_residual_lora: if cell_id > 0 {
                Some(
                    LoRAConfig::new(
                        self.num_cells,
                        self.embedded_dim,
                        self.value_residual_lora_rank,
                        self.head_size,
                        true,
                        ActivationFn::NoOP,
                    )
                    .init(device),
                )
            } else {
                None
            },

            param_key_removal: empty_param.clone(),
            param_key_replacement: empty_param.clone(),

            num_cells: self.num_cells,
            embedded_dim: self.embedded_dim,
            num_heads: self.num_heads,
            head_size: self.head_size,
            packed_infer: Ignored(None),
            cell_id,
        }
    }
}

#[derive(Module, Debug)]
pub struct WeightPrepare<B: Backend> {
    param_receptance: Param<Tensor<B, 3, Float>>,
    param_weight_decay: Param<Tensor<B, 3, Float>>,
    param_key: Param<Tensor<B, 3, Float>>,
    param_value: Param<Tensor<B, 3, Float>>,
    param_learning_rate: Param<Tensor<B, 3, Float>>,

    projection_receptance: Linear<B>,
    projection_key: Linear<B>,
    projection_value: Linear<B>,

    param_weight_decay_lora: LoRA<B>,
    param_learning_rate_lora: LoRA<B>,
    param_value_residual_lora: Option<LoRA<B>>,

    param_key_removal: Param<Tensor<B, 3>>,
    param_key_replacement: Param<Tensor<B, 3>>,

    num_cells: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,

    packed_infer: Ignored<Option<PackedWeightPrepareAny>>,

    #[module(skip)]
    cell_id: usize,
}

impl<B: Backend> WeightPrepare<B> {
    pub fn init_weights(&mut self, device: &B::Device) {
        self.param_receptance = Param::from_tensor(get_token_shift_diff_scale(
            self.num_cells,
            self.embedded_dim,
            self.cell_id,
            0.2,
            device,
        ));

        self.param_weight_decay = Param::from_tensor(get_token_shift_diff_scale(
            self.num_cells,
            self.embedded_dim,
            self.cell_id,
            0.9,
            device,
        ));

        self.param_key = Param::from_tensor(calculate_token_shift_with_offset(
            self.num_cells,
            self.embedded_dim,
            self.cell_id,
            0.7,
            device,
        ));

        self.param_value = Param::from_tensor(calculate_token_shift_with_offset(
            self.num_cells,
            self.embedded_dim,
            self.cell_id,
            0.7,
            device,
        ));

        self.param_learning_rate = Param::from_tensor(get_token_shift_diff_scale(
            self.num_cells,
            self.embedded_dim,
            self.cell_id,
            0.9,
            device,
        ));

        let embedded_dim = self.embedded_dim as f64;

        let receptance_bound = 0.5 / embedded_dim.sqrt();

        let key_bound = 0.05 / embedded_dim.sqrt();

        let value_bound = 0.5 / embedded_dim.sqrt();

        uniform_init(
            &mut self.projection_receptance.weight,
            -receptance_bound,
            receptance_bound,
        );

        uniform_init(&mut self.projection_key.weight, -key_bound, key_bound);
        uniform_init(&mut self.projection_value.weight, -value_bound, value_bound);

        self.param_weight_decay_lora
            .init_weight(self.cell_id, LoRAType::WeightDecay, device);

        self.param_learning_rate_lora
            .init_weight(self.cell_id, LoRAType::LearningRate, device);

        if let Some(ref mut value_residual_lora) = self.param_value_residual_lora {
            value_residual_lora.init_weight(self.cell_id, LoRAType::ValueResidual, device);
        }

        constant_init(&mut self.param_key_removal, 0.71);
        constant_init(&mut self.param_key_replacement, 1.02);
        self.packed_infer = Ignored(None);
    }

    pub fn prepare_inference_cache(&mut self) {
        let PackedGroupedLoRA {
            w_a: lora_w_a,
            w_b: lora_w_b,
            bias: lora_bias,
        } = pack_grouped_lora([
            &self.param_weight_decay_lora,
            &self.param_learning_rate_lora,
        ]);

        self.packed_infer = Ignored(Some(PackedWeightPrepareAny::new(PackedWeightPrepare {
            mix_params: build_mix_params(self),
            projection_weights: pack_grouped_linear_weights([
                &self.projection_receptance,
                &self.projection_key,
                &self.projection_value,
            ]),
            lora_w_a,
            lora_w_b,
            lora_bias,
        })));
    }

    #[cfg_attr(
        feature = "trace",
        tracing::instrument(name = "rwkv.infer.model.weight_prepare", skip_all)
    )]
    pub fn forward(
        &self,
        embedded_context: Tensor<B, 3>,
        value_from_first_cell: Tensor<B, 3>,
        embedded_token_shift: Option<Tensor<B, 2>>,
        context_mask: Option<Tensor<B, 2>>,
    ) -> WeightPrepareOutput<B> {
        let [_, _context_length, _] = embedded_context.dims();
        let token_shifted_diff = {
            #[cfg(feature = "trace")]
            let _token_shift_scope = tracing::trace_span!(
                "rwkv.infer.model.weight_prepare.token_shift_diff",
                cell_id = self.cell_id,
                context_length = _context_length
            )
            .entered();
            token_shift_diff(
                embedded_context.clone(),
                embedded_token_shift,
                context_mask.clone(),
            )
        };

        self.forward_with_token_shifted_diff(
            embedded_context,
            value_from_first_cell,
            token_shifted_diff,
        )
    }

    fn forward_with_token_shifted_diff(
        &self,
        embedded_context: Tensor<B, 3>,
        value_from_first_cell: Tensor<B, 3>,
        token_shifted_diff: Tensor<B, 3>,
    ) -> WeightPrepareOutput<B> {
        // Paper equations implemented:
        // 355: x^{square}_t = lerp(x_t, x_{t-1}, mu_{square})  -- Time shifting
        // 356: a_t = sigmoid(loramlp_a(Identity, x^a_t, bias=True))  -- In-context
        // learning rate 357: k_t = x^k_t W_k  -- Key precursor
        // 358: kappa_t = k_t ⊙ xi  -- Removal key (before normalization)
        // 359: tilde_k_t = k_t ⊙ lerp(1, a_t, alpha)  -- Replacement key
        // 360: nu_t = sigmoid(loramlp_nu(Identity, x^v_t, bias=True))  -- Value
        // residual gate 361-366: v_t computation with residual mixing
        // 367: d_t = loramlp_d(tanh, x^d_t, bias=True)  -- Decay precursor
        // 368: w_t = exp(-e^{-0.5} sigmoid(d_t))  -- Decay
        // 369: r_t = x^r_t W_r  -- Receptance
        // 370: g_t = loramlp_g(sigmoid, x^g_t, bias=False)  -- RWKV gate
        let [batch_size, context_length, embedded_dim] = embedded_context.dims();

        let (num_heads, head_size) = (self.num_heads, self.head_size);

        let mix_params = {
            trace_lite_scope!("rwkv.infer.model.weight_prepare.scale_diff");
            self.packed_infer
                .as_ref()
                .and_then(PackedWeightPrepareAny::downcast_ref::<B>)
                .map(|packed| packed.mix_params.clone())
                .unwrap_or_else(|| build_mix_params(self))
        };

        let mixed_inputs = {
            trace_lite_scope!("rwkv.infer.model.weight_prepare.mix_inputs");
            embedded_context.unsqueeze_dim(0)
                + token_shifted_diff.clone().unsqueeze_dim(0) * mix_params
        };

        let projection_outputs = {
            trace_lite_scope!("rwkv.infer.model.weight_prepare.grouped_projection");
            let projection_inputs = extract_fanout_group(
                mixed_inputs.clone(),
                0,
                3,
                batch_size,
                context_length,
                embedded_dim,
            );
            match self
                .packed_infer
                .as_ref()
                .and_then(PackedWeightPrepareAny::downcast_ref::<B>)
            {
                Some(packed) => grouped_linear_forward_packed(
                    projection_inputs,
                    packed.projection_weights.clone(),
                ),
                None => grouped_linear_forward(
                    projection_inputs,
                    [
                        &self.projection_receptance,
                        &self.projection_key,
                        &self.projection_value,
                    ],
                ),
            }
        };

        let (receptance, key_precursor, value_precursor) = {
            trace_lite_scope!("rwkv.infer.model.weight_prepare.split_projection_outputs");
            (
                extract_fanout_input(
                    projection_outputs.clone(),
                    0,
                    batch_size,
                    context_length,
                    embedded_dim,
                ),
                extract_fanout_input(
                    projection_outputs.clone(),
                    1,
                    batch_size,
                    context_length,
                    embedded_dim,
                ),
                extract_fanout_input(
                    projection_outputs,
                    2,
                    batch_size,
                    context_length,
                    embedded_dim,
                ),
            )
        };

        let lora_outputs = {
            trace_lite_scope!("rwkv.infer.model.weight_prepare.grouped_lora");
            let lora_inputs = extract_fanout_group(
                mixed_inputs.clone(),
                3,
                2,
                batch_size,
                context_length,
                embedded_dim,
            );
            match self
                .packed_infer
                .as_ref()
                .and_then(PackedWeightPrepareAny::downcast_ref::<B>)
            {
                Some(packed) => grouped_lora_forward_packed(
                    lora_inputs,
                    packed.lora_w_a.clone(),
                    packed.lora_w_b.clone(),
                    packed.lora_bias.clone(),
                    [ActivationFn::Tanh, ActivationFn::NoOP],
                ),
                None => grouped_lora_forward(
                    lora_inputs,
                    [
                        &self.param_weight_decay_lora,
                        &self.param_learning_rate_lora,
                    ],
                    [ActivationFn::Tanh, ActivationFn::NoOP],
                ),
            }
        };

        let (weight_decay_lora_result, learning_rate_pre_sigmoid) = {
            trace_lite_scope!("rwkv.infer.model.weight_prepare.split_lora_outputs");
            (
                extract_fanout_input(
                    lora_outputs.clone(),
                    0,
                    batch_size,
                    context_length,
                    embedded_dim,
                ),
                extract_fanout_input(lora_outputs, 1, batch_size, context_length, embedded_dim),
            )
        };

        let value_input = {
            trace_lite_scope!("rwkv.infer.model.weight_prepare.value_input");
            extract_fanout_input(mixed_inputs, 2, batch_size, context_length, embedded_dim)
        };

        let value_from_first_cell = if self.cell_id == 0 {
            value_precursor.clone()
        } else {
            value_from_first_cell
        };

        let learning_rate = {
            trace_lite_scope!("rwkv.infer.model.weight_prepare.learning_rate_lora");
            sigmoid(learning_rate_pre_sigmoid)
        };

        let alpha_modulated = {
            trace_lite_scope!("rwkv.infer.model.weight_prepare.alpha_modulated");
            self.param_key_replacement.val() * (learning_rate.clone() - 1.0) + 1.0
        };

        let replacement_key = {
            trace_lite_scope!("rwkv.infer.model.weight_prepare.replacement_key");
            key_precursor.clone() * alpha_modulated
        };

        let value = if self.cell_id != 0 {
            let nu_t = {
                trace_lite_scope!("rwkv.infer.model.weight_prepare.value_residual_lora");
                sigmoid(
                    self.param_value_residual_lora
                        .as_ref()
                        .unwrap()
                        .forward(value_input),
                )
            };

            {
                trace_lite_scope!("rwkv.infer.model.weight_prepare.value_residual_mix");
                lerp(value_precursor, value_from_first_cell.clone(), nu_t)
            }
        } else {
            value_precursor
        };

        let weight_decay = {
            trace_lite_scope!("rwkv.infer.model.weight_prepare.weight_decay_lora");
            -softplus(-weight_decay_lora_result, 1.0) - 0.5
        };

        let removal_key = {
            trace_lite_scope!("rwkv.infer.model.weight_prepare.removal_key");
            key_precursor * self.param_key_removal.val()
        };

        let removal_key_reshaped = {
            trace_lite_scope!("rwkv.infer.model.weight_prepare.removal_key_reshape");
            removal_key.reshape([batch_size, context_length, num_heads, head_size])
        };

        let removal_key_normalized = {
            trace_lite_scope!("rwkv.infer.model.weight_prepare.normalize_removal_key");
            -normalize(removal_key_reshaped, 2.0, -1, 1e-12).reshape([
                batch_size,
                context_length,
                embedded_dim,
            ])
        };

        let replacement = {
            trace_lite_scope!("rwkv.infer.model.weight_prepare.replacement");
            -removal_key_normalized.clone() * learning_rate
        };

        WeightPrepareOutput {
            token_shifted_diff,
            value_from_first_cell,
            receptance,
            weight_decay,
            replacement_key,
            value,
            removal_key_normalized,
            replacement,
        }
    }
}

#[derive(Debug)]
pub struct WeightPrepareOutput<B: Backend> {
    pub token_shifted_diff: Tensor<B, 3>,
    pub value_from_first_cell: Tensor<B, 3>,
    pub receptance: Tensor<B, 3>,
    pub weight_decay: Tensor<B, 3>,
    pub replacement_key: Tensor<B, 3>,
    pub value: Tensor<B, 3>,
    pub removal_key_normalized: Tensor<B, 3>,
    pub replacement: Tensor<B, 3>,
}

fn extract_fanout_input<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    branch_index: usize,
    batch_size: usize,
    context_length: usize,
    embedded_dim: usize,
) -> Tensor<B, 3> {
    tensor
        .slice([
            branch_index..(branch_index + 1),
            0..batch_size,
            0..context_length,
            0..embedded_dim,
        ])
        .squeeze_dim(0)
}

fn extract_fanout_group<B: Backend>(
    tensor: Tensor<B, 4>,
    start: usize,
    len: usize,
    batch_size: usize,
    context_length: usize,
    embedded_dim: usize,
) -> Tensor<B, 4> {
    tensor.slice([
        start..(start + len),
        0..batch_size,
        0..context_length,
        0..embedded_dim,
    ])
}

fn build_mix_params<B: Backend>(weight_prepare: &WeightPrepare<B>) -> Tensor<B, 4> {
    Tensor::stack::<4>(
        vec![
            weight_prepare.param_receptance.val(),
            weight_prepare.param_key.val(),
            weight_prepare.param_value.val(),
            weight_prepare.param_weight_decay.val(),
            weight_prepare.param_learning_rate.val(),
        ],
        0,
    )
}

fn pack_grouped_linear_weights<B: Backend, const N: usize>(
    linears: [&Linear<B>; N],
) -> Tensor<B, 4> {
    Tensor::stack::<3>(
        linears
            .iter()
            .map(|linear| linear.weight.val())
            .collect::<Vec<_>>(),
        0,
    )
    .unsqueeze_dim::<4>(1)
}

fn grouped_linear_forward<B: Backend, const N: usize>(
    input: Tensor<B, 4>,
    linears: [&Linear<B>; N],
) -> Tensor<B, 4> {
    grouped_linear_forward_packed(input, pack_grouped_linear_weights(linears))
}

fn grouped_linear_forward_packed<B: Backend>(
    input: Tensor<B, 4>,
    weights: Tensor<B, 4>,
) -> Tensor<B, 4> {
    input.matmul(weights)
}

#[derive(Debug)]
struct PackedGroupedLoRA<B: Backend> {
    w_a: Tensor<B, 4>,
    w_b: Tensor<B, 4>,
    bias: Tensor<B, 4>,
}

fn pack_grouped_lora<B: Backend, const N: usize>(loras: [&LoRA<B>; N]) -> PackedGroupedLoRA<B> {
    PackedGroupedLoRA {
        w_a: Tensor::stack::<3>(
            loras.iter().map(|lora| lora.w_a.val()).collect::<Vec<_>>(),
            0,
        )
        .unsqueeze_dim::<4>(1),
        w_b: Tensor::stack::<3>(
            loras.iter().map(|lora| lora.w_b.val()).collect::<Vec<_>>(),
            0,
        )
        .unsqueeze_dim::<4>(1),
        bias: Tensor::stack::<4>(
            loras
                .iter()
                .map(|lora| lora.bias.as_ref().expect("expected LoRA bias").val())
                .collect::<Vec<_>>(),
            0,
        ),
    }
}

fn grouped_lora_forward<B: Backend, const N: usize>(
    input: Tensor<B, 4>,
    loras: [&LoRA<B>; N],
    activation_fns: [ActivationFn; N],
) -> Tensor<B, 4> {
    let PackedGroupedLoRA { w_a, w_b, bias } = pack_grouped_lora(loras);
    grouped_lora_forward_packed(input, w_a, w_b, bias, activation_fns)
}

fn grouped_lora_forward_packed<B: Backend, const N: usize>(
    input: Tensor<B, 4>,
    w_a: Tensor<B, 4>,
    w_b: Tensor<B, 4>,
    bias: Tensor<B, 4>,
    activation_fns: [ActivationFn; N],
) -> Tensor<B, 4> {
    let hidden = input.matmul(w_a);
    let [group_size, batch_size, context_length, hidden_dim] = hidden.dims();

    let activated: Tensor<B, 4> = Tensor::stack::<4>(
        activation_fns
            .into_iter()
            .enumerate()
            .map(|(index, activation_fn)| {
                let branch_hidden = extract_fanout_input(
                    hidden.clone(),
                    index,
                    batch_size,
                    context_length,
                    hidden_dim,
                );

                match activation_fn {
                    ActivationFn::Sigmoid => sigmoid(branch_hidden),
                    ActivationFn::Tanh => burn::tensor::activation::tanh(branch_hidden),
                    ActivationFn::NoOP => branch_hidden,
                }
            })
            .collect::<Vec<_>>(),
        0,
    );
    debug_assert_eq!(group_size, N);

    activated.matmul(w_b) + bias
}

impl<B: Backend> WeightPrepareOutput<B> {
    pub fn reshape_to_wkv7_input(&self, shape: [usize; 4]) -> Wkv7ForwardInput<B> {
        Wkv7ForwardInput {
            receptance: self.receptance.clone().reshape(shape),
            weight_decay: self.weight_decay.clone().reshape(shape),
            replacement_key: self.replacement_key.clone().reshape(shape),
            value: self.value.clone().reshape(shape),
            removal_key_normalized: self.removal_key_normalized.clone().reshape(shape),
            replacement: self.replacement.clone().reshape(shape),
        }
    }
}
