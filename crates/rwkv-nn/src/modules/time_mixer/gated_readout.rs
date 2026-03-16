use burn::{
    config::Config,
    module::{Module, Param},
    nn::{GroupNorm, GroupNormConfig, Linear, LinearConfig},
    prelude::*,
};

use crate::{
    functions::init_weights::{constant_init, get_token_shift_diff_scale, zeros_init},
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
pub struct GatedReadoutConfig {
    num_cells: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,

    output_gate_lora_rank: usize,
}

impl GatedReadoutConfig {
    pub fn init<B: Backend>(&self, cell_id: usize, device: &B::Device) -> GatedReadout<B> {
        let empty_param = Param::from_tensor(Tensor::empty([1, 1, self.embedded_dim], device));

        let projection_output = LinearConfig::new(self.embedded_dim, self.embedded_dim)
            .with_bias(false)
            .init(device);

        GatedReadout {
            param_gate: empty_param,

            param_output_gate_lora: LoRAConfig::new(
                self.num_cells,
                self.embedded_dim,
                self.output_gate_lora_rank,
                self.head_size,
                false,
                ActivationFn::Sigmoid,
            )
            .init(device),

            param_receptance_key_bonus: Param::from_tensor(Tensor::empty(
                [self.num_heads, self.embedded_dim / self.num_heads],
                device,
            )),
            group_norm: GroupNormConfig::new(self.num_heads, self.embedded_dim)
                .with_epsilon(64e-5)
                .init(device),
            projection_output,

            num_cells: self.num_cells,
            embedded_dim: self.embedded_dim,
            num_heads: self.num_heads,
            head_size: self.head_size,

            cell_id,
        }
    }
}

#[derive(Module, Debug)]
pub struct GatedReadout<B: Backend> {
    param_gate: Param<Tensor<B, 3, Float>>,

    param_output_gate_lora: LoRA<B>,

    param_receptance_key_bonus: Param<Tensor<B, 2>>,
    group_norm: GroupNorm<B>,
    projection_output: Linear<B>,

    num_cells: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,

    #[module(skip)]
    cell_id: usize,
}

impl<B: Backend> GatedReadout<B> {
    pub fn init_weights(&mut self, device: &B::Device) {
        self.param_gate = Param::from_tensor(get_token_shift_diff_scale(
            self.num_cells,
            self.embedded_dim,
            self.cell_id,
            0.2,
            device,
        ));

        self.param_output_gate_lora
            .init_weight(self.cell_id, LoRAType::LearningRate, device);

        constant_init(&mut self.param_receptance_key_bonus, -0.04);

        zeros_init(&mut self.projection_output.weight);

        if let Some(ref mut gamma) = self.group_norm.gamma {
            let layer_scale = ((1 + self.cell_id) as f64 / self.num_cells as f64).powf(0.7);

            constant_init(gamma, layer_scale);
        }
    }

    #[cfg_attr(
        feature = "trace",
        tracing::instrument(name = "rwkv.infer.model.gated_readout", skip_all)
    )]
    pub fn forward(&self, gated_readout_input: GatedReadoutInput<B>) -> Tensor<B, 3> {
        let GatedReadoutInput {
            embedded_context,
            token_shifted_diff,
            wkv7_forward_output,
            wkv7_forward_input_receptance,
            wkv7_forward_input_replacement_key,
            wkv7_forward_input_value,
        } = gated_readout_input;

        let [batch_size_per_device, context_length, embedded_dim] = embedded_context.dims();

        let embedded_context_gate = {
            trace_lite_scope!("rwkv.infer.model.gated_readout.gate_input");
            embedded_context + token_shifted_diff * self.param_gate.val()
        };

        let gate = {
            trace_lite_scope!("rwkv.infer.model.gated_readout.gate");
            self.param_output_gate_lora.forward(embedded_context_gate)
        };

        let wkv7_forward_output_normalized = {
            trace_lite_scope!("rwkv.infer.model.gated_readout.group_norm");
            self.group_norm
                .forward(
                    wkv7_forward_output
                        .reshape([batch_size_per_device * context_length, embedded_dim]),
                )
                .reshape([batch_size_per_device, context_length, embedded_dim])
        };

        let bonus: Tensor<B, 4> = {
            trace_lite_scope!("rwkv.infer.model.gated_readout.bonus");
            (wkv7_forward_input_receptance
                * wkv7_forward_input_replacement_key
                * self
                    .param_receptance_key_bonus
                    .val()
                    .unsqueeze_dims(&[0, 1]))
            .sum_dim(3)
                * wkv7_forward_input_value
        };

        let bonus: Tensor<B, 3> = {
            trace_lite_scope!("rwkv.infer.model.gated_readout.bonus_reshape");
            bonus.reshape([batch_size_per_device, context_length, embedded_dim])
        };

        let out_gated = {
            trace_lite_scope!("rwkv.infer.model.gated_readout.apply_gate");
            (wkv7_forward_output_normalized + bonus) * gate
        };

        {
            trace_lite_scope!("rwkv.infer.model.gated_readout.projection_output");
            self.projection_output.forward(out_gated)
        }
    }
}

pub struct GatedReadoutInput<B: Backend> {
    pub embedded_context: Tensor<B, 3>,
    pub token_shifted_diff: Tensor<B, 3>,
    pub wkv7_forward_output: Tensor<B, 4>,
    pub wkv7_forward_input_receptance: Tensor<B, 4>,
    pub wkv7_forward_input_replacement_key: Tensor<B, 4>,
    pub wkv7_forward_input_value: Tensor<B, 4>,
}
