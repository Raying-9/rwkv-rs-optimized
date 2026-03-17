mod gated_readout;
pub mod param_state;
mod weight_prepare;

use burn::{config::Config, module::Module, prelude::*};

use crate::{
    kernels::wkv7_common::{Wkv7ForwardInput, Wkv7Kernel},
    layers::lora::LoRARanks,
};

use crate::functions::context_mask::apply_context_mask;
use crate::functions::token_shift::get_embedded_token_shift;
use crate::modules::time_mixer::gated_readout::GatedReadoutInput;
use gated_readout::{GatedReadout, GatedReadoutConfig};
use weight_prepare::{WeightPrepare, WeightPrepareConfig, WeightPrepareOutput};

const MIN_LORA_RANK: usize = 32;
const LORA_RANK_GRANULARITY: usize = 32;

fn build_lora_ranks(
    embedded_dim: usize,
    weight_decay_lora: usize,
    learning_rate_lora: usize,
    value_residual_lora: usize,
    output_gate_lora: usize,
) -> LoRARanks {
    LoRARanks {
        min_d_model: embedded_dim,
        weight_decay_lora,
        learning_rate_lora,
        value_residual_lora,
        output_gate_lora,
    }
}

fn infer_lora_rank_from_formula(value: f64) -> usize {
    let rounded =
        (value / LORA_RANK_GRANULARITY as f64).round_ties_even() as usize * LORA_RANK_GRANULARITY;

    rounded.max(MIN_LORA_RANK)
}

fn infer_default_lora_ranks(embedded_dim: usize, head_size: usize) -> LoRARanks {
    let channel_dim = embedded_dim as f64;
    let factor = head_size as f64 / 64.0;
    let sqrt_channel_dim = channel_dim.sqrt();

    let weight_decay_lora = infer_lora_rank_from_formula(2.5 * sqrt_channel_dim * factor);
    let learning_rate_lora = infer_lora_rank_from_formula(2.5 * sqrt_channel_dim * factor);
    let value_residual_lora = infer_lora_rank_from_formula(1.7 * sqrt_channel_dim * factor);
    let output_gate_lora = infer_lora_rank_from_formula(5.0 * sqrt_channel_dim);

    build_lora_ranks(
        embedded_dim,
        weight_decay_lora,
        learning_rate_lora,
        value_residual_lora,
        output_gate_lora,
    )
}

fn infer_override_lora_ranks(num_cells: usize, embedded_dim: usize) -> Option<LoRARanks> {
    match (num_cells, embedded_dim) {
        (12, 768) => Some(build_lora_ranks(embedded_dim, 64, 64, 32, 128)),
        (24, 1024) => Some(build_lora_ranks(embedded_dim, 64, 64, 32, 128)),
        (24, 2048) => Some(build_lora_ranks(embedded_dim, 96, 96, 64, 256)),
        (32, 2560) => Some(build_lora_ranks(embedded_dim, 96, 96, 64, 320)),
        (32, 4096) => Some(build_lora_ranks(embedded_dim, 128, 128, 96, 480)),
        (61, 4096) => Some(build_lora_ranks(embedded_dim, 192, 192, 128, 384)),
        _ => None,
    }
}

fn infer_time_mixer_lora_ranks(
    num_cells: usize,
    embedded_dim: usize,
    head_size: usize,
) -> LoRARanks {
    infer_override_lora_ranks(num_cells, embedded_dim)
        .unwrap_or_else(|| infer_default_lora_ranks(embedded_dim, head_size))
}

#[derive(Config, Debug)]
pub struct TimeMixerConfig {
    num_cells: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,
}

impl TimeMixerConfig {
    pub fn init<B: Backend>(&self, cell_id: usize, device: &B::Device) -> TimeMixer<B> {
        let lora_rank =
            infer_time_mixer_lora_ranks(self.num_cells, self.embedded_dim, self.head_size);

        TimeMixer {
            weight_prepare: WeightPrepareConfig::new(
                self.num_cells,
                self.embedded_dim,
                self.num_heads,
                self.head_size,
                lora_rank.weight_decay_lora,
                lora_rank.learning_rate_lora,
                lora_rank.value_residual_lora,
            )
            .init(cell_id, device),
            gated_readout: GatedReadoutConfig::new(
                self.num_cells,
                self.embedded_dim,
                self.num_heads,
                self.head_size,
                lora_rank.output_gate_lora,
            )
            .init(cell_id, device),

            num_cells: self.num_cells,
            embedded_dim: self.embedded_dim,
            num_heads: self.num_heads,
            head_size: self.head_size,

            cell_id,
        }
    }
}

#[derive(Module, Debug)]
pub struct TimeMixer<B: Backend> {
    weight_prepare: WeightPrepare<B>,
    gated_readout: GatedReadout<B>,

    num_cells: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,

    #[module(skip)]
    cell_id: usize,
}

impl<B: Backend> TimeMixer<B> {
    pub fn init_weights(&mut self, device: &B::Device) {
        self.weight_prepare.init_weights(device);
        self.gated_readout.init_weights(device);
    }

    pub fn prepare_inference_cache(&mut self) {
        self.weight_prepare.prepare_inference_cache();
    }

    #[cfg_attr(
        feature = "trace",
        tracing::instrument(name = "rwkv.infer.model.time_mixer", skip_all)
    )]
    pub fn forward<K: Wkv7Kernel<B>>(&self, time_mixer_input: TimeMixerIO<B>) -> TimeMixerIO<B> {
        let TimeMixerIO {
            embedded_context,
            context_mask,
            value_from_first_cell,
            embedded_token_shift,
            state,
        } = time_mixer_input;

        let [batch_size_per_device, context_length, _embedded_dim] = embedded_context.dims();

        let (num_heads, head_size) = (self.num_heads, self.head_size);

        let embedded_context = apply_context_mask(embedded_context, context_mask.clone());
        let value_from_first_cell = apply_context_mask(value_from_first_cell, context_mask.clone());

        let output_embedded_token_shift = embedded_token_shift.as_ref().map(|_| {
            if context_length == 1 {
                embedded_context.clone().squeeze_dim(1)
            } else {
                get_embedded_token_shift(embedded_context.clone())
            }
        });

        let WeightPrepareOutput {
            token_shifted_diff,
            value_from_first_cell,
            receptance,
            weight_decay,
            replacement_key,
            value,
            removal_key_normalized,
            replacement,
        } = self.weight_prepare.forward(
            embedded_context.clone(),
            value_from_first_cell.clone(),
            embedded_token_shift,
            context_mask.clone(),
        );

        let shape = [batch_size_per_device, context_length, num_heads, head_size];
        let wkv7_forward_input_receptance = receptance.clone().reshape(shape);
        let wkv7_forward_input_replacement_key = replacement_key.clone().reshape(shape);
        let wkv7_forward_input_value = value.clone().reshape(shape);
        let wkv7_forward_output = K::forward(
            Wkv7ForwardInput {
                receptance: receptance.reshape(shape),
                weight_decay: weight_decay.reshape(shape),
                replacement_key: replacement_key.reshape(shape),
                value: value.reshape(shape),
                removal_key_normalized: removal_key_normalized.reshape(shape),
                replacement: replacement.reshape(shape),
            },
            state,
            16,
            context_mask.clone(),
        );
        let gated_readout_input = GatedReadoutInput {
            embedded_context,
            token_shifted_diff,
            wkv7_forward_output: wkv7_forward_output.output,
            wkv7_forward_input_receptance,
            wkv7_forward_input_replacement_key,
            wkv7_forward_input_value,
        };

        let output_embedded_context = self.gated_readout.forward(gated_readout_input);

        TimeMixerIO {
            embedded_context: apply_context_mask(output_embedded_context, context_mask.clone()),
            context_mask: context_mask.clone(),
            value_from_first_cell: apply_context_mask(value_from_first_cell, context_mask),
            embedded_token_shift: output_embedded_token_shift,
            state: wkv7_forward_output.next_state,
        }
    }
}

pub struct TimeMixerIO<B: Backend> {
    pub embedded_context: Tensor<B, 3>,
    pub context_mask: Option<Tensor<B, 2>>,
    pub value_from_first_cell: Tensor<B, 3>,
    pub embedded_token_shift: Option<Tensor<B, 2>>,
    pub state: Option<Tensor<B, 4>>,
}
