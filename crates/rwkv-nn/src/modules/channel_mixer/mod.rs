use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    prelude::{Backend, Tensor},
    tensor::activation::relu,
};

use crate::functions::{
    context_mask::apply_context_mask,
    init_weights::{get_token_shift_diff_scale, uniform_init, zeros_init},
    token_shift::{get_embedded_token_shift, token_shift_diff, token_shift_diff_decode},
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
pub struct ChannelMixerConfig {
    num_cells: usize,
    embedded_dim: usize,
}

impl ChannelMixerConfig {
    pub fn init<B: Backend>(&self, cell_id: usize, device: &B::Device) -> ChannelMixer<B> {
        ChannelMixer {
            key: LinearConfig::new(self.embedded_dim, self.embedded_dim * 4)
                .with_bias(false)
                .init(device),
            value: LinearConfig::new(self.embedded_dim * 4, self.embedded_dim)
                .with_bias(false)
                .init(device),
            token_shift_diff_scale: Param::from_tensor(Tensor::empty(
                [1, 1, self.embedded_dim],
                device,
            )),
            num_cells: self.num_cells,
            embedded_dim: self.embedded_dim,
            cell_id,
        }
    }
}

#[derive(Module, Debug)]

pub struct ChannelMixer<B: Backend> {
    pub key: Linear<B>,
    pub value: Linear<B>,
    pub token_shift_diff_scale: Param<Tensor<B, 3>>,

    pub num_cells: usize,
    pub embedded_dim: usize,

    #[module(skip)]
    cell_id: usize,
}

impl<B: Backend> ChannelMixer<B> {
    pub fn init_weights(&mut self, device: &B::Device) {
        let bound = 0.5 / (self.embedded_dim as f64).sqrt();

        uniform_init(&mut self.key.weight, -bound, bound);

        zeros_init(&mut self.value.weight);

        self.token_shift_diff_scale = Param::from_tensor(get_token_shift_diff_scale(
            self.num_cells,
            self.embedded_dim,
            self.cell_id,
            4.0,
            device,
        ));
    }

    #[cfg_attr(
        feature = "trace",
        tracing::instrument(name = "rwkv.infer.model.channel_mixer", skip_all)
    )]
    pub fn forward(&self, channel_mixer_input: ChannelMixerIO<B>) -> ChannelMixerIO<B> {
        let ChannelMixerIO {
            embedded_context,
            embedded_token_shift,
            context_mask,
        } = channel_mixer_input;

        // `context_mask` handles physical left padding (and inactive lanes for batching).
        // Padding timesteps must be strict no-ops for the token-shift state; otherwise the
        // first real token would see an incorrect previous token.
        let embedded_context = {
            trace_lite_scope!("rwkv.infer.model.channel_mixer.apply_context_mask");
            apply_context_mask(embedded_context, context_mask.clone())
        };

        // Left-padding guarantees the last timestep is always valid (per lane), so no gating.
        let output_embedded_token_shift = embedded_token_shift.as_ref().map(|_| {
            trace_lite_scope!("rwkv.infer.model.channel_mixer.output_shift");
            get_embedded_token_shift(embedded_context.clone())
        });

        let [_, _context_length, _] = embedded_context.dims();
        let token_shifted_diff = {
            if _context_length == 1 {
                #[cfg(feature = "trace")]
                let _token_shift_scope = tracing::trace_span!(
                    "rwkv.infer.model.channel_mixer.token_shift_diff_decode_fastpath",
                    cell_id = self.cell_id
                )
                .entered();
                token_shift_diff_decode(
                    embedded_context.clone(),
                    embedded_token_shift,
                    context_mask.clone(),
                )
            } else {
                #[cfg(feature = "trace")]
                let _token_shift_scope = tracing::trace_span!(
                    "rwkv.infer.model.channel_mixer.token_shift_diff",
                    cell_id = self.cell_id,
                    context_length = _context_length
                )
                .entered();
                token_shift_diff(
                    embedded_context.clone(),
                    embedded_token_shift,
                    context_mask.clone(),
                )
            }
        };

        let embedded_context_shift = {
            trace_lite_scope!("rwkv.infer.model.channel_mixer.shift_mix");
            embedded_context.clone() + token_shifted_diff * self.token_shift_diff_scale.val()
        };

        let activated_key = {
            trace_lite_scope!("rwkv.infer.model.channel_mixer.projection_key");
            self.key.forward(embedded_context_shift)
        };
        let activated_key = {
            trace_lite_scope!("rwkv.infer.model.channel_mixer.activation");
            relu(activated_key).powf_scalar(2.0)
        };

        let value = {
            trace_lite_scope!("rwkv.infer.model.channel_mixer.projection_value");
            self.value.forward(activated_key)
        };
        let value = {
            trace_lite_scope!("rwkv.infer.model.channel_mixer.output_mask");
            apply_context_mask(value, context_mask.clone())
        };

        ChannelMixerIO {
            embedded_context: value,
            embedded_token_shift: output_embedded_token_shift,
            context_mask,
        }
    }
}

pub struct ChannelMixerIO<B: Backend> {
    pub embedded_context: Tensor<B, 3>,
    pub context_mask: Option<Tensor<B, 2>>, // [batch_size, context_length]
    pub embedded_token_shift: Option<Tensor<B, 2>>,
}
