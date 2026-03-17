use rwkv::custom::Tensor;
use rwkv::custom::config::Config;
use rwkv::custom::module::Module;
use rwkv::custom::nn::{
    Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
};
use rwkv::custom::prelude::Int;
use rwkv::custom::tensor::backend::Backend;
use rwkv::nn::cells::causal::{MultiCausalCells, MultiCausalCellsConfig, MultiCausalCellsIO};
use rwkv::nn::functions::init_weights::{orthogonal_init, uniform_init};
use rwkv::nn::kernels::wkv7_common::{KernelInfer, Wkv7Backend};
use rwkv::nn::modules::time_mixer::param_state::{StateModule, StateModuleConfig};
use std::mem::take;

/// Controls the unembed layer computation mode.
/// Cf. albatross `full_output`, extended with `Skip` for chunked prefill.
pub enum UnembedMode {
    /// Skip layer_norm + unembed; only update recurrent state. For prefill intermediate chunks.
    Skip,
    /// Unembed only the last token position. For prefill final chunk and decode.
    /// Corresponds to albatross `full_output=False`.
    LastToken,
    /// Unembed all token positions. For evaluation/scoring.
    /// Corresponds to albatross `full_output=True`.
    Full,
}

#[cfg(feature = "training")]
use crate::data::batcher::AutoRegressiveBatch;
#[cfg(feature = "training")]
use rwkv::custom::nn::loss::CrossEntropyLossConfig;
#[cfg(feature = "training")]
use rwkv::custom::tensor::backend::AutodiffBackend;
#[cfg(feature = "training")]
use rwkv::custom::train::{InferenceStep, TrainOutput, TrainStep};
#[cfg(feature = "training")]
use rwkv::nn::kernels::l2wrap::{L2WrapBackend, l2wrap};
#[cfg(feature = "training")]
use rwkv::nn::kernels::wkv7_common::Wkv7Kernel;
#[cfg(feature = "training")]
use rwkv::train::learner::next_token_prediction::NextTokenPredictionOutput;

rwkv::custom_mode!();

#[derive(Config, Debug)]
pub struct AutoRegressiveModelConfig {
    num_cells: usize,
    vocab_size: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,
}

impl AutoRegressiveModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AutoRegressiveModel<B> {
        AutoRegressiveModel {
            embed: EmbeddingConfig::new(self.vocab_size, self.embedded_dim).init(device),
            layer_norm_for_first_cell: LayerNormConfig::new(self.embedded_dim).init(device),
            cells: MultiCausalCellsConfig::new(
                self.num_cells,
                self.embedded_dim,
                self.num_heads,
                self.head_size,
            )
            .init(device),
            layer_norm_for_unembed: LayerNormConfig::new(self.embedded_dim).init(device),
            unembed: LinearConfig::new(self.embedded_dim, self.vocab_size)
                .with_bias(false)
                .init(device),
            state: StateModuleConfig::new(self.num_cells, self.num_heads, self.head_size)
                .init(device),

            num_cells: self.num_cells,
            vocab_size: self.vocab_size,
            embedded_dim: self.embedded_dim,
            num_heads: self.num_heads,
            head_size: self.head_size,
        }
    }
}

#[derive(Module, Debug)]
pub struct AutoRegressiveModel<B: Backend> {
    pub embed: Embedding<B>,
    pub layer_norm_for_first_cell: LayerNorm<B>,
    pub cells: MultiCausalCells<B>,
    pub layer_norm_for_unembed: LayerNorm<B>,
    pub unembed: Linear<B>,
    pub state: StateModule<B>,

    num_cells: usize,
    vocab_size: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,
}

impl<B: Backend> AutoRegressiveModel<B> {
    pub fn init_weights(&mut self, device: &B::Device) {
        uniform_init(&mut self.embed.weight, -1e-4, 1e-4);

        if self.vocab_size > self.embedded_dim {
            orthogonal_init(
                &mut self.unembed.weight,
                Some(0.5 * (self.vocab_size as f32 / self.embedded_dim as f32).sqrt()),
            );
        } else {
            orthogonal_init(&mut self.unembed.weight, Some(0.5));
        }

        self.cells.init_weights(device);
    }

    pub fn prepare_inference_cache(&mut self) {
        self.cells.prepare_inference_cache();
    }

    #[cfg(feature = "training")]
    pub fn forward<K: Wkv7Kernel<B>>(
        &self,
        item: AutoRegressiveBatch<B>,
        embedded_token_shift_for_time_mix: Option<Vec<Tensor<B, 2>>>,
        state: Option<Vec<Tensor<B, 4>>>,
        embedded_token_shift_for_channel_mix: Option<Vec<Tensor<B, 2>>>,
    ) -> (
        Option<Vec<Tensor<B, 2>>>,
        Option<Vec<Tensor<B, 4>>>,
        Option<Vec<Tensor<B, 2>>>,
        NextTokenPredictionOutput<B>,
    )
    where
        B: L2WrapBackend,
    {
        let [batch_size, context_length] = item.inputs.dims();
        let device = &self.embed.devices()[0];

        let inputs = item.inputs.to_device(device);
        let targets = item.targets.to_device(device);

        let embedded_context = self.embed.forward(inputs);
        let embedded_context_normalized = self.layer_norm_for_first_cell.forward(embedded_context);

        let multi_causal_cells_input = MultiCausalCellsIO {
            embedded_context: embedded_context_normalized,
            context_mask: None,
            embedded_token_shift_for_time_mix,
            state,
            embedded_token_shift_for_channel_mix,
        };
        let multi_causal_cells_output = self.cells.forward::<K>(multi_causal_cells_input);
        let embedded_context = multi_causal_cells_output.embedded_context;
        let embedded_context_normalized = self.layer_norm_for_unembed.forward(embedded_context);
        let logits = self.unembed.forward(embedded_context_normalized);

        let num_tokens_per_batch = batch_size * context_length;
        let logits_flat = logits.reshape([num_tokens_per_batch, self.vocab_size]);
        let targets_flat = targets.reshape([num_tokens_per_batch]);

        let logits_flat = l2wrap(logits_flat, num_tokens_per_batch);

        let loss = CrossEntropyLossConfig::new()
            .init(&logits_flat.device())
            .forward(logits_flat, targets_flat);

        // Return the output and loss
        (
            multi_causal_cells_output.embedded_token_shift_for_time_mix,
            multi_causal_cells_output.state,
            multi_causal_cells_output.embedded_token_shift_for_channel_mix,
            NextTokenPredictionOutput { loss },
        )
    }

    #[cfg_attr(
        feature = "trace",
        tracing::instrument(name = "rwkv.infer.model.infer", skip_all)
    )]
    pub fn infer(
        &self,
        contexts: Tensor<B, 2, Int>,
        context_masks: Option<Tensor<B, 2>>,
        embedded_token_shift_for_time_mix: &mut Vec<Tensor<B, 2>>,
        state: &mut Vec<Tensor<B, 4>>,
        embedded_token_shift_for_channel_mix: &mut Vec<Tensor<B, 2>>,
        unembed_mode: UnembedMode,
    ) -> Option<Tensor<B, 3>>
    where
        B: Wkv7Backend,
    {
        debug_assert_eq!(
            embedded_token_shift_for_time_mix.len(),
            self.num_cells,
            "embedded_token_shift_for_time_mix must have num_cells elements"
        );
        debug_assert_eq!(
            state.len(),
            self.num_cells,
            "state must have num_cells elements"
        );
        debug_assert_eq!(
            embedded_token_shift_for_channel_mix.len(),
            self.num_cells,
            "embedded_token_shift_for_channel_mix must have num_cells elements"
        );

        let device = &self.embed.devices()[0];

        let contexts = contexts.to_device(device);
        let context_masks = context_masks.map(|m| m.to_device(device));

        let [batch_size, context_length] = contexts.dims();
        assert!(context_length > 0, "tokens must be non-empty");

        if let Some(mask) = context_masks.as_ref() {
            let [mask_batch_size, mask_context_length] = mask.dims();
            assert_eq!(
                (batch_size, context_length),
                (mask_batch_size, mask_context_length),
                "context_mask shape mismatch with tokens"
            );
        }

        let embedded_context = {
            rwkv_bench::trace_lite_scope!("rwkv.infer.model.embed");
            self.embed.forward(contexts)
        };
        let embedded_context = {
            rwkv_bench::trace_lite_scope!("rwkv.infer.model.layer_norm_pre");
            self.layer_norm_for_first_cell.forward(embedded_context)
        };

        let multi_causal_cells_input = MultiCausalCellsIO {
            embedded_context,
            context_mask: context_masks.clone(),
            embedded_token_shift_for_time_mix: Some(take(embedded_token_shift_for_time_mix)),
            state: Some(take(state)),
            embedded_token_shift_for_channel_mix: Some(take(embedded_token_shift_for_channel_mix)),
        };
        let multi_causal_cells_output = self.cells.forward::<KernelInfer>(multi_causal_cells_input);

        *embedded_token_shift_for_time_mix = multi_causal_cells_output
            .embedded_token_shift_for_time_mix
            .expect("infer requires embedded_token_shift_for_time_mix");
        *state = multi_causal_cells_output
            .state
            .expect("infer requires state");
        *embedded_token_shift_for_channel_mix = multi_causal_cells_output
            .embedded_token_shift_for_channel_mix
            .expect("infer requires embedded_token_shift_for_channel_mix");

        let embedded_context = multi_causal_cells_output.embedded_context;

        match unembed_mode {
            UnembedMode::Skip => None,
            UnembedMode::LastToken => {
                rwkv_bench::trace_lite_scope!("rwkv.infer.model.unembed");
                let last =
                    embedded_context.slice([0..batch_size, (context_length - 1)..context_length]);
                Some(
                    self.unembed
                        .forward(self.layer_norm_for_unembed.forward(last)),
                )
            }
            UnembedMode::Full => {
                rwkv_bench::trace_lite_scope!("rwkv.infer.model.unembed");
                Some(
                    self.unembed
                        .forward(self.layer_norm_for_unembed.forward(embedded_context)),
                )
            }
        }
    }
}

#[cfg(feature = "training")]
impl<B: AutodiffBackend + Wkv7Backend + L2WrapBackend> TrainStep for AutoRegressiveModel<B> {
    type Input = AutoRegressiveBatch<B>;
    type Output = NextTokenPredictionOutput<B>;

    #[cfg(not(any(feature = "statetune", feature = "statepass")))]
    #[allow(unused)]
    fn step(&self, item: AutoRegressiveBatch<B>) -> TrainOutput<NextTokenPredictionOutput<B>> {
        // Run forward pass, calculate gradients and return them along with the output
        use rwkv::nn::kernels::wkv7_common::KernelPretrain;

        let (_, _, _, item) = self.forward::<KernelPretrain>(item, None, None, None);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }

    #[cfg(feature = "statetune")]
    fn step(&self, item: AutoRegressiveBatch<B>) -> TrainOutput<NextTokenPredictionOutput<B>> {
        // Run forward pass, calculate gradients and return them along with the output
        use rwkv::nn::kernels::wkv7_common::KernelStateTune;

        let [batch_size, _] = item.inputs.dims();
        let state = self.state.get_state(batch_size);
        let (_, _, _, item) = self.forward::<KernelStateTune>(item, None, Some(state), None);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }

    #[cfg(feature = "statepass")]
    fn step(&self, item: AutoRegressiveBatch<B>) -> TrainOutput<NextTokenPredictionOutput<B>> {
        // Run forward pass, calculate gradients and return them along with the output
        use rwkv::config::validated::train::TRAIN_CFG;
        use rwkv::nn::kernels::wkv7_common::KernelStatePass;

        let [batch_size, context_length] = item.inputs.dims();
        let device = &item.inputs.device();

        let paragraph_length = TRAIN_CFG.get().unwrap().paragraph_length;

        let mut embedded_token_shift_for_time_mix: Vec<Tensor<B, 2>> = (0..self.num_cells)
            .map(|_| Tensor::zeros([batch_size, self.embedded_dim], device))
            .collect();
        let mut state: Vec<Tensor<B, 4>> = (0..self.num_cells)
            .map(|_| {
                Tensor::zeros(
                    [batch_size, self.num_heads, self.head_size, self.head_size],
                    device,
                )
            })
            .collect();
        let mut embedded_token_shift_for_channel_mix: Vec<Tensor<B, 2>> = (0..self.num_cells)
            .map(|_| Tensor::zeros([batch_size, self.embedded_dim], device))
            .collect();
        let mut sum_loss: Tensor<B, 1> = Tensor::zeros([1], device);

        for paragraph_index in 0..context_length / paragraph_length {
            let paragraph_item = AutoRegressiveBatch {
                inputs: item.inputs.clone().slice([
                    0..batch_size,
                    paragraph_index * paragraph_length..(paragraph_index + 1) * paragraph_length,
                ]),
                targets: item.targets.clone().slice([
                    0..batch_size,
                    paragraph_index * paragraph_length..(paragraph_index + 1) * paragraph_length,
                ]),
            };

            let (
                output_embedded_token_shift_for_time_mix,
                output_state,
                output_embedded_token_shift_for_channel_mix,
                item,
            ) = self.forward::<KernelStatePass>(
                paragraph_item,
                Some(embedded_token_shift_for_time_mix),
                Some(state),
                Some(embedded_token_shift_for_channel_mix),
            );

            embedded_token_shift_for_time_mix = output_embedded_token_shift_for_time_mix
                .unwrap()
                .into_iter()
                .map(|t| t.detach())
                .collect();
            state = output_state
                .unwrap()
                .into_iter()
                .map(|t| t.detach())
                .collect();
            embedded_token_shift_for_channel_mix = output_embedded_token_shift_for_channel_mix
                .unwrap()
                .into_iter()
                .map(|t| t.detach())
                .collect();

            sum_loss = sum_loss + item.loss.mul_scalar(paragraph_length as i32);
        }

        let loss = sum_loss.div_scalar(context_length as i32);
        let grads = loss.backward();

        TrainOutput::new(self, grads, NextTokenPredictionOutput { loss })
    }
}

/// Define validation step
#[cfg(feature = "training")]
impl<B: Backend + Wkv7Backend + L2WrapBackend> InferenceStep for AutoRegressiveModel<B> {
    type Input = AutoRegressiveBatch<B>;
    type Output = NextTokenPredictionOutput<B>;

    fn step(&self, item: AutoRegressiveBatch<B>) -> NextTokenPredictionOutput<B> {
        // Run forward pass and return the output
        NextTokenPredictionOutput {
            loss: Tensor::zeros([1], &item.inputs.device()),
        }
    }
}
