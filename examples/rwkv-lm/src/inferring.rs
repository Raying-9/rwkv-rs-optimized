use itertools::izip;
use rwkv::{
    config::{raw::infer::GenerationConfig, validated::model::FinalModelConfig},
    custom::{
        Tensor,
        cubecl::device::DeviceId,
        module::Module,
        prelude::{Backend, DeviceOps, Int, TensorData},
        record::{FullPrecisionSettings, NamedMpkFileRecorder},
        tensor::{DType, IndexingUpdateOp},
    },
    infer::{
        Error, Result,
        inference_core::{
            InferenceExecutionConfig, InferenceExecutionLoop, ModelForward, SampledToken,
            SamplingConfig, SamplingConfigsTensor, TokenLogprobsConfig,
            build_sampled_token_logprob, sampling_configs_to_tensor,
        },
        model_pool::{LoadedModelGroup, ModelEngineFactory, build_model_group_engines},
    },
    nn::kernels::{
        rapid_sample::{RapidSampleBackend, rapid_sample},
        wkv7_common::Wkv7Backend,
    },
};
use std::{collections::HashMap, marker::PhantomData, sync::Arc};

use crate::model::{AutoRegressiveModel, AutoRegressiveModelConfig, UnembedMode};

rwkv::custom_mode!();

macro_rules! require_infer_cfg {
    ($cfg:expr, $field:ident) => {
        $cfg.$field.ok_or_else(|| {
            Error::BadRequest(format!(
                "missing {} for model {} in infer config",
                stringify!($field),
                &$cfg.model_name,
            ))
        })?
    };
}

pub struct RwkvLmForward<B: Backend> {
    model: AutoRegressiveModel<B>,
    embedded_token_shift_for_time_mix: Vec<Tensor<B, 2>>,
    state: Vec<Tensor<B, 4>>,
    embedded_token_shift_for_channel_mix: Vec<Tensor<B, 2>>,
    rng: Tensor<B, 1, Int>,
    penalties: Tensor<B, 2>,
    device: B::Device,

    max_batch_size: usize,
    vocab_size: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,
}

impl<B: Backend + Wkv7Backend + RapidSampleBackend> RwkvLmForward<B> {
    pub fn new(
        model: AutoRegressiveModel<B>,
        model_cfg: Arc<FinalModelConfig>,
        max_batch_size: usize,
        device: B::Device,
    ) -> Self {
        let num_cells = model_cfg.num_cells;
        let vocab_size = model_cfg.vocab_size;
        let embedded_dim = model_cfg.embedded_dim;
        let num_heads = model_cfg.num_heads;
        let head_size = model_cfg.head_size_auto;

        let init_embedded_token_shift = || {
            (0..num_cells)
                .map(|_| Tensor::zeros([max_batch_size, embedded_dim], &device))
                .collect::<Vec<_>>()
        };
        let init_state = || {
            (0..num_cells)
                .map(|_| Tensor::zeros([max_batch_size, num_heads, head_size, head_size], &device))
                .collect::<Vec<_>>()
        };

        let rng = Tensor::<B, 1, Int>::from_data(
            TensorData::new(
                (0..max_batch_size)
                    .map(|i| (i as i32).wrapping_add(1))
                    .collect(),
                [max_batch_size],
            ),
            &device,
        );

        let penalties = Tensor::<B, 2>::zeros([max_batch_size, vocab_size], (&device, DType::F32));

        Self {
            model,
            embedded_token_shift_for_time_mix: init_embedded_token_shift(),
            state: init_state(),
            embedded_token_shift_for_channel_mix: init_embedded_token_shift(),
            rng,
            penalties,
            device,

            max_batch_size,
            vocab_size,
            embedded_dim,
            num_heads,
            head_size,
        }
    }
}

impl<B> ModelForward for RwkvLmForward<B>
where
    B: Backend + Wkv7Backend + RapidSampleBackend,
{
    fn forward(
        &mut self,
        batch_ids: &[usize],
        contexts: &[&[i32]],
        context_masks: &[&[u8]],
        sampling_configs: &[SamplingConfig],
        token_logprobs: &[Option<TokenLogprobsConfig>],
        need_sample: bool,
    ) -> Result<Vec<SampledToken>> {
        rwkv_bench::trace_scope!("rwkv.infer.executor.forward_step");
        if batch_ids.len() != contexts.len() || contexts.len() != context_masks.len() {
            return Err(Error::BadRequest(
                "forward: inconsistent batch_ids/contexts/context_masks length".to_string(),
            ));
        }
        if batch_ids.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(&batch_id) = batch_ids.iter().find(|&&id| id >= self.max_batch_size) {
            return Err(Error::BadRequest(format!(
                "forward: batch_index {batch_id} out of range (max_batch_size={})",
                self.max_batch_size
            )));
        }

        let context_len = contexts[0].len();
        if context_len == 0 {
            return Err(Error::BadRequest(
                "forward: empty context is not allowed".to_string(),
            ));
        }
        for (idx, (context, mask)) in contexts.iter().zip(context_masks.iter()).enumerate() {
            if context.len() != context_len || mask.len() != context_len {
                return Err(Error::BadRequest(format!(
                    "forward: inconsistent context length at index {idx} (context={}, mask={}, expected={context_len})",
                    context.len(),
                    mask.len(),
                )));
            }
        }
        if sampling_configs.len() != batch_ids.len() {
            return Err(Error::BadRequest(format!(
                "forward: samplings length {} does not match batch size {}",
                sampling_configs.len(),
                batch_ids.len()
            )));
        }
        if token_logprobs.len() != batch_ids.len() {
            return Err(Error::BadRequest(format!(
                "forward: token_logprobs length {} does not match batch size {}",
                token_logprobs.len(),
                batch_ids.len()
            )));
        }

        let batch_size = batch_ids.len();

        let batch_masks = {
            let mut batch_masks = vec![1.0f32; self.max_batch_size];
            for &batch_id in batch_ids {
                batch_masks[batch_id] = 0.0;
            }
            Tensor::from_data(
                TensorData::new(batch_masks, [self.max_batch_size]),
                &self.device,
            )
        };

        let batch_ids_tensor: Tensor<B, 1, Int> = Tensor::from_data(
            TensorData::new(
                batch_ids.iter().map(|&id| id as i32).collect::<Vec<i32>>(),
                [batch_size],
            ),
            &self.device,
        );

        let contexts: Tensor<B, 2, Int> = Tensor::from_data(
            TensorData::new(contexts.concat(), [batch_size, context_len]),
            &self.device,
        );
        let context_masks = Tensor::from_data(
            TensorData::new(
                context_masks.concat().iter().map(|&m| m as f32).collect(),
                [batch_size, context_len],
            ),
            &self.device,
        );

        let (
            mut embedded_token_shift_for_time_mix,
            mut state,
            mut embedded_token_shift_for_channel_mix,
        ) = {
            rwkv_bench::trace_lite_scope!("rwkv.infer.executor.gather_state");
            (
                self.embedded_token_shift_for_time_mix
                    .iter()
                    .map(|x| x.clone().select(0, batch_ids_tensor.clone()))
                    .collect(),
                self.state
                    .iter()
                    .map(|x| x.clone().select(0, batch_ids_tensor.clone()))
                    .collect(),
                self.embedded_token_shift_for_channel_mix
                    .iter()
                    .map(|x| x.clone().select(0, batch_ids_tensor.clone()))
                    .collect(),
            )
        };

        let unembed_mode = if need_sample {
            UnembedMode::LastToken
        } else {
            UnembedMode::Skip
        };

        let logits = {
            rwkv_bench::trace_lite_scope!("rwkv.infer.executor.model_infer");
            self.model.infer(
                contexts,
                Some(context_masks),
                &mut embedded_token_shift_for_time_mix,
                &mut state,
                &mut embedded_token_shift_for_channel_mix,
                unembed_mode,
            )
        };

        let batch_masks_2d = batch_masks.clone().unsqueeze_dim::<2>(1);
        let batch_masks_4d = batch_masks
            .clone()
            .unsqueeze_dim::<2>(1)
            .unsqueeze_dim::<3>(2)
            .unsqueeze_dim::<4>(3);

        rwkv_bench::trace_lite_scope!("rwkv.infer.executor.scatter_state");
        for (
            cell_idx,
            (
                cell_embedded_token_shift_for_time_mix,
                cell_state,
                cell_embedded_token_shift_for_channel_mix,
            ),
        ) in izip!(
            embedded_token_shift_for_time_mix,
            state,
            embedded_token_shift_for_channel_mix,
        )
        .enumerate()
        {
            self.embedded_token_shift_for_time_mix[cell_idx] =
                (self.embedded_token_shift_for_time_mix[cell_idx].clone() * batch_masks_2d.clone())
                    .select_assign(
                        0,
                        batch_ids_tensor.clone(),
                        cell_embedded_token_shift_for_time_mix,
                        IndexingUpdateOp::Add,
                    );

            self.state[cell_idx] = (self.state[cell_idx].clone() * batch_masks_4d.clone())
                .select_assign(
                    0,
                    batch_ids_tensor.clone(),
                    cell_state,
                    IndexingUpdateOp::Add,
                );

            self.embedded_token_shift_for_channel_mix[cell_idx] =
                (self.embedded_token_shift_for_channel_mix[cell_idx].clone()
                    * batch_masks_2d.clone())
                .select_assign(
                    0,
                    batch_ids_tensor.clone(),
                    cell_embedded_token_shift_for_channel_mix,
                    IndexingUpdateOp::Add,
                );
        }

        match logits {
            Some(logits) => {
                rwkv_bench::trace_lite_scope!("rwkv.infer.executor.sample");
                let rng = self.rng.clone().select(0, batch_ids_tensor.clone());
                let penalties = self.penalties.clone().select(0, batch_ids_tensor.clone());

                let SamplingConfigsTensor {
                    inv_temperatures,
                    top_ks,
                    top_ps,
                    presence_penalties,
                    repetition_penalties,
                    penalties_decay,
                } = sampling_configs_to_tensor::<B>(
                    sampling_configs,
                    self.vocab_size,
                    &self.device,
                );

                let sample_output = rapid_sample::<B>(
                    logits.squeeze_dim(1),
                    rng,
                    inv_temperatures,
                    top_ks,
                    top_ps,
                    Some((
                        penalties,
                        presence_penalties,
                        repetition_penalties,
                        penalties_decay,
                    )),
                );

                // Scatter RNG states and penalties back using vectorized operations.
                {
                    rwkv_bench::trace_lite_scope!("rwkv.infer.executor.sample.scatter");

                    // Vectorized RNG scatter: replace N×4 kernel launches with 3 ops total
                    self.rng = (self.rng.clone().float() * batch_masks.clone())
                        .select_assign(
                            0,
                            batch_ids_tensor.clone(),
                            sample_output.states.clone().cast(DType::I32).float(),
                            IndexingUpdateOp::Add,
                        )
                        .int();

                    // Scatter penalties back via mask + select_assign.
                    if let Some(ref updated_penalties) = sample_output.penalties {
                        let penalties_mask_2d = batch_masks_2d.clone().cast(DType::F32);
                        self.penalties = (self.penalties.clone() * penalties_mask_2d)
                            .select_assign(
                                0,
                                batch_ids_tensor.clone(),
                                updated_penalties.clone(),
                                IndexingUpdateOp::Add,
                            );
                    }
                }

                let token_ids = sample_output.token_ids.to_data().to_vec::<i32>().unwrap();
                let probs = token_logprobs
                    .iter()
                    .any(|cfg| cfg.is_some())
                    .then(|| sample_output.probs.to_data().to_vec::<f32>().unwrap());

                Ok(batch_ids
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(row_index, batch_index)| {
                        let token_id = token_ids[row_index];
                        let logprob = token_logprobs[row_index].as_ref().map(|cfg| {
                            let probs = probs.as_ref().expect(
                                "probability buffer must exist when logprobs are requested",
                            );
                            let row_start = row_index * self.vocab_size;
                            let row_end = row_start + self.vocab_size;
                            build_sampled_token_logprob(&probs[row_start..row_end], token_id, cfg)
                        });

                        SampledToken {
                            batch_index,
                            token_id,
                            logprob,
                        }
                    })
                    .collect())
            }
            None => Ok(Vec::new()),
        }
    }

    fn reset(&mut self, batch_index: usize) -> Result<()> {
        rwkv_bench::trace_lite_scope!("rwkv.infer.executor.reset_slot");
        if batch_index >= self.max_batch_size {
            return Err(Error::BadRequest(format!(
                "reset: batch_index {batch_index} out of range (max_batch_size={})",
                self.max_batch_size
            )));
        }

        let zeros_shift = Tensor::<B, 2>::zeros([1, self.embedded_dim], &self.device);
        let reset_shift = |buffers: &mut Vec<Tensor<B, 2>>| {
            for buffer in buffers.iter_mut() {
                *buffer = buffer
                    .clone()
                    .slice_assign([batch_index..batch_index + 1], zeros_shift.clone());
            }
        };

        reset_shift(&mut self.embedded_token_shift_for_time_mix);
        reset_shift(&mut self.embedded_token_shift_for_channel_mix);

        let zeros_state = Tensor::<B, 4>::zeros(
            [1, self.num_heads, self.head_size, self.head_size],
            &self.device,
        );
        for s in self.state.iter_mut() {
            *s = s
                .clone()
                .slice_assign([batch_index..batch_index + 1], zeros_state.clone());
        }

        let seed = Tensor::<B, 1, Int>::from_data(
            TensorData::new(vec![(batch_index as i32).wrapping_add(1)], [1]),
            &self.device,
        );
        self.rng = self
            .rng
            .clone()
            .slice_assign([batch_index..batch_index + 1], seed);

        let zeros_pen = Tensor::<B, 2>::zeros([1, self.vocab_size], (&self.device, DType::F32));
        self.penalties = self.penalties.clone().slice_assign(
            [batch_index..batch_index + 1, 0..self.vocab_size],
            zeros_pen,
        );

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ModelEngineFactory — builds model groups for LoadedModelRegistry
// ---------------------------------------------------------------------------

pub struct RwkvLmEngineFactory<B> {
    _marker: PhantomData<B>,
}

impl<B> RwkvLmEngineFactory<B> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<B> ModelEngineFactory for RwkvLmEngineFactory<B>
where
    B: Backend + Wkv7Backend + RapidSampleBackend + Send + Sync,
{
    fn build_model_groups(
        &self,
        models: &[GenerationConfig],
        model_cfgs: &HashMap<String, Arc<FinalModelConfig>>,
    ) -> Result<HashMap<String, LoadedModelGroup>> {
        build_model_group_engines(models, |generation_cfg| {
            let model_cfg = model_cfgs.get(&generation_cfg.model_name).ok_or_else(|| {
                Error::Internal(format!(
                    "missing model cfg for {}",
                    generation_cfg.model_name
                ))
            })?;
            let (device_type, max_batch_size, paragraph_len, max_context_len, decode_first) = (
                require_infer_cfg!(generation_cfg, device_type),
                require_infer_cfg!(generation_cfg, max_batch_size),
                require_infer_cfg!(generation_cfg, paragraph_len),
                require_infer_cfg!(generation_cfg, max_context_len),
                require_infer_cfg!(generation_cfg, decode_first),
            );

            let mut engines = Vec::new();
            for device_id in &generation_cfg.device_ids {
                let device = B::Device::from_id(DeviceId::new(device_type, *device_id));

                let model_config = AutoRegressiveModelConfig::new(
                    model_cfg.num_cells,
                    model_cfg.vocab_size,
                    model_cfg.embedded_dim,
                    model_cfg.num_heads,
                    model_cfg.head_size_auto,
                );
                let model_runtime = model_config.init::<B>(&device);
                let model_runtime = model_runtime
                    .load_file(
                        &generation_cfg.weights_path,
                        &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
                        &device,
                    )
                    .map_err(|e| {
                        Error::Internal(format!(
                            "failed to load weights {} for model {}: {e}",
                            generation_cfg.weights_path, generation_cfg.model_name
                        ))
                    })?;

                let executor = RwkvLmForward::<B>::new(
                    model_runtime,
                    model_cfg.clone(),
                    max_batch_size,
                    device.clone(),
                );

                let engine = InferenceExecutionLoop::spawn(
                    InferenceExecutionConfig {
                        tokenizer_vocab_path: generation_cfg.tokenizer_vocab_path.clone(),
                        max_batch_size,
                        paragraph_len,
                        max_context_len,
                        decode_first,
                    },
                    Box::new(executor),
                )?;

                log::info!(
                    "engine ready: model_name={} model_cfg={} device_type={} device_id={} \
                     weights_path={}",
                    generation_cfg.model_name,
                    generation_cfg.model_cfg,
                    device_type,
                    device_id,
                    generation_cfg.weights_path
                );
                engines.push(Arc::new(engine));
            }

            Ok(engines)
        })
    }
}
