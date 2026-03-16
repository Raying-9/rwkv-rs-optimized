use std::collections::{HashMap, HashSet};
use std::time::Instant;

use rwkv_data::tokenizer::Tokenizer;
use tokio::sync::{mpsc, oneshot};

use super::batch_scheduler::Scheduler;
use super::request_state::{RequestPhase, RequestState};
use super::request_submit::{InferenceSubmitCommand, InferenceSubmitHandle, InferenceSubmitResult};
use super::special_token::{END_TOKEN_ID, PREFILL_PAD_TOKEN_ID};
use super::tokenizer_loop::{TokenizerCommand, TokenizerLoop};
use super::{
    EntryId, FinishMetadata, FinishReason, OutputToken, OutputTokenCandidate, SampledToken,
    SamplingConfig, StopMatch, TimingBreakdownMs, TokenLogprobsConfig,
};

#[derive(Clone, Debug)]
pub struct InferenceExecutionConfig {
    pub tokenizer_vocab_path: String,
    pub max_batch_size: usize,
    pub paragraph_len: usize,
    pub max_context_len: usize,
    pub decode_first: bool,
}

impl Default for InferenceExecutionConfig {
    fn default() -> Self {
        Self {
            tokenizer_vocab_path: String::new(),
            max_batch_size: 4,
            paragraph_len: 256,
            max_context_len: 4096,
            decode_first: true,
        }
    }
}

/// Model forward inference interface. Prefill and decode are unified through this interface.
pub trait ModelForward: Send + 'static {
    fn forward(
        &mut self,
        batch_ids: &[usize],
        contexts: &[&[i32]],
        masks: &[&[u8]],
        samplings: &[SamplingConfig],
        token_logprobs: &[Option<TokenLogprobsConfig>],
        need_sample: bool,
    ) -> crate::Result<Vec<SampledToken>>;

    fn reset(&mut self, batch_index: usize) -> crate::Result<()>;
}

struct PendingSubmit {
    entry_id: EntryId,
    input_text: String,
    sampling: SamplingConfig,
    stop_suffixes: Vec<String>,
    token_logprobs: Option<TokenLogprobsConfig>,
    submitted_at: Instant,
    runtime_received_at: Instant,
    validate_ms: Option<u64>,
    reply: oneshot::Sender<InferenceSubmitResult>,
}

#[derive(Default)]
struct ForwardBatch {
    entry_ids: Vec<EntryId>,
    batch_ids: Vec<usize>,
    contexts: Vec<Vec<i32>>,
    context_masks: Vec<Vec<u8>>,
    sampling_configs: Vec<SamplingConfig>,
    token_logprob_configs: Vec<Option<TokenLogprobsConfig>>,
    batch_to_entry: HashMap<usize, EntryId>,
}

impl ForwardBatch {
    fn push(
        &mut self,
        entry_id: EntryId,
        batch_index: usize,
        context: Vec<i32>,
        context_mask: Vec<u8>,
        sampling: SamplingConfig,
        token_logprobs: Option<TokenLogprobsConfig>,
    ) {
        self.entry_ids.push(entry_id);
        self.batch_ids.push(batch_index);
        self.contexts.push(context);
        self.context_masks.push(context_mask);
        self.sampling_configs.push(sampling);
        self.token_logprob_configs.push(token_logprobs);
        self.batch_to_entry.insert(batch_index, entry_id);
    }

    fn is_empty(&self) -> bool {
        self.batch_ids.is_empty()
    }

    fn entry_ids(&self) -> &[EntryId] {
        &self.entry_ids
    }
}

#[derive(Clone, Copy, Debug)]
enum SampleSource {
    Decode,
    Prefill,
}

#[derive(Clone, Copy, Debug)]
enum RequestStopReason {
    EndToken,
    StopSuffix(StopMatch),
    Length,
}

struct SampleApplyOutcome {
    entry_id: EntryId,
    output_token: Option<OutputToken>,
    finish_meta: Option<FinishMetadata>,
    finished_batch_index: Option<usize>,
}

pub struct InferenceExecutionLoop {
    cfg: InferenceExecutionConfig,
    tokenizer: Tokenizer,
    scheduler: Scheduler,
    entries: HashMap<EntryId, RequestState>,
    rx: mpsc::Receiver<InferenceSubmitCommand>,
    tokenizer_tx: mpsc::UnboundedSender<TokenizerCommand>,
    executor: Box<dyn ModelForward>,
}

impl InferenceExecutionLoop {
    pub fn spawn(
        cfg: InferenceExecutionConfig,
        executor: Box<dyn ModelForward>,
    ) -> crate::Result<InferenceSubmitHandle> {
        let input_tokenizer = Tokenizer::new(&cfg.tokenizer_vocab_path).map_err(|e| {
            crate::Error::bad_request(format!(
                "failed to load tokenizer vocab {}: {e}",
                cfg.tokenizer_vocab_path
            ))
        })?;
        let output_tokenizer = Tokenizer::new(&cfg.tokenizer_vocab_path).map_err(|e| {
            crate::Error::bad_request(format!(
                "failed to load tokenizer vocab {}: {e}",
                cfg.tokenizer_vocab_path
            ))
        })?;

        let (tx, rx) = mpsc::channel(1024);
        let (tokenizer_tx, tokenizer_rx) = mpsc::unbounded_channel();
        let handle = InferenceSubmitHandle::new(tx.clone());

        tokio::spawn(TokenizerLoop::new(output_tokenizer, tokenizer_rx, tx).run());
        tokio::spawn(Self::new(cfg, input_tokenizer, rx, tokenizer_tx, executor).run());
        Ok(handle)
    }

    pub fn new(
        cfg: InferenceExecutionConfig,
        tokenizer: Tokenizer,
        rx: mpsc::Receiver<InferenceSubmitCommand>,
        tokenizer_tx: mpsc::UnboundedSender<TokenizerCommand>,
        executor: Box<dyn ModelForward>,
    ) -> Self {
        let scheduler = Scheduler::new(cfg.max_batch_size, cfg.paragraph_len);
        Self {
            cfg,
            tokenizer,
            scheduler,
            entries: HashMap::new(),
            rx,
            tokenizer_tx,
            executor,
        }
    }

    pub async fn run(mut self) {
        loop {
            self.drain_pending_commands();

            if !self.has_ready_work() && !self.wait_for_commands().await {
                break;
            }

            while self.has_ready_work() {
                self.drain_pending_commands();

                if !self.has_ready_work() {
                    break;
                }

                if !self.tick_once() {
                    break;
                }
            }
        }
    }

    fn handle_command(
        &mut self,
        cmd: InferenceSubmitCommand,
        pending_submits: &mut Vec<PendingSubmit>,
        cancelled_entry_ids: &mut Vec<EntryId>,
    ) {
        match cmd {
            InferenceSubmitCommand::SubmitText {
                entry_id,
                input_text,
                sampling,
                stop_suffixes,
                token_logprobs,
                submitted_at,
                validate_ms,
                reply,
            } => pending_submits.push(PendingSubmit {
                entry_id,
                input_text,
                sampling,
                stop_suffixes,
                token_logprobs,
                submitted_at,
                runtime_received_at: Instant::now(),
                validate_ms,
                reply,
            }),
            InferenceSubmitCommand::Cancel { entry_id } => {
                cancelled_entry_ids.push(entry_id);
            }
        }
    }

    fn finish_command_batch(
        &mut self,
        pending_submits: Vec<PendingSubmit>,
        cancelled_entry_ids: Vec<EntryId>,
    ) {
        if !cancelled_entry_ids.is_empty() {
            self.cancel_entries(&cancelled_entry_ids);
        }
        self.handle_submit_batch(pending_submits);
    }

    fn drain_pending_commands(&mut self) {
        let mut pending_submits = Vec::new();
        let mut cancelled_entry_ids = Vec::new();
        while let Ok(cmd) = self.rx.try_recv() {
            self.handle_command(cmd, &mut pending_submits, &mut cancelled_entry_ids);
        }
        self.finish_command_batch(pending_submits, cancelled_entry_ids);
    }

    async fn wait_for_commands(&mut self) -> bool {
        let mut pending_submits = Vec::new();
        let mut cancelled_entry_ids = Vec::new();
        let Some(cmd) = self.rx.recv().await else {
            return false;
        };
        self.handle_command(cmd, &mut pending_submits, &mut cancelled_entry_ids);
        while let Ok(cmd) = self.rx.try_recv() {
            self.handle_command(cmd, &mut pending_submits, &mut cancelled_entry_ids);
        }
        self.finish_command_batch(pending_submits, cancelled_entry_ids);
        true
    }

    fn has_ready_work(&self) -> bool {
        self.scheduler.has_ready_decode(&self.entries)
            || self.scheduler.has_ready_prefill(&self.entries)
            || (self.scheduler.has_free_slot() && self.scheduler.has_waiting_entries(&self.entries))
    }

    fn handle_submit_batch(&mut self, submits: Vec<PendingSubmit>) {
        rwkv_bench::trace_scope!("rwkv.infer.execution_loop.handle_submit_batch");
        if submits.is_empty() {
            return;
        }

        let (tokenized, tokenize_ms) = {
            rwkv_bench::trace_lite_scope!("rwkv.infer.execution_loop.tokenize_batch");
            let tokenize_start = Instant::now();
            let texts: Vec<String> = submits
                .iter()
                .map(|submit| submit.input_text.clone())
                .collect();
            let tokenized: Vec<Vec<u16>> = self.tokenizer.encode_batch(texts, false);
            let tokenize_ms = tokenize_start.elapsed().as_millis() as u64;
            (tokenized, tokenize_ms)
        };

        for (submit, token_ids_u16) in submits.into_iter().zip(tokenized.into_iter()) {
            rwkv_bench::trace_lite_scope!("rwkv.infer.execution_loop.register_request");
            let token_ids: Vec<i32> = token_ids_u16.into_iter().map(i32::from).collect();
            if token_ids.len() > self.cfg.max_context_len {
                let _ = submit.reply.send(InferenceSubmitResult::Error {
                    entry_id: submit.entry_id,
                    message: format!(
                        "prefill too long: {} tokens > max_context_len={}",
                        token_ids.len(),
                        self.cfg.max_context_len
                    ),
                });
                continue;
            }

            let stop_suffixes: Vec<String> = submit
                .stop_suffixes
                .into_iter()
                .filter(|suffix| !suffix.is_empty())
                .collect();

            let (output_tx, output_rx) = mpsc::channel(256);
            if self
                .tokenizer_tx
                .send(TokenizerCommand::Register {
                    entry_id: submit.entry_id,
                    output_tx,
                    stop_suffixes: stop_suffixes.clone(),
                })
                .is_err()
            {
                let _ = submit.reply.send(InferenceSubmitResult::Error {
                    entry_id: submit.entry_id,
                    message: "tokenizer loop is unavailable".to_string(),
                });
                continue;
            }

            let mut entry = RequestState::new(
                submit.entry_id,
                submit.sampling,
                stop_suffixes,
                submit.token_logprobs,
            );
            entry.input_token_ids = token_ids;
            entry.validate_ms = submit.validate_ms;
            entry.tokenize_ms = Some(tokenize_ms);
            entry.submitted_at = Some(submit.submitted_at);
            entry.runtime_received_at = Some(submit.runtime_received_at);
            entry.enqueued_at = Some(Instant::now());

            self.entries.insert(submit.entry_id, entry);
            self.scheduler.push_waiting(submit.entry_id);
            let _ = submit.reply.send(InferenceSubmitResult::Receiver {
                entry_id: submit.entry_id,
                rx: output_rx,
            });
        }
    }

    fn cancel_entries(&mut self, entry_ids: &[EntryId]) {
        for entry_id in entry_ids.iter().copied() {
            self.terminate_entry_with_error(
                entry_id,
                RequestPhase::Cancelled,
                "request cancelled".to_string(),
            );
        }
    }

    fn terminate_entry_with_error(
        &mut self,
        entry_id: EntryId,
        phase: RequestPhase,
        message: String,
    ) {
        let batch_index = match self.entries.get_mut(&entry_id) {
            Some(entry) => {
                entry.phase = phase;
                entry.batch_index.take()
            }
            None => return,
        };

        if let Some(batch_index) = batch_index {
            let _ = self.executor.reset(batch_index);
        }
        self.scheduler.on_done(entry_id);
        self.entries.remove(&entry_id);
        let _ = self
            .tokenizer_tx
            .send(TokenizerCommand::Error { entry_id, message });
    }

    fn tick_once(&mut self) -> bool {
        rwkv_bench::trace_scope!("rwkv.infer.execution_loop.tick");
        let step = self.scheduler.schedule(&mut self.entries);
        if !step.has_work() {
            return false;
        }

        if self.cfg.decode_first {
            self.run_decode(&step.decode_ids);
            self.run_prefill_without_output(&step.prefill_without_output_ids);
            self.run_prefill(&step.prefill_ids);
        } else {
            self.run_prefill_without_output(&step.prefill_without_output_ids);
            self.run_prefill(&step.prefill_ids);
            self.run_decode(&step.decode_ids);
        }

        true
    }

    fn run_prefill_without_output(&mut self, entry_ids: &[EntryId]) {
        rwkv_bench::trace_scope!("rwkv.infer.execution_loop.prefill_without_output");
        let batch = self.build_prefill_batch(entry_ids, false);
        if batch.is_empty() {
            return;
        }

        let result = self.forward_batch(&batch, false);
        if let Err(err) = result {
            let chain = err.format_chain();
            log::error!("prefill failed: {chain}");
            self.fail_entries(batch.entry_ids(), format!("prefill failed: {chain}"));
        }
    }

    fn run_prefill(&mut self, entry_ids: &[EntryId]) {
        rwkv_bench::trace_scope!("rwkv.infer.execution_loop.prefill_sample");
        let batch = self.build_prefill_batch(entry_ids, true);
        if batch.is_empty() {
            return;
        }

        match self.forward_batch(&batch, true) {
            Ok(sampled_tokens) => {
                self.apply_sampled_tokens(
                    sampled_tokens,
                    &batch.batch_to_entry,
                    SampleSource::Prefill,
                );
            }
            Err(err) => {
                let chain = err.format_chain();
                log::error!("prefill sample failed: {chain}");
                self.fail_entries(batch.entry_ids(), format!("prefill sample failed: {chain}"));
            }
        }
    }

    fn build_prefill_batch(&mut self, entry_ids: &[EntryId], need_sample: bool) -> ForwardBatch {
        rwkv_bench::trace_lite_scope!("rwkv.infer.execution_loop.build_prefill_batch");
        let mut batch = ForwardBatch::default();
        let paragraph_len = self.cfg.paragraph_len;

        for entry_id in entry_ids.iter().copied() {
            let Some(entry) = self.entries.get_mut(&entry_id) else {
                continue;
            };
            let Some(batch_index) = entry.batch_index else {
                continue;
            };

            let chunk_issued_at = Instant::now();
            if entry.first_prefill_at.is_none() {
                entry.first_prefill_at = Some(chunk_issued_at);
            }
            entry.last_prefill_at = Some(chunk_issued_at);

            let total_chunks = entry.total_prefill_chunks(paragraph_len);
            if total_chunks == 0 || entry.prefill_chunk_cursor >= total_chunks {
                entry.phase = RequestPhase::RunningDecode;
                continue;
            }

            let prefill_len = entry.input_token_ids.len();
            let pad_len = (paragraph_len - (prefill_len % paragraph_len)) % paragraph_len;
            entry.prefill_pad_len = pad_len;

            let start = entry.prefill_chunk_cursor * paragraph_len;
            let end = start + paragraph_len;
            let mut token_ids = Vec::with_capacity(paragraph_len);
            let mut context_mask = vec![1u8; paragraph_len];
            for position in start..end {
                if position < pad_len {
                    token_ids.push(PREFILL_PAD_TOKEN_ID);
                    context_mask[position - start] = 0;
                } else {
                    token_ids.push(entry.input_token_ids[position - pad_len]);
                }
            }
            entry.prefill_chunk_cursor += 1;

            batch.push(
                entry_id,
                batch_index,
                token_ids,
                context_mask,
                entry.sampling,
                need_sample.then(|| entry.token_logprobs.clone()).flatten(),
            );
        }

        batch
    }

    fn run_decode(&mut self, decode_ids: &[EntryId]) {
        rwkv_bench::trace_scope!("rwkv.infer.execution_loop.decode");
        if decode_ids.is_empty() {
            return;
        }

        let mut batch = ForwardBatch::default();
        for entry_id in decode_ids.iter().copied() {
            let Some(entry) = self.entries.get(&entry_id) else {
                continue;
            };
            let Some(batch_index) = entry.batch_index else {
                continue;
            };
            let last_token_id = entry.last_context_token_id().unwrap_or(END_TOKEN_ID);
            batch.push(
                entry_id,
                batch_index,
                vec![last_token_id],
                vec![1u8],
                entry.sampling,
                entry.token_logprobs.clone(),
            );
        }

        if batch.is_empty() {
            return;
        }

        match self.forward_batch(&batch, true) {
            Ok(sampled_tokens) => {
                self.apply_sampled_tokens(
                    sampled_tokens,
                    &batch.batch_to_entry,
                    SampleSource::Decode,
                );
            }
            Err(err) => {
                let chain = err.format_chain();
                log::error!("decode failed: {chain}");
                self.fail_entries(batch.entry_ids(), format!("decode failed: {chain}"));
            }
        }
    }

    fn forward_batch(
        &mut self,
        batch: &ForwardBatch,
        need_sample: bool,
    ) -> crate::Result<Vec<SampledToken>> {
        rwkv_bench::trace_lite_scope!("rwkv.infer.execution_loop.forward_batch");
        let contexts_ref: Vec<&[i32]> = batch.contexts.iter().map(Vec::as_slice).collect();
        let masks_ref: Vec<&[u8]> = batch.context_masks.iter().map(Vec::as_slice).collect();
        self.executor.forward(
            &batch.batch_ids,
            &contexts_ref,
            &masks_ref,
            &batch.sampling_configs,
            &batch.token_logprob_configs,
            need_sample,
        )
    }

    fn apply_sampled_tokens(
        &mut self,
        sampled_tokens: Vec<SampledToken>,
        batch_to_entry: &HashMap<usize, EntryId>,
        source: SampleSource,
    ) {
        rwkv_bench::trace_lite_scope!("rwkv.infer.execution_loop.apply_sampled_tokens");
        let mut seen_batch_indices = HashSet::new();
        for sampled_token in sampled_tokens {
            let batch_index = sampled_token.batch_index;
            let Some(entry_id) = batch_to_entry.get(&batch_index).copied() else {
                continue;
            };
            seen_batch_indices.insert(batch_index);

            let Some(outcome) = self.prepare_sampled_token_outcome(entry_id, sampled_token, source)
            else {
                continue;
            };
            self.emit_sampled_token_outcome(outcome);
        }

        let missing_entry_ids: Vec<EntryId> = batch_to_entry
            .iter()
            .filter_map(|(batch_index, entry_id)| {
                (!seen_batch_indices.contains(batch_index)).then_some(*entry_id)
            })
            .collect();
        if !missing_entry_ids.is_empty() {
            self.fail_entries(
                &missing_entry_ids,
                "executor returned incomplete sample output".to_string(),
            );
        }
    }

    fn prepare_sampled_token_outcome(
        &mut self,
        entry_id: EntryId,
        sampled_token: SampledToken,
        source: SampleSource,
    ) -> Option<SampleApplyOutcome> {
        rwkv_bench::trace_lite_scope!("rwkv.infer.execution_loop.prepare_sampled_token");
        let mut output_token = None;
        let mut finish_meta = None;
        let mut finished_batch_index = None;

        {
            let entry = self.entries.get_mut(&entry_id)?;

            if matches!(source, SampleSource::Decode) {
                let decode_at = Instant::now();
                if entry.first_decode_at.is_none() {
                    entry.first_decode_at = Some(decode_at);
                }
                entry.last_decode_at = Some(decode_at);
            }

            let stop_reason = if sampled_token.token_id == END_TOKEN_ID {
                Some(RequestStopReason::EndToken)
            } else {
                let token_bytes = self
                    .tokenizer
                    .token_bytes(sampled_token.token_id as u16)
                    .to_vec();
                let matched_stop = entry.on_output_token(sampled_token.token_id, &token_bytes);
                if entry.first_emit_at.is_none() {
                    entry.first_emit_at = Some(Instant::now());
                }
                output_token = Some(sampled_token_to_output(sampled_token));

                if let Some(matched) = matched_stop {
                    Some(RequestStopReason::StopSuffix(matched))
                } else if entry.hit_max_new_tokens() {
                    Some(RequestStopReason::Length)
                } else {
                    None
                }
            };

            if let Some(stop_reason) = stop_reason {
                finish_meta = Some(finish_metadata_for_entry(
                    entry,
                    stop_reason,
                    Instant::now(),
                ));
                entry.phase = RequestPhase::Done;
                finished_batch_index = entry.batch_index.take();
            } else if matches!(source, SampleSource::Prefill) {
                entry.phase = RequestPhase::RunningDecode;
            }
        }

        Some(SampleApplyOutcome {
            entry_id,
            output_token,
            finish_meta,
            finished_batch_index,
        })
    }

    fn emit_sampled_token_outcome(&mut self, outcome: SampleApplyOutcome) {
        rwkv_bench::trace_lite_scope!("rwkv.infer.execution_loop.emit_sampled_token");
        let SampleApplyOutcome {
            entry_id,
            output_token,
            finish_meta,
            finished_batch_index,
        } = outcome;

        if let Some(token) = output_token {
            let _ = self
                .tokenizer_tx
                .send(TokenizerCommand::OutputToken { entry_id, token });
        }

        if let Some(meta) = finish_meta {
            let _ = self.tokenizer_tx.send(TokenizerCommand::Finish {
                entry_id,
                finish_meta: meta.clone(),
            });

            if let Some(batch_index) = finished_batch_index {
                let _ = self.executor.reset(batch_index);
            }
            self.scheduler.on_done(entry_id);
            self.entries.remove(&entry_id);
        }
    }

    fn fail_entries(&mut self, entry_ids: &[EntryId], message: String) {
        for entry_id in entry_ids.iter().copied() {
            self.terminate_entry_with_error(entry_id, RequestPhase::Failed, message.clone());
        }
    }
}

fn sampled_token_to_output(sampled_token: SampledToken) -> OutputToken {
    let (logprob, top_logprobs) = match sampled_token.logprob {
        Some(logprob) => (
            Some(logprob.logprob),
            logprob
                .top_logprobs
                .into_iter()
                .map(|candidate| OutputTokenCandidate {
                    token_id: candidate.token_id,
                    logprob: candidate.logprob,
                })
                .collect(),
        ),
        None => (None, Vec::new()),
    };

    OutputToken {
        token_id: sampled_token.token_id,
        logprob,
        top_logprobs,
    }
}

fn finish_metadata_for_entry(
    entry: &RequestState,
    stop_reason: RequestStopReason,
    finished_at: Instant,
) -> FinishMetadata {
    let timings_ms = build_timing_breakdown(entry, finished_at);

    match stop_reason {
        RequestStopReason::EndToken => FinishMetadata {
            reason: FinishReason::Stop,
            matched_stop_suffix: None,
            matched_stop_suffix_index: None,
            max_new_tokens: entry.max_new_tokens,
            generated_tokens: entry.generated_token_count,
            timings_ms,
        },
        RequestStopReason::StopSuffix(matched) => FinishMetadata {
            reason: FinishReason::Stop,
            matched_stop_suffix: entry.matched_stop_suffix(matched.index),
            matched_stop_suffix_index: Some(matched.index),
            max_new_tokens: entry.max_new_tokens,
            generated_tokens: entry.generated_token_count,
            timings_ms,
        },
        RequestStopReason::Length => FinishMetadata {
            reason: FinishReason::Length,
            matched_stop_suffix: None,
            matched_stop_suffix_index: None,
            max_new_tokens: entry.max_new_tokens,
            generated_tokens: entry.generated_token_count,
            timings_ms,
        },
    }
}

fn build_timing_breakdown(entry: &RequestState, finished_at: Instant) -> Option<TimingBreakdownMs> {
    let timings = TimingBreakdownMs {
        validate_ms: entry.validate_ms,
        tokenize_ms: entry.tokenize_ms,
        queue_wait_ms: duration_ms_between(entry.submitted_at, entry.runtime_received_at),
        schedule_wait_ms: duration_ms_between(entry.enqueued_at, entry.scheduled_at),
        prefill_first_ms: duration_ms_between(entry.scheduled_at, entry.first_prefill_at),
        first_emit_ms: duration_ms_between(entry.first_prefill_at, entry.first_emit_at)
            .or_else(|| duration_ms_since(entry.first_prefill_at, finished_at)),
        prefill_total_ms: duration_ms_between(entry.first_prefill_at, entry.last_prefill_at)
            .or_else(|| duration_ms_between(entry.first_prefill_at, entry.first_decode_at))
            .or_else(|| duration_ms_since(entry.first_prefill_at, finished_at)),
        decode_total_ms: duration_ms_between(entry.first_decode_at, entry.last_decode_at)
            .or_else(|| duration_ms_since(entry.first_decode_at, finished_at)),
        request_total_ms: duration_ms_between(entry.submitted_at, Some(finished_at))
            .or_else(|| duration_ms_between(entry.runtime_received_at, Some(finished_at))),
    };

    if timings.validate_ms.is_some()
        || timings.tokenize_ms.is_some()
        || timings.queue_wait_ms.is_some()
        || timings.schedule_wait_ms.is_some()
        || timings.prefill_first_ms.is_some()
        || timings.first_emit_ms.is_some()
        || timings.prefill_total_ms.is_some()
        || timings.decode_total_ms.is_some()
        || timings.request_total_ms.is_some()
    {
        Some(timings)
    } else {
        None
    }
}

fn duration_ms_between(start: Option<Instant>, end: Option<Instant>) -> Option<u64> {
    let (Some(start), Some(end)) = (start, end) else {
        return None;
    };
    Some(end.saturating_duration_since(start).as_millis() as u64)
}

fn duration_ms_since(start: Option<Instant>, end: Instant) -> Option<u64> {
    let Some(start) = start else {
        return None;
    };
    Some(end.saturating_duration_since(start).as_millis() as u64)
}

#[cfg(test)]
mod tests {
    use super::{
        InferenceExecutionConfig, InferenceExecutionLoop, ModelForward, RequestStopReason,
        finish_metadata_for_entry,
    };
    use crate::inference_core::{
        END_TOKEN_ID, EngineEvent, FinishReason, InferenceSubmitResult, RequestState, SampledToken,
        SamplingConfig, TokenLogprobsConfig,
    };
    use rwkv_data::tokenizer::Tokenizer;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tokio::sync::mpsc;
    use tokio::time::timeout;

    const TEST_VOCAB_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/rwkv-lm/assets/rwkv_vocab_v20230424.txt"
    );

    #[derive(Clone, Default)]
    struct TestForwardState {
        forward_calls: Arc<Mutex<Vec<(bool, usize)>>>,
        reset_batch_indices: Arc<Mutex<Vec<usize>>>,
    }

    struct TestForward {
        state: TestForwardState,
        decode_token: i32,
    }

    impl TestForward {
        fn new(state: TestForwardState, decode_token: i32) -> Self {
            Self {
                state,
                decode_token,
            }
        }
    }

    impl ModelForward for TestForward {
        fn forward(
            &mut self,
            batch_ids: &[usize],
            _contexts: &[&[i32]],
            _masks: &[&[u8]],
            _samplings: &[SamplingConfig],
            _token_logprobs: &[Option<TokenLogprobsConfig>],
            need_sample: bool,
        ) -> crate::Result<Vec<SampledToken>> {
            self.state
                .forward_calls
                .lock()
                .expect("forward lock")
                .push((need_sample, batch_ids.len()));

            if need_sample {
                Ok(batch_ids
                    .iter()
                    .copied()
                    .map(|batch_index| SampledToken {
                        batch_index,
                        token_id: self.decode_token,
                        logprob: None,
                    })
                    .collect())
            } else {
                Ok(Vec::new())
            }
        }

        fn reset(&mut self, batch_index: usize) -> crate::Result<()> {
            self.state
                .reset_batch_indices
                .lock()
                .expect("reset lock")
                .push(batch_index);
            Ok(())
        }
    }

    fn test_config() -> InferenceExecutionConfig {
        InferenceExecutionConfig {
            tokenizer_vocab_path: TEST_VOCAB_PATH.to_string(),
            max_batch_size: 2,
            paragraph_len: 256,
            max_context_len: 4096,
            decode_first: true,
        }
    }

    #[test]
    fn finish_metadata_prefers_stop_over_length() {
        let mut entry = RequestState::new(
            uuid::Uuid::new_v4(),
            SamplingConfig {
                max_new_tokens: 1,
                ..SamplingConfig::default()
            },
            vec!["stop".to_string()],
            None,
        );
        entry.generated_token_count = 1;
        let meta = finish_metadata_for_entry(
            &entry,
            RequestStopReason::StopSuffix(crate::inference_core::StopMatch { index: 0, len: 4 }),
            std::time::Instant::now(),
        );
        assert_eq!(meta.reason, FinishReason::Stop);
        assert_eq!(meta.matched_stop_suffix.as_deref(), Some("stop"));
    }

    #[tokio::test]
    async fn submit_returns_receiver_and_collects_length_finish() {
        let state = TestForwardState::default();
        let handle = InferenceExecutionLoop::spawn(
            test_config(),
            Box::new(TestForward::new(state.clone(), 1)),
        )
        .expect("spawn runtime");

        let submit = handle
            .submit_text(
                "hello".to_string(),
                SamplingConfig {
                    max_new_tokens: 1,
                    ..SamplingConfig::default()
                },
                Vec::new(),
                None,
                None,
            )
            .await
            .expect("submit");

        let mut rx = match submit {
            InferenceSubmitResult::Receiver { rx, .. } => rx,
            other => panic!("unexpected submit result: {other:?}"),
        };

        let first = timeout(Duration::from_secs(1), rx.recv())
            .await
            .expect("first event timeout")
            .expect("first event missing");
        match first {
            EngineEvent::Output(delta) => {
                assert!(!delta.text.is_empty() || !delta.tokens.is_empty())
            }
            other => panic!("unexpected first event: {other:?}"),
        }

        let second = timeout(Duration::from_secs(1), rx.recv())
            .await
            .expect("done event timeout")
            .expect("done event missing");
        match second {
            EngineEvent::Done(meta) => {
                assert_eq!(meta.reason, FinishReason::Length);
                assert_eq!(meta.generated_tokens, 1);
            }
            other => panic!("unexpected second event: {other:?}"),
        }
    }

    #[tokio::test]
    async fn end_token_finishes_without_visible_output() {
        let state = TestForwardState::default();
        let handle = InferenceExecutionLoop::spawn(
            test_config(),
            Box::new(TestForward::new(state.clone(), END_TOKEN_ID)),
        )
        .expect("spawn runtime");

        let submit = handle
            .submit_text(
                "hello".to_string(),
                SamplingConfig {
                    max_new_tokens: 4,
                    ..SamplingConfig::default()
                },
                Vec::new(),
                None,
                None,
            )
            .await
            .expect("submit");

        let mut rx = match submit {
            InferenceSubmitResult::Receiver { rx, .. } => rx,
            other => panic!("unexpected submit result: {other:?}"),
        };

        let event = timeout(Duration::from_secs(1), rx.recv())
            .await
            .expect("done timeout")
            .expect("done missing");
        match event {
            EngineEvent::Done(meta) => {
                assert_eq!(meta.reason, FinishReason::Stop);
                assert_eq!(meta.generated_tokens, 0);
            }
            other => panic!("unexpected event: {other:?}"),
        }

        match timeout(Duration::from_millis(50), rx.recv()).await {
            Ok(None) | Err(_) => {}
            Ok(Some(other)) => panic!("unexpected trailing event: {other:?}"),
        }
        assert_eq!(
            *state.forward_calls.lock().expect("forward calls"),
            vec![(true, 1)]
        );
    }

    #[test]
    fn runtime_constructs_with_tokenizer_channel() {
        let state = TestForwardState::default();
        let (_submit_tx, submit_rx) = mpsc::channel(8);
        let (tokenizer_tx, _tokenizer_rx) = mpsc::unbounded_channel();
        let runtime = InferenceExecutionLoop::new(
            test_config(),
            Tokenizer::new(TEST_VOCAB_PATH).expect("tokenizer"),
            submit_rx,
            tokenizer_tx,
            Box::new(TestForward::new(state, 1)),
        );
        assert_eq!(runtime.cfg.max_batch_size, 2);
    }
}
