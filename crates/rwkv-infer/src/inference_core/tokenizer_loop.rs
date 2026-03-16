use std::collections::HashMap;

use rwkv_data::tokenizer::Tokenizer;
use tokio::sync::mpsc;

use super::byte_decoder::ByteDecoder;
use super::{
    END_TOKEN_ID, EngineEvent, EntryId, FinishMetadata, InferenceOutput, InferenceOutputCandidate,
    InferenceSubmitCommand, OutputToken,
};

pub enum TokenizerCommand {
    Register {
        entry_id: EntryId,
        output_tx: mpsc::Sender<EngineEvent>,
        stop_suffixes: Vec<String>,
    },
    OutputToken {
        entry_id: EntryId,
        token: OutputToken,
    },
    Finish {
        entry_id: EntryId,
        finish_meta: FinishMetadata,
    },
    Error {
        entry_id: EntryId,
        message: String,
    },
}

struct TokenizerRequestState {
    output_tx: mpsc::Sender<EngineEvent>,
    byte_decoder: ByteDecoder,
}

pub struct TokenizerLoop {
    tokenizer: Tokenizer,
    rx: mpsc::UnboundedReceiver<TokenizerCommand>,
    engine_tx: mpsc::Sender<InferenceSubmitCommand>,
    requests: HashMap<EntryId, TokenizerRequestState>,
}

impl TokenizerLoop {
    pub fn new(
        tokenizer: Tokenizer,
        rx: mpsc::UnboundedReceiver<TokenizerCommand>,
        engine_tx: mpsc::Sender<InferenceSubmitCommand>,
    ) -> Self {
        Self {
            tokenizer,
            rx,
            engine_tx,
            requests: HashMap::new(),
        }
    }

    pub async fn run(mut self) {
        while let Some(cmd) = self.rx.recv().await {
            match cmd {
                TokenizerCommand::Register {
                    entry_id,
                    output_tx,
                    stop_suffixes,
                } => {
                    rwkv_bench::trace_lite_scope!("rwkv.infer.tokenizer_loop.register");
                    self.requests.insert(
                        entry_id,
                        TokenizerRequestState {
                            output_tx,
                            byte_decoder: ByteDecoder::new(stop_suffixes),
                        },
                    );
                }
                TokenizerCommand::OutputToken { entry_id, token } => {
                    if token.token_id == END_TOKEN_ID {
                        continue;
                    }
                    let Some(state) = self.requests.get_mut(&entry_id) else {
                        continue;
                    };
                    let delta = {
                        rwkv_bench::trace_lite_scope!("rwkv.infer.tokenizer_loop.output");
                        let output = detokenize_output(&self.tokenizer, token);
                        state.byte_decoder.push_output(output)
                    };
                    if has_stream_delta_output(&delta)
                        && state
                            .output_tx
                            .send(EngineEvent::Output(delta))
                            .await
                            .is_err()
                    {
                        self.requests.remove(&entry_id);
                        let _ = self
                            .engine_tx
                            .send(InferenceSubmitCommand::Cancel { entry_id })
                            .await;
                    }
                }
                TokenizerCommand::Finish {
                    entry_id,
                    finish_meta,
                } => {
                    let Some(mut state) = self.requests.remove(&entry_id) else {
                        continue;
                    };
                    let delta = {
                        rwkv_bench::trace_lite_scope!("rwkv.infer.tokenizer_loop.finish");
                        state
                            .byte_decoder
                            .finish(finish_meta.matched_stop_suffix_index)
                    };
                    if has_stream_delta_output(&delta)
                        && state
                            .output_tx
                            .send(EngineEvent::Output(delta))
                            .await
                            .is_err()
                    {
                        let _ = self
                            .engine_tx
                            .send(InferenceSubmitCommand::Cancel { entry_id })
                            .await;
                        continue;
                    }
                    let _ = state.output_tx.send(EngineEvent::Done(finish_meta)).await;
                }
                TokenizerCommand::Error { entry_id, message } => {
                    if let Some(state) = self.requests.remove(&entry_id) {
                        let _ = state.output_tx.send(EngineEvent::Error(message)).await;
                    }
                }
            }
        }
    }
}

fn has_stream_delta_output(delta: &super::StreamDelta) -> bool {
    !delta.text.is_empty() || !delta.tokens.is_empty()
}

fn detokenize_output(tokenizer: &Tokenizer, token: OutputToken) -> InferenceOutput {
    rwkv_bench::trace_lite_scope!("rwkv.infer.tokenizer_loop.detokenize");
    let bytes = tokenizer.token_bytes(token.token_id.max(0) as u16).to_vec();
    let token_text = String::from_utf8_lossy(&bytes).into_owned();
    let top_logprobs = token
        .top_logprobs
        .into_iter()
        .map(|candidate| {
            rwkv_bench::trace_lite_scope!("rwkv.infer.tokenizer_loop.detokenize.top_logprobs");
            let bytes = tokenizer
                .token_bytes(candidate.token_id.max(0) as u16)
                .to_vec();
            InferenceOutputCandidate {
                token: String::from_utf8_lossy(&bytes).into_owned(),
                bytes,
                logprob: candidate.logprob,
            }
        })
        .collect();
    InferenceOutput {
        token: token_text,
        bytes,
        logprob: token.logprob,
        top_logprobs,
    }
}
