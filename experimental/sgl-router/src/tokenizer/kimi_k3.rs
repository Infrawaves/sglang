// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! A bounded pool of persistent Python encoders sharing the worker's K3 semantics.
//! Called only from spawn_blocking. Each process handles one request at a time;
//! a deadline covers both writing the request and reading its response. Failed
//! processes are killed/reaped and restarted in the background, never reused
//! with a pending response from a timed-out request.

use super::options::ChatEncodingOptions;
use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const MAX_RESPONSE_BYTES: u64 = 128 << 20;

#[derive(Debug, thiserror::Error)]
pub enum EncodeError {
    #[error("{0}")]
    Request(String),
    /// The worker returned a valid error response and remains usable.
    #[error("Kimi-K3 tokenizer reported an internal error: {0}")]
    Internal(String),
    #[error("Kimi-K3 encoder unavailable: {0:#}")]
    Unavailable(#[from] anyhow::Error),
}

#[derive(Deserialize)]
struct Reply {
    #[serde(default)]
    ready: bool,
    input_ids: Option<Vec<u32>>,
    error: Option<String>,
    error_type: Option<String>,
}

struct Process {
    child: Child,
    requests: Option<mpsc::Sender<Value>>,
    replies: mpsc::Receiver<Result<Reply>>,
    io_thread: Option<JoinHandle<()>>,
}

enum ExchangeError {
    BeforeDispatch,
    Failed(anyhow::Error),
}

impl Process {
    fn start(source: &str, options: &ChatEncodingOptions, deadline: Instant) -> Result<Self> {
        ensure!(
            deadline > Instant::now(),
            "Kimi-K3 tokenizer startup deadline elapsed before process start"
        );
        let mut child = Command::new(&options.tokenizer_python)
            .args([
                "-u",
                "-m",
                "sglang.srt.entrypoints.openai.kimi_k3_tokenization",
            ])
            .env("TOKENIZERS_PARALLELISM", "false")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("start Kimi-K3 Python tokenizer")?;
        let mut stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        let (requests, request_rx) = mpsc::channel::<Value>();
        let (reply_tx, replies) = mpsc::sync_channel(1);
        let mut process = Self {
            child,
            requests: Some(requests),
            replies,
            io_thread: None,
        };
        process.io_thread = Some(
            std::thread::Builder::new()
                .name("kimi-tokenizer-io".into())
                .spawn(move || {
                    let mut stdout = BufReader::new(stdout);
                    for request in request_rx {
                        let reply = (|| {
                            serde_json::to_writer(&mut stdin, &request)?;
                            stdin.write_all(b"\n")?;
                            stdin.flush()?;
                            let mut line = String::new();
                            let bytes = stdout
                                .by_ref()
                                .take(MAX_RESPONSE_BYTES + 1)
                                .read_line(&mut line)?;
                            ensure!(bytes > 0, "tokenizer process closed stdout");
                            ensure!(
                                bytes as u64 <= MAX_RESPONSE_BYTES && line.ends_with('\n'),
                                "invalid tokenizer response length"
                            );
                            serde_json::from_str(&line).context("invalid tokenizer response")
                        })();
                        let failed = reply.is_err();
                        if reply_tx.send(reply).is_err() || failed {
                            break;
                        }
                    }
                })
                .context("start tokenizer I/O thread")?,
        );
        let reply = process
            .exchange(
                json!({
                    "tokenizer_path": source,
                    "revision": options.tokenizer_revision,
                    "trust_remote_code": options.trust_remote_code,
                    "default_chat_template_kwargs": options.default_chat_template_kwargs,
                }),
                deadline,
            )
            .map_err(|error| match error {
                ExchangeError::BeforeDispatch => {
                    anyhow::anyhow!("Kimi-K3 tokenizer startup deadline elapsed before dispatch")
                }
                ExchangeError::Failed(error) => error,
            })?;
        if let Some(error) = reply.error {
            bail!("tokenizer startup failed: {error}");
        }
        ensure!(reply.ready, "tokenizer did not report ready");
        Ok(process)
    }

    fn exchange(
        &mut self,
        request: Value,
        deadline: Instant,
    ) -> std::result::Result<Reply, ExchangeError> {
        if deadline <= Instant::now() {
            return Err(ExchangeError::BeforeDispatch);
        }
        self.requests
            .as_ref()
            .context("tokenizer stopped")
            .map_err(ExchangeError::Failed)?
            .send(request)
            .map_err(|_| ExchangeError::Failed(anyhow::anyhow!("tokenizer I/O thread stopped")))?;
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(ExchangeError::Failed(anyhow::anyhow!(
                "Kimi-K3 tokenizer deadline elapsed after request dispatch"
            )));
        }
        let reply = self
            .replies
            .recv_timeout(remaining)
            .context("tokenizer response timeout or process exit")
            .map_err(ExchangeError::Failed)?;
        reply.map_err(ExchangeError::Failed)
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        // Killing before joining also unblocks a full stdin pipe or a stalled read.
        let _ = self.child.kill();
        let _ = self.child.wait();
        self.requests.take();
        if let Some(thread) = self.io_thread.take() {
            let _ = thread.join();
        }
    }
}

enum ProcessSlot {
    Ready { generation: u64, process: Process },
    InUse,
    Starting { generation: u64 },
    Empty { generation: u64, error: String },
}

struct PoolState {
    slots: Vec<ProcessSlot>,
}

struct PoolShared {
    state: Mutex<PoolState>,
    returned: Condvar,
}

pub struct KimiK3Pool {
    source: String,
    options: ChatEncodingOptions,
    shared: Arc<PoolShared>,
}

impl KimiK3Pool {
    pub fn new(source: &str, options: &ChatEncodingOptions) -> Result<Self> {
        let count = options
            .tokenizer_workers
            .map_or(1, std::num::NonZeroUsize::get);
        let startup_timeout = Duration::from_secs(options.tokenizer_startup_timeout_secs);
        let processes = (0..count)
            .map(|_| {
                let deadline = Instant::now()
                    .checked_add(startup_timeout)
                    .context("Kimi-K3 tokenizer startup timeout is too large")?;
                Process::start(source, options, deadline)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            source: source.into(),
            options: options.clone(),
            shared: Arc::new(PoolShared {
                state: Mutex::new(PoolState {
                    slots: processes
                        .into_iter()
                        .map(|process| ProcessSlot::Ready {
                            generation: 0,
                            process,
                        })
                        .collect(),
                }),
                returned: Condvar::new(),
            }),
        })
    }

    pub fn encode(&self, request: &Value, deadline: Instant) -> Result<Vec<u32>, EncodeError> {
        let slot_count = self
            .shared
            .state
            .lock()
            .expect("tokenizer pool lock")
            .slots
            .len();
        let mut attempted_restarts = vec![false; slot_count];
        let (slot_index, generation, mut process) = loop {
            let mut state = self.shared.state.lock().expect("tokenizer pool lock");

            // A request that expired while waiting for the blocking task to
            // start must not take and then retire an otherwise healthy process.
            if deadline <= Instant::now() {
                return Err(anyhow::anyhow!(
                    "Kimi-K3 tokenizer deadline elapsed waiting for a process"
                )
                .into());
            }

            if let Some(index) = state
                .slots
                .iter()
                .position(|slot| matches!(slot, ProcessSlot::Ready { .. }))
            {
                let old = std::mem::replace(&mut state.slots[index], ProcessSlot::InUse);
                let ProcessSlot::Ready {
                    generation,
                    process,
                } = old
                else {
                    unreachable!("selected ready tokenizer slot")
                };
                break (index, generation, process);
            }

            let mut restarts = Vec::new();
            for (index, slot) in state.slots.iter_mut().enumerate() {
                if attempted_restarts[index] {
                    continue;
                }
                if let ProcessSlot::Empty { generation, .. } = slot {
                    let next_generation = generation.wrapping_add(1);
                    *slot = ProcessSlot::Starting {
                        generation: next_generation,
                    };
                    attempted_restarts[index] = true;
                    restarts.push((index, next_generation));
                }
            }
            if !restarts.is_empty() {
                drop(state);
                for (index, generation) in restarts {
                    self.schedule_restart(index, generation);
                }
                continue;
            }

            for (index, slot) in state.slots.iter().enumerate() {
                if matches!(slot, ProcessSlot::Starting { .. }) {
                    attempted_restarts[index] = true;
                }
            }

            let restart_in_progress = state
                .slots
                .iter()
                .any(|slot| matches!(slot, ProcessSlot::InUse | ProcessSlot::Starting { .. }));
            if !restart_in_progress {
                let error = state
                    .slots
                    .iter()
                    .find_map(|slot| match slot {
                        ProcessSlot::Empty { error, .. } => Some(error.as_str()),
                        _ => None,
                    })
                    .unwrap_or("no tokenizer process is available");
                return Err(anyhow::anyhow!("Kimi-K3 tokenizer restart failed: {error}").into());
            }

            let remaining = deadline.saturating_duration_since(Instant::now());
            let (guard, _) = self
                .shared
                .returned
                .wait_timeout(state, remaining)
                .expect("tokenizer pool lock");
            drop(guard);
        };

        let exchange = process.exchange(request.clone(), deadline);
        let (result, restart) = match exchange {
            Err(ExchangeError::BeforeDispatch) => (
                Err(
                    anyhow::anyhow!("Kimi-K3 tokenizer deadline elapsed before request dispatch")
                        .into(),
                ),
                false,
            ),
            Err(ExchangeError::Failed(error)) => {
                // Once dispatched, a timed-out or malformed exchange may leave
                // a late reply in the stream, so this worker cannot be reused.
                (Err(EncodeError::Unavailable(error)), true)
            }
            Ok(reply) if reply.error.is_some() => {
                let error = reply.error.expect("checked tokenizer error response");
                if reply.error_type.as_deref() == Some("request") {
                    (Err(EncodeError::Request(error)), false)
                } else {
                    // A well-formed error line leaves the JSONL stream in sync.
                    (Err(EncodeError::Internal(error)), false)
                }
            }
            Ok(reply) => match reply.input_ids {
                None => (
                    Err(anyhow::anyhow!("tokenizer response missing input_ids").into()),
                    true,
                ),
                Some(ids) if ids.is_empty() => (
                    Err(anyhow::anyhow!("tokenizer produced empty input_ids").into()),
                    true,
                ),
                Some(ids) => (Ok(ids), false),
            },
        };

        if restart {
            let next_generation = generation.wrapping_add(1);
            drop(process);
            {
                let mut state = self.shared.state.lock().expect("tokenizer pool lock");
                state.slots[slot_index] = ProcessSlot::Starting {
                    generation: next_generation,
                };
            }
            self.schedule_restart(slot_index, next_generation);
        } else {
            let mut state = self.shared.state.lock().expect("tokenizer pool lock");
            state.slots[slot_index] = ProcessSlot::Ready {
                generation,
                process,
            };
            self.shared.returned.notify_all();
        }
        result
    }

    fn schedule_restart(&self, slot_index: usize, generation: u64) {
        let shared = Arc::clone(&self.shared);
        let source = self.source.clone();
        let options = self.options.clone();
        let startup_timeout = Duration::from_secs(options.tokenizer_startup_timeout_secs);
        let worker = std::thread::Builder::new()
            .name("kimi-tokenizer-restart".into())
            .spawn(move || {
                let result = (|| {
                    let deadline = Instant::now()
                        .checked_add(startup_timeout)
                        .context("Kimi-K3 tokenizer startup timeout is too large")?;
                    Process::start(&source, &options, deadline)
                })();
                let new_slot = match result {
                    Ok(process) => ProcessSlot::Ready {
                        generation,
                        process,
                    },
                    Err(error) => ProcessSlot::Empty {
                        generation,
                        error: format!("{error:#}"),
                    },
                };
                let mut state = shared.state.lock().expect("tokenizer pool lock");
                if matches!(
                    state.slots.get(slot_index),
                    Some(ProcessSlot::Starting { generation: current }) if *current == generation
                ) {
                    state.slots[slot_index] = new_slot;
                    shared.returned.notify_all();
                }
            });
        if let Err(error) = worker {
            let mut state = self.shared.state.lock().expect("tokenizer pool lock");
            state.slots[slot_index] = ProcessSlot::Empty {
                generation,
                error: format!("start tokenizer restart thread: {error}"),
            };
            self.shared.returned.notify_all();
        }
    }
}

/// Text and image Chat requests can share the worker's K3 prompt encoder.
/// Image URLs remain in messages for worker-side loading and placeholder
/// expansion. Audio, video and custom templates retain worker-side encoding.
pub fn is_supported_request(request: &Value) -> bool {
    request.get("input_ids").is_none_or(Value::is_null)
        && request.get("chat_template").is_none_or(Value::is_null)
        && request
            .get("messages")
            .and_then(Value::as_array)
            .is_some_and(|messages| {
                messages.iter().all(|message| {
                    message
                        .get("content")
                        .and_then(Value::as_array)
                        .is_none_or(|parts| {
                            parts.iter().all(|part| {
                                match part.get("type").and_then(Value::as_str) {
                                    Some("text") => true,
                                    Some("image_url") => part.get("modalities").is_none_or(|v| {
                                        v.is_null()
                                            || matches!(v.as_str(), Some("image" | "multi-images"))
                                    }),
                                    _ => false,
                                }
                            })
                        })
                })
            })
}

/// Media placeholders are not the scheduler's final IDs: workers expand them
/// and add media hashes. Never use these IDs for prefix lookup or exact length,
/// including when the IDs came from the caller.
pub fn requires_media_processing(request: &Value) -> bool {
    request
        .get("messages")
        .and_then(Value::as_array)
        .is_some_and(|messages| {
            messages.iter().any(|message| {
                message
                    .get("content")
                    .and_then(Value::as_array)
                    .is_some_and(|parts| {
                        parts
                            .iter()
                            .any(|part| part.get("type").and_then(Value::as_str) != Some("text"))
                    })
            })
        })
}
