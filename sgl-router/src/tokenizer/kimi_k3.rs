// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! A bounded pool of persistent Python encoders sharing the worker's K3 semantics.
//! Called only from spawn_blocking. Each process handles one request at a time;
//! a deadline covers both writing the request and reading its response. Failed
//! processes are killed/reaped and restarted in the background, never reused
//! with a pending response from a timed-out request.

use super::options::ChatEncodingOptions;
use crate::server::metrics::{EncoderMetrics, EncoderOutcome, EncoderQueueGuard};
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
    #[error("Kimi-K3 tokenizer deadline elapsed waiting for a process")]
    QueueTimeout,
    #[error("Kimi-K3 tokenizer response timed out: {0:#}")]
    Timeout(anyhow::Error),
    #[error("Kimi-K3 encoder unavailable: {0:#}")]
    Unavailable(#[from] anyhow::Error),
}

impl EncodeError {
    pub(crate) fn outcome(&self) -> EncoderOutcome {
        match self {
            Self::Request(_) => EncoderOutcome::RequestError,
            Self::Internal(_) => EncoderOutcome::InternalError,
            Self::QueueTimeout => EncoderOutcome::QueueTimeout,
            Self::Timeout(_) => EncoderOutcome::EncodeTimeout,
            Self::Unavailable(_) => EncoderOutcome::Unavailable,
        }
    }
}

#[derive(Deserialize)]
struct Reply {
    #[serde(default)]
    ready: bool,
    /// The encoder's rendering digest, in its ready reply.
    fingerprint: Option<String>,
    input_ids: Option<Vec<u32>>,
    error: Option<String>,
    error_type: Option<String>,
}

struct Process {
    /// The rendering digest workers must report to take this pool's IDs.
    fingerprint: String,
    child: Child,
    requests: Option<mpsc::Sender<Value>>,
    replies: mpsc::Receiver<Result<Reply>>,
    io_thread: Option<JoinHandle<()>>,
}

enum ExchangeError {
    BeforeDispatch,
    Timeout(anyhow::Error),
    Failed(anyhow::Error),
}

impl Process {
    fn start(
        source: &str,
        options: &ChatEncodingOptions,
        defaults: &Value,
        deadline: Instant,
    ) -> Result<Self> {
        ensure!(
            deadline > Instant::now(),
            "Kimi-K3 tokenizer startup deadline elapsed before process start"
        );
        let mut command = Command::new(&options.tokenizer_python);
        // Its own process group, so cleanup also reaches processes a wrapper
        // `--tokenizer-python` starts, which would otherwise hold the pipes open.
        #[cfg(unix)]
        std::os::unix::process::CommandExt::process_group(&mut command, 0);
        let mut child = command
            .args([
                "-u",
                "-m",
                "sglang.srt.entrypoints.openai.kimi_k3_tokenization",
            ])
            // The encoder exits when this process does, even mid-startup.
            .args(["--parent-pid", &std::process::id().to_string()])
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
            fingerprint: String::new(),
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
                    "default_chat_template_kwargs": defaults,
                }),
                deadline,
            )
            .map_err(|error| match error {
                ExchangeError::BeforeDispatch => {
                    anyhow::anyhow!("Kimi-K3 tokenizer startup deadline elapsed before dispatch")
                }
                ExchangeError::Timeout(error) | ExchangeError::Failed(error) => error,
            })?;
        if let Some(error) = reply.error {
            bail!("tokenizer startup failed: {error}");
        }
        ensure!(reply.ready, "tokenizer did not report ready");
        process.fingerprint = reply
            .fingerprint
            .filter(|fingerprint| !fingerprint.is_empty())
            .context("Kimi-K3 encoder reported no rendering fingerprint")?;
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
            return Err(ExchangeError::Timeout(anyhow::anyhow!(
                "Kimi-K3 tokenizer deadline elapsed after request dispatch"
            )));
        }
        let reply = self.replies.recv_timeout(remaining).map_err(|error| {
            let timed_out = matches!(error, mpsc::RecvTimeoutError::Timeout);
            let error =
                anyhow::Error::new(error).context("tokenizer response timeout or process exit");
            if timed_out {
                ExchangeError::Timeout(error)
            } else {
                ExchangeError::Failed(error)
            }
        })?;
        reply.map_err(ExchangeError::Failed)
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        // Killing before joining also unblocks a full stdin pipe or a stalled read.
        // The unreaped child keeps its PID, so its process group cannot be reused yet.
        #[cfg(unix)]
        let group = i32::try_from(self.child.id()).ok();
        #[cfg(unix)]
        if let Some(group) = group {
            // SAFETY: kill(2) only signals; `group` is the group created for this child.
            unsafe { libc::kill(-group, libc::SIGKILL) };
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
        #[cfg(unix)]
        if let Some(group) = group {
            reap_adopted(group);
        }
        self.requests.take();
        if let Some(thread) = self.io_thread.take() {
            let _ = thread.join();
        }
    }
}

/// Reap members of the killed `group` this process adopted. A wrapper's
/// children are reparented to the router when it is PID 1 or a subreaper, as in
/// a container without an init, and would otherwise stay zombies. Any adopted
/// members were reparented before the wrapper's own reap returned, and are
/// already SIGKILLed; elsewhere none are this process's children.
#[cfg(unix)]
fn reap_adopted(group: i32) {
    loop {
        // SAFETY: waitpid(2) on this process's own children in `group` only.
        let reaped = unsafe { libc::waitpid(-group, std::ptr::null_mut(), 0) };
        if reaped > 0
            || (reaped < 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::EINTR))
        {
            continue;
        }
        break;
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
    /// Every process's rendering digest; a replacement must report the same.
    fingerprint: String,
    options: ChatEncodingOptions,
    /// Worker `--default-chat-template-kwargs`, sent to every process at startup.
    defaults: Value,
    shared: Arc<PoolShared>,
    metrics: Arc<EncoderMetrics>,
}

impl KimiK3Pool {
    pub(crate) fn new_with_metrics(
        source: &str,
        options: &ChatEncodingOptions,
        defaults: Value,
        metrics: Arc<EncoderMetrics>,
    ) -> Result<Self> {
        let count = options
            .tokenizer_workers
            .map_or(1, std::num::NonZeroUsize::get);
        let deadline = Instant::now()
            .checked_add(Duration::from_secs(options.tokenizer_startup_timeout_secs))
            .context("Kimi-K3 tokenizer startup timeout is too large")?;
        // Each process loads the tokenizer, so they start together under one
        // deadline; on an error the others are dropped, which stops them.
        let processes = std::thread::scope(|scope| {
            let starts: Vec<_> = (0..count)
                .map(|_| scope.spawn(|| Process::start(source, options, &defaults, deadline)))
                .collect();
            starts
                .into_iter()
                .map(|start| {
                    start
                        .join()
                        .expect("Kimi-K3 tokenizer startup thread panicked")
                })
                .collect::<Result<Vec<_>>>()
        })?;
        let fingerprint = processes[0].fingerprint.clone();
        ensure!(
            processes
                .iter()
                .all(|process| process.fingerprint == fingerprint),
            "Kimi-K3 encoder processes rendered differently at startup"
        );
        Ok(Self {
            source: source.into(),
            fingerprint,
            options: options.clone(),
            defaults,
            metrics,
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

    pub(crate) fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub(crate) fn metrics(&self) -> Arc<EncoderMetrics> {
        Arc::clone(&self.metrics)
    }

    pub(crate) fn process_counts(&self) -> [usize; 4] {
        let state = self.shared.state.lock().expect("tokenizer pool lock");
        let mut counts = [0; 4];
        for slot in &state.slots {
            counts[match slot {
                ProcessSlot::Ready { .. } => 0,
                ProcessSlot::InUse => 1,
                ProcessSlot::Starting { .. } => 2,
                ProcessSlot::Empty { .. } => 3,
            }] += 1;
        }
        counts
    }

    pub(crate) fn encode_queued(
        &self,
        request: &Value,
        deadline: Instant,
        mut queue: EncoderQueueGuard,
    ) -> Result<Vec<u32>, EncodeError> {
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
                return Err(EncodeError::QueueTimeout);
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

        queue.start_encoding();
        let started = Instant::now();
        let exchange = process.exchange(request.clone(), deadline);
        self.metrics.observe_encoding(started);
        let (result, restart) = match exchange {
            Err(ExchangeError::BeforeDispatch) => (Err(EncodeError::QueueTimeout), false),
            Err(ExchangeError::Timeout(error)) => (Err(EncodeError::Timeout(error)), true),
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
        let defaults = self.defaults.clone();
        let metrics = Arc::clone(&self.metrics);
        let fingerprint = self.fingerprint.clone();
        let startup_timeout = Duration::from_secs(options.tokenizer_startup_timeout_secs);
        let worker = std::thread::Builder::new()
            .name("kimi-tokenizer-restart".into())
            .spawn(move || {
                let result = (|| {
                    let deadline = Instant::now()
                        .checked_add(startup_timeout)
                        .context("Kimi-K3 tokenizer startup timeout is too large")?;
                    let process = Process::start(&source, &options, &defaults, deadline)?;
                    // Workers were matched against the pool's digest.
                    ensure!(
                        process.fingerprint == fingerprint,
                        "the restarted Kimi-K3 encoder renders differently"
                    );
                    Ok(process)
                })();
                metrics.restart(result.is_ok());
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
            self.metrics.restart(false);
            let mut state = self.shared.state.lock().expect("tokenizer pool lock");
            state.slots[slot_index] = ProcessSlot::Empty {
                generation,
                error: format!("start tokenizer restart thread: {error}"),
            };
            self.shared.returned.notify_all();
        }
    }
}

/// What the Python encoder loads for `--tokenizer-path`: the directory of a
/// local tokenizer file, which also holds the `config.json` the encoder reads,
/// or the same HF repo id the Rust tokenizer resolves.
pub fn encoder_source(tokenizer_path: &str) -> String {
    let path = std::path::Path::new(tokenizer_path);
    match path.parent() {
        // A bare file name's parent is empty, which is the working directory.
        Some(dir) if path.is_file() && dir.as_os_str().is_empty() => ".".to_owned(),
        Some(dir) if path.is_file() => dir.to_string_lossy().into_owned(),
        _ => tokenizer_path.to_owned(),
    }
}

/// Request-supplied templates stay on the workers, which gate them behind
/// `--trust-request-chat-template`; serving reads them from `chat_template_kwargs`.
fn has_request_chat_template(request: &Value) -> bool {
    let is_set = |v: &Value| !v.is_null();
    request.get("chat_template").is_some_and(is_set)
        || request
            .get("chat_template_kwargs")
            .and_then(|kwargs| kwargs.get("chat_template"))
            .is_some_and(is_set)
}

/// Text and image Chat requests can share the worker's K3 prompt encoder.
/// Image URLs remain in messages for worker-side loading and placeholder
/// expansion. Audio, video and custom templates retain worker-side encoding.
pub fn is_supported_request(request: &Value) -> bool {
    ["input_ids", "kimi_k3_input_ids"]
        .iter()
        .all(|field| request.get(*field).is_none_or(Value::is_null))
        && !has_request_chat_template(request)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn chat(extra: Value) -> Value {
        let mut request = json!({"messages": [{"role": "user", "content": "hi"}]});
        request
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        request
    }

    #[test]
    fn request_chat_templates_bypass_the_encoder() {
        for extra in [
            json!({"chat_template": "{{ custom }}"}),
            json!({"chat_template_kwargs": {"chat_template": "{{ custom }}"}}),
            json!({"chat_template_kwargs": {"chat_template": "", "custom_flag": 1}}),
        ] {
            assert!(!is_supported_request(&chat(extra.clone())), "{extra}");
        }
        for extra in [
            json!({}),
            json!({"chat_template": null}),
            json!({"chat_template_kwargs": null}),
            json!({"chat_template_kwargs": {"chat_template": null}}),
            json!({"chat_template_kwargs": {"custom_flag": "retained"}}),
        ] {
            assert!(is_supported_request(&chat(extra.clone())), "{extra}");
        }
    }

    /// A pool of one fake encoder (`tests/fixtures/kimi_k3/fake_tokenizer.py`)
    /// answering `replies` in turn; with `wrapped`, started through a shell
    /// that does not exec, as a `--tokenizer-python` wrapper can be.
    fn fake_pool(dir: &std::path::Path, replies: &[&str], wrapped: bool) -> KimiK3Pool {
        let fixture = json!({"replies": replies, "restart_delay_millis": 0, "generated_ids": [7]});
        std::fs::write(dir.join("fixture.json"), fixture.to_string()).unwrap();
        let encoder = dir.join("fake-tokenizer");
        std::os::unix::fs::symlink(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/kimi_k3/fake_tokenizer.py"),
            &encoder,
        )
        .unwrap();
        let python = if wrapped {
            let wrapper = dir.join("wrapper.sh");
            let script = format!("#!/bin/sh\n{} \"$@\"\n", encoder.display());
            std::fs::write(&wrapper, script).unwrap();
            std::fs::set_permissions(
                &wrapper,
                std::os::unix::fs::PermissionsExt::from_mode(0o700),
            )
            .unwrap();
            wrapper
        } else {
            encoder
        };
        let options = ChatEncodingOptions {
            chat_encoder: super::super::options::ChatEncoderKind::KimiK3,
            tokenizer_python: python.to_str().unwrap().into(),
            ..Default::default()
        };
        KimiK3Pool::new_with_metrics("unused", &options, json!({}), EncoderMetrics::new("k3"))
            .unwrap()
    }

    /// A job whose deadline passed while it waited for a blocking thread must
    /// not take, and then retire, a healthy process.
    #[test]
    fn an_expired_deadline_leaves_the_process_ready() {
        let dir = tempfile::tempdir().unwrap();
        let pool = fake_pool(dir.path(), &[r#"{"input_ids": [7]}"#], false);
        let request = chat(json!({}));
        let expired = Instant::now() - Duration::from_millis(1);
        assert!(matches!(
            pool.encode_queued(&request, expired, pool.metrics().queue()),
            Err(EncodeError::QueueTimeout)
        ));
        assert_eq!(pool.process_counts(), [1, 0, 0, 0]);
        let deadline = Instant::now() + Duration::from_secs(10);
        let ids = pool.encode_queued(&request, deadline, pool.metrics().queue());
        assert_eq!(ids.unwrap(), vec![7]);
        let captured = std::fs::read_to_string(dir.path().join("requests.jsonl")).unwrap();
        assert_eq!(captured.lines().count(), 1);
    }

    /// A timed-out encoder started through a wrapper that does not exec must
    /// die with the wrapper: a survivor holds the pipes open, so the slot is
    /// never freed and every later request times out.
    #[test]
    fn a_timed_out_wrapped_encoder_is_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let pool = fake_pool(dir.path(), &["__stall__", r#"{"input_ids": [7]}"#], true);
        let (done, finished) = mpsc::channel();
        std::thread::spawn(move || {
            let request = chat(json!({}));
            let deadline = Instant::now() + Duration::from_millis(500);
            let first = pool.encode_queued(&request, deadline, pool.metrics().queue());
            let deadline = Instant::now() + Duration::from_secs(10);
            let second = pool.encode_queued(&request, deadline, pool.metrics().queue());
            let _ = done.send((
                first.map_err(|e| e.to_string()),
                second.map_err(|e| e.to_string()),
            ));
        });
        let (first, second) = finished
            .recv_timeout(Duration::from_secs(20))
            .expect("the timed-out encoder blocked its slot");
        assert!(first.unwrap_err().contains("timed out"));
        assert_eq!(second.unwrap(), vec![7]);
    }

    /// As PID 1 or a subreaper (a container without an init), the router adopts
    /// a killed wrapper's encoder and must reap it, or each timeout leaks a zombie.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_killed_wrapper_leaves_no_zombie_under_a_subreaper() {
        // SAFETY: prctl(2) only marks this test process as a subreaper.
        assert_eq!(unsafe { libc::prctl(libc::PR_SET_CHILD_SUBREAPER, 1) }, 0);
        let dir = tempfile::tempdir().unwrap();
        let pool = fake_pool(dir.path(), &["__stall__", r#"{"input_ids": [7]}"#], true);
        let group = match &pool.shared.state.lock().unwrap().slots[0] {
            ProcessSlot::Ready { process, .. } => process.child.id().to_string(),
            _ => unreachable!("the pool starts ready"),
        };
        let request = chat(json!({}));
        let deadline = Instant::now() + Duration::from_millis(500);
        assert!(pool
            .encode_queued(&request, deadline, pool.metrics().queue())
            .is_err());
        // Fields after the command name: state, ppid, pgrp.
        let members: Vec<String> = std::fs::read_dir("/proc")
            .unwrap()
            .filter_map(|entry| std::fs::read_to_string(entry.ok()?.path().join("stat")).ok())
            .filter_map(|stat| {
                let fields: Vec<&str> = stat.rsplit_once(')')?.1.split_whitespace().collect();
                (fields.get(2) == Some(&group.as_str())).then(|| stat.clone())
            })
            .collect();
        assert!(members.is_empty(), "group {group} left: {members:?}");
        // The replacement must be up before the fixture directory goes away.
        let deadline = Instant::now() + Duration::from_secs(10);
        let ids = pool.encode_queued(&request, deadline, pool.metrics().queue());
        assert_eq!(ids.unwrap(), vec![7]);
    }
}
