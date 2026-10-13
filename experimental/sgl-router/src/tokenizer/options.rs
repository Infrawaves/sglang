// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use clap::{Args, ValueEnum};
use serde_json::{Map, Value};
use std::num::NonZeroUsize;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub enum ChatEncoderKind {
    /// Existing HF/Jinja encoder. Kimi-K3 model names select the Python encoder.
    #[default]
    Auto,
    /// Use the shared SGLang Kimi-K3 encoder in persistent Python processes.
    KimiK3,
}

/// K3 settings must match the serving workers' tokenizer revision and defaults.
#[derive(Args, Debug, Clone)]
pub struct ChatEncodingOptions {
    #[arg(long, value_enum, default_value = "auto")]
    pub chat_encoder: ChatEncoderKind,

    /// Python executable in the SGLang environment (Kimi-K3 only).
    #[arg(long, default_value = "python3")]
    pub tokenizer_python: String,

    /// Load the same tokenizer snapshot as the P/D workers (Kimi-K3 only).
    #[arg(long)]
    pub tokenizer_revision: Option<String>,

    /// Allow the Kimi-K3 tokenizer's custom Python code, as on the P/D workers.
    #[arg(long)]
    pub trust_remote_code: bool,

    /// Match the K3 workers' --default-chat-template-kwargs JSON object.
    #[arg(long, default_value = "{}", value_parser = parse_json_object)]
    pub default_chat_template_kwargs: Map<String, Value>,

    /// Concurrent prompt preparation jobs; K3 defaults to one Python process.
    #[arg(long)]
    pub tokenizer_workers: Option<NonZeroUsize>,

    /// Per-request time limit for Kimi-K3 queueing and encoding.
    #[arg(long, default_value_t = 300, value_parser = clap::value_parser!(u64).range(1..))]
    pub tokenizer_timeout_secs: u64,

    /// Time limit for initial and background Kimi-K3 Python process startup.
    #[arg(long, default_value_t = 300, value_parser = clap::value_parser!(u64).range(1..))]
    pub tokenizer_startup_timeout_secs: u64,
}

impl Default for ChatEncodingOptions {
    fn default() -> Self {
        Self {
            chat_encoder: ChatEncoderKind::Auto,
            tokenizer_python: "python3".into(),
            tokenizer_revision: None,
            trust_remote_code: false,
            default_chat_template_kwargs: Map::new(),
            tokenizer_workers: None,
            tokenizer_timeout_secs: 300,
            tokenizer_startup_timeout_secs: 300,
        }
    }
}

impl ChatEncodingOptions {
    pub fn is_kimi_k3(&self, model_id: &str) -> bool {
        let id = model_id.to_ascii_lowercase().replace('_', "-");
        self.chat_encoder == ChatEncoderKind::KimiK3 || id.contains("kimi-k3")
    }

    /// K3-only flags set to a non-default value. Other encoders ignore these,
    /// so reject them there instead of accepting settings that have no effect.
    /// A flag passed explicitly with its default value cannot be detected.
    pub fn non_default_kimi_only_flags(&self) -> Vec<&'static str> {
        let defaults = Self::default();
        [
            (
                self.tokenizer_python != defaults.tokenizer_python,
                "--tokenizer-python",
            ),
            (self.tokenizer_revision.is_some(), "--tokenizer-revision"),
            (self.trust_remote_code, "--trust-remote-code"),
            (
                !self.default_chat_template_kwargs.is_empty(),
                "--default-chat-template-kwargs",
            ),
            (self.tokenizer_workers.is_some(), "--tokenizer-workers"),
            (
                self.tokenizer_timeout_secs != defaults.tokenizer_timeout_secs,
                "--tokenizer-timeout-secs",
            ),
            (
                self.tokenizer_startup_timeout_secs != defaults.tokenizer_startup_timeout_secs,
                "--tokenizer-startup-timeout-secs",
            ),
        ]
        .into_iter()
        .filter_map(|(set, flag)| set.then_some(flag))
        .collect()
    }
}

fn parse_json_object(raw: &str) -> Result<Map<String, Value>, String> {
    serde_json::from_str(raw)
        .map_err(|error| format!("expected a JSON object for chat template defaults: {error}"))
}
