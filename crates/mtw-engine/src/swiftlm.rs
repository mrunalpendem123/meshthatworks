//! Local SwiftLM process as an [`InferenceEngine`].
//!
//! `SwiftLMEngine` can either:
//! - **Spawn** a SwiftLM binary as a child process (`SwiftLMEngine::spawn`) —
//!   used when `mtw serve` owns the inference process lifecycle.
//! - **Attach** to an already-running SwiftLM (`SwiftLMEngine::attach`) — used
//!   when a SwiftLM server is managed externally.
//!
//! Either way, we talk to it over its OpenAI-compatible HTTP API at
//! `/v1/models` (readiness) and `/v1/chat/completions` (inference).

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::process::{Child, Command};
use tokio::time::sleep;

use crate::{ChatRequest, ChatResponse, InferenceEngine, ModelInfo};

const READY_POLL_INTERVAL: Duration = Duration::from_millis(500);
const DEFAULT_READY_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_CHAT_TIMEOUT: Duration = Duration::from_secs(900);

/// Options for spawning a SwiftLM child process.
#[derive(Debug, Clone)]
pub struct SwiftLMOptions {
    pub binary: PathBuf,
    pub model_dir: PathBuf,
    pub port: u16,
    pub mem_limit_mb: Option<u32>,
    pub stream_experts: bool,
    pub ssd_prefetch: bool,
    pub draft_model_dir: Option<PathBuf>,
    pub extra_args: Vec<String>,
}

impl SwiftLMOptions {
    pub fn new(binary: impl Into<PathBuf>, model_dir: impl Into<PathBuf>) -> Self {
        Self {
            binary: binary.into(),
            model_dir: model_dir.into(),
            port: 9876,
            mem_limit_mb: Some(4096),
            stream_experts: true,
            ssd_prefetch: true,
            draft_model_dir: None,
            extra_args: Vec::new(),
        }
    }

    fn build_args(&self) -> Vec<String> {
        let mut args = vec![
            "--model".into(),
            self.model_dir.display().to_string(),
            "--port".into(),
            self.port.to_string(),
        ];
        if self.stream_experts {
            args.push("--stream-experts".into());
        }
        if self.ssd_prefetch {
            args.push("--ssd-prefetch".into());
        }
        if let Some(mb) = self.mem_limit_mb {
            args.push("--mem-limit".into());
            args.push(mb.to_string());
        }
        if let Some(draft) = &self.draft_model_dir {
            args.push("--draft-model".into());
            args.push(draft.display().to_string());
        }
        args.extend(self.extra_args.clone());
        args
    }
}

pub struct SwiftLMEngine {
    base_url: String,
    model_info: ModelInfo,
    client: reqwest::Client,
    /// `Some` if we spawned the process; `None` if we only attached to an existing one.
    /// Held for its `Drop` side-effect: `Command::kill_on_drop(true)` ensures the
    /// child dies when `SwiftLMEngine` drops.
    #[allow(dead_code)]
    child: Option<Child>,
}

impl SwiftLMEngine {
    /// Spawn SwiftLM as a child process, wait for readiness, return the engine.
    /// The child is killed when this `SwiftLMEngine` is dropped.
    pub async fn spawn(opts: SwiftLMOptions) -> anyhow::Result<Self> {
        if !opts.binary.exists() {
            bail!("SwiftLM binary not found at {}", opts.binary.display());
        }
        if !opts.model_dir.exists() {
            bail!("model dir not found at {}", opts.model_dir.display());
        }

        let args = opts.build_args();
        let binary_dir = opts
            .binary
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();

        tracing::info!(binary = %opts.binary.display(), port = opts.port, "spawning SwiftLM");
        let child = Command::new(&opts.binary)
            .args(&args)
            .current_dir(&binary_dir)
            .kill_on_drop(true)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .with_context(|| format!("spawn {}", opts.binary.display()))?;

        let base_url = format!("http://127.0.0.1:{}", opts.port);
        let client = reqwest::Client::builder()
            .timeout(DEFAULT_CHAT_TIMEOUT)
            .build()?;

        wait_until_ready(&client, &base_url, DEFAULT_READY_TIMEOUT)
            .await
            .context("SwiftLM did not become ready within timeout")?;

        let model_info = fetch_model_info(&opts.model_dir).await.unwrap_or_else(|_| {
            // Fallback if we can't read the config — shouldn't break startup.
            ModelInfo {
                name: opts.model_dir.file_name().map(|s| s.to_string_lossy().into()).unwrap_or_default(),
                num_layers: 0,
                hidden_size: 0,
                num_experts: None,
                num_experts_per_tok: None,
            }
        });

        Ok(Self {
            base_url,
            model_info,
            client,
            child: Some(child),
        })
    }

    /// Attach to an already-running SwiftLM — we do not own its lifecycle.
    pub async fn attach(base_url: impl Into<String>, model_dir: Option<&Path>) -> anyhow::Result<Self> {
        let base_url = base_url.into();
        let client = reqwest::Client::builder()
            .timeout(DEFAULT_CHAT_TIMEOUT)
            .build()?;
        wait_until_ready(&client, &base_url, Duration::from_secs(5))
            .await
            .with_context(|| format!("SwiftLM not reachable at {base_url}"))?;

        let model_info = if let Some(dir) = model_dir {
            fetch_model_info(dir).await.unwrap_or_else(|_| placeholder_info("remote"))
        } else {
            placeholder_info("remote")
        };

        Ok(Self {
            base_url,
            model_info,
            client,
            child: None,
        })
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

fn placeholder_info(name: &str) -> ModelInfo {
    ModelInfo {
        name: name.into(),
        num_layers: 0,
        hidden_size: 0,
        num_experts: None,
        num_experts_per_tok: None,
    }
}

async fn fetch_model_info(model_dir: &Path) -> anyhow::Result<ModelInfo> {
    let config_path = model_dir.join("config.json");
    let text = tokio::fs::read_to_string(&config_path)
        .await
        .with_context(|| format!("read {}", config_path.display()))?;
    let cfg: serde_json::Value = serde_json::from_str(&text)?;
    let name = model_dir
        .file_name()
        .map(|s| s.to_string_lossy().into())
        .unwrap_or_else(|| "unknown".into());
    Ok(ModelInfo {
        name,
        num_layers: cfg.get("num_hidden_layers").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
        hidden_size: cfg.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
        num_experts: cfg.get("num_experts").and_then(|v| v.as_u64()).map(|v| v as usize),
        num_experts_per_tok: cfg
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize),
    })
}

async fn wait_until_ready(
    client: &reqwest::Client,
    base_url: &str,
    timeout: Duration,
) -> anyhow::Result<()> {
    let start = Instant::now();
    let url = format!("{base_url}/v1/models");
    loop {
        if client.get(&url).send().await.map(|r| r.status().is_success()).unwrap_or(false) {
            return Ok(());
        }
        if start.elapsed() >= timeout {
            bail!("timeout waiting for {url}");
        }
        sleep(READY_POLL_INTERVAL).await;
    }
}

#[derive(Serialize)]
struct OpenAIChatRequest<'a> {
    model: &'a str,
    messages: &'a [crate::ChatMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    stream: bool,
}

#[derive(Deserialize)]
struct OpenAIChatResponse {
    choices: Vec<OpenAIChoice>,
    usage: OpenAIUsage,
}

#[derive(Deserialize)]
struct OpenAIChoice {
    message: OpenAIChoiceMessage,
}

#[derive(Deserialize)]
struct OpenAIChoiceMessage {
    content: String,
}

#[derive(Deserialize)]
struct OpenAIUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
}

#[async_trait]
impl InferenceEngine for SwiftLMEngine {
    fn model_info(&self) -> &ModelInfo {
        &self.model_info
    }

    async fn chat_complete(&self, req: ChatRequest) -> anyhow::Result<ChatResponse> {
        let t0 = Instant::now();
        let body = OpenAIChatRequest {
            model: &self.model_info.name,
            messages: &req.messages,
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            stream: false,
        };
        let resp = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(&body)
            .send()
            .await
            .context("POST /v1/chat/completions")?
            .error_for_status()
            .context("SwiftLM returned error status")?
            .json::<OpenAIChatResponse>()
            .await
            .context("parse SwiftLM response")?;
        let choice = resp
            .choices
            .into_iter()
            .next()
            .context("SwiftLM returned no choices")?;
        Ok(ChatResponse {
            content: choice.message.content,
            prompt_tokens: resp.usage.prompt_tokens,
            completion_tokens: resp.usage.completion_tokens,
            latency_ms: t0.elapsed().as_millis(),
        })
    }
}
