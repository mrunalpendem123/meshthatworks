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

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::process::{Child, Command};
use tokio::task::JoinHandle;
use tokio::time::sleep;

use crate::{ActivationTensor, ChatRequest, ChatResponse, InferenceEngine, LayerPeer, ModelInfo};

const READY_POLL_INTERVAL: Duration = Duration::from_millis(500);
const DEFAULT_READY_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_CHAT_TIMEOUT: Duration = Duration::from_secs(900);
const MEM_SAMPLE_INTERVAL: Duration = Duration::from_millis(500);

/// Aborts a background task when dropped — used so the memory sampler dies
/// with its `SwiftLMEngine`.
struct AbortOnDrop(JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

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
    /// Bumped on every `chat_complete` and `/v1/layer-forward` call. The memory
    /// sampler reads this so RSS samples are correlated with request count —
    /// the OOM-after-N-requests pattern needs that correlation to be debuggable.
    request_count: Arc<AtomicU64>,
    /// Background task that polls the SwiftLM child's RSS and writes annotated
    /// samples to `MTW_SWIFTLM_LOG`. `None` for attached engines (we don't own
    /// the PID lifecycle). Aborted on drop.
    #[allow(dead_code)]
    sampler: Option<AbortOnDrop>,
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
        // Pipe SwiftLM's stdout/stderr into a log file so it's debuggable.
        // Override path with MTW_SWIFTLM_LOG.
        let log_path = std::env::var("MTW_SWIFTLM_LOG")
            .unwrap_or_else(|_| "/tmp/mtw-swiftlm.log".into());
        let stdout_log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .with_context(|| format!("open SwiftLM log {log_path}"))?;
        let stderr_log = stdout_log.try_clone()?;
        let child = Command::new(&opts.binary)
            .args(&args)
            .current_dir(&binary_dir)
            .kill_on_drop(true)
            .stdout(std::process::Stdio::from(stdout_log))
            .stderr(std::process::Stdio::from(stderr_log))
            .spawn()
            .with_context(|| format!("spawn {}", opts.binary.display()))?;

        let child_pid = child.id();
        let request_count = Arc::new(AtomicU64::new(0));
        let sampler = child_pid.map(|pid| {
            AbortOnDrop(tokio::spawn(memory_sampler(
                pid,
                log_path.clone(),
                Arc::clone(&request_count),
            )))
        });

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
            request_count,
            sampler,
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
            request_count: Arc::new(AtomicU64::new(0)),
            sampler: None,
        })
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Hand out a clone of the request counter so the HTTP proxy in `mtw-api`
    /// can bump it on every forwarded chat completion. Both increment paths
    /// converge on the same atomic, so the memory sampler's `requests=` field
    /// is consistent regardless of which path the client took.
    pub fn request_counter(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.request_count)
    }

    /// MeshThatWorks: send raw token IDs to SwiftLM's `/v1/layer-forward`.
    ///
    /// Used by a "first peer" in a layer-split pipeline — the peer that owns
    /// `embedTokens` and the model's first layers. The remote SwiftLM was
    /// launched with `MTW_LAYER_RANGE=0,K` so its `callAsFunctionPartial`
    /// embeds the tokens then runs layers `0..=K` and returns the post-layer
    /// activation `[batch, seq, hidden_size]`.
    pub async fn run_partial_tokens(
        &self,
        tokens: &[i32],
        shape: Vec<usize>,
    ) -> anyhow::Result<ActivationTensor> {
        let body = OutgoingLayerForward {
            input_kind: "tokens",
            shape: shape.clone(),
            data: tokens.iter().map(|&t| t as f64).collect(),
        };
        self.post_layer_forward(body).await
    }

    /// MeshThatWorks: forward an existing activation tensor through this
    /// peer's loaded slice via `/v1/layer-forward`.
    ///
    /// Used by middle and last peers — the input is the previous peer's
    /// output activation. If this is the last peer in the pipeline (loaded
    /// range ends at `hiddenLayers - 1`), the response is logits
    /// `[batch, seq, vocab_size]`; otherwise it's another activation
    /// `[batch, seq, hidden_size]` for the next peer.
    pub async fn run_partial_activation(
        &self,
        input: ActivationTensor,
    ) -> anyhow::Result<ActivationTensor> {
        if !input.is_well_formed() {
            anyhow::bail!(
                "activation data.len()={} but shape {:?} implies {}",
                input.data.len(),
                input.shape,
                input.expected_len()
            );
        }
        let body = OutgoingLayerForward {
            input_kind: "activation",
            shape: input.shape,
            data: input.data.into_iter().map(|f| f as f64).collect(),
        };
        self.post_layer_forward(body).await
    }

    async fn post_layer_forward(
        &self,
        body: OutgoingLayerForward<'_>,
    ) -> anyhow::Result<ActivationTensor> {
        self.request_count.fetch_add(1, Ordering::Relaxed);
        let resp = self
            .client
            .post(format!("{}/v1/layer-forward", self.base_url))
            .json(&body)
            .send()
            .await
            .context("POST /v1/layer-forward")?
            .error_for_status()
            .context("SwiftLM /v1/layer-forward returned error status")?
            .json::<IncomingLayerForward>()
            .await
            .context("parse /v1/layer-forward response")?;

        let shape: Vec<usize> = resp.shape.into_iter().map(|n| n as usize).collect();
        let expected: usize = shape.iter().product();
        if resp.data.len() != expected {
            anyhow::bail!(
                "remote sent shape {:?} ({} elems) but data has {}",
                shape,
                expected,
                resp.data.len()
            );
        }
        Ok(ActivationTensor {
            shape,
            data: resp.data,
        })
    }
}

#[async_trait]
impl LayerPeer for SwiftLMEngine {
    async fn run_partial_tokens(
        &self,
        tokens: &[i32],
        shape: Vec<usize>,
    ) -> anyhow::Result<ActivationTensor> {
        SwiftLMEngine::run_partial_tokens(self, tokens, shape).await
    }

    async fn run_partial_activation(
        &self,
        input: ActivationTensor,
    ) -> anyhow::Result<ActivationTensor> {
        SwiftLMEngine::run_partial_activation(self, input).await
    }
}

#[derive(Serialize)]
struct OutgoingLayerForward<'a> {
    input_kind: &'a str,
    shape: Vec<usize>,
    data: Vec<f64>,
}

#[derive(Deserialize)]
struct IncomingLayerForward {
    shape: Vec<i64>,
    data: Vec<f32>,
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

/// Sample SwiftLM child RSS every `MEM_SAMPLE_INTERVAL` and append a line per
/// sample to the SwiftLM log file. Each line carries the request count so an
/// "OOM after N requests" trace is reconstructable post-crash.
///
/// Format: `[mtw-mem] t_ms=<elapsed> rss_kb=<rss> requests=<n>`
///
/// Exits when `ps` reports the PID is gone (SwiftLM died) — no need for an
/// explicit shutdown signal; `AbortOnDrop` covers the engine-drop path.
async fn memory_sampler(pid: u32, log_path: String, request_count: Arc<AtomicU64>) {
    let t0 = Instant::now();
    let pid_str = pid.to_string();
    loop {
        sleep(MEM_SAMPLE_INTERVAL).await;

        let output = match Command::new("ps")
            .args(["-o", "rss=", "-p", &pid_str])
            .output()
            .await
        {
            Ok(o) => o,
            Err(err) => {
                tracing::debug!(%err, "memory_sampler: ps failed; stopping");
                return;
            }
        };
        if !output.status.success() {
            // PID is gone — SwiftLM exited or was killed.
            return;
        }
        let rss_kb: u64 = match std::str::from_utf8(&output.stdout)
            .ok()
            .and_then(|s| s.trim().parse().ok())
        {
            Some(n) => n,
            None => continue,
        };

        let line = format!(
            "[mtw-mem] t_ms={} rss_kb={} requests={}\n",
            t0.elapsed().as_millis(),
            rss_kb,
            request_count.load(Ordering::Relaxed),
        );
        // Best-effort append. If the log file vanished, the next loop drops it.
        if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(&log_path) {
            let _ = f.write_all(line.as_bytes());
        }
    }
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
        self.request_count.fetch_add(1, Ordering::Relaxed);
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
