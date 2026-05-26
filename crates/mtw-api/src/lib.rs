//! OpenAI-compatible HTTP proxy on `localhost:9337`.
//!
//! Today: forwards every `/v1/*` request to a local SwiftLM instance. The
//! mesh layer adds no routing logic yet.
//!
//! Tomorrow: this is the layer where we plug in peer selection — "is this
//! request best served locally, or should we delegate to a peer over iroh?".
//! Keeping it as an HTTP entry point means any OpenAI client (Claude Code,
//! curl, Python `openai` SDK) can drive the mesh without being mesh-aware.

use std::io::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anyhow::Context;
use axum::{
    Json, Router,
    body::Body,
    extract::{Path, State},
    http::{HeaderMap, Method, StatusCode, Uri},
    response::IntoResponse,
    routing::{any, get},
};
use bytes::Bytes;
use futures_util::TryStreamExt;
use mtw_engine::ModelInfo;
use serde::{Deserialize, Serialize};

/// Lightweight snapshot of what a node is doing, served at `GET /status`.
/// Consumed by `mtw dashboard` and by any future health-monitoring UI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStatus {
    pub endpoint_id: String,
    /// Human-friendly node name (for the discovery list). Defaults to the host
    /// name; falls back to a short id when unset.
    #[serde(default)]
    pub name: String,
    pub proxy_url: String,
    pub upstream_url: String,
    pub alpns: Vec<String>,
    pub model: ModelInfo,
    pub started_at_unix: u64,
    pub version: String,
}

/// Configuration for the HTTP proxy.
#[derive(Clone)]
pub struct ProxyConfig {
    /// When set, `/v1/chat/completions` is answered by this engine (e.g. a
    /// `LayerSplitEngine` driving inference across paired devices over iroh)
    /// instead of being forwarded to a single upstream SwiftLM.
    pub chat_engine: Option<Arc<dyn mtw_engine::InferenceEngine>>,
    /// Live swarm-discovery registry, served at `GET /discovered`.
    pub discovered: Option<mtw_core::gossip::Discovered>,
    /// Address our proxy listens on (typically `127.0.0.1:9337`).
    pub bind: std::net::SocketAddr,
    /// Base URL of the upstream SwiftLM we forward to (typically `http://127.0.0.1:9876`).
    pub upstream: String,
    /// Optional banner printed at startup.
    pub model_label: Option<String>,
    /// Snapshot of node identity + model, served at `/status`.
    pub status: NodeStatus,
    /// Counter bumped on every `/v1/chat/completions` and `/v1/completions`
    /// request that we forward upstream. Shared with the engine's memory
    /// sampler so RSS samples are correlated with request boundaries — the
    /// "OOM after N requests" pattern needs this to be debuggable.
    pub request_counter: Option<Arc<AtomicU64>>,
    /// Path of the log file the engine sampler writes to. When set, the
    /// proxy emits `[mtw-req]` markers there too so the timeline is one
    /// stream. Defaults to `MTW_SWIFTLM_LOG` env var.
    pub trace_log_path: Option<String>,
}

#[derive(Clone)]
struct AppState {
    upstream: Arc<str>,
    client: reqwest::Client,
    status: Arc<NodeStatus>,
    request_counter: Option<Arc<AtomicU64>>,
    trace_log_path: Option<Arc<str>>,
    started: Instant,
    chat_engine: Option<Arc<dyn mtw_engine::InferenceEngine>>,
    discovered: Option<mtw_core::gossip::Discovered>,
}

pub async fn run(cfg: ProxyConfig) -> anyhow::Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(900))
        .build()?;
    let state = AppState {
        upstream: Arc::from(cfg.upstream.clone()),
        client,
        status: Arc::new(cfg.status),
        request_counter: cfg.request_counter,
        trace_log_path: cfg.trace_log_path.map(Arc::from),
        started: Instant::now(),
        chat_engine: cfg.chat_engine,
        discovered: cfg.discovered,
    };

    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/status", get(status_handler))
        .route("/discovered", get(discovered_handler))
        .route("/v1/models", any(proxy_root))
        .route("/v1/{*path}", any(proxy_passthrough))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(cfg.bind)
        .await
        .with_context(|| format!("bind {}", cfg.bind))?;

    println!("mtw-api: listening on http://{}", cfg.bind);
    println!("mtw-api: forwarding to {}", cfg.upstream);
    if let Some(label) = cfg.model_label {
        println!("mtw-api: model label for clients = {label}");
    }
    println!("mtw-api: endpoints: /v1/models, /v1/chat/completions, /v1/completions, /v1/embeddings");

    axum::serve(listener, app).await.context("axum::serve")?;
    Ok(())
}

async fn healthz() -> &'static str {
    "ok"
}

async fn status_handler(State(state): State<AppState>) -> Json<Arc<NodeStatus>> {
    Json(state.status.clone())
}

/// Live list of swarm nodes seen on the gossip room (for the discovery UI).
async fn discovered_handler(
    State(state): State<AppState>,
) -> Json<Vec<mtw_core::gossip::DiscoveredNode>> {
    match &state.discovered {
        Some(d) => Json(d.list().await),
        None => Json(Vec::new()),
    }
}

async fn proxy_root(
    State(state): State<AppState>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    proxy(state, method, uri, headers, body).await
}

async fn proxy_passthrough(
    Path(_rest): Path<String>,
    State(state): State<AppState>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    proxy(state, method, uri, headers, body).await
}

/// Answer `/v1/chat/completions` from a local engine, formatted as a minimal
/// SSE stream (one delta + `[DONE]`) so the existing client stream parser works.
/// Used in split mode where the engine pipelines a forward pass across peers.
async fn serve_chat_via_engine(
    engine: Arc<dyn mtw_engine::InferenceEngine>,
    body: &Bytes,
) -> axum::response::Response {
    #[derive(serde::Deserialize)]
    struct InMsg {
        role: String,
        content: String,
    }
    #[derive(serde::Deserialize)]
    struct InReq {
        messages: Vec<InMsg>,
        #[serde(default)]
        max_tokens: Option<usize>,
        #[serde(default)]
        temperature: Option<f32>,
    }
    let req: InReq = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, format!("bad chat request: {e}")).into_response(),
    };
    let chat_req = mtw_engine::ChatRequest {
        messages: req
            .messages
            .into_iter()
            .map(|m| mtw_engine::ChatMessage { role: m.role, content: m.content })
            .collect(),
        max_tokens: req.max_tokens,
        temperature: req.temperature,
    };
    let content = match engine.chat_complete(chat_req).await {
        Ok(resp) => resp.content,
        Err(e) => {
            let msg = format!("{e:#}").to_lowercase();
            if msg.contains("dns")
                || msg.contains("resolve")
                || msg.contains("no addressing")
                || msg.contains("timed out")
                || msg.contains("connect")
                || msg.contains("offline")
            {
                "⚠️ Split peer is offline or unreachable. The other Mac must be online, on the \
                 same model, and set to “Be worker” — then try again. (Or switch this node to \
                 Single mode to chat on its own.)"
                    .to_string()
            } else {
                format!("⚠️ Split error: {e:#}")
            }
        }
    };
    let chunk = serde_json::json!({ "choices": [{ "index": 0, "delta": { "content": content } }] });
    let sse = format!("data: {chunk}\n\ndata: [DONE]\n\n");
    ([(axum::http::header::CONTENT_TYPE, "text/event-stream")], sse).into_response()
}

async fn proxy(
    state: AppState,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> axum::response::Response {
    // Split mode: answer chat from the local engine (a LayerSplitEngine driving
    // inference across paired devices over iroh) instead of forwarding to one
    // upstream SwiftLM.
    if uri.path() == "/v1/chat/completions" {
        if let Some(engine) = state.chat_engine.clone() {
            return serve_chat_via_engine(engine, &body).await;
        }
    }
    let path_and_query = uri.path_and_query().map(|p| p.as_str()).unwrap_or(uri.path());
    let target_url = format!("{}{}", state.upstream, path_and_query);

    // Bump request counter and emit a [mtw-req] marker for inference paths so
    // the engine's memory sampler can correlate RSS deltas with request
    // boundaries. Models endpoint and status are too chatty to count.
    if matches!(
        uri.path(),
        "/v1/chat/completions" | "/v1/completions" | "/v1/layer-forward"
    ) {
        if let Some(c) = &state.request_counter {
            let n = c.fetch_add(1, Ordering::Relaxed) + 1;
            if let Some(p) = state.trace_log_path.as_ref() {
                let line = format!(
                    "[mtw-req] t_ms={} path={} requests={}\n",
                    state.started.elapsed().as_millis(),
                    uri.path(),
                    n,
                );
                if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(p.as_ref()) {
                    let _ = f.write_all(line.as_bytes());
                }
            }
        }
    }

    let mut builder = state.client.request(method.clone(), &target_url);
    for (name, value) in headers.iter() {
        // Hop-by-hop and axum-synthetic headers we don't want to forward.
        let n = name.as_str();
        if matches!(
            n,
            "host" | "connection" | "transfer-encoding" | "keep-alive"
                | "proxy-connection" | "upgrade" | "te" | "trailer" | "content-length"
        ) {
            continue;
        }
        builder = builder.header(name, value);
    }
    if !body.is_empty() {
        builder = builder.body(body);
    }

    let upstream = match builder.send().await {
        Ok(resp) => resp,
        Err(err) => {
            tracing::warn!(%err, target = %target_url, "proxy upstream error");
            return (
                StatusCode::BAD_GATEWAY,
                format!("upstream error: {err}"),
            )
                .into_response();
        }
    };

    let status = upstream.status();
    let mut out_headers = HeaderMap::new();
    for (name, value) in upstream.headers().iter() {
        let n = name.as_str();
        if matches!(
            n,
            "connection" | "transfer-encoding" | "keep-alive" | "content-length"
        ) {
            continue;
        }
        out_headers.insert(name, value.clone());
    }

    // Stream the body back. This preserves SSE for `stream: true` chat completions.
    let body_stream = upstream.bytes_stream().map_err(std::io::Error::other);
    let body = Body::from_stream(body_stream);

    let mut resp = axum::response::Response::builder().status(status);
    if let Some(h) = resp.headers_mut() {
        *h = out_headers;
    }
    resp.body(body)
        .unwrap_or_else(|_| (StatusCode::INTERNAL_SERVER_ERROR, "body build failed").into_response())
}
