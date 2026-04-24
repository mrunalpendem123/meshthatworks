mod dashboard;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use clap::{Parser, Subcommand};
use mtw_api::{NodeStatus, ProxyConfig};
use mtw_engine::{
    ChatMessage, ChatRequest, InferenceEngine, MockEngine, SwiftLMEngine,
    swiftlm::SwiftLMOptions,
};

const DEFAULT_SWIFTLM_BINARY: &str =
    "/Users/mrunalpendem/Desktop/meshthatworks-deps/SwiftLM/.build/release/SwiftLM";
const DEFAULT_MODEL_DIR: &str =
    "/Users/mrunalpendem/Desktop/meshthatworks-deps/models/Qwen3-30B-A3B-4bit";
const DEFAULT_SWIFTLM_PORT: u16 = 9876;
const DEFAULT_PROXY_PORT: u16 = 9337;

#[derive(Parser)]
#[command(
    name = "mtw",
    version,
    about = "MeshThatWorks — run frontier MoE models across your devices"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the always-on mesh node: spawns SwiftLM, serves iroh mesh, and
    /// exposes an OpenAI-compatible HTTP proxy on port 9337.
    Serve {
        /// Path to the model directory (must contain config.json + safetensors).
        #[arg(long, default_value = DEFAULT_MODEL_DIR)]
        model: PathBuf,
        /// Path to the SwiftLM binary.
        #[arg(long, default_value = DEFAULT_SWIFTLM_BINARY)]
        swiftlm: PathBuf,
        /// Port SwiftLM listens on internally.
        #[arg(long, default_value_t = DEFAULT_SWIFTLM_PORT)]
        swiftlm_port: u16,
        /// Port our OpenAI-compatible proxy listens on (user-facing).
        #[arg(long, default_value_t = DEFAULT_PROXY_PORT)]
        proxy_port: u16,
        /// GPU memory limit for SwiftLM (MB).
        #[arg(long, default_value_t = 4096)]
        mem_limit: u32,
        /// Skip SwiftLM spawn; use MockEngine (for testing iroh mesh only).
        #[arg(long)]
        mock: bool,
        /// Attach to an already-running SwiftLM at this URL instead of spawning.
        #[arg(long, conflicts_with_all = &["mock", "swiftlm"])]
        attach: Option<String>,
    },
    /// Show local info and ping every paired peer.
    Status,
    /// Transport diagnostics — iroh echo listen/dial.
    #[command(subcommand)]
    Echo(EchoCmd),
    /// Print an invite code and wait for a peer to join.
    Pair,
    /// Join an existing mesh using an invite code.
    Join {
        /// Invite string printed by `mtw pair` on the other device.
        invite: String,
    },
    /// List peers saved at ~/.mtw/peers.json.
    Peers,
    /// Launch a terminal UI that shows this node's state and its peers.
    Dashboard {
        /// Base URL of the mtw-api proxy to query for /status.
        #[arg(long, default_value = "http://127.0.0.1:9337")]
        url: String,
        /// Refresh cadence in milliseconds.
        #[arg(long, default_value_t = 500)]
        tick_ms: u64,
        /// How often to re-ping peers, in seconds.
        #[arg(long, default_value_t = 10)]
        ping_every_s: u64,
    },
    /// Send a single chat completion to a running server (SwiftLM or mtw-api),
    /// or delegate to a paired peer over iroh (`mtw/infer/0`).
    Chat {
        /// Prompt text. Combine multiple words without quoting.
        #[arg(trailing_var_arg = true, required = true)]
        prompt: Vec<String>,
        /// Base URL of the local server (ignored when --peer is set).
        #[arg(long, default_value = "http://127.0.0.1:9337")]
        url: String,
        /// Max tokens to generate.
        #[arg(long, default_value_t = 120)]
        max_tokens: usize,
        /// Sampling temperature.
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,
        /// Optional path to the model directory for populating model info.
        #[arg(long)]
        model_dir: Option<PathBuf>,
        /// Ask a paired peer (by endpoint id) to answer instead of hitting
        /// the local server. Uses the `mtw/infer/0` iroh protocol.
        #[arg(long)]
        peer: Option<String>,
        /// Seconds to wait for the peer's reply before giving up.
        #[arg(long, default_value_t = 600)]
        peer_timeout: u64,
    },
}

#[derive(Subcommand)]
enum EchoCmd {
    Listen,
    Dial {
        target: String,
        #[arg(trailing_var_arg = true, required = true)]
        message: Vec<String>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // The dashboard is a full-screen TUI in raw mode; any writer that touches
    // stderr will paint over the frame. Route logs to a file instead. Users
    // can tail `/tmp/mtw-dashboard.log` (or set MTW_LOG_FILE) for details.
    // Other commands keep stderr logging.
    if matches!(cli.command, Command::Dashboard { .. }) {
        init_tracing_file()?;
    } else {
        init_tracing_stderr();
    }

    match cli.command {
        Command::Serve {
            model,
            swiftlm,
            swiftlm_port,
            proxy_port,
            mem_limit,
            mock,
            attach,
        } => {
            serve_cmd(ServeArgs {
                model,
                swiftlm,
                swiftlm_port,
                proxy_port,
                mem_limit,
                mock,
                attach,
            })
            .await
        }
        Command::Status => {
            let secret = mtw_core::identity::load_or_create()?;
            mtw_core::status::run(secret).await
        }
        Command::Echo(EchoCmd::Listen) => mtw_core::echo::listen().await,
        Command::Echo(EchoCmd::Dial { target, message }) => {
            let joined = message.join(" ");
            mtw_core::echo::dial(&target, &joined).await
        }
        Command::Pair => {
            let secret = mtw_core::identity::load_or_create()?;
            mtw_core::pair::pair(secret).await
        }
        Command::Join { invite } => {
            let secret = mtw_core::identity::load_or_create()?;
            mtw_core::pair::join(secret, &invite).await
        }
        Command::Peers => peers_cmd(),
        Command::Dashboard {
            url,
            tick_ms,
            ping_every_s,
        } => dashboard::run(dashboard::DashboardArgs {
            url,
            tick: std::time::Duration::from_millis(tick_ms),
            ping_every: std::time::Duration::from_secs(ping_every_s),
        })
        .await,
        Command::Chat {
            prompt,
            url,
            max_tokens,
            temperature,
            model_dir,
            peer,
            peer_timeout,
        } => {
            let joined = prompt.join(" ");
            if let Some(peer_id) = peer {
                chat_via_peer_cmd(&peer_id, joined, max_tokens, temperature, peer_timeout).await
            } else {
                chat_cmd(joined, url, max_tokens, temperature, model_dir).await
            }
        }
    }
}

struct ServeArgs {
    model: PathBuf,
    swiftlm: PathBuf,
    swiftlm_port: u16,
    proxy_port: u16,
    mem_limit: u32,
    mock: bool,
    attach: Option<String>,
}

async fn serve_cmd(args: ServeArgs) -> anyhow::Result<()> {
    // Preflight: fail fast if the ports we need are held by stale processes,
    // before spawning SwiftLM or binding iroh.
    preflight_port(args.proxy_port, "proxy")?;
    if args.attach.is_none() && !args.mock {
        preflight_port(args.swiftlm_port, "SwiftLM")?;
    }

    let secret = mtw_core::identity::load_or_create()?;

    // Build the engine according to the flags.
    let engine: Arc<dyn InferenceEngine> = if args.mock {
        println!("mtw serve: --mock set, using MockEngine (no real inference)");
        Arc::new(MockEngine::olmoe())
    } else if let Some(url) = args.attach.clone() {
        println!("mtw serve: attaching to SwiftLM at {url}");
        let engine = SwiftLMEngine::attach(&url, Some(&args.model))
            .await
            .with_context(|| format!("attach SwiftLM at {url}"))?;
        Arc::new(engine)
    } else {
        println!("mtw serve: spawning SwiftLM");
        println!("  binary: {}", args.swiftlm.display());
        println!("  model:  {}", args.model.display());
        println!("  port:   {}", args.swiftlm_port);
        let mut opts = SwiftLMOptions::new(&args.swiftlm, &args.model);
        opts.port = args.swiftlm_port;
        opts.mem_limit_mb = Some(args.mem_limit);
        opts.stream_experts = true;
        opts.ssd_prefetch = true;
        let engine = SwiftLMEngine::spawn(opts)
            .await
            .context("spawn SwiftLM")?;
        Arc::new(engine)
    };

    // Proxy forwards to whatever engine's URL is. For MockEngine there is
    // no upstream; we skip the proxy in that case.
    let upstream = if !args.mock {
        Some(match &args.attach {
            Some(url) => url.clone(),
            None => format!("http://127.0.0.1:{}", args.swiftlm_port),
        })
    } else {
        None
    };

    let started = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let endpoint_id_str = secret.public().to_string();
    let node_status = NodeStatus {
        endpoint_id: endpoint_id_str,
        proxy_url: format!("http://127.0.0.1:{}", args.proxy_port),
        upstream_url: upstream.clone().unwrap_or_default(),
        alpns: vec!["mtw/health/0".into(), "mtw/infer/0".into()],
        model: engine.model_info().clone(),
        started_at_unix: started,
        version: env!("CARGO_PKG_VERSION").into(),
    };

    // Run the iroh mesh + the HTTP proxy concurrently. First error wins OR
    // a shutdown signal aborts both and we drop the engine to propagate
    // `kill_on_drop` to the SwiftLM child.
    let mesh_engine = engine.clone();
    let mut mesh_handle =
        tokio::spawn(async move { mtw_core::serve::run(secret, mesh_engine).await });

    // We always start the proxy so /status and /healthz are available; even
    // in --mock mode with no upstream, /v1/* will simply 502, but the
    // dashboard and health checks still work.
    let status_for_proxy = node_status.clone();
    let proxy_cfg = Some(ProxyConfig {
        bind: format!("127.0.0.1:{}", args.proxy_port).parse().unwrap(),
        upstream: upstream.unwrap_or_else(|| "http://127.0.0.1:0".into()),
        model_label: Some(engine.model_info().name.clone()),
        status: status_for_proxy,
    });
    let mut proxy_handle = proxy_cfg.map(|cfg| tokio::spawn(async move { mtw_api::run(cfg).await }));

    let shutdown = async {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};
            let mut term = signal(SignalKind::terminate()).expect("install SIGTERM handler");
            let mut int = signal(SignalKind::interrupt()).expect("install SIGINT handler");
            tokio::select! {
                _ = term.recv() => "SIGTERM",
                _ = int.recv() => "SIGINT",
            }
        }
        #[cfg(not(unix))]
        {
            tokio::signal::ctrl_c().await.ok();
            "ctrl-c"
        }
    };

    let proxy_fut = async {
        match proxy_handle.as_mut() {
            Some(h) => h.await,
            None => std::future::pending().await,
        }
    };

    let result: anyhow::Result<()> = tokio::select! {
        res = &mut mesh_handle => match res {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(e.context("mesh serve")),
            Err(e) if e.is_cancelled() => Ok(()),
            Err(e) => Err(anyhow::anyhow!("mesh task: {e}")),
        },
        res = proxy_fut => match res {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(e.context("api proxy")),
            Err(e) if e.is_cancelled() => Ok(()),
            Err(e) => Err(anyhow::anyhow!("proxy task: {e}")),
        },
        sig = shutdown => {
            println!();
            println!("mtw serve: {sig} received, shutting down");
            Ok(())
        }
    };

    // Abort any still-running tasks. Do NOT re-await — tokio panics if you
    // poll a JoinHandle that already completed inside the select above.
    mesh_handle.abort();
    if let Some(h) = &proxy_handle {
        h.abort();
    }

    // Give the runtime a tick to propagate the abort, so each task's future
    // (and its Arc<dyn InferenceEngine> clone) drops.
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;

    // Only .await handles that weren't polled to completion by the select.
    if !mesh_handle.is_finished() {
        let _ = mesh_handle.await;
    }
    if let Some(h) = proxy_handle {
        if !h.is_finished() {
            let _ = h.await;
        }
    }

    // Only our `engine` Arc remains now; dropping it runs SwiftLMEngine::drop
    // which kills the child thanks to Command::kill_on_drop(true).
    drop(engine);
    // Grace period for the child to receive SIGKILL and exit.
    tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    result
}

fn preflight_port(port: u16, label: &str) -> anyhow::Result<()> {
    match std::net::TcpListener::bind(("127.0.0.1", port)) {
        Ok(l) => {
            drop(l);
            Ok(())
        }
        Err(e) => {
            anyhow::bail!(
                "port {port} ({label}) is already in use: {e}\n\n\
                 another process is listening there. free it with:\n\
                 \n\
                 \tkill -9 $(lsof -ti :{port})\n\
                 \n\
                 or kill all mtw/SwiftLM processes at once:\n\
                 \n\
                 \tpkill -9 -f 'mtw serve' 2>/dev/null; pkill -9 -f SwiftLM 2>/dev/null\n\
                 \n\
                 then retry."
            );
        }
    }
}

fn env_filter() -> tracing_subscriber::EnvFilter {
    tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"))
}

fn init_tracing_stderr() {
    tracing_subscriber::fmt()
        .with_env_filter(env_filter())
        .init();
}

fn init_tracing_file() -> anyhow::Result<()> {
    let log_path = std::env::var("MTW_LOG_FILE")
        .unwrap_or_else(|_| "/tmp/mtw-dashboard.log".into());
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("open log file {log_path}"))?;
    tracing_subscriber::fmt()
        .with_writer(file)
        .with_ansi(false)
        .with_env_filter(env_filter())
        .init();
    Ok(())
}

fn peers_cmd() -> anyhow::Result<()> {
    let list = mtw_core::peers::load()?;
    if list.peers.is_empty() {
        println!("no peers paired yet. run `mtw pair` on one device and `mtw join <invite>` on the other.");
        return Ok(());
    }
    println!("{} peer(s) in ~/.mtw/peers.json:", list.peers.len());
    for peer in &list.peers {
        println!("  {}  (paired at unix {})", peer.id, peer.paired_at);
    }
    Ok(())
}

async fn chat_via_peer_cmd(
    peer_id_str: &str,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    timeout_secs: u64,
) -> anyhow::Result<()> {
    use iroh::{Endpoint, EndpointId, endpoint::presets};
    use mtw_engine::{ChatMessage, ChatRequest};
    use std::time::Duration;

    let peer_id: EndpointId = peer_id_str
        .parse()
        .with_context(|| format!("parse peer id {peer_id_str:?}"))?;
    let secret = mtw_core::identity::load_or_create()?;
    let endpoint = Endpoint::builder(presets::N0)
        .secret_key(secret)
        .bind()
        .await
        .context("bind iroh endpoint")?;

    eprintln!("asking peer {peer_id} (iroh mtw/infer/0)");

    let req = ChatRequest {
        messages: vec![ChatMessage::user(prompt)],
        max_tokens: Some(max_tokens),
        temperature: Some(temperature),
    };
    let resp = mtw_core::infer::infer_on_peer(
        &endpoint,
        peer_id,
        req,
        Duration::from_secs(timeout_secs),
    )
    .await?;

    println!("{}", resp.content);
    eprintln!();
    eprintln!(
        "  [{} prompt + {} completion tokens in {}ms ≈ {:.2} tok/s via peer]",
        resp.prompt_tokens,
        resp.completion_tokens,
        resp.latency_ms,
        resp.completion_tokens as f64 / (resp.latency_ms as f64 / 1000.0).max(0.01),
    );
    endpoint.close().await;
    Ok(())
}

async fn chat_cmd(
    prompt: String,
    url: String,
    max_tokens: usize,
    temperature: f32,
    model_dir: Option<PathBuf>,
) -> anyhow::Result<()> {
    let engine = SwiftLMEngine::attach(&url, model_dir.as_deref()).await?;
    let info = engine.model_info();
    eprintln!("connected to {url}");
    eprintln!("model: {} (layers={}, hidden={})", info.name, info.num_layers, info.hidden_size);
    eprintln!();

    let req = ChatRequest {
        messages: vec![ChatMessage::user(prompt)],
        max_tokens: Some(max_tokens),
        temperature: Some(temperature),
    };
    let resp = engine.chat_complete(req).await?;

    println!("{}", resp.content);
    eprintln!();
    eprintln!(
        "  [{} prompt + {} completion tokens in {}ms ≈ {:.2} tok/s]",
        resp.prompt_tokens,
        resp.completion_tokens,
        resp.latency_ms,
        resp.completion_tokens as f64 / (resp.latency_ms as f64 / 1000.0).max(0.01),
    );
    Ok(())
}
