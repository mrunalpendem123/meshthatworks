mod catalog;
mod dashboard;
mod doctor;

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

const DEFAULT_SWIFTLM_PORT: u16 = 9876;
const DEFAULT_PROXY_PORT: u16 = 9337;

/// `~/.meshthatworks-deps` — where bootstrap.sh installs SwiftLM and downloads models.
pub fn default_deps_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".meshthatworks-deps")
}

pub fn default_swiftlm_binary() -> PathBuf {
    default_deps_dir().join("SwiftLM/.build/arm64-apple-macosx/release/SwiftLM")
}

/// Resolution order for the model directory:
///   1. ~/.mtw/active-model (set by the dashboard's Models tab)
///   2. fallback: ~/.meshthatworks-deps/models/OLMoE-1B-7B-0125-Instruct-4bit
pub fn default_model_dir() -> PathBuf {
    if let Ok(Some(p)) = mtw_core::active_model::load() {
        return p;
    }
    default_deps_dir().join("models/OLMoE-1B-7B-0125-Instruct-4bit")
}

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
        /// Defaults to `~/.meshthatworks-deps/models/OLMoE-1B-7B-0125-Instruct-4bit`.
        #[arg(long)]
        model: Option<PathBuf>,
        /// Path to the SwiftLM binary. Defaults to
        /// `~/.meshthatworks-deps/SwiftLM/.build/arm64-apple-macosx/release/SwiftLM`.
        #[arg(long)]
        swiftlm: Option<PathBuf>,
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
        /// Optional draft model directory for speculative decoding (e.g.
        /// Qwen3-0.6B-4bit). Pairs a tiny draft with the main model so the
        /// big model verifies N draft tokens in one parallel pass — typical
        /// 1.5–3× speedup when the draft accepts well, at ~400 MB extra RAM.
        #[arg(long)]
        draft_model: Option<PathBuf>,
        /// Number of draft tokens per speculation round (only when --draft-model is set).
        #[arg(long, default_value_t = 4)]
        num_draft_tokens: u32,
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
    /// Connectivity self-check: IPv6 reachability, public IPv4, CGNAT detection,
    /// NAT type (via STUN), and macOS firewall state. Predicts whether pairing
    /// will be direct or fall back to relay.
    Doctor,
    /// One-command launch: starts the engine in the background, waits for
    /// it to be ready, and opens the live dashboard. Ctrl-C stops both.
    Start {
        /// Path to the model directory.
        #[arg(long)]
        model: Option<PathBuf>,
        /// Path to the SwiftLM binary.
        #[arg(long)]
        swiftlm: Option<PathBuf>,
        /// GPU memory limit for SwiftLM (MB).
        #[arg(long, default_value_t = 4096)]
        mem_limit: u32,
    },
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

    // The dashboard is a full-screen TUI in raw mode, so its logs go to a file.
    // `mtw serve` is long-running and gets very chatty (iroh net_report, hyper,
    // etc.); when the user runs `mtw serve` and `mtw dashboard` in the same
    // terminal, serve's stderr paints over the dashboard. Send serve's logs to
    // a file too. Short-lived commands (pair/join/status/...) keep stderr.
    match cli.command {
        Command::Dashboard { .. } | Command::Start { .. } => {
            init_tracing_file("/tmp/mtw-dashboard.log")?;
        }
        Command::Serve { .. } => {
            init_tracing_file("/tmp/mtw-serve.log")?;
            eprintln!("[mtw] logs → /tmp/mtw-serve.log  (override with MTW_LOG_FILE=...)");
        }
        _ => init_tracing_stderr(),
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
            draft_model,
            num_draft_tokens,
        } => {
            serve_cmd(ServeArgs {
                model: model.unwrap_or_else(default_model_dir),
                swiftlm: swiftlm.unwrap_or_else(default_swiftlm_binary),
                swiftlm_port,
                proxy_port,
                mem_limit,
                mock,
                attach,
                draft_model,
                num_draft_tokens,
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
        Command::Doctor => doctor::run().await,
        Command::Start {
            model,
            swiftlm,
            mem_limit,
        } => {
            start_cmd(StartArgs {
                model: model.unwrap_or_else(default_model_dir),
                swiftlm: swiftlm.unwrap_or_else(default_swiftlm_binary),
                mem_limit,
            })
            .await
        }
        Command::Dashboard {
            url,
            tick_ms,
            ping_every_s,
        } => {
            // Auto-start the engine if it isn't already responding, and run
            // a supervisor that respawns it whenever the dashboard signals
            // (e.g. user picked a different model in the Models tab).
            let restart = Arc::new(tokio::sync::Notify::new());
            let shutdown = Arc::new(tokio::sync::Notify::new());

            let supervisor = if engine_is_up().await {
                None
            } else {
                eprintln!("[mtw dashboard] no engine running — auto-spawning one…");
                Some(tokio::spawn(supervise_engine(
                    restart.clone(),
                    shutdown.clone(),
                )))
            };

            let result = dashboard::run(dashboard::DashboardArgs {
                url,
                tick: std::time::Duration::from_millis(tick_ms),
                ping_every: std::time::Duration::from_secs(ping_every_s),
                engine_restart: Some(restart.clone()),
            })
            .await;

            shutdown.notify_one();
            if let Some(h) = supervisor {
                let _ = h.await;
            }
            result
        }
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
    draft_model: Option<PathBuf>,
    num_draft_tokens: u32,
}

async fn serve_cmd(args: ServeArgs) -> anyhow::Result<()> {
    // Preflight: fail fast if the ports we need are held by stale processes,
    // before spawning SwiftLM or binding iroh.
    preflight_port(args.proxy_port, "proxy")?;
    if args.attach.is_none() && !args.mock {
        preflight_port(args.swiftlm_port, "SwiftLM")?;
    }

    let secret = mtw_core::identity::load_or_create()?;

    // Build the engine according to the flags. Pull the SwiftLM request
    // counter and a LayerPeer handle out alongside the engine — the HTTP
    // proxy uses the counter to share with the engine's memory sampler;
    // mtw_core::serve uses the LayerPeer to expose `mtw/layer-forward/0`.
    let (engine, request_counter, layer_peer): (
        Arc<dyn InferenceEngine>,
        Option<std::sync::Arc<std::sync::atomic::AtomicU64>>,
        Option<Arc<dyn mtw_engine::LayerPeer>>,
    ) = if args.mock {
        println!("mtw serve: --mock set, using MockEngine (no real inference)");
        (Arc::new(MockEngine::olmoe()), None, None)
    } else if let Some(url) = args.attach.clone() {
        println!("mtw serve: attaching to SwiftLM at {url}");
        let engine = SwiftLMEngine::attach(&url, Some(&args.model))
            .await
            .with_context(|| format!("attach SwiftLM at {url}"))?;
        let counter = engine.request_counter();
        let arc: Arc<SwiftLMEngine> = Arc::new(engine);
        (
            arc.clone() as Arc<dyn InferenceEngine>,
            Some(counter),
            Some(arc as Arc<dyn mtw_engine::LayerPeer>),
        )
    } else {
        println!("mtw serve: spawning SwiftLM");
        println!("  binary: {}", args.swiftlm.display());
        println!("  model:  {}", args.model.display());
        println!("  port:   {}", args.swiftlm_port);
        if let Some(d) = &args.draft_model {
            println!("  draft:  {} ({} tokens/round)", d.display(), args.num_draft_tokens);
        }
        let mut opts = SwiftLMOptions::new(&args.swiftlm, &args.model);
        opts.port = args.swiftlm_port;
        opts.mem_limit_mb = Some(args.mem_limit);
        opts.stream_experts = true;
        opts.ssd_prefetch = true;
        if let Some(d) = args.draft_model.clone() {
            opts.draft_model_dir = Some(d);
            opts.extra_args.push("--num-draft-tokens".into());
            opts.extra_args.push(args.num_draft_tokens.to_string());
        }
        let engine = SwiftLMEngine::spawn(opts)
            .await
            .context("spawn SwiftLM")?;
        let counter = engine.request_counter();
        let arc: Arc<SwiftLMEngine> = Arc::new(engine);
        (
            arc.clone() as Arc<dyn InferenceEngine>,
            Some(counter),
            Some(arc as Arc<dyn mtw_engine::LayerPeer>),
        )
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
    let mesh_layer_peer = layer_peer.clone();
    let mut mesh_handle = tokio::spawn(async move {
        mtw_core::serve::run(secret, mesh_engine, mesh_layer_peer).await
    });

    // We always start the proxy so /status and /healthz are available; even
    // in --mock mode with no upstream, /v1/* will simply 502, but the
    // dashboard and health checks still work.
    let status_for_proxy = node_status.clone();
    let proxy_cfg = Some(ProxyConfig {
        bind: format!("127.0.0.1:{}", args.proxy_port).parse().unwrap(),
        upstream: upstream.unwrap_or_else(|| "http://127.0.0.1:0".into()),
        model_label: Some(engine.model_info().name.clone()),
        status: status_for_proxy,
        request_counter,
        trace_log_path: Some(
            std::env::var("MTW_SWIFTLM_LOG").unwrap_or_else(|_| "/tmp/mtw-swiftlm.log".into()),
        ),
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

struct StartArgs {
    model: PathBuf,
    swiftlm: PathBuf,
    mem_limit: u32,
}

/// Probe `localhost:9337/healthz` with a tight timeout — used to decide
/// whether the user already has a `mtw serve` running.
async fn engine_is_up() -> bool {
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(500))
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    client
        .get(format!("http://127.0.0.1:{}/healthz", DEFAULT_PROXY_PORT))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

/// Build a default `ServeArgs` from current persistent settings (active
/// model, default SwiftLM binary). Returns `None` if either prereq is
/// missing — caller should let the dashboard banner say so instead of
/// trying to spawn an engine that will fail.
fn default_serve_args_or_none() -> Option<ServeArgs> {
    let swiftlm = default_swiftlm_binary();
    let model = default_model_dir();
    if !swiftlm.is_file() {
        return None;
    }
    if !model.is_dir() || !model.join("config.json").is_file() {
        return None;
    }
    Some(ServeArgs {
        model,
        swiftlm,
        swiftlm_port: DEFAULT_SWIFTLM_PORT,
        proxy_port: DEFAULT_PROXY_PORT,
        mem_limit: 4096,
        mock: false,
        attach: None,
        draft_model: None,
        num_draft_tokens: 4,
    })
}

/// Long-running supervisor that owns a single engine task and re-spawns it
/// whenever a notification arrives. The notification path is what makes
/// "Enter on a new model in the dashboard" produce a live engine swap.
async fn supervise_engine(
    restart: Arc<tokio::sync::Notify>,
    shutdown: Arc<tokio::sync::Notify>,
) {
    loop {
        let args = match default_serve_args_or_none() {
            Some(a) => a,
            None => {
                // Wait for either a restart hint (user fixed the prereq +
                // signalled) or shutdown.
                tokio::select! {
                    _ = restart.notified() => continue,
                    _ = shutdown.notified() => return,
                }
            }
        };
        eprintln!(
            "[mtw] starting engine: model={}",
            args.model.display()
        );
        let mut task = tokio::spawn(serve_cmd(args));
        tokio::select! {
            // The engine task exited on its own (crash or normal stop).
            res = &mut task => {
                if let Ok(Err(e)) = res {
                    eprintln!("[mtw] engine exited with error: {e:#}");
                }
                // Wait for a restart hint or shutdown before respawning so
                // we don't tight-loop.
                tokio::select! {
                    _ = restart.notified() => continue,
                    _ = shutdown.notified() => return,
                }
            }
            _ = restart.notified() => {
                eprintln!("[mtw] active model changed — restarting engine");
                task.abort();
                let _ = task.await;
                // tiny pause so the SwiftLM child fully releases the port
                tokio::time::sleep(std::time::Duration::from_millis(750)).await;
                continue;
            }
            _ = shutdown.notified() => {
                task.abort();
                let _ = task.await;
                return;
            }
        }
    }
}

/// `mtw start` — one-command flow: spawn `mtw serve` in the background, wait
/// for it to be ready, then open the dashboard. Ctrl-C in the dashboard
/// brings both down.
async fn start_cmd(args: StartArgs) -> anyhow::Result<()> {
    if !args.swiftlm.is_file() {
        eprintln!("mtw start: SwiftLM binary not found at {}", args.swiftlm.display());
        eprintln!();
        eprintln!("Run  mtw doctor  for setup instructions, or pass --swiftlm <path>.");
        anyhow::bail!("missing SwiftLM");
    }
    if !args.model.is_dir() || !args.model.join("config.json").is_file() {
        eprintln!("mtw start: no model installed yet.");
        eprintln!();
        eprintln!("Open the dashboard and pick one from the Models tab:");
        eprintln!("    mtw dashboard");
        eprintln!();
        eprintln!("(or pass --model <path> if you already have one elsewhere.)");
        anyhow::bail!("missing model");
    }

    println!("mtw start: launching engine + dashboard…");
    println!("  model:  {}", args.model.display());
    println!("  engine: {}", args.swiftlm.display());
    println!();

    let serve_args = ServeArgs {
        model: args.model.clone(),
        swiftlm: args.swiftlm.clone(),
        swiftlm_port: DEFAULT_SWIFTLM_PORT,
        proxy_port: DEFAULT_PROXY_PORT,
        mem_limit: args.mem_limit,
        mock: false,
        attach: None,
        draft_model: None,
        num_draft_tokens: 4,
    };
    let serve_handle = tokio::spawn(serve_cmd(serve_args));

    let healthz = format!("http://127.0.0.1:{}/healthz", DEFAULT_PROXY_PORT);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
    println!("waiting for engine to come up (model load can take ~30s cold)…");
    loop {
        if serve_handle.is_finished() {
            return match serve_handle.await {
                Ok(Ok(())) => anyhow::bail!("mtw serve exited before becoming ready"),
                Ok(Err(e)) => Err(e.context("mtw serve failed during startup")),
                Err(je) => anyhow::bail!("serve task crashed: {je}"),
            };
        }
        if client
            .get(&healthz)
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
        {
            break;
        }
        if std::time::Instant::now() >= deadline {
            serve_handle.abort();
            anyhow::bail!("mtw serve did not become ready within 300s");
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }

    println!("✓ engine ready — opening dashboard (Ctrl-C to stop)");
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;

    // Hand the dashboard a Notify it can poke when the user picks a new
    // model. We tear down the boot serve_handle and switch to the
    // supervisor so the live-swap path is active.
    serve_handle.abort();
    let _ = serve_handle.await;
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let restart = Arc::new(tokio::sync::Notify::new());
    let shutdown = Arc::new(tokio::sync::Notify::new());
    let supervisor = tokio::spawn(supervise_engine(restart.clone(), shutdown.clone()));

    let dashboard_args = dashboard::DashboardArgs {
        url: format!("http://127.0.0.1:{}", DEFAULT_PROXY_PORT),
        tick: std::time::Duration::from_millis(500),
        ping_every: std::time::Duration::from_secs(10),
        engine_restart: Some(restart),
    };
    let dashboard_result = dashboard::run(dashboard_args).await;

    shutdown.notify_one();
    let _ = supervisor.await;
    dashboard_result
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

fn init_tracing_file(default_path: &str) -> anyhow::Result<()> {
    let log_path = std::env::var("MTW_LOG_FILE").unwrap_or_else(|_| default_path.into());
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
