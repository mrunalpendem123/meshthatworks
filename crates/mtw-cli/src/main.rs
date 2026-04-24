use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use mtw_engine::{ChatMessage, ChatRequest, InferenceEngine, MockEngine, SwiftLMEngine};

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
    /// Run the always-on mesh node.
    Serve,
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
    /// Send a single chat completion to a running SwiftLM server.
    Chat {
        /// Prompt text. Combine multiple words without quoting.
        #[arg(trailing_var_arg = true, required = true)]
        prompt: Vec<String>,
        /// Base URL of the SwiftLM server.
        #[arg(long, default_value = "http://127.0.0.1:9876")]
        url: String,
        /// Max tokens to generate.
        #[arg(long, default_value_t = 120)]
        max_tokens: usize,
        /// Sampling temperature.
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,
        /// Optional path to the model directory, used for populating model info.
        #[arg(long)]
        model_dir: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum EchoCmd {
    /// Listen for echo connections and print the endpoint id.
    Listen,
    /// Dial an echo listener and send a message.
    Dial {
        /// Endpoint id printed by `mtw echo listen`.
        target: String,
        /// Message to echo. Multiple words are joined with spaces.
        #[arg(trailing_var_arg = true, required = true)]
        message: Vec<String>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Serve => {
            let secret = mtw_core::identity::load_or_create()?;
            let engine: Arc<dyn InferenceEngine> = Arc::new(MockEngine::olmoe());
            mtw_core::serve::run(secret, engine).await
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
        Command::Chat {
            prompt,
            url,
            max_tokens,
            temperature,
            model_dir,
        } => chat_cmd(prompt.join(" "), url, max_tokens, temperature, model_dir).await,
    }
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

async fn chat_cmd(
    prompt: String,
    url: String,
    max_tokens: usize,
    temperature: f32,
    model_dir: Option<PathBuf>,
) -> anyhow::Result<()> {
    let engine = SwiftLMEngine::attach(&url, model_dir.as_deref()).await?;
    let info = engine.model_info();
    eprintln!("connected to SwiftLM at {url}");
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

