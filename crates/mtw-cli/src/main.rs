use clap::{Parser, Subcommand};

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
    /// Transport diagnostics — iroh echo listen/dial.
    #[command(subcommand)]
    Echo(EchoCmd),
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
        Command::Echo(EchoCmd::Listen) => mtw_core::echo::listen().await,
        Command::Echo(EchoCmd::Dial { target, message }) => {
            let joined = message.join(" ");
            mtw_core::echo::dial(&target, &joined).await
        }
    }
}
