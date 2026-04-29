//! End-to-end layer-split demo.
//!
//! Assumes you've started two SwiftLM processes ahead of time, each with a
//! complementary `MTW_LAYER_RANGE`. The example assembles a `LayerSplitEngine`
//! across them and runs a chat request.
//!
//! Bring-up:
//!
//!     # peer A — first half of the model
//!     MTW_LAYER_RANGE="0,23" \
//!       ~/Desktop/meshthatworks-deps/SwiftLM/.build/release/SwiftLM \
//!       --model ~/Desktop/meshthatworks-deps/models/Qwen3-30B-A3B-4bit \
//!       --port 9337 --stream-experts --ssd-prefetch --mem-limit 4096
//!
//!     # peer B — second half + final norm + lm_head
//!     MTW_LAYER_RANGE="24,47" \
//!       ~/Desktop/meshthatworks-deps/SwiftLM/.build/release/SwiftLM \
//!       --model ~/Desktop/meshthatworks-deps/models/Qwen3-30B-A3B-4bit \
//!       --port 9437 --stream-experts --ssd-prefetch --mem-limit 4096
//!
//!     # then in a third terminal
//!     cargo run -p mtw-engine --example layer_split_demo

use std::path::PathBuf;
use std::sync::Arc;

use mtw_engine::{
    ChatMessage, ChatRequest, InferenceEngine, LayerSplitEngine, SwiftLMEngine,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let peer_a_url = std::env::var("PEER_A_URL").unwrap_or_else(|_| "http://127.0.0.1:9337".into());
    let peer_b_url = std::env::var("PEER_B_URL").unwrap_or_else(|_| "http://127.0.0.1:9437".into());
    let model_dir = std::env::var("MODEL_DIR").unwrap_or_else(|_| {
        format!(
            "{}/Desktop/meshthatworks-deps/models/Qwen3-30B-A3B-4bit",
            std::env::var("HOME").unwrap_or_else(|_| ".".into())
        )
    });
    let model_dir = PathBuf::from(&model_dir);

    let prompt = std::env::args().nth(1).unwrap_or_else(|| {
        "In one short sentence, what is the capital of France?".to_string()
    });

    println!("→ peer A: {peer_a_url}");
    println!("→ peer B: {peer_b_url}");
    println!("→ model:  {}", model_dir.display());
    println!();

    println!("attaching to peers…");
    let peer_a = Arc::new(
        SwiftLMEngine::attach(peer_a_url.clone(), Some(&model_dir)).await?,
    );
    let peer_b = Arc::new(
        SwiftLMEngine::attach(peer_b_url.clone(), Some(&model_dir)).await?,
    );
    println!("✓ both peers attached");

    let engine = LayerSplitEngine::new(vec![peer_a, peer_b], &model_dir)?;
    let info = engine.model_info();
    println!(
        "✓ LayerSplitEngine ready: {} ({} layers · hidden {})",
        info.name, info.num_layers, info.hidden_size
    );
    println!();

    let req = ChatRequest {
        messages: vec![
            ChatMessage::system("You are a concise helpful assistant. Answer in one short sentence."),
            ChatMessage::user(&prompt),
        ],
        max_tokens: Some(20), // keep short — quadratic without KV cache
        temperature: None,
    };

    println!("prompt:   {}", prompt);
    println!("(generating up to {} tokens — without KV cache, expect a few seconds per token)", 20);
    println!();
    let started = std::time::Instant::now();
    let resp = engine.chat_complete(req).await?;
    let elapsed = started.elapsed();

    println!("=================================================================");
    println!("  LAYER-SPLIT INFERENCE ACROSS 2 PEERS");
    println!("=================================================================");
    println!("response: {}", resp.content);
    println!();
    println!(
        "  prompt_tokens={}  completion_tokens={}  total_elapsed={:.2}s  ≈ {:.2} tok/s",
        resp.prompt_tokens,
        resp.completion_tokens,
        elapsed.as_secs_f64(),
        resp.completion_tokens as f64 / elapsed.as_secs_f64().max(0.001),
    );
    println!();

    Ok(())
}
