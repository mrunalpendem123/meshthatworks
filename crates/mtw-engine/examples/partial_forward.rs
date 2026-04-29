//! End-to-end check of `SwiftLMEngine::run_partial_tokens` against a live
//! SwiftLM that has been launched with `MTW_LAYER_RANGE` set.
//!
//! Run flow (the example doesn't spawn SwiftLM — it expects you to have one
//! running already on `--swiftlm-url`):
//!
//!     # one terminal
//!     MTW_LAYER_RANGE="0,11" \
//!       ~/Desktop/meshthatworks-deps/SwiftLM/.build/release/SwiftLM \
//!       --model ~/Desktop/meshthatworks-deps/models/Qwen3-30B-A3B-4bit \
//!       --port 9999 --stream-experts --ssd-prefetch --mem-limit 4096
//!
//!     # another terminal
//!     cargo run -p mtw-engine --example partial_forward -- \
//!       --url http://127.0.0.1:9999

use std::time::Duration;

use mtw_engine::{ActivationTensor, SwiftLMEngine};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let url = std::env::args()
        .nth(2) // skip arg0 and "--url"
        .unwrap_or_else(|| "http://127.0.0.1:9999".to_string());

    println!("→ attaching to SwiftLM at {url}");
    let engine = SwiftLMEngine::attach(url, None).await?;
    println!("→ attached");

    // 4 toy token IDs. The actual values don't matter for proving the wire
    // path — we just need a well-formed shape SwiftLM will accept.
    let tokens: Vec<i32> = vec![1234, 5678, 910, 1112];
    println!("\n--- run_partial_tokens (first peer position) ---");
    println!("input: {} tokens, shape [1, {}]", tokens.len(), tokens.len());
    let started = std::time::Instant::now();
    let out = engine
        .run_partial_tokens(&tokens, vec![1, tokens.len()])
        .await?;
    let elapsed = started.elapsed();
    println!("output shape: {:?}", out.shape);
    println!("output data len: {}", out.data.len());
    println!(
        "first 5 values: {:?}",
        &out.data[..out.data.len().min(5)]
    );
    println!("elapsed: {:?}", elapsed);

    // Round-trip: feed the output back through as an activation. If this
    // SwiftLM is configured as a "first peer" (range 0..K, K<last layer), it
    // wouldn't accept activation input — but a "middle peer" or "last peer"
    // setup would. We try it anyway and just report whether it succeeds.
    println!("\n--- run_partial_activation (echo same tensor back) ---");
    let started = std::time::Instant::now();
    match engine.run_partial_activation(out.clone()).await {
        Ok(echoed) => {
            println!("output shape: {:?}", echoed.shape);
            println!("output data len: {}", echoed.data.len());
            println!("elapsed: {:?}", started.elapsed());
            // Sanity: `partialCallAsFunction` always runs the loaded layers,
            // so feeding an activation back will run another N-layer pass.
            // Useful for confirming the activation-input path works.
        }
        Err(e) => {
            println!("(activation input rejected — expected for first-peer setup) {e}");
        }
    }

    let _ = ActivationTensor::zeros(vec![1]); // silence unused-import lint
    let _ = Duration::from_secs(0);
    Ok(())
}
