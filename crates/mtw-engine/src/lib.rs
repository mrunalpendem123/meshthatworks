//! Per-node inference engine for MeshThatWorks.
//!
//! Wraps the forked MLX/SwiftLM runtime: SSD expert streaming, KV cache,
//! and expert-activation profiling hooks. Exposes a small Rust surface
//! (`run_layer(activations) -> activations`) consumed by `mtw-core`.
