# MeshThatWorks

Run frontier open-source MoE models across 2–3 consumer Apple Silicon devices, using each device's SSD as extended memory.

**Target**: Qwen3-Coder-30B-A3B on 2× 8GB Macs at 8–15 tok/s.

**Status**: Pre-alpha. Scaffolding only. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full technical specification.

## Workspace

| Crate | Purpose |
| --- | --- |
| `mtw-core` | Mesh coordination: peer discovery, shard assignment, routing. |
| `mtw-engine` | Per-node inference: MLX FFI, SSD expert streaming, KV cache. |
| `mtw-cache` | Usage-adaptive expert caching. |
| `mtw-api` | OpenAI-compatible HTTP API on `localhost:9337`. |
| `mtw-cli` | `mtw` binary entry point. |

## Build

```
cargo check --workspace
```

## End-to-end smoke test

`scripts/demo.sh` verifies the whole local stack is working: Metal toolchain, the patched SharpAI/mlx fork (symbols + metallib kernel), the Rust workspace (cargo tests + build), an iroh echo round-trip, and a real MLX inference against OLMoE-1B-7B-0125-Instruct. 15 checks, ~20 s on a warm machine.

```
./scripts/demo.sh
```

Prerequisites: OLMoE weights at `~/Desktop/meshthatworks-deps/models/OLMoE-1B-7B-0125-Instruct-4bit/`, the patched MLX fork built at `~/Desktop/meshthatworks-deps/mlx/build/libmlx.a`, and a Python venv at `~/Desktop/meshthatworks-deps/.venv` with `mlx-lm` installed. See `docs/ENGINE_RESEARCH.md` §0 for the MLX build fix.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
