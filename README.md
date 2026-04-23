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

## License

Apache-2.0. See [`LICENSE`](LICENSE).
