# MeshThatWorks

Run frontier open-source MoE models across 2–3 consumer Apple Silicon devices, using each device's SSD as extended memory.

**Target**: Qwen3-Coder-30B-A3B on 2× 8 GB Macs at 8–15 tok/s.

**Status**: SSD streaming proven on 8 GB Mac with Qwen3-30B-A3B-4bit (RSS oscillates 30–906 MB on an 18 GB model). Single-node + dashboard work end-to-end. Cross-machine layer-split scaffolding lands via `mtw/layer-forward/0`. Sustained throughput at low mem-limit is bandwidth-bound; throughput tuning + adaptive caching are the next milestones. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the spec and [`docs/BASELINES.md`](docs/BASELINES.md) for measured numbers.

## Workspace

| Crate | Purpose |
| --- | --- |
| `mtw-core` | Mesh: peer discovery, ALPNs (`mtw/health/0`, `mtw/infer/0`, `mtw/layer/0`, `mtw/layer-forward/0`), pairing. |
| `mtw-engine` | Per-node inference: SwiftLM child-process driver, `LayerPeer` trait, `LayerSplitEngine` orchestrator. |
| `mtw-cache` | Rolling expert-activation histogram and hot/warm/cold tiering. |
| `mtw-api` | OpenAI-compatible HTTP proxy on `localhost:9337`. |
| `mtw-cli` | `mtw` binary: `serve`, `pair`, `join`, `status`, `chat`, `dashboard`, `doctor`. |

## Quickstart

```
make install      # build + install mtw to ~/.local/bin
mtw doctor        # tells you what's still missing (Xcode Metal, SwiftLM, model)
mtw serve         # runs the node: spawns SwiftLM, exposes :9337, mesh over iroh
mtw dashboard     # live TUI: chat, peers, model hub
```

Two devices? Pair them:

```
# device A
mtw pair                              # prints an invite code
# device B
mtw join mtw-invite:<id>-<passcode>
```

Hand any OpenAI-compatible client at `http://127.0.0.1:9337`:

```
curl -s http://127.0.0.1:9337/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-30B-A3B-4bit","messages":[{"role":"user","content":"hi"}],"max_tokens":50}'
```

## Prerequisites

| What | Why | Where |
| --- | --- | --- |
| Rust 1.89+ | builds `mtw` | https://rustup.rs |
| Xcode 26 + Metal Toolchain | required by SwiftLM | App Store + `xcodebuild -downloadComponent MetalToolchain` |
| SwiftLM binary | per-node inference engine | clone https://github.com/SharpAI/SwiftLM, run `swift build -c release` |
| MoE weights | the model | e.g. `mlx-community/Qwen3-30B-A3B-4bit` from Hugging Face |

`mtw doctor` checks each of these and tells you the exact next step. Default paths assume the layout in `crates/mtw-cli/src/main.rs` — override with `--swiftlm` / `--model`.

## Build / test

```
make build      # cargo build --release --bin mtw
make test       # cargo test --workspace  (32 tests across 4 ALPNs + cache + engine)
make demo       # scripts/demo.sh — 15-check end-to-end smoke
make clean
```

## Memory + request tracing

`mtw serve` writes a continuous trace of SwiftLM RSS and request boundaries to `MTW_SWIFTLM_LOG` (default `/tmp/mtw-swiftlm.log`):

```
[mtw-mem] t_ms=12345 rss_kb=204876 requests=2
[mtw-req] t_ms=12410 path=/v1/chat/completions requests=3
```

The two markers share one atomic counter so RSS deltas are correlated with request boundaries — useful for diagnosing OOM-after-N-requests patterns.

## End-to-end smoke test

`scripts/demo.sh` verifies the local stack: Metal toolchain, the patched SharpAI/mlx fork (symbols + metallib kernel), the Rust workspace, an iroh echo round-trip, and a real MLX inference against OLMoE. 15 checks, ~20 s on a warm machine.

```
./scripts/demo.sh
```

Prerequisites: OLMoE weights at `~/Desktop/meshthatworks-deps/models/OLMoE-1B-7B-0125-Instruct-4bit/`, the patched MLX fork at `~/Desktop/meshthatworks-deps/mlx/build/libmlx.a`, and a Python venv at `~/Desktop/meshthatworks-deps/.venv` with `mlx-lm` installed. See `docs/ENGINE_RESEARCH.md` §0 for the MLX build fix.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
