# MeshThatWorks

**Frontier AI on the Macs you already own.**

The biggest open AI models — DeepSeek, Qwen3-Coder, Llama — are free to download. Most people still can't run them, because the math wants 16 GB of RAM per device and most Macs have 8.

MeshThatWorks lowers that floor. It does two things at once that nobody else does together:

- **Treats your SSD as memory.** The model loads weights from disk on demand instead of holding all of them in RAM. A model that asks for 18 GB of RAM runs on a Mac with 8.
- **Splits the work across devices.** Pair two Macs and the model spreads across both. More memory, less waiting. Add a third device and bigger models open up.

Other projects do one or the other. SwiftLM streams from SSD, but assumes a single 32 GB+ Mac. Mesh-LLM splits across devices, but assumes each one already has 16 GB. We do both. The per-device floor drops from 16 GB to 4–8 GB.

Your old MacBook is now in the game.

No cloud. No accounts. Your prompts and your data stay on your machines.

## Install

```
curl -sSL https://raw.githubusercontent.com/mrunalpendem123/meshthatworks/master/scripts/bootstrap.sh | sh
```

This installs Rust if missing, clones the repo, clones and builds [SwiftLM](https://github.com/SharpAI/SwiftLM), and builds `mtw`. You need Xcode for the SwiftLM build — the installer tells you if it is missing. First run is around 30 minutes and uses about 1.5 GB of disk.

No model is downloaded yet — the dashboard's Models tab is the catalog.

## Pick a model

```
mtw dashboard
```

Open the **Models** tab. Pick one from the catalog (small models for trying it out, big MoE models for the real use case). The dashboard handles the download.

## Run it

```
mtw start
```

That is the whole flow. It launches the engine in the background, waits for it to come up, and opens the live dashboard in the same terminal. Hit Ctrl-C to stop both.

If you want to look around first:

```
mtw doctor      # what is set up, what is missing, and the next command
mtw serve       # just the engine, no dashboard
mtw dashboard   # just the dashboard, attaches to a running engine
```

## Use it from any tool

The proxy speaks OpenAI. Point Claude Code, Cursor, the OpenAI Python SDK, an Ollama-compatible client, or your own scripts at `http://localhost:9337`. No code changes.

```
export OPENAI_BASE_URL=http://localhost:9337/v1
export OPENAI_API_KEY=local
```

## Add a second device

On the first Mac:

```
mtw pair
```

It prints an invite. On the second Mac:

```
mtw join <invite>
```

Both Macs share the model now — bigger models become reachable, and throughput goes up.

## How it works

```
  your app  →  /v1/chat/completions  →  mtw-api proxy on :9337
                                              │
                                              ↓
                              mtw-engine (per-node SwiftLM driver)
                                              │
                                ┌─────────────┴─────────────┐
                                │                            │
                          this Mac's SwiftLM           other Mac's SwiftLM
                          (layers 0..N/2)              (layers N/2..N)
                                │                            │
                                └────── iroh QUIC ───────────┘
                                  (mtw/layer-forward/0 ALPN)
```

| Crate | What it does |
| --- | --- |
| `mtw-core` | Peer discovery, mesh ALPNs (`health`, `infer`, `layer`, `layer-forward`), pairing |
| `mtw-engine` | Per-node inference: SwiftLM driver, `LayerPeer` trait, `LayerSplitEngine` orchestrator |
| `mtw-cache` | Adaptive expert caching: rolling activation histogram, hot/warm/cold tiering, OS page-residency advisor |
| `mtw-api` | OpenAI-compatible HTTP proxy on `localhost:9337` |
| `mtw-cli` | The `mtw` binary and the `ratatui` dashboard |

The transport is [iroh](https://www.iroh.computer): QUIC under the hood, end-to-end encrypted, NAT-traversed, falls back to a relay only if a direct path is impossible. Identities are Ed25519 keys at `~/.mtw/identity.bin`.

The engine is [SwiftLM](https://github.com/SharpAI/SwiftLM) — Apple-Silicon-optimized MLX inference with SSD expert streaming. We drive it as a child process and add the mesh layer on top.

## Status (honest)

| | |
| --- | --- |
| Single-device runs end-to-end | ✓ |
| Streaming primitive proven on 8 GB Mac with 18 GB model | ✓ — RSS oscillates 30 KB ↔ 906 MB while the kernel pages experts |
| OpenAI proxy + dashboard + pairing | ✓ |
| Cross-device layer-split bridge (`mtw/layer-forward/0`) | ✓ — built, unit-tested with stub peers |
| Live two-Mac demo | ⏳ — bridge in place, awaiting a second device run |
| Sustained throughput at target (≥2 tok/s on 8 GB Mac × 30B) | ⏳ — currently bandwidth-bound by macOS unified memory; cache fix designed and built, not yet wired |
| Adaptive expert tap from SwiftLM | ⏳ — Swift patch in `patches/`, pending apply + rebuild |

Live measurements, the analysis behind the throughput problem, and the plan to fix it are in [`docs/BASELINES.md`](docs/BASELINES.md). Full architecture and milestone roadmap in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Why this exists

Frontier AI is gated by hardware availability, not by model availability. The biggest open models are downloadable; most people can't run them because the per-device RAM floor (16 GB+) is above what they own.

The cloud answer is to rent inference from someone else's H100. That works but it's not free, your data leaves your hardware, and you depend on a company's rate limits and policies.

The local answer is to lower the floor. Two compounding tricks:

1. **Disk is slow RAM.** Modern SSDs are fast enough that streaming weight pages on demand beats not running the model at all. SwiftLM proved this on big Macs; we extend it to small ones.
2. **Idle devices are extra capacity.** Most people own two or three Apple devices that sit idle most of the time. Pair them, and a model that no single one of them could host runs across all of them — over QUIC, encrypted, on your home network.

Combined, the floor drops from 16 GB to 4–8 GB per device. An old MacBook plus a Mac Mini plus a current iPhone Pro can together host a model that none of them could on their own.

That's the bet.

## Build from source

```
git clone https://github.com/mrunalpendem123/meshthatworks
cd meshthatworks
make install
```

## Common commands

```
make build       # cargo build --release --bin mtw
make install     # build + install to ~/.local/bin
make test        # cargo test --workspace  (32 tests)
make doctor      # mtw doctor
make demo        # 15-check end-to-end smoke test
make clean
```

## Contribute

Issues, PRs, and benchmarks on hardware we haven't tested are all welcome. The interesting open questions are listed in `docs/BASELINES.md`'s "What's not yet measured" section. Cross-device benchmarks and quantizations beyond MLX 4-bit are especially valuable.

## License

MIT. See [`LICENSE`](LICENSE).
