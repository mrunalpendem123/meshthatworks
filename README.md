# MeshThatWorks

**Frontier AI on the Macs you already own.**

The biggest open AI models — DeepSeek, Qwen3-Coder, Llama, Mixtral — are free to download. Most people still cannot run them, because the math wants 16+ GB of RAM per device and most consumer Macs have 8.

MeshThatWorks lowers the floor by combining two tricks that nobody had combined before:

- **Your SSD becomes memory.** Weights load from disk on demand instead of sitting in RAM. A model that asks for 18 GB of RAM runs on a Mac with 8.
- **Your devices share the work.** Pair two Macs (or three) and the model splits across them. Each device only holds a slice. More memory available, less waiting.

Other projects do one or the other. [SwiftLM](https://github.com/SharpAI/SwiftLM) streams from SSD, but assumes a single 32 GB+ Mac. [Mesh-LLM](https://github.com/Mesh-LLM/mesh-llm) splits across devices, but assumes each one already has 16 GB. We do both. The per-device floor drops from 16 GB to 4–8 GB.

Your old MacBook is now in the game.

No cloud. No accounts. Your prompts and your data stay on your machines.

---

## Quickstart

Three commands. The first one takes ~30 minutes the first time (it builds SwiftLM); the others are seconds.

```bash
# 1. Install — clones, builds, sets up ~/.local/bin/mtw
curl -sSL https://raw.githubusercontent.com/mrunalpendem123/meshthatworks/master/scripts/bootstrap.sh | sh

# 2. Pick a model — opens the dashboard's Models tab, choose one, hit Enter
mtw dashboard

# 3. Run — engine + dashboard in one terminal, Ctrl-C to stop
mtw start
```

Then point any OpenAI-compatible app at `http://localhost:9337`:

```bash
export OPENAI_BASE_URL=http://localhost:9337/v1
export OPENAI_API_KEY=local
```

That's it. The rest of this README is for when you want to know *how* and *why*.

---

## Demo

<!--
  To embed a screen recording: drag a .mp4 / .mov / .gif into a comment on
  any GitHub issue or PR in this repo, then copy the resulting `https://github.com/...
  /assets/...` URL and paste it where the placeholder below points to.

  Replace the line below with:
      https://github.com/mrunalpendem123/meshthatworks/assets/<id>/<filename>

  GitHub renders that URL inline as an HTML5 video player.
-->

> 📹 Demo video coming soon. In the meantime, here is what the install + first run looks like:

```text
$ curl -sSL https://raw.githubusercontent.com/mrunalpendem123/meshthatworks/master/scripts/bootstrap.sh | sh
==> Checking base tools
==> Cloning / updating ~/.meshthatworks
==> Building mtw (first time ~5 min)
   Compiling mtw-core v0.1.0
   Compiling mtw-engine v0.1.0
   ...
   Finished `release` profile [optimized] target(s) in 4m 33s
==> Installed: ~/.local/bin/mtw
==> Cloning + building SwiftLM (~30 min, ~3 GB disk)
==> All set. Running mtw doctor:

mtw doctor — environment check

local setup
  mtw on PATH     ✓
  Xcode + Metal   ✓
  SwiftLM binary  ✓  ~/.meshthatworks-deps/SwiftLM/.build/arm64-apple-macosx/release/SwiftLM
  Model           ✗  no model installed yet — open `mtw dashboard` and pick one from the Models tab

running network probes (~5–10s)…

  IPv6            ✓ reachable  (local: 2401:4900:88f4:d88f:1da:da0f:164c:2dc1)
  IPv4 (public)   ✓ 122.177.245.210
  NAT type        ✓ endpoint-independent (122.177.245.210:13425)  hole-punching works
  macOS firewall  ? unknown

  Verdict         ✓ IPv6 available — direct connection essentially always works

──────────────────────────────────────────────────────────
Next: pick a model.
       mtw dashboard          # opens the Models tab — choose one and download
```

```text
$ mtw start
mtw start: launching engine + dashboard…
  model:  ~/.meshthatworks-deps/models/OLMoE-1B-7B-0125-Instruct-4bit
  engine: ~/.meshthatworks-deps/SwiftLM/.build/arm64-apple-macosx/release/SwiftLM

waiting for engine to come up (model load can take ~30s cold)…
✓ engine ready — opening dashboard (Ctrl-C to stop)

┌ mtw dashboard ──────────────────────────────────────────────────────────────┐
│ 1 Dashboard   2 Chat   3 Peers   4 Models   5 Help                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ Identity       7b3a9c1e…1f02                                                 │
│ Model          OLMoE-1B-7B-0125-Instruct-4bit  (16 layers, 64 experts)       │
│ Proxy          ● up        http://localhost:9337                             │
│ Peers          0 paired                                                      │
│                                                                              │
│ Activity                                                                     │
│   12:04:33  ok    proxy ready on :9337                                       │
│   12:04:35  ok    SwiftLM healthz responded                                  │
│                                                                              │
│ mesh: 0 peers   proxy: ● up   net: ● direct (IPv6)   state: ● idle           │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Table of contents

1. [Quickstart](#quickstart)
2. [Demo](#demo)
3. [The problem](#the-problem)
4. [How it works](#how-it-works)
5. [Install](#install)
6. [Pick a model](#pick-a-model)
7. [Run it](#run-it)
8. [Use it from Claude Code, Cursor, anything](#use-it-from-claude-code-cursor-anything)
9. [Pair a second device](#pair-a-second-device)
10. [Architecture in depth](#architecture-in-depth)
11. [Configuration reference](#configuration-reference)
12. [Performance and tracing](#performance-and-tracing)
13. [Troubleshooting](#troubleshooting)
14. [Status — what works today](#status--what-works-today)
15. [Roadmap](#roadmap)
16. [Build from source](#build-from-source)
17. [Common commands](#common-commands)
18. [Contribute](#contribute)
19. [Acknowledgements](#acknowledgements)
20. [License](#license)

---

## The problem

A modern frontier model has tens of billions of parameters. At 4-bit quantization a 30B-parameter Mixture-of-Experts (MoE) model still wants ~18 GB of RAM resident — plus working memory for the KV cache, activations, and the OS itself. That comfortably exceeds an 8 GB Mac and is tight even on a 16 GB Mac.

The cloud answer is to rent inference from someone else's H100. That works but it has costs:

- **It is not free.** Tokens billed per million. Easy to spend more on inference than on the hardware that runs it locally.
- **Your data leaves your hardware.** Your prompts, your code, your conversations — they pass through somebody else's machine.
- **You depend on a company's policies.** Rate limits, content rules, model deprecations, pricing changes, region availability.

The local answer is to lower the floor on what hardware can run a model. There are two ways to do that:

1. **Use slower memory.** Modern NVMe SSDs can sustain 2–5 GB/s. That is fast enough that streaming weight pages on demand is dramatically better than not running the model at all. SwiftLM proved this works on big Macs.
2. **Use more devices.** Most people own two or three Apple devices that sit idle most of the day. If you can pipeline a forward pass across them, a model that no one of them could host alone runs across all of them. Mesh-LLM proved this works when each device has enough RAM.

MeshThatWorks does both at once.

---

## How it works

### MoE models in one paragraph

A Mixture-of-Experts model has a "router" at each layer that decides which subset of expert sub-networks (typically 8 of 64 or 8 of 128) handles a given token. At any moment, most of the model's weights are not being used — they are dormant on disk or in RAM, waiting for a token whose router selects them. This is the property MeshThatWorks exploits.

### Trick 1 — SSD as memory

Instead of loading every expert into RAM up front, MeshThatWorks (via SwiftLM) memory-maps the weight file from SSD and only pulls the experts the router actually selects, per token. The OS page cache holds frequently-used experts; cold ones live on disk and pay one SSD read on first use.

```
                          per token, 8 experts fire
                          ↓
   model on SSD  ─────────●●●●●●●●─────●──────●●─────────────────
                          ↓
                          page cache (RAM)
                          ↓
                          GPU (Metal)
```

Without streaming, an 18 GB model never fits in 8 GB of RAM. With streaming, the *active* working set per token is small — a few hundred MB — and the kernel pages experts in and out as the model uses them. Slow on a cold cache, much faster as the cache warms.

### Trick 2 — Mesh distribution

Pair two devices. The model's transformer layers split across them: device A holds layers 0..N/2, device B holds layers N/2..N. A forward pass enters device A as tokens, leaves as a hidden activation, crosses the network to device B, and finishes there as logits.

```
  prompt ──→ device A (layers 0..N/2) ──[activation, ~2 MB]──→ device B (layers N/2..N) ──→ logits
```

Each device only has to hold half the model's weights, half the page-cache pressure. The activation that crosses the wire is small — a few megabytes per token — so a Wi-Fi or Thunderbolt connection between two Macs in the same room handles it without sweating. Three devices? Three slices, each smaller still.

### The combination

```
  ┌──────────────── device A (8 GB Mac) ────────────────┐
  │                                                       │
  │   SSD ───→ page cache ───→ MLX (Metal) on layers 0..K │
  │                                  │                    │
  └──────────────────────────────────┼────────────────────┘
                                     │ activation
                                     ▼   over QUIC (iroh)
  ┌──────────────── device B (8 GB Mac) ────────────────┐
  │                                                       │
  │   SSD ───→ page cache ───→ MLX (Metal) on layers K..N │
  │                                  │                    │
  └──────────────────────────────────┼────────────────────┘
                                     │ logits
                                     ▼
                                next token
```

Each device runs SwiftLM with `MTW_LAYER_RANGE=K,M` so it only loads its slice. The slices share weights only by coincidence — most of an MoE's parameters live inside experts, and routing is independent per layer. So splitting layers across devices does not waste capacity.

The orchestrator (the device the user typed at) drives the pipeline: tokenize, send tokens to the first peer, receive activation, forward to the next peer, receive logits, sample next token, loop.

### One inference step, traced

What happens when you ask the mesh to generate a single token, with two paired devices:

```
  user (device A)                        device A engine                     device B engine
        │                                       │                                   │
        │  POST /v1/chat/completions            │                                   │
        ├──────────────────────────────────────▶│                                   │
        │                                       │                                   │
        │                                       │  tokenize prompt                  │
        │                                       │  build ChatML template            │
        │                                       │                                   │
        │                                       │  POST /v1/layer-forward           │
        │                                       │  { tokens, MTW_LAYER_RANGE=0..K } │
        │                                       │     │                             │
        │                                       │     ▼                             │
        │                                       │  embed + run layers 0..K          │
        │                                       │  (router selects 8 experts/layer; │
        │                                       │   MLX pulls them from page cache, │
        │                                       │   page cache pulls from SSD if    │
        │                                       │   they were not resident)         │
        │                                       │                                   │
        │                                       │  activation [batch, seq, hidden]  │
        │                                       │  ────── bincode over iroh ──────▶ │
        │                                       │       (mtw/layer-forward/0)       │
        │                                       │                                   │  run layers K..N
        │                                       │                                   │  norm + lm_head
        │                                       │                                   │  → logits
        │                                       │ ◀────── bincode over iroh ──────  │
        │                                       │                                   │
        │                                       │  argmax (greedy) or sample        │
        │                                       │  → next token                     │
        │                                       │                                   │
        │  SSE: chunk { delta: " hi" }          │                                   │
        │ ◀─────────────────────────────────────┤                                   │
        │                                       │  loop until EOS or max_tokens     │
```

A few things worth noting:

- The activation payload is small (~2 MB for `hidden=4096`, `seq=128`). A direct QUIC link between two Macs in the same room (Wi-Fi, Thunderbolt, or even cellular hotspot) handles it without bottlenecking.
- The orchestrator is whichever device the user is talking to. The other device is "just" a layer-forward target; it does not need a special role.
- bincode is used because JSON would 6× the payload size on f32 tensors. Same `serde` derives as JSON, no extra type-level cost.
- If a peer is unreachable, the orchestrator falls back to running the whole pass locally (slower, but the request still completes).

---

## Install

```
curl -sSL https://raw.githubusercontent.com/mrunalpendem123/meshthatworks/master/scripts/bootstrap.sh | sh
```

The installer:

1. Checks for `git` and `cargo` (Rust). Installs Rust via `rustup` if missing.
2. Clones this repo to `~/.meshthatworks`.
3. Builds the `mtw` binary and installs it to `~/.local/bin/mtw`.
4. Clones [SwiftLM](https://github.com/SharpAI/SwiftLM) to `~/.meshthatworks-deps/SwiftLM` and runs `swift build -c release`. **This is the slow part — about 30 minutes the first time.**
5. Adds `~/.local/bin` to your `PATH` if it is not already there.
6. Runs `mtw doctor` and tells you the next command.

You need Xcode (full app, not just Command Line Tools) plus the Metal Toolchain — SwiftLM cannot build without them. The installer detects this and tells you exactly what to install.

Total disk: about 1.5 GB (Rust toolchain, SwiftLM build artifacts).

> No model is downloaded by the installer. The dashboard's **Models** tab is the catalog.

---

## Pick a model

```
mtw dashboard
```

Hit `4` to go to the **Models** tab. You see two lists:

- **Installed** — models already on disk under `~/.meshthatworks-deps/models/`. Pick one with `D` to delete and free disk.
- **Available** — a curated catalog from `mlx-community` on Hugging Face, filtered by use case (general chat, coding, vision, small/fast). Hit Enter on a row to download — the dashboard streams the safetensors and config files into `~/.meshthatworks-deps/models/<name>/`.

For a first run, pick something small (1–4 GB) so you can see the system end-to-end without waiting on a 20 GB download. **OLMoE-1B-7B-0125-Instruct-4bit** (3.6 GB) is a good MoE starter. **Qwen3-Coder-30B-A3B-4bit** (~18 GB) is the eventual target.

---

## Run it

```
mtw start
```

That is the whole flow. `mtw start`:

1. Spawns the inference engine (SwiftLM as a child) in the background.
2. Polls `http://localhost:9337/healthz` until it answers.
3. Opens the live dashboard in the same terminal.

Hit Ctrl-C and both come down cleanly.

If you prefer the moving parts separated:

```
mtw doctor      # what is set up, what is missing, and the next command
mtw serve       # just the engine — long-running, prints endpoint id and ALPNs
mtw dashboard   # just the dashboard — attaches to a running engine
```

---

## Use it from Claude Code, Cursor, anything

The proxy at `localhost:9337` speaks the OpenAI HTTP API. Any tool that talks to OpenAI works without code changes — just point it at the local URL.

```bash
export OPENAI_BASE_URL=http://localhost:9337/v1
export OPENAI_API_KEY=local
```

Some tools want the Anthropic env names:

```bash
export ANTHROPIC_BASE_URL=http://localhost:9337
export ANTHROPIC_API_KEY=local
```

Or just hit it directly:

```bash
curl http://localhost:9337/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "any-name",
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 200
  }'
```

Streaming works (`"stream": true`). The proxy passes the upstream Server-Sent Events through unchanged.

The `mtw-api` proxy serves these endpoints:

- `GET /v1/models` — list of loaded models
- `POST /v1/chat/completions` — OpenAI chat
- `POST /v1/completions` — legacy text completion
- `POST /v1/embeddings` — embeddings (if SwiftLM has the model loaded)
- `GET /healthz` — liveness probe
- `GET /status` — JSON snapshot consumed by the dashboard

---

## Pair a second device

On the first Mac, inside the dashboard go to **Peers** (`3`) and press `P`. You see an invite line:

```
mtw-invite:7b3a9c…1f02-deadbeef
```

That is two pieces glued together — a 64-hex iroh endpoint id and an 8-character random passcode. Read it to your other device any way you like (AirDrop, message, photo of the screen).

On the second Mac, also inside the dashboard at the **Peers** tab, press `J` and paste the invite. Both devices verify each other and the pairing is saved to `~/.mtw/peers.json` on both sides. From then on they discover each other automatically — no need to re-pair.

To remove stale peers (devices you paired in the past that are now offline forever): from the **Peers** tab press `X`. That forgets every peer that is currently DOWN.

### What pairing actually does

- A short-lived **pairing endpoint** binds on ALPN `mtw/pair/0` on the first device. It only accepts a connection that knows the passcode.
- The second device dials it, both sides exchange Ed25519 public keys.
- Each side appends the other's public key to its `~/.mtw/peers.json`.
- Future communication between the two uses iroh's persistent identity — direct over QUIC if NAT-traversal succeeds, otherwise via an n0 relay.

The pairing endpoint times out and shuts down after a minute or two. The passcode prevents anyone listening on the same LAN from sneaking in.

---

## Architecture in depth

### The crates

| Crate | What it does |
| --- | --- |
| `mtw-core` | The mesh layer. Peer discovery, identity, persistent peer list, four ALPNs, pairing protocol. Built on iroh. |
| `mtw-engine` | The per-node inference layer. `InferenceEngine` trait, `LayerPeer` trait, `MockEngine` for tests, `SwiftLMEngine` (drives a SwiftLM child via HTTP), `LayerSplitEngine` (orchestrates layer-split across multiple `LayerPeer`s). |
| `mtw-cache` | Adaptive expert caching. `ActivationHistogram` (rolling N-prompt window), `ExpertTier` (Hot/Warm/Cold), `MemoryAdvisor` (mmap + `madvise`/`mlock`), `parse_expert_layout` (safetensors header walker), `apply_tiering` (turns histograms into syscalls). |
| `mtw-api` | OpenAI-compatible HTTP proxy on `localhost:9337`. Forwards `/v1/*` to the local engine, exposes `/status` for the dashboard, emits `[mtw-req]` markers correlated with the engine's RSS sampler. |
| `mtw-cli` | The `mtw` binary. Commands: `serve`, `start`, `dashboard`, `pair`, `join`, `chat`, `status`, `peers`, `doctor`, `echo`. The dashboard is a `ratatui` TUI with five tabs (Dashboard / Chat / Peers / Models / Help). |

### The four ALPNs

iroh routes connections by Application-Layer Protocol Negotiation strings. MeshThatWorks ships four:

| ALPN | Direction | Wire | Purpose |
| --- | --- | --- | --- |
| `mtw/health/0` | request → reply | JSON | Ping. Returns the responder's model info, peer count, and a nonce. Drives the dashboard's UP/DOWN/RTT view. |
| `mtw/infer/0` | request → reply | JSON | Whole-chat delegation. Send a chat request, get a chat response. Used when one device wants another to handle a prompt entirely. |
| `mtw/layer/0` | request → reply | bincode | Single-layer forward. One RPC = one transformer layer's forward pass. Generic over `engine.run_layer()`. Used today only by `MockEngine` tests. |
| `mtw/layer-forward/0` | request → reply | bincode | Range-based forward — the one wired to real SwiftLM. Inputs are either tokens (first peer) or an activation (middle/last peers). Output is an activation or logits. This is the bridge that makes cross-machine layer-split work with real inference. |

bincode for activations because an f32 activation tensor for hidden=4096 / seq=128 is ~2 MB raw. JSON would 6× that. JSON is fine for chat (small payloads, debuggable).

### The data flow when you send a chat

```
   you                                               the model's reply
    │                                                       ▲
    ▼                                                       │
  any OpenAI client                                         │
    │                                                       │
    │   POST /v1/chat/completions                           │
    ▼                                                       │
  mtw-api proxy   :9337  ──→ counts request, emits          │
    │                          [mtw-req] log line           │
    │                                                       │
    │   forward HTTP                                        │
    ▼                                                       │
  SwiftLM         :9876                                     │
    │                                                       │
    │   For each new token:                                 │
    │     • router selects 8 experts per layer              │
    │     • mmap'd weight pages → page cache → MLX (Metal)  │
    │     • forward pass through layer slice                │
    │                                                       │
    │   If layer-split is on:                               │
    │     • this peer runs MTW_LAYER_RANGE=0,K              │
    │     • activation goes back to mtw-engine              │
    │     • mtw-engine dials peer B over mtw/layer-forward/0│
    │     • peer B (MTW_LAYER_RANGE=K+1,N-1) finishes        │
    │     • logits come back → sample next token            │
    │                                                       │
    │   stream chunks back as Server-Sent Events ───────────┘
    ▼
  mtw-api passes the SSE stream through
```

### Where state lives

```
~/.mtw/
  identity.bin            Ed25519 secret key, mode 0600.
                          Your device's persistent identity.
  peers.json              List of peers you have paired with.

~/.meshthatworks/         The repo (cloned by bootstrap).
~/.meshthatworks-deps/
  SwiftLM/                The SwiftLM source + build outputs.
  models/<name>/          Per-model directories. config.json,
                          tokenizer.json, model.safetensors.

~/.local/bin/mtw          The mtw binary.

/tmp/mtw-serve.log        Log output from `mtw serve` (iroh, hyper, ...).
/tmp/mtw-swiftlm.log      SwiftLM stdout/stderr + the [mtw-mem] / [mtw-req]
                          memory + request trace. Override via
                          MTW_SWIFTLM_LOG.
/tmp/mtw-dashboard.log    Dashboard tracing log.
```

---

## Configuration reference

### `mtw serve` flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `--swiftlm <path>` | `~/.meshthatworks-deps/SwiftLM/.build/arm64-apple-macosx/release/SwiftLM` | Path to the SwiftLM binary. |
| `--model <dir>` | `~/.meshthatworks-deps/models/OLMoE-1B-7B-0125-Instruct-4bit` | Directory with `config.json` + `model.safetensors`. |
| `--swiftlm-port <u16>` | `9876` | Port SwiftLM listens on internally. |
| `--proxy-port <u16>` | `9337` | Port the OpenAI-compat proxy listens on (user-facing). |
| `--mem-limit <u32>` | `4096` | Metal memory ceiling for SwiftLM, in MB. |
| `--mock` | off | Use `MockEngine` instead of spawning SwiftLM. For mesh testing. |
| `--attach <url>` | unset | Attach to an already-running SwiftLM instead of spawning one. |
| `--draft-model <dir>` | unset | Optional draft model for speculative decoding (~1.5–3× speedup when the draft accepts well). |
| `--num-draft-tokens <u32>` | `4` | Tokens per speculation round. |

### Environment variables

| Var | Effect |
| --- | --- |
| `MTW_SWIFTLM_LOG` | Where the engine sampler writes `[mtw-mem]` and `[mtw-req]` lines. Default `/tmp/mtw-swiftlm.log`. |
| `MTW_LOG_FILE` | Where `mtw serve` redirects tracing. Default `/tmp/mtw-serve.log`. |
| `SWIFTLM_TOP_K` | Override the MoE router's top-k. Lower = fewer experts per token = faster, lower quality. ~20–40% gain on Qwen3-30B-A3B. |
| `MTW_LAYER_RANGE=K,M` | Tells SwiftLM to load only layers K..M. Set per peer in a layer-split deployment. |
| `MTW_EXPERT_LOG=<path>` | When the Swift patch in `patches/` is applied, SwiftLM appends `[mtw-expert] layer=L indices=[...]` lines for the cache to consume. |
| `HOME` | Used to compute defaults. Override to relocate everything (e.g. `HOME=/tmp/test mtw pair`). |

### Settings files

`~/.mtw/identity.bin` and `~/.mtw/peers.json` are managed by `mtw`. You can delete them to start fresh — `mtw` will mint a new identity on next launch.

---

## Performance and tracing

### What is measured

`docs/BASELINES.md` keeps a running table of `(runner, model, hardware, tok/s, date, notes)`. Currently includes:

- Upstream MLX 0.31.2 on OLMoE-1B-7B-0125-Instruct-4bit, 8 GB Mac → **3.6 tok/s** baseline (no streaming, full RAM).
- SwiftLM `--stream-experts` on LFM2-8B-A1B-4bit, 8 GB Mac → **1.6–2.78 tok/s** first request, OOM after 1–2 requests.
- `mtw serve` → SwiftLM on Qwen3-30B-A3B-4bit, 8 GB Mac, `--mem-limit 4096` → streams cleanly (RSS 30 KB ↔ 906 MB on an 18 GB model), throughput severely bandwidth-bound.
- Same with `--mem-limit 6000` → **0.045 tok/s warm decode**. Cause identified: macOS unified-memory page-cache thrashing. `MEM_DEMAND` climbed 6.7 → 7.0 GB across two requests.

The single-device throughput target (≥2 tok/s on 8 GB × 30B) is the open work item. The cache layer designed to fix it (`mtw-cache::apply_tiering`) is built but not yet wired into the serve loop. Full diagnosis and fix plan are in `docs/BASELINES.md`.

### The trace format

Every `mtw serve` writes timestamped events to a single log file (default `/tmp/mtw-swiftlm.log`). Two markers:

```
[mtw-mem] t_ms=12345 rss_kb=204876 requests=3
[mtw-req] t_ms=12410 path=/v1/chat/completions requests=4
```

Same `requests=` counter on both — it is incremented by the `mtw-api` proxy on every chat request and read by the in-process RSS sampler that polls `ps -o rss=` every 500 ms. So you can correlate a memory spike with the exact request that caused it.

```
$ grep '\[mtw-' /tmp/mtw-swiftlm.log | tail -20

[mtw-mem] t_ms=11420 rss_kb=141824 requests=0
[mtw-mem] t_ms=11930 rss_kb=141824 requests=0
[mtw-req] t_ms=12010 path=/v1/chat/completions requests=1
[mtw-mem] t_ms=12440 rss_kb=355648 requests=1   # request 1 underway
[mtw-mem] t_ms=12952 rss_kb=566224 requests=1
...
[mtw-mem] t_ms=22100 rss_kb=8400  requests=1    # idle, fully evicted
[mtw-req] t_ms=24500 path=/v1/chat/completions requests=2
[mtw-mem] t_ms=25010 rss_kb=566224 requests=2   # request 2 reloaded experts
```

The `requests=` field is the lever for spotting OOM-after-N-requests bugs. If RSS climbs *across* requests instead of returning to the idle baseline, something is leaking.

---

## Troubleshooting

**`mtw doctor` says SwiftLM binary not found.**
The bootstrap could not build SwiftLM, usually because Xcode is missing or only Command Line Tools is installed. Install Xcode from the App Store, then `xcodebuild -downloadComponent MetalToolchain`, then re-run `bootstrap.sh`.

**`mtw start` says "no model installed yet".**
You skipped the catalog. Run `mtw dashboard`, hit `4` for Models, pick a row.

**Dashboard says `proxy: ⊘ down`.**
The engine is not running. Either you launched `mtw dashboard` directly without `mtw serve`, or the engine crashed. Run `mtw start` instead — it manages both.

**Every peer is DOWN.**
They are saved in `~/.mtw/peers.json` from past pairings but unreachable now (they are off, on a different network, or de-paired). On the Peers tab, press `X` to forget all DOWN peers.

**`mtw pair` prints an invite but the other device hits "connection refused".**
Both devices need outbound network. The pairing handshake uses iroh, which prefers a direct QUIC path and falls back to relay. If both devices are on weird networks, run `mtw doctor` on each to see the NAT verdict.

**MoE inference is much slower than `estimated_tok_s`.**
You are seeing macOS page-cache thrashing. `MEM_DEMAND` (in SwiftLM's logs) is exceeding `system_ram_gb`. The fix in flight is the `mtw-cache` page advisor; until it lands, smaller models or two-device layer-split are the workarounds.

**`mtw pair` works but `mtw chat --peer <id>` times out.**
The other device's `mtw serve` may not be running, or its firewall is blocking inbound iroh traffic. `mtw doctor` on the other side will say.

---

## Status — what works today

| Capability | State |
| --- | --- |
| Single-device runs end-to-end | ✓ |
| Streaming primitive proven on 8 GB Mac with 18 GB model | ✓ — RSS oscillates 30 KB ↔ 906 MB |
| OpenAI-compatible HTTP proxy | ✓ — chat, completions, embeddings, models |
| Dashboard (chat, peers, model hub, pairing in TUI) | ✓ |
| Pair / join over iroh | ✓ — invite-code flow, persistent identity |
| Mesh ALPNs (health, infer, layer, layer-forward) | ✓ — all wired into `mtw serve` |
| Cross-device layer-split bridge | ✓ — built, unit-tested with stub peers (3 iroh integration tests) |
| Live two-Mac demo with real SwiftLM | ⏳ — bridge in place, awaiting a second device run |
| Sustained throughput at target (≥2 tok/s on 8 GB × 30B) | ⏳ — currently bandwidth-bound; cache fix designed and built, not yet wired |
| Adaptive expert tap from SwiftLM | ⏳ — Swift patch in `patches/`, pending apply + rebuild |
| `mtw-cache::apply_tiering` integrated into `mtw serve` | ⏳ — primitives ready, integration pending |

Live measurements, the analysis behind the throughput problem, and the plan to fix it are in [`docs/BASELINES.md`](docs/BASELINES.md). Full architecture, milestone list, and references in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Roadmap

Roughly in order:

1. **Wire `mtw-cache` into `mtw serve`.** Take the `[mtw-expert]` log feed (after the Swift patch is applied), feed it to `ActivationHistogram::record_prompt`, recompute tiers every N prompts, and call `apply_tiering` to issue `madvise(WILLNEED|DONTNEED)` against the safetensors mmap. Goal: bring single-device 30B throughput from 0.045 tok/s into the 1+ tok/s range by escaping page-cache thrashing.
2. **Live two-Mac demo.** Run `MTW_LAYER_RANGE=0,23` SwiftLM on Mac A, `MTW_LAYER_RANGE=24,47` on Mac B, drive a forward pass across both via `mtw/layer-forward/0`, measure tok/s vs single-Mac.
3. **Apply the Swift expert-tap patch upstream.** `patches/qwen3moe-expert-tap.patch` works locally; submit it to SwiftLM so users do not need to re-patch on every upgrade.
4. **Adaptive re-sharding.** Once per-device tier stats exist, redistribute experts across devices based on observed usage. A device whose hot experts overlap heavily with another device's hot experts is wasting space.
5. **Linux / Windows.** Mac-only today because SwiftLM is Mac-only. CUDA / ROCm equivalents are a port, not a redesign.
6. **Speculative decoding.** Already plumbed (`--draft-model`), needs a curated set of small draft models in the Models catalog.
7. **Public mesh.** Optional — let strangers donate compute to other strangers' models. Out of scope for v1; would need verified computation, anti-abuse, and probably tokens. v2+.

---

## Build from source

```
git clone https://github.com/mrunalpendem123/meshthatworks
cd meshthatworks
make install
```

You need:

- Rust 1.89+ (`rustup` recommended).
- Xcode 26+ with the Metal Toolchain (App Store + `xcodebuild -downloadComponent MetalToolchain`).
- A working `git` (or Xcode Command Line Tools).

`make install` runs `cargo build --release --bin mtw` and copies the binary to `~/.local/bin/mtw`. It does not build SwiftLM — for that, run `scripts/bootstrap.sh`, which does both.

---

## Common commands

```
make build       # cargo build --release --bin mtw
make install     # build + install to ~/.local/bin
make test        # cargo test --workspace  (32 tests across 4 ALPNs and the cache)
make doctor      # mtw doctor
make demo        # 15-check end-to-end smoke test
make serve       # mtw serve (engine only)
make claude-env  # print env vars to point Claude Code at the local node
make clean       # cargo clean
```

The `make demo` target runs `scripts/demo.sh` — a 15-step smoke test that verifies the Metal toolchain, the SharpAI/mlx fork is built with the right symbols, the workspace tests pass, the iroh echo round-trip works, and a real MLX inference completes against a known model. It is the fastest way to confirm a fresh checkout is healthy.

---

## Contribute

The interesting open work items, in order of "would pay off most":

1. Wire `mtw-cache::apply_tiering` into `mtw serve` and re-bench. (Issue 1 on the roadmap.) Pure code — no second machine needed.
2. Run a two-Mac layer-split benchmark and post the numbers. Needs two paired Macs. The tooling is all in place.
3. Submit the Swift expert-tap patch to SwiftLM upstream.
4. Test on hardware we have not — M1 Air, 16 GB Mini, M4 Pro, etc. Add rows to `docs/BASELINES.md`.
5. Quantizations beyond MLX 4-bit (FP8, Q5_K, etc) — both for SwiftLM compatibility and for cache-policy measurement.
6. Cross-platform engine port (Linux + CUDA, Linux + ROCm, Windows).

PRs welcome. The bar: tests pass, the change is described in the commit message, and any new behaviour has a unit or integration test that exercises it. CI is `cargo test --workspace`.

---

## Acknowledgements

MeshThatWorks stands on three open-source projects:

- **[iroh](https://www.iroh.computer)** by [n0-computer](https://github.com/n0-computer) — QUIC-based P2P networking with NAT traversal, encrypted by default, optional relay fallback. Apache-2.0/MIT.
- **[SwiftLM](https://github.com/SharpAI/SwiftLM)** by [SharpAI](https://github.com/SharpAI) — Apple-Silicon-optimized MLX inference engine with SSD expert streaming. MIT.
- **[mlx-swift](https://github.com/ml-explore/mlx-swift)** by Apple — Swift bindings to the MLX numerical library. MIT.

Mesh-LLM, Petals, Prima.cpp, and the broader distributed-inference research line all influenced the design. Specific paper credits are in `docs/ARCHITECTURE.md`.

---

## License

MIT. See [`LICENSE`](LICENSE).
