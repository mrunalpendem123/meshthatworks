# MeshThatWorks — Technical Specification

**Version**: 0.1 (draft)
**Status**: Pre-implementation planning
**Target start**: Week of April 28, 2026

---

## 1. What we're building

A tool that runs frontier open-source AI models across 2–3 consumer devices the user already owns, by splitting the model across devices and using each device's SSD as extended memory.

**In one sentence**: Run Qwen3-Coder-30B-A3B on 2× 8GB Macs at 8–15 tok/s, via per-node SSD expert streaming inside a distributed mesh.

**Scope for v1**: home use only. Pair devices by invite code, use locally, done. No accounts, no cloud, no tokens, no public mesh.

---

## 2. Why this is novel

Every existing solution either:

- **Assumes datacenter hardware** (llm-d, NVIDIA Dynamo) — needs H100s, Kubernetes, not consumer-accessible
- **Assumes each node has 16GB+ RAM** (Petals, Mesh-LLM, Prima.cpp) — excludes the majority of consumer devices
- **Works on low-memory single devices but very slowly** (AirLLM, llama.cpp mmap) — 0.2–0.7 tok/s, unusable

**MeshThatWorks's contribution**: per-node SSD expert streaming *inside* a distributed mesh. Each node streams experts from its local SSD while participating in a multi-device pipeline. Drops the hardware floor from 16GB/device to 4GB/device.

**The exact gap**: SwiftLM ships SSD streaming for a single 64GB+ Mac. Mesh-LLM ships distributed inference assuming RAM-only per-node storage. Nobody has combined them. That's the bet.

---

## 3. System architecture

### 3.1 Layered stack

```
┌─────────────────────────────────────────────────────────┐
│  Agent integration layer                                │
│  - OpenAI-compatible API on localhost:9337              │
│  - Works with Claude Code, Goose, opencode, etc.        │
├─────────────────────────────────────────────────────────┤
│  Mesh coordination layer                                │
│  - Peer discovery and invite pairing                    │
│  - Shard assignment                                     │
│  - Adaptive expert re-sharding based on usage           │
├─────────────────────────────────────────────────────────┤
│  Transport layer                                        │
│  - iroh (QUIC, NAT traversal, relay fallback)           │
│  - Node identity via Ed25519 keys                       │
├─────────────────────────────────────────────────────────┤
│  Per-node inference engine                              │
│  - SwiftLM-forked MLX with SSD streaming                │
│  - Usage-adaptive expert caching                        │
│  - Disk-backed KV cache                                 │
├─────────────────────────────────────────────────────────┤
│  Hardware                                               │
│  - Apple Silicon Macs (primary)                         │
│  - 4–8GB RAM + 20GB free SSD per device                 │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Typical deployment

Two devices in the same home, connected via Wi-Fi or Thunderbolt:

- **Node A** (M2 MacBook Pro, 8GB RAM): holds layers 0–23, streams cold experts from its SSD
- **Node B** (iPhone Pro, M4 iPad, or older 8GB Mac): holds layers 24–47, streams cold experts from its SSD

User points Claude Code at `http://localhost:9337`. Claude Code sends a prompt. The coordinator node runs its layers, passes activations to the other node over iroh, that node runs its layers, returns activations or final logits, tokens stream back to Claude Code.

---

## 4. Core dependencies (all MIT/Apache 2.0)

### 4.1 Networking
- **iroh** — QUIC P2P transport, NAT traversal, node identity
  - Repo: https://github.com/n0-computer/iroh
  - Crate: `iroh = "0.91"` (check crates.io for latest)
  - What we use: `Endpoint`, `Connection`, `open_bi` streams
- **iroh-gossip** — optional, for peer capability announcements
  - Repo: https://github.com/n0-computer/iroh/tree/main/iroh-gossip

### 4.2 Inference engine (per-node)
- **SharpAI/mlx** — MLX fork that carries the SSD streaming primitive. This is the fork we actually build against.
  - Repo: https://github.com/SharpAI/mlx
  - Key files: `mlx/backend/metal/ssd_streamer.{mm,h}`, `mlx/core/moe_stream_op.{cpp,h}`, `mlx/backend/metal/kernels/{fence,moe_stream}.metal`
  - Reference commit: `9a9c214` ("feat: custom ssd-streaming kernels and custom MLX I/O fast loaders")
  - License: MIT
  - See `docs/ENGINE_RESEARCH.md` for a file:line walkthrough.
- **SwiftLM** — Swift wrapper that sits on top of MLX. Not on the critical path for v1; we may or may not consume any of it depending on whether we need its Swift-side tokenization/weight-loading helpers.
  - Repo: https://github.com/SharpAI/SwiftLM
  - License: MIT
  - Author: Eric Lake (SharpAI)
- **SharpAI/mlx-c** — C bindings for MLX. Candidate for the `mtw-engine` FFI layer.
  - Repo: https://github.com/SharpAI/mlx-c

### 4.3 Mesh coordination reference
- **Mesh-LLM** — reference for iroh usage patterns and agent integration
  - Repo: https://github.com/Mesh-LLM/mesh-llm
  - Docs: https://docs.anarchai.org/
  - License: Apache 2.0
  - We borrow: relay deployment (`relay/` folder), agent launcher pattern (Goose/Claude Code integration), OpenAI API proxy
  - Author: Michael Neale (Block/Goose team)

### 4.4 Distributed inference reference
- **Prima.cpp** — reference for pipelined-ring parallelism
  - Repo: https://github.com/Lizonghang/prima.cpp
  - Paper: arXiv:2504.08791 (ICLR 2026)
  - License: Apache 2.0
  - We borrow: Halda scheduler concepts, PRP pipelining

### 4.5 Model
- **Qwen3-Coder-30B-A3B** — primary target model
  - Source: Hugging Face, Qwen organization
  - Architecture: MoE, ~128 experts, 8 active per token
  - Dense backbone: ~2–3GB (always resident)
  - Total disk (Q4): ~18GB
  - Per-token active weights: ~400MB

### 4.6 Optional future additions
- **KVSwap** (arXiv:2511.11907) — disk-backed KV cache for long context
- **CATS** (arXiv:2404.08763, https://github.com/ScalingIntelligence/CATS) — activation sparsity thresholding
- **Fate prefetching** (arXiv:2502.12224) — expert activation prediction

---

## 5. The novel contributions (what we build, not borrow)

### 5.1 Per-node SSD streaming inside a mesh

Upstream reality (see `docs/ENGINE_RESEARCH.md`): SharpAI/mlx already ships the SSD→GPU primitive (pinned `MTLBuffer` + `pread` + a Metal 4-bit GEMM kernel), but the hot path is **synchronous and per-forward-pass** — no expert-level cache, no prefetch. The `io_queue_` and `MTL::SharedEvent shared_event_` hooks are constructed but inert.

What we build:
- **Expert-aware LRU cache** keyed by `active_expert` id, living in `mtw-engine` / `mtw-cache`. Deferred deallocation past end of forward pass, so hot experts stay resident.
- **Async prefetch path**: reactivate the dormant `io_queue_` for non-blocking `load_async`, wire `shared_event_` into a Metal command buffer wait so compute overlaps with next-expert load.
- **Allocator ceiling tuning** via `MetalAllocator::set_memory_limit()` for the 4–8GB unified-memory budget. No `MLX` source changes needed for this — caller-side init only.
- **4–8GB budget constants** live in our cache layer, not upstream; MLX has no such knob today.

### 5.2 Usage-adaptive expert caching
- Per-node instrumentation: record which experts fire on user's actual prompts
- Build a histogram over a rolling window (last 500 prompts)
- Classify experts into tiers: hot (top 20% by activation), warm (next 30%), cold (rest)
- Pin hot experts in RAM, keep warm experts in mmap, stream cold experts on demand
- Rebalance the cache once per hour or when the histogram shifts significantly

### 5.3 Mesh-level adaptive re-sharding
- After each node has built its local expert histogram, gossip the top-N experts to the mesh
- Compute a global assignment that minimizes total SSD streaming time
- Example: if Node A's user codes all day and Node B's user writes emails, different experts get pinned per node
- Re-shard once per week or when usage patterns shift

### 5.4 Home-mesh invite flow
- Simple pairing UX: first device prints a short code, second device enters it
- Under the hood: iroh NodeId exchange + shared secret for authentication
- No accounts, no cloud, no discovery servers for v1

---

## 6. Component-level design

### 6.1 `mtw-core` (Rust crate)

Responsibilities:
- Entry point binary (`mtw serve`, `mtw pair`, `mtw status`)
- iroh endpoint management
- Peer connection lifecycle
- Shard assignment logic
- Inference request coordination

Key types:
```rust
struct Node {
    id: NodeId,              // iroh Ed25519 public key
    capacity: NodeCapacity,  // RAM, SSD, bandwidth
    assigned_layers: Range<usize>,
    expert_shard: Vec<ExpertId>,
}

struct MeshState {
    self_node: Node,
    peers: HashMap<NodeId, Node>,
    model: ModelInfo,
    router: RequestRouter,
}
```

### 6.2 `mtw-engine` (Rust crate wrapping C++)

Responsibilities:
- FFI wrapper around forked MLX/SwiftLM
- SSD streaming coordination
- KV cache management
- Expert usage profiling (instrumentation)

Implementation:
- Uses SharpAI/mlx-c bindings
- Adds profiling hooks to record expert activation per forward pass
- Exposes a clean Rust API: `run_layer(activations) -> activations`

### 6.3 `mtw-cache` (Rust crate)

Responsibilities:
- Expert usage histogram
- Hot/warm/cold tier classification
- RAM budget management (how many experts to pin)
- Rebalancing trigger logic

### 6.4 `mtw-api` (Rust crate)

Responsibilities:
- OpenAI-compatible HTTP API on localhost:9337
- SSE streaming for token generation
- Request routing to local vs remote nodes
- Web dashboard on localhost:3131 (optional v1.1)

### 6.5 `mtw-cli` (binary)

Commands:
```
mtw serve --auto             # Start node, pair with existing mesh
mtw pair                     # Print invite code for first device
mtw join <invite-code>       # Join an existing home mesh
mtw status                   # Show mesh health and assignment
mtw models                   # List available models
mtw download <model>         # Fetch a model to local SSD
```

---

## 7. Critical technical decisions

### 7.1 Why Rust for the mesh layer
- iroh is Rust-native
- Memory safety for network-facing code
- Tokio async runtime matches iroh's async model
- Good FFI to C++ (for MLX bindings)

### 7.2 Why fork SwiftLM instead of building from scratch
- SwiftLM already ships working SSD→GPU streaming on Apple Silicon via Metal
- Their `ssd_streamer.mm` and `fence.air` solve problems we'd otherwise spend months on
- Adapting their code to smaller RAM is easier than writing it from zero
- Eric Lake is approachable — reach out early

### 7.3 Why iroh over libp2p or raw QUIC
- Iroh purpose-built for P2P with NAT traversal included
- Matches Mesh-LLM's choice, enabling potential interop later
- Simpler API surface than libp2p
- Active development, battle-tested at scale

### 7.4 Why Qwen3-Coder-30B-A3B as the target
- MoE architecture (required for SSD streaming to make sense)
- 30B total, 3B active — quality comparable to dense 27B, speed comparable to dense 3B
- Coding-focused — matches our primary user (developers on Claude Code)
- Apache 2.0 license
- Available on Hugging Face in GGUF and MLX formats

### 7.5 Why Mac-first for v1
- Apple Silicon's unified memory architecture makes SSD streaming viable (SSD→GPU direct paths)
- MLX is a modern, well-designed framework (unlike legacy CoreML)
- User's own primary hardware is M2 MacBook Pro
- Large Mac user base among the target audience (developers)
- Linux/Windows can come in v2

---

## 8. Milestones and deliverables

### Milestone 1 (Month 1): Baselines and environment
**Goal**: Understand what we're building on top of.

- [ ] Install SwiftLM on a 32GB+ Mac, run Qwen3.5-122B, benchmark
- [ ] Install Mesh-LLM on M2 + second device, run Qwen3-Coder-30B-A3B, benchmark
- [ ] Install Prima.cpp, run a benchmark on the same hardware
- [ ] Read `ssd_streamer.mm` and `fence.air` in full, document findings
- [ ] Build iroh echo example between M2 and another device
- [ ] Write a one-page "state of the baselines" document

**Deliverable**: `docs/BASELINES.md` with numbers and gaps.

### Milestone 2 (Month 2): Per-node SSD streaming on 8GB
**Goal**: Get Qwen3-Coder-30B-A3B running on a single 8GB Mac.

Revised task list after the research in `docs/ENGINE_RESEARCH.md`:

- [ ] Build SharpAI/mlx locally at `9a9c214`, verify `streamed_gather_mm` works end-to-end via a standalone C++ test (no Rust yet)
- [ ] C++ shim crate exposing `SSDStreamer::new/load_sync/free`, `streamed_gather_mm`, and the existing `mlx_ssd_metrics_snapshot` as `extern "C"`
- [ ] Rust FFI in `mtw-engine` (`cxx` crate or hand-rolled `bindgen`) over the shim
- [ ] Expert-aware LRU cache in `mtw-cache`, plugged in between the mesh scheduler and `SSDStreamer::load_sync`
- [ ] Async prefetch: reactivate `io_queue_` in `ssd_streamer.mm`, wire `shared_event_` into the Metal command buffer so compute and next-expert load overlap
- [ ] `MetalAllocator::set_memory_limit()` called at engine init for the 4–8GB budget
- [ ] Test on M2 8GB with Qwen3-Coder-30B-A3B, target 2–4 tok/s initially
- [ ] Profile expert activation patterns on 500 coding prompts (feeds Milestone 4)
- [ ] Publish results as draft blog post

**Deliverable**: Single-node inference running at 2–5 tok/s on 8GB M2.

### Milestone 3 (Month 3): Two-node mesh with SSD streaming
**Goal**: Combine SSD streaming with distributed inference.

- [ ] Build `mtw-core` Rust crate with iroh-based mesh coordination
- [ ] Integrate `mtw-engine` as the per-node inference engine
- [ ] Implement basic shard assignment (layers 0..N/2 on Node A, rest on Node B)
- [ ] Get two 8GB Macs to run Qwen3-Coder-30B-A3B together
- [ ] Benchmark: expected 6–10 tok/s

**Deliverable**: Working two-node mesh with SSD streaming per node.

### Milestone 4 (Month 4): Adaptive caching
**Goal**: System improves with usage.

- [ ] Implement `mtw-cache` with rolling expert histogram
- [ ] Hot/warm/cold tier classification
- [ ] RAM budget rebalancing based on observed usage
- [ ] Implement mesh-level adaptive re-sharding (weekly rebalance)
- [ ] Benchmark: expected 8–15 tok/s after 1 week of warm-up

**Deliverable**: Self-optimizing mesh that improves speed over time.

### Milestone 5 (Month 5): Polish, release, paper
**Goal**: Ship it.

- [ ] OpenAI-compatible API (`mtw-api` crate)
- [ ] Invite-code pairing flow (`mtw pair` / `mtw join`)
- [ ] Claude Code integration (write config file + launcher)
- [ ] README, installation docs, troubleshooting guide
- [ ] Open source under Apache 2.0 or MIT
- [ ] Write up MLSys / ICLR workshop submission

**Deliverable**: Public repo, installable binary, paper draft.

---

## 9. Success criteria

### Technical
- [ ] Qwen3-Coder-30B-A3B runs on 2× 8GB Macs at ≥8 tok/s sustained
- [ ] Single-device mode runs on a 16GB Mac at ≥15 tok/s (fallback)
- [ ] Adaptive caching improves tok/s by ≥30% after 1 week of usage
- [ ] Mesh survives a peer disconnecting and reconnecting
- [ ] Works with Claude Code without config changes

### Research
- [ ] Paper-quality benchmark suite with comparisons against:
  - llama.cpp mmap (single node)
  - AirLLM (single node, low memory)
  - SwiftLM (single node, high memory)
  - Mesh-LLM (distributed, RAM-only)
  - Prima.cpp (distributed, RAM-only)
- [ ] Ablation study: effect of SSD streaming, adaptive caching, mesh re-sharding individually

### Product
- [ ] One-command install: `curl ... | bash`
- [ ] Pair two devices in under 60 seconds
- [ ] Works on macOS 13+ on Apple Silicon
- [ ] Documentation that a developer can follow without assistance

---

## 10. Risks and mitigations

### Risk 1: Apple NVMe not fast enough for real-time streaming
- **Likelihood**: medium
- **Impact**: high (would cap speed at 2–3 tok/s)
- **Mitigation**: adaptive caching should bring hit rate above 80%, reducing SSD pressure. Speculative decoding with draft model doubles effective throughput.
- **Fallback**: if 8GB is too tight, target 16GB Macs (still novel — SwiftLM targets 64GB+).

### Risk 2: SwiftLM fork diverges too much from upstream MLX
- **Likelihood**: medium
- **Impact**: medium (makes maintenance hard)
- **Mitigation**: keep fork minimal. Upstream patches to SwiftLM where possible. Document our diff clearly.

### Risk 3: iroh API changes between versions
- **Likelihood**: low
- **Impact**: low
- **Mitigation**: pin version. Iroh is approaching 1.0 and committing to wire-protocol stability.

### Risk 4: Qwen3-Coder-30B-A3B expert distribution doesn't shard well
- **Likelihood**: low (MoE models are designed to shard)
- **Impact**: medium (may need to use a different target model)
- **Mitigation**: have fallback targets (Mixtral 8×7B, DeepSeek-MoE, other MoE variants).

### Risk 5: Nobody wants a two-device mesh
- **Likelihood**: low (developers already own multiple devices)
- **Impact**: high (product doesn't matter if no users)
- **Mitigation**: ship single-device mode too. SSD streaming alone on 16GB Mac is already useful.

---

## 11. Out of scope (deliberately)

- **Public mesh / strangers helping strangers**: v2 concern
- **Verified computation**: v2 concern
- **Tokens or economic incentives**: v2 concern
- **Windows and Linux**: v2 concern, after Mac is stable
- **Android/iOS**: v2+ concern, after desktop is stable
- **Training / fine-tuning**: not our problem; users use MLX LoRA or cloud
- **Model catalog curation**: users bring their own MoE models; we don't curate
- **Multi-tenant serving**: single user per mesh for v1

---

## 12. Reference materials for the implementer

### Papers to read in full
- Prima.cpp (arXiv:2504.08791) — understand PRP and Halda
- KVSwap (arXiv:2511.11907) — disk-backed KV cache
- CATS (arXiv:2404.08763) — activation sparsity
- Fate (arXiv:2502.12224) — expert prefetching
- LLM in a Flash (arXiv:2312.11514) — foundational Apple paper

### Codebases to study
- SwiftLM: read `ssd_streamer.mm`, `fence.air`, main loop
- Mesh-LLM: read `relay/`, `mesh-llm/src/` for iroh patterns
- Prima.cpp: read the scheduler in full
- iroh: read the echo example, then the router example

### Docs to bookmark
- iroh: https://docs.rs/iroh and https://www.iroh.computer
- MLX: https://ml-explore.github.io/mlx/
- Apple Metal Shading Language spec (for understanding `fence.air`)

---

## 13. Open questions to answer in Milestone 1

1. Does Mesh-LLM's MoE expert parallelism already work acceptably on 2× 8GB Macs?
2. If yes, how much does adding per-node SSD streaming improve it?
3. What's the real-world SSD bandwidth on M2 for random 8MB reads? (spec says 3–5 GB/s, we need to measure)
4. How often does Qwen3-Coder-30B-A3B's expert distribution shift across a coding session? (stable enough for caching to help, or does it thrash?)
5. Does iroh's NAT traversal work between a home Mac and a mobile device on cellular? (important for the "my Mac + my phone" use case)

---

## 14. Contact points

- **SwiftLM**: Eric Lake (@sharpai on GitHub)
- **Mesh-LLM**: Michael Neale (@michaelneale on GitHub, Block/Goose)
- **Prima.cpp**: Li Zonghang (@Lizonghang on GitHub, MBZUAI)
- **iroh**: n0 Computer team, Discord linked from iroh.computer

---

## 15. Repository layout (planned)

```
meshthatworks/
├── Cargo.toml              # workspace
├── README.md
├── LICENSE                 # Apache 2.0
├── crates/
│   ├── mtw-core/           # mesh coordination
│   ├── mtw-engine/         # MLX FFI + SSD streaming
│   ├── mtw-cache/          # adaptive expert caching
│   ├── mtw-api/            # OpenAI-compatible API
│   └── mtw-cli/            # binary entry point
├── mlx-fork/               # our fork of SharpAI/mlx (submodule) — carries ssd_streamer + moe_stream_op
├── swiftlm-fork/           # optional; only if we end up reusing SwiftLM's Swift-side helpers
├── docs/
│   ├── BASELINES.md        # milestone 1 deliverable
│   ├── ARCHITECTURE.md     # this document, refined
│   ├── BENCHMARKS.md       # milestone 5 deliverable
│   └── PAPER.md            # draft research paper
├── scripts/
│   ├── install.sh
│   └── benchmark.sh
└── relay/                  # optional self-hosted iroh relay
```

---

## 16. Build and run (target UX)

```
# Install
curl -fsSL https://meshthatworks.dev/install.sh | bash

# First device: start and show invite
mtw pair
> Mesh created. Invite code: mtw-7f3a9c2e

# Second device: join
mtw join mtw-7f3a9c2e
> Joined mesh. Syncing model weights...

# Both devices now part of a mesh. Use from Claude Code:
# (Claude Code autoconfigured via ~/.config/claude/providers.json)
claude
> (Claude Code runs against Qwen3-Coder-30B-A3B on your local mesh)
```

---

**End of technical specification.**

For questions, implementation concerns, or scope adjustments, see Section 13 (Open Questions) or discuss with project owner.
