# Adaptive Collective Inference on Heterogeneous Edge Devices

*Working title — placeholder. Project specification, v3.*

**Changes in v3:** §6.5 added — networking layer specified. Iroh is adopted as the stage-5 transport; the earlier blanket rejection was of `prime-iroh` specifically, not Iroh, and that distinction is now explicit. §8 stage 5 updated.

**Changes in v2:** §3.2 corrected (cross-model entropy requires per-model calibration — v1's routing rule was wrong as written). §2.1 added (sharpened novelty claim). §4 expanded with four papers that contest the original framing. §6.3 revised (NVIDIA PAIR replaces Phonon for the offline stage). §3.4 added (quantization × calibration). §8 revised.

---

## 1. What this is

A small cluster of consumer phones, each running a **complete** small language model from a different model family. A prompt is answered by all of them in parallel. The system then measures where the collective is uncertain, decides which single node is best placed to resolve that uncertainty, sends one targeted follow-up to that node only, and synthesizes a final answer.

Two things this is explicitly **not**:

- Not one large model sharded across several phones (tensor/pipeline parallelism).
- Not plain Mixture-of-Agents, where every node answers and an aggregator blends the results with no diagnosis and no selective second pass.

The novel component is the middle: **measure the gap, then allocate one extra inference to the node most likely to close it.**

---

## 2. The claim

> A group of heterogeneous small language models, coordinated adaptively, closes a meaningful fraction of the quality gap to a much larger model — at substantially lower energy per query, with no data leaving the local network.

**The claim is not** that four phones beat a frontier model on answer quality. They will not. The gap between a 1B model and a frontier model is a capability gap, not a coordination problem.

The defensible headline metric is **quality per joule**, not quality in isolation.

### 2.1 The sharpened novelty claim

*This is the framing that survives contact with the related work in §4, and it should lead every abstract and proposal.*

> **Every existing uncertainty-routing system assumes a stronger model exists to escalate to. This one does not.**

Chuang et al. route SLM → stronger LLM. MOSAIC runs on a 4-GPU server. FrugalGPT, Hybrid LLM, and RouteLLM all route weak → strong. The entire literature is built on the premise that when the small model is unsure, something better is available.

Remove that premise and the problem changes shape:

- There is no oracle. The correct answer may exist in **no** member of the ensemble.
- Selection is **among equals** — you are choosing on complementary strengths, not on a capability ordering.
- The escalation decision becomes a genuine cost/benefit optimization rather than "escalate when unsure," because the extra pass costs battery on a device the user is holding.

Routing among peers under an energy budget, with no stronger model available, is the contribution. Everything else is assembly.

### Why it might work

Mixture-of-Agents (Wang et al., 2024) established that coordination can beat a larger single model: their mixture of open-source LLMs reached 65.1% on AlpacaEval 2.0 against 57.5% for GPT-4 Omni, and responses from *heterogeneous* models contributed significantly more than repeated samples from the same model.

### Why it might not

That result used 70B-class models on datacenter GPUs. Nobody has shown the effect survives when the members shrink to phone-size. A 1B model may not be able to recognize a 1B model's error. **Establishing this cleanly in either direction is the contribution.** A negative result, properly measured, is publishable.

---

## 3. Mechanism

### 3.1 Uncertainty measurement

The gap detector is built on **semantic entropy** (Farquhar, Kossen, Kuhn & Gal, *Nature*, 2024): cluster generations by meaning using bidirectional entailment, then compute entropy over the semantic clusters.

Three properties make it right here:

1. **The authors' own application list includes "get extra help when answering questions that are likely to be answered wrong."** That is this architecture, described by the people who invented the measurement.
2. **The cost objection does not apply.** Semantic entropy normally needs a 5–10× compute increase for multiple samples. This system generates multiple answers across multiple devices in parallel regardless. *The expensive part is free by construction — this belongs in the abstract.*
3. **The entailment model is phone-sized** (~400M, DeBERTa-scale). No large model hides inside the small-model system.

### 3.2 Cross-model entropy requires per-model calibration

**This section replaces v1's routing rule, which was wrong.**

Farquhar samples K generations from a **single** model. The entropy measures that model's uncertainty, and the paper's calibration evidence is for that quantity.

Sampling once from each of four different models gives **cross-model disagreement** — a different quantity. High disagreement can occur when one model is right and three are wrong, a case where the ensemble is fine and the detector fires anyway.

Worse, and this is the part v1 got wrong: Chuang et al. (arXiv 2502.04428) found across 1500+ settings that **uncertainty distributions depend more on the specific SLM and the chosen UQ method than on the downstream data.** Each of your four models has a differently *shaped* uncertainty distribution. Comparing raw entropy values across nodes therefore routes systematically to whichever model is most overconfident — not to the one most likely to be right.

**Corrected design:**

- Sample K ≈ 5 per node (≈20 generations across four devices).
- **Cross-node cluster divergence** → detects *that* a gap exists.
- **Per-node semantic entropy, z-score normalized per model against a held-out calibration split** → determines *who to ask*. Only normalized values may be compared across nodes.
- Fit normalization once per model on held-out data. Never on the evaluation set.

Chuang et al. also open-sourced a calibration hold-out set and a calibration-data construction pipeline; use it rather than building one.

Their third finding is a warning to take seriously: simple confidence measures performed well, and they conclude in favour of robust simple solutions over complex routing mechanisms. This architecture is the complex kind. The §5.2 ablations must show the complexity earns its keep.

*Cheaper alternative worth testing:* Semantic Entropy Probes (arXiv 2406.15927) approximate semantic entropy from a single generation via linear probes on hidden states. Cheaper, but needs access to internals — a constraint under some mobile runtimes.

### 3.3 Aggregation constraint

The final synthesis model **must be the same size class as the proposers**, and the spec must report which node performed it. MoA's gains almost certainly depend in part on a strong aggregator. If synthesis calls anything larger than a node model, a large model has been smuggled into a small-model system and the result is void.

### 3.4 Secondary research question: does calibration survive quantization?

*Added in v2. Testable on a laptop, and it is the thread with the clearest hardware relevance.*

To run on an NPU, models are quantized to INT8/INT4. Quantization changes a model's output distribution — and semantic entropy is a measurement *of* that distribution. If quantized models are miscalibrated, the gap detector degrades exactly where it must work.

Test: compare entropy-vs-error AUROC for FP16, INT8, and INT4 of the same model. If calibration degrades, that is a finding with a concrete correction attached ("uncertainty-based routing requires quantization-aware calibration"). If it doesn't, that is a reassuring result worth reporting. As far as I can establish, this is unexamined.

---

## 4. Positioning against prior work

*v2 note: the field is considerably more crowded than v1 assumed. Each of the first four rows claims a component of the original framing. §2.1 exists because of them.*

| Work | What it claims | What it leaves to you |
|---|---|---|
| **Confident or Seek Stronger** (Chuang et al., arXiv 2502.04428) | **The closest paper to this mechanism.** Uncertainty-based routing from on-device SLMs to stronger LLMs, benchmarked over 1500+ settings. Establishes that UQ-method choice dominates routing performance and that uncertainty distributions are model-specific. | Assumes a stronger model exists. Single device, single SLM, no peer structure, no energy budget. Their calibration insight is a dependency, not competition — cite it and build on it. |
| **MOSAIC** (arXiv 2606.03014) | MoA scheduling with a confidence-aware adaptive aggregation gate that bypasses the aggregator when experts converge; ILP-based expert-to-worker placement; benchmarked against round-robin on a 4-GPU server. | Your §6.4 cost check, already published — but in a datacenter, optimizing wall-clock, with no energy term and no device constraints. Cite as the closest systems precedent and ablate against it. |
| **Symbolic-MoE** (Chen et al.) | Adaptive instance-level skill-based routing across heterogeneous models, with task-specific aggregator selection. | Skill profiling is done offline from a model pool. Yours is computed per-query from measured uncertainty, on devices with battery state. |
| **MMoA** (arXiv 2605.19194) | Recurrent MoA with LSTM gating; dynamically activates fewer agents to cut overhead. | Learned gating over a static pool; no uncertainty semantics, no physical substrate. |
| **Mixture-of-Agents** (arXiv 2406.04692) | Layered heterogeneous collaboration beats the best single model; heterogeneity beats repetition. | Datacenter-scale. Static proposer sets, every node in every layer. No uncertainty measurement, no selective allocation, no energy accounting. |
| **Distributed MoA** (Mitra, Kaswan & Ulukus, arXiv 2412.21200) | MoA on independent edge devices with decentralized gossip; queuing-stability conditions bounding average queue size. | Peer selection is uniform random (`random.sample`), k fixed. The released code runs 70B models on one GPU and *simulates* distributed timing in post-processing — no networking, no on-device execution. |
| **Semantic entropy** (Farquhar et al., *Nature* 2024) | Meaning-level uncertainty detects confabulation; explicitly proposed as a trigger for seeking extra help. | Single-model, single-device. Never a routing signal across a device network, never under an energy budget. |
| **LLM-Based Multi-Agent Blackboard System** (Salemi et al., arXiv 2510.01285) | Agents volunteering on self-assessed capability beat master–slave control by 13–57% relative — a central coordinator cannot hold precise knowledge of every sub-agent's expertise. | **A direct challenge.** Their setting has no energy, thermal, or bandwidth constraints — and those are precisely what a node cannot self-assess in isolation. Include a volunteer-mode arm as condition 7 rather than assuming the coordinator wins. |

**Adjacent, context only:** PicoSpec (arXiv 2603.19133) — heterogeneous edge–cloud collaboration via asynchronous speculative decoding, up to 2.9× speedup. PolyLink (arXiv 2510.02395) — decentralized crowdsourced inference with verification and incentives.

**Not prior work, despite surface similarity.** AntSeed, OpenHydra, Warden, Gensyn, Prime Intellect, Hyperbolic and PolyLink are all about *distributing compute* — moving a workload to hardware with capacity. This project is about *coordinating reasoning*. Same vocabulary, no shared mechanism. Do not spend reading time here.

**Framing note:** Google and UC San Diego are building a datacenter from roughly 2,000 Pixel phones, expected online fall 2026. Better opening than "phones are cheap." NVIDIA also released PAIR in September 2026 — a local-network inference router that explicitly routes each request to one node without splitting models — which is direct industrial validation of the one-complete-model-per-node premise.

---

## 5. Experimental design

### 5.1 The baseline that decides everything

Run this **first**.

> **Four heterogeneous models × 1 sample** vs. **one model × 4 samples with self-consistency**, at matched total token budget.

This isolates model diversity from extra sampling. If heterogeneity does not beat self-consistency at equal compute, the premise fails and all selection machinery is decoration.

### 5.2 Ablation set

1. Single model, 1× budget (floor)
2. Single model, 4× budget, self-consistency ← **the real baseline**
3. Four models, first pass only
4. Four models + second pass to **all** nodes
5. Four models + second pass to a **random** node
6. Four models + second pass to the **selected** node ← the contribution
7. Volunteer mode — nodes self-nominate (the blackboard arm)

Condition 6 must beat 5, or selection is decoration. It must beat 2, or the project fails.

### 5.3 Benchmarks

**Primary — checkable ground truth only.** An uncertainty measure cannot be validated against a judge with no ground truth.

- TriviaQA (matches Farquhar's validation setting)
- GSM8K (multi-step reasoning, unambiguous correctness)
- A domain set where model families are expected to diverge (code vs. multilingual vs. commonsense)

**Secondary, comparability only:** AlpacaEval 2.0, to sit on the MoA axis. LLM-judge-based and length-biased; MoA themselves report conciseness as their weakest metric. Never the primary signal.

### 5.4 Metrics

- Task accuracy per condition
- **Energy per query (joules)** — headline axis
- End-to-end latency including coordination overhead
- Bytes transferred between nodes
- **Detector calibration: AUROC of the gap signal against actual error.** Report independently — it stands alone as a finding, and it is the kill criterion.
- Second-pass trigger rate
- Per-model entropy distribution shape (evidence for §3.2)

### 5.5 Energy measurement

- Android `batterystats` / Battery Historian gives attribution, but values are **modelled, not measured**, and noisy at single-inference granularity.
- An **inline USB power meter per device** is cruder but yields actual joules. For quality-per-watt, measured beats modelled.

Confounds:

- **Thermal throttling** drifts tokens/sec over long runs. Randomize question order and **interleave conditions**.
- **Battery-aware degradation.** Phonon's sidecar reduces load when unplugged below 20% battery. Pin devices to a fixed power and thermal state; log temperature throughout.

---

## 6. System architecture

### 6.1 Models

Phone-class shortlist, confirmed by what ships in production on Android (Nataris runs Qwen 2.5 0.5B, Llama 3.2 1B, Phi-3 Mini 3.8B on Android 8.0+, 4GB RAM minimum, 6–8GB recommended). Phi-3 Mini is roughly the ceiling.

- Qwen 2.5 (0.5B / 1.5B)
- Llama 3.2 (1B / 3B)
- Phi-3 Mini
- Gemma family
- LFM2.x

**Start homogeneous.** Same model on all four nodes first, so any gain is attributable to coordination rather than model differences. Introduce heterogeneity only after that baseline exists.

### 6.2 Devices and runtime

3–4 Android phones, Android 14+, NPU-equipped (Pixel Tensor or Qualcomm Hexagon). LiteRT-LM for NPU execution; llama.cpp as fallback. Local Wi-Fi or Wi-Fi Direct. **No blockchain in v1.**

### 6.3 Substrate — revised in v2

**Offline and laptop stages: NVIDIA PAIR** (Apache 2.0, released September 2026). It discovers machines via mDNS, secures with mTLS, schedules on node readiness, engine state, model presence and GPU utilization, and presents Ollama- and OpenAI-compatible endpoints. Critically, **PAIR routes each request to one node — it does not pool GPU memory or split a model across systems**, which is exactly this project's premise.

Two consequences. First, you can run stages 1–3 against a maintained, cross-platform, industrially-backed router instead of alpha software. Second, PAIR's stock routing is **capacity-based**, which gives you a strong published baseline: *PAIR routes by capacity; this routes by reasoning gap.*

Build your coordinator as a **client in front of PAIR's proxy** — fan out, collect, compute entropy, issue the targeted follow-up, synthesize. Do not modify PAIR's Go internals until the result exists.

**Phone stage: Phonon** (`chezgoulet/phonon`, MIT). PAIR is Windows/macOS/Linux only, so the phone path still needs Phonon's Kotlin sidecar, which runs LiteRT-LM on Tensor and Hexagon NPUs and reports battery, temperature and processing state every 30 seconds. Caveats: alpha, single maintainer, no maintenance guarantee. Its in-development **Shard Mode** (pipelined-ring parallelism across phones) is explicitly *not* this architecture.

**Rejected substrates.** `prime-pipeline` and `prime-vllm` are model sharding — the thing §1 rules out. `prime-iroh` is rejected too, but see §6.5: the objection is to that wrapper, not to Iroh.

**Optional: browser harness.** The mechanism layer — clustering, entropy, routing, ablations — needs no GPU and can run as a static page over a dumped `generations.jsonl`. Iterating on the entropy formulation in seconds rather than minutes is worth the afternoon it costs. Live in-browser inference via WebLLM (WebGPU) is possible but cannot validate latency or energy, since one tab has one GPU.

### 6.4 Coordinator responsibilities

```
prompt
  → fan out to N nodes, K samples each
  → collect generations + per-node telemetry
  → semantic clustering (bidirectional entailment)
  → cross-node divergence           → is there a gap?
  → per-node entropy, normalized    → who resolves it?     [§3.2]
  → cost check: is a second pass worth its energy/latency/bandwidth?
  → targeted second inference (one node)
  → synthesis (node-class model only)
```

The cost check is not garnish. It converts a heuristic into an explicit optimization: **answer quality vs. compute + battery + communication + latency.** It is also the piece MOSAIC has already published in a datacenter setting, so the energy term is what makes yours distinct.

### 6.5 Networking — added in v3

**Stages 1–4: none required.** One machine, then two machines on a LAN. mDNS plus plain HTTP is sufficient, and PAIR already provides it (mDNS discovery, mTLS). No transport work belongs in these stages — nothing in the §5.2 ablation table changes when the transport changes, so it buys no result.

**Stage 5: Iroh** (`n0-computer/iroh`). QUIC with authenticated end-to-end encryption, endpoints addressed by Ed25519 key rather than IP, direct connections via hole-punching with relay fallback when no direct path exists.

The topology maps cleanly. Iroh's `Router` dispatches incoming connections to protocol handlers by ALPN string, so several protocols share one endpoint:

- **`iroh-gossip`** for the broadcast first pass — publish-subscribe message broadcast to groups of endpoints by topic.
- **A custom QUIC protocol** on the same endpoint for the targeted second-pass unicast.

Two protocols, one endpoint, exactly the two message patterns this system needs.

`iroh-gossip` is also the right fit for the hardware: it is built to scale requiring only the resources an average phone can handle, and its PlumTree active/passive membership split was chosen specifically because star topologies that compensate for high network churn with many connections are not viable on mobile. The n0 team has run 2000-node gossip stress tests.

**Why not `prime-iroh`.** Its `Node` sends to exactly one peer and receives from exactly one peer — a ring, built for pipeline parallelism. This project needs broadcast plus targeted unicast. Use Iroh directly.

**Integration cost.** Iroh is Rust. On the phone side that is fine (the sidecar is Kotlin/JNI territory anyway). For a Python coordinator, `iroh-ffi` provides bindings, and there are Dart and Go ports, but Python is not first-class — budget real time for this, and do it only once stages 1–4 have produced a result.

**Payoff.** Iroh earns its place when the four devices are on *different* networks — one on cellular, one on home Wi-Fi — rather than a single LAN. That is where NAT traversal stops being overhead and becomes the only thing that works, and "four phones anywhere" is a materially stronger demo than "four phones on a desk." It also aligns the implementation with Distributed MoA's gossip framing, which that paper describes but implements as `random.sample` in a simulation.

---

## 7. Risks and kill criteria

| Risk | Test | Kill criterion |
|---|---|---|
| Small models can't detect each other's errors | AUROC of gap signal vs. actual error | Near-chance AUROC → detector fails; reframe as a negative result |
| Cross-model entropy is miscalibrated | Per-model entropy distributions; normalized vs. raw routing | Handle via §3.2. If normalization doesn't fix it, selection cannot work |
| Heterogeneity isn't load-bearing | Condition 6 vs. 5 | No significant difference → selection is decoration; simplify to broadcast |
| Coordination loses to more sampling | Condition 1 vs. 2 | Self-consistency wins at equal budget → premise fails |
| Simple beats complex (Chuang et al.'s finding) | Condition 6 vs. a naive max-confidence rule | If a one-line heuristic matches the full mechanism, publish the heuristic |
| Energy overhead swamps the gain | Joules per query, all conditions | Adaptive pass costs more than it saves → quality-per-watt claim is dead |
| Quantization breaks calibration | AUROC at FP16 / INT8 / INT4 | Not a kill — a finding either way (§3.4) |

---

## 8. Staging

1. **Offline, single machine.** Four models via Ollama, K=5 samples, dump `generations.jsonl` over TriviaQA and GSM8K. Run §5.1. No network, no PAIR.
2. **Detector validation.** Both entropy signals plus per-model normalization. Report AUROC. **This is the entire scientific risk and it costs nothing.** If AUROC < 0.6, stop and rethink before going further.
3. **Routing, still offline.** Full ablation set over saved generations. One CSV, one plot: accuracy vs. tokens, seven conditions.
4. **Two machines via PAIR.** Coordinator as a client in front of PAIR's proxy. Real latency, real concurrency.
5. **Phones.** Phonon sidecar for NPU inference; Iroh + `iroh-gossip` for transport (§6.5); energy, thermal, bandwidth measurement. Optionally split into 5a (all devices on one LAN) and 5b (devices on separate networks), since 5a needs no NAT traversal and 5b is what justifies Iroh.

Stages 1–3 produce a paper without a single phone purchased. Stages 4–5 make it a system.

---

## 9. References

- Chuang, Y.-N., Yu, L., Wang, G., Zhang, L., Liu, Z., Cai, X., Sui, Y., Braverman, V. & Hu, X. *Confident or Seek Stronger: Exploring Uncertainty-Based On-device LLM Routing From Benchmarking to Generalization.* arXiv:2502.04428.
- Farquhar, S., Kossen, J., Kuhn, L. & Gal, Y. *Detecting hallucinations in large language models using semantic entropy.* Nature 630(8017):625–630, 2024.
- Kossen, J. et al. *Semantic Entropy Probes.* arXiv:2406.15927.
- Wang, J., Wang, J., Athiwaratkun, B., Zhang, C. & Zou, J. *Mixture-of-Agents Enhances Large Language Model Capabilities.* arXiv:2406.04692. Code: `togethercomputer/moa` (Apache 2.0).
- Mitra, P., Kaswan, P. & Ulukus, S. *Distributed Mixture-of-Agents for Edge Inference with Large Language Models.* arXiv:2412.21200. Code: `purbeshmitra/distributed_moa`.
- Salemi, A. et al. *LLM-Based Multi-Agent Blackboard System for Information Discovery in Data Science.* arXiv:2510.01285.
- *MOSAIC: Efficient Mixture-of-Agent Scheduling via Adaptive Aggregation and Inference Concurrency.* arXiv:2606.03014.
- *MMoA: An AI-Agent framework with recurrence for Memoried Mixture-of-Agent.* arXiv:2605.19194.
- Chen et al. *Symbolic Mixture-of-Experts: Adaptive Skill-based Routing for Heterogeneous Reasoning.*
- Zhang, Y. et al. *PicoSpec.* arXiv:2603.19133.
- *PolyLink.* arXiv:2510.02395. Code: `IMCL-PolyLink/PolyLink`.
- NVIDIA Personal AI Router (PAIR), Apache 2.0 — `developer.nvidia.com/blog/nvidia-pair-virtual-inference-router-expands-available-compute-on-your-local-network/`
- Phonon — `chezgoulet/phonon`, MIT.
- Iroh — `n0-computer/iroh` (n0, inc.). QUIC + NAT traversal. Protocols: `iroh-gossip`, `iroh-blobs`, `iroh-docs`. Bindings: `iroh-ffi`. Docs: `docs.iroh.computer`.

**Cited in earlier notes, not independently verified — check before use:** Khoshsirat, Perin & Rossi, *Decentralized LLM Inference over Edge Networks with Energy Harvesting* (arXiv:2408.15907); *Demystifying Small Language Models for Edge Deployment* (ACL 2025). Symbolic-MoE's arXiv ID and venue also need confirming.
