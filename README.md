# MeshThatWorks

**Adaptive collective inference on heterogeneous edge devices.**

A small cluster of consumer devices, each running a **complete** small language model from a different model family. A prompt is answered by all of them in parallel. The system measures where the collective is uncertain, decides which single node is best placed to resolve that uncertainty, sends **one targeted follow-up to that node only**, and synthesizes a final answer.

Two things this is explicitly **not**:

- Not one large model sharded across several devices (tensor/pipeline parallelism).
- Not plain Mixture-of-Agents, where every node answers and an aggregator blends the results with no diagnosis and no selective second pass.

The novel component is the middle: **measure the gap, then allocate one extra inference to the node most likely to close it.**

## The claim

> A group of heterogeneous small language models, coordinated adaptively, closes a meaningful fraction of the quality gap to a much larger model — at substantially lower energy per query, with no data leaving the local network.

And the sharpened version that positions it against every existing uncertainty-routing system:

> **Every existing uncertainty-routing system assumes a stronger model exists to escalate to. This one does not.**

There is no oracle. Selection is among equals — on complementary strengths, not a capability ordering — and the extra pass costs battery on a device the user is holding. Routing among peers under an energy budget, with no stronger model available, is the contribution. The headline metric is **quality per joule**, not quality in isolation.

Full specification, mechanism, prior-work positioning, kill criteria: **[`docs/SPEC.md`](docs/SPEC.md)**.

## How the mechanism works

```
prompt
  → fan out to N nodes, K samples each
  → collect generations + per-node telemetry
  → semantic clustering (bidirectional entailment)
  → cross-node divergence           → is there a gap?
  → per-node entropy, normalized    → who resolves it?
  → cost check: is a second pass worth its energy/latency/bandwidth?
  → targeted second inference (one node)
  → synthesis (node-class model only)
```

The gap detector is built on **semantic entropy** (Farquhar et al., *Nature* 2024). Semantic entropy normally needs a 5–10× compute increase for multiple samples — but this system generates multiple answers across multiple devices in parallel regardless, so *the expensive part is free by construction*. Cross-model entropy values are never compared raw: each model's entropy distribution is z-score normalized against a held-out calibration split (Chuang et al., arXiv:2502.04428, found uncertainty distributions are model-specific — comparing raw values routes to whichever model is most overconfident).

## Staging

| Stage | What | Status |
|---|---|---|
| 1 | **Offline, single machine.** Four models via Ollama, K≈5 samples each, dump `generations.jsonl` over TriviaQA + GSM8K. Run the heterogeneity-vs-self-consistency baseline. | `harness/` — built |
| 2 | **Detector validation.** Cross-node divergence + per-node normalized entropy vs. actual error. Report AUROC. *This is the entire scientific risk and it costs nothing.* AUROC < 0.6 → stop and rethink. | `harness/` — built |
| 3 | **Routing, still offline.** Full 7-condition ablation over saved generations. One CSV, one plot. | `harness/` — built |
| web | **Live browser mesh.** Any phone/laptop opens a URL, loads a model in the browser (WebLLM/WebGPU), joins over WebSocket; the coordinator runs the real gap→select→second-pass loop. | `mesh-web/` — built |
| 4 | **Two machines via NVIDIA PAIR.** Coordinator as a client in front of PAIR's proxy. | planned |
| 5 | **Phones.** Phonon sidecar for NPU inference; Iroh + `iroh-gossip` transport; energy/thermal/bandwidth measurement. | planned |

Stages 1–3 produce a paper without a single phone purchased. Stages 4–5 make it a system.

## Try it live with your own devices

```bash
python3 -m venv .venv && .venv/bin/pip install -r harness/requirements.txt fastapi 'uvicorn[standard]' websockets
.venv/bin/python mesh-web/server.py               # coordinator on :8020
cloudflared tunnel --url http://localhost:8020    # HTTPS URL for phones (WebGPU needs it)
```

Open the URL on every device, pick a model per device (different families!), join,
and ask. Each round shows every node's answers, per-node semantic entropy, the gap
verdict, the one selected node, and the synthesized answer. Details: [`mesh-web/README.md`](mesh-web/README.md).

## Running the offline harness (stages 1–3)

```bash
cd harness
pip install -r requirements.txt

# Stage 1 — generate. Needs Ollama running with the four models pulled.
ollama pull qwen2.5:1.5b llama3.2:1b phi3:mini gemma2:2b
python generate.py --dataset triviaqa --n-questions 400
python generate.py --dataset gsm8k    --n-questions 400

# Stage 2 — detector. Fits per-model calibration on a held-out split, then AUROC.
python calibrate.py
python detector_eval.py

# Stage 3 — ablation. Seven conditions, accuracy vs. tokens, CSV + plot.
python ablate.py
```

See [`harness/README.md`](harness/README.md) for the condition table, file formats, and the kill criteria each stage checks.

## The baseline that decides everything

> **Four heterogeneous models × 1 sample** vs. **one model × 4 samples with self-consistency**, at matched total token budget.

If heterogeneity does not beat self-consistency at equal compute, the premise fails and all selection machinery is decoration. The harness runs this first (conditions 2 vs. 3), and the selected-second-pass condition must beat both the random-second-pass condition and the self-consistency baseline, or the project reframes as a negative result — which, properly measured, is publishable.

## Previous direction

This repository previously held a different system: running one large MoE model split across consumer Macs via per-node SSD expert streaming inside an iroh mesh (SwiftLM + layer-split + Tauri desktop app). That work is complete and preserved on the [`legacy-ssd-streaming`](https://github.com/mrunalpendem123/meshthatworks/tree/legacy-ssd-streaming) branch, including the shipped macOS app (see Releases). It is exactly the thing the current spec's §1 rules out — the two directions share a name and a belief in consumer hardware, not a mechanism.

## License

MIT. See [`LICENSE`](LICENSE).
