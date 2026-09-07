# mesh-web — join the mesh from any browser

The live testing harness (spec §6.3's browser harness, taken further): every
phone or laptop opens one URL, loads a small model **in the browser** via WebLLM
(WebGPU), and becomes a mesh node over WebSocket. This server runs the real
mechanism from §6.4 — fan-out, semantic clustering, gap detection, one targeted
second pass to the node with the lowest calibrated entropy, synthesis.

This validates the mechanism and the coordination, not latency or energy —
those need stages 4–5 (real transports, NPUs, power meters).

## Run

```bash
# from the repo root (one-time: python3 -m venv .venv && .venv/bin/pip install -r harness/requirements.txt fastapi 'uvicorn[standard]' websockets)
.venv/bin/python mesh-web/server.py        # http://localhost:8020
```

- **Laptops on the same machine/LAN**: open `http://localhost:8020` (or `http://<your-ip>:8020` —
  Chrome treats plain-HTTP LAN pages as insecure, so WebGPU may be blocked; the tunnel below fixes that).
- **Phones**: WebGPU needs a secure context, so give them HTTPS with one command:

  ```bash
  cloudflared tunnel --url http://localhost:8020
  ```

  Open the printed `https://….trycloudflare.com` URL on each phone. Works from
  anywhere, not just your Wi-Fi.

On each device: enter a name, pick a model sized for the device (SmolLM2-360M /
Qwen-0.5B for phones, 1B–3B for laptops — pick a *different family* per device),
hit **load model & join mesh** (first load downloads the model into browser
cache), then ask a question from any device. Short-answer questions show the
mechanism best.

What you'll see per round: every node's greedy answer + samples, its semantic
entropy, whether a gap was detected (greedy answers in >1 semantic cluster), which
node was selected for the one targeted second pass, and the synthesized final
answer with total token cost.

## Notes

- **Clustering**: normalized-string / containment / numeric equivalence, upgraded
  by bidirectional NLI (deberta-large-mnli, downloads ~1.6 GB on first server
  start; MPS/CUDA if available). `MTW_NLI=0` disables NLI.
- **Calibration**: per-model z-normalization warms up from round history
  (`(uncalib)` shows until a model has 5 rounds; before that raw entropies are
  compared, which §3.2 warns about — fine for plumbing tests, not for claims).
- **Rounds are logged** to `harness/data/rounds.jsonl` for later analysis.
- `?mock=1` on the URL joins a fake instant node (no WebGPU needed) — for testing
  the coordination without model loads. `mock_node.py` does the same headlessly:

  ```bash
  .venv/bin/python mesh-web/mock_node.py --name a --answers "Paris,Paris,Paris,Lyon"
  ```
- Need ≥2 ready nodes to run a round; one round at a time.
