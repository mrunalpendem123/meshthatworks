# mesh-web — join the mesh from any browser

The live testing harness (spec §6.3's browser harness, taken further): every
phone or laptop opens one URL, loads a small model **in the browser**, and
becomes a mesh node over WebSocket. Two backends, picked automatically:

- **WebLLM (WebGPU)** — fast, on devices with GPU access in the browser
- **transformers.js (WASM)** — slower, but runs on *any* browser, including
  phones without WebGPU

Per round the coordinator runs the §6.4 loop, tuned for live open-ended use:

1. every node writes **one proper answer** plus K=3 **short one-line probes**
   (the probes are the cheap entropy signal — full-length samples were 4× slower)
2. probes are clustered semantically across nodes → how many distinct views?
3. on disagreement, the node with the lowest calibrated entropy becomes the
   round's **synthesizer**: it reads everyone's answers, keeps agreed points and
   unique correct additions, drops what's wrong, and *writes* the final merged
   answer — one targeted second inference, never a broadcast
4. every device sees **the mesh's answer next to what it said alone**

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

On each device: enter a name, accept the suggested model (the page auto-suggests
a model *family not yet in the mesh* — heterogeneity is where the value comes
from), hit **load model & join mesh**, then ask from any device. Devices without
WebGPU automatically get the no-GPU (WASM) model list.

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
