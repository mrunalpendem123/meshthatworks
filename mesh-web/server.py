"""Web-mesh coordinator (spec §6.4, live over browser nodes).

Every device — phone or laptop — opens the same page, loads a small model in the
browser (WebLLM on WebGPU, transformers.js on WASM as fallback), and joins over
WebSocket. Per round:

  1. every node writes ONE proper answer + K short one-line probes (for entropy)
  2. probes are clustered semantically across nodes → how many distinct views?
  3. if views disagree, the node with the lowest calibrated entropy becomes the
     round's synthesizer: it reads everyone's answers, notes what each adds and
     what is missing, and WRITES the final merged answer (one targeted second
     inference — never a broadcast)
  4. everyone sees the mesh's answer next to what their own device said alone

Rounds are logged to harness/data/rounds.jsonl.

  .venv/bin/python mesh-web/server.py             # http://localhost:8020
  cloudflared tunnel --url http://localhost:8020  # HTTPS for phones (WebGPU/WASM
                                                  # both want a secure context)
"""

import asyncio
import json
import statistics
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from cluster import cluster, extract_answer, semantic_entropy

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "harness" / "data"
K_PROBES = 3             # short one-line probes per node (entropy signal)
ANSWER_MAX_TOKENS = 170
PROBE_MAX_TOKENS = 32
SYNTH_MAX_TOKENS = 240
GEN_TIMEOUT = 300        # WASM phones are slow; stream progress so it feels alive
MIN_CALIB_ROUNDS = 5

app = FastAPI()


class Node:
    def __init__(self, ws: WebSocket, name: str, model: str, device: str, backend: str):
        self.ws = ws
        self.id = uuid.uuid4().hex[:8]
        self.name = name
        self.model = model
        self.device = device
        self.backend = backend
        self.status = "ready"
        self.pending: dict[str, asyncio.Future] = {}


class Mesh:
    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.viewers: set[WebSocket] = set()
        self.round_lock = asyncio.Lock()
        self.round_no = 0
        self.entropy_hist: dict[str, list[float]] = defaultdict(list)

    def roster(self) -> list[dict]:
        return [{"id": n.id, "name": n.name, "model": n.model, "device": n.device,
                 "backend": n.backend, "status": n.status} for n in self.nodes.values()]

    async def broadcast(self, msg: dict):
        data = json.dumps(msg)
        dead = []
        for ws in [n.ws for n in self.nodes.values()] + list(self.viewers):
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.viewers.discard(ws)

    async def broadcast_roster(self):
        await self.broadcast({"type": "roster", "nodes": self.roster()})

    def znorm(self, model: str, raw: float) -> tuple[float, bool]:
        hist = self.entropy_hist[model]
        if len(hist) < MIN_CALIB_ROUNDS:
            return raw, False
        mean = statistics.mean(hist)
        std = statistics.pstdev(hist) or 1e-6
        return (raw - mean) / std, True


mesh = Mesh()


def answer_prompt(question: str) -> str:
    return ("Answer the question below well, in at most 4 short sentences. "
            "Be concrete and correct; no filler, no headings.\n\n"
            f"Question: {question}\nAnswer:")


def probe_prompt(question: str) -> str:
    return ("Answer in ONE short sentence — the single most important point only."
            f"\n\nQuestion: {question}\nAnswer:")


def synthesis_prompt(question: str, entries: list[tuple[str, str]]) -> str:
    parts = "\n\n".join(f"[{model}]\n{ans[:500]}" for model, ans in entries)
    return ("You are the synthesizer for a mesh of small AI models. They answered "
            "this question independently:\n\n"
            f"Question: {question}\n\n{parts}\n\n"
            "Compare the answers: where they agree, trust them; where one adds a "
            "correct point the others missed, keep it; drop anything wrong or "
            "repeated. Then write the single best final answer — complete, "
            "accurate, at most 5 short sentences. Output ONLY the final answer.")


async def node_generate(node: Node, req: dict) -> dict | None:
    fut = asyncio.get_event_loop().create_future()
    node.pending[req["req_id"]] = fut
    try:
        await node.ws.send_text(json.dumps(req))
        return await asyncio.wait_for(fut, timeout=GEN_TIMEOUT)
    except (asyncio.TimeoutError, Exception):
        return None
    finally:
        node.pending.pop(req["req_id"], None)


async def run_round(question: str, asked_by: str):
    mesh.round_no += 1
    rid = f"r{mesh.round_no}"
    nodes = [n for n in mesh.nodes.values() if n.status == "ready"]
    if len(nodes) < 2:
        await mesh.broadcast({"type": "round", "round": rid, "phase": "error",
                              "message": "need at least 2 ready nodes"})
        return
    t0 = time.time()
    await mesh.broadcast({"type": "round", "round": rid, "phase": "fanout",
                          "question": question, "asked_by": asked_by,
                          "nodes": [{"id": n.id, "name": n.name, "model": n.model}
                                    for n in nodes]})

    # ---- first pass: 1 proper answer + K one-line probes, streamed as done ----
    async def first_pass(n: Node):
        r = await node_generate(n, {
            "type": "generate", "req_id": f"{rid}:{n.id}:first", "round": rid,
            "kind": "first",
            "answer_prompt": answer_prompt(question),
            "probe_prompt": probe_prompt(question),
            "k_probes": K_PROBES, "temperature": 1.0,
            "answer_max_tokens": ANSWER_MAX_TOKENS,
            "probe_max_tokens": PROBE_MAX_TOKENS,
        })
        await mesh.broadcast({"type": "round", "round": rid, "phase": "node_done",
                              "node": n.id, "name": n.name, "ok": r is not None,
                              "preview": (r["answer"]["text"][:90] if r else None)})
        return r

    results = await asyncio.gather(*[first_pass(n) for n in nodes])
    live = [(n, r) for n, r in zip(nodes, results) if r]
    if len(live) < 2:
        await mesh.broadcast({"type": "round", "round": rid, "phase": "error",
                              "message": "fewer than 2 nodes answered"})
        return

    # ---- analysis: cluster probes jointly, entropy per node, distinct views ----
    flat, owner = [], []
    for n, r in live:
        for p in r["probes"]:
            flat.append(extract_answer(p["text"]))
            owner.append(n.id)
    joint = await asyncio.to_thread(cluster, question, flat)
    per_node_ids = defaultdict(list)
    for cid, nid in zip(joint, owner):
        per_node_ids[nid].append(cid)

    modal = {}
    per_node = {}
    for n, r in live:
        ids = per_node_ids[n.id]
        raw = semantic_entropy(ids)
        z, calibrated = mesh.znorm(n.model, raw)
        mesh.entropy_hist[n.model].append(raw)
        pos = [c for c in ids if c >= 0]
        modal[n.id] = Counter(pos).most_common(1)[0][0] if pos else -abs(hash(n.id)) % 10**6
        per_node[n.id] = {
            "name": n.name, "model": n.model, "backend": n.backend,
            "answer": r["answer"]["text"].strip(),
            "probes": [extract_answer(p["text"]) for p in r["probes"]],
            "entropy_raw": round(raw, 3), "entropy_z": round(z, 3),
            "calibrated": calibrated,
            "tokens": r["answer"]["tokens"] + sum(p["tokens"] for p in r["probes"]),
        }

    # view label per node: nodes sharing a modal probe cluster share a view
    view_of = {}
    for nid, m in modal.items():
        view_of[nid] = m
    distinct = {m: chr(65 + i) for i, m in enumerate(dict.fromkeys(view_of.values()))}
    views = {nid: distinct[m] for nid, m in view_of.items()}
    n_views = len(distinct)
    gap = n_views > 1
    for nid in per_node:
        per_node[nid]["view"] = views[nid]
    await mesh.broadcast({"type": "round", "round": rid, "phase": "analysis",
                          "per_node": per_node, "n_views": n_views, "gap": gap})

    # ---- synthesis: ONE selected node reads everyone and writes the answer ----
    selected = min(live, key=lambda nr: (per_node[nr[0].id]["entropy_z"], nr[0].id))[0]
    synth_tokens = 0
    if gap:
        await mesh.broadcast({"type": "round", "round": rid, "phase": "second_pass",
                              "selected": selected.id, "name": selected.name,
                              "reason": "most self-consistent node this round"})
        entries = [(per_node[n.id]["model"], per_node[n.id]["answer"]) for n, _ in live]
        r2 = await node_generate(selected, {
            "type": "generate", "req_id": f"{rid}:{selected.id}:second", "round": rid,
            "kind": "second",
            "answer_prompt": synthesis_prompt(question, entries),
            "k_probes": 0, "temperature": 0.0,
            "answer_max_tokens": SYNTH_MAX_TOKENS, "probe_max_tokens": 0,
        })
        if r2:
            final = r2["answer"]["text"].strip()
            synth_tokens = r2["answer"]["tokens"]
            final_how = "synthesized"
        else:
            final = per_node[selected.id]["answer"]
            final_how = "selected (synthesis timed out)"
    else:
        final = per_node[selected.id]["answer"]
        final_how = "consensus"

    total_tokens = sum(a["tokens"] for a in per_node.values()) + synth_tokens
    record = {"round": rid, "ts": time.time(), "question": question,
              "final": final, "final_by": {"id": selected.id, "name": selected.name,
                                           "model": selected.model},
              "final_how": final_how, "gap": gap, "n_views": n_views,
              "per_node": per_node, "elapsed_s": round(time.time() - t0, 1),
              "total_tokens": total_tokens}
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / "rounds.jsonl", "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    await mesh.broadcast({"type": "round", "round": rid, "phase": "final", **record})


async def locked_round(question: str, asked_by: str):
    async with mesh.round_lock:
        try:
            await run_round(question, asked_by)
        except Exception as e:
            await mesh.broadcast({"type": "round", "round": f"r{mesh.round_no}",
                                  "phase": "error", "message": str(e)})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    node: Node | None = None
    mesh.viewers.add(ws)
    await ws.send_text(json.dumps({"type": "roster", "nodes": mesh.roster()}))
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            t = msg.get("type")
            if t == "register":
                mesh.viewers.discard(ws)
                node = Node(ws, msg.get("name") or "anon", msg["model"],
                            msg.get("device", "?"), msg.get("backend", "?"))
                mesh.nodes[node.id] = node
                await ws.send_text(json.dumps({"type": "registered", "node_id": node.id}))
                await mesh.broadcast_roster()
            elif t == "status" and node:
                node.status = msg.get("status", "ready")
                await mesh.broadcast_roster()
            elif t == "generation" and node:
                fut = node.pending.get(msg.get("req_id", ""))
                if fut and not fut.done():
                    fut.set_result(msg)
            elif t == "ask":
                q = (msg.get("question") or "").strip()
                if not q:
                    continue
                if mesh.round_lock.locked():
                    await ws.send_text(json.dumps({"type": "error",
                                                   "message": "a round is already running"}))
                    continue
                asker = node.name if node else "viewer"
                # background task: the asker may itself be a node, and its
                # generation replies arrive on this same receive loop
                asyncio.create_task(locked_round(q, asker))
    except WebSocketDisconnect:
        pass
    finally:
        mesh.viewers.discard(ws)
        if node:
            mesh.nodes.pop(node.id, None)
            for fut in node.pending.values():
                if not fut.done():
                    fut.cancel()
            await mesh.broadcast_roster()


@app.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


if __name__ == "__main__":
    import os
    import threading

    if os.environ.get("MTW_NLI") != "0":
        def _warm():
            try:
                from cluster import _nli_equivalent
                _nli_equivalent("warmup", "yes", "no")
                print("[nli] entailment model ready")
            except Exception as e:
                print(f"[nli] warmup failed, lite equivalence only: {e}")
        threading.Thread(target=_warm, daemon=True).start()

    print("mesh-web coordinator on http://localhost:8020")
    print("phones need HTTPS → cloudflared tunnel --url http://localhost:8020")
    uvicorn.run(app, host="0.0.0.0", port=8020, log_level="warning")
