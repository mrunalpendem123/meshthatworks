"""Web-mesh coordinator (spec §6.4, live over browser nodes).

Every device — phone or laptop — opens the same page, loads a small model in the
browser via WebLLM (WebGPU), and joins over WebSocket. This process runs the
mechanism: fan out, cluster, measure the gap, pick ONE node for the targeted
second pass, synthesize. Rounds are logged to data/rounds.jsonl.

  .venv/bin/python mesh-web/server.py            # http://localhost:8020
  cloudflared tunnel --url http://localhost:8020  # HTTPS URL for phones (WebGPU
                                                  # needs a secure context)
"""

import asyncio
import json
import statistics
import time
import uuid
from collections import defaultdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from cluster import cluster, extract_answer, semantic_entropy

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "harness" / "data"
K_SAMPLES = 3            # per node per round — phones are slow, keep live K small
GEN_TIMEOUT = 240        # seconds to wait for a node's generations
MIN_CALIB_ROUNDS = 5     # rounds of history before z-normalization kicks in

app = FastAPI()


class Node:
    def __init__(self, ws: WebSocket, name: str, model: str, device: str):
        self.ws = ws
        self.id = uuid.uuid4().hex[:8]
        self.name = name
        self.model = model
        self.device = device
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
                 "status": n.status} for n in self.nodes.values()]

    async def broadcast(self, msg: dict):
        dead = []
        for ws in [n.ws for n in self.nodes.values()] + list(self.viewers):
            try:
                await ws.send_text(json.dumps(msg))
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


def first_pass_prompt(question: str) -> str:
    return ("Answer as briefly as possible — just the answer itself, one line, "
            f"no explanation.\n\nQ: {question}\nA:")


def second_pass_prompt(question: str, candidates: list[str]) -> str:
    cands = "\n".join(f"- {c}" for c in dict.fromkeys(candidates))
    return ("Different answerers gave these candidate answers:\n" + cands +
            "\n\nThink about which is correct (or give a better one), then reply "
            f"with just the final answer on one line.\n\nQ: {question}\nA:")


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
                          "nodes": [n.id for n in nodes], "k": K_SAMPLES})

    # ---- first pass: every node, greedy + K samples, in parallel ----
    reqs = [node_generate(n, {
        "type": "generate", "req_id": f"{rid}:{n.id}:first", "round": rid,
        "kind": "first", "prompt": first_pass_prompt(question),
        "k_samples": K_SAMPLES, "temperature": 1.0, "max_tokens": 80,
    }) for n in nodes]
    results = await asyncio.gather(*reqs)

    live: list[tuple[Node, dict]] = []
    for n, r in zip(nodes, results):
        if r is None:
            await mesh.broadcast({"type": "round", "round": rid, "phase": "node_timeout",
                                  "node": n.id})
        else:
            live.append((n, r))
    if len(live) < 2:
        await mesh.broadcast({"type": "round", "round": rid, "phase": "error",
                              "message": "fewer than 2 nodes answered"})
        return

    # ---- analysis: joint clustering, per-node entropy, gap ----
    flat, owner = [], []
    for n, r in live:
        for s in r["samples"]:
            flat.append(extract_answer(s["text"]))
            owner.append(("sample", n.id))
        flat.append(extract_answer(r["greedy"]["text"]))
        owner.append(("greedy", n.id))
    joint = await asyncio.to_thread(cluster, question, flat)

    per_node_ids = defaultdict(list)
    greedy_cid, greedy_ans = {}, {}
    for cid, (kind, nid), ans in zip(joint, owner, flat):
        if kind == "sample":
            per_node_ids[nid].append(cid)
        else:
            greedy_cid[nid] = cid
            greedy_ans[nid] = ans

    analysis = {}
    for n, r in live:
        raw = semantic_entropy(per_node_ids[n.id])
        z, calibrated = mesh.znorm(n.model, raw)
        mesh.entropy_hist[n.model].append(raw)
        analysis[n.id] = {
            "name": n.name, "model": n.model,
            "greedy_answer": greedy_ans[n.id],
            "sample_answers": [extract_answer(s["text"]) for s in r["samples"]],
            "entropy_raw": round(raw, 3), "entropy_z": round(z, 3),
            "calibrated": calibrated,
            "tokens": r["greedy"]["tokens"] + sum(s["tokens"] for s in r["samples"]),
        }
    n_clusters = len(set(greedy_cid.values()))
    gap = n_clusters > 1
    await mesh.broadcast({"type": "round", "round": rid, "phase": "analysis",
                          "per_node": analysis, "n_greedy_clusters": n_clusters,
                          "gap": gap})

    # ---- targeted second pass: ONE node, lowest (z-normalized) entropy ----
    second = None
    selected_id = None
    if gap:
        selected = min(live, key=lambda nr: (analysis[nr[0].id]["entropy_z"], nr[0].id))[0]
        selected_id = selected.id
        candidates = [a for a in greedy_ans.values() if a]
        await mesh.broadcast({"type": "round", "round": rid, "phase": "second_pass",
                              "selected": selected.id,
                              "reason": "lowest normalized semantic entropy"})
        r2 = await node_generate(selected, {
            "type": "generate", "req_id": f"{rid}:{selected.id}:second", "round": rid,
            "kind": "second", "prompt": second_pass_prompt(question, candidates),
            "k_samples": 0, "temperature": 0.0, "max_tokens": 120,
        })
        if r2:
            second = {"node": selected.id, "answer": extract_answer(r2["greedy"]["text"]),
                      "tokens": r2["greedy"]["tokens"]}

    # ---- synthesis: weighted cluster vote (greedy=1, second pass=2) ----
    votes = [(greedy_ans[n.id], 1.0) for n, _ in live]
    if second and second["answer"]:
        votes.append((second["answer"], 2.0))
    vote_answers = [v[0] for v in votes]
    vote_ids = await asyncio.to_thread(cluster, question, vote_answers)
    weight = defaultdict(float)
    for (a, w), cid in zip(votes, vote_ids):
        weight[cid] += w
    win = max(weight, key=lambda c: weight[c])
    final = next(a for (a, _), cid in zip(votes, vote_ids) if cid == win and a)

    total_tokens = sum(a["tokens"] for a in analysis.values()) + (second["tokens"] if second else 0)
    record = {"round": rid, "ts": time.time(), "question": question, "final": final,
              "gap": gap, "selected": selected_id, "second": second,
              "per_node": analysis, "elapsed_s": round(time.time() - t0, 1),
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
                            msg.get("device", "?"))
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
                # Run in a background task: the asker may itself be a node, and its
                # generation replies arrive on this same receive loop.
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
    print("phones need HTTPS for WebGPU → cloudflared tunnel --url http://localhost:8020")
    uvicorn.run(app, host="0.0.0.0", port=8020, log_level="warning")
