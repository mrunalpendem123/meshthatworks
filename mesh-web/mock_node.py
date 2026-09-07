"""Headless mock node for testing the coordinator without browsers.

  python mock_node.py --name a --answers "Paris,Paris,Paris,Paris"
  python mock_node.py --name b --answers "Paris,London,Paris,London"

--answers: cycled through for samples; first entry is the greedy answer.
A second-pass request always returns the greedy answer.
"""

import argparse
import asyncio
import itertools
import json

import websockets


async def run(name: str, answers: list[str], url: str):
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"type": "register", "name": name,
                                  "model": f"mock-{name}", "device": "mock"}))
        cycle = itertools.cycle(answers)
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") != "generate":
                continue
            greedy = {"text": answers[0], "tokens": 8}
            samples = [{"text": next(cycle), "tokens": 8}
                       for _ in range(msg["k_samples"])]
            await ws.send(json.dumps({"type": "generation", "req_id": msg["req_id"],
                                      "greedy": greedy, "samples": samples}))
            print(f"[{name}] {msg['kind']}: greedy={greedy['text']} "
                  f"samples={[s['text'] for s in samples]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--answers", required=True, help="comma-separated answer pool")
    ap.add_argument("--url", default="ws://localhost:8020/ws")
    args = ap.parse_args()
    asyncio.run(run(args.name, args.answers.split(","), args.url))
