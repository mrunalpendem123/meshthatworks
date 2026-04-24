#!/usr/bin/env python3
"""Interactive chat against a local SwiftLM (or any OpenAI-compatible) server.

Streams tokens as they arrive. Maintains conversation history across turns.

Usage:
    python3 scripts/chat_api.py                          # defaults to :9876
    python3 scripts/chat_api.py --url http://127.0.0.1:9876
    python3 scripts/chat_api.py --model qwen3 --max-tokens 400

Commands at the prompt:
    /exit or /quit     leave
    /reset             wipe conversation history
    /url <new>         switch to a different endpoint (useful when mesh'd)
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error


def stream_chat(url: str, model: str, messages, max_tokens: int, temperature: float):
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    chunks = 0
    parts: list[str] = []
    t_start = time.time()
    t_first = None

    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = ev.get("choices", [{}])[0].get("delta", {})
            text = delta.get("content", "")
            if text:
                if t_first is None:
                    t_first = time.time() - t_start
                print(text, end="", flush=True)
                parts.append(text)
                chunks += 1

    dt = time.time() - t_start
    return "".join(parts), chunks, dt, t_first


def main() -> int:
    ap = argparse.ArgumentParser(description="Stream chat against a local OpenAI-compatible server")
    ap.add_argument("--url", default="http://127.0.0.1:9876")
    ap.add_argument("--model", default="qwen3")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--temp", type=float, default=0.7)
    args = ap.parse_args()

    print(f"chat_api.py → {args.url}  (model={args.model}, max_tokens={args.max_tokens})")
    print("commands: /exit  /reset  /url <new-url>")
    print()

    history: list[dict] = []
    url = args.url

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user:
            continue
        if user in ("/exit", "/quit"):
            break
        if user == "/reset":
            history.clear()
            print("(history cleared)\n", file=sys.stderr)
            continue
        if user.startswith("/url "):
            url = user.split(None, 1)[1]
            print(f"(url → {url})\n", file=sys.stderr)
            continue

        history.append({"role": "user", "content": user})
        print("asst> ", end="", flush=True)
        try:
            reply, chunks, dt, ttft = stream_chat(
                url, args.model, history, args.max_tokens, args.temp
            )
        except urllib.error.HTTPError as e:
            print(f"\n  [HTTP {e.code}: {e.reason}]\n")
            history.pop()
            continue
        except urllib.error.URLError as e:
            print(f"\n  [connection error: {e.reason}]  — is SwiftLM running on {url}?\n")
            history.pop()
            continue
        except Exception as e:
            print(f"\n  [error: {type(e).__name__}: {e}]\n")
            history.pop()
            continue

        history.append({"role": "assistant", "content": reply})
        ttft_str = f"ttft={ttft:.1f}s, " if ttft is not None else ""
        print(
            f"\n  \033[2m[{ttft_str}{chunks} chunks, {dt:.1f}s, "
            f"~{chunks / max(dt, 0.01):.2f} chunks/s]\033[0m"
        )

    print("bye.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
