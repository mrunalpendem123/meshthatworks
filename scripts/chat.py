#!/usr/bin/env python3
"""Interactive chat against OLMoE running on local MLX.

Usage:
    source ~/Desktop/meshthatworks-deps/.venv/bin/activate
    python3 scripts/chat.py

Type a message and hit enter. Type /exit, /quit, or Ctrl-D to leave.
Type /reset to wipe the conversation history.
"""
import os
import sys
import time

MODEL_PATH = os.path.expanduser(
    "~/Desktop/meshthatworks-deps/models/OLMoE-1B-7B-0125-Instruct-4bit"
)
MAX_TOKENS_PER_REPLY = 512

try:
    from mlx_lm import load, stream_generate
except ImportError:
    sys.stderr.write(
        "mlx_lm not available. Activate the venv first:\n"
        "    source ~/Desktop/meshthatworks-deps/.venv/bin/activate\n"
    )
    sys.exit(1)


def main() -> int:
    print(f"loading {os.path.basename(MODEL_PATH)}...", file=sys.stderr, flush=True)
    t0 = time.time()
    model, tokenizer = load(MODEL_PATH)
    print(f"ready in {time.time() - t0:.1f}s. commands: /exit  /reset\n", file=sys.stderr)

    history: list[dict[str, str]] = []

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

        history.append({"role": "user", "content": user})
        prompt = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )

        print("olmoe> ", end="", flush=True)
        reply_parts: list[str] = []
        t_start = time.time()
        n_tok = 0
        for chunk in stream_generate(
            model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS_PER_REPLY
        ):
            print(chunk.text, end="", flush=True)
            reply_parts.append(chunk.text)
            n_tok += 1
        dt = time.time() - t_start
        reply = "".join(reply_parts)
        print(f"\n  \033[2m[{n_tok} tok, {dt:.1f}s, {n_tok / max(dt, 0.01):.1f} tok/s]\033[0m")

        history.append({"role": "assistant", "content": reply})

    print("bye.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
