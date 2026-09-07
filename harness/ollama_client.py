"""Thin Ollama client. One complete model per node — no sharding (spec §1)."""

import requests

from common import OLLAMA_URL


def generate(model: str, prompt: str, temperature: float, seed: int | None = None,
             max_tokens: int = 512) -> dict:
    """Returns {"text": str, "tokens": int} where tokens = prompt eval + generation."""
    options = {"temperature": temperature, "num_predict": max_tokens}
    if seed is not None:
        options["seed"] = seed
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": options},
        timeout=300,
    )
    r.raise_for_status()
    body = r.json()
    return {
        "text": body.get("response", ""),
        "tokens": body.get("prompt_eval_count", 0) + body.get("eval_count", 0),
    }


def check_models(models: list[str]) -> list[str]:
    """Returns the subset of `models` not present locally."""
    r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    r.raise_for_status()
    have = {m["name"] for m in r.json().get("models", [])}
    have |= {n.split(":")[0] for n in have}
    return [m for m in models if m not in have]
