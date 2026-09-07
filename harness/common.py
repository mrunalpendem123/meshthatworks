"""Shared config and IO for the offline harness (spec §8, stages 1-3)."""

import json
import os
import re
import string
from pathlib import Path

# The four heterogeneous nodes (spec §6.1). Ollama tags.
# Start homogeneous first if you want coordination-only attribution — see SPEC §6.1.
MODELS = [
    "qwen2.5:1.5b",
    "llama3.2:1b",
    "phi3:mini",
    "gemma2:2b",
]

K_SAMPLES = 5          # samples per node at T=1.0 (spec §3.2: K ≈ 5)
SAMPLE_TEMP = 1.0
CALIB_FRACTION = 0.3   # held-out split for per-model entropy calibration, never evaluated on

DATA_DIR = Path(os.environ.get("MTW_HARNESS_DATA", Path(__file__).parent / "data"))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

ENTAILMENT_MODEL = "microsoft/deberta-large-mnli"  # ~400M, phone-sized (spec §3.1)


def generations_path(dataset: str) -> Path:
    return DATA_DIR / f"generations_{dataset}.jsonl"


def second_pass_path(dataset: str) -> Path:
    return DATA_DIR / f"second_pass_{dataset}.jsonl"


def calibration_path() -> Path:
    return DATA_DIR / "calibration.json"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------- answer normalization / scoring ----------

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str) -> str:
    """SQuAD/TriviaQA-style normalization: lowercase, strip punctuation and articles."""
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def extract_number(text: str) -> str | None:
    """GSM8K: prefer the number after '####', else the last number in the text."""
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if not m:
        nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
        if not nums:
            return None
        m_val = nums[-1]
    else:
        m_val = m.group(1)
    try:
        v = float(m_val.replace(",", ""))
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return None


def is_correct(row_answer: str | None, gold, dataset: str) -> bool:
    if row_answer is None:
        return False
    if dataset == "gsm8k":
        return row_answer == gold
    # triviaqa: gold is a list of aliases
    norm = normalize_answer(row_answer)
    return any(norm == normalize_answer(g) or normalize_answer(g) in norm for g in gold)
