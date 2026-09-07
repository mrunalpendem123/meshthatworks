"""Live semantic equivalence for web-mesh rounds.

Equivalence ladder: normalized string match → extracted-number match → optional
bidirectional NLI (harness entailment model) when MTW_NLI=1 and torch is present.
Short-answer prompts keep the cheap rungs effective; NLI upgrades quality later
without changing anything else.
"""

import math
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))
from common import extract_number, normalize_answer  # noqa: E402

_NLI_OK = None


def _nli_equivalent(question: str, a: str, b: str) -> bool:
    global _NLI_OK
    if _NLI_OK is False:
        return False
    try:
        from entailment import equivalent as harness_equivalent
        result = harness_equivalent(question, a, b, "triviaqa")
        _NLI_OK = True
        return result
    except Exception as e:  # torch/transformers missing or model load failed
        if _NLI_OK is None:
            print(f"[cluster] NLI unavailable, using lite equivalence only: {e}")
        _NLI_OK = False
        return False


def equivalent(question: str, a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    xa, xb = normalize_answer(a), normalize_answer(b)
    if xa == xb:
        return True
    # containment: "the capital of france is paris" ≡ "paris"
    short, long_ = sorted([xa, xb], key=len)
    if len(short) >= 3 and f" {short} " in f" {long_} ":
        return True
    na, nb = extract_number(a), extract_number(b)
    if na is not None and nb is not None and ("#" in a or "#" in b or a.strip()[0].isdigit() or b.strip()[0].isdigit()):
        if na == nb:
            return True
    if os.environ.get("MTW_NLI") != "0":  # NLI on by default; falls back if torch missing
        return _nli_equivalent(question, a, b)
    return False


def cluster(question: str, answers: list[str | None]) -> list[int]:
    """Greedy clustering by `equivalent`. None answers get negative singleton ids."""
    reps: list[str] = []
    ids: list[int] = []
    next_singleton = -1
    for a in answers:
        if a is None:
            ids.append(next_singleton)
            next_singleton -= 1
            continue
        for ci, rep in enumerate(reps):
            if equivalent(question, a, rep):
                ids.append(ci)
                break
        else:
            reps.append(a)
            ids.append(len(reps) - 1)
    return ids


def semantic_entropy(cluster_ids: list[int]) -> float:
    n = len(cluster_ids)
    if n == 0:
        return 0.0
    counts = Counter(cluster_ids)
    return -sum((c / n) * math.log(c / n) for c in counts.values())


def extract_answer(text: str) -> str | None:
    """Short-answer extraction for live rounds: '#### N' if present, else first
    non-empty line, trimmed."""
    num = None
    if "####" in text:
        num = extract_number(text)
    if num is not None:
        return num
    for line in text.strip().split("\n"):
        line = line.strip().strip('"').rstrip(".")
        if line:
            return line[:200]
    return None
