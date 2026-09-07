"""Bidirectional-entailment semantic clustering (Kuhn et al. / Farquhar et al.).

Two answers are semantically equivalent iff each entails the other, judged by a
phone-sized NLI model (spec §3.1). GSM8K answers are numeric, so equivalence there
is exact numeric match and the NLI model is never loaded.
"""

import threading
from functools import lru_cache

from common import ENTAILMENT_MODEL, normalize_answer

_nli = None
_nli_lock = threading.Lock()


def _get_nli():
    global _nli
    with _nli_lock:
        return _load_nli()


def _load_nli():
    global _nli
    if _nli is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(ENTAILMENT_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(ENTAILMENT_MODEL)
        model.eval()
        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        _nli = (tok, model, device)
    return _nli


@lru_cache(maxsize=200_000)
def _entails(premise: str, hypothesis: str) -> bool:
    import torch

    tok, model, device = _get_nli()
    with _nli_lock:  # one NLI forward at a time — MPS is not thread-safe
        inputs = tok(premise, hypothesis, return_tensors="pt", truncation=True,
                     max_length=256).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
    # deberta-large-mnli labels: 0=contradiction, 1=neutral, 2=entailment
    return int(logits.argmax(dim=-1)) == 2


def equivalent(question: str, a: str, b: str, dataset: str) -> bool:
    if dataset == "gsm8k":
        return a == b  # answers are pre-extracted numbers
    if normalize_answer(a) == normalize_answer(b):
        return True
    ca = f"{question} {a}"
    cb = f"{question} {b}"
    return _entails(ca, cb) and _entails(cb, ca)


def cluster(question: str, answers: list[str | None], dataset: str) -> list[int]:
    """Greedy clustering: assign each answer to the first cluster whose
    representative it is bidirectionally equivalent with. None answers (extraction
    failed) each get their own cluster — they carry no shared meaning.
    Returns a cluster id per answer."""
    reps: list[str] = []          # representative answer per cluster
    ids: list[int] = []
    next_singleton = -1
    for a in answers:
        if a is None:
            ids.append(next_singleton)  # negative ids = unclusterable singletons
            next_singleton -= 1
            continue
        for ci, rep in enumerate(reps):
            if equivalent(question, a, rep, dataset):
                ids.append(ci)
                break
        else:
            reps.append(a)
            ids.append(len(reps) - 1)
    return ids
