"""Benchmark loading (spec §5.3): checkable ground truth only."""

import hashlib

from common import CALIB_FRACTION, extract_number


def _split_of(qid: str) -> str:
    """Deterministic calib/eval assignment by qid hash. Calibration is fit only on
    'calib' rows (spec §3.2: never on the evaluation set)."""
    h = int(hashlib.sha256(qid.encode()).hexdigest(), 16) % 1000
    return "calib" if h < CALIB_FRACTION * 1000 else "eval"


def load_questions(dataset: str, n: int) -> list[dict]:
    from datasets import load_dataset

    out = []
    if dataset == "triviaqa":
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        for i, ex in enumerate(ds):
            if len(out) >= n:
                break
            qid = f"triviaqa:{ex['question_id']}"
            gold = list({ex["answer"]["value"], *ex["answer"].get("aliases", [])})
            out.append({"qid": qid, "dataset": dataset, "split": _split_of(qid),
                        "question": ex["question"], "gold": gold})
    elif dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        for i, ex in enumerate(ds):
            if len(out) >= n:
                break
            qid = f"gsm8k:{i}"
            out.append({"qid": qid, "dataset": dataset, "split": _split_of(qid),
                        "question": ex["question"], "gold": extract_number(ex["answer"])})
    else:
        raise ValueError(f"unknown dataset {dataset!r} (triviaqa | gsm8k)")
    return out


def build_prompt(question: str, dataset: str) -> str:
    if dataset == "gsm8k":
        return (f"Solve this problem step by step. End with the final numeric answer "
                f"on its own line in the form '#### <number>'.\n\n{question}")
    return (f"Answer this question as briefly as possible — just the answer, "
            f"no explanation.\n\nQ: {question}\nA:")


def extract_answer(text: str, dataset: str) -> str | None:
    if dataset == "gsm8k":
        return extract_number(text)
    ans = text.strip().split("\n")[0].strip()
    return ans if ans else None
