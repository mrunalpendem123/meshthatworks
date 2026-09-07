"""Groups generations.jsonl by question and computes/caches uncertainty signals.

The NLI clustering is the expensive step, so signals are cached to
data/signals_<dataset>.jsonl and reruns are incremental.
"""

from collections import defaultdict
from pathlib import Path

from common import DATA_DIR, append_jsonl, generations_path, read_jsonl
from entropy import question_signals


def signals_path(dataset: str) -> Path:
    return DATA_DIR / f"signals_{dataset}.jsonl"


def load_grouped(dataset: str) -> dict[str, dict]:
    """qid -> {meta, greedy: {node: row}, samples: {node: [rows]}} — only questions
    with complete data for every node present in the dump."""
    rows = read_jsonl(generations_path(dataset))
    if not rows:
        raise SystemExit(f"no generations for {dataset} — run generate.py first")
    by_q: dict[str, dict] = {}
    all_nodes = set()
    for r in rows:
        q = by_q.setdefault(r["qid"], {
            "qid": r["qid"], "dataset": dataset, "split": r["split"],
            "question": r["question"], "gold": r["gold"],
            "greedy": {}, "samples": defaultdict(list),
        })
        all_nodes.add(r["node"])
        if r["kind"] == "greedy":
            q["greedy"][r["node"]] = r
        else:
            q["samples"][r["node"]].append(r)
    complete = {qid: q for qid, q in by_q.items()
                if set(q["greedy"]) == all_nodes and set(q["samples"]) == all_nodes}
    dropped = len(by_q) - len(complete)
    if dropped:
        print(f"note: dropped {dropped} incomplete questions")
    return complete


def compute_signals(dataset: str) -> dict[str, dict]:
    """qid -> signals dict (see entropy.question_signals), cached on disk."""
    grouped = load_grouped(dataset)
    path = signals_path(dataset)
    cached = {r["qid"]: r for r in read_jsonl(path)}
    todo = [q for qid, q in grouped.items() if qid not in cached]
    if todo:
        print(f"computing signals for {len(todo)} questions ({dataset})…")
    for i, q in enumerate(todo):
        sig = question_signals(
            q["question"], dataset,
            {n: [r["answer"] for r in sorted(rs, key=lambda r: r["sample_idx"])]
             for n, rs in q["samples"].items()},
            {n: r["answer"] for n, r in q["greedy"].items()},
        )
        # JSON keys must be strings
        row = {"qid": q["qid"],
               **{k: v for k, v in sig.items() if not isinstance(v, dict)},
               "greedy_cluster_ids": {str(k): v for k, v in sig["greedy_cluster_ids"].items()},
               "per_node_entropy": {str(k): v for k, v in sig["per_node_entropy"].items()},
               "per_node_sample_ids": {str(k): v for k, v in sig["per_node_sample_ids"].items()},
               "per_node_modal_cluster": {str(k): v for k, v in sig["per_node_modal_cluster"].items()}}
        append_jsonl(path, [row])
        cached[q["qid"]] = row
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(todo)}")
    return {qid: cached[qid] for qid in grouped}
