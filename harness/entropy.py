"""Uncertainty signals (spec §3.1-3.2).

Two distinct quantities, never conflated:
  - cross-node divergence over a JOINT clustering  → detects THAT a gap exists
  - per-node semantic entropy, z-normalized per model → decides WHO to ask
Only normalized per-node values may be compared across nodes (§3.2).
"""

import json
import math
from collections import Counter, defaultdict

from common import calibration_path
from entailment import cluster


def semantic_entropy(cluster_ids: list[int]) -> float:
    """Discrete semantic entropy over cluster proportions."""
    n = len(cluster_ids)
    if n == 0:
        return 0.0
    counts = Counter(cluster_ids)
    return -sum((c / n) * math.log(c / n) for c in counts.values())


def _dist(cluster_ids: list[int], support: list[int]) -> list[float]:
    counts = Counter(cluster_ids)
    n = len(cluster_ids) or 1
    return [counts.get(c, 0) / n for c in support]


def _jsd(p: list[float], q: list[float]) -> float:
    m = [(a + b) / 2 for a, b in zip(p, q)]

    def kl(x, y):
        return sum(a * math.log(a / b) for a, b in zip(x, y) if a > 0)

    return (kl(p, m) + kl(q, m)) / 2


def question_signals(question: str, dataset: str,
                     node_answers: dict[int, list[str | None]],
                     node_greedy: dict[int, str | None]) -> dict:
    """All per-question uncertainty signals from one joint clustering pass.

    node_answers: node -> K sampled answers; node_greedy: node -> greedy answer.
    """
    nodes = sorted(node_answers)
    flat, owner = [], []
    for n in nodes:
        for a in node_answers[n]:
            flat.append(a)
            owner.append(("sample", n))
    for n in nodes:
        flat.append(node_greedy[n])
        owner.append(("greedy", n))

    joint = cluster(question, flat, dataset)

    per_node_ids = defaultdict(list)
    greedy_ids = {}
    for cid, (kind, n) in zip(joint, owner):
        if kind == "sample":
            per_node_ids[n].append(cid)
        else:
            greedy_ids[n] = cid

    support = sorted({c for ids in per_node_ids.values() for c in ids})
    dists = {n: _dist(per_node_ids[n], support) for n in nodes}
    pairs = [(a, b) for i, a in enumerate(nodes) for b in nodes[i + 1:]]
    mean_jsd = (sum(_jsd(dists[a], dists[b]) for a, b in pairs) / len(pairs)) if pairs else 0.0

    all_sample_ids = [c for n in nodes for c in per_node_ids[n]]
    return {
        "joint_entropy": semantic_entropy(all_sample_ids),
        "mean_pairwise_jsd": mean_jsd,
        "n_greedy_clusters": len(set(greedy_ids.values())),
        "greedy_cluster_ids": greedy_ids,               # node -> joint cluster id
        "per_node_entropy": {n: semantic_entropy(per_node_ids[n]) for n in nodes},
        "per_node_sample_ids": {n: per_node_ids[n] for n in nodes},
        "per_node_modal_cluster": {
            n: Counter(per_node_ids[n]).most_common(1)[0][0] for n in nodes if per_node_ids[n]
        },
    }


# ---------- per-model calibration (spec §3.2) ----------

def load_calibration() -> dict:
    with open(calibration_path()) as f:
        return json.load(f)


def normalized_entropy(raw: float, model: str, calib: dict) -> float:
    stats = calib[model]
    return (raw - stats["mean"]) / max(stats["std"], 1e-6)
