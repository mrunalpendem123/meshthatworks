"""Stage 2 (spec §8): detector validation. The entire scientific risk, measured.

For each eval question: does the gap signal predict that the collective's
first-pass answer (cluster-majority over greedy answers) is wrong?
Reports AUROC per signal per dataset. Kill criterion: AUROC < 0.6 → stop and
rethink before building anything further (spec §7).

  python detector_eval.py [--datasets triviaqa gsm8k]
"""

import argparse
from collections import Counter

from common import is_correct
from signals import compute_signals, load_grouped

SIGNALS = ["joint_entropy", "mean_pairwise_jsd", "n_greedy_clusters", "max_norm_entropy"]


def majority_greedy_answer(q: dict, sig: dict) -> str | None:
    """Collective first-pass answer: the greedy answer from the largest greedy
    cluster (ties → lowest node id)."""
    ids = {int(n): c for n, c in sig["greedy_cluster_ids"].items()}
    counts = Counter(ids.values())
    best = max(counts.items(), key=lambda kv: (kv[1], -min(n for n, c in ids.items() if c == kv[0])))
    node = min(n for n, c in ids.items() if c == best[0])
    return q["greedy"][node]["answer"]


def auroc(scores: list[float], labels: list[int]) -> float:
    from sklearn.metrics import roc_auc_score

    if len(set(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["triviaqa", "gsm8k"])
    args = ap.parse_args()

    try:
        from entropy import load_calibration, normalized_entropy
        calib = load_calibration()
    except FileNotFoundError:
        raise SystemExit("calibration.json missing — run calibrate.py first")

    for ds in args.datasets:
        grouped = load_grouped(ds)
        sigs = compute_signals(ds)
        scores = {s: [] for s in SIGNALS}
        labels = []
        for qid, q in grouped.items():
            if q["split"] != "eval":
                continue
            sig = sigs[qid]
            ans = majority_greedy_answer(q, sig)
            labels.append(0 if is_correct(ans, q["gold"], ds) else 1)  # 1 = error
            norm = [normalized_entropy(e, q["greedy"][int(n)]["model"], calib)
                    for n, e in sig["per_node_entropy"].items()]
            scores["joint_entropy"].append(sig["joint_entropy"])
            scores["mean_pairwise_jsd"].append(sig["mean_pairwise_jsd"])
            scores["n_greedy_clusters"].append(sig["n_greedy_clusters"])
            scores["max_norm_entropy"].append(max(norm))

        n_err = sum(labels)
        print(f"\n{ds}: {len(labels)} eval questions, first-pass error rate "
              f"{n_err / max(len(labels), 1):.1%}")
        for s in SIGNALS:
            a = auroc(scores[s], labels)
            verdict = "" if a != a else ("  ← KILL (<0.6)" if a < 0.6 else "  ✓")
            print(f"  AUROC[{s:20s}] = {a:.3f}{verdict}")


if __name__ == "__main__":
    main()
