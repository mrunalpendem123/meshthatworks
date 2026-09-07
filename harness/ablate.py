"""Stage 3 (spec §8): the seven-condition ablation over saved generations.

  1. single model, 1x greedy                          (floor)
  2. single model, K samples, self-consistency        (THE real baseline, §5.1)
  3. four models, first pass only (greedy majority)
  4. four models + second pass to ALL nodes
  5. four models + second pass to a RANDOM node
  6. four models + second pass to the SELECTED node   (the contribution)
  7. volunteer mode — node self-nominates on raw (uncalibrated) confidence

Condition 6 must beat 5, or selection is decoration. It must beat 2, or the
project fails (spec §5.2). Second-pass generations go through Ollama and are
cached to data/second_pass_<dataset>.jsonl, so reruns are free.

Gap trigger (v0): the four greedy answers fall into more than one semantic
cluster. Selection rule (v0): lowest z-normalized per-node semantic entropy.
Synthesis rule (v0): weighted cluster vote — greedy answers weight 1,
second-pass answers weight 2; ties broken by lowest node id.

  python ablate.py [--datasets triviaqa gsm8k]
"""

import argparse
import csv
import random
from collections import Counter, defaultdict

from common import (DATA_DIR, MODELS, append_jsonl, is_correct, read_jsonl,
                    second_pass_path)
from datasets_util import extract_answer
from entailment import cluster
from entropy import load_calibration, normalized_entropy
from ollama_client import generate
from signals import compute_signals, load_grouped

CONDITIONS = ["1_single_greedy", "2_self_consistency", "3_first_pass_majority",
              "4_second_pass_all", "5_second_pass_random", "6_second_pass_selected",
              "7_volunteer"]


# ---------- second pass ----------

def second_pass_prompt(q: dict, candidates: list[str]) -> str:
    cands = "\n".join(f"- {c}" for c in dict.fromkeys(candidates))
    if q["dataset"] == "gsm8k":
        return (f"Different solvers disagree on this problem. Their final answers were:\n"
                f"{cands}\n\nSolve it carefully yourself, step by step. End with the final "
                f"numeric answer on its own line in the form '#### <number>'.\n\n{q['question']}")
    return (f"Different answerers gave these candidate answers to a question:\n{cands}\n\n"
            f"Decide which is correct (or give a better one). Reply with just the answer.\n\n"
            f"Q: {q['question']}\nA:")


def get_second_pass(dataset: str, q: dict, node: int, model: str,
                    candidates: list[str], cache: dict) -> dict:
    key = (q["qid"], node)
    if key in cache:
        return cache[key]
    max_tokens = 512 if dataset == "gsm8k" else 96
    g = generate(model, second_pass_prompt(q, candidates), temperature=0.0,
                 max_tokens=max_tokens)
    row = {"qid": q["qid"], "node": node, "model": model, "text": g["text"],
           "answer": extract_answer(g["text"], dataset), "tokens": g["tokens"]}
    append_jsonl(second_pass_path(dataset), [row])
    cache[key] = row
    return row


# ---------- voting ----------

def weighted_vote(question: str, dataset: str,
                  votes: list[tuple[str | None, float, int]]) -> str | None:
    """votes: (answer, weight, node). Cluster answers, sum weights per cluster,
    return an answer from the heaviest cluster (ties → lowest node id)."""
    answers = [v[0] for v in votes]
    ids = cluster(question, answers, dataset)
    weight = defaultdict(float)
    best_node = {}
    for (a, w, n), cid in zip(votes, ids):
        weight[cid] += w
        best_node[cid] = min(best_node.get(cid, n), n)
    winner = max(weight, key=lambda c: (weight[c], -best_node[c]))
    for (a, _, _), cid in zip(votes, ids):
        if cid == winner:
            return a
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["triviaqa", "gsm8k"])
    args = ap.parse_args()
    calib = load_calibration()

    results = []
    for ds in args.datasets:
        grouped = load_grouped(ds)
        sigs = compute_signals(ds)
        sp_cache = {(r["qid"], r["node"]): r for r in read_jsonl(second_pass_path(ds))}
        eval_qs = [q for q in grouped.values() if q["split"] == "eval"]
        print(f"\n{ds}: {len(eval_qs)} eval questions")

        correct = {c: 0 for c in CONDITIONS}
        tokens = {c: 0 for c in CONDITIONS}
        triggered = 0
        per_model_1 = Counter()
        per_model_2 = Counter()

        for q in eval_qs:
            sig = sigs[q["qid"]]
            nodes = sorted(q["greedy"])
            models = {n: q["greedy"][n]["model"] for n in nodes}
            greedy_ans = {n: q["greedy"][n]["answer"] for n in nodes}
            greedy_tok = sum(q["greedy"][n]["tokens"] for n in nodes)
            sample_tok = sum(r["tokens"] for n in nodes for r in q["samples"][n])
            sample_answers = {
                n: [r["answer"] for r in sorted(q["samples"][n], key=lambda r: r["sample_idx"])]
                for n in nodes}

            # 1 + 2: single-model floors (averaged over the four models)
            for n in nodes:
                if is_correct(greedy_ans[n], q["gold"], ds):
                    per_model_1[models[n]] += 1
                sc_votes = [(a, 1.0, n) for a in sample_answers[n]] + [(greedy_ans[n], 0.5, n)]
                sc = weighted_vote(q["question"], ds, sc_votes)
                if is_correct(sc, q["gold"], ds):
                    per_model_2[models[n]] += 1
            tokens["1_single_greedy"] += greedy_tok // len(nodes)
            tokens["2_self_consistency"] += sample_tok // len(nodes)

            # 3: heterogeneous first pass, greedy majority
            fp_votes = [(greedy_ans[n], 1.0, n) for n in nodes]
            fp = weighted_vote(q["question"], ds, fp_votes)
            if is_correct(fp, q["gold"], ds):
                correct["3_first_pass_majority"] += 1
            tokens["3_first_pass_majority"] += greedy_tok

            # gap detection for 4-7
            gap = sig["n_greedy_clusters"] > 1
            base_tok = greedy_tok + sample_tok  # detector consumes the K samples
            candidates = [a for a in greedy_ans.values() if a is not None]
            norm = {n: normalized_entropy(sig["per_node_entropy"][str(n)], models[n], calib)
                    for n in nodes}
            raw = {n: sig["per_node_entropy"][str(n)] for n in nodes}
            rng = random.Random(q["qid"])

            picks = {
                "4_second_pass_all": nodes,
                "5_second_pass_random": [rng.choice(nodes)],
                "6_second_pass_selected": [min(nodes, key=lambda n: (norm[n], n))],
                "7_volunteer": [min(nodes, key=lambda n: (raw[n], n))],
            }
            if gap:
                triggered += 1
            for cond, chosen in picks.items():
                tokens[cond] += base_tok
                if not gap:
                    final = fp
                else:
                    votes = [(greedy_ans[n], 1.0, n) for n in nodes]
                    for n in chosen:
                        sp = get_second_pass(ds, q, n, models[n], candidates, sp_cache)
                        votes.append((sp["answer"], 2.0, n))
                        tokens[cond] += sp["tokens"]
                    final = weighted_vote(q["question"], ds, votes)
                if is_correct(final, q["gold"], ds):
                    correct[cond] += 1

        nq = len(eval_qs)
        correct["1_single_greedy"] = sum(per_model_1.values()) / len(MODELS)
        correct["2_self_consistency"] = sum(per_model_2.values()) / len(MODELS)

        print(f"  gap trigger rate: {triggered / nq:.1%}")
        for m in sorted(per_model_1):
            print(f"    {m:20s} greedy {per_model_1[m] / nq:.1%}   "
                  f"self-consistency {per_model_2[m] / nq:.1%}")
        for cond in CONDITIONS:
            acc = correct[cond] / nq
            tk = tokens[cond] / nq
            print(f"  {cond:24s} acc={acc:.1%}  tokens/q={tk:.0f}")
            results.append({"dataset": ds, "condition": cond, "accuracy": round(acc, 4),
                            "tokens_per_q": round(tk, 1),
                            "trigger_rate": round(triggered / nq, 4)})

    out = DATA_DIR / "results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\n→ {out}")
    plot(results)


def plot(results: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted({r["dataset"] for r in results})
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        rows = [r for r in results if r["dataset"] == ds]
        for r in rows:
            ax.scatter(r["tokens_per_q"], r["accuracy"], s=60)
            ax.annotate(r["condition"].split("_")[0], (r["tokens_per_q"], r["accuracy"]),
                        textcoords="offset points", xytext=(6, 4))
        ax.set_title(ds)
        ax.set_xlabel("tokens per question")
        ax.set_ylabel("accuracy")
        ax.grid(alpha=0.3)
    out = DATA_DIR / "results.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
