"""Stage 1 (spec §8): fan out every question to all nodes, dump generations.jsonl.

Per question, per node: 1 greedy answer (accuracy) + K samples at T=1.0 (entropy).
Resumable — already-generated (qid, model) pairs are skipped on rerun.

  python generate.py --dataset triviaqa --n-questions 400
  python generate.py --dataset gsm8k    --n-questions 400
"""

import argparse
import sys

from common import K_SAMPLES, MODELS, SAMPLE_TEMP, append_jsonl, generations_path, read_jsonl
from datasets_util import build_prompt, extract_answer, load_questions
from ollama_client import check_models, generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["triviaqa", "gsm8k"])
    ap.add_argument("--n-questions", type=int, default=400)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--k", type=int, default=K_SAMPLES)
    args = ap.parse_args()

    missing = check_models(args.models)
    if missing:
        sys.exit(f"models not pulled in Ollama: {missing} — run: ollama pull {' '.join(missing)}")

    path = generations_path(args.dataset)
    done = {(r["qid"], r["model"]) for r in read_jsonl(path)}
    questions = load_questions(args.dataset, args.n_questions)
    print(f"{args.dataset}: {len(questions)} questions × {len(args.models)} nodes × "
          f"(1 greedy + {args.k} samples); {len(done)} (qid, model) pairs already done")

    max_tokens = 512 if args.dataset == "gsm8k" else 96
    for qi, q in enumerate(questions):
        for node, model in enumerate(args.models):
            if (q["qid"], model) in done:
                continue
            prompt = build_prompt(q["question"], args.dataset)
            rows = []
            g = generate(model, prompt, temperature=0.0, max_tokens=max_tokens)
            rows.append({**q, "model": model, "node": node, "kind": "greedy", "sample_idx": -1,
                         "text": g["text"], "answer": extract_answer(g["text"], args.dataset),
                         "tokens": g["tokens"]})
            for k in range(args.k):
                s = generate(model, prompt, temperature=SAMPLE_TEMP, seed=1000 + k,
                             max_tokens=max_tokens)
                rows.append({**q, "model": model, "node": node, "kind": "sample", "sample_idx": k,
                             "text": s["text"], "answer": extract_answer(s["text"], args.dataset),
                             "tokens": s["tokens"]})
            append_jsonl(path, rows)
        if (qi + 1) % 20 == 0:
            print(f"  {qi + 1}/{len(questions)}")
    print(f"done → {path}")


if __name__ == "__main__":
    main()
