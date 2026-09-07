"""Fit per-model entropy normalization on the calibration split (spec §3.2).

Raw semantic-entropy values are not comparable across models — each model's
uncertainty distribution has its own shape (Chuang et al., 2502.04428). This fits
mean/std per model on 'calib' questions only and writes calibration.json.

  python calibrate.py [--datasets triviaqa gsm8k]
"""

import argparse
import json
import statistics

from common import MODELS, calibration_path
from signals import compute_signals, load_grouped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["triviaqa", "gsm8k"])
    args = ap.parse_args()

    per_model: dict[str, list[float]] = {m: [] for m in MODELS}
    for ds in args.datasets:
        grouped = load_grouped(ds)
        sigs = compute_signals(ds)
        for qid, q in grouped.items():
            if q["split"] != "calib":
                continue
            for node_str, ent in sigs[qid]["per_node_entropy"].items():
                model = q["greedy"][int(node_str)]["model"]
                per_model.setdefault(model, []).append(ent)

    calib = {}
    for model, vals in per_model.items():
        if len(vals) < 10:
            print(f"warning: only {len(vals)} calib points for {model}")
        if not vals:
            continue
        calib[model] = {
            "mean": statistics.mean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 1.0,
            "n": len(vals),
        }
        print(f"{model:20s}  mean={calib[model]['mean']:.3f}  "
              f"std={calib[model]['std']:.3f}  n={calib[model]['n']}")

    with open(calibration_path(), "w") as f:
        json.dump(calib, f, indent=2)
    print(f"→ {calibration_path()}")


if __name__ == "__main__":
    main()
