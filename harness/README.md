# Offline harness — stages 1–3

Implements spec §8 stages 1–3: everything that produces a paper without buying a phone.
All state lives in `data/` (gitignored): `generations_<ds>.jsonl`, `signals_<ds>.jsonl`,
`second_pass_<ds>.jsonl`, `calibration.json`, `results.csv`, `results.png`.

## Pipeline

```
generate.py        stage 1   Ollama fan-out: per question × node, 1 greedy + K=5 samples @ T=1.0
calibrate.py       §3.2      per-model entropy mean/std, fit on the calib split ONLY
detector_eval.py   stage 2   AUROC of each gap signal vs. actual first-pass error  ← kill criterion
ablate.py          stage 3   7 conditions → results.csv + results.png
```

Supporting modules: `entailment.py` (bidirectional-entailment clustering,
`microsoft/deberta-large-mnli`, ~400M — phone-sized per §3.1), `entropy.py`
(semantic entropy, cross-node JSD, z-normalization), `signals.py` (per-question
signal cache), `datasets_util.py` (TriviaQA rc.nocontext + GSM8K, checkable ground
truth only per §5.3), `ollama_client.py`, `common.py` (config).

Everything is resumable: generation, signal computation, and second passes are
cached per (question, node) and skipped on rerun.

## Setup

```bash
pip install -r requirements.txt
ollama pull qwen2.5:1.5b && ollama pull llama3.2:1b && ollama pull phi3:mini && ollama pull gemma2:2b
```

Models are set in `common.py::MODELS`. **Start homogeneous** (same model four times)
if you want gains attributable to coordination before introducing heterogeneity (§6.1).

## Run

```bash
python generate.py --dataset triviaqa --n-questions 400
python generate.py --dataset gsm8k    --n-questions 400
python calibrate.py
python detector_eval.py
python ablate.py
```

## The seven conditions (§5.2)

| # | Condition | Reads as |
|---|---|---|
| 1 | single model, 1× greedy | floor |
| 2 | single model, K samples, self-consistency | **the real baseline** |
| 3 | four models, first pass only | is heterogeneity worth anything? |
| 4 | + second pass to **all** nodes | upper bound on the second pass |
| 5 | + second pass to a **random** node | is selection better than luck? |
| 6 | + second pass to the **selected** node | **the contribution** |
| 7 | volunteer mode (raw-confidence self-nomination) | the blackboard arm |

Decision rules: **6 must beat 5** or selection is decoration. **6 must beat 2** or
the premise fails. Both outcomes are results; only an unmeasured outcome is not.

## Kill criteria checked here (§7)

- `detector_eval.py`: any gap-signal AUROC < 0.6 on eval → the detector fails; stop
  and rethink (or write the negative result).
- `ablate.py`: condition 1 vs 2 (self-consistency at equal budget) and 6 vs 5/2 as above.

## v0 design decisions (revisit deliberately, not accidentally)

- **Gap trigger**: the four greedy answers fall into >1 semantic cluster.
  The cost check (§6.4) is not modeled offline — it enters at stage 4/5 where
  energy and latency are real.
- **Selection**: lowest z-normalized per-node semantic entropy (per §3.2, raw
  entropies are never compared across models).
- **Synthesis**: weighted semantic-cluster vote (greedy = 1, second pass = 2) —
  node-class computation only, no aggregator model, satisfying §3.3 trivially.
  A proper node-model synthesis prompt is a stage-4 upgrade.
- **Volunteer mode (7)**: approximated offline as raw (uncalibrated) minimum
  entropy — self-assessment without global knowledge, per the blackboard framing.
- GSM8K semantic equivalence = extracted-number equality (no NLI needed);
  TriviaQA uses normalized string match, then bidirectional entailment.
