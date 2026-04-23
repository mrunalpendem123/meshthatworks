# Baselines

First M1 deliverable — concrete performance numbers for what exists today, so we know what to beat.

## Hardware under test

- Apple Silicon M-series, **8 GB unified RAM**, 8 CPU cores
- macOS 26.2 (Darwin 25.2.0, build 25C56)
- Xcode 26 + Metal Toolchain 17E188
- Rust 1.96 nightly
- Disk free at first test: 42 GB

This is the target hardware class the project was designed for — the 8 GB constraint in the spec applies here directly.

## Test matrix

| Runner | Model | Load | Gen tok/s | Date | Notes |
|---|---|---|---|---|---|
| upstream MLX 0.31.2 (pip) | OLMoE-1B-7B-0125-Instruct-4bit | 8.7 s | 3.6 | 2026-04-23 | full model in RAM, no streaming |
| SharpAI/mlx fork (patched) | OLMoE-1B-7B-0125-Instruct-4bit | — | — | — | **built 2026-04-23**, `libmlx.a` 34 MB, needs `patches/sharpai-mlx-cmake-hookup.patch`. Integration test not yet written. |
| SharpAI/mlx fork + `streamed_gather_mm` | OLMoE-1B-7B-0125-Instruct-4bit | — | — | — | SSD streaming path, not yet exercised |
| SwiftLM (`--stream-experts`) | LFM2-8B-A1B-4bit (5 GB) | 2 s | **1.6–1.8** | 2026-04-23 | works, but crashes after 1–2 requests on 8 GB Mac (MEM_DEMAND ≈ 6.6 GB) |
| SwiftLM (`--stream-experts --ssd-prefetch`) | LFM2-8B-A1B-4bit (5 GB) | 2 s | **2.78** (1st req only) | 2026-04-23 | PAPPS 16-worker prefetch on — ~50% uplift before OOM |
| SwiftLM (no streaming) | Qwen3-8B-4bit (4.3 GB dense) | — | — | 2026-04-23 | `SWAP-ASSISTED` mode, first request killed server. Dense 8B too big for 8 GB Mac single-node. |
| SwiftLM (no streaming) | LFM2-8B-A1B-4bit | — | — | 2026-04-23 | Both requests timed out at 90 s — streaming is *required*, not optional, on this hardware |
| Mesh-LLM | Qwen3-Coder-30B-A3B | — | — | — | not tried, requires 2 devices |
| Prima.cpp | matched | — | — | — | not tried |
| mtw two-node mesh (target) | Qwen3-Coder-30B-A3B | — | **≥8** | target | spec §9 success criterion |

## 2026-04-23 — upstream MLX on OLMoE

Ran `mlx_lm.generate` against the mlx-community 4-bit OLMoE-1B-7B-0125-Instruct:

```python
from mlx_lm import load, generate
model, tokenizer = load(".../OLMoE-1B-7B-0125-Instruct-4bit")
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Write a three-line haiku about SSDs."}],
    tokenize=False, add_generation_prompt=True,
)
text = generate(model, tokenizer, prompt=prompt, max_tokens=60)
```

Timings:

- Load: **8.7 s** (reading 3.9 GB safetensors + setting up 4-bit quant structures)
- Generation: **16.8 s** for 60 tokens → **3.6 tok/s** sustained

Output (coherent, approximately haiku-shaped):

```
Fast, reliable, SSDs
Solid State Drives, fast and strong
Pure speed, no delay
```

### Interpretation

This is the **upstream-MLX pip package**, not SharpAI's SSD-streaming fork. The whole 3.9 GB of quantized weights is memory-mapped into RAM up front and generation runs from there. On an 8 GB machine this is near the edge — there's swap pressure from model + tokenizer + OS + whatever else is running, but generation completes without failure.

**3.6 tok/s is the floor the project exists to stay above** while cutting peak RAM footprint by streaming cold experts off SSD instead of mapping everything.

## What's not yet measured

- SharpAI/mlx fork with `streamed_gather_mm` active (the actual SSD streaming case — next task).
- Mesh-LLM two-node pipelined inference on matched 8 GB hardware.
- Prima.cpp pipelined-ring on matched hardware.
- Qwen1.5-MoE-A2.7B (~8 GB Q4) or Qwen3-Coder-30B-A3B (~18 GB Q4, needs sharding to fit on 8 GB).
- Time-to-first-token vs sustained generation (current 3.6 tok/s is 60-token average, not steady-state).
- RAM high-water-mark during generation.

Each entry above becomes a row in the matrix as we fill it in.
