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
| `mtw serve` → SwiftLM (`--stream-experts --ssd-prefetch`, `--mem-limit 4096`) | Qwen3-30B-A3B-4bit (~18 GB) | ~30 s | **0.17 (3 tok cold) / < 0.4 sustained** | 2026-04-28 | streams cleanly on 8 GB Mac, RSS oscillates 30 KB–906 MB. 200-token poetry hit the 900 s proxy timeout; 50-token count hit 120 s curl timeout — sustained throughput is bandwidth-bound at this mem-limit. |
| `mtw serve` → SwiftLM (`--stream-experts --ssd-prefetch`, `--mem-limit 6000`) | Qwen3-30B-A3B-4bit (~18 GB) | 35.6 s prefill (15 tok) | **0.045 tok/s warm decode** | 2026-04-29 | **Confirmed not a 4 GB cap problem.** SwiftLM internal `estimated_tok_s=4.6`; we measured ~22 s per generated token (10 tokens in 252 s wall, 8 inter-token gaps averaging 21.9 s). MEM_DEMAND climbed 6.7 → 7.0 GB across two requests on an 8.6 GB-RAM Mac — page-cache thrashing dominates. Cache work needs to budget OS-level pages, not just MLX VRAM. |
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

## 2026-04-28 — Qwen3-30B-A3B-4bit on the 8 GB Mac via `mtw serve`

End-to-end through `mtw serve` (release build) → SwiftLM child (`--stream-experts --ssd-prefetch`, `--mem-limit 4096`). The new in-process memory sampler (`crates/mtw-engine/src/swiftlm.rs::memory_sampler`) emitted 16,234 RSS samples to `/tmp/mtw-swiftlm.log` over the run.

**Streaming works.** Across the full trace, RSS oscillated between **2.9 MB (idle, fully evicted)** and **906 MB (peak, prompt-processing burst)**, with steady-state generation in the 50–165 MB band. The 30B model on disk is ~18 GB; we never came close to mapping it. Adaptive expert paging from SSD is real and observable.

**But sustained throughput is severely bandwidth-bound at 4 GB mem-limit.**
- *Request 1* — `"Write exactly: HELLO WORLD"`, 30 max_tokens, cold cache: **17.4 s wall-clock for 3 generated tokens** (load + first inference). 0.17 tok/s.
- *Request 2* — 10-line poem, 200 max_tokens: hit the **900 s proxy timeout** without finishing. SwiftLM emitted at least one token (`Nodes`) ~3 minutes in.
- *Request 3* — `"Count from 1 to 10"`, 50 max_tokens, with the cache already hot from req 2: hit a **120 s curl deadline** with only one token (`1`) emitted.

Effective sustained rate: **< 0.4 tok/s** at this mem-limit, well below the 2–4 tok/s M2 target in §8 of the spec. The streaming machinery isn't broken — it's working too hard. Each forward pass on Qwen3-30B-A3B activates 8 experts × ~256 MB each, so the working set per token already exceeds the 4 GB ceiling. Every token forces multi-expert evictions.

**Hypothesis to verify next.** Re-run with `--mem-limit 6500` (within the 8 GB envelope, leaves 1.5 GB for OS) — that puts the per-token expert working set comfortably inside the budget and should bring sustained tok/s up by 5–10×. If it does, this confirms what `mtw-cache` needs to compute: the *minimum* RAM-resident expert footprint that keeps the next token's experts in cache.

The earlier user-confirmed "Qwen3-30B-A3B works on a single node" was almost certainly at a higher mem-limit. Today's run proves the streaming pipeline; it does not establish a usable tok/s.

## 2026-04-29 — `--mem-limit 6000` re-bench (hypothesis disconfirmed)

Same setup, mem-limit raised to 6000 MB. Result: **decode tok/s went from <0.4 to 0.045** — *worse*, not better. The 4 GB cap was not the bottleneck.

What SwiftLM reported on startup:
```
strategy=ssd_streaming, model_weight_gb=17.2, kv_cache_gb=0.4,
total_required_gb=21, system_ram_gb=8.6, estimated_tok_s=4.6,
overcommit_ratio=4.58
```

What we measured across two warm requests:
- Prefill: 35.6 s for 15 tokens, then 37.2 s for 14 tokens
- Generation: 10 tokens in ~220 s of decode wall-clock → ~22 s per token (median inter-token gap from SSE timestamps: 19–24 s)
- MEM_DEMAND: 6.7 GB after request 1, **7.0 GB after request 2**
- OS_RAM: 3.4 → 3.6 GB
- RSS (process-resident): 100–650 MB throughout

**Diagnosis: macOS unified-memory page-cache thrashing.** RSS stays small because SwiftLM faithfully evicts MLX-resident expert weights — the streaming primitive works. But `total_required_gb=21` of mmap'd weight pages still get reached through the OS page cache, and `MEM_DEMAND` is creeping up past `system_ram_gb`. Once the kernel can't keep working-set pages resident, every expert load becomes a real disk read instead of a cache hit. SwiftLM's `estimated_tok_s=4.6` assumes those page reads stay in cache; on an 8 GB Mac they cannot.

**Implication for `mtw-cache`.** The cache-policy abstraction can't just track "which experts to pin in MLX VRAM." It also needs to budget *OS page residency* — keep some experts unmapped so others stay hot, rather than relying on the kernel to make the right LRU choice across mmap'd weights it doesn't understand. Concretely: a hot expert isn't just `set_memory_limit`-pinned in MLX, it's `mlock`'d (or the equivalent `madvise(MADV_WILLNEED)` + active reads to keep it warm). Cold experts should be `madvise(MADV_DONTNEED)`'d so they free OS pages instead of competing for them.

This is now the central design question. The `mtw-cache` rolling histogram (`crates/mtw-cache/src/lib.rs`) classifies experts; the missing half is a thin OS-budget layer that turns those classifications into syscalls SwiftLM can't make from inside MLX.

## What's not yet measured

- SharpAI/mlx fork with `streamed_gather_mm` active (the actual SSD streaming case — next task).
- Mesh-LLM two-node pipelined inference on matched 8 GB hardware.
- Prima.cpp pipelined-ring on matched hardware.
- Qwen1.5-MoE-A2.7B (~8 GB Q4) or Qwen3-Coder-30B-A3B (~18 GB Q4, needs sharding to fit on 8 GB).
- Time-to-first-token vs sustained generation (current 3.6 tok/s is 60-token average, not steady-state).
- RAM high-water-mark during generation.

Each entry above becomes a row in the matrix as we fill it in.
