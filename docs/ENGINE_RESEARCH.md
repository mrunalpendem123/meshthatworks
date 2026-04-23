# MLX SSD-streaming research note

**Source**: SharpAI/mlx at commit `9a9c214` ("feat: custom ssd-streaming kernels and custom MLX I/O fast loaders").
**Local clone**: `~/Desktop/meshthatworks-deps/mlx` — not vendored in this repo, read-only reference.
**Reproduce**: `git clone https://github.com/SharpAI/mlx.git && git checkout 9a9c214`.

---

> **2026-04-23 correction.** The original section 0 of this note claimed "no end-to-end test or benchmark exercises the streamer" (point 4) and treated expert-level cache + async prefetch as new work we'd have to build from scratch. **That framing was wrong.** SharpAI's *SwiftLM* — a separate Swift-based inference runtime in the `SwiftLM` repo, not the `mlx` repo — is the caller of `streamed_gather_mm`. It ships concurrent `pread` with QD=24, runtime top-k expert selection, and an asyncEval pipeline that fuses compute with next-expert load. Published benchmarks: 10.8 tok/s on a 26B MoE in 22 GB RAM; 10× speedup on 122B models. The integration we assumed was missing *exists*, just inside SwiftLM, not inside the MLX repo we initially looked at. Points 2 and 3 below (no cache, no prefetch) were correct observations about `SharpAI/mlx` in isolation, but wrong takeaways about the overall SharpAI stack. The project's novel work is therefore the *mesh* layer on top of SwiftLM, not reimplementing SwiftLM's streaming.

## 0. Key findings that diverge from ARCHITECTURE.md

Six things the spec describes that don't match the code on disk. Flagging up front so the milestone plan can adjust.

**0 (added 2026-04-23). The fork never hooked the streaming code into its build system.** `moe_stream_op.{cpp,h}`, `ssd_streamer.{mm,h}`, and `moe_stream.metal` all exist in the tree but are **not referenced by any `CMakeLists.txt`**. A default `cmake --build` produces a `libmlx.a` that contains zero streaming symbols — the files are orphans. Three one-line CMake additions fix this (captured in `patches/sharpai-mlx-cmake-hookup.patch`):

- `mlx/CMakeLists.txt`: add `${CMAKE_CURRENT_SOURCE_DIR}/core/moe_stream_op.cpp` to `target_sources(mlx PRIVATE ...)`
- `mlx/backend/metal/CMakeLists.txt`: add `${CMAKE_CURRENT_SOURCE_DIR}/ssd_streamer.mm` to `target_sources(mlx PRIVATE ...)`
- `mlx/backend/metal/kernels/CMakeLists.txt`: add `build_kernel(moe_stream)` after the fence block

After the patch, `libmlx.a` ships all the expected symbols: `mlx::core::streamed_gather_mm`, `mlx::core::fast::SSDStreamer::{ctor,dtor,load_sync}`, `mlx::core::LoadSSDExpert::eval_impl`, the `extern "C" mlx_ssd_metrics_snapshot`, and `streamed_moe_gemm` inside `mlx/backend/metal/kernels/mlx.metallib`. Verified locally on macOS 26.2 + Xcode 26 + Metal Toolchain 17E188.



1. **Files live in MLX, not SwiftLM.** The spec (§4.2, §5.1) says we fork `ssd_streamer.mm` and `fence.air` from SwiftLM. They're actually in the MLX fork at `mlx/backend/metal/`. SwiftLM is a Swift wrapper around MLX and contains no `.mm` or `.air` files. Practical impact: our fork target is `SharpAI/mlx`, not `SharpAI/SwiftLM` (or both, with MLX carrying the streaming layer).

2. **There is no expert-level hot cache yet.** Spec §5.1 talks about "smaller hot cache, aggressive LRU eviction". The upstream fork has no such cache — each expert is loaded synchronously on demand and its buffer is released back to the Metal allocator's general pool after compute. The only pressure-based eviction is in the generic `BufferCache<MTL::Buffer>`, not expert-aware. Our cache work is **net-new**, not constant-tuning.

3. **There is no async prefetch.** Spec §5.1 mentions "async prefetch coordination with mesh scheduler". The current hot path is strictly synchronous `pread`. A `dispatch_queue_t io_queue_` and an `MTL::SharedEvent shared_event_` are constructed in `SSDStreamer` but never invoked on the hot path. Async prefetch is also new work.

4. **No end-to-end test or benchmark exercises the streamer.** No `.py`, `.swift`, or fixture calls `streamed_gather_mm`. We are first-mover for integration tests; the reference harness is on us.

5. **FFI surface is narrower than it looks.** Only `mlx_ssd_metrics_snapshot()` is `extern "C"`. The primitives we actually want (`SSDStreamer::load_sync`, `streamed_gather_mm`) require a C++ bridge — `cxx` crate or a custom C facade in `mtw-engine`. Metal C++ types (`MTL::SharedEvent`) cannot cross FFI directly.

---

## 1. Hot-path architecture

### Entry point: `streamed_gather_mm`

Public C++ entry point is `mlx::core::streamed_gather_mm()` at `mlx/core/moe_stream_op.h:31`. Arguments:
- `x`: input token matrix (not used in compute; kept for ABI compat)
- `w_shape`: shape array specifying output/input dims
- `active_expert`: `uint32_t` expert index
- `streamer`: `shared_ptr<SSDStreamer>`
- `expert_offsets`: vector of byte offsets into the `.safetensors` file; `expert_offsets[i..i+1]` gives start/end
- `s`: `StreamOrDevice`

At `mlx/core/moe_stream_op.cpp:170–187` the function wraps compute in a custom MLX primitive called `LoadSSDExpert`.

### Hot path: `LoadSSDExpert::eval_impl`

`mlx/core/moe_stream_op.cpp:63–139`, synchronous on the graph evaluator thread:

1. Bounds check against `expert_offsets_.size()` (line 67).
2. Compute byte offset and matrix size from offsets (lines 71–72).
3. `allocator::malloc(matrix_bytes)` → pinned, unified-memory `MTLBuffer` (line 75).
4. **Synchronous disk read**: `streamer_->load_sync(block_offset, matrix_bytes, o.data<void>())` blocks until the expert is resident (line 78).
5. Metrics accumulation (lines 82–84).
6. 10-second windowed throughput logging to stderr (lines 99–138).

### Memory flow

```
SSD file (.safetensors)
    |
    v   pread (blocking)
pinned MTLBuffer (unified, GPU-visible)
    |
    v   streamed_moe_gemm kernel
float32 output
```

### Metal kernel: `streamed_moe_gemm`

`mlx/backend/metal/kernels/moe_stream.metal:14–58`:
- Inputs: `x` (bf16 tokens as `uint16_t*`), `w` (4-bit packed as `uint32_t*`), `out` (`float*`), plus M/K/N constants.
- 2D thread grid `[N, M]`, one thread per output element.
- Unpacks 8 × 4-bit nibbles per `uint32_t`, scales to `[-8..7] / 8.0`, reinterprets bf16 via `as_type<float>()` (line 51), accumulates dot product across K, writes float32.

### Synchronization

No explicit async prefetch. The flow is: CPU `pread` blocking → GPU compute → next iteration. The `shared_event_` constructed at `ssd_streamer.mm:33` is dormant.

---

## 2. Hot cache sizing

**There is no expert-specific hot cache.** There is a general-purpose Metal buffer pool.

### Buffer pool parameters

`mlx/backend/metal/allocator.cpp:34–66`:

```cpp
block_limit_ = std::min(1.5 * max_rec_size, 0.95 * memsize);
gc_limit_ = std::min(static_cast<size_t>(0.95 * max_rec_size), block_limit_);
max_pool_size_ = block_limit_;
```

Where `max_rec_size = device_->recommendedMaxWorkingSetSize()` (typically 40–50% of total) and `memsize` is total device memory.

Static tuning constants in `mlx/backend/metal/allocator.h:51–52`:
- `small_size_ = 256` — small-buffer heap threshold
- `heap_size_ = 1 << 20` — 1 MB Metal heap pool

### SSDStreamer buffer sizing

`mlx/backend/metal/ssd_streamer.h:25`:

```cpp
SSDStreamer(const std::string& file_path, size_t buffer_size_bytes, int num_buffers = 2);
```

`buffer_size_bytes` is the maximum expert size accepted. No default; caller-set. Enforced at `ssd_streamer.mm:42`:

```cpp
if (length > buffer_size_bytes_) {
  throw std::invalid_argument("[SSDStreamer] Load length exceeds Pinned Buffer capacity.");
}
```

`num_buffers` (default 2) is accepted but currently unused — `ssd_streamer.h:20–21` notes: "The StreamBuffer pool is removed in favor of caller-provided pinned memory."

### Tuning for 4–8 GB

The handle is `MetalAllocator::set_memory_limit()` at `mlx/backend/metal/allocator.cpp:76–82`:

```cpp
allocator::allocator().set_memory_limit(size_t(8) << 30);  // 8 GB
```

This sets `block_limit_`; `gc_limit_` becomes 95% of it. **No expert-specific hot-cache constant exists to tune** — the relevant knob is the general allocator ceiling plus whatever cache layer we build ourselves in `mtw-engine` / `mtw-cache`.

---

## 3. Eviction policy

No expert-specific eviction. The Metal buffer cache is pressure-based via `BufferCache<MTL::Buffer>` in `mlx/backend/common/buffer_cache.h`.

Trigger at `mlx/backend/metal/allocator.cpp:124`:

```cpp
if (mem_required >= gc_limit_ || num_resources_ >= resource_limit_) {
  num_resources_ -= buffer_cache_.release_cached_buffers(mem_required - gc_limit_);
}
```

Post-compute recycle at `allocator.cpp:174–176`:

```cpp
if (get_cache_memory() > max_pool_size_) {
  num_resources_ -= buffer_cache_.release_cached_buffers(
      get_cache_memory() - max_pool_size_);
}
```

Eviction fires when active + cached > `gc_limit_` OR outstanding handles > `resource_limit_`. Internal ordering in `BufferCache` is LRU, but there is no expert-aware prioritization.

---

## 4. Async prefetch

**Absent.** The path is strictly synchronous:

1. `LoadSSDExpert::eval_impl` (line 78 of `moe_stream_op.cpp`) → blocking.
2. `SSDStreamer::load_sync` (line 41 of `ssd_streamer.mm`) → synchronous `pread(fd_, ...)`.
3. Metal compute runs after the buffer is populated.

Dormant hooks:
- `dispatch_queue_t io_queue_` created at `ssd_streamer.mm:25` but never dispatched onto. Comments suggest `dispatch_io_read` was tried but reverted due to copy overhead.
- `MTL::SharedEvent shared_event_` created at `ssd_streamer.mm:33`, not exposed in public API.

Async prefetch for MeshThatWorks means: new `load_async(offset, length, dst, MTL::SharedEvent*) → future`, reactivate `io_queue_`, and make `LoadSSDExpert` enqueue a GPU-side wait on the event rather than blocking the evaluator thread.

---

## 5. FFI surface for Rust

### `ssd_streamer.h` — C++ only

```cpp
class SSDStreamer {
public:
  SSDStreamer(const std::string& file_path, size_t buffer_size_bytes, int num_buffers = 2);
  ~SSDStreamer();
  SSDStreamer(const SSDStreamer&) = delete;
  SSDStreamer& operator=(const SSDStreamer&) = delete;

  void load_sync(off_t byte_offset, size_t length, void* dst_ptr);

  MTL::SharedEvent* get_shared_event() const;
  const std::string& get_file_path() const;
};
```

Not C-callable: `std::string`, deleted copy/move, Metal C++ return. Needs a bridge.

### `moe_stream_op.h` — mostly C++

```cpp
namespace mlx::core {
  array streamed_gather_mm(
      const array& x,
      const array& w_shape,
      uint32_t active_expert,
      std::shared_ptr<fast::SSDStreamer> streamer,
      const std::vector<off_t>& expert_offsets,
      StreamOrDevice s = {}
  );
}

extern "C" {
  struct MlxSSDMetricsSnapshot;
  void mlx_ssd_metrics_snapshot(struct MlxSSDMetricsSnapshot* out);
}
```

Only `mlx_ssd_metrics_snapshot` is `extern "C"`. Everything else uses `mlx::core::array`, `std::shared_ptr`, `std::vector`.

### `mtw-engine` FFI strategy

1. **Wrap `SSDStreamer`** via `cxx` crate (or a hand-rolled C facade):
   - `mtw_ssd_streamer_new(path, buf_size) -> *mut SSDStreamer`
   - `mtw_ssd_streamer_load_sync(*mut SSDStreamer, off_t, size_t, *mut void) -> i32`
   - `mtw_ssd_streamer_free(*mut SSDStreamer)`
2. **Wrap `streamed_gather_mm`** with a C++ helper that accepts Rust-friendly inputs (slices, plain ints) and constructs the C++ types internally.
3. **Call `mlx_ssd_metrics_snapshot` directly** from Rust — it already has C ABI.
4. **Do not** try to expose `MTL::SharedEvent` across FFI; keep it behind an opaque handle.

---

## 6. Metal shaders

The `.air` files in `mlx/backend/metal/kernels/` are compiled Metal AIR bitcode. The human-readable sources are sibling `.metal` files.

### `fence.metal` → `fence.air`

`mlx/backend/metal/kernels/fence.metal:1–53`, three system-scope sync kernels:

1. `input_coherent` (15–26) — reads and re-writes a coherent device buffer, then `metal::atomic_thread_fence(mem_device, seq_cst)`. Forces CPU→GPU visibility without doing work.
2. `fence_update` (29–37) — single-threaded, writes a new timestamp and issues a fence. CPU signals GPU.
3. `fence_wait` (40–52) — single-threaded, busy-spins reading a timestamp until it hits a target. GPU waits for CPU. Expensive; not on the SSD streamer hot path.

None of these fire in the current streamer path — they're utilities available for future async coordination.

### `moe_stream.metal` → `moe_stream.air`

`mlx/backend/metal/kernels/moe_stream.metal:1–59`, one kernel `streamed_moe_gemm` (see §1). Input bf16 + packed 4-bit weights → float32 GEMM. Compiled offline via `xcrun -sdk macosx metal -c moe_stream.metal -o moe_stream.air` and loaded at runtime.

---

## 7. What to change for a 4–8 GB budget

### Trivial — just call the existing API

1. `allocator::allocator().set_memory_limit(size_t(8) << 30);` at init.
2. Construct `SSDStreamer` with `buffer_size_bytes` set to the largest single expert's byte size.
3. Poll `mlx_ssd_metrics_snapshot` for backpressure signal.

### Non-trivial — new code in `mtw-engine` / `mtw-cache`

1. **Expert-aware hot cache.** Keep K hottest experts pinned in device memory across forward passes. Requires:
   - An LRU or LFU indexed by `active_expert`.
   - Hook between `LoadSSDExpert::eval_impl` and `load_sync` that consults the cache first.
   - Deferred deallocation (currently the buffer is released immediately after compute).
   - Touches `mlx/core/moe_stream_op.cpp` and a new header alongside `ssd_streamer.h`.
2. **Async prefetch.** Add `load_async()` returning an `MTL::SharedEvent*`; rewire `LoadSSDExpert` to issue a GPU-side `wait_on_event` instead of blocking. Reactivates `io_queue_`.
3. **Multi-expert batching.** For MoE with 8-of-N routing, current code loads each of the 8 experts serially. An 8-wide async load pipeline would change the top-of-loop structure.

### Structural, not quick

- Right now eviction is purely memory-pressure driven. Usage-adaptive caching (§5.2 of the spec — histograms of activations, hot/warm/cold tiers) has no upstream precedent; we build it.
- Mesh-level re-sharding (§5.3) requires usage telemetry gossiped between `mtw-core` instances; also new.

---

## 8. Caller examples

None. No Python test, no Swift test, no C++ test, no fixture exercises `streamed_gather_mm` end-to-end in `SharpAI/mlx` at this commit. The only public anchor is `mlx_ssd_metrics_snapshot()`.

**Integration-test plan** for when `mtw-engine` ships FFI:

1. Build a minimal `.safetensors` file with two ~100 MB expert matrices.
2. Construct `SSDStreamer` with a 200 MB buffer.
3. Call `streamed_gather_mm` twice with different `active_expert` values.
4. Assert output shape, then snapshot metrics to confirm reads happened.

This test belongs in `mtw-engine/tests/` once the C++ bridge exists. It's the first artifact that proves the FFI round-trips.

---

## 9. Reading order for a new engineer

1. `mlx/backend/metal/ssd_streamer.h` — the primitive.
2. `mlx/backend/metal/ssd_streamer.mm` — implementation, focus on `load_sync` (line 41) and the dormant async hooks.
3. `mlx/core/moe_stream_op.h` — public C++ API plus the `extern "C"` metrics.
4. `mlx/core/moe_stream_op.cpp` — `LoadSSDExpert::eval_impl` (lines 63–139) is the entire hot path.
5. `mlx/backend/metal/kernels/moe_stream.metal` — the 4-bit MoE GEMM kernel.
6. `mlx/backend/metal/allocator.{h,cpp}` — where the memory ceiling lives, if you want to understand backpressure.
7. `mlx/backend/metal/kernels/fence.metal` — optional; relevant only when building the async path.

All paths relative to the MLX repo root (`~/Desktop/meshthatworks-deps/mlx` locally).
