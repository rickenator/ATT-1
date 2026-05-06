# ATT-1 CUDA backend skeleton

Milestone 12 adds an opt-in CUDA backend path without changing default CPU
behavior. Normal builds do not require CUDA headers, libraries, or a GPU.

## Build modes

Default CPU-only build:

```sh
make
```

Opt-in CUDA build (requires CUDA toolkit and cuBLAS):

```sh
make CUDA=1
```

`CUDA=1` defines `ATT1_ENABLE_CUDA`, adds `$(CUDA_HOME)/include`, adds
`$(CUDA_HOME)/lib64`, and links `-lcudart -lcublas`. `CUDA_HOME` defaults to
`/usr/local/cuda` and can be overridden:

```sh
make CUDA=1 CUDA_HOME=/opt/cuda
```

## Availability

`att1_backend_cuda_available()` returns nonzero only when ATT-1 is built with
`CUDA=1` and the CUDA runtime reports at least one CUDA device.

Without `CUDA=1`, CUDA APIs compile as stubs:

- `att1_backend_cuda_available()` returns `0`
- `att1_backend_cuda_create()` returns `ATT1_ERR_UNSUPPORTED`
- CUDA copy helpers return `ATT1_ERR_UNSUPPORTED` after argument validation

## Lifecycle

The CUDA backend exposes the same lifecycle hooks as other backends:

- `att1_backend_cuda_create`
- `att1_backend_destroy`
- `alloc`
- `free`
- `sync`

When CUDA is enabled and available, `alloc`, `free`, and `sync` use CUDA runtime
calls. Host/device copy helpers are available for explicit transfers:

- `att1_backend_cuda_copy_host_to_device`
- `att1_backend_cuda_copy_device_to_host`

## Kernel status

### Milestone 14: matmul_f32 (cuBLAS)

`matmul_f32` is implemented using `cublasSgemm`.  Input and output buffers are
host-resident; the implementation allocates temporary device memory per call,
copies inputs to device, executes the GEMM, copies the result back, and frees
device memory.  This proves CUDA execution correctness without requiring
device-side buffer management in callers.

Row-major semantics are preserved via the standard column-major trick:

```
C^T = B^T × A^T
cublasSgemm(handle, N, N, cols, rows, inner,
            alpha, d_rhs, cols, d_lhs, inner,
            beta,  d_dst, cols)
```

A cuBLAS handle is created and destroyed per call.  This is intentionally
simple for the prototype; a persistent handle will be introduced when the
backend data struct gains a proper destroy hook.

### Milestone 15: rmsnorm_f32 (cuBLAS)

`rmsnorm_f32` is implemented without introducing a separate CUDA kernel build
path. The current prototype stays inside the existing `cc` + cuBLAS toolchain:

1. `cublasSdot` computes `sum(src[i] * src[i])`
2. the host computes `scale = 1 / sqrt((sum / count) + epsilon)`
3. `cublasSdgmm` performs elementwise `src[i] * weight[i]`
4. `cublasSscal` applies the final normalization scale in-place

This keeps default builds CUDA-free, keeps `make CUDA=1` opt-in, and provides a
real CUDA-backed RMSNorm path using CPU f32 RMSNorm as the correctness reference.

The prototype also adds a private cuBLAS-based vector multiply helper used by
RMSNorm. Standalone CUDA SiLU, add, and other elementwise primitives are not yet
exposed through the backend API because the current milestone only needs the
multiply primitive to validate normalization.

### Milestone 16: ffn_swiglu_f32 (CUDA-backed)

`ffn_swiglu_f32` is implemented in the CUDA backend and now provides a real
CUDA-backed SwiGLU helper path through the existing backend API.

Implementation details:

1. SiLU(gate) is computed deterministically on host for the prototype path.
2. SiLU(gate) is copied to device.
3. Elementwise multiply with `value` runs through the existing cuBLAS-backed
   vector multiply helper.
4. Result is copied back to host.

This is intentionally scoped to the current API and avoids introducing a new
CUDA kernel toolchain requirement while providing a real CUDA execution path for
FFN gating behavior.

### Milestone 17: rope_f32 (CUDA-backed)

`rope_f32` is implemented in the CUDA backend for Q/K vectors with semantics
matched to `att1_rope_f32`.

Implementation details:

1. Input vector is copied to device.
2. Each even/odd pair (`[0,1]`, `[2,3]`, ...) is rotated on device using
   `cublasSrot`.
3. Per-pair angle matches CPU reference exactly:
   `angle = position * (1 / pow(theta, i / count))` where `i` is the even index.
4. Result is copied back to host.

To map BLAS rotation convention to RoPE convention, the implementation uses
`-sin(angle)` with `cublasSrot`, yielding the RoPE matrix:

```
[c -s]
[s  c]
```

### Milestone 18: attention_forward (via component ops + softmax)

`attention_forward` is implemented as a full forward pass through the existing
backend API. It is not a separate kernel operation; instead, it composes the
existing CUDA operators (matmul, rope, softmax) to perform causal self-attention
for batch size 1.

The existing CPU attention function (`att1_attention_forward_backend`) already
uses the backend ops API to delegate Q/K/V projection, RoPE, and softmax to
backend implementations. For the CUDA backend, this means:

1. Projection matmuls (`Wq`, `Wk`, `Wv`, `Wo`) → `cuda_backend_matmul_f32`
2. RoPE application → `cuda_backend_rope_f32`  
3. Softmax over attention scores → `cuda_backend_softmax_f32` (new)
4. KV cache operations (append/lookup) → CPU-side
5. Attention weight accumulation → CPU-side loops (small vectors, O(cache_length²) scoring)

Implementation of `cuda_backend_softmax_f32`:

The softmax operation receives host-resident float arrays and produces normalized
probabilities in-place. Since cuBLAS does not provide a softmax operation, softmax
is implemented using numerically stable CPU computation:

1. Find max value for stability: `max(values[i])`
2. Compute `exp(values[i] - max)` in-place, accumulate sum
3. Divide by sum to normalize

This keeps the CUDA backend within the existing C11 + cuBLAS toolchain while
providing correct numerical softmax that works with the causal masking semantics
of the attention forward pass.

### Milestone 19: transformer_block_forward (via composed CUDA ops)

`transformer_block_forward` is enabled for CUDA through the existing backend API
composition path in `att1_transformer_block_forward_backend`.

No dedicated CUDA block kernel is introduced. Instead, block execution uses the
already validated CUDA-backed primitives for batch size 1 decode:

1. pre-attention RMSNorm via `cuda_backend_rmsnorm_f32`
2. causal self-attention via `att1_attention_forward_backend`, which uses:
   `cuda_backend_matmul_f32`, `cuda_backend_rope_f32`, and
   `cuda_backend_softmax_f32`
3. residual add (host-side, same backend API path)
4. pre-FFN RMSNorm via `cuda_backend_rmsnorm_f32`
5. FFN projections via `cuda_backend_matmul_f32`
6. SwiGLU via `cuda_backend_ffn_swiglu_f32`
7. final projection via `cuda_backend_matmul_f32`
8. final residual add (host-side, same backend API path)

This milestone intentionally keeps the patch narrow and does not attempt full
CUDA model inference. CUDA q8 remains out of scope.

### Milestone 20: single-tile inference integration (CUDA backend)

Single-tile decode now runs through the CUDA backend when selected by caller
(`att1_infer_set_backend` with a CUDA backend handle, or `att1-bench`
`--mode single --backend cuda`).

Integration details:

1. The existing single-tile inference loop in `att1_infer_decode_token`
   already routes per-layer execution through
   `att1_transformer_block_forward_backend`.
2. With a CUDA backend selected, each block executes via the Milestone 19
   composed CUDA primitive path.
3. Final RMSNorm and logits projection in `att1_infer_decode_token` also use
   backend ops, so CUDA selection applies consistently for the full single-tile
   decode step.

Behavioral scope:

- Byte tokenizer and greedy sampler behavior are unchanged.
- CUDA unsupported/unavailable remains explicit through
  `att1_backend_cuda_create` returning `ATT1_ERR_UNSUPPORTED`.
- CUDA cluster inference is intentionally out of scope for this milestone.

### Not yet implemented

`matmul_q8xf32` still returns failure. Full transformer inference via CUDA is
not attempted until all operator kernels are validated.

## Tests

`tests/test_cuda_matmul.c` validates the CUDA f32 matmul prototype:

1. **Tiny known** — 2×3 × 3×1 with hand-checkable expected values [50, 122].
2. **Larger deterministic** — 4×8 × 8×4 with deterministic fill, compared
   to the CPU f32 reference within 1e-3 tolerance.
3. **Shape handling** — NULL and zero-dimension arguments fail cleanly on
   both CPU reference and CUDA backend.
4. **CUDA unavailable** — `att1_backend_cuda_create` returns
   `ATT1_ERR_UNSUPPORTED` when CUDA is not compiled in or no GPU is present.
5. **No silent fallback** — Backend name is asserted as `"cuda"` before
   calling `matmul_f32`; result is compared to CPU reference to confirm
   CUDA execution produced correct output.

`tests/test_cuda_norm.c` validates the CUDA RMSNorm prototype:

1. **Tiny RMSNorm** — small deterministic input compared to CPU RMSNorm.
2. **Larger deterministic RMSNorm** — 32-element generated input and weights,
   compared to the CPU f32 reference within 1e-3 tolerance.
3. **Shape handling** — NULL arguments, zero count, and nonpositive epsilon
   fail cleanly on both CPU reference and CUDA backend.
4. **CUDA unavailable** — `att1_backend_cuda_create` returns
   `ATT1_ERR_UNSUPPORTED` when CUDA is not compiled in or no GPU is present.
5. **No silent fallback** — Backend name is asserted as `"cuda"` before
   calling `rmsnorm_f32`; result is compared to CPU reference.

`tests/test_cuda_ffn.c` validates the CUDA FFN/SwiGLU prototype:

1. **Tiny hand-checkable FFN** — deterministic `d_model=2`, `d_ff=4` composed
   through matmul + SwiGLU + matmul and compared to CPU reference.
2. **Medium deterministic FFN** — deterministic `d_model=4`, `d_ff=8` end-to-end
   FFN comparison against CPU within 1e-3 tolerance.
3. **Zero weights** — confirms FFN output is zero for zero weights.
4. **Activation behavior** — mixed negative/positive values verify SwiGLU matches
   CPU SiLU gating semantics.
5. **Shape failure / unsupported** — invalid dimensions fail cleanly and
   unavailable CUDA reports unsupported.
6. **No silent fallback** — backend name is asserted as `"cuda"` and output is
   compared against CPU reference.

`tests/test_cuda_rope.c` validates the CUDA RoPE prototype:

1. **Position 0 identity** — RoPE with `position=0` leaves vectors unchanged.
2. **Pair layout** — validates independent rotations for pairs `[0,1]`, `[2,3]`,
   `[4,5]`.
3. **Odd dimension failure** — odd `head_dim` fails cleanly.
4. **Deterministic multi-head/multi-position** — per-head CPU vs CUDA comparisons
   across several positions within 1e-3 tolerance.
5. **Unsupported path and no silent fallback** — unavailable CUDA is reported as
   unsupported, and CUDA-selected backend is asserted as `"cuda"` before result
   comparison against CPU reference.

`tests/test_cuda_attention.c` validates the CUDA causal attention forward pass:

1. **Position 0 causal mask** — Position 0 attends only to token 0, reproducing
   attention input as output (identity Wo). CUDA output matches CPU f32 within
   1e-3 tolerance.
2. **Position N causal mask** — Position N attends to tokens 0..N. Output
   accumulates weighted average of visible history. CUDA output matches CPU
   reference within 1e-3 tolerance.
3. **Future KV no effect** — A future token added to KV cache does not affect
   earlier position outputs due to causal masking. CUDA and CPU both enforce
   causal causality.
4. **Softmax numerical stability** — Multiple positions and heads produce
   stable, normalized probabilities; CUDA output matches CPU within tolerance.
5. **Multi-head deterministic** — 2-head attention with deterministic weights
   produces consistent CUDA vs CPU output across both heads within tolerance.
6. **Invalid KV range fails cleanly** — Mismatched position vs cache length is
   rejected by the attention forward function on both backends.
7. **No silent fallback** — Backend name is asserted as `"cuda"` before running
   attention, confirming no silent CPU fallback occurs.

When CUDA is unavailable, the test suite skips gracefully with a message
indicating that CUDA tests were skipped.

`tests/test_cuda_transformer_block.c` validates CUDA-backed single-block
execution against CPU f32 reference:

1. **Zero attention/FFN weights preserve residual** — verifies output equals
   input when block projections are zero.
2. **CPU vs CUDA tiny deterministic** — compares 1-head tiny block output
   between CPU and CUDA within 1e-3 tolerance.
3. **KV cache position updates match CPU** — verifies cache length progression
   and position semantics are consistent across CUDA and CPU.
4. **Multi-head medium deterministic** — compares multi-head, medium-size block
   outputs across sequential positions within 1e-3 tolerance.
5. **No silent fallback** — asserts backend name is `"cuda"` and block forward
   succeeds only through CUDA-selected backend path.

`tests/test_cuda_infer.c` validates CUDA single-tile inference integration
against CPU f32 reference on `models/dummy/model.att1`:

1. **Next-token equivalence** — CPU and CUDA produce identical next token for
   the same byte input.
2. **Generated sequence equivalence (2 and 4 tokens)** — CPU and CUDA produce
   identical greedy-generated token sequences.
3. **Prompt prefill position parity** — decode position progression matches CPU
   during prompt prefill.
4. **KV cache update parity** — per-layer KV lengths match CPU after each
   prefill decode step.
5. **No silent fallback** — CUDA backend name is asserted and inference decode
   succeeds via explicit CUDA backend selection.
6. **Unsupported path clarity** — non-CUDA builds/devices must report
   `ATT1_ERR_UNSUPPORTED` from `att1_backend_cuda_create` and skip CUDA infer
   execution cleanly.

### Milestone 21: benchmark and trace integration

Benchmark tool (`att1-bench`) now exposes CUDA single-tile inference with trace
counters. Single-tile mode with `--backend cuda` reports per-token timing, total
timing, generated token count, and backend name.

`tests/test_cuda_bench.c` validates benchmark and trace integration:

1. **Single-tile mode exits successfully** — `att1-bench --backend cuda --mode single`
   exits with code 0, reports `backend=cuda`, `mode=single`, and trace counters
   (`tokens_decoded`, `token_time_us_total`, etc.).
2. **Cluster mode availability behavior** — `att1-bench --backend cuda --mode cluster`
   exits with code 0 and reports `backend=cuda` only when CUDA is compiled in
   and available at runtime; non-CUDA builds/devices report unsupported cleanly.
3. **Deterministic token generation** — CPU f32 and CUDA benchmarks produce
   identical token sequences for fixed dummy model and prompt, confirming
   inference correctness.
4. **Backend label clarity** — benchmark output explicitly prints `backend=cuda`
   (or `backend=cpu-f32`, `backend=cpu-q8` for other selections); no missing or
   misreported backend.
5. **CUDA unsupported message** — non-CUDA builds report `unsupported` error when
   `--backend cuda` is selected and CUDA is unavailable.

Trace output format (from `att1_trace_t` callbacks):

- `tokens_decoded=<count>` — number of tokens generated
- `token_time_us_total=<microseconds>` — total time across all tokens
- `token_time_us_max=<microseconds>` — max time for any single token
- `layer_time_us_total=<microseconds>` — total time spent in layer operations
- `layer[i].executions`, `layer[i].time_us`, `layer[i].kv_appends` — per-layer breakdown
- Per-tile activation/logit byte counts and fabric packet counts (single-tile: mostly 0)

Benchmark output format:

```
mode=single
backend=cuda
requested_tokens=8
benchmark_tokens=<actual>
generated_tokens=<count>
last_token=<token_id>
tokens_decoded=<count>
token_time_us_total=<us>
...
layer[0].executions=<n> time_us=<us> kv_appends=<m>
...
```

### Milestone 22: cluster inference integration

Cluster inference can now use the CUDA backend for per-tile/layer execution
through the existing backend vtable. The fabric simulation, packet accounting,
activation/logit byte counters, and trace output remain on the same host-side
cluster path as CPU cluster inference. CPU cluster remains the correctness
reference.

`tests/test_cuda_cluster.c` validates CUDA cluster inference:

1. **Next-token equivalence** — CPU cluster and CUDA cluster produce the same
   next token for the dummy `.att1` model.
2. **Generated sequence equivalence** — CPU cluster and CUDA cluster produce the
   same 2-token and 4-token generated sequences for a fixed prompt.
3. **Trace/counter preservation** — CUDA cluster fabric packet counts remain
   nonzero, and activation/logit byte counters match CPU cluster.
4. **No silent CPU fallback** — the selected backend is verified as `"cuda"`
   before it is installed into the cluster inference context.
5. **Unavailable CUDA clarity** — non-CUDA builds/devices still report
   `ATT1_ERR_UNSUPPORTED` from `att1_backend_cuda_create` and skip CUDA cluster
   execution cleanly.

CUDA q8 remains unsupported; CUDA cluster integration is limited to the f32
backend operations already exposed by the CUDA backend.

## CLI

`att1-bench` accepts:

```sh
./build/att1-bench --model models/dummy/model.att1 --prompt hello --tokens 8 --mode single --backend cpu-f32
./build/att1-bench --model models/dummy/model.att1 --prompt hello --tokens 8 --mode single --backend cpu-q8
./build/att1-bench --model models/dummy/model.att1 --prompt hello --tokens 8 --mode single --backend cuda
./build/att1-bench --model models/dummy/model.att1 --prompt hello --tokens 8 --mode cluster --tiles 2 --backend cuda
```

`cpu-f32` remains the default. `cpu-q8` can run the current f32 inference path
because it provides f32 ops plus q8 matmul support. `cuda` reports unsupported
or unavailable when CUDA is not compiled in or no CUDA runtime device is
available.

`att1-q8-bench` accepts `--backend cpu-q8|cuda`. The CUDA option is a skeleton
path and is not expected to run q8 matmul until CUDA kernels are implemented.
