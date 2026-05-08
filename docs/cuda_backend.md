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

Full transformer inference through q8 CUDA weights is not implemented. CUDA q4
is not implemented.

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
6. **q8xf32 prototype** — CUDA `matmul_q8xf32` is validated against CPU
   q8xf32 for tiny hand-checkable, deterministic medium, zero-scale row, and
   saturated-value fixtures. Medium q8 output is also compared against CPU f32
   within the documented q8 tolerance.

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

### Milestone 23: q8xf32 matmul prototype

The CUDA backend now implements `matmul_q8xf32` behind the existing backend
vtable. It consumes the same per-row int8 quantization format as the CPU q8
path and keeps activations as float32. The prototype dequantizes q8 rows into a
temporary f32 matrix in `rhs[in, out]` layout, then executes the multiply through
the existing CUDA f32 matmul path. This keeps the implementation narrow while
validating CUDA-selected q8 operator dispatch and preserving the CPU q8 path as
the correctness reference.

`tests/test_cuda_matmul.c` covers Milestone 23 q8 cases:

1. **Tiny q8 known case** — hand-checkable q8 matrix/vector output.
2. **Medium deterministic q8** — CUDA q8 output matches CPU q8 and stays within
   the documented q8 tolerance versus CPU f32.
3. **Zero-scale row** — q8 rows with scale `0.0` produce zero contribution,
   matching CPU q8 behavior.
4. **Saturation edges** — `-127` and `127` values match CPU q8 behavior.
5. **No silent CPU fallback** — backend name is asserted as `"cuda"` before
   calling `matmul_q8xf32`, and the CUDA-selected q8 op must succeed.
6. **Unsupported path clarity** — non-CUDA builds/devices still report
   `ATT1_ERR_UNSUPPORTED` from `att1_backend_cuda_create` and skip CUDA q8
   execution cleanly.

At this milestone only the CUDA q8 matmul operator is added; single-tile CUDA
q8 inference integration is documented below. CUDA q8 cluster inference and
CUDA q4 remain unsupported.

### Milestone 25: CUDA q8 single-tile inference

The CUDA backend exposes an explicit `cuda-q8` backend name through
`att1_backend_cuda_q8_create`. It shares the existing CUDA f32 operators plus
the Milestone 23 `matmul_q8xf32` implementation, but the distinct backend name
lets single-tile inference select the q8 weight path without changing
`--backend cuda` f32 behavior.

Single-tile inference treats `cuda-q8` like `cpu-q8` for weight preparation:
the `.att1` file remains float32, owned runtime q8 copies are built for
attention, FFN, and output projection weights, and activations stay float32.
Those q8 matmuls dispatch through the selected CUDA backend. Non-matmul ops
remain the existing CUDA f32 ops.

`tests/test_cuda_infer.c` covers Milestone 25 q8 inference:

1. **CUDA q8 logits** — one-token CUDA q8 logits match CPU q8 logits within
   `1e-3`.
2. **Generated tokens** — the current dummy-model short prompt matches CPU q8;
   future token divergence remains acceptable when logits are within tolerance.
3. **No silent CPU fallback** — the selected backend must report
   `backend=cuda-q8`, not `cpu-q8` or `cpu-f32`.
4. **Unsupported path clarity** — non-CUDA builds/devices report
   `ATT1_ERR_UNSUPPORTED` from `att1_backend_cuda_q8_create` and CLI
   `--backend cuda-q8` exits nonzero cleanly.

`tests/test_cuda_bench.c` also validates `att1-bench --mode single --backend
cuda-q8` output labeling and unsupported-path behavior.

CUDA q8 cluster inference is implemented at Milestone 28. `att1-bench --mode cluster
--backend cuda-q8` exits zero on CUDA builds and reports `backend=cuda-q8`. Non-CUDA
builds reject the combination with an explicit `unsupported` message. CUDA q4
remains unsupported.

## CLI

`att1-bench` accepts:

```sh
./build/att1-bench --model models/dummy/model.att1 --prompt hello --tokens 8 --mode single --backend cpu-f32
./build/att1-bench --model models/dummy/model.att1 --prompt hello --tokens 8 --mode single --backend cpu-q8
./build/att1-bench --model models/dummy/model.att1 --prompt hello --tokens 8 --mode single --backend cuda
./build/att1-bench --model models/dummy/model.att1 --prompt hello --tokens 8 --mode single --backend cuda-q8
./build/att1-bench --model models/dummy/model.att1 --prompt hello --tokens 8 --mode cluster --tiles 2 --backend cuda
./build/att1-bench --model models/dummy/model.att1 --prompt hello --tokens 8 --mode cluster --tiles 2 --backend cuda-q8
```

`cpu-f32` remains the default. `cpu-q8` runs single-tile inference with runtime
q8 copies of projection and output weights. `cuda` runs the CUDA f32 path, while
`cuda-q8` runs single-tile CUDA q8 inference. CUDA selections report unsupported
or unavailable when CUDA is not compiled in or no CUDA runtime device is
available.

`att1-q8-bench` accepts `--backend cpu-q8|cuda`. The CUDA option exercises the
CUDA q8xf32 matmul operator. It does not imply full q8 model inference.

## Milestone 26: q8 benchmark and trace integration

`tests/test_q8_bench.c` validates q8 benchmark behavior for both CPU and CUDA
backends across all four supported inference backends:

1. **CPU q8 single mode** — `att1-bench --mode single --backend cpu-q8` exits
   zero and reports `backend=cpu-q8`, `mode=single`, trace counters
   (`tokens_decoded`, `logits_bytes_produced`, `kv_appends`, etc.).
2. **CPU f32 vs CPU q8 deterministic** — both backends generate the same
   `last_token` for the fixed dummy model prompt.
3. **CPU q8 vs CUDA q8 deterministic** — both backends generate the same
   `last_token` (CUDA-only; skipped when CUDA unavailable).
4. **CUDA q8 cluster mode** — `att1-bench --mode cluster --backend cuda-q8`
   exits zero on CUDA builds; cpu-only builds reject it with an `unsupported`
   message. Validated in Milestone 28.
5. **CUDA q8 unsupported on CPU-only build** — `--backend cuda-q8` exits
   non-zero with an `unsupported` message when CUDA is not compiled in.

The four supported benchmark backend names and their semantics:

| Backend    | Matmul path     | Norm/RoPE/etc | Cluster | Notes              |
|------------|-----------------|---------------|---------|--------------------|
| `cpu-f32`  | f32             | f32 CPU       | yes     | correctness ref    |
| `cpu-q8`   | q8×f32 CPU      | f32 CPU       | yes     | quantization ref   |
| `cuda`     | f32 cuBLAS      | f32 CUDA ops  | yes     | CUDA f32 path      |
| `cuda-q8`  | f32 cuBLAS*     | f32 CUDA ops  | yes     | CUDA q8 path       |

\* Cluster inference uses float32 weights via `att1_transformer_block_forward_backend()`;
the q8×f32 matmul path is used only in single-tile inference.

Cluster mode is supported for `cpu-f32`, `cpu-q8`, `cuda`, and `cuda-q8`.
All four backends use `att1_cluster_infer_set_backend()` to wire into the
cluster path.

## Milestone 28: CUDA q8 cluster inference integration

`att1_cluster_infer_set_backend()` now accepts the `cuda-q8` backend without
restriction. The cuda-q8 backend dispatches transformer block computations
through the existing cuBLAS f32 operators (same as the `cuda` backend) since
cluster inference drives `att1_transformer_block_forward_backend()` with
float32 model weights. Activations remain float32.

`tests/test_cuda_q8_cluster.c` validates CUDA q8 cluster inference:

1. **Next-token equivalence** — CPU q8 cluster and CUDA q8 cluster produce the
   same greedy-sampled token for the same input.
2. **Generated sequence equivalence** — CPU q8 cluster and CUDA q8 cluster
   produce the same 2-token and 4-token sequences on the dummy model.
3. **Trace/counter preservation** — CUDA q8 cluster fabric packet counts and
   activation/logit byte counters are nonzero and match CPU q8 cluster counters.
4. **No silent fallback** — the `cuda-q8` backend name (`ops->name`) is
   verified as `"cuda-q8"` before the backend is installed into the cluster
   context.

`tests/test_q8_bench.c` also validates `att1-bench --mode cluster --backend
cuda-q8` output labeling, counter sanity, last-token agreement with CPU q8
cluster, and unsupported-path behavior on CPU-only builds.

## Milestone 86: CUDA q4 implementation plan

Documentation-only.  No CUDA q4 kernel or C source changes.

### Current status (resolved in M88)

`--backend cuda-q4 --mode single` exits zero on a CUDA-capable host as of M88.
Cluster mode remains rejected until M89.

### Recommended approach

**M87 prototype:** dequantize-on-the-fly custom CUDA kernel
(`cuda_backend_matmul_q4xf32`).  Each thread group reads packed nibbles and
per-group float32 scales from global memory, dequantizes to float32 in
registers, and accumulates the dot product without allocating an intermediate
f32 weight buffer.  This matches the CPU dequantize-then-multiply approach and
preserves the memory-reduction benefit of q4.

A pre-dequantize path (unpack q4 → f32 device buffer, then cuBLAS) is
acceptable as a correctness-first fallback if the custom kernel is blocked.

CUTLASS-style tiled kernels are deferred past M90.

### Backend name and no-silent-fallback policy

The CUDA q4 backend must:
- set `ops->name = "cuda-q4"`,
- register a non-NULL `matmul_q4xf32` device function,
- never fall back to the CPU q4 path silently.

Any attempt to run CUDA q4 on a CPU-only build must exit non-zero with a clear
`unsupported` message on stderr, consistent with current behavior.

### Wire format passed to CUDA

The kernel receives exactly the on-disk layout:
- packed nibble bytes (`rows * cols / 2` bytes, contiguous),
- float32 scale array (`rows * (cols / group_size) * 4` bytes, immediately
  after packed bytes),
- `group_size` as a launch parameter (32 by default, from `tensor.flags & 0xFF`).

No reformatting or transposing at load time.  The kernel handles the low/high
nibble split and signed `[-7, 7]` range directly.

### Milestone split

| Milestone | Scope |
|-----------|-------|
| M87 | Custom dequantize-on-the-fly CUDA q4×f32 matmul kernel; unit tests vs CPU q4; no inference |
| M88 | CUDA q4 single-tile inference; `--backend cuda-q4 --mode single` exits zero |
| M89 | CUDA q4 cluster inference; `--backend cuda-q4 --mode cluster` exits zero |
| M90 | Backend matrix + `validate_public_q4_backends.py --include-cuda` pass for cuda-q4 |

## Milestone 87: CUDA q4 matmul prototype

`cuda_backend_matmul_q4xf32()` implemented using the
pre-dequantize-on-CPU-then-cuBLAS approach (Option B from M86 plan).

### Implementation

- Dequantizes the entire weight matrix to float32 on the CPU before the cuBLAS
  call.  For each weight row, iterates over groups of `group_size` packed
  nibbles: low nibble = even column index, high nibble = odd column index.
  Sign-extends 4-bit values from `[0,15]` to signed `[-7,7]` by subtracting 16
  when the nibble value > 7.  Multiplies by the per-group float32 scale.
- Stores the result in **transposed column-major** layout
  (`dequant_rhs[col * rows + row]`) matching the `cublasSgemm` column-major
  convention used by the existing q8 path.
- Delegates to `cuda_backend_matmul_f32()` for the cuBLAS SGEMM call.
- `cuda_q4_backend_ops` ("cuda-q4"): `matmul_q4xf32` populated; all inference
  ops (`rmsnorm_f32`, `softmax_f32`, `rope_f32`, `ffn_swiglu_f32`) set to NULL
  to prevent accidental use for full inference before M88.
- `att1_backend_cuda_q4_create()` exported.

### Tolerance

CUDA q4 vs CPU q4: `1e-4f` (identical dequant algorithm, f32 arithmetic
rounding only).  CPU q4 vs CPU f32: `0.35f` (quantisation loss).

### Deferred

Custom CUDA dequantize-on-the-fly kernel deferred to a future milestone.
Inference wiring (`att1_infer_create_q4` CUDA path) deferred to M88.

## Milestone 88: CUDA q4 single-tile inference

`--backend cuda-q4 --mode single` now exits zero on a CUDA-capable host.

### Implementation

- `cuda_q4_backend_ops` inference ops (`rmsnorm_f32`, `softmax_f32`,
  `rope_f32`, `ffn_swiglu_f32`) populated with the existing CUDA functions
  shared by `cuda_backend_ops` and `cuda_q8_backend_ops`.
- `att1_attention_forward_backend_q4` and
  `att1_transformer_block_forward_backend_q4` route all `matmul_q4xf32` calls
  through new static helpers (`attention_q4_matmul`, `block_q4_matmul`) that
  dispatch through `backend->ops->matmul_q4xf32` when non-NULL, with a CPU
  fallback for the `cpu-q4` backend (which leaves the slot NULL).
- `infer_backend_is_q4()` extended to accept "cuda-q4" in addition to
  "cpu-q4".
- Output projection in `att1_infer_decode_token` routed through new
  `infer_matmul_q4()` helper (same vtable-or-fallback pattern).
- `att1-bench --backend cuda-q4 --mode single` now creates a cpu-q4 infer
  context (to load q4 weights) then swaps to the cuda-q4 backend via
  `att1_infer_set_backend`.
- Cluster mode with `--backend cuda-q4` remains rejected with a clear error.
- `tests/test_cuda_infer_q4.c`: 3 tests (no-silent-fallback, logits match
  cpu-q4 within `Q4_LOGIT_TOL=0.35f`, generated tokens identical to cpu-q4).
  Skipped on CPU-only builds.
- `test_q4_bench.c` `cuda-q4` test updated: expects exit-zero + `backend=cuda-q4`
  when CUDA is available; exit-nonzero with error message otherwise.

### Pattern: cuda-q4 single-tile inference

```c
att1_infer_create_q4(model, &infer);          /* load q4 weights (cpu-q4) */
att1_backend_cuda_q4_create(&cuda_backend);   /* create cuda-q4 backend   */
att1_infer_set_backend(infer, cuda_backend);  /* swap; triggers q4 prep   */
/* cuda_backend now owned by infer */
```

### Cluster deferred

`att1_cluster_infer_create_q4` does not accept "cuda-q4".  CUDA q4 cluster
inference is Milestone 89.

## Milestone 89: CUDA q4 cluster inference

`--backend cuda-q4 --mode cluster` now exits zero on a CUDA-capable host.

### Implementation

- `cluster_backend_is_q4()` in `src/cluster_infer.c` extended to accept
  "cuda-q4" in addition to "cpu-q4".  This enables `use_q4 = 1` in the decode
  loop and causes `att1_cluster_infer_set_backend` to call `cluster_prepare_q4`
  when swapping to cuda-q4 (idempotent; releases and reloads q4 views).
- `cluster_matmul_q4()` static helper added: dispatches output-projection
  matmul through `backend->ops->matmul_q4xf32` when non-NULL, with CPU fallback.
- Output projection in `att1_cluster_infer_decode_token` routed through
  `cluster_matmul_q4` (same vtable-or-fallback pattern as M88).
- `att1-bench` `run_cluster` and `run_cluster_external`:
  - cuda-q4 early-rejection blocks removed.
  - `is_q4` check extended to include "cuda-q4".
  - After `att1_cluster_infer_create_q4`, a cuda-q4 backend-swap block calls
    `att1_cluster_infer_set_backend` when `backend_name == "cuda-q4"`.
  - Error messages generalised from "cpu-q4" to "q4".
- `tests/test_cuda_cluster_infer_q4.c`: 4 tests (no-silent-fallback, fabric
  counters nonzero, logits match cpu-q4 cluster within `Q4_LOGIT_TOL=0.35f`,
  generated tokens identical to cpu-q4 cluster).  Skipped on CPU-only builds.
- `test_q4_bench.c` extended with `test_q4_bench_cuda_q4_cluster_status`:
  cluster mode with `--tiles 2 --backend cuda-q4`; same CUDA-conditional logic
  as the single-mode check.

### Pattern: cuda-q4 cluster inference

```c
att1_cluster_infer_create_q4(model, &cfg, &infer);  /* load q4 weights (cpu-q4) */
att1_backend_cuda_q4_create(&cuda_backend);         /* create cuda-q4 backend   */
att1_cluster_infer_set_backend(infer, cuda_backend);/* swap; triggers q4 prep   */
/* cuda_backend now owned by infer */
```


---

## Milestone 90: CUDA q4 benchmark and backend-matrix integration

`test_backend_matrix` now includes cuda-q4 single and cluster runtime entries,
giving complete CPU/CUDA × single/cluster coverage for q4.

### Backend matrix — group 4 (q4_tiny fixture)

| Backend  | Mode    | Shard plan | Requires CUDA | Group |
|----------|---------|------------|---------------|-------|
| cpu-q4   | single  | runtime    | no            | 4     |
| cpu-q4   | cluster | runtime    | no            | 4     |
| cuda-q4  | single  | runtime    | yes           | 4     |
| cuda-q4  | cluster | runtime    | yes           | 4     |

On a CUDA host all four entries pass and produce the same `last_token` (cross-
backend consistency).  On a CPU-only host the cuda-q4 entries are skipped.

### Validation

```
make clean && make && make test
# backend_matrix: 14/28 passed, 14 skipped, 0 failed  (CPU-only host)

make clean && make CUDA=1 && make test CUDA=1
# backend_matrix: 28/28 passed, 0 skipped, 0 failed  (CUDA host — verified)
```

### No new inference code

All inference wiring was completed in M88 (single) and M89 (cluster).  M90
adds only two `k_matrix[]` entries in `tests/test_backend_matrix.c` and
relaxes the M85 Python/C smoke assertion from `status=unsupported` to
`result: pass`.

---

## Milestone 91: CUDA q4 public-model validation report

`compiler/validate_public_q4_cuda.py` produces a per-row CUDA q4 validation
report for a local converted q4 ATT-1 artifact.

### Validation matrix

| Backend  | Mode    | Shard plan | CPU-only status | CUDA host status |
|----------|---------|------------|-----------------|------------------|
| cpu-q4   | single  | n/a        | pass            | pass             |
| cpu-q4   | cluster | runtime    | pass            | pass             |
| cpu-q4   | cluster | metadata   | plan_unsupported| plan_unsupported |
| cuda-q4  | single  | n/a        | unavailable     | pass             |
| cuda-q4  | cluster | runtime    | unavailable     | pass             |

`plan_unsupported` and `unavailable` are expected outcomes; neither fails the
overall report.

### Backend-name verification

cuda-q4 rows that exit zero but report a backend from
`{cpu-q4, cpu-q8, cpu-f32, cuda, cuda-q8}` are marked `fail` (silent
fallback detected).

### Fabric packet verification

All `status=pass` cluster rows must have `fabric_packets_sent > 0`.

### No new inference code or CUDA kernels

All inference wiring is from M88/M89.  M91 adds only the Python script and a
C smoke test (`check_public_q4_cuda_smoke` in `tests/test_bench_smoke.c`).

---

## Milestone 92: Backend comparison report

`compiler/backend_comparison_report.py` runs all f32/q8/q4 × CPU/CUDA ×
single/cluster bench combinations and produces a per-row comparison report.

### Comparison matrix

| Backend  | Mode    | Artifact | Required | CPU-only status | CUDA host status     |
|----------|---------|----------|----------|-----------------|----------------------|
| cpu-f32  | single  | f32      | yes      | pass            | pass                 |
| cpu-f32  | cluster | f32      | yes      | pass            | pass                 |
| cpu-q8   | single  | q8       | yes      | pass            | pass                 |
| cpu-q8   | cluster | q8       | yes      | pass            | pass                 |
| cpu-q4   | single  | q4       | yes      | pass            | pass                 |
| cpu-q4   | cluster | q4       | yes      | pass            | pass                 |
| cuda     | single  | f32      | no       | pending         | pass                 |
| cuda     | cluster | f32      | no       | pending         | pass                 |
| cuda-q8  | single  | q8       | no       | pending         | pass                 |
| cuda-q8  | cluster | q8       | no       | pending         | pass                 |
| cuda-q4  | single  | q4       | no       | pending         | pass                 |
| cuda-q4  | cluster | q4       | no       | pending         | pass                 |

CUDA rows require `--include-cuda`.  Without it they report `status=pending`
(not a failure).  On a CPU-only host with `--include-cuda` they report
`status=unavailable` (also not a failure).  Only `status=fail` on
`required=True` rows fails the overall report.

### Fabric-packet and silent-fallback checks

Same logic as M91: required cluster rows must have `fabric_packets_sent > 0`;
cuda-q4 silent-fallback detection via `_FORBIDDEN_FALLBACKS`.

### C smoke test

`check_backend_comparison_smoke` in `tests/test_bench_smoke.c` exercises
the script with the checked-in real_tiny fixtures (no `--include-cuda`) and
confirms `result: pass`, CPU rows present, CUDA rows `status=pending`, notes
present, and JSON keys correct.

CUDA host verification: pending RTX 3090 signoff.
