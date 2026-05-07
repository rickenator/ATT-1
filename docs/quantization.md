# ATT-1 int8 quantization

Milestone 10 added a standalone int8 weight path for kernel correctness and
measurement. Milestone 24 wires CPU q8 into single-tile inference without
changing the model file format or adding q4. Milestone 49 adds checked-in q8
`.att1` tiny-model artifacts using the same per-row q8 representation.

## Representation

`att1_q8_matrix` stores row-major int8 weights plus one float32 scale per row:

```c
typedef struct att1_q8_matrix {
    size_t rows;
    size_t cols;
    int8_t *values;
    float *scales;
} att1_q8_matrix;
```

Rows are output rows. Columns are input dimensions. This means a dense
projection weight that is normally multiplied as `lhs[batch, in] *
rhs[in, out]` is quantized in transposed logical layout as
`weights_q8[out, in]`.

Activations stay float32.

## Per-row quantization

Each output row gets an independent symmetric scale:

```text
scale = max(abs(row_values)) / 127
q = round(value / scale)
```

Zero rows are valid. They use `scale = 1.0` and all quantized values are zero.

Only finite float32 inputs are accepted. Quantization fails cleanly for
non-finite values.

## Saturation

The supported int8 range is `[-127, 127]`. Quantized values are rounded to the
nearest integer and then clamped to that range. `-128` is intentionally unused
so the range remains symmetric around zero.

## q8xf32 matmul

`att1_matmul_q8xf32` computes:

```text
dst[batch, out] = lhs[batch, in] * dequant(weights_q8[out, in])^T
dequant(q) = (float)q * row_scale
```

The float32 matmul path remains the correctness reference. Tests compare q8
outputs to float32 outputs with a documented tolerance of `0.035` for the
current tiny fixtures.

## q8 `.att1` tensor layout

Milestone 49 activates dtype code `2` for q8 model-file tensors while keeping
the `.att1` container version unchanged.  A q8 tensor is a 2-D matrix in the
runtime q8 layout:

```text
shape = [out_dim, in_dim]
payload = int8 values[rows * cols] followed by float32 scales[rows]
nbytes = rows * cols + rows * 4
```

Eligible projection and output matrices are converted from the f32 ATT-1
logical shape into this `[out,in]` q8 layout.  `tok_embeddings.weight`,
RMSNorm weights, activations, KV cache data, residuals, and logits remain
float32.

Single-tile and cluster q8 backends borrow dtype-2 q8 tensors directly from a
q8 `.att1` file.  For f32 `.att1` files, those backends retain the existing
behavior of building owned runtime q8 copies during backend selection.

## CPU q8 single-tile inference

Milestone 24 adds `--backend cpu-q8` support to the existing single-tile
inference path. When a single-tile inference context selects the `cpu-q8`
backend, it uses dtype-2 file tensors directly when present, or builds owned
runtime q8 copies of projection and output weights from f32 tensors.

The q8 path uses `matmul_q8xf32` for:

- attention projections `wq`, `wk`, `wv`, and `wo`
- FFN projections `w_gate`, `w_up`, and `w_down`
- final output logits projection

RMSNorm, RoPE, softmax, SwiGLU, KV cache storage, residual adds, and
activations remain float32. CPU f32 remains the correctness reference.

For the dummy `.att1` fixture, tests compare one-token CPU f32 and CPU q8
logits with a maximum absolute tolerance of `0.15` and assert that the current
short generated-token fixture remains identical. Token equivalence is not a
general quantization contract for future models; when logits are close enough
to move an argmax boundary, generated tokens may diverge even if the q8 backend
is working correctly.

## CUDA q8xf32 prototype

Milestone 23 adds CUDA backend support for `matmul_q8xf32` without changing the
q8 representation. The CUDA path consumes the same `att1_q8_matrix` layout as
CPU q8:

- q8 weights are stored as row-major `weights_q8[out, in]`.
- one float32 scale is stored per output row.
- activations stay float32.
- dequantization uses `(float)q * row_scale`.

The prototype dequantizes q8 rows into a temporary float32 RHS matrix and runs
the multiply through the existing CUDA f32 matmul path. CPU q8 remains the
correctness reference for the CUDA q8 operator. Tests also compare q8 outputs
against CPU f32 where the existing q8 tolerance applies.

CUDA q4 is not implemented.

## CUDA q8 single-tile inference

Milestone 25 adds explicit `--backend cuda-q8` selection for single-tile
inference. It uses the same runtime q8 weight copies as the CPU q8 path, keeps
activations float32, and dispatches q8 projection/logit matmuls through the
CUDA `matmul_q8xf32` backend op.

CPU q8 remains the direct correctness reference for CUDA q8. Tests compare
one-token CUDA q8 logits against CPU q8 logits with a maximum absolute
tolerance of `1e-3`. The existing CPU q8 versus CPU f32 checks remain the f32
reference coverage for quantization error and use the documented `0.15`
dummy-model logit tolerance.

For the current dummy `.att1` fixture, CUDA q8 generated tokens are expected to
match CPU q8 for the short tested prompt. As with CPU q8, token equivalence is
not a general quantization contract for future models because small logit
differences can move an argmax boundary.

q4 is not implemented.

## Benchmark

### att1-q8-bench (kernel measurement)

`build/att1-q8-bench` runs a deterministic small f32 matmul and q8xf32 matmul
over the same data and prints timing and error counters:

```sh
./build/att1-q8-bench
./build/att1-q8-bench --iterations 10000
```

This benchmark is for kernel measurement only. `--backend cuda` exercises the
CUDA q8xf32 operator when CUDA is compiled in and available at runtime;
otherwise it reports unsupported cleanly.

### att1-bench q8 single-tile and cluster (Milestones 26-27)

`build/att1-bench` supports four inference backends for single-tile mode:

```sh
./build/att1-bench --model models/dummy/model.att1 --prompt hello --tokens 8 \
    --mode single --backend cpu-f32
./build/att1-bench --model models/dummy/model.att1 --prompt hello --tokens 8 \
    --mode single --backend cpu-q8
./build/att1-bench --model models/dummy/model.att1 --prompt hello --tokens 8 \
    --mode single --backend cuda
./build/att1-bench --model models/dummy/model.att1 --prompt hello --tokens 8 \
    --mode single --backend cuda-q8
```

Each run prints:

- `mode=single`, `backend=<name>` — explicit backend identification
- `generated_tokens=<n>`, `last_token=<id>` — token generation summary
- `tokens_decoded=<n>`, `token_time_us_total=<us>`, `token_time_us_max=<us>` — timing
- `logits_bytes_produced=<bytes>`, `kv_appends=<n>`, `kv_key_reads=<n>` — trace counters
- `layer[i].executions=<n> time_us=<us> kv_appends=<n>` — per-layer breakdown

CPU q8 cluster mode runs through the cluster inference path using q8 projection
matmuls with float32 activations and float32 KV/cache/state tensors preserved.
CPU f32 cluster inference remains the correctness reference.

Milestone 27 adds explicit cluster q8 validation:

- `att1-bench --mode cluster --backend cpu-q8` exits zero and reports
    `backend=cpu-q8`.
- Fabric packet counters and activation/logit byte counters remain active and
    nonzero in cpu-q8 cluster mode.
- CPU f32 cluster vs CPU q8 cluster logits are compared on the dummy model with
    max-abs tolerance `0.15` (same documented dummy-model q8 tolerance used for
    single-tile checks).
- Dummy-model generated token sequence is deterministic for current fixtures
    and is asserted in tests; for future models, token divergence can still occur
    when logits remain within tolerance around argmax boundaries.

Milestone 28 adds CUDA q8 cluster support using the same
`att1_cluster_infer_set_backend()` path.  For f32 model files, cluster q8
builds runtime q8 copies before decode.  For M49 q8 model files, it borrows
dtype-2 q8 tensors directly.  Activations remain float32.

CUDA q8 cluster mode validation:

- `att1-bench --mode cluster --backend cuda-q8` exits zero on CUDA builds and reports
  `backend=cuda-q8`.
- Fabric packet counters and activation/logit byte counters remain active and nonzero.
- CUDA q8 cluster vs CPU q8 cluster generated token sequences match exactly on the dummy
  model (greedy argmax is invariant to minor float32 rounding differences between CPU and
  cuBLAS).
- Non-CUDA builds reject `--backend cuda-q8` cluster mode with an explicit `unsupported`
  message and nonzero exit code.
- The cuda-q8 backend name (`ops->name == "cuda-q8"`) is verified before use to ensure no
  silent fallback to cpu-q8, cpu-f32, or cuda.

`tests/test_q8_bench.c` validates benchmark/trace behavior including cpu-q8
cluster exit/status, backend labeling, counters, cuda-q8 cluster success (CUDA-only),
and cuda-q8 cluster unsupported behavior on CPU-only builds.

`tests/test_q8_cluster.c` validates CPU f32 vs CPU q8 cluster logits tolerance,
trace/fabric counter parity, and deterministic dummy-model generated tokens.

`tests/test_cuda_q8_cluster.c` validates CPU q8 cluster vs CUDA q8 cluster token
sequence equivalence, trace/fabric counter parity, and no-silent-fallback behavior.

## Converted tiny q8 model

`models/real_tiny_q8/model.att1` is emitted from
`compiler/fixtures/tiny_llama_2l.safetensors` with:

```bash
python3 compiler/convert_llama_to_att1.py \
    --config compiler/fixtures/tiny_llama_config.json \
    --safetensors compiler/fixtures/tiny_llama_2l.safetensors \
    --weight-format q8 \
    --out models/real_tiny_q8/model.att1
```

`att1-inspect` reports q8 tensors with `dtype_name=q8`,
`quant=per-row-q8`, `q8_values`, and `q8_scales`.  Converter validation runs
the q8 artifact through CPU q8 single and cluster benchmarks, checks cluster
fabric packets, and compares the tiny fixture's CPU q8 `last_token` against
the f32 converted model running through the CPU q8 backend.  On CUDA builds
with an available runtime, the same fixture is validated with CUDA q8 single
and cluster modes.

Token equivalence is only asserted for the deterministic tiny fixture.  For
larger or future models, q8 logits may move an argmax boundary; numerical logit
tolerances remain the correctness contract.

## q8 conversion of BF16-source public models (M68)

Milestone 68 extends the q8 conversion path to BF16-source `.safetensors`
files.  The existing per-row q8 rules are unchanged; only the source dtype
handling in the comparison harness is updated.

### Source dtype handling

`compiler/compare_att1_to_source.py` now accepts BF16 and F16 source tensors
(coerced to F32 via `_coerce_bf16`/`_coerce_f16` from `load_safetensors.py`).
The static comparison checks the ATT-1 f32/q8 values against these coerced
float values; the error bounds are the same as for F32-source models.

### q8 conversion workflow for public models

```sh
# Convert a BF16-source model to f32 and q8 ATT-1 artifacts.
python3 compiler/convert_llama_to_att1.py \
    --safetensors ~/Models/<model>/model.safetensors \
    --config ~/Models/<model>/config.json \
    --out ~/Models/att1/<model>/model_f32.att1

python3 compiler/convert_llama_to_att1.py \
    --safetensors ~/Models/<model>/model.safetensors \
    --config ~/Models/<model>/config.json \
    --weight-format q8 \
    --out ~/Models/att1/<model>/model_q8.att1
```

The `--weight-format q8` flag quantises all eligible 2-D projection and output
weight matrices using per-row symmetric int8 quantisation.  `tok_embeddings.weight`,
RMSNorm weights, activations, KV cache entries, residuals, and logits remain
float32.

### Tolerance and token divergence for public models

| Check | Expected range |
|-------|---------------|
| f32 static max\_abs\_error (BF16 source) | ≈ 0 (BF16→F32 coercion is lossless) |
| q8 static max\_abs\_error | < 0.6 (per-row int8 symmetric; same tolerance as tiny fixture) |
| f32 vs q8 last\_token | Typically identical; may diverge at argmax boundaries |
| q8 cluster vs q8 single | Must be identical |

Token divergence between f32 and q8 is not a q8 defect when logits are
within the documented tolerance.  The correctness contract is the logit
tolerance; token sequences are secondary for models larger than the
deterministic tiny fixtures.

### Validation

`tests/test_bench_smoke.c` `check_q8_conversion()` (M68) validates:
1. BF16 fixture → q8 artifact (`build/m68_q8/model.att1`)
2. `att1-inspect`: `dtype_name=q8`, `quant=per-row-q8`
3. `att1-bench --backend cpu-q8 --mode single`
4. `att1-bench --backend cpu-q8 --mode cluster --tiles 2`
5. `compare_att1_to_source.py`: `q8_status: pass`, `result: pass` (numpy-skippable)

## Ownership

`att1_q8_matrix_alloc` allocates owned `values` and `scales` buffers.
`att1_q8_matrix_free` releases both buffers, clears the struct, and accepts
`NULL`.
