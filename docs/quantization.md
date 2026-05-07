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

## Public q8 source comparison report (M69)

`compiler/compare_att1_to_source.py` accepts local public-model source
directories and external q8 artifacts:

```sh
python3 compiler/compare_att1_to_source.py \
    --model-dir ~/Models/<model> \
    --att1-f32 ~/Models/att1/<model>/model_f32.att1 \
    --att1-q8  ~/Models/att1/<model>/model_q8.att1 \
    --prompt-ids 1,2,3 \
    --q8-backend cpu-q8 \
    --report
```

The q8 report records the artifact path, dtype, backend, prompt token IDs,
logits shape from the source reference, max absolute/relative static tensor
error, q8 tolerance, next-token comparison, and pass/fail status.  Generated
public `.att1` files should remain outside Git, alongside the local source
model files.

## Ownership

`att1_q8_matrix_alloc` allocates owned `values` and `scales` buffers.
`att1_q8_matrix_free` releases both buffers, clears the struct, and accepts
`NULL`.

---

## q4 quantization planning (M73)

Milestone 73 is documentation-only.  No C source, Makefile, `.att1` format,
converter, or runtime is changed.

### Why q4 matters

| Motivation | Detail |
|------------|--------|
| Memory reduction | q4 halves storage relative to q8; a 7B-parameter f32 model (~28 GiB) fits in ~3.5 GiB q4 |
| Larger local/public models | Models that do not fit in CPU RAM at f32 or q8 may fit at q4 |
| AIMU storage efficiency | Tile-local SRAM is scarce; smaller weight payloads per tile reduce fabric transfer cost |

q8 remains the primary validated quantization path.  q4 is a future
compression tier for deployment, not a correctness reference.

### Candidate q4 schemes

| Scheme | Summary | Decision |
|--------|---------|----------|
| Simple symmetric int4 per-row | One scale per row, range [-7, 7] | Too coarse for rows with mixed magnitude |
| Grouped int4 with per-group scales | One scale per fixed-size group of values | **Recommended first format** |
| GPTQ/AWQ-style calibrated formats | Require activation-aware calibration data | Future/non-goal for now |
| MXFP4 | Block floating-point, hardware-specific | Future/non-goal unless AIMU adopts it |

### Recommended first q4 format: deterministic grouped int4

#### Value representation

- **Signed**: int4, range `[-7, 7]`.  `-8` is excluded to maintain a
  symmetric range around zero (mirrors q8 excluding `-128`).
- **Group size**: 32 (default candidate) or 64.  32 gives better accuracy;
  64 saves scale storage.  The group size is fixed at quantization time and
  stored in the tensor's quantization metadata field.
- **Scale per group**: one `float32` scale stored per group.
  `scale = max(abs(group_values)) / 7`  (or `1.0` for all-zero groups).
- **Zero-point policy**: symmetric, no explicit zero point.  The
  quantization grid is centered at zero.

#### Byte packing

Two int4 values are packed per byte, **low nibble first**:

```text
byte = (signed_hi_nibble << 4) | (signed_lo_nibble & 0xF)
values at index 2k   → low nibble  (bits 3:0)
values at index 2k+1 → high nibble (bits 7:4)
```

Signs are stored as two's complement 4-bit: value `-1` → nibble `0xF`,
value `7` → nibble `0x7`, value `-7` → nibble `0x9`, value `-8` is not used.

Row length must be divisible by 2 (so packing is complete) and ideally by
`group_size` (so group boundaries align with row boundaries).  The first
implementation may require `cols % group_size == 0` as a strict validation
rule.

#### Payload layout per tensor

```text
shape = [out_dim, in_dim]   (same row-major convention as q8)
n_groups_per_row = in_dim / group_size
packed_bytes_per_row = in_dim / 2
payload:
  uint8  packed[out_dim * in_dim / 2]      — packed int4 values, row-major
  float32 scales[out_dim * n_groups_per_row]  — one scale per group, row-major
```

Total bytes: `rows * cols / 2 + rows * (cols / group_size) * 4`.

### `.att1` format implications

| Concern | Plan |
|---------|------|
| dtype enum | Reserve `DTYPE_Q4 = 3`.  Loader already rejects unknown dtype values, so no existing model breaks.  The loader must not accept dtype 3 until the q4 tensor layout is fully specified and validated (M74). |
| Quantization metadata | `group_size` must be recoverable from the file.  Options: a fixed-width quantization parameter field in the tensor header, a future `quant_params` varint, or a separate section.  M74 will specify the wire encoding. |
| Alignment and padding | Packed rows must not cross file alignment boundaries unexpectedly.  The first implementation enforces `cols % (group_size * 2) == 0` to keep alignment simple. |
| Hostile-input validation | Loader must validate: `cols % 2 == 0`, `cols % group_size == 0`, `group_size` is a power-of-two in `[16, 128]`, payload size matches formula exactly.  Unknown group_size values must be rejected, not silently ignored. |
| Versioning compatibility | Adding dtype 3 with strict unknown-value rejection means no `.att1` version bump is required at M74; existing v1/v2 loaders already fail cleanly on unknown dtype.  If a quantization-metadata tensor-header field is added, a version bump may be required; M74 will decide. |

`.att1` format change decisions are deferred to M74.  This milestone (M73) does
not touch the format.

### Converter implications

| Concern | Plan |
|---------|------|
| f32 source → q4 artifact | `--weight-format q4` flag in `convert_llama_to_att1.py`; same tensor selection as q8 (2-D projections and output, not embeddings or norms) |
| Determinism | Same f32 input → same q4 output; no stochastic rounding.  SHA256 of artifact must be stable across runs. |
| Group size configurability | `--q4-group-size 32` (default) or `64`; validated against tensor column counts before conversion begins |
| Validation against f32/q8 | `compare_att1_to_source.py` will dequantize q4 → f32 and report `max_abs_error`; expected tolerance TBD (likely `< 0.4` for deterministic tiny fixtures at group_size=32) |
| BF16 source | Already supported via existing `_coerce_bf16`; q4 converter uses the same coerced f32 values as input |

q4 conversion output is deferred to M77.  M73 specifies the strategy only.

### Runtime/backend implications

| Concern | Plan |
|---------|------|
| CPU q4 matmul | First implementation target (M76): dequantize-then-multiply, analogous to the q8 prototype path; correctness over throughput |
| CPU q4 optimised matmul | Future: nibble-level SIMD unpack and fused multiply; deferred |
| CUDA q4 matmul | Prototype after CPU path is validated (M79): dequantize q4 row → float32 buffer, then run through existing CUDA f32 matmul |
| q4 cluster inference | Blocked until single-tile q4 inference is validated (M78+) |
| Activation dtype | Activations stay float32.  q4 is weights-only, same as q8. |
| KV cache | Remains float32.  KV quantization is a separate future concern. |

### Test plan

| Test | Method | Pass condition |
|------|--------|---------------|
| Tiny hand-checkable packing | 4 values, 1 group; verify packed byte and dequantized output manually | Exact byte match; dequantized error < `scale/2` |
| Zero row | All-zero input row | `scale = 1.0`, all packed bytes `0x00` |
| Saturation | Input value `abs > 7 * scale` | Clamps to `±7`; no UB |
| f32 vs q4 tolerance | Tiny 2-layer fixture at group_size=32 | `max_abs_error < 0.4` (to be confirmed at M77) |
| q4 model inspect | `att1-inspect` on q4 artifact | Reports `dtype_name=q4`, `quant=grouped-q4-g32`, `q4_groups=N` |
| q4 model load/reject | Bad group_size; cols not divisible; truncated payload | Loader returns error, no UB |

### Milestone split

| Milestone | Goal |
|-----------|------|
| M74 | q4 format and schema: `ATT1_MODEL_DTYPE_Q4=3` enum, nbytes formula + group_size validation in loader, `ATT1_ERR_UNSUPPORTED` from validate_decoder, `att1-inspect` q4 reporting, `test_quant_q4` (9 checks) |
| M75 | CPU q4 packing/unpacking primitives: `att1_q4_group_scale()`, `att1_q4_pack_group()`, `att1_q4_unpack_group()`, `att1_q4_quantize_group()`, `att1_q4_dequantize_group()`; `test_quant_q4_pack` (9 checks) |
| M76 | CPU q4 matmul prototype: `att1_q4_matrix` struct, `att1_quantize_q4_per_group()`, `att1_matmul_q4xf32()` (dequantize-then-multiply, activations stay float32); `test_matmul_q4` (8 checks) |
| M77 | q4 `.att1` fixture: `--weight-format q4` converter output, dtype-3 loader support, `att1-inspect` q4 reporting, checked-in tiny q4 model |
| M78 | CPU q4 single-tile inference: `--backend cpu-q4` in `att1-bench`, single-tile decode validated against cpu-f32 |
| M79 | CUDA q4 matmul planning/prototype: dequantize-then-multiply in CUDA, tests against CPU q4 reference |

---

## q4 format and schema (M74)

This section documents the q4 wire format decisions that were implemented in M74.
No q4 runtime kernels or q4 inference are included; this is schema and loader only.

### dtype wire value

`ATT1_MODEL_DTYPE_Q4 = 3` is the third entry in the `att1_model_dtype` enum.
Values 1 (f32) and 2 (q8) are unchanged.

### group_size encoding in flags

The group size for a q4 tensor is encoded in bits `[7:0]` of the existing
`flags` field of the tensor descriptor (the field was previously always zero):

| `flags & 0xFF` | Meaning |
|---------------|---------|
| `0` | Use default group size: 32 |
| `16`–`128` (power of two) | Explicit group size |
| Any other value | Invalid — loader returns error |

Bits `[31:8]` of `flags` must be zero for q4 tensors; any non-zero upper bits
are rejected by the loader.

Constants defined in `include/att1_quant.h`:

```c
#define ATT1_Q4_GROUP_SIZE_DEFAULT 32u
#define ATT1_Q4_GROUP_SIZE_MIN     16u
#define ATT1_Q4_GROUP_SIZE_MAX     128u
#define ATT1_Q4_FLAGS_GROUP_MASK   0xFFu
```

### Payload layout

For a q4 tensor with shape `[rows, cols]`:

```
uint8  packed[rows * cols / 2]              -- low-nibble-first int4 pairs
float32 scales[rows * (cols / group_size)]  -- one f32 scale per group
```

Total bytes = `rows*cols/2 + rows*(cols/group_size)*4`.

Values are signed symmetric int4 in `[-7, 7]` (value `-8` is excluded).

### `.att1` version decision

No format version bump is required for q4 tensors.  The `flags` field existed
in the v1 tensor descriptor and was always zero before M74; using bits `[7:0]`
to encode group size is backward-compatible because v1 loaders that predate M74
will reject dtype=3 tensors before reading `flags`.

### Hostile-input validation rules (loader)

The following checks are performed by `att1_tensor_nbytes_expected()` for q4 tensors:

1. `tensor->ndims == 2` — q4 tensors must be 2-D.
2. `(tensor->flags & ~ATT1_Q4_FLAGS_GROUP_MASK) == 0` — upper 24 bits of `flags` must be zero.
3. Resolved `group_size` must be a power of two in `[ATT1_Q4_GROUP_SIZE_MIN, ATT1_Q4_GROUP_SIZE_MAX]`.
4. `tensor->shape[1] % 2 == 0` — `cols` must be even (nibble packing).
5. `tensor->shape[1] % group_size == 0` — `cols` must be divisible by `group_size`.
6. All intermediate multiplications use overflow-safe `att1_u64_mul()`; on overflow the loader returns error.
7. The loader cross-checks computed nbytes against the stored nbytes field; mismatch returns error.

### Inference rejection

`att1_model_view_validate_decoder()` scans all tensors and returns
`ATT1_ERR_UNSUPPORTED` if any tensor has `dtype == ATT1_MODEL_DTYPE_Q4`.
There is no silent fallback to q8 or f32.

### `att1-inspect` q4 output

For each q4 tensor, `att1-inspect` prints:

```
dtype_name=q4 quant=grouped-q4-g<G> q4_groups=<N> q4_packed_bytes=<P> q4_scale_bytes=<S>
```

where `G` is the resolved group size, `N = rows*(cols/G)`, `P = rows*cols/2`,
and `S = N*4`.

### Test coverage (`test_quant_q4`)

Nine test cases in `tests/test_quant_q4.c`:

| # | Name | What it checks |
|---|------|---------------|
| 1 | `test_q4_load_valid` | rows=4, cols=64, group_size=32 loads successfully |
| 2 | `test_q4_valid_group_sizes` | group_size ∈ {0,16,32,64,128} with cols=128 all load |
| 3 | `test_q4_bad_group_size` | group_size ∈ {1,3,7,15,48,100,129,200,255} all rejected |
| 4 | `test_q4_bad_cols_alignment` | cols=48, group_size=32 (48 % 32 ≠ 0) rejected |
| 5 | `test_q4_bad_nbytes` | correct shape, wrong nbytes in file rejected |
| 6 | `test_q4_flags_reserved_bits` | flags with bit 8 set rejected |
| 7 | `test_q4_inference_rejected` | load succeeds, `att1_model_view_validate_decoder()` returns `ATT1_ERR_UNSUPPORTED` |
| 8 | `test_q4_inspect_output` | `att1-inspect` exits 0, output contains `dtype_name=q4`, `quant=grouped-q4-g32`, correct counts |
| 9 | `test_existing_dtypes_unaffected` | existing f32 and q8 fixtures still load correctly |

---

## q4 packing and unpacking primitives (M75)

Five per-group helper functions added to `src/quant.c` and declared in
`include/att1_quant.h`. No q4 matmul or q4 inference is included.

### API

| Function | Purpose |
|----------|---------|
| `att1_q4_group_scale(src, group_size, out_scale)` | Compute per-group float32 scale: `max(|src[i]|) / 7.0`, or `1.0` for a zero row |
| `att1_q4_pack_group(src_int4, group_size, dst_packed)` | Pack `group_size` int4 values (clamped to `[-7,7]`) into `group_size/2` bytes |
| `att1_q4_unpack_group(src_packed, group_size, dst_int4)` | Unpack `group_size/2` bytes back to `group_size` int4 values (sign-extended) |
| `att1_q4_quantize_group(src, group_size, dst_packed, out_scale)` | Compute scale, round-and-clamp to int4, pack; rejects non-finite input |
| `att1_q4_dequantize_group(src_packed, group_size, scale, dst)` | Unpack and multiply by scale to recover float32 values |

All functions return `0` on success, `-1` on invalid arguments (null pointers,
invalid group size, non-finite input).

### Valid group sizes

Same constraint as M74: powers of two in `[ATT1_Q4_GROUP_SIZE_MIN,
ATT1_Q4_GROUP_SIZE_MAX]` (16 to 128 inclusive).  `group_size=0` is **not**
valid as a function argument (the default-32 shorthand is a wire-format concept
only; callers must resolve the group size before calling these functions).

### Nibble packing convention

Matches the M74 wire format:
- `packed[i/2]` low nibble  = element `i`   (even index)
- `packed[i/2]` high nibble = element `i+1` (odd index)

Signed int4 values use two's-complement bit pattern.  The value `-8` is
excluded; the range is `[-7, 7]`.

### Saturation

Values outside `[-7, 7]` passed to `att1_q4_pack_group()` or produced by
rounding in `att1_q4_quantize_group()` are clamped to `[-7, 7]` before packing.

### Round-trip tolerance

For a deterministic 32-element vector covering `[-1, 1]`, the max absolute
reconstruction error is at most one quantization step (`scale * 1`).  With
`group_size=32` and `max_abs=1.0`, `scale = 1/7 ≈ 0.143`; expected
`max_abs_error < 0.143`.

### Test coverage (`test_quant_q4_pack`)

Nine test cases in `tests/test_quant_q4_pack.c`:

| # | Name | What it checks |
|---|------|---------------|
| 1 | `test_nibble_order` | Exact byte encoding for `{-3, 5}` pair; round-trip via unpack |
| 2 | `test_pack_unpack_all_values` | All 15 distinct int4 values in `[-7, 7]` survive pack/unpack |
| 3 | `test_saturation` | `±127` input clamps to `±7` after pack/unpack |
| 4 | `test_zero_row` | Zero vector → scale=1.0, all-zero output |
| 5 | `test_round_trip_tolerance` | 32-element evenly-spaced vector; `max_err ≤ scale` |
| 6 | `test_invalid_group_size` | `{0,1,3,15,48,256}` all rejected by every function |
| 7 | `test_null_args` | Null pointers rejected by every function |
| 8 | `test_nonfinite_input` | `inf`/`nan` in `src` rejected by `group_scale` and `quantize_group` |
| 9 | `test_all_valid_group_sizes` | `{16,32,64,128}` all quantize/dequantize successfully |

---

## q4 matmul prototype (M76)

`att1_q4_matrix` struct and four associated functions added to `src/quant.c`
and declared in `include/att1_quant.h`.  No q4 inference path; activations
stay float32 throughout.

### `att1_q4_matrix` struct

| Field | Type | Description |
|-------|------|-------------|
| `rows` | `size_t` | Number of weight matrix rows |
| `cols` | `size_t` | Number of weight matrix columns |
| `group_size` | `uint32_t` | Per-row group size used during quantization |
| `packed` | `uint8_t *` | `rows * cols / 2` bytes of packed nibbles |
| `scales` | `float *` | `rows * (cols / group_size)` float32 scale values |

### API

| Function | Purpose |
|----------|---------|
| `att1_q4_matrix_alloc(matrix, rows, cols, group_size)` | Allocate and zero `packed` and `scales` buffers; validates all args including `cols` even, `cols % group_size == 0`, valid group_size |
| `att1_q4_matrix_free(matrix)` | Free both buffers, zero all fields |
| `att1_quantize_q4_per_group(matrix, weights, rows, cols, group_size)` | Allocate matrix then quantize row by row, group by group using `att1_q4_quantize_group()` |
| `att1_matmul_q4xf32(dst, lhs, lhs_rows, lhs_cols, weights)` | Matrix multiply: for each group, unpack int4 values, multiply by group scale and float32 activation, accumulate into `dst` |

All functions return `0` on success, `-1` on invalid arguments.
`att1_q4_matrix_free()` is null-safe and always returns cleanly.

### Algorithm

`att1_matmul_q4xf32` uses a dequantize-then-multiply inner loop:
for each `(lhs_row, weight_row, group)` triple it calls
`att1_q4_unpack_group()` into a temporary `int8_t` buffer of at most
`ATT1_Q4_GROUP_SIZE_MAX` elements, then accumulates
`lhs[...] * (int4_val * scale)` into the output element.

Output layout: `dst[lhs_row * weights->rows + weight_row]`.
Requires `lhs_cols == weights->cols`.

### Tolerance

`Q4_TOLERANCE = 0.35f`.  Derived from one signed-int4 quantization step
(`scale/7`) per element accumulated over a full group (32 elements default,
scale ≤ 0.8/7 ≈ 0.114), rounded up conservatively to accommodate two groups
per row in the medium test.

### Test coverage (`test_matmul_q4`)

Eight test cases in `tests/test_matmul_q4.c`:

| # | Name | What it checks |
|---|------|---------------|
| 1 | `test_alloc_free` | Alloc/free lifecycle; field values after alloc and after free |
| 2 | `test_alloc_invalid` | Null matrix, zero rows/cols, odd cols, cols not divisible by group_size, invalid group_size all rejected |
| 3 | `test_tiny_hand_computed` | 2×16 all-0.5 and alternating-±1 weight matrix; lhs=all-1; q4 output within Q4_TOLERANCE of f32 reference |
| 4 | `test_zero_matrix` | All-zero weights → all-zero output |
| 5 | `test_medium_vs_f32` | 8×64 sin-based weights, 2×64 cos-based lhs, group_size=32; max_abs_error ≤ Q4_TOLERANCE vs f32 reference |
| 6 | `test_q4_vs_q8` | 4×32 sin-based weights quantized both ways; q4 vs q8 output within Q4_TOLERANCE |
| 7 | `test_dimension_mismatch` | `lhs_cols ≠ weights->cols` and zero `lhs_rows` both rejected |
| 8 | `test_null_args` | Null dst/lhs/weights in matmul; null src/matrix in quantize all rejected |
