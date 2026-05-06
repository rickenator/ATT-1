# ATT-1 int8 quantization

Milestone 10 added a standalone int8 weight path for kernel correctness and
measurement. Milestone 24 wires CPU q8 into single-tile inference without
changing the model file format or adding q4.

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

## CPU q8 single-tile inference

Milestone 24 adds `--backend cpu-q8` support to the existing single-tile
inference path. The `.att1` model continues to store float32 tensors. When a
single-tile inference context selects the `cpu-q8` backend, it builds owned
runtime q8 copies of projection and output weights using the same
`weights_q8[out, in]` representation described above.

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

This is not full CUDA quantized model inference. CUDA q8 inference and CUDA q4
are not implemented.

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

CUDA q8 cluster inference is not implemented. q4 is not implemented.

## Benchmark

`build/att1-q8-bench` runs a deterministic small f32 matmul and q8xf32 matmul
over the same data and prints timing and error counters:

```sh
./build/att1-q8-bench
./build/att1-q8-bench --iterations 10000
```

This benchmark is for kernel measurement only. `--backend cuda` exercises the
CUDA q8xf32 operator when CUDA is compiled in and available at runtime;
otherwise it reports unsupported cleanly.

## Ownership

`att1_q8_matrix_alloc` allocates owned `values` and `scales` buffers.
`att1_q8_matrix_free` releases both buffers, clears the struct, and accepts
`NULL`.
