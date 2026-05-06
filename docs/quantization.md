# ATT-1 int8 quantization

Milestone 10 adds a standalone int8 weight path for kernel correctness and
measurement. It does not change the model file format, does not add q4, and
does not route full model inference through quantized weights yet.

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

## Benchmark

`build/att1-q8-bench` runs a deterministic small f32 matmul and q8xf32 matmul
over the same data and prints timing and error counters:

```sh
./build/att1-q8-bench
./build/att1-q8-bench --iterations 10000
```

This benchmark is for kernel measurement only. It is not full quantized model
inference.

## Ownership

`att1_q8_matrix_alloc` allocates owned `values` and `scales` buffers.
`att1_q8_matrix_free` releases both buffers, clears the struct, and accepts
`NULL`.
