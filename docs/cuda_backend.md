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

### Not yet implemented

`matmul_q8xf32`, `rmsnorm_f32`, `softmax_f32`, `rope_f32`, and
`ffn_swiglu_f32` still return failure.  Full transformer inference via CUDA
is not attempted until all operator kernels are validated.

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

## CLI

`att1-bench` accepts:

```sh
./build/att1-bench --model models/dummy/model.att1 --prompt hello --tokens 8 --mode single --backend cpu-f32
./build/att1-bench --model models/dummy/model.att1 --prompt hello --tokens 8 --mode single --backend cpu-q8
./build/att1-bench --model models/dummy/model.att1 --prompt hello --tokens 8 --mode single --backend cuda
```

`cpu-f32` remains the default. `cpu-q8` can run the current f32 inference path
because it provides f32 ops plus q8 matmul support. `cuda` reports unsupported
or unavailable unless a future milestone adds CUDA kernels.

`att1-q8-bench` accepts `--backend cpu-q8|cuda`. The CUDA option is a skeleton
path and is not expected to run q8 matmul until CUDA kernels are implemented.
