# ATT-1 CUDA backend skeleton

Milestone 12 adds an opt-in CUDA backend path without changing default CPU
behavior. Normal builds do not require CUDA headers, libraries, or a GPU.

## Build modes

Default CPU-only build:

```sh
make
```

Opt-in CUDA build:

```sh
make CUDA=1
```

`CUDA=1` defines `ATT1_ENABLE_CUDA`, adds `$(CUDA_HOME)/include`, adds
`$(CUDA_HOME)/lib64`, and links `-lcudart`. `CUDA_HOME` defaults to
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

The CUDA backend does not implement transformer inference or CUDA operator
kernels yet. Operator entries such as `matmul_f32`, `matmul_q8xf32`,
`rmsnorm_f32`, `softmax_f32`, `rope_f32`, and `ffn_swiglu_f32` currently return
failure.

This keeps the backend usable for lifecycle, allocation, copy, and integration
testing while preventing accidental CUDA inference.

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
