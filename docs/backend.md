# ATT-1 backend abstraction

Milestone 11 separates operator dispatch from inference. The default inference
path still uses CPU float32 and should remain behavior-compatible with the
pre-backend implementation.

## Context and ops

`att1_backend` is a small context that points at an ops table:

```c
struct att1_backend {
    const att1_backend_ops *ops;
    void *user_data;
};
```

The ops table currently includes:

- `alloc` and `free`
- `sync`
- `matmul_f32`
- `matmul_q8xf32`
- `rmsnorm_f32`
- `softmax_f32`
- `rope_f32`
- `ffn_swiglu_f32`

Every op receives the backend context as its first argument. CPU backends ignore
`user_data` today; future CUDA or device-backed contexts can use it for streams,
handles, workspaces, or device memory state.

## Lifecycle

Use one of the lifecycle helpers:

```c
att1_backend *backend = NULL;
att1_backend_default_create(&backend);
att1_backend_destroy(backend);
```

`att1_backend_default_create` returns the CPU f32 backend. `att1_backend_destroy`
accepts `NULL`.

## CPU f32 backend

The CPU f32 backend forwards all float32 ops to the existing scalar C kernels.
Single-tile and cluster inference create this backend by default and dispatch
transformer-block, attention, final norm, and final logits projection through
the ops table.

The public `att1_attention_forward_f32` and
`att1_transformer_block_forward_f32` functions remain available. They are now
wrappers around backend-aware variants using the default backend.

## CPU q8 backend

The CPU q8 backend is a matmul-capable backend for Milestone 10 q8 weights. It
provides:

- normal CPU f32 ops for non-quantized operators
- `matmul_q8xf32` for float32 activations multiplied by per-row q8 weights

This is not full quantized model inference. The model format is unchanged and
inference still loads and executes the dummy model through f32 weights by
default.

`build/att1-q8-bench` uses the CPU q8 backend to compare f32 and q8 matmul
timing/error on a deterministic tiny fixture.

## Non-goals

Milestone 11 does not add CUDA, q4 execution, device memory management, or
backend selection in model files. Those are future extensions layered behind
the same ops table.
