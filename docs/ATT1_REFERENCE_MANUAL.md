# ATT-1 Reference Manual (M146)

**Scope:** Software, artifact, and runtime reference for the ATT-1 tiled tensor
simulator. This manual covers the `.att1` artifact format, C11 runtime API,
backends, inference modes, CLI tools, model conversion flow, planning and
control-plane pipeline, testing policy, and artifact hygiene.

**Not in scope:** AIMU intrinsics, AIMU command set, MMIO register map, DMA
protocol, fabric packet format, and execution phases are documented separately
in the [AIMU Intrinsics and Operations Reference Manual](AIMU_INTRINSICS_OPERATIONS_REFERENCE.md) (M147).

---

## Table of Contents

1. [ATT-1 Overview](#1-att-1-overview)
2. [Artifact Format](#2-artifact-format)
3. [Tensor Naming and Model Assumptions](#3-tensor-naming-and-model-assumptions)
4. [Dtypes and Quantization](#4-dtypes-and-quantization)
5. [Runtime Architecture](#5-runtime-architecture)
6. [Backend Model](#6-backend-model)
7. [Inference Modes](#7-inference-modes)
8. [CLI Reference](#8-cli-reference)
9. [Model Conversion Flow](#9-model-conversion-flow)
10. [Placement and Reporting Pipeline](#10-placement-and-reporting-pipeline)
11. [Testing and Validation](#11-testing-and-validation)
12. [Artifact and Repository Hygiene](#12-artifact-and-repository-hygiene)
13. [Error and Status Conventions](#13-error-and-status-conventions)
14. [Non-Goals](#14-non-goals)
15. [Roadmap References](#15-roadmap-references)

---

## 1. ATT-1 Overview

ATT-1 is a C11 software simulator for clustered LLM inference on a tiled
tensor architecture. It models a multi-tile AIMU (AI Memory Unit) system in
which each tile owns a portion of the model weights, a local KV cache, and a
fabric interface. Phase 1 is entirely in-process software simulation; no
physical PCIe, FPGA, or ASIC hardware is required or implied.

### Component Relationships

```
.att1 artifact
    │
    ├─► att1_model_load()          model loader (C11)
    │       │
    │       ├─► att1_infer_t       single-tile inference context
    │       │       └─► att1_backend  dispatch table (cpu-f32 / cpu-q8 / cpu-q4 / cuda*)
    │       │
    │       └─► att1_cluster_infer_t  multi-tile cluster inference context
    │               ├─► att1_shard_plan  per-tile layer assignment
    │               ├─► att1_fabric_t    in-process packet fabric
    │               └─► att1_backend     per-tile backend
    │
    ├─► att1-inspect               artifact inspection tool (CLI)
    ├─► att1-bench                 inference benchmark tool (CLI)
    └─► att1-size                  capacity planning + placement report (CLI)

compiler/ (Python tools)
    ├─► placement report pipeline  (M98–M102: validator, advisory, scenarios)
    ├─► command-plan pipeline      (M109, M129: placement→commands, exec→commands)
    ├─► fabric-route pipeline      (M115–M119: mapper, validator, BW sim, replay)
    └─► AIMU host/MMIO emulator    (M121–M122: userspace emulator, MMIO replay)
```

*CUDA paths require `make CUDA=1` and an NVIDIA GPU. CPU-only is the default.*

---

## 2. Artifact Format

### 2.1 File Layout

An `.att1` file is a little-endian binary with three sequential sections:

| Section | Version 1 size | Version 2 size | Notes |
|---------|---------------|---------------|-------|
| Header | 80 bytes | 96 bytes | Magic, version, section offsets and sizes |
| Config | 36 bytes | 36 bytes | Model hyperparameters |
| Tensor descriptors | `tensor_count × 128 bytes` | same | Name, dtype, shape, offset, size, shard, flags |
| Tensor payloads | variable | variable | Packed tensor data (f32 / q8 / q4) |
| Shard metadata section | optional | optional | Per-tile tensor ownership records |
| Tokenizer metadata section | absent in V1 | optional in V2 | Tokenizer type, vocab, special token IDs |

### 2.2 Header Fields

| Field | Type | Offset | Notes |
|-------|------|--------|-------|
| `magic` | `uint8[8]` | 0 | `"ATT1MODL"` |
| `version` | `uint32` | 8 | 1 (V1) or 2 (V2) |
| `header_size` | `uint32` | 12 | Must equal 80 (V1) or 96 (V2) |
| `config_size` | `uint32` | 16 | Must equal 36 |
| `config_offset` | `uint64` | 20 | Byte offset of config from file start |
| `tensor_count` | `uint64` | 28 | Number of tensor descriptors |
| `tensor_desc_offset` | `uint64` | 36 | Byte offset of first tensor descriptor |
| `data_offset` | `uint64` | 44 | Byte offset of tensor payload region |
| `data_size` | `uint64` | 52 | Total byte size of tensor payload region |
| `shard_offset` | `uint64` | 60 | Byte offset of shard metadata (0 = absent) |
| `shard_size` | `uint64` | 68 | Byte size of shard metadata (0 = absent) |
| `tok_meta_offset` | `uint64` | 80 | V2 only: byte offset of tokenizer metadata |
| `tok_meta_size` | `uint64` | 88 | V2 only: byte size of tokenizer metadata |

Constants: `ATT1_MODEL_HEADER_SIZE = 80`, `ATT1_MODEL_HEADER_SIZE_V2 = 96`,
`ATT1_MODEL_CONFIG_SIZE = 36`, `ATT1_MODEL_TENSOR_DESC_SIZE = 128`.

### 2.3 Config Fields (`att1_model_config`)

| Field | Type | Description |
|-------|------|-------------|
| `vocab_size` | `uint32` | Token vocabulary size |
| `n_layers` | `uint32` | Transformer decoder layer count |
| `n_heads` | `uint32` | Attention head count |
| `d_model` | `uint32` | Hidden state dimension |
| `d_ff` | `uint32` | FFN intermediate dimension |
| `max_seq_len` | `uint32` | Maximum sequence length |
| `rope_dim` | `uint32` | RoPE rotary embedding dimension per head |
| `n_tiles` | `uint32` | Target AIMU tile count (advisory; not enforced by loader) |
| `shard_count` | `uint32` | Number of shard metadata records (0 = absent) |

### 2.4 Tensor Descriptor Fields (`att1_model_tensor`)

| Field | Type | Size | Notes |
|-------|------|------|-------|
| `name` | `char[64]` | 64 B | Null-terminated tensor name |
| `dtype` | `uint32` | 4 B | `1`=f32, `2`=q8, `3`=q4 |
| `ndims` | `uint32` | 4 B | Number of shape dimensions (1–4) |
| `shape[4]` | `uint64[4]` | 32 B | Dimension sizes; unused dims are 0 |
| `offset` | `uint64` | 8 B | Byte offset of payload within data region |
| `nbytes` | `uint64` | 8 B | Byte size of payload |
| `shard_id` | `uint32` | 4 B | Owning shard/tile index |
| `flags` | `uint32` | 4 B | q4: low byte encodes group size |
| *(padding)* | — | 4 B | Reserved, must be zero |

### 2.5 Quantization Payloads

**q8 tensors** store one `int8` per element followed by one `float32` scale
per group (group size is stored in the shard metadata or defaults to 32).

**q4 tensors** store elements as nibble pairs followed by per-group `float32`
scales. Full layout for shape `[rows, cols]` with group size `G`:

```
uint8  packed[rows * cols / 2]             -- low nibble = even element
float32 scales[rows * (cols / G)]          -- one scale per group
```

Values are signed two's-complement int4 in the range `[-7, 7]`. `-8` is
excluded to maintain a symmetric range. The group size `G` must be a power of
two in `[16, 128]`; the default is 32 (`ATT1_Q4_GROUP_SIZE_DEFAULT`).

### 2.6 Metadata Sections

**Shard metadata** records (120 bytes each, optional) describe per-tile tensor
ownership, memory layout, and slice information. See `docs/shard_metadata.md`.

**Tokenizer metadata** (V2 only, optional) records the tokenizer type, vocab
size, special token IDs, normalization settings, and asset hash. See
`docs/tokenizer_metadata.md` and `include/att1_tok_meta.h`.

### 2.7 Hostile-Input Validation Policy

The loader (`att1_model_load`) treats every file as hostile input:

- Validates magic bytes exactly.
- Validates version against `ATT1_MODEL_VERSION` and `ATT1_MODEL_VERSION_2`.
- Validates header size constants.
- Rejects `config_size` = 0 or out-of-range.
- Rejects `config_offset` or `data_offset` that would overflow a `uint64` range check.
- Rejects `tensor_count` values that would cause allocation overflow.
- Rejects tensor descriptors with unsupported dtype values.
- Rejects `shard_size` present without matching `shard_offset`.
- All-zeros files and truncated files are rejected.

Passing a hostile or malformed file to `att1_model_load` must return a
non-`ATT1_OK` status. It must never silently succeed or corrupt memory.

### 2.8 Versioning Policy

- V1 (`ATT1_MODEL_VERSION = 1`): 80-byte header; no tokenizer metadata section.
- V2 (`ATT1_MODEL_VERSION_2 = 2`): 96-byte header; optional tokenizer metadata.
- Loaders reject unknown version values.
- Field additions require a version bump.
- The format is considered stable for V1 and V2. Changes to `att1_model_config`
  fields would require a V3 definition.

---

## 3. Tensor Naming and Model Assumptions

ATT-1 targets LLaMA-style autoregressive decoder-only transformers. Other
model families (encoder-decoder, MoE, encoder-only) are not supported.

### 3.1 Tensor Name Convention

| Tensor name | Shape | Notes |
|-------------|-------|-------|
| `tok_embeddings.weight` | `[vocab_size, d_model]` | Input embedding matrix |
| `layers.N.attention_norm.weight` | `[d_model]` | Pre-attention RMSNorm |
| `layers.N.attention.wq.weight` | `[d_model, d_model]` | Query projection |
| `layers.N.attention.wk.weight` | `[d_model, d_model]` | Key projection |
| `layers.N.attention.wv.weight` | `[d_model, d_model]` | Value projection |
| `layers.N.attention.wo.weight` | `[d_model, d_model]` | Output projection |
| `layers.N.ffn_norm.weight` | `[d_model]` | Pre-FFN RMSNorm |
| `layers.N.ffn.w_gate.weight` | `[d_model, d_ff]` | FFN gate projection (SwiGLU) |
| `layers.N.ffn.w_up.weight` | `[d_model, d_ff]` | FFN up projection (SwiGLU) |
| `layers.N.ffn.w_down.weight` | `[d_ff, d_model]` | FFN down projection |
| `output_norm.weight` | `[d_model]` | Final RMSNorm before lm_head |
| `output.weight` | `[d_model, vocab_size]` | LM head output projection |

`N` is a zero-based layer index string (`0`, `1`, …, `n_layers-1`).

### 3.2 Unsupported Variants

- Grouped Query Attention (GQA): K/V shape differences are not handled in the
  runtime. The converter imports GQA tensors but the runtime expects
  `wk`/`wv` with the full `d_model` width.
- Mixture of Experts (MoE): No router, gate, or expert selection logic.
- Encoder-only or encoder-decoder: Causal attention mask is always applied.
- Sliding window attention: Not implemented.
- BF16 and F16 at runtime: Only f32 activations are used at runtime. BF16 and
  F16 source tensors from external model files are coerced to f32 by the
  converter; see §9.

---

## 4. Dtypes and Quantization

### 4.1 Dtype Summary

| Dtype | Enum value | Wire format | Runtime activations |
|-------|-----------|-------------|---------------------|
| `f32` | `ATT1_MODEL_DTYPE_F32 = 1` | IEEE 754 `float` | f32 |
| `q8` | `ATT1_MODEL_DTYPE_Q8 = 2` | `int8` values + `float32` scales | f32 (dequantized per matmul) |
| `q4` | `ATT1_MODEL_DTYPE_Q4 = 3` | Nibble-packed + `float32` scales | f32 (dequantized per matmul) |

The runtime always computes activations in f32. Quantized weights are
dequantized on access during matmul operations.

### 4.2 q8 Format

- Per-row or per-group `int8` values with one `float32` scale per group.
- Scale = `max(|x_i|) / 127.0` per group.
- Tolerance: logits `max_abs_error < 0.15` versus f32 reference (typical).

### 4.3 q4 Format

See §2.5 for wire layout. Key parameters:

| Constant | Value | Meaning |
|----------|-------|---------|
| `ATT1_Q4_GROUP_SIZE_DEFAULT` | 32 | Default group size |
| `ATT1_Q4_GROUP_SIZE_MIN` | 16 | Minimum valid group size |
| `ATT1_Q4_GROUP_SIZE_MAX` | 128 | Maximum valid group size |
| Int4 range | `[-7, 7]` | `-8` excluded for symmetry |
| Scale policy | `max(|x_i|) / 7.0` | Per group; `1.0` if all-zero |

- Tolerance: logits `max_abs_error < 0.35` versus f32 reference (typical).
- Token output may diverge from f32 reference for real models; divergence is
  expected and within documented tolerance.

### 4.4 Projected/Future Dtypes

- F16 and BF16 are defined in `att1_tensor_dtype` (`ATT1_DTYPE_F16`,
  `ATT1_DTYPE_BF16`) as tensor-type identifiers for the converter only.
  Runtime inference with F16/BF16 weights is not implemented.
- Additional quantization schemes (e.g., per-channel q8, fp8) are not
  specified or planned in the current version.

### 4.5 Unsupported Dtype Behavior

Loading a tensor with an unrecognized dtype value causes `att1_model_load` to
return `ATT1_ERR_BAD_FORMAT`. There is no silent fallback to another dtype.

---

## 5. Runtime Architecture

### 5.1 C11 Runtime

The runtime is written in C11 (`-std=c11 -Wall -Wextra -Wpedantic -Werror`).
POSIX threading (`-pthread`) is used for tile threads and fabric simulation.
CUDA is opt-in via `make CUDA=1`; the default build is CUDA-free.

### 5.2 Model Loader (`att1_model_load`)

```c
att1_status_t att1_model_load(const char *path, att1_model *model);
void          att1_model_free(att1_model *model);
const att1_model_tensor *att1_model_find_tensor(const att1_model *model,
                                                const char *name);
```

- Reads the entire file into memory; the model owns the buffer until `att1_model_free`.
- All tensor `data` pointers borrow into `model->file_data`; they are valid
  only while the `att1_model` is alive.
- `att1_model` must not be shallow-copied; it owns `tensors[]` and `file_data`.

### 5.3 Tensor Math (`att1_tensor`)

```c
int  att1_tensor_alloc_f32(att1_tensor *tensor, uint32_t rank, const size_t *shape);
void att1_tensor_free(att1_tensor *tensor);
```

- `att1_tensor` is an owned f32 buffer with shape metadata.
- `att1_tensor` must not be shallow-copied; it owns `data`.
- Returns `-1` on failure; `0` on success (uses `int`, not `att1_status_t`).

### 5.4 KV Cache and KV-MMU

The `att1_kv_cache_t` provides per-layer paged key-value storage. The
`att1_kv_mmu_t` implements a simulated hardware-shaped page table for KV
memory management. Both are session-local and not thread-safe.

```c
att1_status_t att1_kv_cache_init(att1_kv_cache *cache, /* ... */);
att1_status_t att1_kv_mmu_create(const att1_kv_mmu_config *config,
                                 att1_kv_mmu **out_mmu);
void att1_kv_mmu_destroy(att1_kv_mmu *mmu);
```

- `att1_kv_cache_init` returns `att1_status_t` as of M151.
- `att1_kv_mmu` is an opaque handle as of M151. Callers create it through
  `att1_kv_mmu_create` and release it with `att1_kv_mmu_destroy`.
- `att1_kv_mmu_create` rejects invalid capacity shapes, including
  `max_positions > max_pages * page_tokens`.

See `docs/kv_mmu.md`.

### 5.5 Tokenizer and Pretokenized Input

Three tokenizer modes are supported:

| Mode | Flag | Description |
|------|------|-------------|
| `byte` | `--tokenizer byte` (default) | Raw byte values 0–255 used as token IDs |
| `metadata` | `--tokenizer metadata` | Reads tokenizer section from V2 `.att1`; hard errors if absent |
| `external` | `--tokenizer external` | Accepts pretokenized IDs via `--input-token-ids` or `--tokens-file` |

See `docs/tokenizer_metadata.md`.

### 5.6 Sampler

The sampler applies a temperature-scaled softmax and supports greedy argmax
selection. Nucleus (top-p) and top-k sampling are defined in the API but the
current implementation uses greedy decode for reproducibility.

### 5.7 Inference Context (`att1_infer_t`)

```c
att1_status_t att1_infer_create(const att1_model *model, att1_infer_t **out_infer);
att1_status_t att1_infer_create_q4(const att1_model *model, att1_infer_t **out_infer);
void          att1_infer_destroy(att1_infer_t *infer);

att1_status_t att1_infer_decode_token(att1_infer_t *infer, uint32_t token_id,
                                      uint32_t *out_token);
att1_status_t att1_infer_generate(att1_infer_t *infer, const unsigned char *prompt,
                                  size_t prompt_bytes, size_t generated_token_count,
                                  uint32_t *out_tokens, size_t out_token_capacity,
                                  size_t *out_token_count);
const float  *att1_infer_logits(const att1_infer_t *infer, size_t *out_count);
att1_status_t att1_infer_set_backend(att1_infer_t *infer, att1_backend *backend);
att1_status_t att1_infer_set_trace(att1_infer_t *infer, att1_trace_t *trace);
```

- `att1_infer_set_backend`: takes **ownership** of backend.
- `att1_infer_set_trace`: **borrows** trace; caller must keep trace alive while
  attached and until `att1_infer_destroy` or a subsequent `set_trace(NULL)`.
- `att1_infer_logits`: returns a **borrow** into the internal logit buffer;
  valid only until the next decode call or `att1_infer_destroy`.
- `att1_infer_t` must not be shallow-copied.

### 5.8 Cluster Inference Context (`att1_cluster_infer_t`)

```c
att1_status_t att1_cluster_infer_create(const att1_model *model,
    const att1_cluster_infer_config *config, att1_cluster_infer_t **out_infer);
att1_status_t att1_cluster_infer_create_q4(const att1_model *model,
    const att1_cluster_infer_config *config, att1_cluster_infer_t **out_infer);
void att1_cluster_infer_destroy(att1_cluster_infer_t *infer);
att1_status_t att1_cluster_infer_decode_token(att1_cluster_infer_t *infer,
    uint32_t token_id, uint32_t *out_token);
```

The cluster context creates `tile_count` simulated tile threads, assigns
layers to tiles by the shard plan, and exchanges activations through
`att1_fabric_t`. See §7 for shard plan modes.

### 5.9 Trace and Counter Model

`att1_trace_t` records per-decode counters: fabric packets sent/received,
payload bytes, KV appends/reads, logit bytes, local op counts, and
prefill/decode split timestamps. The benchmark tool prints these as
`key=value` lines. See `docs/tracing.md`.

---

## 6. Backend Model

### 6.1 Available Backends

| Backend name | Creation function | CUDA required | Status |
|-------------|-------------------|---------------|--------|
| `cpu-f32` | `att1_backend_cpu_f32_create()` | No | Implemented; correctness reference |
| `cpu-q8` | `att1_backend_cpu_q8_create()` | No | Implemented |
| `cpu-q4` | `att1_backend_cpu_q4_create()` | No | Implemented |
| `cuda` (f32) | `att1_backend_cuda_create()` | Yes | Implemented; manual signoff required |
| `cuda-q8` | `att1_backend_cuda_q8_create()` | Yes | Implemented; manual signoff required |
| `cuda-q4` | `att1_backend_cuda_q4_create()` | Yes | Implemented; manual signoff required |

`att1_backend_default_create()` selects `cpu-f32` unless overridden.

### 6.2 Backend Dispatch Table (`att1_backend_ops`)

| Slot | Signature | Implemented by |
|------|-----------|----------------|
| `matmul_f32` | `(dst, lhs, rhs, rows, cols, inner)` | All backends |
| `matmul_q8xf32` | `(dst, lhs, lhs_rows, lhs_cols, weights)` | cpu-q8, cuda-q8 |
| `matmul_q4xf32` | `(dst, lhs, lhs_rows, lhs_cols, weights)` | cpu-q4, cuda-q4 |
| `rmsnorm_f32` | `(dst, src, weight, count, epsilon)` | All backends |
| `softmax_f32` | `(values, count)` | All backends |
| `rope_f32` | `(values, count, position, theta)` | All backends |
| `ffn_swiglu_f32` | `(dst, gate, value, count)` | All backends |
| `alloc` / `free` / `sync` | Memory management | All backends |

Backends that do not support a slot set it to `NULL`. Calling a `NULL` slot
is a programming error and will crash; the inference path validates backend
capability before use.

### 6.3 No-Silent-Fallback Rule

If a requested backend cannot be created, `att1_backend_*_create` returns an
error. The runtime does **not** silently fall back to another backend. The
caller must handle the error explicitly.

### 6.4 CUDA Opt-In Policy

CUDA support is compiled in only when `CUDA=1` is passed to `make`:

```sh
make CUDA=1        # build with CUDA
make               # default: CPU-only, CUDA-free
```

CPU-only builds compile all CUDA backend source files but all CUDA-dependent
code paths are guarded by `#ifdef CUDA_ENABLED`. CUDA test bodies are skipped
on hosts without a GPU. The CPU f32 backend is always the correctness
reference.

---

## 7. Inference Modes

### 7.1 Single-Tile Mode

Single-tile inference runs all transformer layers on one context. It uses
`att1_infer_t` and a single backend.

```sh
att1-bench --model MODEL.att1 --tokens N --mode single [--backend cpu-f32|cpu-q8|cpu-q4|cuda|cuda-q8] [--prompt TEXT]
```

### 7.2 Cluster Mode

Multi-tile cluster inference distributes layers across `tile_count` simulated
tiles, each running on its own thread. Activations are exchanged through the
in-process fabric. It uses `att1_cluster_infer_t`.

```sh
att1-bench --model MODEL.att1 --tokens N --mode cluster --tiles N [--backend cpu-f32|cpu-q8|cpu-q4|cuda|cuda-q8] [--prompt TEXT]
```

### 7.3 Shard Plan Modes

The `att1_cluster_infer_config.shard_plan_mode` field controls how layers are
assigned to tiles:

| Mode | Enum value | Description |
|------|-----------|-------------|
| `runtime` | `ATT1_SHARD_PLAN_RUNTIME = 0` | Distribute layers evenly across tiles (default) |
| `metadata` | `ATT1_SHARD_PLAN_METADATA` | Use tile assignments from `.att1` shard metadata section |

`ATT1_SHARD_PLAN_METADATA` is not supported in q4 cluster inference; use
`ATT1_SHARD_PLAN_RUNTIME`.

### 7.4 Prefill vs Decode Reporting

When a prompt is provided, the runtime records trace counters at the
prefill/decode boundary and emits separate fields:

| Field | Description |
|-------|-------------|
| `prompt_tokens` | Number of prompt tokens consumed |
| `decode_tokens` | Number of new tokens generated |
| `prefill_time_us_total` | Microseconds spent in prefill |
| `decode_time_us_total` | Microseconds spent in decode |
| `prefill_kv_appends` / `decode_kv_appends` | KV append counts per phase |
| `prefill_fabric_packets` / `decode_fabric_packets` | Cluster mode only |

### 7.5 Benchmark Output Fields

`att1-bench` prints `key=value` lines to stdout. Key fields:

| Field | Notes |
|-------|-------|
| `mode=single|cluster` | Inference mode |
| `backend=<name>` | Backend selected |
| `tokens=N` | Tokens generated |
| `tokenizer=byte|metadata|external` | Tokenizer mode |
| `shard_plan=runtime|metadata` | Cluster only |
| `fabric_packets_sent=N` | Cluster only |
| `kv_appends=N` | Total KV append operations |
| `decode_time_us_total=N` | Decode phase microseconds |

See `docs/tracing.md` for the full field list.

---

## 8. CLI Reference

### 8.1 `att1-inspect`

**Purpose:** Print human-readable metadata from a `.att1` file.

```sh
att1-inspect <model.att1>
```

**Output:** Config fields, tensor list (name, dtype, shape, offset, nbytes,
shard, flags), shard metadata summary and records (if present), tokenizer
metadata (if present), validation violations (if any).

**Failure:** Exits non-zero if the file cannot be loaded.

---

### 8.2 `att1-bench`

**Purpose:** Run inference on a `.att1` model and report trace counters.

```sh
att1-bench --model PATH --tokens N --mode single|cluster
           [--backend cpu-f32|cpu-q8|cpu-q4|cuda|cuda-q8|cuda-q4]
           [--prompt TEXT]
           [--tiles N]
           [--tokenizer byte|metadata|external]
           [--input-token-ids ID,ID,...]
           [--tokens-file PATH]
           [--shard-plan runtime|metadata]
```

| Flag | Notes |
|------|-------|
| `--model PATH` | Path to `.att1` file |
| `--tokens N` | Number of decode tokens to generate |
| `--mode single\|cluster` | Inference mode |
| `--backend NAME` | Backend selection; default `cpu-f32` |
| `--prompt TEXT` | Byte-mode prompt (required for byte/metadata tokenizer) |
| `--tiles N` | Tile count for cluster mode |
| `--tokenizer MODE` | Tokenizer mode (`byte` default) |
| `--input-token-ids IDs` | Comma-separated token IDs for external mode |
| `--tokens-file PATH` | File containing comma-separated token IDs |
| `--shard-plan MODE` | Shard plan mode for cluster (`runtime` default) |

**Failure:** Exits non-zero on load error, invalid arguments, or inference
failure. q4 models print an error and exit non-zero if the model does not
contain q4 tensors.

---

### 8.3 `att1-size`

**Purpose:** Capacity planning and tensor placement report generation.

```sh
# Preset mode
att1-size --preset tiny-dummy|gpt-oss-120b-shape [--tiles N] [--context N] [--dtype f32|f16|q8|q4]

# Config file mode
att1-size --config PATH [--tiles N] [--context N] [--dtype f32|f16|q8|q4] [--json]

# Manual mode
att1-size --layers N --d-model N --heads N --d-ff N --vocab-size N \
          [--context N] [--tiles N] [--dtype f32|f16|q8|q4] [--json]

# Capacity/bandwidth flags (any mode)
att1-size ... [--tile-memory-mib N] [--tile-memory-gib N]
              [--sessions N] [--target-tokens-per-sec N] [--fabric-gib-sec N]
              [--placement-report-json PATH]
```

| Flag | Notes |
|------|-------|
| `--preset NAME` | Built-in preset; `tiny-dummy` = 4-d dummy model |
| `--config PATH` | JSON config file (LLaMA `config.json` format) |
| `--dtype NAME` | Dtype for size estimates (`f32`, `f16`, `q8`, `q4`) |
| `--tiles N` | Target tile count for per-tile estimates |
| `--context N` | Sequence length for KV size estimates |
| `--sessions N` | Concurrent sessions for KV pressure |
| `--tile-memory-mib N` | Per-tile memory budget (MiB) |
| `--fabric-gib-sec N` | Fabric bandwidth for PASS/WARN/FAIL check |
| `--placement-report-json PATH` | Write M98-schema JSON placement report |
| `--json` | Machine-readable JSON output |

**Output:** Size estimates (bytes and GiB), per-tile capacity and bandwidth
status (PASS/WARN/FAIL/UNKNOWN), optional JSON placement report. All estimates
are architectural projections; preset shapes are for planning purposes only.

---

### 8.4 `att1-aimu-mmio-emulator`

**Purpose:** Userspace MMIO emulator smoke flow (M121). Runs a complete
probe → enumerate → setup → command sequence against a mmap-backed BAR0
register file.

```sh
att1-aimu-mmio-emulator [--bar0-file PATH] [--tiles N] [--tile-memory-mib N]
                         [--kv-memory-mib N] [--run-smoke] [--report-json PATH] [--verbose]
```

---

### 8.5 `att1-aimu-replay`

**Purpose:** In-process M109 command-plan JSON replay against the M112 host
harness (M113).

```sh
att1-aimu-replay --plan PATH [--strict] [--report-json PATH]
```

---

### 8.6 `att1-aimu-mmio-replay`

**Purpose:** M109 command-plan JSON replay through the M121 userspace MMIO
emulator (M122).

```sh
att1-aimu-mmio-replay --plan PATH [--bar0-file PATH] [--tiles N]
                       [--tile-memory-mib N] [--kv-memory-mib N]
                       [--strict] [--report-json PATH] [--verbose]
```

---

### 8.7 Python Compiler Tools

All Python tools live in `compiler/`. They require Python 3; no third-party
packages are required for the planning pipeline tools.

| Tool | Purpose | Key flags |
|------|---------|-----------|
| `convert_llama_to_att1.py` | Convert LLaMA safetensors to `.att1` | `--safetensors`, `--weight-format f32\|q8\|q4`, `--tiles N`, `--shard-meta`, `--report-json` |
| `scan_safetensors.py` | Inspect safetensors tensor metadata | `--model-dir` |
| `scan_tokenizer.py` | Import tokenizer asset report | `--model-config`, `--report`, `--report-json` |
| `tokenize_hf.py` | Pretokenize text via Hugging Face tokenizer | `--model-dir`, `--text` |
| `compare_att1_to_source.py` | Compare `.att1` tensors against source | `--att1`, `--model-dir`, `--backend`, `--report-json` |
| `check_llama_compat.py` | Check LLaMA model directory compatibility | `--model-dir` |
| `validate_tensor_placement_report.py` | Validate M98 placement report JSON | `--report`, `--strict`, `--report-json` |
| `propose_tensor_placement.py` | Advisory tensor placement proposals | `--report`, `--report-json` |
| `propose_tensor_scenarios.py` | Multi-scenario capacity comparison | `--report`, `--tile-memory-gib`, `--report-json` |
| `map_placement_to_commands.py` | Placement report → AIMU command plan | `--report`, `--plan-json`, `--strict` |
| `plan_tensor_execution.py` | Command/route → execution plan | `--placement-report`, `--command-plan`, `--plan-json` |
| `validate_tensor_execution_plan.py` | Validate M125 execution plan JSON | `--plan`, `--strict`, `--report-json` |
| `map_execution_plan_to_commands.py` | Execution plan → command plan | `--plan`, `--plan-json`, `--strict` |
| `map_commands_to_fabric_routes.py` | Command plan → fabric route report | `--plan`, `--report-json` |
| `validate_fabric_routes.py` | Validate M115 fabric route report | `--report`, `--strict`, `--report-json` |
| `simulate_fabric_bandwidth.py` | Fabric bandwidth/latency simulator | `--route-report`, `--target-tokens-per-sec`, `--fabric-gib-sec`, `--report-json` |
| `replay_fabric_routes.py` | Fabric route replay simulator | `--route-report`, `--report-json` |
| `replay_aimu_command_plan.py` | Python wrapper for `att1-aimu-replay` | `--plan`, `--strict`, `--report-json` |
| `replay_command_plan_via_mmio.py` | Python wrapper for `att1-aimu-mmio-replay` | `--plan`, `--tiles N`, `--report-json` |
| `run_aimu_planning_pipeline.py` | 8-stage end-to-end planning pipeline (M119) | `--placement-report`, `--workdir`, `--report-json`, `--strict` |
| `run_execution_replay_pipeline.py` | 6-stage execution/replay pipeline (M132) | `--execution-plan`, `--workdir`, `--report-json`, `--strict` |
| `check_schema_compat.py` | Schema version compatibility checker | `--schema TYPE`, `--input`, `--strict`, `--report-json` |
| `check_hostile_inputs.py` | Hostile-input validation (superset of schema compat) | `--schema TYPE`, `--input`, `--strict`, `--report-json` |
| `check_golden_regressions.py` | Golden baseline regression checker | `--check` (default), `--update-golden` |
| `run_full_regression.py` | 8-step local regression runner | `--no-build`, `--cuda`, `--report-json` |
| `fuzz_json_schemas.py` | JSON schema fuzz/smoke harness (M143) | (no flags; runs all 40 cases) |

All tools exit `0` on success, `1` on validation/logic failure, `2` on
parse/argument error. Most accept `--strict` to promote warnings to errors.

---

### 8.8 Demo and Regression Scripts

| Script | Purpose |
|--------|---------|
| `tools/demo_tiny_att1.sh` | 14-step end-to-end tiny fixture demo (M144) |
| `compiler/run_full_regression.py` | 8-step local regression runner (M136) |
| `.github/workflows/ci.yml` | GitHub Actions CPU-only CI definition (M137) |

---

## 9. Model Conversion Flow

### 9.1 Supported Source Format

The converter (`compiler/convert_llama_to_att1.py`) imports from:

- **safetensors** (HuggingFace format): `--safetensors PATH`
- **LLaMA config.json** directory: `--model-dir PATH` (auto-discovers safetensors)

BF16 and F16 source tensors are coerced to f32 during import. Other exotic
source dtypes are not supported.

### 9.2 Conversion Modes

| Flag | Weight dtype in output `.att1` |
|------|-------------------------------|
| (default) | f32 |
| `--weight-format q8` | q8 |
| `--weight-format q4` | q4 |

### 9.3 Tiny Fixtures

The following tiny fixtures are committed to the repository and require no
external model downloads:

| Model | Path | Dtype | Notes |
|-------|------|-------|-------|
| Dummy | `models/dummy/model.att1` | f32 | 4-d model; the primary test fixture |
| Tokenizer metadata | `models/tok_meta/model.att1` | f32 | V2 with tokenizer section |
| Shard metadata | `models/shard_meta/model.att1` | f32 | With 2-tile shard metadata |
| Converted stub meta | `models/converted_stub_meta/model.att1` | f32 | 2-tile converter output |
| Real tiny f32 | `models/real_tiny_f32/model.att1` | f32 | Converted from tiny safetensors fixture |
| Real tiny q8 | `models/real_tiny_q8/model.att1` | q8 | |
| Real tiny q4 | `models/real_tiny_q4/model.att1` | q4 | |
| q4 tiny | `models/q4_tiny/model.att1` | q4 | |
| m61, m63 variants | `models/m61_*/`, `models/m63_*/` | f32/q8 | Larger tiny fixtures |

### 9.4 Public Model Artifact Policy

Public model weights and any `.att1` files generated from them must **not** be
committed to the repository. Place them in local directories outside the
repository root. See §12 and `docs/EXTERNAL_REVIEW_PACKAGE.md`.

### 9.5 Source Comparison

`compiler/compare_att1_to_source.py` compares tensor values between an `.att1`
file and its source safetensors, accounting for dtype coercion. Reports
`max_abs_error` per tensor. f32 models typically achieve `max_abs_error = 0`;
q8 typically `< 0.6`; q4 typically `< 1.5` per tensor element.

---

## 10. Placement and Reporting Pipeline

The planning pipeline is a control-plane simulation and reporting system only.
It does **not** execute inference, load real model weights, or access physical
hardware. All tools are advisory unless explicitly stated.

### 10.1 Schema Versions

| Schema | Version | Definition |
|--------|---------|------------|
| Placement report | 1 | `docs/tensor_placement_report.md` |
| Command plan | 1 | `docs/aimu_pcie_command_requirements.md` |
| Fabric route report | 1 | `docs/aimu_fabric_routing.md` §11 |
| Execution plan | 1 | `docs/tensor_execution_plan.md` |
| Pipeline report | 132 | `compiler/run_execution_replay_pipeline.py` |

### 10.2 Pipeline Stages

| Stage | Tool | Input | Output |
|-------|------|-------|--------|
| Capacity planning | `att1-size` | Model config | Placement report JSON |
| Placement validation | `validate_tensor_placement_report.py` | Placement report | Validation JSON |
| Placement advisory | `propose_tensor_placement.py` | Placement report | Advisory JSON |
| Scenario comparison | `propose_tensor_scenarios.py` | Placement report | Scenario table JSON |
| Placement → commands | `map_placement_to_commands.py` | Placement report | Command plan JSON |
| Command replay (in-process) | `att1-aimu-replay` / `replay_aimu_command_plan.py` | Command plan | Replay report JSON |
| Command replay (MMIO emulator) | `att1-aimu-mmio-replay` / `replay_command_plan_via_mmio.py` | Command plan | MMIO replay report JSON |
| Commands → fabric routes | `map_commands_to_fabric_routes.py` | Command plan | Fabric route report JSON |
| Fabric route validation | `validate_fabric_routes.py` | Fabric route report | Validation JSON |
| Fabric bandwidth simulation | `simulate_fabric_bandwidth.py` | Fabric route report | BW simulation JSON |
| Fabric route replay | `replay_fabric_routes.py` | Fabric route report | Replay report JSON |
| Execution plan | `plan_tensor_execution.py` | Placement + command + route | Execution plan JSON |
| Execution plan validation | `validate_tensor_execution_plan.py` | Execution plan | Validation JSON |
| Execution plan → commands | `map_execution_plan_to_commands.py` | Execution plan | Command plan JSON |
| Integrated pipeline (M119) | `run_aimu_planning_pipeline.py` | Placement report | Integrated JSON report |
| Execution/replay pipeline (M132) | `run_execution_replay_pipeline.py` | Execution plan | Integrated JSON report |

### 10.3 Report vs Execution

All pipeline stages listed above are **report-only** (planning and
simulation). They do not:
- Execute transformer inference.
- Load real model weights.
- Access physical PCIe registers or MMIO.
- Implement a kernel driver.
- Transfer data over a real fabric.

---

## 11. Testing and Validation

### 11.1 Test Commands

| Command | Description | Expected result |
|---------|-------------|-----------------|
| `make clean && make && make test` | Build and run all C tests | 781 PASS 0 FAIL |
| `make regression` | Python validation layers and M152 fuzz coverage | All steps PASS |
| `make clean && make test-asan` | ASAN-instrumented build and test | 781 PASS 0 FAIL |
| `make clean && make test-ubsan` | UBSAN-instrumented build and test | 781 PASS 0 FAIL |
| `make fuzz-smoke` | Binary loader + JSON schema fuzz smoke + coverage guard | 22 loader + 45 JSON PASS; 67 total |
| `./tools/demo_tiny_att1.sh` | End-to-end tiny fixture demo | 14/14 PASS |

### 11.2 Regression Layers

`make regression` invokes `compiler/run_full_regression.py`, which runs:

1. `make clean` and `make` (build validation)
2. `make test` (C test suite)
3. `compiler/check_golden_regressions.py` — 12 golden baseline checks
4. `compiler/test_schema_compat.py` — 31 schema compatibility tests
5. `compiler/test_hostile_inputs.py` — 42 hostile-input tests
6. `compiler/run_execution_replay_pipeline.py` (M132 pipeline smoke)
7. Python cache artifact check
8. `compiler/check_docs.py` (M149 docs lint/link check)
9. `make fuzz-smoke` (M152 deterministic fuzz smoke + coverage guard)

### 11.3 Sanitizer Targets

| Target | Flags | Build dir |
|--------|-------|-----------|
| `make test-asan` | `-fsanitize=address` | `build-asan/` |
| `make test-ubsan` | `-fsanitize=undefined` | `build-ubsan/` |
| `make clean-asan` / `make clean-ubsan` | — | Removes sanitizer build dirs |

Normal `make clean` does **not** remove sanitizer directories.

### 11.4 Fuzz Smoke Targets

| Target | Tool | Cases |
|--------|------|-------|
| `make fuzz-loader` | `tests/fuzz_model_loader.c` | 22 (2 valid + 20 hostile) |
| `make fuzz-json` | `compiler/fuzz_json_schemas.py` | 45 (32 hostile + 2 valid + 11 inline mutations) |
| `make fuzz-coverage` | `compiler/report_fuzz_coverage.py` | Static corpus guard |
| `make fuzz-smoke` | Loader + JSON + coverage | 67 total |
| `make fuzz-libfuzzer` | Optional `tests/fuzz_model_loader_guided.c` | Builds libFuzzer binary when `clang` exists |
| `make fuzz-afl` | Optional `tests/fuzz_model_loader_guided.c` | Builds AFL-compatible binary when `afl-clang-fast` exists |

### 11.5 CI Policy

The GitHub Actions CPU-only CI (`.github/workflows/ci.yml`) runs on every
push and pull request:
- Installs `build-essential python3 make`.
- Runs `make clean && make && make test`.
- Runs `python3 compiler/run_full_regression.py --no-build`.
- Checks for tracked Python cache artifacts.

CI does **not** validate CUDA kernels, CUDA runtime paths, or public model
inference.

### 11.6 CUDA Manual Signoff

CUDA validation requires a manual run on a CUDA-capable host:

```sh
make clean && make CUDA=1 && make test CUDA=1
python3 compiler/run_full_regression.py --cuda --report-json cuda_signoff_$(date +%Y%m%d).json
```

See `docs/CUDA_VALIDATION_PLAN.md` for the full signoff policy.

---

## 12. Artifact and Repository Hygiene

### 12.1 Allowed Tracked Files

- Tiny fixture `.att1` models in `models/` (see §9.3).
- Planning and schema fixtures in `compiler/fixtures/` (including
  `golden/`, `hostile/`, `schema_compat/`).
- Source code, headers, Makefile, tests, documentation.
- `.github/` workflow definitions.

### 12.2 Forbidden Tracked Files

| File type | Policy |
|-----------|--------|
| Public model weights (safetensors, bin, gguf, etc.) | Never commit |
| Generated large `.att1` files from real models | Never commit |
| Local model directories | Never commit |
| `__pycache__/`, `*.pyc`, `*.pyo` | Never commit |
| Local absolute paths in text files | Do not commit; relative paths only |
| API keys, tokens, secrets | Never commit |
| Patent claim drafts or private invention disclosures | Never commit |

Check before every commit:

```sh
git ls-files | grep -E '(__pycache__|\.pyc$|\.pyo$)' || echo "OK"
```

### 12.3 External Reviewer Package

See `docs/EXTERNAL_REVIEW_PACKAGE.md` for the full reviewer checklist,
pre-sharing validation commands, and `git archive` tarball instructions.

---

## 13. Error and Status Conventions

### 13.1 `att1_status_t` (Runtime API)

| Code | Value | Meaning |
|------|-------|---------|
| `ATT1_OK` | 0 | Success |
| `ATT1_ERR_INVALID_ARG` | -1 | Null pointer, out-of-range value, or invalid argument |
| `ATT1_ERR_OOM` | -2 | Memory allocation failure |
| `ATT1_ERR_IO` | -3 | File I/O error |
| `ATT1_ERR_BAD_FORMAT` | -4 | Malformed or version-mismatch binary input |
| `ATT1_ERR_NOT_FOUND` | -5 | Tensor, page, or resource not found |
| `ATT1_ERR_SHAPE` | -6 | Shape mismatch or invalid dimension |
| `ATT1_ERR_QUEUE_FULL` | -7 | Fabric or command queue capacity exceeded |
| `ATT1_ERR_STATE` | -8 | Invalid state transition or lifecycle violation |
| `ATT1_ERR_UNSUPPORTED` | -9 | Feature or path not supported |
| `ATT1_ERR_QUEUE_EMPTY` | -10 | Queue polled when empty |
| `ATT1_ERR_TIMEOUT` | -11 | Operation timed out |
| `ATT1_ERR_ALREADY_STARTED` | -12 | Context already started; double-start prevented |

### 13.2 Two Error Systems

ATT-1 has two distinct error systems by design:

| System | Header | Values | Used by |
|--------|--------|--------|---------|
| `att1_status_t` | `include/att1_status.h` | Negative integers | Runtime C API |
| `att1_aimu_result` | `include/att1_aimu_cmdq.h` | Hex codes | AIMU command-plane |

These systems are intentionally separate. Do not mix them. AIMU result codes
are documented in the AIMU Intrinsics and Operations Reference Manual (M147).

### 13.3 Remaining `int`-Returning Allocation Functions

M151 migrated `att1_kv_cache_init` to `att1_status_t` and replaced the
stack-allocated `att1_kv_mmu_init` API with opaque `att1_kv_mmu_create` /
`att1_kv_mmu_destroy`. The remaining early allocation functions that still use
plain `int` are:

| Function | Returns |
|----------|---------|
| `att1_tensor_alloc_f32` | `int`: `0` success, `-1` failure |
| `att1_q8_matrix_alloc` | `int`: `0` success, `-1` failure |
| `att1_q4_matrix_alloc` | `int`: `0` success, `-1` failure |

This remaining inconsistency was identified in the M141 API ownership review
and is left for a later, narrower refactor.

### 13.4 No-Silent-Fallback Rule

No failure path in the runtime silently falls back to a different backend,
dtype, or operation. All failures must produce a non-success return value or
an explicit error message on `stderr`. This rule applies to:
- Backend creation failures.
- Dtype unsupported errors.
- Model load validation failures.
- Schema validation failures in Python tools.
- AIMU command dispatch returning `UNSUPPORTED_OP`.

---

## 14. Non-Goals

The following are explicitly outside the scope of ATT-1:

| Non-goal | Notes |
|----------|-------|
| Production ASIC design or tape-out | Not planned in this project |
| Real PCIe endpoint or BAR0 MMIO | Userspace emulator only |
| Linux kernel driver | Not implemented |
| FPGA RTL or synthesis | Feasibility notes only (`docs/fpga_feasibility.md`) |
| Mobile, Android, Vulkan, or OpenCL targets | Not planned |
| Encoder-only or encoder-decoder models | Causal decoder only |
| Grouped Query Attention at runtime | Not implemented |
| Patent claim language | Must not appear anywhere in the repository |
| Public model weights in Git | External only; never committed |
| Automatic CUDA fallback | No silent fallback |
| Full production inference serving | Reference simulator only |
| Real-time fabric or hardware interconnect | In-process simulation only |

---

## 15. Roadmap References

| Milestone | Title | Status |
|-----------|-------|--------|
| M147 | AIMU Intrinsics and Operations Reference Manual | Complete — [docs/AIMU_INTRINSICS_OPERATIONS_REFERENCE.md](AIMU_INTRINSICS_OPERATIONS_REFERENCE.md) |
| M148 | Reference manual consistency pass: cross-check README, INDEX, both reference manuals, RELEASE_READINESS, EXTERNAL_REVIEW_PACKAGE, testing, CUDA validation, and OPERATION_LOG for link consistency, terminology, and status accuracy | Complete — see `docs/OPERATION_LOG.md` |
| M149 | Documentation lint and link checker | Complete — see `compiler/check_docs.py` |
| M150 | Release candidate checkpoint | Complete — see [docs/RELEASE_CANDIDATE_M150.md](RELEASE_CANDIDATE_M150.md) |
| M151 | API opacity and refactor plan | Complete — KV cache/MMU status APIs, opaque KV-MMU handle, status alias cleanup |
| M152 | Deeper fuzzing and coverage expansion | Complete — 67 deterministic fuzz cases, coverage guard, optional libFuzzer/AFL harnesses |
| M153+ | Release dry-run, external review response log, or hardware prototype work | TBD |

See `docs/OPERATION_LOG.md` for the full milestone history and
`docs/RELEASE_READINESS.md` for the current readiness gate status.

---

*End of ATT-1 Reference Manual (M146)*
