# ATT-1 Real Tiny Model Import Plan

Milestone 45.  This document defines the full implementation plan for importing
a real tiny LLaMA-style model into ATT-1.  No converter code changes are made
here.  Safetensors parsing is explicitly deferred.

---

## Scope

### First supported source model type

| Property | Value |
|----------|-------|
| Architecture | Decoder-only LLaMA-style transformer |
| Attention | Multi-head (no GQA, no MQA) |
| FFN | SwiGLU (`w_gate`, `w_up`, `w_down`) |
| MoE | Out of scope |
| Position encoding | Absolute RoPE (theta default `10000.0`, configurable) |
| Weight dtype | `bfloat16` or `float32` source; ATT-1 target is `f32` first, then `q8` |
| q4 weights | Deferred |
| Tied embeddings | Allowed; converter writes explicit copy of `output.weight` |
| Tokenizer | Deferred (M50) |

### Explicitly out of scope for M46–M49

- Grouped-query attention (GQA) or multi-query attention (MQA)
- Mixture-of-experts (MoE) layers
- q4 quantized output
- GPT-OSS 120B or any large-model real inference
- Tokenizer vocabulary import or BPE execution
- Any Python dependency inside `make test` or the C runtime

---

## Expected source files

| File | Required | Notes |
|------|----------|-------|
| `config.json` | Yes | Provides all shape constants; already parsed by converter |
| `model.safetensors` | Yes (M47) | Single-shard; multi-shard via `model.safetensors.index.json` deferred |
| `generation_config.json` | Optional | Ignored by the converter |
| `tokenizer.json` / `tokenizer.model` | No | Deferred to M50 |
| `special_tokens_map.json` | No | Deferred |

All weight loading is Python-only under `compiler/`.  The C runtime never
loads safetensors directly.

---

## Required tensor mappings

Source key names follow Hugging Face LLaMA conventions.  ATT-1 target names
follow the flat naming convention in `src/model_view.c`.

| Source key (HF LLaMA) | ATT-1 name | Shape in ATT-1 | Notes |
|----------------------|------------|----------------|-------|
| `model.embed_tokens.weight` | `tok_embeddings.weight` | `[vocab_size, d_model]` | copy as-is |
| `model.layers.L.input_layernorm.weight` | `layers.L.attention_norm.weight` | `[d_model]` | |
| `model.layers.L.self_attn.q_proj.weight` | `layers.L.attention.wq.weight` | `[d_model, d_model]` | transpose required |
| `model.layers.L.self_attn.k_proj.weight` | `layers.L.attention.wk.weight` | `[d_model, d_model]` | transpose required |
| `model.layers.L.self_attn.v_proj.weight` | `layers.L.attention.wv.weight` | `[d_model, d_model]` | transpose required |
| `model.layers.L.self_attn.o_proj.weight` | `layers.L.attention.wo.weight` | `[d_model, d_model]` | transpose required |
| `model.layers.L.post_attention_layernorm.weight` | `layers.L.ffn_norm.weight` | `[d_model]` | |
| `model.layers.L.mlp.gate_proj.weight` | `layers.L.ffn.w_gate.weight` | `[d_model, d_ff]` | transpose required |
| `model.layers.L.mlp.up_proj.weight` | `layers.L.ffn.w_up.weight` | `[d_model, d_ff]` | transpose required |
| `model.layers.L.mlp.down_proj.weight` | `layers.L.ffn.w_down.weight` | `[d_ff, d_model]` | transpose required |
| `model.norm.weight` | `output_norm.weight` | `[d_model]` | |
| `lm_head.weight` | `output.weight` | `[d_model, vocab_size]` | transpose required; if absent and tied, copy `tok_embeddings.weight` and transpose |

`L` is the zero-based layer index.  The converter iterates layers
`0 … n_layers-1` in order.

### Transpose rules

ATT-1 projection weights are stored `[out_dim, in_dim]` (row-major,
output-major).  HF LLaMA stores them `[out_dim, in_dim]` as well, but
PyTorch's convention for linear layers is that `weight` is the transposed
kernel — i.e. the stored array is already in `[out_features, in_features]`
order matching a right-multiply `y = x @ W.T`.

Concretely:

| Tensor | Source shape | ATT-1 shape | Action |
|--------|-------------|-------------|--------|
| `q_proj.weight` | `[d_model, d_model]` | `[d_model, d_model]` | transpose (swap axes) |
| `k_proj.weight` | `[d_model, d_model]` | `[d_model, d_model]` | transpose |
| `v_proj.weight` | `[d_model, d_model]` | `[d_model, d_model]` | transpose |
| `o_proj.weight` | `[d_model, d_model]` | `[d_model, d_model]` | transpose |
| `gate_proj.weight` | `[d_ff, d_model]` | `[d_model, d_ff]` | transpose |
| `up_proj.weight` | `[d_ff, d_model]` | `[d_model, d_ff]` | transpose |
| `down_proj.weight` | `[d_model, d_ff]` | `[d_ff, d_model]` | transpose |
| `lm_head.weight` | `[vocab_size, d_model]` | `[d_model, vocab_size]` | transpose |
| `embed_tokens.weight` | `[vocab_size, d_model]` | `[vocab_size, d_model]` | copy, no transpose |
| RMSNorm weights | `[d_model]` | `[d_model]` | copy, no transpose |

The converter must call `numpy.ascontiguousarray(w.T)` (or equivalent) for all
projection weights before writing.

---

## Shape conventions and validation

The following checks must pass before any tensor is written:

```text
vocab_size   > 0
n_layers     > 0
n_heads      > 0
d_model      > 0
d_ff         > 0
max_seq_len  > 0
rope_dim     > 0
d_model % n_heads == 0
head_dim = d_model // n_heads
rope_dim <= head_dim
```

Per-tensor checks (after transpose):

```text
tok_embeddings.weight      shape == (vocab_size, d_model)
layers.L.attention.wq      shape == (d_model, d_model)
layers.L.attention.wk      shape == (d_model, d_model)
layers.L.attention.wv      shape == (d_model, d_model)
layers.L.attention.wo      shape == (d_model, d_model)
layers.L.attention_norm    shape == (d_model,)
layers.L.ffn_norm          shape == (d_model,)
layers.L.ffn.w_gate        shape == (d_model, d_ff)
layers.L.ffn.w_up          shape == (d_model, d_ff)
layers.L.ffn.w_down        shape == (d_ff, d_model)
output_norm.weight         shape == (d_model,)
output.weight              shape == (d_model, vocab_size)
```

Any shape mismatch must be a hard error (`sys.exit(1)`) with a descriptive
message identifying the tensor name, expected shape, and actual shape.

---

## RoPE conventions

| Property | Value |
|----------|-------|
| `rope_dim` | Derived as `head_dim` unless `config.json` provides `partial_rotary_factor` — in that case `rope_dim = int(head_dim * partial_rotary_factor)`, rounded down to even |
| `rope_theta` | From `config.json` field `rope_theta` (aliases: `rope_scaling.rope_theta`); default `10000.0` |
| Position encoding | Applied per-head to the first `rope_dim` elements of each query/key vector |
| Variants deferred | YaRN, NTK-aware scaling, LongRoPE, `rope_scaling.type != null` other than `"linear"` with factor `1.0` |

If `config.json` contains a non-trivial `rope_scaling` dict, the converter
must emit a warning and proceed only if `type == "linear"` and `factor == 1.0`
(effectively no scaling), or exit with an error for unsupported variants.

---

## dtype conversion

### f32 output (M48)

1. Load source weights as `float32` using `compiler/load_safetensors.py`.
   BF16/F16 coercion remains deferred; M48 accepts only `F32` source tensors.
2. Transpose where required (see table above).
3. Write ATT-1 dtype code `1` (`ATT1_DTYPE_F32`) for every tensor.
4. Compute `nbytes = product(shape) * 4`.

### q8 output (M49)

q8 output is built from the f32 converter path, so the safetensors reader still
loads the source payloads as F32:

1. Run the f32 converter to get the intermediate numpy arrays.
2. For each eligible 2-D projection/output tensor except
   `tok_embeddings.weight`, transpose the ATT-1 f32 `[in,out]` matrix into the
   runtime q8 `[out,in]` layout and apply per-row quantization:
   ```text
   scale[row] = max(abs(row)) / 127   (or 1.0 if row is all-zero)
   q[row, col] = clamp(round(val / scale), -127, 127)
   ```
3. Store dtype-2 tensor payloads as all row-major int8 values followed by one
   float32 scale per row.  The tensor shape is the q8 runtime shape
   `[out_dim, in_dim]`, and `nbytes = rows * cols + rows * 4`.
4. Write ATT-1 dtype code `2` (`ATT1_DTYPE_Q8`) for quantized tensors.
5. RMSNorm weights and token embeddings remain `f32` (dtype `1`).

The `.att1` container remains version `1`; M49 only activates the previously
reserved dtype code `2` and extends hostile-input byte-count validation for q8
matrices.

### q4 output

Deferred indefinitely.

---

## Hostile-input validation requirements

The converter must fail fast and clearly on any of the following:

| Condition | Error message pattern |
|-----------|-----------------------|
| `config.json` missing | `error: config file not found: <path>` |
| `architecture` not in supported list | `error: unsupported architecture: <name>` |
| Missing required `config.json` field | `error: missing required field: <name>` |
| `d_model % n_heads != 0` | `error: d_model (<v>) is not divisible by n_heads (<v>)` |
| `rope_dim > head_dim` | `error: rope_dim (<v>) exceeds head_dim (<v>)` |
| Source tensor not found in safetensors | `error: missing tensor: <name>` |
| Source tensor wrong shape | `error: tensor <name>: expected shape <s>, got <s>` |
| Source tensor wrong dtype (not float16/bfloat16/float32) | `error: tensor <name>: unsupported source dtype <d>` |
| File offset out of bounds in safetensors | `error: safetensors: offset <o> exceeds file size <s>` |
| Unsupported `rope_scaling` type | `error: unsupported rope_scaling type: <type>` |

All errors write to `stderr` and `sys.exit(1)`.  The output `.att1` file must
not be written (or must be deleted) on any error.

---

## Validation ladder

| Stage | Model | What is tested | Milestone |
|-------|-------|---------------|-----------|
| 1 | Deterministic synthetic stub | Converter round-trip; loader acceptance; bench runs | M32 ✓ |
| 1a | Stub with shard metadata | Metadata plan; inspect; cluster bench | M42–M44 ✓ |
| 2 | Tiny real config + synthetic tensors | safetensors metadata scan; tensor layout check | M46–M47 |
| 3 | Tiny trained toy model (real weights) | f32 converted artifact loads and runs; last-token sanity | M48 |
| 4 | q8 converted tiny model | q8 path; dtype-2 loader; cross-backend last-token match | M49 |
| 5 | Small public LLaMA-like model | Tokenizer spot-check after M50 | post-M50 |

---

## Exact future milestone split

### M46 — safetensors metadata scanner (complete)

**Goal:** Parse the JSON metadata header of a `.safetensors` file and emit a
human-readable tensor inventory, without loading any tensor data.

**Implementation:**

- **`compiler/scan_safetensors.py`** — standalone module + CLI.
  Public API:
  - `ScanError(Exception)` — raised for all fatal parse errors.
  - `scan_safetensors(path) -> dict` — returns `file_size`, `header_len`,
    `data_offset`, `data_size`, `metadata`, `tensors` (list of dicts with
    `name`, `dtype`, `shape`, `data_offsets`, `nbytes`, `element_size`,
    `expected_nbytes`), and `errors` (non-fatal issues).
  - `llama_required_keys(n_layers) -> list` — full required key list.
  - `check_llama_tensors(scan, n_layers, allow_tied_lm_head=True) -> list`
    — returns list of missing required keys.
  - `format_scan_report(scan, show_tensors=True) -> str` — human-readable
    output with `key=value` header block followed by per-tensor lines.

  CLI flags: `path`, `--check-llama`, `--n-layers N`, `--config PATH`,
  `--json`, `--no-tensors`.
  Exit codes: 0 = ok, 1 = fatal scan error, 2 = missing LLaMA tensors.

  Fatal error conditions (exit 1):
  - File not found.
  - File < 8 bytes (cannot read header length prefix).
  - `header_len` extends beyond file size (truncated/corrupt).
  - JSON header is not valid UTF-8 or not valid JSON.
  - Any tensor's `data_offsets[1]` exceeds `data_size`.

  Non-fatal (logged in `errors`, exit 0 unless `--check-llama` fails):
  - Missing `dtype`, `shape`, or `data_offsets` fields in a tensor entry.
  - `data_offsets[0] > data_offsets[1]`.
  - `nbytes` inconsistent with `shape × element_size`.
  - Overlapping data regions between tensors.

- **`compiler/fixtures/make_tiny_safetensors.py`** — generates a minimal
  valid `.safetensors` fixture for testing:
  - vocab=16, d\_model=8, d\_ff=16, n\_heads=2, n\_layers=2
  - 21 tensors, all `F32`, ~6304 bytes data + ~2013 bytes JSON header
    = 8325 bytes total

- **`compiler/fixtures/tiny_llama_2l.safetensors`** — checked-in fixture
  (8325 bytes, 21 tensors, all required LLaMA keys present).

- **`tests/test_bench_smoke.c`** — `check_scanner()` added (Python-skippable):
  - Runs basic scan, asserts `tensor_count=21`, `scan_errors=0`, `scan: ok`,
    `model.embed_tokens.weight`, `lm_head.weight`, `F32`.
  - Runs `--check-llama --n-layers 2`, asserts `llama_check: ok`,
    `llama_missing: 0`.

- No changes to C source, Makefile, or `.att1` format.
- `make test` passes (39 tests).

### M47 — safetensors tensor reader (complete)

**Goal:** Load actual tensor payload bytes from a `.safetensors` file, decode
F32 values, and return a deterministic Python data structure — using Python
standard library only (no numpy, no external dependencies).

**Implementation:**

- **`compiler/load_safetensors.py`** — new module + CLI built on the M46
  scanner.  Public API:
  - `LoadError(Exception)` — raised for all load-level failures.
  - `TensorData` namedtuple — `name`, `dtype`, `shape`, `values`
    (tuple of float, row-major), `nbytes`.
  - `load_tensor(path, name, expected_dtype=None, expected_shape=None)`
    — validates dtype and shape, reads payload bytes from the file's data
    region, decodes as little-endian F32 via `struct.unpack`.  Raises
    `ScanError` (propagated from M46) or `LoadError`.
  - `load_all(path)` — loads all readable tensors; returns
    `(loaded, skipped, errors)`.
  - `format_tensor_report(td, max_values=8)` — one-tensor human output.
  - `format_summary(loaded, skipped, errors)` — multi-tensor summary.

  Supported dtype: F32 only (M47).  BF16/F16 coercion deferred to M48.

  CLI flags: `--tensor NAME`, `--expected-dtype DTYPE`,
  `--expected-shape DIM,...`, `--summary`, `--check-values`.
  Exit codes: 0 = success, 1 = scan/load error, 2 = NaN/Inf found.

  Error conditions:
  - Tensor name not found → `LoadError`.
  - `expected_dtype` mismatch → `LoadError`.
  - Unsupported dtype (BF16, F16, etc.) → `LoadError`.
  - `expected_shape` mismatch → `LoadError`.
  - Payload bytes shorter than declared → `LoadError` (defensive).
  - Truncated file (offsets exceed data region) → `ScanError` (M46).

- **`compiler/test_load_safetensors.py`** — self-contained Python test (10
  checks): embed/q\_proj/gate\_proj/norm/lm\_head loads; expected\_dtype
  mismatch; expected\_shape mismatch; BF16 unsupported dtype; truncated file;
  known non-zero value decode.  Outputs `self_test: ok` on success.

- **`tests/test_bench_smoke.c`** — `check_tensor_reader()` added
  (Python-skippable): loads embed (dtype/shape/elements), q\_proj (elements),
  gate\_proj (elements), norm.weight (elements), `--check-values` (21 tensors
  finite), `compiler/test_load_safetensors.py` self-test.

- No changes to C source (other than `test_bench_smoke.c`), Makefile, or
  `.att1` format.  `make test` passes (39 tests).

### M48 — real f32 converted tiny model (complete)

**Goal:** Produce a real (non-synthetic) `.att1` artifact from a tiny
LLaMA-style model's weights and confirm it loads and runs.

**Implementation:**

- `compiler/convert_llama_to_att1.py` keeps the synthetic stub path as the
  default.  Passing `--safetensors PATH` switches emission to real f32 tensor
  payloads loaded through the M47 safetensors reader.
- The checked-in fixture pair is:
  - `compiler/fixtures/tiny_llama_config.json`
  - `compiler/fixtures/tiny_llama_2l.safetensors`
- The fixture shape is intentionally tiny: `vocab_size=16`, `n_layers=2`,
  `n_heads=2`, `d_model=8`, `d_ff=16`.
- The checked-in converted artifact is
  `models/real_tiny_f32/model.att1` (21 f32 tensors, no shard metadata).
- The C decoder validation now accepts any nonzero `vocab_size` instead of
  only 256, while all tensor shape checks still bind to the model config.

**Converter command:**

```bash
python3 compiler/convert_llama_to_att1.py \
    --config compiler/fixtures/tiny_llama_config.json \
    --safetensors compiler/fixtures/tiny_llama_2l.safetensors \
    --out models/real_tiny_f32/model.att1
```

**Validation:**

- `att1-inspect` loads the converted model through the hostile-input C loader.
- `tests/test_converter_validation.c` verifies config fields and transposed
  ATT-1 matrix shapes such as `ffn.w_gate.weight [8,16]`,
  `ffn.w_down.weight [16,8]`, and `output.weight [8,16]`.
- `att1-bench --mode single --backend cpu-f32` runs on the converted model.
- `att1-bench --mode cluster --tiles 2 --backend cpu-f32` also runs and
  reports nonzero fabric packets.
- No tokenizer import is included.  Until tokenizer support lands, prompts for
  this 16-token fixture must be raw byte IDs in the range `0..15`.
- No q8/q4 conversion is included, and the `.att1` binary format is unchanged.

### M49 — q8 converted tiny model

**Goal:** Produce a q8 `.att1` artifact from the checked-in tiny safetensors
fixture and validate it through inspect plus CPU/CUDA q8 inference paths.

**Scope:**
- `compiler/convert_llama_to_att1.py --weight-format q8` emits per-row
  quantized projection/output weights as dtype `2`.
- The checked-in artifact is `models/real_tiny_q8/model.att1`.
- The loader validates dtype-2 tensor shapes and byte counts as
  `rows * cols + rows * sizeof(float)`.
- `att1-inspect` reports `dtype_name=q8`, `quant=per-row-q8`, `q8_values`,
  and `q8_scales` for quantized tensors.
- `tests/test_converter_validation.c` validates the q8 artifact without
  invoking Python during `make test`.
- CPU q8 single and cluster benchmarks run on the q8 artifact; cluster mode
  must report nonzero fabric packets.
- CUDA q8 single and cluster benchmarks run on CUDA builds when the CUDA
  runtime is available; CPU-only builds report unsupported cleanly.

**Manual command:**

```bash
python3 compiler/convert_llama_to_att1.py \
    --config compiler/fixtures/tiny_llama_config.json \
    --safetensors compiler/fixtures/tiny_llama_2l.safetensors \
    --weight-format q8 \
    --out models/real_tiny_q8/model.att1

./build/att1-inspect models/real_tiny_q8/model.att1

./build/att1-bench \
    --model models/real_tiny_q8/model.att1 \
    --prompt $'\x01' --tokens 2 \
    --mode single --backend cpu-q8

./build/att1-bench \
    --model models/real_tiny_q8/model.att1 \
    --prompt $'\x01' --tokens 2 \
    --mode cluster --tiles 2 --backend cpu-q8
```

For the tiny fixture, tests compare q8 converted CPU behavior against the f32
converted model running through the CPU q8 backend and assert matching
`last_token`.  This is a fixture-level regression check, not a general token
equivalence contract: q8 logits may move an argmax boundary on future models.
The quantization tolerance remains the documented q8 logit tolerance in
`docs/quantization.md`.

### M50 — tokenizer import plan

**Goal:** Document and prototype the path from `tokenizer.json` or
`tokenizer.model` (SentencePiece) to `att1_tokenizer_t`.

**Scope (documentation only in M50):**
- Required vocabulary fields; BPE merge rules; byte-fallback handling.
- ATT-1 tokenizer binary format extension (or new file alongside `.att1`).
- No implementation in M50 — plan only, mirroring the approach of M45.

---

## Open questions (to resolve before M46)

1. **Single vs multi-shard safetensors:** Initial target is single-file
   `model.safetensors`.  Multi-shard (`model-00001-of-00002.safetensors`) is
   out of scope until after M48.

2. **Tied weights:** If `lm_head.weight` is absent (tied to `embed_tokens`),
   the converter writes an explicit transposed copy.  No flag needed.

3. **GQA/MQA:** Not supported.  If `num_key_value_heads` differs from
   `num_attention_heads`, the converter emits an error.

4. **Bias terms:** LLaMA-style models typically have no bias in projection
   layers.  If bias tensors are present, the converter must error (not silently
   drop them).

5. **`rope_scaling` non-null:** The converter errors unless
   `rope_scaling.type == "linear"` and `rope_scaling.factor == 1.0`.

6. **f32 file size:** A 2-layer, d_model=64, vocab_size=256, d_ff=128 toy
   model is approximately 256 KB as f32.  Safe to check in.  A 2-layer,
   d_model=256, vocab_size=32000 model is approximately 60 MB — too large.
   The first real fixture should use a synthetic tiny vocab.

---

## Related Documents

- [real_model_conversion.md](real_model_conversion.md) — existing converter
  plan covering tensor naming, dtype rules, matrix conventions, and milestone
  history through M44
- [shard_metadata.md](shard_metadata.md) — shard metadata binary format and
  validation requirements
- [model_format.md](model_format.md) — `.att1` binary format specification
- [aimu_architecture.md](aimu_architecture.md) — AIMU tiled tensor architecture
