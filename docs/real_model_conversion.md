# ATT-1 Real Model Conversion Plan

Milestone 30 documents the path from real model artifacts to a valid `.att1`
file that can run through the existing simulator inference stack.  No converter
is implemented here.

## Scope

### First target architecture

Tiny LLaMA-style decoder-only transformer:

- Single embedding table (token embeddings only)
- Per-layer stacked projection weights: `wq`, `wk`, `wv`, `wo`, `w_gate`,
  `w_up`, `w_down`
- Per-layer RMSNorm weights: `attention_norm`, `ffn_norm`
- Output-side RMSNorm weight: `output_norm`
- Output projection weight: `output_weight` (shared with embedding is allowed
  if the source model uses tied weights; the converter must write an explicit
  copy)
- Absolute RoPE positional encoding (theta convention: `10000.0` default;
  configurable)

Multi-head attention with no grouped-query or multi-query variants for the
initial target.

### Explicitly out of scope for this milestone

- Mixture-of-experts (MoE) layers
- q4 weights
- GPT-OSS 120B or any large-model real inference
- Tokenizer import or in-process BPE execution
- safetensors or GGUF parsing in C runtime code
- Any Python code required by `make test` or the C runtime

---

## Tensor naming convention

The `.att1` tensor `name[64]` field encodes each tensor with a flat ASCII key.
Layer-scoped tensors use the `layers.%u.` prefix where `%u` is the zero-based
layer index, matching the C `layer_name()` helper in `src/model_view.c`.

```text
tok_embeddings.weight               shape [vocab_size, d_model]    f32
layers.0.attention_norm.weight      shape [d_model]                f32
layers.0.attention.wq.weight        shape [d_model, d_model]       f32
layers.0.attention.wk.weight        shape [d_model, d_model]       f32
layers.0.attention.wv.weight        shape [d_model, d_model]       f32
layers.0.attention.wo.weight        shape [d_model, d_model]       f32
layers.0.ffn_norm.weight            shape [d_model]                f32
layers.0.ffn.w_gate.weight          shape [d_model, d_ff]          f32
layers.0.ffn.w_up.weight            shape [d_model, d_ff]          f32
layers.0.ffn.w_down.weight          shape [d_ff, d_model]          f32
...                                 (repeat for each layer)
output_norm.weight                  shape [d_model]                f32
output.weight                       shape [d_model, vocab_size]    f32
```

All names must be null-terminated within the 64-byte field.  Names that
exceed 63 characters are rejected by the loader.

---

## Supported dtypes

| Code | Value | Meaning                        |
|------|-------|--------------------------------|
| `1`  | 1     | float32 (current, required)    |
| `2`  | 2     | q8 (int8 + per-row f32 scales) |

The loader must reject any other dtype value.  q8 tensors in `.att1` files are
not yet implemented; dtype `2` is reserved for a future format update and must
not appear in milestone-30 files.  The converter must write dtype `1` for all
tensors regardless of the quantization path used at inference time — runtime q8
copies are always derived on load.

---

## Matrix conventions

- All matrices are stored **row-major** in the `.att1` file.
- Projection weights follow the convention used by the runtime:
  `weight[out_dim, in_dim]`.  This is the transposed form relative to the
  typical `[in_dim, out_dim]` layout in PyTorch; the converter must transpose
  before writing.
- The loader does not perform transposition.  Shape validation must match the
  convention above exactly.

---

## Per-row q8 quantization rules (inference-time only)

When the runtime builds q8 copies from f32 `.att1` tensors:

```text
scale[row] = max(abs(row_values)) / 127   (or 1.0 if row is all-zero)
q[row, col] = clamp(round(row_values[col] / scale[row]), -127, 127)
```

`-128` is intentionally excluded to preserve symmetric range.
Scales are float32.  Activations remain float32 throughout.

---

## Shard metadata expectations

For converted single-shard models, `shard_metadata_offset` and
`shard_metadata_size` should be zero (absent).  Cluster-mode sharding is
performed at runtime by `att1_shard_plan_build()` and does not require
pre-baked shard metadata in the file.  The shard metadata section of the format
is reserved for future hardware-targeting use.

---

## Format versioning rules

- `version` is currently `1`.
- Backwards-incompatible changes (field relocation, dtype removal, tensor
  renaming) require a version bump.
- The loader must reject files with unknown versions.
- The converter must write `version = 1` until a version bump is agreed.
- New optional fields or new dtype codes are permissible in `version = 1` only
  if the loader already rejects unknown values (which it does for dtype).

---

## Hostile-input validation requirements

The loader already enforces:

- Magic bytes `ATT1MODL` exact match
- Known version value
- `header_size` matches expected size
- `ndims <= 4`
- No zero-size dimensions
- Tensor data ranges within declared `tensor_data_size`
- Tensor data section does not extend past the file

The converter must produce files that pass all existing checks.  Additional
shape validation for converted models:

- `vocab_size`, `d_model`, `d_ff`, `n_heads`, `n_layers` must be nonzero and
  consistent across all tensors.
- `d_model % n_heads == 0` must hold.
- `rope_dim <= d_model / n_heads` must hold.
- Each tensor's `nbytes` must equal `product(shape) * sizeof(dtype)`.
- No duplicate tensor names in a single file.

---

## How single and cluster backends consume converted models

Single-tile inference (`att1_infer_t`):
1. Loads the `.att1` file via `att1_model_load()`.
2. Validates the model with `att1_model_view_validate_decoder()`.
3. If a q8 backend is selected, builds runtime q8 copies of all projection
   weights before decoding begins.
4. Decodes tokens via `att1_infer_decode_token()`.

Cluster inference (`att1_cluster_infer_t`):
1. Loads the same `.att1` file identically.
2. Builds a shard plan at runtime via `att1_shard_plan_build()`.
3. Dispatches each tile's layers through `att1_transformer_block_forward_backend()`.
4. The backend selection (cpu-f32, cpu-q8, cuda, cuda-q8) is independent of the
   file format.

No file format changes are required to support cluster mode.

---

## Validation ladder

| Stage | Model | Purpose | Status |
|-------|-------|---------|--------|
| 1 | Converted tiny synthetic model | Converter round-trip; loader acceptance | **complete (M32)** |
| 2 | Tiny trained toy model (e.g. char-level) | End-to-end logit sanity | pending |
| 3 | Small public LLaMA-like model | Real tokenizer output; greedy token spot-check | pending |
| 4 | Larger architecture | Simulation only; timing and memory counters | pending |

---

## Risks

| Risk | Mitigation |
|------|-----------|
| Tensor naming mismatch | Canonical name table in this document; converter asserts all names present |
| Transposed weights | Converter tests round-trip: load → multiply → compare to original |
| RoPE convention mismatch | Converter accepts `--rope-theta` argument; documented default is `10000.0` |
| q8 scale mismatch | Runtime quantizes from f32 file; no scale in file format until dtype `2` is defined |
| Tokenizer mismatch | Tokenizer is out of scope; converter emits raw byte IDs only for initial tests |
| Model format drift | Version field enforced; loader rejects unknown versions |
| safetensors C dependency | Converter is Python-only under `compiler/`; runtime never links it |

---

## Converter status (Milestone 32)

`compiler/convert_llama_to_att1.py` supports validated config parsing and
deterministic stub emission.  Real safetensors weight loading is not yet
implemented.

**Supported architectures:** `llama`, `mistral`

**Fixture:** `compiler/fixtures/tiny_llama/config.json`

### Manual validation sequence

```bash
# 1. Emit the deterministic stub model
python3 compiler/convert_llama_to_att1.py \
    --config compiler/fixtures/tiny_llama/config.json \
    --out models/converted_stub/model.att1

# 2. Inspect the emitted artifact
./build/att1-inspect models/converted_stub/model.att1

# 3. Run CPU f32 single-tile inference on the stub
./build/att1-bench \
    --model models/converted_stub/model.att1 \
    --prompt hello --tokens 4 \
    --mode single --backend cpu-f32
```

All three commands must exit 0.  The bench output should report
`generated_tokens=4` and nonzero `tokens_decoded`.

### Not yet implemented

- Safetensors or `.bin` shard weight loading
- Tokenizer vocabulary import
- Real weight transposition and shape alignment
- q8 pre-quantized `.att1` file emission (dtype `2`)

---

## Shard metadata emission (Milestone 42)

`compiler/convert_llama_to_att1.py` gains two new flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--tiles N` | 1 | Set `n_tiles` in the model config and control layer→tile assignment |
| `--shard-meta` | off | Emit a shard metadata section |

When `--shard-meta` is specified, the emitted artifact includes a shard
metadata section with one 120-byte record per tensor.  Layer tensors
(`layers.L.*`) are assigned to tiles using the same ceiling-division algorithm
as `att1_shard_plan_build()` in `src/shard.c`.

### Example: 2-tile stub with shard metadata

```bash
python3 compiler/convert_llama_to_att1.py \
    --config compiler/fixtures/tiny_llama_config.json \
    --tiles 2 --shard-meta \
    --out models/converted_stub_meta/model.att1

# Inspect (shows shard_meta: 21 records, plan_entries=2)
./build/att1-inspect models/converted_stub_meta/model.att1

# Cluster bench — runtime plan
./build/att1-bench \
    --model models/converted_stub_meta/model.att1 \
    --prompt hello --tokens 8 \
    --mode cluster --tiles 2 --shard-plan runtime --backend cpu-f32

# Cluster bench — metadata plan (no silent fallback)
./build/att1-bench \
    --model models/converted_stub_meta/model.att1 \
    --prompt hello --tokens 8 \
    --mode cluster --tiles 2 --shard-plan metadata --backend cpu-f32
```

The fixture `models/converted_stub_meta/model.att1` is checked into the
repository; no Python is required for `make test`.

---

## Shard plan report (Milestone 43)

`compiler/convert_llama_to_att1.py` gains two new reporting flags that work
with or without `--output` (dry-run compatible):

| Flag | Description |
|------|-------------|
| `--report` | Print a human-readable shard plan report to stdout |
| `--report-json PATH` | Write a JSON shard plan report to PATH |

Both flags are independent; they can be combined with `--shard-meta`,
`--tiles`, and `--output` in any combination.

### Report contents

| Section | Fields |
|---------|--------|
| Config summary | `vocab_size`, `n_layers`, `n_heads`, `d_model`, `d_ff`, `max_seq_len`, `rope_dim`, `rope_theta`, `n_tiles` |
| Shard metadata | `present`/`absent`, `tensor_count`, `tile_count`, `aimu_count` |
| Dtype summary | `dtype_f32`, `dtype_q8`, `quant_none` |
| Tile ownership | per-tile: `tile_id`, `aimu_id`, `tensor_count`, `layer_range` |
| Layer assignment | per-layer: `layer_id` → `tile_id` |
| Validation | `status` (`ok`/`failed`), error list |
| Tensors (JSON only) | per-tensor: `name`, `shape`, `dtype`, `quant`, `tile_id`, `bytes` |

### Example usage

```bash
# Human-readable report (dry run — no artifact emitted)
python3 compiler/convert_llama_to_att1.py \
    --config compiler/fixtures/tiny_llama_config.json \
    --tiles 2 --shard-meta --report

# JSON report written to file
python3 compiler/convert_llama_to_att1.py \
    --config compiler/fixtures/tiny_llama_config.json \
    --tiles 2 --shard-meta --report-json build/my_report.json

# Emit artifact and print report in one pass
python3 compiler/convert_llama_to_att1.py \
    --config compiler/fixtures/tiny_llama_config.json \
    --tiles 2 --shard-meta --report \
    --out models/converted_stub_meta/model.att1
```

### Smoke test

`tests/test_bench_smoke.c` includes `check_converter_report()` which:
1. Invokes `--report` and verifies key fields in the text output.
2. Invokes `--report-json` and verifies the JSON structure.

The check is skipped (not failed) when `python3` is absent from `$PATH`,
preserving the Python-free `make test` guarantee.

---

## Converter executable metadata plan validation (Milestone 44)

M44 proves that a converter-generated `.att1` stub with shard metadata can be
inspected and benchmarked end-to-end using both shard-plan modes.

### New C test: `tests/test_converter_validation.c`

Uses the checked-in `models/converted_stub_meta/model.att1` fixture — no
Python required at test time.

| Check | What it verifies |
|-------|------------------|
| `check_inspect()` | `att1-inspect` reports `tensor_count=21`, `n_tiles=2`, `shard_meta: 21 records`, `shard_meta_assigned=21`, `shard_meta_unassigned=0`, `shard_meta_dtype_f32=21`, both tile groups present |
| `check_bench_consistency()` | `att1-bench --shard-plan runtime` and `--shard-plan metadata` produce identical `last_token`, `logits_bytes_produced`, and `fabric_packets_sent` |

### Validation script: `compiler/validate_converter_flow.sh`

Developer-facing script (not run by `make test`) that exercises the full
pipeline end-to-end including Python generation:

```bash
bash compiler/validate_converter_flow.sh         # run and clean up
bash compiler/validate_converter_flow.sh --keep  # keep build/m44_validation/
```

Steps performed:

1. Generate converted stub via `convert_llama_to_att1.py --shard-meta`
2. Generate shard plan report (`--report --report-json`)
3. Run `att1-inspect` and validate key fields
4. Run `att1-bench --shard-plan runtime`
5. Run `att1-bench --shard-plan metadata`
6. Compare `last_token`, `logits_bytes_produced`, `fabric_packets_sent` — must agree

All artefacts written to `build/m44_validation/` (cleaned unless `--keep`).
