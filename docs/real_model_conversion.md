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
supported for M49 tiny converted artifacts.  A dtype-2 tensor must be a 2-D
matrix with payload layout:

```text
int8 values[rows * cols]  followed by  float32 scales[rows]
```

The tensor shape is the q8 runtime layout `[out_dim, in_dim]`, so q8
projection tensors are transposed relative to their f32 ATT-1 file shape.
RMSNorm tensors and `tok_embeddings.weight` remain dtype `1`.

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

## Per-row q8 quantization rules

When the runtime builds q8 copies from f32 `.att1` tensors, or when the
converter emits dtype-2 q8 tensors:

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
- Each f32 tensor's `nbytes` must equal `product(shape) * 4`.
- Each q8 tensor's `nbytes` must equal `rows * cols + rows * 4`.
- No duplicate tensor names in a single file.

---

## How single and cluster backends consume converted models

Single-tile inference (`att1_infer_t`):
1. Loads the `.att1` file via `att1_model_load()`.
2. Validates the model with `att1_model_view_validate_decoder()`.
3. If a q8 backend is selected, either borrows dtype-2 q8 tensors directly from
   the model file or builds runtime q8 copies from dtype-1 f32 tensors before
   decoding begins.
4. Decodes tokens via `att1_infer_decode_token()`.

Cluster inference (`att1_cluster_infer_t`):
1. Loads the same `.att1` file identically.
2. Builds a shard plan at runtime via `att1_shard_plan_build()`.
3. Dispatches each tile's layers through the selected backend.  q8 cluster
   backends borrow dtype-2 q8 tensors directly when present, or build runtime
   q8 copies from dtype-1 f32 tensors.
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
| q8 scale mismatch | dtype-2 files store deterministic per-row f32 scales; f32 files still derive runtime q8 copies |
| Tokenizer mismatch | Tokenizer is out of scope; converter emits raw byte IDs only for initial tests |
| Model format drift | Version field enforced; loader rejects unknown versions |
| safetensors C dependency | Converter is Python-only under `compiler/`; runtime never links it |

---

## Converter status (Milestones 32, 48, and 49)

`compiler/convert_llama_to_att1.py` supports validated config parsing,
deterministic stub emission, M48 f32 safetensors import, and M49 q8 safetensors
conversion.  With `--safetensors PATH`, real F32 tensor payloads are loaded
under `compiler/`.  `--weight-format f32` emits dtype-1 tensors, while
`--weight-format q8` emits dtype-2 projection/output tensors plus dtype-1
embedding and norm tensors.

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

### Still not implemented

- Tokenizer vocabulary import and runtime tokenizer selection
- Multi-shard safetensors or `.bin` shard weight loading
- BF16/F16 source upcast
- q4 conversion

### Tiny f32 safetensors conversion

```bash
python3 compiler/convert_llama_to_att1.py \
    --config compiler/fixtures/tiny_llama_config.json \
    --safetensors compiler/fixtures/tiny_llama_2l.safetensors \
    --out models/real_tiny_f32/model.att1

./build/att1-inspect models/real_tiny_f32/model.att1

# The fixture vocab is 16, so prompts must use raw byte IDs 0..15 until
# tokenizer import is implemented.
./build/att1-bench \
    --model models/real_tiny_f32/model.att1 \
    --prompt $'\x01' --tokens 2 \
    --mode single --backend cpu-f32

./build/att1-bench \
    --model models/real_tiny_f32/model.att1 \
    --prompt $'\x01' --tokens 2 \
    --mode cluster --tiles 2 --backend cpu-f32
```

The checked-in tiny f32 artifact is used by
`tests/test_converter_validation.c`, so `make test` does not invoke Python for
this validation path.

### Tiny q8 safetensors conversion

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

On CUDA builds with an available runtime, the same artifact is also validated
with `--backend cuda-q8` in single and cluster modes.  The checked-in q8
artifact is part of `tests/test_converter_validation.c`, so normal `make test`
remains Python-free.

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

---

## Real tiny model import plan (Milestone 45)

M45 is documentation-only.  It defines the full implementation plan for
importing a real tiny LLaMA-style model — covering source files, tensor
mappings, transpose rules, RoPE conventions, dtype conversion, hostile-input
validation, and the exact future milestone split.

See [docs/real_tiny_model_import.md](real_tiny_model_import.md) for the
complete plan.

## Tokenizer import plan (Milestone 50)

M50 is documentation-only.  The current byte tokenizer remains the runtime
default for tests and benchmarks.  Real tokenizer import is planned as a
converter-side Python path first, using `tokenizer.json`,
`tokenizer_config.json`, `special_tokens_map.json`, and later
`tokenizer.model` for SentencePiece models.

The import plan requires token ID stability, `vocab_size` agreement across
config, embedding rows, lm_head rows, and tokenizer vocab count, explicit
BOS/EOS/PAD/UNK handling, and documented byte-fallback behavior.  Runtime
tokenizer selection and optional `.att1` tokenizer metadata are deferred to
later milestones.

## Tokenizer metadata schema (Milestone 51)

M51 defines the optional tokenizer metadata schema in
[docs/tokenizer_metadata.md](tokenizer_metadata.md). The schema is logical only
in M51; it does not add a tokenizer metadata section to `.att1`, does not
require metadata for existing models, and does not change the current byte
tokenizer default.

The schema covers versioning, tokenizer type, vocabulary size, special token
IDs, byte fallback, normalization and pretokenizer policies, tokenizer asset
hashing, reserved future embedded asset offset/size fields, compatibility
checks against model vocabulary dimensions, hostile-input validation, and
runtime fallback rules.

## Public source comparison report (Milestone 69)

M69 extends `compiler/compare_att1_to_source.py` for public-model validation
using only local files.  The comparison harness now accepts a local
`--model-dir` containing `config.json` and `model.safetensors`, plus explicit
converted f32/q8 `.att1` artifact paths that may live outside the Git tree.

No model weights and no generated public `.att1` artifacts are checked in.
The runtime remains C11, the `.att1` binary format is unchanged, q4 remains
unimplemented, and tokenizer parsing stays outside C.

Manual validation after local conversion:

```sh
python3 compiler/compare_att1_to_source.py \
    --model-dir ~/Models/SmolLM2-135M \
    --att1-f32 ~/Models/att1/SmolLM2-135M/model_f32.att1 \
    --att1-q8  ~/Models/att1/SmolLM2-135M/model_q8.att1 \
    --prompt-ids 1,2,3 \
    --report \
    --report-json ~/Models/att1/SmolLM2-135M/source_comparison.json
```

The report includes:

- source model path, `config.json`, and `model.safetensors`
- ATT-1 artifact path, dtype, and selected backend
- prompt token IDs
- logits shape and source-reference next-token result
- f32/q8 `max_abs_error`, `max_rel_error`, tolerance, and pass/fail status

The numpy reference path loads the source safetensors directly and compares
f32/q8 artifacts against that local source reference.  q8 next-token
divergence remains a warning when static tensor error is within tolerance,
because near-tied logits can flip greedy argmax under quantisation.

## Public backend smoke validation (Milestone 70)

M70 adds a manual backend smoke driver for locally converted public artifacts:
`compiler/validate_public_backends.py`.  It is Python tooling only and wraps
the existing C11 `att1-bench` binary.  Runtime code, the `.att1` binary format,
q4, and C tokenizer parsing are unchanged.

The validator requires local external paths for the source model directory,
converted f32 artifact, converted q8 artifact, and a token IDs file:

```sh
# Optional: create an external token IDs file from the local tokenizer.
python3 compiler/tokenize_hf.py \
    --tokenizer ~/Models/SmolLM2-135M \
    --text "The answer is" \
    --out ~/Models/att1/SmolLM2-135M/prompt.ids \
    --add-special-tokens true

# Run backend smoke validation.
python3 compiler/validate_public_backends.py \
    --model-dir ~/Models/SmolLM2-135M \
    --att1-f32 ~/Models/att1/SmolLM2-135M/model_f32.att1 \
    --att1-q8  ~/Models/att1/SmolLM2-135M/model_q8.att1 \
    --tokens-file ~/Models/att1/SmolLM2-135M/prompt.ids \
    --tokens 1 \
    --tiles 2 \
    --report-json ~/Models/att1/SmolLM2-135M/backend_smoke.json
```

The validator runs:

| Artifact | Required CPU paths | Optional CUDA paths |
|----------|--------------------|---------------------|
| f32 | `cpu-f32` single, `cpu-f32` cluster | `cuda` single, `cuda` cluster |
| q8 | `cpu-q8` single, `cpu-q8` cluster | `cuda-q8` single, `cuda-q8` cluster |

Each report row includes artifact path, backend, mode, shard plan,
generated-token count, last token, total token timing, and pass/fail status.
CUDA rows that report unsupported or unavailable are marked `unsupported` and
do not fail the validation; CPU rows must pass.

## Public tokenized end-to-end validation (Milestone 71)

M71 adds `compiler/validate_public_tokenized.py`, a manual end-to-end driver
for the selected public model.  It first invokes the local HF tokenizer helper
from M59 (`compiler/tokenize_hf.py`) to generate token IDs, then runs the
pretokenized M58 `att1-bench --tokenizer external --tokens-file` path across
the f32/q8 CPU and optional CUDA backend smoke matrix.

All paths are local and external to Git:

```sh
python3 compiler/validate_public_tokenized.py \
    --model-dir ~/Models/SmolLM2-135M \
    --tokenizer-dir ~/Models/SmolLM2-135M \
    --prompt-text "The answer is" \
    --att1-f32 ~/Models/att1/SmolLM2-135M/model_f32.att1 \
    --att1-q8  ~/Models/att1/SmolLM2-135M/model_q8.att1 \
    --tokens-file ~/Models/att1/SmolLM2-135M/prompt.ids \
    --tokenizer-json ~/Models/att1/SmolLM2-135M/prompt_tokens.json \
    --tokens 1 \
    --tiles 2 \
    --report-json ~/Models/att1/SmolLM2-135M/tokenized_e2e.json
```

The text report records prompt text, generated token IDs and token count,
artifact path, backend, mode, generated-token count, last token, total token
timing, and pass/fail status.  CUDA rows that report unsupported or
unavailable are printed as `status=unsupported`; CPU rows must pass.

No tokenizer assets, public weights, or generated public `.att1` artifacts
are committed.  Runtime remains C11, the `.att1` format is unchanged, q4 is
not implemented, and BPE/SentencePiece parsing remains out of C.

### Future milestone sequence

| Milestone | Goal |
|-----------|------|
| M46 | safetensors metadata scanner (Python; no weight loading) |
| M47 | safetensors tensor reader (Python; dtype coercion, transpose) |
| M48 | Real f32 converted tiny model — first non-synthetic `.att1` artifact |
| M49 | q8 converted tiny model — dtype-2 loader extension, cross-backend check |
| M50 | Tokenizer import plan (documentation only) |
| M51 | Tokenizer metadata schema (documentation only) |
| M52 | Tokenizer scanner/parser skeleton |
| M53 | Tokenizer fixture import |
| M54 | Optional `.att1` tokenizer metadata section |
| M55 | Runtime tokenizer selection plan (documentation only) |
| M56 | Tokenizer selection CLI stub |
| M57 | Metadata tokenizer validation path |
| M58 | External tokenizer preprocessing mode |
| M59 | BPE tokenizer parser prototype |
| M60 | Tokenizer-aware converted model validation |
| M61 | Source-model comparison harness (Python; numpy forward pass vs ATT-1 bench) |
| M62 | Source comparison report integration (`--report`, `--report-json`, `--backend`) |
| M63 | Larger tiny-model fixture (vocab=64, d_model=32) and validation |
| M64 | Public small-model import candidate selection (documentation only) |
| M65 | Public model acquisition and import instructions (documentation only) |
| M66 | Public model compatibility scanner (`compiler/check_llama_compat.py`) |
| M67 | BF16/F16 source dtype coercion in `load_safetensors.py`; `_coerce_bf16`/`_coerce_f16` helpers; `TensorData.coerced` field; compat scanner promotes BF16/F16 to warning; BF16 fixture + smoke check |
| M68 | q8 conversion of BF16-source public model: `compare_att1_to_source.py` BF16 support; q8 smoke check (`check_q8_conversion`); manual validation workflow for public models; documented tolerance and token-divergence behaviour |
| M69 | Public model source comparison report: `--model-dir`, external f32/q8 artifact paths, source-safetensors numpy reference, richer pass/fail report fields |
| M70 | Public backend smoke validation: local f32/q8 artifacts across CPU single/cluster and optional CUDA single/cluster paths |
| M71 | Public tokenized end-to-end validation: local HF tokenizer IDs plus f32/q8 CPU and optional CUDA backend smoke report |
| M72 | Larger-model scaling and placement report: `att1-size --config`, `--layers/--d-model/--heads/--d-ff/--vocab-size`, `--json`; per-category storage, KV-cache by context, AIMU tile plan, backend feasibility; `check_scaling_report()` smoke test |
| M73 | q4 quantization planning: strategy document only; grouped int4 format spec, `.att1` format implications, converter and runtime plan, test plan, M74–M79 milestone split |
| M74 | q4 format and schema: `ATT1_MODEL_DTYPE_Q4=3` enum; `flags[7:0]` group_size encoding (0=default 32, powers-of-two in [16,128]); nbytes formula `rows*cols/2 + rows*(cols/group_size)*4` with overflow-safe validation in loader; `ATT1_ERR_UNSUPPORTED` from `att1_model_view_validate_decoder()` for any q4 tensor (no silent fallback); `att1-inspect` q4 reporting (`dtype_name=q4`, `quant=grouped-q4-g%u`, `q4_groups`, `q4_packed_bytes`, `q4_scale_bytes`); `tests/test_quant_q4.c` (9 checks); no `.att1` version bump; no q4 inference |
| M75 | CPU q4 packing/unpacking primitives: `att1_q4_group_scale()`, `att1_q4_pack_group()`, `att1_q4_unpack_group()`, `att1_q4_quantize_group()`, `att1_q4_dequantize_group()` in `src/quant.c`; 9 tests in `test_quant_q4_pack`; no q4 matmul, no q4 inference |
| M76 | CPU q4 matmul prototype: `att1_q4_matrix` struct, `att1_quantize_q4_per_group()`, `att1_matmul_q4xf32()` (dequantize-then-multiply, activations stay float32); `test_matmul_q4` (8 checks); no q4 inference. 45 tests. |
| M77 | q4 `.att1` fixture: `--weight-format q4` converter output, dtype-3 loader, `att1-inspect` q4 reporting, checked-in tiny q4 model |
| M78 | CPU q4 single-tile inference: `--backend cpu-q4`, single-tile decode validated against cpu-f32 |
| M79 | CUDA q4 matmul planning/prototype: dequantize-then-multiply in CUDA, tests against CPU q4 reference |
| M80 | GQA support: `n_kv_heads` config field, converter, runtime attention |
| M81 | SmolLM2-135M import and validation (first real public model) |
