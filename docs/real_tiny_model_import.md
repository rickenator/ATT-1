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
| Tokenizer | Byte-level runtime default remains; real tokenizer import is converter-side first and runtime integration is deferred |

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
| `tokenizer.json` | No | Primary future tokenizer source; parsed by converter-side tooling first |
| `tokenizer.model` | No | Future SentencePiece source if a target fixture uses it |
| `tokenizer_config.json` | No | Future source for model-specific tokenizer behavior and special IDs |
| `special_tokens_map.json` | No | Future source for BOS/EOS/PAD/UNK names and IDs |

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

**Goal:** Define how real tokenizer assets will be imported into ATT-1
conversion tooling without implementing parsing or runtime integration yet.

**Scope:** documentation/spec only.  No C source changes, Makefile changes,
external dependencies, `.att1` model-format changes, tokenizer parser, or
runtime tokenizer selection are part of M50.

**First supported tokenizer path:**

- Current tests and benchmarks keep the existing byte-level tokenizer behavior.
  The tiny real-model fixtures still require raw byte IDs in the valid
  vocabulary range until a later runtime milestone.
- Real tokenizer import starts in Python converter tooling under `compiler/`.
  The first parser should inspect and validate tokenizer assets, then report
  metadata needed for future `.att1` integration.
- Runtime tokenizer integration is deferred.  No tokenizer import is required
  by `make test`.

**Expected tokenizer assets:**

| File | Role |
|------|------|
| `tokenizer.json` | First target for Hugging Face byte-level BPE tokenizer metadata, vocab entries, merges, normalizer/pre-tokenizer declarations, and added tokens |
| `tokenizer.model` | Future SentencePiece source if a target model uses SentencePiece instead of `tokenizer.json` |
| `tokenizer_config.json` | Source for tokenizer class, byte fallback settings, normalization flags, model max length, and declared special token fields |
| `special_tokens_map.json` | Source for BOS, EOS, PAD, UNK, and any additional special-token strings |

**Vocabulary mapping requirements:**

- Token IDs must be stable: the imported vocabulary index for every token must
  exactly match the source tokenizer ID.
- `vocab_size` must agree across `config.json`, `tok_embeddings.weight` rows,
  and `lm_head.weight` rows.  Any mismatch is a hard validation error.
- BOS, EOS, PAD, and UNK handling must preserve both token string and token ID
  when present.  Missing PAD is allowed only if the source tokenizer omits it;
  missing BOS/EOS/UNK must be reported explicitly.
- Added tokens must not shift base vocabulary IDs.  Duplicate token strings or
  duplicate IDs are fatal unless the source format explicitly aliases them and
  the aliasing rule is documented.
- Byte fallback behavior must be detected and reported.  If byte fallback is
  absent, future runtime code must not silently substitute the current byte
  tokenizer.

**Runtime options for future milestones:**

- Keep the current byte tokenizer as the default path for existing tests,
  dummy models, and tiny fixtures.
- Add optional tokenizer metadata to `.att1` in a later milestone only after a
  schema is documented and hostile-input validation rules are defined.
- Allow converted models to remain weight-only; tokenizer import must stay
  optional and must not become a dependency of `make test`.

**Validation risks:**

| Risk | Required mitigation |
|------|---------------------|
| Mismatched vocab size | Compare config, embedding rows, lm_head rows, and tokenizer vocab count before conversion succeeds |
| Shifted token IDs | Verify selected known tokens and specials retain exact source IDs |
| Special-token mismatch | Report BOS/EOS/PAD/UNK string and ID from all available tokenizer files and fail on disagreement |
| UTF-8 normalization differences | Record normalizer/pre-tokenizer settings; future runtime must match or explicitly reject unsupported settings |
| BPE/SentencePiece incompatibility | Treat tokenizer type as part of metadata; do not parse one format as the other |
| Prompt encoding divergence | Add future golden prompt-to-ID fixtures before runtime tokenizer selection is enabled |

**Future milestone split:**

| Milestone | Goal |
|-----------|------|
| M51 | Tokenizer metadata schema: document fields, binary/sidecar options, versioning, and hostile-input validation rules |
| M52 | Tokenizer scanner/parser skeleton under `compiler/`; inventory assets and report tokenizer type without runtime integration |
| M54 | Optional `.att1` tokenizer metadata section; C parser, loader detection, inspect reporting, Python fixture generator. ✅ |
| M55 | Runtime tokenizer selection plan: modes, CLI policy, compatibility checks, failure policy, and milestone split. ✅ |
| M56 | Tokenizer selection CLI stub; byte wired; metadata/external stub-fail. |
| M57 | Metadata tokenizer validation path; no BPE parser yet. |
| M58 | External tokenizer preprocessing mode. ✅ |
| M59 | Local HF tokenizer helper: `compiler/tokenize_hf.py` — local text → token IDs; no C changes; no network. ✅ |
| M60 | Converted model validation with pretokenized input: `att1-bench --tokenizer external` on `real_tiny_f32` and `real_tiny_q8`; cpu-f32/q8 single+cluster; CUDA skipped if absent. ✅ |
| M61 | Source-model comparison harness: Python harness validates ATT-1 f32/q8 artifacts against source safetensors; static tensor mapping with transpose rules; numpy LLaMA forward pass; next-token match; m61 fixture with seeded random weights. ✅ |
| M62 | Source comparison report integration: `--report`, `--report-json`, `--tokens-file`, `--backend`, `max_rel_error`, `logits_shape`; structured JSON result dict; hard fail on bad report path; smoke test extended. ✅ |
| M63 | Larger tiny-model fixture and validation: vocab=64, d_model=32, d_ff=64, n_heads=4, n_layers=2; seeded safetensors; f32+q8 models; all four bench modes validated; source comparison passes. ✅ |
| M64 | Public small-model import candidate selection: five candidates evaluated (SmolLM2-135M recommended); shared BF16 blocker documented; GQA format+runtime+converter requirements documented; prerequisite milestone plan M65–M67. ✅ |

### M51 — tokenizer metadata schema

**Goal:** Define the optional tokenizer metadata schema for future converter
import and runtime selection without changing the current `.att1` model format.

**Schema summary:**

- Metadata is optional; existing `.att1` models remain valid without it.
- Existing byte-level tokenizer behavior remains the default for tests and
  benchmarks.
- Version `1` records tokenizer type (`byte`, `bpe_json`, `sentencepiece`, or
  `unknown`), vocabulary size, BOS/EOS/PAD/UNK IDs, byte-fallback flag,
  normalization policy, pretokenizer policy, tokenizer asset hash, and reserved
  future asset offset/size fields.
- Future compatibility checks must compare metadata `vocab_size` against
  `config.json`, embedding rows, and lm_head rows.
- Hostile-input validation and runtime fallback behavior are defined before any
  parser or runtime integration work begins.

See [tokenizer_metadata.md](tokenizer_metadata.md) for the complete schema.

### M52 — tokenizer asset scanner (complete)

**Goal:** Scan a directory of tokenizer asset files, parse available metadata,
and report tokenizer type, vocabulary, special-token IDs, and checksums —
without implementing a tokenizer or modifying runtime behavior.

**Implementation:**

- **`compiler/scan_tokenizer.py`** — new module + CLI.  Public API:
  - `TokenizerScanError(Exception)` — raised for fatal I/O or JSON errors.
  - `scan_tokenizer_dir(dirpath, config_path=None) -> dict` — discovers and
    parses `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`,
    and detects `tokenizer.model` (unsupported in M52).  Returns:
    `dir`, `assets`, `asset_hashes` (SHA-256), `tokenizer_type` (`bpe_json`,
    `sentencepiece`, `byte`, or `unknown`), `vocab_size`, `bos_id`, `eos_id`,
    `pad_id`, `unk_id`, `byte_fallback`, `normalizer`, `pretokenizer`,
    `config_vocab_size`, `vocab_size_match`, `warnings`, `errors`.
  - `format_tokenizer_report(result, show_detail=True) -> str` — human-readable
    `key=value` output followed by per-asset listing and `scan: ok`.

  CLI flags: `dirpath`, `--config PATH`, `--json`, `--no-detail`.
  Exit codes: 0 = ok, 1 = fatal error or scan errors, 2 = vocab_size mismatch.

  Error conditions:
  - Directory not found → `TokenizerScanError`.
  - `tokenizer.json` malformed (bad JSON, non-object root) → exit 1.
  - `tokenizer_config.json` malformed → exit 1.
  - `tokenizer.model` present → `tokenizer_type=sentencepiece`, error reported,
    exit 1 (unsupported in M52).
  - `config.vocab_size` mismatch → `vocab_size_match=no`, exit 2.
  - No assets found → `tokenizer_type=byte`, warning only, exit 0.

- **`compiler/fixtures/tiny_tokenizer/tokenizer.json`** — minimal BPE tokenizer
  with 16-token vocab (UNK=0, BOS=1, EOS=2, plus 13 byte tokens), ByteLevel
  pre-tokenizer, no normalizer, no merges.
- **`compiler/fixtures/tiny_tokenizer/tokenizer_config.json`** — declares
  `bos_token="<s>"`, `eos_token="</s>"`, `unk_token="<unk>"`,
  `vocab_size=16`.
- **`compiler/fixtures/tiny_tokenizer/special_tokens_map.json`** — maps
  `bos_token`, `eos_token`, `unk_token`.

- **`tests/test_bench_smoke.c`** — `check_tokenizer_scanner()` added
  (Python-skippable): basic scan asserts `tokenizer_type=bpe_json`,
  `vocab_size=16`, `bos_id=1`, `eos_id=2`, `scan: ok`; config cross-check
  asserts `vocab_size_match=yes`.

- No C source change (other than `test_bench_smoke.c`), no Makefile change,
  no `.att1` format change.  `make test` passes (39 tests).

### M53 — tokenizer fixture import report (complete)

**Goal:** Extend the M52 tokenizer scanner to produce a deterministic
human-readable and JSON tokenizer import report — including import readiness,
unsupported-field listing, and a canonical composite asset hash — without
embedding tokenizer metadata into `.att1` and without changing runtime behavior.

**New public API in `compiler/scan_tokenizer.py`:**

- `build_import_report(result: dict) -> dict` — extends a `scan_tokenizer_dir`
  result with three additional keys:
  - `import_ready` (`bool`) — `True` when no scan errors, no vocab_size
    mismatch, tokenizer type is `bpe_json`, and no unsupported fields are
    present.
  - `unsupported_fields` (`list[str]`) — names every detected-but-unsupported
    item: `tokenizer.model` (binary SentencePiece), `tokenizer_type` (unknown),
    `tokenizer_type=sentencepiece`, or non-null normalizer types.
  - `canonical_hash` (`str`) — SHA-256 hex digest of all present asset files
    concatenated in canonical order (`tokenizer.json`, `tokenizer.model`,
    `tokenizer_config.json`, `special_tokens_map.json`); absent files are
    skipped.  Deterministic for a given set of present assets.

- `format_import_report(result: dict, show_detail: bool = True) -> str` —
  human-readable import report with sections: header fields, `# compatibility`,
  `# unsupported fields`, `# assets`, errors/warnings, and `report: ok`.

**New CLI flags:**

| Flag | Meaning |
|------|---------|
| `--report` | Print human-readable import report to stdout |
| `--report-json PATH` | Write JSON import report to `PATH` (no stdout output when used alone) |
| `--model-config PATH` | Alias for `--config` (friendly name) |

Both `--report` and `--report-json PATH` can be combined: text goes to stdout,
JSON goes to the file.

**Determinism guarantees:**
- `canonical_hash` is identical for identical asset file contents.
- Field ordering in text and JSON output is fixed.
- Two invocations on the same fixture produce byte-identical reports.

**Error/edge-case behavior (unchanged from M52):**
- Directory not found → `TokenizerScanError` → exit 1.
- `tokenizer.json` malformed → exit 1.
- `tokenizer.model` present → `unsupported_fields=[tokenizer.model]`,
  `import_ready=no`, exit 1.
- `vocab_size` mismatch → `import_ready=no`, `vocab_size_match=no`, exit 2.

**`tests/test_bench_smoke.c`** — `check_tokenizer_import_report()` added
(Python-skippable, wired into `main()`):
1. Runs `--report --model-config ...` → asserts `tokenizer_type=bpe_json`,
   `vocab_size=16`, `bos_id=1`, `eos_id=2`, `vocab_size_match=yes`,
   `import_ready=yes`, `canonical_hash=`, `unsupported_fields=none`,
   `report: ok`.
2. Runs `--report-json build/tok_import_report.json` → asserts JSON contains
   `"tokenizer_type"`, `"bpe_json"`, `"import_ready"`, `"canonical_hash"`,
   `"unsupported_fields"`.
3. Creates a mismatched config (`vocab_size=32`) → asserts `vocab_size_match=no`
   and `import_ready=no`.

No C source change (other than `test_bench_smoke.c`), no Makefile change,
no `.att1` format change.  `make test` passes (39 tests).

### M54 — optional `.att1` tokenizer metadata section (complete)

**Goal:** Add an optional 96-byte tokenizer metadata section to `.att1` v2
files.  No runtime tokenization behavior changes.  Old v1 models remain
fully compatible.

**`.att1` version 2:**  Header grows from 80 to 96 bytes.  Bytes 80–95 carry
two new `uint64` fields: `tok_meta_offset` and `tok_meta_size`.  The version
field is `2` and `header_size` is `96`.  Version 1 models (80-byte header)
load unchanged; `tok_meta.present` is `0` for them.

**New files:**

| File | Purpose |
|------|---------|
| `include/att1_tok_meta.h` | Wire layout constants, `att1_tok_meta` struct, `att1_tok_meta_parse()`, enum name helpers |
| `src/tok_meta.c` | `att1_tok_meta_parse()` implementation with full hostile-input validation |
| `compiler/make_tok_meta_fixture.py` | Generates `models/tok_meta/model.att1` (868 bytes) |
| `models/tok_meta/model.att1` | Checked-in 868-byte v2 fixture (bpe_json, vocab=16, bos=1, eos=2) |
| `tests/test_tok_meta.c` | 8 test cases (absent, valid, bad_version, bad_tok_type, vocab_mismatch, truncated, bad_flags, v1_compat) |

**Modified files:**

| File | Change |
|------|--------|
| `include/att1_model.h` | `ATT1_MODEL_VERSION_2=2`, `ATT1_MODEL_HEADER_SIZE_V2=96`, `tok_meta` field in `att1_model` |
| `src/model_loader.c` | v1/v2 dispatch; tok_meta offset/size range validation; calls `att1_tok_meta_parse()` |
| `tools/att1-inspect.c` | Reports `tok_meta: present` with all fields, or `tok_meta: absent` |
| `Makefile` | `tok_meta.c` in `COMMON_SRCS`; `test_tok_meta` in `TEST_NAMES` |

**Fixture tokenizer metadata wire values:**
`tokenizer_type=bpe_json`, `vocab_size=16`, `bos=1`, `eos=2`, `pad=-1`,
`unk=0`, `byte_fallback=0`, `normalization=none`, `pretokenizer=byte_level`,
`asset_hash_kind=sha256`,
`asset_hash=da38c00aa0b62d2699c46cb2d1ccadce14b36815d784265680b78a7a31ab1034`,
`flags=0`, `asset_offset=0`, `asset_size=0`.

**Validation rules enforced by `att1_tok_meta_parse()`:**
- Section size must be exactly 96 bytes.
- `schema_version` must be `1`.
- `tokenizer_type` must be in `[0, 3]`.
- `vocab_size > 0` and must equal model config `vocab_size`.
- Special token IDs must be `−1` or in `[0, vocab_size)`.
- `byte_fallback` must be `0` or `1`.
- `normalization_policy` must be in `[0, 4]`.
- `pretokenizer_policy` must be in `[0, 5]`.
- `asset_hash_kind` must be `0` or `1`.
- `flags` must be `0`.
- `asset_offset` and `asset_size` must be both zero or both nonzero.

`make test` passes (40 tests).  No inference behavior change.  No C tokenizer
parser added.

### M55 — runtime tokenizer selection plan (complete)

**Goal:** Define how ATT-1 will select between byte-level tokenization and
metadata-declared tokenization, without implementing any runtime tokenizer
parser.  Documentation-only.  No C source, Makefile, or `.att1` format changes.
No inference behavior change.

See `docs/tokenizer_metadata.md` §M55 for the full spec.  Summary:

- Byte tokenizer remains the sole active runtime path.
- Three future modes: `byte` (default), `metadata`, `external`.
- Future `--tokenizer byte|metadata|external` CLI flag (M56 stub).
- `--tokenizer metadata` without a present, valid, supported tok_meta is a
  hard error — no silent fallback.
- Compatibility checks before metadata mode: vocab_size match, special token
  ID range, normalization/pretokenizer support, asset hash match.
- Testing strategy: byte baseline unchanged; metadata selection tests precede
  implementation; golden prompt-to-ID fixtures required before M60.
- Milestone split: M56 CLI stub → M57 validation path → M58 external mode →
  M59 BPE parser → M60 tokenizer-aware model validation.

`make test` passes (40 tests).  No code changes.

### M56–M58 — tokenizer CLI and external preprocessing (complete)

M56 added the `--tokenizer byte|metadata|external` CLI flag to `att1-bench`.
M57 added runtime validation for the `metadata` path (`att1_tok_meta_check_runtime()`).
M58 wired `--tokenizer external` end-to-end: `--input-token-ids` and
`--tokens-file` supply pre-tokenized IDs, validated against `vocab_size`.

See `docs/tokenizer_metadata.md` §M56–M58 for full details.

`make test` passes (41 tests) after M58.

### M59 — local HF tokenizer helper (complete)

**Goal:** Add `compiler/tokenize_hf.py` to bridge external Hugging Face
tokenizers to the ATT-1 external preprocessing mode without adding a BPE/
SentencePiece parser to the C runtime.

**File:** `compiler/tokenize_hf.py`

**What it does:**

- Accepts `--tokenizer PATH` (directory containing `tokenizer.json`),
  `--text TEXT` or `--text-file PATH` for input, and optional
  `--out PATH` (one-ID-per-line), `--json-out PATH`, `--add-special-tokens`,
  and `--timing` flags.
- Loads the tokenizer locally using the `tokenizers` library (preferred) or
  `transformers` with `local_files_only=True`.  No network access.
- Prints comma-separated token IDs to stdout (usable with
  `att1-bench --input-token-ids`) and optionally writes a one-per-line IDs file
  (usable with `att1-bench --tokens-file`).
- Exits 1 on path/argument error, 2 if neither `tokenizers` nor `transformers`
  is installed (with install guidance), 3 on tokenization error.

**Typical workflow:**

```sh
# Tokenize and pass to att1-bench in one step
IDS=$(python3 compiler/tokenize_hf.py \
        --tokenizer /path/to/llama/tokenizer \
        --text "Once upon a time" \
        --add-special-tokens false)
./build/att1-bench --model MODEL.att1 --tokens 32 --mode single \
    --tokenizer external --input-token-ids "$IDS"

# Or: write IDs to file, then pass with --tokens-file
python3 compiler/tokenize_hf.py \
    --tokenizer /path/to/tokenizer --text "Once upon a time" \
    --out build/prompt_ids.txt --add-special-tokens false
./build/att1-bench --model MODEL.att1 --tokens 32 --mode single \
    --tokenizer external --tokens-file build/prompt_ids.txt
```

**Constraints unchanged:** Python under `compiler/` only; C runtime unchanged;
`make test` Python-free except existing skippable smoke tests.

See `docs/tokenizer_metadata.md` §M59 for full specification.

`make test` passes (41 tests) after M59.

### M60 — converted model validation with pretokenized input (complete)

**Goal:** Validate that `att1-bench --tokenizer external` works end-to-end
with the checked-in `real_tiny_f32` and `real_tiny_q8` model artifacts.

**Why external mode is required here:** Both models have `vocab_size=16`.
The byte tokenizer maps text to ASCII code-points, which for any typical
English text produce IDs ≥ 32 — all out of range.  External mode with
fixture IDs (1,3,5 — BOS, 'a', 'c' in the synthetic tiny tokenizer) is the
correct path.

**Coverage added:**

| Check | Model | Mode | Backend |
|-------|-------|------|---------| 
| `--tokens-file` and `--input-token-ids` single | `real_tiny_f32` | single | cpu-f32 |
| `--tokens-file` cluster | `real_tiny_f32` | cluster | cpu-f32 |
| `--tokens-file` single | `real_tiny_q8` | single | cpu-q8 |
| `--tokens-file` cluster | `real_tiny_q8` | cluster | cpu-q8 |
| CUDA f32/q8 single+cluster | both | single+cluster | cuda/cuda-q8 |
| M59 helper pipeline (Python-skippable) | both | single+cluster | cpu-f32/q8 |

CUDA tests skip gracefully when CUDA is unavailable.  The M59 pipeline
check (`check_pretokenized_pipeline` in `test_bench_smoke.c`) runs
`tokenize_hf.py` on the tiny tokenizer fixture and feeds the output to the
bench tool; it is skipped when neither `tokenizers` nor `transformers` is
installed.

**Files changed:** `tests/test_converter_validation.c`,
`tests/test_bench_smoke.c`.  No C source changes.  No Makefile changes.

See `docs/tokenizer_metadata.md` §M60 for full specification.

`make test` passes (41 tests) after M60.

### M61 — source-model comparison harness (complete)

**Goal:** Add a Python comparison harness that validates ATT-1 converted
f32/q8 model artifacts against the source LLaMA-style safetensors weights
and against a Python-native reference forward pass.  No C changes.  No
`.att1` format changes.  No network access.  Checked-in fixtures only.

**New files:**

| File | Purpose |
|------|---------|
| `compiler/fixtures/make_m61_fixture.py` | Generates `m61_llama_2l.safetensors` with seeded random weights |
| `compiler/fixtures/m61_llama_2l.safetensors` | Non-trivial fixture: seed=61, scale=0.1 (8347 bytes, 21 tensors) |
| `models/m61_f32/model.att1` | Converted f32 artifact (9108 bytes; `last_token=10` for prompt `[5]`) |
| `models/m61_q8/model.att1` | Converted q8 artifact (5524 bytes) |
| `compiler/read_att1.py` | Stdlib-only ATT-1 binary reader (v1/v2, f32 and q8 dequantize) |
| `compiler/compare_att1_to_source.py` | Full comparison harness (static + forward-pass) |

**Modified files:**

| File | Change |
|------|--------|
| `tests/test_bench_smoke.c` | Added `check_source_comparison()` (Python/numpy-skippable) |

**Static tensor comparison** verifies all 21 LLaMA tensor mappings (embedding,
norms, Q/K/V/O projections, gate/up/down FFN, output norm, lm_head) applying
the same transpose rules as the converter:
- Projection and output matrices (wq, wk, wv, wo, w_gate, w_up, w_down,
  output.weight): transposed (HF stores [out, in]; ATT-1 stores [in, out]).
- Embeddings and norm weights: no transpose.

Results: f32 `max_abs_error=0.000e+00` (exact float copy), q8
`max_abs_error=5.407e-01` (within tolerance 0.6).

**Forward-pass comparison** (`_python_forward_pass` in
`compare_att1_to_source.py`) implements:
- RMSNorm: `x / sqrt(mean(x²) + 1e-6) * weight`
- RoPE: per-pair `(x0·cos − x1·sin, x0·sin + x1·cos)`, frequency =
  `1/theta^(i/count)`, matching `att1_rope_f32` exactly
- SwiGLU: `silu(gate) * up`, where `silu(x) = x / (1 + exp(−x))`
- Causal attention: per-head scaled dot-product with incremental KV cache
- All math via ATT-1 weight layout (`output = input @ weight`)

For prompt `[5]` on the m61 fixture: `ref_last_token=10`, `bench_last_token=10`,
`forward_match=yes`, `q8_bench_last_token=10`, `q8_forward_match=yes`.

**Why a new fixture?** The existing `tiny_llama_2l.safetensors` has all-zero
weights, making numerics trivial — any implementation produces the same
argmax (all-zero logits → argmax=0).  The m61 fixture uses seeded random
weights to ensure a non-trivial, deterministic next-token prediction.

`make test` passes (41 tests) after M61.  No new C test binary.

### M62 — source comparison report integration (complete)

**Goal:** Extend `compiler/compare_att1_to_source.py` to produce deterministic
human-readable and JSON reports, expose `max_rel_error` and `logits_shape`
fields, and add a `--tokens-file` / `--backend` option.  No C changes.  No
`.att1` format changes.

**New CLI flags:**

| Flag | Meaning |
|------|---------|
| `--report` | Rich structured text report (`report: ok` at the end) |
| `--report-json PATH` | Write JSON report to PATH; exit 1 on IO error |
| `--tokens-file PATH` | Load prompt token IDs from file (overrides `--prompt-ids`) |
| `--backend {cpu-f32,cpu-q8,cuda,cuda-q8}` | Backend for f32 forward-pass bench |

**New output fields (default key=value mode):**

- `f32_max_rel_error` — maximum relative error across f32 static tensor check
- `q8_max_rel_error` — maximum relative error across q8 static tensor check
- `logits_shape` — shape of the reference forward-pass output tensor

**JSON report structure (`--report-json`):**

```json
{
  "date":        "YYYY-MM-DD",
  "safetensors": "...",
  "config_path": "...",
  "config":      { "vocab_size": ..., "n_layers": ..., ... },
  "f32_static":  { "tensors_checked": ..., "max_abs_error": ..., "max_rel_error": ...,
                   "tolerance": ..., "status": "pass|fail", "per_tensor": [...] },
  "q8_static":   { ... },
  "forward":     { "prompt_ids": [...], "logits_shape": [...], "ref_last_token": ...,
                   "f32_forward_match": ..., "q8_forward_match": ..., ... },
  "result":      "pass|fail"
}
```

**Failure policy:**
- `--report-json` with an unwritable path → print error to stderr, exit 1
  (hard fail, no warning-only fallback).
- Static error above tolerance → `status: fail`, overall `result: fail`.
- Forward-pass argmax mismatch (f32) → hard fail.
- Forward-pass argmax mismatch (q8) → warning only (quantisation rounding
  can shift near-tied logits).

**Smoke test additions (`tests/test_bench_smoke.c`):**

1. `--report` mode: asserts `report: ok` and `result: pass` in output.
2. `--report-json`: asserts JSON file contains `"result"`, `"f32_static"`,
   `"q8_static"`, `"forward"`, `"config"`.
3. Bad `--report-json` path: uses new `command_fails()` helper to assert
   non-zero exit status.

`make test` passes (41 tests) after M62.  No new C test binary.

### M63 — larger tiny-model fixture and validation (complete)

**Goal:** Replace the minimal M61 tiny fixture (vocab=16, d_model=8) with a
wider model fixture that produces a more numerically meaningful forward pass.
No C changes.  No `.att1` format changes.  No network access.

**Fixture dimensions:**

| Parameter | M61 value | M63 value |
|-----------|-----------|-----------|
| vocab_size | 16 | 64 |
| d_model | 8 | 32 |
| n_heads | 2 | 4 |
| d_ff | 16 | 64 |
| n_layers | 2 | 2 |
| max_seq_len | 128 | 32 |
| weight_scale | 0.10 | 0.02 |
| seed | 61 | 63 |

The smaller weight scale (0.02 vs 0.1) prevents logit saturation in the wider
model where each matrix product accumulates more terms.

**New files:**

| File | Purpose |
|------|---------|
| `compiler/fixtures/make_m63_fixture.py` | Generator — seeded random F32 weights, same pattern as M61 |
| `compiler/fixtures/m63_llama_config.json` | LLaMA-style config for the M63 fixture |
| `compiler/fixtures/m63_llama_2l.safetensors` | 21 tensors, 101064 bytes |
| `models/m63_f32/model.att1` | Converted f32 artifact (101748 bytes) |
| `models/m63_q8/model.att1` | Converted q8 artifact (36724 bytes) |

**Modified files:**

| File | Change |
|------|--------|
| `tests/test_bench_smoke.c` | Added `check_m63_validation()` (Python+numpy-skippable) |

**Validation results:**

| Mode | Backend | last_token |
|------|---------|-----------|
| single | cpu-f32 | 11 |
| cluster (tiles=2) | cpu-f32 | 11 |
| single | cpu-q8 | 11 |
| cluster (tiles=2) | cpu-q8 | 11 |

**Source comparison** (prompt `[5, 20, 40]`):

- `f32_max_abs_error = 0.000e+00` (exact copy)
- `q8_max_abs_error = 1.190e-01` (well below 0.6 tolerance)
- `f32_forward_match = yes`, `q8_forward_match = yes`
- `result = pass`

**Smoke test additions (`tests/test_bench_smoke.c`):**

`check_m63_validation()` (Python+numpy-skippable, wired into `main()`):
1. f32 single: asserts `mode=single`, `backend=cpu-f32`, `last_token=`.
2. f32 cluster: asserts `mode=cluster`, `fabric_packets_sent=`.
3. q8 single: asserts `mode=single`, `backend=cpu-q8`, `last_token=`.
4. q8 cluster: asserts `mode=cluster`, `backend=cpu-q8`, `fabric_packets_sent=`.
5. `--report`: asserts `result:              pass`, `report:              ok`.
6. `--report-json`: asserts JSON contains `"result"`, `"f32_static"`, `"forward"`.

`make test` passes (41 tests) after M63.  No new C test binary.
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

### M64 — public small-model import candidate selection (complete)

**Goal:** Identify and document a suitable small public LLaMA-style model for
the next real ATT-1 conversion milestone.  Documentation and planning only.
No C source changes, Makefile changes, `.att1` format changes, or new
converter logic.

#### Candidate summary

| # | Model | Params | Attn | f32 | q8 | License | Shards | Verdict |
|---|-------|--------|------|-----|----|---------|--------|---------|
| A | `HuggingFaceTB/SmolLM2-135M` | 135 M | GQA 3:1 | ~540 MB | ~135 MB | Apache 2.0 | 1 | **Recommended** |
| B | `HuggingFaceTB/SmolLM2-360M` | 360 M | GQA 3:1 | ~1.4 GB | ~360 MB | Apache 2.0 | 1 | Alternate |
| C | `openlm-research/open_llama_3b_v2` | 3.4 B | MHA | ~13.5 GB | ~3.4 GB | Apache 2.0 | 2+ | No-GQA alt |
| D | `Qwen/Qwen2-0.5B` | 494 M | GQA 7:1 | ~1.9 GB | ~490 MB | Apache 2.0 | 1 | Arch gap |
| E | `meta-llama/Llama-3.2-1B` | 1.24 B | GQA 4:1 | ~4.9 GB | ~1.24 GB | Llama 3.2 | 1 | Licensed |

#### Shared blocker (all candidates)

All five models store weights as **bfloat16** in their HF safetensors files.
The current `compiler/load_safetensors.py` raises `LoadError` for any non-F32
dtype.  BF16→F32 coercion must be added to the tensor reader before any public
model can be converted.  This is a **Python-only change** — no C, Makefile, or
`.att1` format changes required.

#### Candidate A — SmolLM2-135M (recommended)

| Config field | Value |
|-------------|-------|
| HF repo | `HuggingFaceTB/SmolLM2-135M` |
| model_type | `llama` |
| vocab_size | 49152 |
| hidden_size (d_model) | 576 |
| intermediate_size (d_ff) | 1536 |
| num_hidden_layers (n_layers) | 30 |
| num_attention_heads (n_heads) | 9 |
| num_key_value_heads (n_kv_heads) | 3 |
| head_dim | 64 |
| max_position_embeddings | 2048 |
| rope_theta | 10000.0 |
| Source dtype | bfloat16 |
| Tokenizer | BPE JSON, vocab=49152 |
| License | Apache 2.0 |
| Safetensors | Single shard (~270 MB BF16) |

**Tensor naming:** Identical to HF LLaMA conventions used by the existing
converter.  A 30-layer model has 1 + 30×9 + 2 = **273 tensors** total.
All 21 mapping rules in the current converter apply without modification.

**Config field resolution:** All fields resolve through the existing
`FIELD_ALIASES` and `ROPE_THETA_KEYS` maps in `convert_llama_to_att1.py`.
No converter config-reading changes are needed.

**Known incompatibilities:**

1. **BF16 source** — shared blocker; requires BF16→F32 coercion in
   `compiler/load_safetensors.py` (Python only, no C/format change).

2. **GQA (n_kv_heads=3, n_heads=9).** The K and V projection weights have
   source shape `[n_kv_heads × head_dim, d_model]` = `[192, 576]` rather
   than the full MHA shape `[576, 576]`.  After the standard ATT-1 transpose
   these become `[d_model, n_kv × head_dim]` = `[576, 192]`.

   Supporting GQA requires three coordinated changes:

   a. **Format change** — add a `n_kv_heads` field to `att1_model_config`.
      The field is a `uint32_t` appended after `shard_count`, increasing
      `ATT1_MODEL_CONFIG_SIZE` from 36 to 40 bytes.  A new format version
      (v3) is required.  Old v1/v2 models load as full MHA
      (`n_kv_heads = n_heads`).

   b. **Converter change** — read `num_key_value_heads` from `config.json`;
      accept and transpose wk/wv tensors with shape
      `[d_model, n_kv × head_dim]`; write `n_kv_heads` into the new config
      field.

   c. **Runtime attention change** — `src/attention.c` reads `n_kv_heads`
      from `att1_model_config`; for each query head group, reads the
      corresponding shared K/V head.

3. **Tokenizer** — external token IDs via `--tokenizer external` are the
   correct path.  Raw byte IDs 0–255 are in-range for vocab_size=49152 but
   are not semantically meaningful without the BPE tokenizer.

**Pros:** smallest available Apache 2.0 LLaMA-style model; single shard;
~540 MB f32 artifact; standard tensor naming; `model_type="llama"` already
in `SUPPORTED_ARCH`.

**Cons:** GQA requires format change + C runtime change + converter change;
BF16 source requires minor Python extension; 49152-token vocab makes the
output weight alone ~108 MB f32.

**Local requirements (SmolLM2-135M):**

| Step | Peak RAM | Disk |
|------|---------|------|
| Download source safetensors | — | ~270 MB (BF16) |
| Conversion (numpy in-memory) | ~1.5 GB | — |
| f32 `.att1` artifact | — | ~540 MB |
| q8 `.att1` artifact | — | ~135 MB |
| att1-bench inference | ~600 MB | — |

#### Candidate B — SmolLM2-360M

| Config field | Value |
|-------------|-------|
| HF repo | `HuggingFaceTB/SmolLM2-360M` |
| vocab_size | 49152 |
| d_model | 960 |
| d_ff | 2560 |
| n_layers | 32 |
| n_heads | 15 |
| n_kv_heads | 5 |
| head_dim | 64 |
| License | Apache 2.0 |
| Safetensors | Single shard |

Same architecture family as Candidate A; identical blockers (BF16, GQA 3:1).
Produces 1 + 32×9 + 2 = 291 tensors.  Prefer A unless 360 M parameter scale
is specifically needed.

#### Candidate C — Open-LLaMA 3B v2 (no-GQA alternative)

| Config field | Value |
|-------------|-------|
| HF repo | `openlm-research/open_llama_3b_v2` |
| model_type | `llama` |
| vocab_size | 32000 |
| d_model | 3200 |
| d_ff | 8640 |
| n_layers | 26 |
| n_heads | 32 |
| n_kv_heads | 32 (full MHA) |
| head_dim | 100 |
| rope_theta | 10000.0 |
| Source dtype | bfloat16 |
| License | Apache 2.0 |
| Safetensors | Multi-shard (~7 GB BF16 source) |

**Known incompatibilities:**

1. **BF16 source** — shared blocker.
2. **Multi-shard safetensors** — weights are split across multiple shards
   referenced by `model.safetensors.index.json`.  The current scanner, reader,
   and converter handle single-file safetensors only.  Multi-shard support
   requires reading the index file and loading tensors from multiple shards;
   this is a Python-only converter extension, but it is not yet implemented.
3. **Large artifact** — ~13.5 GB f32; requires ~15 GB free RAM to convert and
   ~14 GB free disk for the resulting artifact.  Not suitable for checking in.

**Pros:** Full MHA — zero runtime or format changes once BF16 coercion and
multi-shard reader are added; standard HF LLaMA naming; Apache 2.0.

**Cons:** Multi-shard support not yet implemented; very large artifact;
impractical for CI or portable local testing.

#### Candidate D — Qwen2-0.5B

| Config field | Value |
|-------------|-------|
| HF repo | `Qwen/Qwen2-0.5B` |
| model_type | `qwen2` |
| vocab_size | 151936 |
| d_model | 896 |
| d_ff | 4864 |
| n_layers | 24 |
| n_heads | 14 |
| n_kv_heads | 2 |
| head_dim | 64 |
| rope_theta | 1000000.0 |
| License | Apache 2.0 |
| Safetensors | Single shard |

**Known incompatibilities:**

1. **BF16 source** — shared blocker.
2. **GQA (7:1 ratio)** — same C/format blockers as Candidate A, but a more
   aggressive grouping ratio.
3. **`model_type = "qwen2"`** — not in the current converter's `SUPPORTED_ARCH`
   set; requires adding `"qwen2"` (a trivial Python change) after verifying
   that no undocumented architecture differences exist.
4. **`rope_theta = 1000000.0`** — non-default value; already handled by the
   converter's `rope_theta` config resolution (`--rope-theta` override or
   direct config field reading).
5. **Very large vocabulary (151936)** — the output weight tensor is
   `[896, 151936]` ≈ 137 M elements × 4 bytes ≈ 548 MB f32, larger than all
   other model parameters combined.

Not recommended as first target due to the architecture-type gap, extreme
vocab size, and aggressive GQA ratio.

#### Candidate E — Llama-3.2-1B

| Config field | Value |
|-------------|-------|
| HF repo | `meta-llama/Llama-3.2-1B` |
| model_type | `llama` |
| vocab_size | 128256 |
| d_model | 2048 |
| d_ff | 8192 |
| n_layers | 16 |
| n_heads | 32 |
| n_kv_heads | 8 |
| head_dim | 64 |
| rope_theta | 500000.0 |
| License | Llama 3.2 Community License |
| Safetensors | Single shard |

**Known incompatibilities:**

1. **BF16 source** — shared blocker.
2. **GQA (4:1 ratio)** — same blockers as Candidates A, B, D.
3. **Gated license** — requires accepting Meta's Llama 3.2 Community License
   on HuggingFace Hub.  Permissive for non-commercial research and product
   development under 700 M monthly active users, but less open than
   Apache 2.0.  Adds friction to a fully open reference workflow.
4. **Large vocab (128256)** — output weight ~1 GB f32 alone.

Not recommended as first target; license friction and large size are
unnecessary complications at this stage.

#### Recommended candidate and prerequisite milestone plan

**Primary recommendation: Candidate A (SmolLM2-135M)** — smallest available
model with Apache 2.0 licensing, standard HF LLaMA tensor naming, and a
single-shard safetensors distribution.  It is the most practical target for
exercising the full ATT-1 conversion and validation pipeline on a real public
model.

Prerequisite milestones before conversion:

| Milestone | Goal | Scope |
|-----------|------|-------|
| M65 | Public model acquisition and import instructions | Exact local workflow for acquiring SmolLM2-135M and preflight validation; documentation only |
| M66 | Converter compatibility scanner | Document gaps between current converter and SmolLM2-135M requirements; documentation only |
| M67 | BF16/F16 source dtype coercion | Extend `load_safetensors.py` to upcast BF16/F16 payloads to F32; Python only; validate with a synthetic BF16 fixture |
| M68 | GQA support | Add `n_kv_heads` to `att1_model_config` (format version bump); extend converter; update `src/attention.c` runtime |
| M69 | SmolLM2-135M import and validation | Full pipeline: scan → coerce BF16 → convert f32/q8 → inspect → source comparison → bench smoke |

**Immediate no-GQA-change alternative:** Candidate C (Open-LLaMA-3B-v2)
needs only BF16 coercion (M65, Python) and a multi-shard safetensors reader
extension (Python converter) before conversion.  No C or format changes.  The
practical blocker is size (~15 GB RAM and disk).

#### Validation plan (SmolLM2-135M, post-M66)

1. **Scan:** `python3 compiler/scan_safetensors.py model.safetensors --check-llama --n-layers 30`  
   Expect: `tensor_count=273`, `llama_check: ok`, `llama_missing: 0`.

2. **Read tensors:** `python3 compiler/load_safetensors.py model.safetensors --check-values`  
   BF16 coercion active; all 273 values finite after upcast.

3. **Convert f32:**
   ```sh
   python3 compiler/convert_llama_to_att1.py \
       --safetensors model.safetensors --config config.json \
       --out models/smollm2_135m_f32/model.att1
   ```
   Expect: `wrote ~540 MB`.

4. **Inspect:** `./build/att1-inspect models/smollm2_135m_f32/model.att1`  
   Verify: `vocab_size=49152`, `n_heads=9`, `n_kv_heads=3`, `d_model=576`,
   `tensor_count=273`.

5. **Source comparison:**
   ```sh
   python3 compiler/compare_att1_to_source.py \
       --safetensors model.safetensors --config config.json \
       --att1-f32 models/smollm2_135m_f32/model.att1 \
       --prompt-ids 1,2,3 --report
   ```
   Expect: f32 `max_abs_error ≈ 0`, `forward_match=yes`, `result=pass`.

6. **Convert q8:**
   ```sh
   python3 compiler/convert_llama_to_att1.py \
       --safetensors model.safetensors --config config.json \
       --weight-format q8 \
       --out models/smollm2_135m_q8/model.att1
   ```
   Expect: projection tensors as dtype-2.

7. **Backend matrix smoke:** `att1-bench --tokenizer external --input-token-ids "1,2,3"`
   in single and cluster (tiles=2) modes with cpu-f32 and cpu-q8 backends.

`make test` passes (41 tests) after M64.  No code changes; documentation only.

---

### M65 — public model acquisition and import instructions (complete)

**Goal:** Document the exact local workflow for acquiring `HuggingFaceTB/SmolLM2-135M`
and preparing it for ATT-1 conversion.  Documentation only — no C source changes,
no Makefile changes, no `.att1` format changes, no model weights committed.

#### Selected model

| Property | Value |
|----------|-------|
| HF repo | `HuggingFaceTB/SmolLM2-135M` |
| model_type | `llama` |
| vocab_size | 49152 |
| d_model | 576 |
| d_ff | 1536 |
| n_layers | 30 |
| n_heads | 9 |
| n_kv_heads | 3 (GQA 3:1) |
| head_dim | 64 |
| max_position_embeddings | 2048 |
| rope_theta | 10000.0 |
| Source dtype | bfloat16 |
| Tokenizer | BPE JSON, vocab=49152 |
| Safetensors | Single shard (~270 MB BF16) |
| License | Apache 2.0 |

#### License and private-experimentation caveats

SmolLM2-135M is released under **Apache 2.0**.  No gated access, no
agreement required, no attribution beyond the LICENSE file.  The model
weights are suitable for private experimentation, research, and commercial
use under the terms of Apache 2.0.

Although the license is permissive, the model weights are large binary
assets (~270 MB BF16) and must **not** be committed to the ATT-1 repository.
All weight files reside exclusively in a local directory outside the repo
(see §Local directory layout below).

#### Local directory layout (outside the repo)

All downloaded files live under `~/Models/SmolLM2-135M/` — a path outside
the ATT-1 repo root.

```
~/Models/
  SmolLM2-135M/
    config.json              # model architecture (required)
    model.safetensors        # single-shard BF16 weights (~270 MB, required)
    tokenizer.json           # BPE vocabulary and merge rules (required for scan)
    tokenizer_config.json    # tokenizer class and special-token strings
    special_tokens_map.json  # BOS/EOS/PAD/UNK name → token string
    generation_config.json   # optional; ignored by converter
```

The converter reads `config.json` and `model.safetensors` directly via
explicit `--config` / `--safetensors` flags.  The tokenizer assets live in
the same directory and are read by `scan_tokenizer.py` by directory path.

#### What must not be committed

Never commit `model.safetensors` or any other weight file to the ATT-1 repo.
The `compiler/fixtures/` directory holds only small synthetic safetensors
fixtures (≤ 200 KB each) that are generated by the fixture scripts and are
automatically reproducible — the real public model weights are not.

No `.gitignore` update is required because the canonical model location
(`~/Models/`) is outside the repo.  The existing fixtures at
`compiler/fixtures/*.safetensors` must remain tracked; a blanket
`*.safetensors` exclusion in `.gitignore` would break them.

#### Download options

**Option 1 — `huggingface-cli` (recommended):**

```sh
pip install huggingface_hub   # once; already present if transformers is installed
mkdir -p ~/Models/SmolLM2-135M
huggingface-cli download HuggingFaceTB/SmolLM2-135M \
    --local-dir ~/Models/SmolLM2-135M \
    --include "config.json" \
              "model.safetensors" \
              "tokenizer.json" \
              "tokenizer_config.json" \
              "special_tokens_map.json"
```

This downloads only the listed files, skipping large blobs like PyTorch
`.bin` shards.  The `--include` filter prevents accidental multi-file
downloads if the repo layout changes.

**Option 2 — `git lfs`:**

```sh
brew install git-lfs   # or: apt install git-lfs
git lfs install
mkdir -p ~/Models
cd ~/Models
GIT_LFS_SKIP_SMUDGE=1 git clone \
    https://huggingface.co/HuggingFaceTB/SmolLM2-135M SmolLM2-135M
cd SmolLM2-135M
git lfs pull --include "model.safetensors"
```

`GIT_LFS_SKIP_SMUDGE=1` clones only the pointer file first, then
`git lfs pull --include` fetches only the specified binary.  This avoids
pulling other LFS blobs (e.g., `.bin` shards if present).

**Option 3 — manual download:**

1. Browse `https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/main`.
2. Click each file and use the download icon to save it.
3. Place all files under `~/Models/SmolLM2-135M/`.

Manual download is suitable when neither `huggingface-cli` nor `git` is
available, or for air-gapped transfers.

#### Preflight validation commands

Run these from the ATT-1 repo root after downloading.  All commands are
Python-only and do not modify any files.

**Step 1 — Scan safetensors metadata:**

```sh
python3 compiler/scan_safetensors.py \
    ~/Models/SmolLM2-135M/model.safetensors \
    --check-llama --config ~/Models/SmolLM2-135M/config.json \
    --no-tensors
```

Expected output (values may differ slightly across model revisions):

```
file_size=283068616
header_len=...
data_offset=...
data_size=...
tensor_count=273
scan_errors=0

scan: ok
llama_check: ok
llama_missing: 0
```

Key assertions:
- `tensor_count=273` — 1 embedding + 30×9 layer tensors + 2 output tensors.
- `scan_errors=0` — no offset overlaps, no malformed entries.
- `llama_check: ok` and `llama_missing: 0` — all expected HF LLaMA keys
  present.
- Dtype for all weight tensors will be `BF16` — expected; BF16 coercion is
  a known blocker addressed in M67.

**Step 2 — Scan tokenizer assets:**

```sh
python3 compiler/scan_tokenizer.py \
    ~/Models/SmolLM2-135M \
    --config ~/Models/SmolLM2-135M/config.json
```

Expected output:

```
tokenizer_dir=~/Models/SmolLM2-135M
tokenizer_type=bpe_json
vocab_size=49152
bos_id=1
eos_id=2
pad_id=none
unk_id=none
...
# compatibility
config_vocab_size=49152
vocab_size_match=yes
...
report: ok
```

Key assertions:
- `tokenizer_type=bpe_json` — standard HF tokenizers BPE JSON.
- `vocab_size=49152` — matches `config.json`.
- `vocab_size_match=yes` — tokenizer and config agree.
- `report: ok` — no fatal errors.

**Step 3 — Inspect config fields (manual check):**

```sh
python3 -c "
import json, sys
c = json.load(open('$HOME/Models/SmolLM2-135M/config.json'))
for k in ['model_type','vocab_size','hidden_size','intermediate_size',
          'num_hidden_layers','num_attention_heads',
          'num_key_value_heads','max_position_embeddings','rope_theta']:
    print(f'{k}: {c.get(k)}')" 
```

Expected output:

```
model_type: llama
vocab_size: 49152
hidden_size: 576
intermediate_size: 1536
num_hidden_layers: 30
num_attention_heads: 9
num_key_value_heads: 3
max_position_embeddings: 2048
rope_theta: 10000.0
```

**Note:** There is no `--check-only` or `--dry-run` flag in
`convert_llama_to_att1.py`.  Do not run the converter against the public
model yet — the BF16 blocker (M67) and GQA blocker (M68) must be resolved
first.  The preflight is limited to the scanner and tokenizer scripts.

#### Local disk and RAM expectations

| Operation | Disk delta | Peak RAM |
|-----------|-----------|----------|
| Download source safetensors | +270 MB (BF16) | — |
| Download all tokenizer assets | +~2 MB | — |
| `scan_safetensors.py` (header only) | 0 | < 50 MB |
| `scan_tokenizer.py` | 0 | < 50 MB |
| Conversion (post-M67, post-M68) | +540 MB f32, +135 MB q8 | ~1.5 GB |
| `att1-bench` inference (post-conversion) | 0 | ~600 MB |

Minimum recommended free disk for the full future pipeline: **1 GB** after
download (for f32 + q8 artifacts alongside source).  Minimum recommended
free RAM during conversion: **2 GB**.

#### Failure triage

| Symptom | Likely cause | Resolution |
|---------|-------------|------------|
| `error: file not found: ...model.safetensors` | Download incomplete or wrong path | Re-run download; verify `~/Models/SmolLM2-135M/model.safetensors` exists |
| `scan_errors=1` and `error: ... 'model.safetensors.index.json'` mentioned in header | Multi-shard distribution | Model repo switched to multi-shard layout; download all shards and use shard merger (not yet implemented; fall back to `open_llama_3b_v2` is not viable — see M64 §C) |
| `llama_missing: N` with N > 0 | Tensor naming change in model revision | Check HF commit history; use `--no-tensors` scan to list all present keys |
| `scan_errors=N` with offset errors | File truncated during download | Delete and re-download |
| `model_type` in config.json is not `llama` | Wrong model or architecture variant | Verify repo identity; `qwen2` and other types require converter extension |
| `vocab_size_match=no` | Tokenizer vocab count differs from config | Check config.json `vocab_size` field vs. tokenizer.json vocabulary length; possible model revision mismatch |
| `tokenizer_type=unknown` or `report: 1 error(s)` | Missing tokenizer.json | Re-download tokenizer assets |
| `num_key_value_heads: 3` in config but converter errors on GQA | GQA blocker not yet resolved | GQA support is planned for M68; do not run converter until M68 is complete |
| HTTP 401 or 403 during `huggingface-cli download` | Access-gated model | SmolLM2-135M is not gated; verify you are not accidentally targeting `meta-llama/Llama-3.2-1B` or another gated repo; run `huggingface-cli login` only if required |
| `dtype: BF16` in scan output | Expected for all public models | This is a known blocker; BF16 coercion is planned for M67 |

#### Next milestone

M66 — compatibility scanner (`compiler/check_llama_compat.py`): inspects a
local model directory and produces a pass/fail compat report with a list of
required converter changes.  Validated against the checked-in tiny fixture.

`make test` passes (41 tests) after M65.  No code changes; documentation only.

---

### M66 — public model compatibility scanner (complete)

**Goal:** Add `compiler/check_llama_compat.py` — a converter-side compatibility
scanner that inspects a local model directory and reports whether the model is
ready for ATT-1 LLaMA conversion.  Validated against the checked-in tiny
fixture.  No C source changes, no Makefile changes, no `.att1` format changes,
no model weights committed.

#### New file: `compiler/check_llama_compat.py`

```
Usage:
  python3 compiler/check_llama_compat.py --model-dir PATH
  python3 compiler/check_llama_compat.py --model-dir PATH \\
      --safetensors PATH/model.safetensors
  python3 compiler/check_llama_compat.py --model-dir PATH --no-tensors
  python3 compiler/check_llama_compat.py --model-dir PATH --json

Exit codes:
  0  compat pass (model ready for ATT-1 conversion)
  1  fatal error (dir not found, config missing/unparseable)
  2  compat fail (required converter changes present)
```

The scanner validates:

| Check | Field |
|-------|-------|
| Directory exists | `--model-dir` path |
| `config.json` present and parseable | always |
| `model_type` in `SUPPORTED_ARCH` | `{"llama", "mistral"}` |
| All 6 required fields present | `vocab_size`, `n_layers`, `n_heads`, `d_model`, `d_ff`, `max_seq_len` |
| `d_model % n_heads == 0` | derived validation |
| GQA detected and reported | `num_key_value_heads` ≠ `num_attention_heads` |
| MoE detected and reported | `num_local_experts` etc. |
| `model.safetensors` present | unless `--no-tensors` |
| Safetensors header scans clean | via `scan_safetensors.py` |
| All required LLaMA tensor keys present | via `check_llama_tensors()` |
| Source dtype (F32 passes, BF16/F16 → required change) | per-tensor dtype scan |
| Tokenizer assets in model directory | via `scan_tokenizer_dir()` |
| `vocab_size` cross-check between tokenizer and config | if tokenizer present |

#### Report output (human-readable)

Passing case against the tiny MHA fixture (`compiler/fixtures/m66_compat_fixture`):

```
# ATT-1 LLaMA compatibility scan

# config
config_path               compiler/fixtures/m66_compat_fixture/config.json
model_type                llama
vocab_size                16
n_layers                  2
n_heads                   2
d_model                   8
d_ff                      16
max_seq_len               128
n_kv_heads                absent (MHA assumed)
gqa_detected              no

# safetensors
safetensors_path          compiler/fixtures/tiny_llama_2l.safetensors
tensor_count              21
source_dtype              F32
llama_check               ok
llama_missing             0

# tokenizer
tokenizer_type            bpe_json
vocab_size                16
vocab_size_match          yes
bos_id                    1
eos_id                    2

# artifact size estimates
n_params                  1608
n_tensors_est             21
f32_bytes                 9200
q8_bytes                  5360
f32_size                  9.0 KB
q8_size                   5.2 KB

# required converter changes
required_changes          none

compat: pass
```

Failing case against a SmolLM2-135M-like config (GQA + no tokenizer assets):

```
n_kv_heads                3
gqa_detected              yes
...
# required converter changes
  required_change: GQA: num_key_value_heads=3, num_attention_heads=9 (ratio
    3:1); GQA requires format change (n_kv_heads in att1_model_config),
    converter change, and runtime attention change (M68)

compat: fail
```

#### New fixture: `compiler/fixtures/m66_compat_fixture/`

Combined fixture directory for compat scanner tests:

| File | Source |
|------|--------|
| `config.json` | `tiny_llama_config.json` (vocab=16, d_model=8, n_heads=2, n_layers=2) |
| `tokenizer.json` | `tiny_tokenizer/tokenizer.json` |
| `tokenizer_config.json` | `tiny_tokenizer/tokenizer_config.json` |
| `special_tokens_map.json` | `tiny_tokenizer/special_tokens_map.json` |

The safetensors file is **not** copied — the scanner test passes
`--safetensors compiler/fixtures/tiny_llama_2l.safetensors` explicitly.

#### Test integration

`check_compat_scanner()` in `tests/test_bench_smoke.c` (Python-skippable,
no numpy required) runs four sub-checks:

1. Human-readable pass report: `compat: pass`, all key fields present.
2. JSON pass report: `"compat"`, `"pass"`, `"arch"`, `"llama"`, etc.
3. Missing model-dir: exit ≠1, `compat: error` in output.
4. Missing safetensors (no `--no-tensors`): exit ≠0, `safetensors file not found`.

#### Manual scanner command for SmolLM2-135M (when available)

After downloading to `~/Models/SmolLM2-135M/` (see M65 acquisition guide):

```sh
python3 compiler/check_llama_compat.py \
    --model-dir ~/Models/SmolLM2-135M
```

Expected outcome (before M67/M68 blockers are resolved):

```
gqa_detected              yes
...
# required converter changes
  required_change: GQA: num_key_value_heads=3, num_attention_heads=9 ...
  required_change: source dtype ['BF16']: BF16/F16 coercion to F32 required ...

compat: fail
```

This command is not required by `make test`; it requires a local download
of the model (see M65).

#### Next milestone

M67 — BF16/F16 source dtype coercion in `compiler/load_safetensors.py`:
extend the tensor reader to upcast BF16 and F16 payloads to F32 before
returning values.  Python only; validate with a synthetic BF16 fixture.

`make test` passes (41 tests) after M66.  No C source changes, no Makefile
changes, no `.att1` format changes.

---

## Related Documents

- [real_model_conversion.md](real_model_conversion.md) — existing converter
  plan covering tensor naming, dtype rules, matrix conventions, and milestone
  history through M44
- [shard_metadata.md](shard_metadata.md) — shard metadata binary format and
  validation requirements
- [model_format.md](model_format.md) — `.att1` binary format specification
- [tokenizer_metadata.md](tokenizer_metadata.md) — optional future tokenizer
  metadata schema and validation rules
- [aimu_architecture.md](aimu_architecture.md) — AIMU tiled tensor architecture
