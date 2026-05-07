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
- [tokenizer_metadata.md](tokenizer_metadata.md) — optional future tokenizer
  metadata schema and validation rules
- [aimu_architecture.md](aimu_architecture.md) — AIMU tiled tensor architecture
