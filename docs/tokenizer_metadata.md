# ATT-1 Tokenizer Metadata Schema

Milestone 51 defines the optional tokenizer metadata schema for future
converter-side tokenizer import and runtime tokenizer selection. This is a
documentation-only milestone. No parser is implemented, no C runtime behavior
changes, and the current `.att1` binary model format is not changed.

## Scope

Tokenizer metadata is optional. Existing `.att1` models without tokenizer
metadata remain valid, and the byte-level tokenizer remains the default for
tests and benchmarks.

M51 does not:

- parse `tokenizer.json`
- import SentencePiece
- add external dependencies
- change the Makefile
- require tokenizer metadata for any model
- change inference behavior
- add a tokenizer section to `.att1`

## Future Placement

The schema is defined as a logical record first. A later milestone may store it
as either:

- a sidecar metadata file emitted by converter tooling, or
- an optional `.att1` tokenizer metadata section.

If embedded in `.att1` later, the model header must remain backward compatible:
older models with no tokenizer metadata must load and run exactly as they do
now. The future embedded form must use an explicit offset and size, and the
loader must reject invalid ranges as hostile input.

## Logical Schema

All integer fields are unsigned unless explicitly marked signed. Sentinel token
IDs use `-1` when the source tokenizer does not define the token.

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `schema_version` | `uint32` | yes | Tokenizer metadata schema version. M51 defines version `1`. |
| `tokenizer_type` | enum | yes | Tokenizer implementation family. |
| `vocab_size` | `uint32` | yes | Number of tokenizer IDs; must match model vocabulary. |
| `bos_token_id` | `int32` | yes | BOS token ID, or `-1` if absent. |
| `eos_token_id` | `int32` | yes | EOS token ID, or `-1` if absent. |
| `pad_token_id` | `int32` | yes | PAD token ID, or `-1` if absent. |
| `unk_token_id` | `int32` | yes | UNK token ID, or `-1` if absent. |
| `byte_fallback` | `uint32` boolean | yes | `1` if tokenizer defines byte fallback; otherwise `0`. |
| `normalization_policy` | enum | yes | Text normalization behavior required before tokenization. |
| `pretokenizer_policy` | enum | yes | Pretokenization behavior required before model tokenization. |
| `asset_hash_kind` | enum | yes | Hash algorithm used for tokenizer asset verification. |
| `asset_hash` | fixed bytes | yes | Hash digest of canonical tokenizer asset bytes. |
| `asset_offset` | `uint64` | no | Future embedded tokenizer asset byte offset; `0` when absent. |
| `asset_size` | `uint64` | no | Future embedded tokenizer asset byte size; `0` when absent. |
| `flags` | `uint32` | yes | Reserved feature flags; unknown bits are rejected. |

Version `1` reserves all unassigned enum values and flag bits. Parsers must
reject unknown required values unless a future schema version defines them.

## Tokenizer Type

| Code | Name | Meaning |
|------|------|---------|
| `0` | `unknown` | Unknown or unsupported tokenizer. Metadata may be reported but must not be selected for runtime tokenization. |
| `1` | `byte` | Existing byte-level fallback tokenizer. |
| `2` | `bpe_json` | Hugging Face `tokenizer.json` BPE-style tokenizer. |
| `3` | `sentencepiece` | SentencePiece tokenizer from `tokenizer.model`. |

Runtime fallback behavior depends on this field:

- Missing metadata: use current byte tokenizer.
- `byte`: use current byte tokenizer.
- `bpe_json` or `sentencepiece`: future runtime may use it only when the
  matching tokenizer implementation exists and compatibility checks pass.
- `unknown`: do not silently select it; report unsupported and fall back to
  byte only when the caller did not explicitly request real-tokenizer runtime.

## Normalization Policy

The policy records what text normalization the tokenizer requires.

| Code | Name | Meaning |
|------|------|---------|
| `0` | `unknown` | Unknown or unsupported normalization. |
| `1` | `none` | No normalization before tokenization. |
| `2` | `utf8` | Validate UTF-8 only; do not alter code points. |
| `3` | `nfkc` | Unicode NFKC normalization required. |
| `4` | `custom` | Source tokenizer defines a custom normalizer graph. |

M51 does not implement Unicode normalization. Future runtime selection must
reject unsupported policies rather than producing divergent prompt IDs.

## Pretokenizer Policy

| Code | Name | Meaning |
|------|------|---------|
| `0` | `unknown` | Unknown or unsupported pretokenizer. |
| `1` | `none` | Model tokenizer consumes raw normalized text. |
| `2` | `byte_level` | Byte-level BPE pretokenization. |
| `3` | `whitespace` | Whitespace-oriented pretokenization. |
| `4` | `sentencepiece` | SentencePiece model handles segmentation. |
| `5` | `custom` | Source tokenizer defines a custom pretokenizer graph. |

Future parser work must report the source tokenizer's pretokenizer policy
exactly. Runtime code must reject policies it cannot reproduce.

## Asset Hash

The metadata must include a stable checksum over the canonical tokenizer asset
bytes used by converter tooling.

| Code | Name | Digest bytes |
|------|------|--------------|
| `0` | `none` | 0 |
| `1` | `sha256` | 32 |

Version `1` should use `sha256` for `tokenizer.json` or `tokenizer.model`.
When multiple tokenizer files contribute to metadata, the canonical hash input
is the concatenation of file labels and raw bytes in this order:

```text
tokenizer.json
tokenizer.model
tokenizer_config.json
special_tokens_map.json
```

Missing optional files are omitted from the hash input. The exact canonical
hash procedure must be tested before parser implementation lands.

## Optional Embedded Asset Fields

`asset_offset` and `asset_size` are reserved for a future `.att1` embedded
tokenizer asset section. In M51 they are schema fields only and must be `0` in
any sidecar or draft metadata fixture.

Future embedded validation rules:

- `asset_offset == 0` iff `asset_size == 0`
- nonzero ranges must be inside the file
- `asset_offset + asset_size` must not overflow
- tokenizer asset ranges must not overlap tensor data or shard metadata
- asset bytes must match `asset_hash`

## Compatibility Checks

Before future converter emission succeeds, tokenizer metadata must be
compatible with model weights:

- `metadata.vocab_size == config.vocab_size`
- `metadata.vocab_size == tok_embeddings.weight.shape[0]`
- if `lm_head.weight` is present, its source rows must match
  `metadata.vocab_size`
- every nonnegative special token ID must be `< vocab_size`
- tokenizer type must match the source asset family
- byte fallback must match the source tokenizer declaration

Any mismatch is a hard converter error. Runtime code must also re-check
metadata against loaded model config before selecting a real tokenizer.

## Hostile-Input Validation

Future loaders/parsers must treat tokenizer metadata as hostile input:

- reject unknown `schema_version`
- reject unknown enum values unless explicitly allowed by the schema
- reject reserved flag bits
- reject `vocab_size == 0`
- reject special token IDs `< -1`
- reject special token IDs `>= vocab_size`
- reject invalid embedded asset ranges
- reject integer overflow in all offset/size calculations
- reject hash kind/digest length mismatches
- reject metadata that names a tokenizer type not supported by the runtime when
  explicitly requested
- reject conflicting metadata and model config vocabulary sizes

Validation failure must not change inference behavior for existing byte-token
paths. Explicit real-tokenizer selection must fail clearly and nonzero.

## Runtime Fallback Behavior

The byte tokenizer remains the default.

Future runtime policy:

- no tokenizer metadata: use byte tokenizer
- metadata present but caller does not request real tokenizer: use byte
  tokenizer unless a later milestone changes the default explicitly
- caller requests real tokenizer and metadata is valid/supported: use selected
  tokenizer
- caller requests real tokenizer and metadata is missing/invalid/unsupported:
  fail clearly; do not silently fall back to byte tokenization
- unknown tokenizer type: report unsupported

This preserves current tests while preventing prompt encoding divergence when a
real tokenizer is explicitly requested.

## Future Milestones

| Milestone | Goal |
|-----------|------|
| M52 | Tokenizer scanner/parser skeleton under `compiler/`; inventory tokenizer assets and report this schema. ✅ |
| M53 | Deterministic tokenizer import report (`--report`, `--report-json PATH`, `--model-config PATH`); import readiness, unsupported-field listing, canonical composite asset hash. ✅ |
| M54 | Optional `.att1` tokenizer metadata section; C parser, loader detection, inspect reporting, Python fixture generator. ✅ |
| M55 | Runtime tokenizer selection plan: modes, CLI policy, compatibility checks, failure policy, and milestone split. ✅ |
| M56 | Tokenizer selection CLI stub (`--tokenizer byte|metadata|external`); byte mode only wired; others stub-fail. ✅ |
| M57 | Metadata tokenizer validation path: load, range-check, vocab cross-check; no BPE parser. |
| M58 | External tokenizer preprocessing mode: pipe external process for prompt encoding. |
| M59 | BPE tokenizer parser prototype (if chosen). |
| M60 | Tokenizer-aware converted model validation. |

## M54 Binary Section Layout

M54 embeds the tokenizer metadata schema above as an optional 96-byte section
in `.att1` version 2 files.

### v2 Header Extension

Version 2 models have `version=2` and `header_size=96`.  Bytes 80–95 are two
new `uint64` fields:

| Offset | Size | Field | Notes |
|--------|------|-------|-------|
| 80 | 8 | `tok_meta_offset` | Byte offset of tokenizer metadata section, or `0` |
| 88 | 8 | `tok_meta_size` | Size in bytes of the section (must be 96 when nonzero) |

Version 1 models (`version=1`, `header_size=80`) have no tok_meta fields.
The loader accepts both versions; `tok_meta.present` is `0` for v1 models.

### Tokenizer Metadata Section Wire Layout (96 bytes, all little-endian)

| Offset | Size | Field | Type | Notes |
|--------|------|-------|------|-------|
| 0 | 4 | `schema_version` | `uint32` | Must be `1` |
| 4 | 4 | `tokenizer_type` | `uint32` | `0`=unknown `1`=byte `2`=bpe_json `3`=sentencepiece |
| 8 | 4 | `vocab_size` | `uint32` | >0; must equal model config `vocab_size` |
| 12 | 4 | `bos_token_id` | `int32` | −1=absent |
| 16 | 4 | `eos_token_id` | `int32` | −1=absent |
| 20 | 4 | `pad_token_id` | `int32` | −1=absent |
| 24 | 4 | `unk_token_id` | `int32` | −1=absent |
| 28 | 4 | `byte_fallback` | `uint32` | `0` or `1` |
| 32 | 4 | `normalization_policy` | `uint32` | `0`=unknown `1`=none `2`=utf8 `3`=nfkc `4`=custom |
| 36 | 4 | `pretokenizer_policy` | `uint32` | `0`=unknown `1`=none `2`=byte_level `3`=whitespace `4`=sentencepiece `5`=custom |
| 40 | 4 | `asset_hash_kind` | `uint32` | `0`=none `1`=sha256 |
| 44 | 4 | `flags` | `uint32` | Reserved; must be `0` |
| 48 | 32 | `asset_hash[32]` | `uint8[32]` | SHA-256 digest when `asset_hash_kind=1`; otherwise all zeros |
| 80 | 8 | `asset_offset` | `uint64` | `0` when no embedded asset |
| 88 | 8 | `asset_size` | `uint64` | `0` when no embedded asset |

Validation rules (implemented in `src/tok_meta.c`):
- Section size must equal exactly 96.
- `schema_version` must be `1`.
- `tokenizer_type` must be in `[0, 3]`.
- `vocab_size` must be > 0 and must equal the model config `vocab_size`.
- Each special token ID must be −1 or in `[0, vocab_size)`.
- `byte_fallback` must be `0` or `1`.
- `normalization_policy` must be in `[0, 4]`.
- `pretokenizer_policy` must be in `[0, 5]`.
- `asset_hash_kind` must be `0` or `1`.
- `flags` must be `0`.
- Exactly one of `asset_offset`/`asset_size` may not be zero; both must be zero
  together or both nonzero together.

### Fixture

`models/tok_meta/model.att1` (868 bytes) is the checked-in test fixture.
It is generated by `compiler/make_tok_meta_fixture.py`.

| Section | Offset | Size |
|---------|--------|------|
| header | 0 | 96 |
| config | 96 | 36 |
| tensor descriptors | 132 | 128 |
| tensor data | 260 | 512 |
| tokenizer metadata | 772 | 96 |

Fixture tokenizer metadata: `bpe_json`, `vocab_size=16`, `bos=1`, `eos=2`,
`pad=-1`, `unk=0`, `byte_fallback=0`, `normalization=none`,
`pretokenizer=byte_level`, `asset_hash_kind=sha256`.

## M55 Runtime Tokenizer Selection Plan

M55 is documentation-only.  No C source changes.  No Makefile changes.  No
`.att1` format changes.  No BPE or SentencePiece parser implemented.
The existing byte-level tokenizer remains the default and the only active
runtime path.

### Current Behavior

- The byte-level tokenizer (`att1_tokenizer`) is the sole runtime tokenizer.
- Tokenizer metadata (`tok_meta`) is loaded from v2 `.att1` files and exposed
  via `att1_model.tok_meta`, but it is descriptive only — no runtime selection
  logic reads it.
- `att1-bench` and inference tests always use the byte tokenizer.
- Missing or invalid tokenizer metadata does not affect tokenization.

### Future Runtime Tokenizer Modes

| Mode | Token encoding source | When available |
|------|-----------------------|----------------|
| `byte` | Existing `att1_tokenizer` byte-level encoder | Always (default) |
| `metadata` | Tokenizer declared in `tok_meta`; runtime parser reads embedded or external asset | After M57+ |
| `external` | Caller-supplied pre-encoded token array; ATT-1 runtime skips encoding | After M58+ |

A fourth implicit mode exists: when tokenizer metadata is absent the runtime
falls back to `byte` only.  There is no silent promotion from `byte` to
`metadata` mode.

### CLI Policy

Future `att1-bench` and tool flag (M56 stub):

```
--tokenizer byte        Use byte tokenizer (default; always works)
--tokenizer metadata    Use tokenizer declared in tok_meta (fails clearly if absent or unsupported)
--tokenizer external    Accept pre-encoded token IDs from caller (future; M58)
```

Rules:
- Default is `byte` regardless of whether tok_meta is present.
- `--tokenizer metadata` without tok_meta in the loaded model → hard error, nonzero exit.
- `--tokenizer metadata` with unsupported tokenizer type → hard error, nonzero exit.
- No flag auto-switches the mode based on model content.
- Tools that do not accept `--tokenizer` always use `byte`.

### Compatibility Checks (for future `metadata` mode)

Before a metadata-declared tokenizer may be selected, the runtime must verify:

1. `tok_meta.present == 1` — metadata section is present and parsed.
2. `tok_meta.vocab_size == model.config.vocab_size` — already enforced by loader
   (M54), but must be re-verified at selection time.
3. `tok_meta.tokenizer_type` is a supported value (not `unknown`).
4. All nonnegative special token IDs are `< vocab_size`.
5. `normalization_policy` and `pretokenizer_policy` are supported by the
   runtime's tokenizer implementation.
6. `asset_hash_kind` and `asset_hash` match the asset bytes when an asset is
   present.

Any failed check is a hard error.  There is no partial fallback to byte mode
when `--tokenizer metadata` was explicitly requested.

### Runtime Failure Policy

| Condition | Policy |
|-----------|--------|
| `--tokenizer metadata` + tok_meta absent | Hard error, exit 1, message: `tokenizer metadata absent` |
| `--tokenizer metadata` + unsupported type | Hard error, exit 1, message: `tokenizer type unsupported: <name>` |
| `--tokenizer metadata` + vocab mismatch | Hard error, exit 1, message: `vocab_size mismatch: tok_meta=%u model=%u` |
| `--tokenizer metadata` + bad normalization policy | Hard error, exit 1 |
| `--tokenizer metadata` + asset hash mismatch | Hard error, exit 1, message: `tokenizer asset hash mismatch` |
| `--tokenizer byte` + tok_meta present | OK; tok_meta is ignored for encoding |
| No `--tokenizer` flag | `byte` mode; tok_meta ignored for encoding |

Silent tokenizer switching is never permitted.  If the caller wants byte mode,
they get byte mode; if they want metadata mode, a failed precondition is a
hard error, not a fallback.

### Testing Strategy

- **Byte mode baseline** — all existing tests use byte tokenizer and must
  continue to pass unchanged after any tokenizer-selection code lands.
- **Metadata selection tests (before implementation)** — once the M56 CLI stub
  exists, tests cover: `--tokenizer metadata` fails on v1 model (absent),
  `--tokenizer metadata` fails on unsupported type, `--tokenizer metadata`
  succeeds on valid fixture (fixture only; no real prompt tested until M57).
- **Real tokenizer fixture tests (M57+)** — golden prompt-to-ID fixtures
  validated against the metadata-declared tokenizer before the metadata path
  is considered production-ready.
- **External mode tests (M58+)** — caller-supplied token arrays are accepted
  without encoding; existing byte-tokenizer callers are unaffected.

### Future Milestone Split

| Milestone | Goal |
|-----------|------|
| M56 | Tokenizer selection CLI stub: `--tokenizer byte\|metadata\|external`; byte wired; others stub-fail clearly. ✅ |
| M57 | Metadata tokenizer validation path: selection precondition checks, vocab/hash cross-check; no BPE parser yet. |
| M58 | External tokenizer preprocessing mode: caller provides pre-encoded token IDs. |
| M59 | BPE tokenizer parser prototype (if chosen). |
| M60 | Tokenizer-aware converted model validation: golden prompt-to-ID fixtures, round-trip checks. |

## M56 Tokenizer Selection CLI Stub

M56 adds the `--tokenizer byte|metadata|external` flag to `att1-bench`.
Byte tokenizer remains the only active runtime path.  `metadata` and
`external` validate preconditions then fail with a clear "not implemented
yet" message.

**Changes:**

| File | Change |
|------|--------|
| `tools/att1-bench.c` | Added `--tokenizer byte\|metadata\|external` flag; `check_tokenizer_mode()` helper; `tokenizer=<name>` printed in bench output |
| `tests/test_bench_smoke.c` | Added `check_tokenizer_selection()` (6 cases); asserts `tokenizer=byte` in existing bench output |

**`check_tokenizer_mode()` logic:**

| Mode | Precondition | Result |
|------|-------------|--------|
| `byte` (default) | — | OK; inference runs normally |
| `metadata` | tok_meta absent | Hard error: `tokenizer metadata absent` |
| `metadata` | tok_meta present, type=unknown | Hard error: `tokenizer type unsupported: unknown` |
| `metadata` | tok_meta present, type known | Hard error: `metadata tokenizer runtime not implemented yet` |
| `external` | — | Hard error: `external tokenizer mode not implemented yet` |
| other | — | Usage error; exit 1 |

**Test cases in `check_tokenizer_selection()`:**
1. Default mode prints `tokenizer=byte`.
2. `--tokenizer byte` prints `tokenizer=byte`.
3. `--tokenizer metadata` on v1 model (no tok_meta) → exit nonzero, `tokenizer metadata absent`.
4. `--tokenizer metadata` on v2 tok_meta fixture (bpe_json) → exit nonzero, `not implemented yet`.
5. `--tokenizer external` → exit nonzero, `not implemented yet`.
6. `--tokenizer bogus` → exit nonzero.

`make test` passes (40 tests).  No `.att1` format change.  No backend change.

## Canonical Asset Hash (M53)

The composite hash defined in the _Asset Hash_ section above is implemented by
`_canonical_asset_hash()` in `compiler/scan_tokenizer.py`.  The hash input
concatenates the raw bytes of each present file in this order:

1. `tokenizer.json`
2. `tokenizer.model`
3. `tokenizer_config.json`
4. `special_tokens_map.json`

Absent files are skipped.  The digest is SHA-256 (32 bytes, hex-encoded).
The hash is stable for a given set of present asset files and is reported as
`canonical_hash=<hex>` in the import report and as `"canonical_hash": "<hex>"`
in the JSON report.
