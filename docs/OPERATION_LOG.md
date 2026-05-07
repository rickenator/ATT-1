# Operation Log

## Current Objective

Build ATT-1: a C11 programmable tensor-tile simulator for clustered LLM inference, using binary `.att1` model artifacts, CPU reference execution, CUDA live-model validation later, and ATT-1 custom PCIe silicon as the future Phase 3 hardware target. 

## Current Milestone

Milestone 62: source comparison report integration.

## Hard Rules

- Runtime is C11.
- Tests are C11.
- Python may generate artifacts under `compiler/`, but must not be required by `make test`.
- `.att1` format must remain versioned and validated.
- Loader must treat binary files as hostile input.
- Default build must not require CUDA.
- CUDA must be opt-in, for example `make CUDA=1`.
- CPU f32 backend remains the correctness reference.
- Do not add Vulkan, Android, OpenCL, or mobile targets.
- Do not implement full CUDA transformer inference until the CUDA skeleton is clean.
- Keep milestones narrow and commit-ready.

## Completed Milestones

- Milestone 0: project skeleton, build, smoke test, and baseline docs.
- Milestone 1: tensor/runtime core object lifetimes and structured status handling.
- Milestone 2: float32 tensor primitives and local transformer math building blocks.
- Milestone 3: paged KV cache and KV-MMU simulator with append validation.
- Milestone 4: packet fabric simulator with bounded queues, barriers, broadcast, and counters.
- Milestone 5: tile runtime threads, command dispatch, and lifecycle coverage.
- Milestone 6: versioned `.att1` model format, hostile-input loader checks, inspect tool, and dummy model generator.
- Milestone 7: single-tile tiny LLM inference from loaded `.att1` model artifacts.
- Milestone 8: multi-tile sharding and cluster inference path.
- Milestone 9: tracing, benchmark tools, and synthetic sizing/reporting flow.
- Milestone 10: q8 quantization primitives and q8xf32 matmul support.
- Milestone 11: backend abstraction with CPU f32 and CPU q8 backends wired through inference.
- Milestone 12: CUDA backend skeleton — opt-in CUDA build path, lifecycle hooks, copy helpers, and stub operator vtable.
- Milestone 13: CUDA validation plan — `docs/CUDA_VALIDATION_PLAN.md` and environment-aware bench smoke test.
- Milestone 14: CUDA f32 matmul prototype — `cublasSgemm` wired into `cuda_backend_matmul_f32`; five-case `test_cuda_matmul` validation suite.
- Milestone 15: CUDA RMSNorm prototype — `cuda_backend_rmsnorm_f32` implemented with cuBLAS primitives and validated on both CPU-only and CUDA-capable test paths.
- Milestone 16: CUDA FFN/SwiGLU prototype — `cuda_backend_ffn_swiglu_f32` implemented and validated against CPU f32 reference with deterministic tiny/medium FFN tests.
- Milestone 17: CUDA RoPE prototype — `cuda_backend_rope_f32` implemented with per-pair `cublasSrot` rotation and validated against CPU f32 RoPE reference.
- Milestone 18: CUDA causal attention — `cuda_backend_softmax_f32` implemented with numerically stable CPU softmax; attention forward pass uses all component CUDA operators (matmul, rope, softmax); comprehensive 7-case test suite validates causal masking, numerical stability, and multi-head determinism.
- Milestone 19: CUDA transformer block prototype — single-block backend path validated on CUDA backend using composed primitives (RMSNorm, causal attention, matmul, SwiGLU) against CPU f32 reference.
- Milestone 20: CUDA single-tile inference integration — existing single-tile decode path now validated with CUDA backend selection and CPU-vs-CUDA equivalence checks on dummy `.att1` model.
- Milestone 28: CUDA q8 cluster inference enabled — removed cpu-only rejection from `run_cluster()`; validated cuda-q8 cluster path matches cpu-q8 last token; new `test_cuda_q8_cluster` and `test_q8_bench` coverage.
- Milestone 29: Backend matrix regression harness — `tests/test_backend_matrix.c` exercises all 8 (backend × mode) combos against the dummy `.att1` model; reports pass/fail, timing, token, and counter info.
- Milestone 30: Real model conversion plan — `docs/real_model_conversion.md` documents tensor naming, dtypes, matrix conventions, q8 rules, and validation ladder; no code changes.
- Milestone 31: ATT-1 converter skeleton — `compiler/convert_llama_to_att1.py` validates a LLaMA-style `config.json`, resolves field aliases, derives `rope_dim`, and prints a planned ATT-1 config; exits cleanly on unsupported arch or missing config; `compiler/fixtures/tiny_llama/` fixture added.
- Milestone 32: Deterministic converter stub output — `compiler/convert_llama_to_att1.py` extended with binary `.att1` emitter; tensor names match `src/model_view.c`; `--out`/`--config` aliases added; stub validated with `att1-inspect` and `att1-bench`; determinism confirmed by SHA256.
- Milestone 33: AIMU tiled tensor architecture document — `docs/aimu_architecture.md` added covering core concept, near-memory execution model, prototype mapping table, conceptual stack, future shard metadata fields, Phase 1/2/3 hardware roadmap, and non-goals; `README.md` updated with AIMU fabric paragraph.
- Milestone 34: ATT-1 shard metadata design — `docs/shard_metadata.md` defining the future `.att1` shard metadata section for AIMU tensor-tile ownership: 13 fields, fixed 120-byte record layout, versioning rules, hostile-input validation requirements. No C source changes.
- Milestone 35: Optional shard metadata C parser skeleton — `include/att1_shard_meta.h`, `src/shard_meta.c`, `att1_model.shard_meta` field, `model_loader.c` overlap check and parser call, `tests/test_shard_meta.c` (8 cases).
- Milestone 36: Deterministic shard metadata fixture generation — `compiler/make_shard_meta_fixture.py`, checked-in `models/shard_meta/model.att1` (14 876 bytes), `att1-inspect` per-record shard output, `tests/test_shard_meta_fixture.c` (4 cases).
- Milestone 37: Shard metadata reporting and trace integration — `att1_shard_meta_summarize()`, `att1-inspect` summary header, `att1-bench` `shard_meta=present/absent`, `tests/test_shard_meta_report.c` (4 cases).
- Milestone 38: Shard metadata consistency validation — `att1_shard_meta_validate()`, `att1-inspect` violations block, `tests/test_shard_meta_consistency.c` (8 cases).
- Milestone 39: Metadata-driven shard plan proposal — `att1_meta_plan_build()`, `att1_meta_plan_compare()`, `att1-inspect` plan block, `tests/test_shard_meta_plan.c` (5 cases). Advisory only; no inference or backend change. 36 tests pass.
- Milestone 40: Opt-in metadata shard plan execution — `att1_shard_plan_mode` enum, `att1_shard_plan_from_meta()`, `shard_plan_mode` field in `att1_cluster_infer_config`, `--shard-plan runtime|metadata` CLI flag in `att1-bench`, `shard_plan=` key=value output, `tests/test_shard_meta_exec.c` (8 cases). No silent fallback. 37 tests pass.
- Milestone 41: Backend matrix validation for shard plans — `tests/test_backend_matrix.c` extended to 16 entries (4 groups: single/dummy, cluster/dummy/runtime, cluster/shard_meta/runtime, cluster/shard_meta/metadata); 3 consistency groups cross-validate runtime vs metadata plans; `att1-bench` single mode now prints `shard_plan=runtime`; `tests/test_bench_smoke.c` checks `shard_plan=runtime` in both single and cluster output; `docs/shard_metadata.md` §13 added. `make test` passes (37 tests; backend_matrix 8/16 passed, 8 skipped on CPU-only build).
- Milestone 42: Converter stub with shard metadata — `compiler/convert_llama_to_att1.py` extended with `--tiles N` and `--shard-meta` flags; `models/converted_stub_meta/model.att1` (2-tile, 21 shard_meta records, layer 0 → tile 0, layer 1 → tile 1) checked in; `tests/test_backend_matrix.c` extended to 24 entries with group 3 cross-validating runtime vs metadata on the 2-tile stub; `docs/real_model_conversion.md` and `docs/shard_metadata.md` §14 updated. `make test` passes (37 tests; backend_matrix 12/24 passed, 12 skipped on CPU-only build).

- Milestone 43: Converter shard metadata plan report — `compiler/convert_llama_to_att1.py` extended with `--report` (human-readable stdout) and `--report-json PATH` flags; `build_shard_plan_report()` and `format_report_text()` added; `tests/test_bench_smoke.c` `check_converter_report()` added (Python-skippable smoke test); `docs/real_model_conversion.md` §Shard plan report added; `docs/shard_metadata.md` §15 added. No `.att1` format change. No C source change. `make test` passes (38 tests).
- Milestone 44: Converter executable metadata plan validation — `tests/test_converter_validation.c` added (inspect + bench consistency on `models/converted_stub_meta/model.att1`; no Python at test time); `compiler/validate_converter_flow.sh` added (full dev pipeline including generation, report, inspect, bench runtime vs metadata, deterministic output comparison); `docs/real_model_conversion.md` §M44 added; `docs/shard_metadata.md` §16 added. No `.att1` format change. No backend change. `make test` passes (39 tests; backend_matrix 12/24 passed, 12 skipped on CPU-only build).
- Milestone 45: Real tiny model import plan — `docs/real_tiny_model_import.md` added (source files, tensor mappings, transpose rules, RoPE conventions, dtype conversion, hostile-input validation, validation ladder, M46–M50 milestone split); `docs/real_model_conversion.md` §M45 cross-reference added; no C source change, no Makefile change, no `.att1` format change. `make test` passes (39 tests).
- Milestone 46: safetensors metadata scanner — `compiler/scan_safetensors.py` added (module + CLI: `scan_safetensors()`, `llama_required_keys()`, `check_llama_tensors()`, `format_scan_report()`; `ScanError` for fatal issues; exit codes 0/1/2; validates file size, header length bounds, UTF-8/JSON parse, tensor `data_offsets[1]` against data region size, non-overlapping regions); `compiler/fixtures/make_tiny_safetensors.py` generator added; `compiler/fixtures/tiny_llama_2l.safetensors` (21 tensors, 8325 bytes, all required LLaMA keys) checked in; `tests/test_bench_smoke.c` `check_scanner()` added (Python-skippable, asserts `tensor_count=21`, `scan: ok`, `llama_check: ok`, `llama_missing: 0`); `docs/real_tiny_model_import.md` §M46 updated. No C source change. No Makefile change. No `.att1` format change. `make test` passes (39 tests).
- Milestone 47: safetensors tensor reader — `compiler/load_safetensors.py` added (module + CLI: `load_tensor()`, `load_all()`, `format_tensor_report()`, `format_summary()`; `LoadError`; `TensorData` namedtuple; F32 payload decode via `struct.unpack` only; `--tensor NAME`, `--expected-dtype`, `--expected-shape`, `--check-values`, `--summary` CLI flags; exit codes 0/1/2); `compiler/test_load_safetensors.py` added (10 error-case checks, outputs `self_test: ok`); `tests/test_bench_smoke.c` `check_tensor_reader()` added (Python-skippable, checks embed/q_proj/gate_proj/norm loads, `check-values` all-finite, self-test); `docs/real_tiny_model_import.md` §M47 updated. No C source change (other than smoke test). No Makefile change. No `.att1` format change. `make test` passes (39 tests).
- Milestone 48: f32 ATT-1 conversion from tiny safetensors fixture — `compiler/convert_llama_to_att1.py` now supports `--safetensors PATH` to load real F32 payloads through the M47 reader, map Hugging Face LLaMA tensor names to ATT-1 names, transpose projection/output matrices, and emit dtype-1 `.att1` data without changing the binary format; tiny config fixtures now match `compiler/fixtures/tiny_llama_2l.safetensors` (`vocab_size=16`, `d_model=8`, `d_ff=16`); checked-in `models/real_tiny_f32/model.att1` loads via `att1-inspect` and runs `att1-bench` single and cluster cpu-f32; CPU and CUDA test suites pass on the RTX 3090 host; backend matrix passes `24/24`; `tests/test_converter_validation.c` validates the converted fixture without Python at test time; tokenizer import and q8/q4 conversion remain deferred.
- Milestone 49: q8 ATT-1 conversion from tiny safetensors fixture — `compiler/convert_llama_to_att1.py` now supports `--weight-format q8` with `--safetensors PATH`, reusing the M48 F32 tensor reader/mapping/transpose path and emitting dtype-2 per-row q8 projection/output tensors while keeping embeddings and norms f32; dtype-2 `.att1` tensor payloads are validated as row-major int8 values followed by per-row float32 scales; checked-in `models/real_tiny_q8/model.att1` loads via the hostile-input C loader and reports q8 metadata in `att1-inspect`; CPU q8 single and cluster bench paths exit zero on the q8 artifact with nonzero fabric packets in cluster mode; CUDA q8 single and cluster bench paths exit zero on the RTX 3090 host; `tests/test_converter_validation.c` validates the f32 and q8 tiny converted fixtures without Python at test time; `make clean && make && make test` passes; `make clean && make CUDA=1 && make test CUDA=1` passes with backend matrix `24/24`; tokenizer import, q4 conversion, BF16/F16 source upcast, and multi-shard real-model import remain pending.
- Milestone 50: tokenizer import plan — documentation-only plan added to `docs/real_tiny_model_import.md` and summarized in `docs/real_model_conversion.md`; current byte-level tokenizer remains the runtime default and real tokenizer import starts converter-side first; expected assets are `tokenizer.json`, future `tokenizer.model`, `tokenizer_config.json`, and `special_tokens_map.json`; plan covers token ID stability, `vocab_size` agreement across config/embeddings/lm_head/tokenizer vocab, BOS/EOS/PAD/UNK handling, byte-fallback behavior, validation risks, and M51-M55 follow-on split. No C source change. No Makefile change. No `.att1` format change. No tokenizer parser implementation. No new dependencies. `make clean && make && make test` passes.
- Milestone 51: tokenizer metadata schema — `docs/tokenizer_metadata.md` added as the optional future tokenizer metadata schema; `docs/real_tiny_model_import.md` and `docs/real_model_conversion.md` cross-reference it. Schema version 1 covers tokenizer type (`byte`, `bpe_json`, `sentencepiece`, `unknown`), `vocab_size`, BOS/EOS/PAD/UNK IDs, byte fallback, normalization and pretokenizer policies, tokenizer asset hash, reserved future embedded asset offset/size fields, compatibility checks against model config and tensor vocab dimensions, hostile-input validation, and runtime fallback rules. Metadata remains optional; existing byte-level tokenizer remains default; no parser, SentencePiece import, dependency, Makefile, C source, `.att1` format, or inference behavior changes. Validation: `make clean && make && make test` passes.
- Milestone 52: tokenizer asset scanner — `compiler/scan_tokenizer.py` added (module + CLI: `scan_tokenizer_dir()`, `format_tokenizer_report()`; `TokenizerScanError`; discovers and parses `tokenizer.json`/`tokenizer_config.json`/`special_tokens_map.json`; detects `tokenizer.model` as unsupported; reports `tokenizer_type`, `vocab_size`, `bos_id`, `eos_id`, `pad_id`, `unk_id`, `byte_fallback`, `normalizer`, `pretokenizer`, SHA-256 per-asset; cross-checks `vocab_size` against `--config`; exit codes 0/1/2); `compiler/fixtures/tiny_tokenizer/tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` (16-token BPE fixture, BOS=1, EOS=2, UNK=0) added; `tests/test_bench_smoke.c` `check_tokenizer_scanner()` added (Python-skippable, asserts `tokenizer_type=bpe_json`, `vocab_size=16`, `bos_id=1`, `eos_id=2`, `scan: ok`, `vocab_size_match=yes`); `docs/real_tiny_model_import.md` §M52 added. No C source change (other than smoke test). No Makefile change. No `.att1` format change. `make test` passes (39 tests).
- Milestone 53: tokenizer fixture import report — `compiler/scan_tokenizer.py` extended with `build_import_report()` (adds `import_ready`, `unsupported_fields`, `canonical_hash` to scan result), `format_import_report()` (sectioned human-readable report), `--report` (stdout import report), `--report-json PATH` (JSON import report to file), `--model-config PATH` (alias for `--config`); `_canonical_asset_hash()` added (SHA-256 over present asset files in canonical order: tokenizer.json, tokenizer.model, tokenizer_config.json, special_tokens_map.json); `tests/test_bench_smoke.c` `check_tokenizer_import_report()` added (Python-skippable: asserts `import_ready=yes`, `canonical_hash=`, `unsupported_fields=none`, `report: ok` on text report; asserts JSON keys on file report; asserts `vocab_size_match=no` and `import_ready=no` on mismatch); `docs/tokenizer_metadata.md` canonical hash section and M53 milestone status added; `docs/real_tiny_model_import.md` §M53 added. No C source change (other than smoke test). No Makefile change. No `.att1` format change. `make test` passes (39 tests).
- Milestone 54: optional `.att1` tokenizer metadata section — `include/att1_tok_meta.h` added (96-byte section wire layout constants, `att1_tok_meta` struct, `att1_tok_meta_parse()` declaration, `att1_tok_type_name()`/`att1_tok_norm_name()`/`att1_tok_pretok_name()` helpers); `src/tok_meta.c` added (full hostile-input validation: size==96, schema_version==1, tokenizer_type in [0,3], vocab_size>0 and matches model config, special token IDs -1 or in [0,vocab_size), byte_fallback 0/1, norm/pretok/hash enums, flags==0, asset_offset/size both-zero-or-both-nonzero); `.att1` version 2 defined (96-byte header, bytes 80–87 `tok_meta_offset` u64, bytes 88–95 `tok_meta_size` u64; version 1 80-byte models remain valid); `include/att1_model.h` extended (`ATT1_MODEL_VERSION_2=2`, `ATT1_MODEL_HEADER_SIZE_V2=96`, `tok_meta` field in `att1_model`); `src/model_loader.c` updated (v1/v2 dispatch on version+header_size, tok_meta range validation, `att1_tok_meta_parse()` call); `tools/att1-inspect.c` updated (`tok_meta: present` with full field report, or `tok_meta: absent`); `compiler/make_tok_meta_fixture.py` added (generates `models/tok_meta/model.att1`); `models/tok_meta/model.att1` (868 bytes, version=2, bpe_json, vocab=16, bos=1, eos=2, pad=-1, unk=0, hash=da38c00aa0b62d2699c46cb2d1ccadce14b36815d784265680b78a7a31ab1034) checked in; `tests/test_tok_meta.c` added (8 cases: absent, valid, bad_version, bad_tok_type, vocab_mismatch, truncated, bad_flags, v1_compat); `docs/tokenizer_metadata.md` §M54 binary layout added and milestone table updated; `docs/real_tiny_model_import.md` §M54 added. No inference behavior change. No C tokenizer parser. `make test` passes (40 tests).
- Milestone 55: runtime tokenizer selection plan — documentation-only. Defines three future runtime tokenizer modes: `byte` (default, always active), `metadata` (uses tok_meta-declared tokenizer, after M57+), `external` (caller-supplied pre-encoded token IDs, after M58+). Specifies future `--tokenizer byte|metadata|external` CLI flag (M56 stub): default is `byte`; `--tokenizer metadata` without a present/valid/supported tok_meta is a hard error with no silent fallback. Documents compatibility checks for metadata mode (vocab_size match, special token ID range, norm/pretok policy support, asset hash match) and failure policy table (absent tok_meta, unsupported type, vocab mismatch, hash mismatch all → exit 1). Testing strategy: byte baseline unchanged; metadata selection tests before implementation; golden prompt-to-ID fixtures before M60. Milestone split: M56 CLI stub → M57 validation path → M58 external mode → M59 BPE parser → M60 tokenizer-aware model validation. No C source change. No Makefile change. No `.att1` format change. `make test` passes (40 tests).
- Milestone 56: tokenizer selection CLI stub — `--tokenizer byte|metadata|external` added to `att1-bench`; `check_tokenizer_mode()` helper validates preconditions and emits clear error messages: `tokenizer metadata absent` when tok_meta is not present, `tokenizer type unsupported: unknown` for unknown type, `metadata tokenizer runtime not implemented yet` when tok_meta is present and type is known, `external tokenizer mode not implemented yet` for external mode; default is `byte`; `tokenizer=<name>` printed in bench output for both single and cluster modes; `tests/test_bench_smoke.c` `check_tokenizer_selection()` added (6 cases: default byte, explicit byte, metadata-absent fail, metadata-notimpl fail, external fail, invalid-mode fail); existing bench output checks updated to assert `tokenizer=byte`; `docs/tokenizer_metadata.md` §M56 added. No `.att1` format change. No backend change. No BPE/SentencePiece parser. `make test` passes (40 tests).
- Milestone 57: metadata tokenizer validation path — `att1_tok_meta_check_runtime()` added to `include/att1_tok_meta.h` and `src/tok_meta.c` (selection-time defense-in-depth: validates present flag, schema_version, tokenizer_type, vocab match against model config, special token ID range, byte_fallback ≤ 1, norm/pretok/hash enums, flags==0, asset half-set); `check_tokenizer_mode()` in `tools/att1-bench.c` updated to call `att1_tok_meta_check_runtime()` and emit differentiated error messages (`tokenizer type unsupported: <name>` for UNSUPPORTED, `tokenizer metadata incompatible with model` for BAD_FORMAT); `tests/test_tok_meta_select.c` added (15 unit-test cases: valid, null, absent, bad_schema_version, unknown_type, bad_type, zero_vocab, vocab_mismatch, bos_out_of_range, unk_out_of_range, absent_ids_ok, bad_byte_fallback, nonzero_flags, sentpiece_type_ok, asset_half_set); `Makefile` TEST_NAMES extended with `tok_meta_select`; `docs/tokenizer_metadata.md` §M57 added and milestone table updated. No `.att1` format change. No backend change. No BPE parser. `make test` passes (40 tests).
- Milestone 58: external tokenizer preprocessing mode — `include/att1_tok_ext.h` and `src/tok_ext.c` added (`att1_tok_ext_parse_ids_str()`, `att1_tok_ext_parse_ids_file()` with per-token range validation against `vocab_size`, stderr diagnostics, and clear error codes); `tools/att1-bench.c` extended with `--input-token-ids` and `--tokens-file` CLI flags, `run_single_external()` and `run_cluster_external()` decode paths, `tokenizer=external` output, and `--prompt` only required for byte/metadata modes; `check_tokenizer_mode()` updated to accept external mode (ID-source validation moved to `main()`); `tests/test_tok_ext.c` added (15 unit-test cases: valid string, single zero, max ID, empty string, null string, malformed alpha, out-of-range, negative, empty segment, trailing comma, null out-param, file valid, file with comments, file not-found, file out-of-range); `tests/test_bench_smoke.c` `check_external_tokenizer()` added (9 integration checks); `Makefile` TEST_NAMES and COMMON_SRCS updated; `docs/tokenizer_metadata.md` §M58 added and milestone tables updated; `docs/OPERATION_LOG.md` updated. No `.att1` format change. No backend change. No BPE parser. `make test` passes (41 tests).
- Milestone 59: local Hugging Face tokenizer helper — `compiler/tokenize_hf.py` added; converts text to pretokenized token IDs using a local Hugging Face tokenizer directory; outputs comma-separated IDs to stdout (for `att1-bench --input-token-ids`) and optionally one-ID-per-line files (for `att1-bench --tokens-file`) and JSON metadata; tries `tokenizers` library first then falls back to `transformers` with `local_files_only=True`; exits 1 on missing path, 2 on missing package (with install guidance), 3 on tokenization error; `tests/test_bench_smoke.c` `check_hf_tokenizer()` added (always tests missing-path error; skips positive-path tests if neither package is installed); `docs/tokenizer_metadata.md` §M59 added and milestone tables updated; `docs/real_tiny_model_import.md` §M59 and §M56–M58 added; no C source change, no Makefile change, no `.att1` format change. `make test` passes (41 tests).
- Milestone 60: converted model validation with pretokenized input — `tests/test_converter_validation.c` extended with `check_real_tiny_pretokenized()`: writes fixture IDs file (1,3,5), runs `--tokenizer external` bench with `--tokens-file` and `--input-token-ids` on `real_tiny_f32` and `real_tiny_q8` for cpu-f32/cpu-q8 single and cluster modes, asserts `tokenizer=external`/`prompt_tokens=3`/`generated_tokens=2`/`fabric_packets_sent>0` (cluster), skips CUDA tests when CUDA unavailable; `tests/test_bench_smoke.c` `check_pretokenized_pipeline()` added (Python-skippable: runs `tokenize_hf.py` on tiny tokenizer fixture then benches both real_tiny models via `--tokens-file`); milestone tables and §M60 sections added to `docs/tokenizer_metadata.md` and `docs/real_tiny_model_import.md`; no C source change, no Makefile change, no `.att1` format change. `make test` passes (41 tests).
- Milestone 61: source-model comparison harness — `compiler/fixtures/make_m61_fixture.py` generates `compiler/fixtures/m61_llama_2l.safetensors` (seeded random weights, seed=61, scale=0.1, vocab_size=16, d_model=8, d_ff=16, n_layers=2, n_heads=2); `models/m61_f32/model.att1` and `models/m61_q8/model.att1` converted from fixture and checked in; `compiler/read_att1.py` added (stdlib-only ATT-1 binary reader, v1/v2, f32 and q8 decode including `dequantize()`); `compiler/compare_att1_to_source.py` added (full comparison harness: static tensor mapping with transpose rules for all 21 LLaMA tensors, max_abs_error report, numpy LLaMA reference forward pass with RMSNorm/RoPE/SwiGLU/causal attention, att1-bench subprocess call for f32 and q8 forward-pass next-token comparison; f32 max_abs_error=0, f32 forward_match=yes, q8 max_abs_error<0.6, q8 forward_match=yes); `tests/test_bench_smoke.c` `check_source_comparison()` added (Python-skippable, numpy-skippable: runs harness and checks `result: pass` and `forward_match: yes`); no C source change, no Makefile change, no `.att1` format change. `make test` passes (41 tests).
- Milestone 62: source comparison report integration — `compiler/compare_att1_to_source.py` extended with `--report` (rich structured text report ending with `report: ok`), `--report-json PATH` (JSON report, hard exit 1 on IO error), `--tokens-file PATH` (load prompt IDs from file, overrides `--prompt-ids`), `--backend {cpu-f32,cpu-q8,cuda,cuda-q8}` (select f32 bench backend); `_compare_static` updated to return 5-tuple including `max_rel_error` and per-tensor list; `_call_att1_bench_generic` replaces old per-backend variants; structured `rpt` dict added (`date`, `safetensors`, `config_path`, `config`, `f32_static`, `q8_static`, `forward`, `result`); `logits_shape` and `max_rel_error` fields added to default key=value output; `tests/test_bench_smoke.c` `check_source_comparison()` extended with `command_fails()` helper and M62 sub-tests (`--report`, `--report-json`, bad path fails); no C source change, no Makefile change, no `.att1` format change. `make test` passes (41 tests).

## Next Prompt for Codex

## Active Task

Milestone 62 complete. Await Milestone 63 scope.

## Known Risks

- CLI drift between docs and tools.
- CUDA accidentally becoming required by default build.
- Backend abstraction leaking CPU-specific assumptions.
- Integer overflow in shape/offset math.
- Tensor bounds validation.
- Hidden Python dependency in tests.
- `.att1` model format drift without versioning.
- Public API ownership and copy hazards.
- Trace counters changing inference behavior.
- q8 path diverging from f32 reference without tolerance tests.
- CUDA skeleton silently growing into partial inference without clear tests.

## Standard Post-Milestone Checklist

- Run `make clean && make && make test`.
- If applicable, run the opt-in CUDA build and tests.
- Run relevant tools for the milestone.
- Inspect `git diff --stat`.
- Inspect `git diff`.
- Commit with a milestone-specific message.
- Update `docs/OPERATION_LOG.md`.

Milestone 29 complete:
- Backend matrix regression harness validates supported dummy-model paths.
- CPU-only build/test remains CUDA-free.
- CUDA=1 build/test passes on RTX 3090 host.
- Supported matrix: cpu-f32, cpu-q8, cuda-f32, cuda-q8 across single and cluster modes.
- q4 and real-model conversion remain unimplemented.

Milestone 28 complete:
- CUDA q8 cluster inference is integrated.
- CPU f32, CUDA f32, CPU q8, and CUDA q8 all work in single and cluster modes for the dummy `.att1` model.
- Default CPU-only build/test remains CUDA-free.
- CUDA=1 build/test passes on RTX 3090 host.
- q4 and real-model conversion are not implemented yet.

Milestone 27 complete:
- CPU q8 cluster inference validated against CPU f32 cluster reference.
- `make test` passes on both CPU-only and RTX 3090 (38 tests).
- Fabric counters remain active in cpu-q8 cluster mode.
- Logits tolerance 0.15 documented and enforced.
- Explicit unsupported behavior for `--backend cuda-q8` cluster mode on CPU-only builds.

Milestone 22 complete:
- CUDA cluster inference works on the RTX 3090 host.
- `make CUDA=1` and `make test CUDA=1` pass outside the Codex sandbox.
- CUDA cluster benchmark exits zero and reports `backend=cuda`.
- Fabric counters remain active in CUDA cluster mode.
- Codex sandbox cannot see `/dev/nvidia*`, so CUDA availability checks may skip inside the sandbox.
- CUDA q8 and real-model conversion remain unimplemented.

Milestone 23 complete:
- CUDA q8xf32 matmul validates against CPU q8xf32.
- Default CPU-only build remains CUDA-free.
- CUDA=1 tests pass on RTX 3090 host.
- Full CUDA q8 inference is still not implemented.
