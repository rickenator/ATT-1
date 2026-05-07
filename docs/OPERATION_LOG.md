# Operation Log

## Current Objective

Build ATT-1: a C11 programmable tensor-tile simulator for clustered LLM inference, using binary `.att1` model artifacts, CPU reference execution, CUDA live-model validation later, and ATT-1 custom PCIe silicon as the future Phase 3 hardware target. 

## Current Milestone

Milestone 52: tokenizer scanner/parser skeleton.

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

## Active Task

Milestone 51 complete. Await Milestone 52 scope.

## Next Prompt for Codex

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
