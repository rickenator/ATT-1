# Operation Log

## Current Objective

Build ATT-1: a C11 programmable tensor-tile simulator for clustered LLM inference, using binary `.att1` model artifacts, CPU reference execution, CUDA live-model validation later, and ATT-1 custom PCIe silicon as the future Phase 3 hardware target.

## Current Milestone

Milestone 93: AIMU/PCIe prototype requirements document (complete).

## Active Task

Prepare Milestone 94.

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
- Milestone 14: CUDA f32 matmul prototype — `cublasSgemm` wired into `cuda_backend_matmul_f32`; five-case validation suite.
- Milestone 15: CUDA RMSNorm prototype — `cuda_backend_rmsnorm_f32` with cuBLAS primitives; validated on CPU-only and CUDA paths.
- Milestone 16: CUDA FFN/SwiGLU prototype — `cuda_backend_ffn_swiglu_f32` validated against CPU f32 reference.
- Milestone 17: CUDA RoPE prototype — `cuda_backend_rope_f32` with per-pair `cublasSrot`; validated against CPU f32 reference.
- Milestone 18: CUDA causal attention — `cuda_backend_softmax_f32`; attention forward uses all CUDA operators; 7-case test suite.
- Milestone 19: CUDA transformer block prototype — single-block backend path validated using composed CUDA primitives vs CPU f32.
- Milestone 20: CUDA single-tile inference integration — single-tile decode validated with CUDA backend; CPU-vs-CUDA equivalence on dummy model.
- Milestone 22: CUDA cluster inference — `run_cluster()` validated on CUDA backend; bench exits zero with `backend=cuda`; fabric counters active.
- Milestone 23: CUDA q8xf32 matmul — CUDA q8xf32 matmul validated against CPU q8xf32 reference; CPU-only build remains CUDA-free.
- Milestone 27: CPU q8 cluster inference — validated against CPU f32 cluster reference; logits tolerance 0.15 enforced; 38 tests pass.
- Milestone 28: CUDA q8 cluster inference enabled — removed cpu-only rejection from `run_cluster()`; cuda-q8 cluster path validated; `test_cuda_q8_cluster` and `test_q8_bench` added.
- Milestone 29: backend matrix regression harness — `tests/test_backend_matrix.c` covers all 8 (backend × mode) combos against the dummy model.
- Milestone 30: real model conversion plan — `docs/real_model_conversion.md` added; tensor naming, dtypes, matrix conventions, q8 rules, and validation ladder documented.
- Milestone 31: ATT-1 converter skeleton — `compiler/convert_llama_to_att1.py` validates LLaMA `config.json` and prints planned ATT-1 config; `compiler/fixtures/tiny_llama/` fixture added.
- Milestone 32: deterministic converter stub output — binary `.att1` emitter; tensor names match `src/model_view.c`; determinism confirmed by SHA256.
- Milestone 33: AIMU tiled tensor architecture document — `docs/aimu_architecture.md` added; `README.md` updated with AIMU fabric paragraph.
- Milestone 34: ATT-1 shard metadata design — `docs/shard_metadata.md` added; 13 fields, fixed 120-byte record layout, versioning and validation rules.
- Milestone 35: optional shard metadata C parser skeleton — `include/att1_shard_meta.h`, `src/shard_meta.c`, model loader overlap check; `tests/test_shard_meta.c` (8 cases).
- Milestone 36: deterministic shard metadata fixture — `compiler/make_shard_meta_fixture.py`; `models/shard_meta/model.att1` checked in; `att1-inspect` per-record output; 4 fixture tests.
- Milestone 37: shard metadata reporting and trace integration — `att1_shard_meta_summarize()`; `att1-inspect` summary header; `att1-bench` `shard_meta=present/absent`; 4 report tests.
- Milestone 38: shard metadata consistency validation — `att1_shard_meta_validate()`; `att1-inspect` violations block; 8 consistency tests.
- Milestone 39: metadata-driven shard plan proposal — `att1_meta_plan_build()`, `att1_meta_plan_compare()`; `att1-inspect` plan block; 5 plan tests. Advisory only.
- Milestone 40: opt-in metadata shard plan execution — `att1_shard_plan_mode` enum, `att1_shard_plan_from_meta()`; `--shard-plan runtime|metadata` CLI flag; `shard_plan=` output; 8 exec tests. No silent fallback.
- Milestone 41: backend matrix validation for shard plans — `tests/test_backend_matrix.c` extended to 16 entries (4 groups); 3 consistency cross-validation groups.
- Milestone 42: converter stub with shard metadata — `--tiles N` and `--shard-meta` converter flags; `models/converted_stub_meta/model.att1` (2-tile, 21 records) checked in; backend matrix extended to 24 entries.
- Milestone 43: converter shard metadata plan report — `--report` and `--report-json` converter flags; `build_shard_plan_report()` added; Python-skippable smoke test; 38 tests pass.
- Milestone 44: converter executable metadata plan validation — `tests/test_converter_validation.c` added (inspect + bench consistency, no Python at test time); `compiler/validate_converter_flow.sh` added.
- Milestone 45: real tiny model import plan — `docs/real_tiny_model_import.md` added (tensor mappings, transpose rules, RoPE, dtype conversion, validation ladder, M46–M50 split).
- Milestone 46: safetensors metadata scanner — `compiler/scan_safetensors.py` added; `compiler/fixtures/tiny_llama_2l.safetensors` (21 tensors) checked in; Python-skippable smoke test.
- Milestone 47: safetensors tensor reader — `compiler/load_safetensors.py` added; `compiler/test_load_safetensors.py` (10 error-case checks); tensor reader smoke test.
- Milestone 48: f32 ATT-1 conversion from tiny safetensors — `--safetensors PATH` converter flag; `models/real_tiny_f32/model.att1` checked in; converter validation test updated.
- Milestone 49: q8 ATT-1 conversion from tiny safetensors — `--weight-format q8` converter flag; `models/real_tiny_q8/model.att1` checked in; CUDA=1 backend matrix 24/24.
- Milestone 50: tokenizer import plan — documentation-only plan in `docs/real_tiny_model_import.md` and `docs/real_model_conversion.md`; M51–M55 split defined.
- Milestone 51: tokenizer metadata schema — `docs/tokenizer_metadata.md` added; schema version 1 covering tokenizer type, vocab_size, special token IDs, normalization, asset hash, runtime fallback rules.
- Milestone 52: tokenizer asset scanner — `compiler/scan_tokenizer.py` added; `compiler/fixtures/tiny_tokenizer/` (16-token BPE) added; Python-skippable smoke test.
- Milestone 53: tokenizer fixture import report — `build_import_report()`, `--report`, `--report-json`, `--model-config` added to `scan_tokenizer.py`; import report smoke test.
- Milestone 54: optional `.att1` tokenizer metadata section — `include/att1_tok_meta.h`, `src/tok_meta.c`; `.att1` version 2 defined (96-byte header); `models/tok_meta/model.att1` checked in; `tests/test_tok_meta.c` (8 cases).
- Milestone 55: runtime tokenizer selection plan — documentation-only; three modes defined (`byte`, `metadata`, `external`); M56–M60 split; hard-error policy for unsupported tok_meta.
- Milestone 56: tokenizer selection CLI stub — `--tokenizer byte|metadata|external` added to `att1-bench`; `tokenizer=<name>` in bench output; 6 selection smoke tests.
- Milestone 57: metadata tokenizer validation path — `att1_tok_meta_check_runtime()` added; differentiated error messages in `check_tokenizer_mode()`; `tests/test_tok_meta_select.c` (15 cases).
- Milestone 58: external tokenizer preprocessing mode — `include/att1_tok_ext.h`, `src/tok_ext.c`; `--input-token-ids` and `--tokens-file` CLI flags; `tokenizer=external` bench output; `tests/test_tok_ext.c` (15 cases).
- Milestone 59: local Hugging Face tokenizer helper — `compiler/tokenize_hf.py` added; outputs comma-separated IDs for `att1-bench --input-token-ids`; HF-package-skippable smoke test.
- Milestone 60: converted model validation with pretokenized input — `test_converter_validation.c` extended with external-tokenizer bench round-trip; pretokenized pipeline smoke test.
- Milestone 61: source-model comparison harness — `compiler/read_att1.py` and `compiler/compare_att1_to_source.py` added; `models/m61_f32/` and `models/m61_q8/` checked in; f32 max_abs_error=0, q8 max_abs_error<0.6.
- Milestone 62: source comparison report integration — `--report`, `--report-json`, `--tokens-file`, `--backend` flags added to `compare_att1_to_source.py`; structured `rpt` dict; M62 smoke sub-tests.
- Milestone 63: larger tiny-model fixture — `compiler/fixtures/m63_llama_2l.safetensors` (vocab=64, d_model=32); `models/m63_f32/` and `models/m63_q8/` checked in; all four bench modes validated.
- Milestone 64: public small-model import candidate selection — `HuggingFaceTB/SmolLM2-135M` recommended; BF16 blocker and GQA requirements documented; M65–M69 prereq plan defined.
- Milestone 65: public model acquisition and import instructions — SmolLM2-135M download options, directory layout, preflight validation commands, and failure triage documented.
- Milestone 66: public model compatibility scanner — `compiler/check_llama_compat.py` added; `compiler/fixtures/m66_compat_fixture/` added; compat report smoke test (4 checks).
- Milestone 67: BF16/F16 source dtype coercion — `_coerce_bf16()` and `_coerce_f16()` added to `load_safetensors.py`; `compiler/fixtures/m67_bf16_llama_2l.safetensors` checked in; BF16 coercion smoke test; 42 tests pass.
- Milestone 68: q8 conversion of BF16-source public model — `compare_att1_to_source.py` extended to coerce BF16/F16 source tensors; q8 public-model workflow in `docs/quantization.md`; q8 conversion smoke test (5 checks).
- Milestone 69: public model source comparison report — `compare_att1_to_source.py` accepts `--model-dir`; GQA-shaped K/V tensors supported in Python reference; source comparison report documented.
- Milestone 70: public model backend smoke validation — `compiler/validate_public_backends.py` added; CPU f32/q8 single and cluster paths validated; CUDA rows report `unsupported`; no public artifacts committed.
- Milestone 71: public model end-to-end tokenized validation — `compiler/validate_public_tokenized.py` added; `tokenize_hf.py` + pretokenized bench path combined; HF-package-skippable smoke test.
- Milestone 72: larger-model scaling and placement report — `tools/att1-size.c` extended with `--config`, manual shape, `--json` modes; per-category storage breakdown, KV-cache table, AIMU tile placement, backend feasibility notes; scaling report smoke test (6 checks).
- Milestone 73: q4 quantization planning — documentation-only; grouped int4 format spec in `docs/quantization.md`; M74–M79 implementation split defined.
- Milestone 74: q4 format and schema — `ATT1_MODEL_DTYPE_Q4=3` added; q4 wire-format constants; hostile-input validation in loader; q4 tensor reporting in `att1-inspect`; `tests/test_quant_q4.c` (9 cases).
- Milestone 75: CPU q4 packing and unpacking primitives — five per-group helpers in `src/quant.c`; nibble convention, signed int4 [-7,7], saturation, non-finite rejection; `tests/test_quant_q4_pack.c` (9 cases).
- Milestone 76: CPU q4 matmul prototype — `att1_q4_matrix` struct, `att1_quantize_q4_per_group()`, `att1_matmul_q4xf32()`; dequantize-then-multiply; Q4_TOLERANCE=0.35f; `tests/test_matmul_q4.c` (8 cases).
- Milestone 77: q4 fixture generation and validation — `compiler/make_q4_fixture.py`; `models/q4_tiny/model.att1` checked in; `att1-inspect` `inference_status=q4_unsupported`; `tests/test_quant_q4_fixture.c` (8 cases).
- Milestone 78: CPU q4 single-tile inference integration — `att1_infer_create_q4()`, `cpu-q4` backend, zero-copy q4 weight views, q4 attention and transformer block forward; `tests/test_infer_q4.c` (7 cases).
- Milestone 79: CPU q4 cluster inference integration — `att1_cluster_infer_create_q4()`; `cluster_prepare_q4()`; `use_q4` branch in decode loop; metadata plan returns `ATT1_ERR_UNSUPPORTED`; `tests/test_cluster_infer_q4.c` (8 cases).
- Milestone 80: q4 bench and backend matrix integration — `att1-bench --backend cpu-q4` single and cluster validated; `--backend cuda-q4` hard-rejected; `test_q4_bench` (4 checks); backend matrix extended with cpu-q4 consistency group.
- Milestone 81: public-model q4 converter — `compiler/convert_llama_to_att1.py` extended with `--weight-format q4`; `models/real_tiny_q4/model.att1` checked in; `test_converter_validation.c` extended with q4 round-trip.
- Milestone 82: public-model q4 conversion plan — documentation-only; 211 eligible q4 tensors for SmolLM2-135M; size estimates; tolerance targets; M83–M85 split defined.
- Milestone 83: public-model q4 conversion path — `--model-dir` auto-discovery in converter; `compiler/validate_public_q4.py` added; `compiler/fixtures/m83_model_dir/` fixture; q4 public model smoke test.
- Milestone 84: public-model q4 source comparison report — `--att1-q4` and `--q4-tol` flags in `compare_att1_to_source.py`; q4 decode support in `compiler/read_att1.py`; q4 source comparison smoke test.
- Milestone 85: public-model q4 backend smoke validation — `compiler/validate_public_q4_backends.py` added; cpu-q4 single/cluster validated; cuda-q4 explicitly unsupported; q4 backend smoke test (3 parts).
- Milestone 86: CUDA q4 implementation plan — documentation-only; Option A/B/C design choices in `docs/quantization.md`; M87–M90 milestone split in `docs/cuda_backend.md`; no C or Makefile change.
- Milestone 87: CUDA q4 matmul prototype — `cuda_backend_matmul_q4xf32()` (dequantize-on-CPU then cuBLAS); `cuda_q4_backend_ops` ("cuda-q4"); `att1_backend_cuda_q4_create()`; `matmul_q4xf32` slot in `att1_backend_ops`; all CPU backends updated to NULL; `tests/test_cuda_matmul_q4.c` (8 cases); 46 tests pass.
- Milestone 88: CUDA q4 single-tile inference — `cuda_q4_backend_ops` populated with full inference ops (rmsnorm, softmax, rope, ffn_swiglu); `att1_attention_forward_backend_q4` and `att1_transformer_block_forward_backend_q4` route matmuls through backend vtable via `attention_q4_matmul`/`block_q4_matmul` helpers; `infer_backend_is_q4` accepts "cuda-q4"; output projection in `att1_infer_decode_token` routed via `infer_matmul_q4`; `att1-bench` cuda-q4 single mode enabled (cluster remains rejected); `test_q4_bench` updated for CUDA-conditional cuda-q4 check; `tests/test_cuda_infer_q4.c` (3 cases: no-fallback, logits match, generated tokens); 48 tests pass.
- Milestone 89: CUDA q4 cluster inference — `cluster_backend_is_q4()` extended to accept "cuda-q4"; `cluster_matmul_q4()` static helper routes output projection through backend vtable when supported; cuda-q4 cluster rejection removed from `att1-bench` `run_cluster`/`run_cluster_external`; cuda-q4 backend swap wired after `att1_cluster_infer_create_q4`; `tests/test_cuda_cluster_infer_q4.c` (4 cases: no-fallback, fabric counters, logits match, generated tokens); `test_q4_bench` extended with cuda-q4 cluster status check; 53 tests pass (7 CUDA-skipped on CPU-only host).
- Milestone 90: CUDA q4 benchmark and backend-matrix integration — `test_backend_matrix` extended with cuda-q4 single and cluster runtime entries (group 4, q4_tiny fixture, CUDA-conditional); `validate_public_q4_backends.py` M85 smoke assertion relaxed from `status=unsupported` to `result: pass`; 28-entry matrix; M85 CUDA q4 policy note updated; CPU-only host: 14/28 passed, 14 skipped, 0 failed; CUDA host: 28/28 passed, 0 skipped, 0 failed (verified).
- Milestone 91: CUDA q4 public-model validation report — `compiler/validate_public_q4_cuda.py` added; 5-case validation matrix (cpu-q4 × single/cluster/metadata, cuda-q4 × single/cluster); plan_unsupported and unavailable are non-failure outcomes; backend silent-fallback check; fabric-packet nonzero check; q4 notes embedded; JSON report; `check_public_q4_cuda_smoke` added to `test_bench_smoke.c`; CPU-only: bench smoke passes; CUDA host: all cuda-q4 rows pass (verified).
- Milestone 92: Backend comparison report — `compiler/backend_comparison_report.py` added; 12-case matrix (f32/q8/q4 × CPU/CUDA × single/cluster); pending/unavailable/pass CUDA status; backend silent-fallback check; fabric-packet nonzero check; q4 notes; JSON report; `check_backend_comparison_smoke` added to `test_bench_smoke.c`; CPU-only: all 6 CPU rows pass, 6 CUDA rows pending (no --include-cuda); CUDA host: all 12 rows pass (verified on RTX 3090).
- Milestone 93: AIMU/PCIe prototype requirements — Section 8 added to `docs/aimu_architecture.md`; covers prototype goal, proof criteria, runtime/CUDA relationships, tile responsibilities, local memory sizing (f32/q8/q4), host control plane, fabric/interconnect requirements, KV-MMU requirements, counter/trace requirements, four MVP options (software-emulated/FPGA/PCIe card/ASIC), data movement assumptions, dtype tolerances, non-goals, open engineering questions, and M94 proposal.

## Next Prompt for Codex

Implement Milestone 94 only.
- Default make/make test must remain CPU-only and CUDA-free.
- CUDA remains opt-in with make CUDA=1.
- Do not implement CUDA q4 cluster inference yet.
- Do not change .att1 binary format.
- Do not change CPU q4 behavior.
- Do not change f32/q8/cuda-f32/cuda-q8 behavior.
- `att1_infer_create_q4()` must accept a "cuda-q4" backend.
- `att1-bench --backend cuda-q4 --mode single` must exit zero.
- CUDA q4 logits must match CPU q4 within Q4_TOLERANCE=0.35f.
- Update docs/quantization.md.
- Update docs/cuda_backend.md.
- Update docs/OPERATION_LOG.md.
- Run make clean && make && make test.
- On CUDA host, run make clean && make CUDA=1 && make test CUDA=1.

## Known Risks

- OPERATION_LOG ordering drift.
- CLI/docs drift.
- CUDA accidentally required by default build.
- Silent backend fallback.
- q4 tolerance/token divergence not documented.
- q4 metadata/packing mismatch.
- CUDA q4 kernel mismatch vs CPU q4.
- Hostile-input validation gaps.
- Public model artifacts accidentally committed.
- Python cache artifacts accidentally tracked.
- Tokenizer mismatch / token ID drift.
- ATT-1 format drift without versioning.

## Standard Post-Milestone Checklist

- `make clean && make && make test`
- `make clean && make CUDA=1 && make test CUDA=1` when CUDA touched. CUDA validation rule:
For milestones touching CUDA code, CUDA benchmarks, CUDA reports, or CUDA backend behavior, the milestone is not complete until Rick manually validates on the RTX 3090 host with:

make clean && make CUDA=1 && make test CUDA=1

and any milestone-specific CUDA smoke commands.

Until that signoff, the milestone status is:
"CPU-validated; CUDA signoff pending."
- `git diff --stat`
- `git diff`
- `git ls-files` cache check:
  `git ls-files | grep -E '(__pycache__|\.pyc$|\.pyo$)' || echo "No tracked Python cache artifacts"`
- Commit with a milestone-specific message.
- Tag the milestone.
- Push.
- Update `docs/OPERATION_LOG.md`.
