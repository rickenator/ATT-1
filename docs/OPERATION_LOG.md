# Operation Log

## Current Objective

Build ATT-1: a C11 programmable tensor-tile simulator for clustered LLM inference, using binary `.att1` model artifacts, CPU reference execution, CUDA live-model validation later, and ATT-1 custom PCIe silicon as the future Phase 3 hardware target. 

## Current Milestone

Milestone 36: Deterministic shard metadata fixture generation.

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

## Active Task

Milestone 39: Metadata-driven shard plan proposal.
- Added `att1_meta_plan_entry`, `att1_meta_plan`, `att1_meta_plan_diff` types to `include/att1_shard.h`.
- Added `att1_meta_plan_build()`, `att1_meta_plan_free()`, `att1_meta_plan_compare()` to `include/att1_shard.h` and `src/shard.c`.
  - `att1_meta_plan_build()`: derives proposed layer→tile plan from shard metadata tensor names; counts extra (non-layer) records and conflicting tile assignments per layer.
  - `att1_meta_plan_compare()`: compares proposed against runtime `att1_shard_plan`; reports matching, mismatch, missing, extra, conflict.
- Updated `tools/att1-inspect.c` — builds proposed plan + runtime plan, prints `shard_meta_plan_entries/extra/conflict/matching/mismatch/missing` when metadata is present.
- Added `tests/test_shard_meta_plan.c` (5 cases): absent, consistent, missing layer, tile conflict, inspect output.
- No inference or backend behavior changed. No placement enforcement added.
- `make test` passes (36 tests).

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
