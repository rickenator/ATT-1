# Operation Log

## Current Objective

Build ATT-1: a C11 programmable tensor-tile simulator for clustered LLM inference, using binary `.att1` model artifacts, CPU reference execution, CUDA live-model validation later, and ATT-1 custom PCIe silicon as the future Phase 3 hardware target.

## Current Milestone

Milestone 21: CUDA benchmark and trace integration.

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

## Active Task

Milestone 23: CUDA q8xf32 matmul prototype.
- Goal: Add CUDA-backed q8xf32 matmul behind the existing backend API.
- CPU q8xf32 remains the correctness reference.
- Existing per-row int8 quantization format is unchanged and activations remain float32.
- CUDA q8 outputs compare against CPU q8 within tolerance and against CPU f32 where existing q8 tests do so.
- Full CUDA q8 inference and CUDA q4 remain out of scope.
- Non-CUDA builds/devices report CUDA q8 unsupported cleanly.
- Test suite: updated `test_cuda_matmul.c` q8 coverage.

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
