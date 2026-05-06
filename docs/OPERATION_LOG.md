# Operation Log

## Current Objective

Build ATT-1: a C11 programmable tensor-tile simulator for clustered LLM inference, using binary `.att1` model artifacts, CPU reference execution, CUDA live-model validation later, and ATT-1 custom PCIe silicon as the future Phase 3 hardware target.

## Current Milestone

Milestone 16: CUDA softmax prototype.

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

## Active Task

Milestone 15 CUDA RMSNorm prototype is complete and validated on both the
default CPU-only build and the CUDA-enabled build.

Milestone 15 scope:
- `src/backend_cuda.c`: `cuda_backend_rmsnorm_f32` uses cuBLAS primitives only
  (`cublasSdot`, `cublasSdgmm`, `cublasSscal`) so the project keeps the current
  `cc` + cuBLAS build path and does not require `nvcc`.
- `src/backend_cuda.c`: adds a private cuBLAS-backed vector-multiply helper used
  by RMSNorm.
- `tests/test_backend.c`: updated CUDA backend coverage to verify `rmsnorm_f32`
  succeeds and matches the CPU reference within CUDA tolerance.
- `tests/test_cuda_norm.c`: added dedicated five-case CUDA RMSNorm validation
  (tiny deterministic, larger deterministic, shape handling, unavailable path,
  and no silent fallback).
- `docs/cuda_backend.md`: updated kernel-status and CUDA test coverage.

Milestone 15 validation status:
- `make clean && make && make test` passes locally on the default CPU-only build
  (27/27 tests pass, including `cuda_norm`).
- `make clean && make CUDA=1 && make test CUDA=1` passes on a CUDA-capable
  machine.

## Next Prompt for Codex

Run Milestone 16 only.

Goal:
Add CUDA softmax kernel — numerically stable float32 normalization using the
existing backend API and CPU f32 as the correctness reference.

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