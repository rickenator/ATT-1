# Operation Log

## Current Objective

Build ATT-1: a C11 programmable tensor-tile simulator for clustered LLM inference, using binary `.att1` model artifacts, CPU reference execution, CUDA live-model validation later, and ATT-1 custom PCIe silicon as the future Phase 3 hardware target.

## Current Milestone

## Current Milestone

Milestone 12 hardening: CUDA backend skeleton validation.

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

## Active Task

Milestone 12 remains current: CUDA backend skeleton.

Current repo state already includes:
- opt-in CUDA build plumbing in `Makefile` via `CUDA=1`
- `src/backend_cuda.c` and public CUDA backend declarations in `include/att1_backend.h`
- CUDA availability detection plus create/destroy/alloc/free/sync hooks
- host/device copy helpers
- `docs/cuda_backend.md`
- backend and bench smoke coverage for the skeleton path

Milestone 12 scope remains CUDA plumbing only:
- keep normal builds CPU-only by default
- keep CUDA lifecycle and copy hooks available behind the backend API
- keep unsupported/unavailable CUDA paths explicit and testable
- do not add full CUDA transformer inference yet

## Next Prompt for Codex

Continue Milestone 12 only: harden the CUDA backend skeleton without expanding scope.

Goal:
Keep CUDA opt-in and integration-safe while CPU remains the correctness reference.

Requirements:
- Preserve default `make` and `make test` behavior without CUDA.
- Preserve opt-in CUDA builds with `make CUDA=1`.
- Keep `att1_backend_cuda_available`, `att1_backend_cuda_create`, alloc/free/sync, and copy helpers behind the current backend API.
- Keep CLI/backend selection behavior aligned with the repo docs and tests.
- Keep CUDA unavailable paths returning clear unsupported/status results.
- Do not add CUDA operator kernels, cuBLAS, or full transformer inference.
- Do not change CPU f32 or CPU q8 behavior except for shared backend plumbing.
- Run `make clean`, `make`, and `make test`.
- If CUDA is available in the environment, also run `make clean`, `make CUDA=1`, and `make test CUDA=1`.

Before finishing, update `docs/OPERATION_LOG.md`.

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