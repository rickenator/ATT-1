# Operation Log

## Current Objective

Build ATT-1: a C11 programmable tensor-tile simulator for clustered LLM inference, using binary `.att1` model artifacts, CPU reference execution, CUDA live-model validation later, and ATT-1 custom PCIe silicon as the future Phase 3 hardware target.

## Current Milestone

Milestone 13: CUDA validation plan.

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

Create `docs/CUDA_VALIDATION_PLAN.md` for execution on a CUDA-capable machine.

Milestone 12 hardening status (local non-CUDA environment):
- Default CPU build validated with `make clean`, `make`, and `make test`.
- CLI CUDA selection now fails explicitly with unsupported/status messaging when CUDA is unavailable.
- `CUDA=1` build/test path is pending validation on real CUDA hardware.

Milestone 12 validation status:
- Default CPU build/test passes on APEXX and RTX 3090.
- CUDA-unavailable behavior is validated on APEXX.
- CUDA-capable behavior is validated on RTX 3090.
- CUDA backend skeleton remains plumbing only; no CUDA operator kernels or full inference yet.

## Next Prompt for Codex

After Milestone 13 plan completion and successful CUDA-host execution, run Milestone 14 only.

Goal:
Start CUDA matmul/cuBLAS prototyping while preserving current CPU correctness behavior.

Requirements:
- Milestone 14 only: CUDA matmul/cuBLAS prototype.
- Do not implement full transformer inference yet.
- Keep CUDA opt-in (`make CUDA=1`) and non-CUDA default build behavior unchanged.
- Keep CPU f32 as correctness reference and avoid regressions in CPU q8 behavior.

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