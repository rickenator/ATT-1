# ATT-1 Release Candidate Checkpoint — M150

**Milestone**: M150  
**Date**: 2026-05-09  
**Commit**: afdc02b (`milestone-149-docs-lint-checker`)  
**Branch**: `master`  
**Decision**: [See §10](#10-review-decision)

---

## 1. Release Candidate Summary

| Field | Value |
|-------|-------|
| Project | ATT-1 — Attention Tile Transformer runtime and simulator |
| Milestone | M150 release candidate checkpoint |
| Package type | Reviewable software/runtime/simulator repository |
| Runtime language | C11 (`-std=c11 -Wall -Wextra -Wpedantic -Werror -pthread`) |
| Build system | GNU Make; CPU-only by default; `make CUDA=1` opt-in |
| Test baseline | 781 PASS 0 FAIL (C unit and integration tests) |
| Regression baseline | ALL STEPS PASSED (9 steps) |
| CI | CPU-only GitHub Actions (ubuntu-latest) |
| CUDA status | Manual signoff on RTX 3090; not validated by CI |
| Hardware | None — software simulator only |
| Patent disclosure | None — engineering architecture language only |
| Public model deployment | None — no public model weights in repository |
| Network access required | No |
| Public model downloads required | No |

This checkpoint marks ATT-1 as ready for **limited outside technical review**
of the software, simulator, documentation, and validation infrastructure.
It is not a production release, a hardware tape-out, or a patent disclosure.

---

## 2. Validation Commands and Baselines

Run the following commands on a clean checkout. All must exit 0.

```sh
# 1. CPU-only build and test (required)
make clean && make && make test

# 2. Full regression suite (9 steps)
make regression

# 3. Documentation lint and link checker
make docs-check

# 4. ASAN build and test
make clean && make test-asan

# 5. UBSAN build and test
make clean && make test-ubsan

# 6. Fuzz and loader smoke tests
make fuzz-smoke

# 7. End-to-end tiny fixture demo
./tools/demo_tiny_att1.sh

# 8. Python cache artifact check
git ls-files | grep -E '(__pycache__|\.pyc$|\.pyo$)' || echo "No tracked Python cache artifacts"
```

### Confirmed baselines at afdc02b

| Command | Expected result | M150 result |
|---------|----------------|-------------|
| `make clean && make` | Build succeeds, 0 errors | PASS |
| `make test` | 781 PASS 0 FAIL | 781 PASS 0 FAIL |
| `make regression` | ALL STEPS PASSED (9 steps) | ALL STEPS PASSED |
| `make docs-check` | PASS (0 errors) | PASS (0 errors) |
| `make test-asan` | 781 PASS 0 FAIL (ASAN) | (run locally before review) |
| `make test-ubsan` | 781 PASS 0 FAIL (UBSAN) | (run locally before review) |
| `make fuzz-smoke` | fuzz_loader 17/17; fuzz_json 40/40 | 17/17 PASS; 40/40 PASS |
| `./tools/demo_tiny_att1.sh` | 14 PASS 0 FAIL; `ATT-1 tiny demo: PASS` | 14 PASS 0 FAIL |
| Cache artifact check | No tracked Python cache artifacts | No tracked Python cache artifacts |

---

## 3. CI Status

| Workflow | Runner | Scope | Status |
|----------|--------|-------|--------|
| `.github/workflows/ci.yml` | `ubuntu-latest` (GitHub Actions) | CPU-only build, test, regression, ASAN, UBSAN, fuzz-smoke, demo, cache check, docs-check | Active; must be green on `master` |
| `.github/workflows/cuda-self-hosted.example.yml` | Self-hosted RTX 3090 | CUDA build + test | Inactive example; not run in CI |

**CI does not validate CUDA kernels, download public model weights, or require
any public artifacts.** The CPU CI gate is a necessary but not sufficient
condition for a CUDA-touching change.

---

## 4. CUDA Status

| Item | Status |
|------|--------|
| CUDA backend implemented | Yes (`src/backend_cuda.c`, `make CUDA=1`) |
| CUDA CI runner | Not active (manual only) |
| CUDA signoff host | RTX 3090-class, local |
| CUDA signoff for this checkpoint | Not performed — CPU CI only |
| CUDA labeled in docs | Yes — every CUDA-touching milestone entry is labeled "manual signoff pending" or "validated on RTX 3090" |

Any CUDA-touching change must complete manual signoff before that change is
considered validated. CPU CI green does not imply CUDA correctness.

Manual CUDA signoff command sequence:

```sh
make clean && make CUDA=1 && make test CUDA=1
python3 compiler/run_full_regression.py --cuda
```

See [CUDA_VALIDATION_PLAN.md](CUDA_VALIDATION_PLAN.md) for full policy.

---

## 5. Capability Summary

### 5.1 Runtime and inference

| Capability | Status |
|------------|--------|
| `.att1` binary format (magic, version, section fields) | Implemented |
| Hostile `.att1` loader validation (bounds, magic, version) | Implemented — 17 hostile fixture tests |
| CPU f32 single-tile inference | Implemented |
| CPU q8 quantized inference | Implemented |
| CPU q4 quantized inference | Implemented |
| CUDA f32 inference | Implemented (manual RTX 3090 signoff required) |
| CUDA q8 inference | Implemented (manual RTX 3090 signoff required) |
| CUDA q4 inference | Implemented (manual RTX 3090 signoff required) |
| Single-tile inference pipeline | Implemented |
| Multi-tile cluster inference (simulated fabric) | Implemented |
| Source text inference (`--prompt`) | Implemented |
| Pre-tokenized input (`--input-token-ids`) | Implemented |

### 5.2 Model tooling

| Capability | Status |
|------------|--------|
| Real tiny-model conversion path | Implemented (`compiler/make_dummy_model.py`) |
| Public-model conversion example path | Documented; weights stay external |
| Tensor placement reports (advisory) | Implemented (M98/M99) |
| Placement advisory and scenario comparison | Implemented |

### 5.3 AIMU control-plane simulator

| Capability | Status |
|------------|--------|
| AIMU command queue simulation | Implemented |
| AIMU userspace MMIO emulator | Implemented |
| AIMU device/DMA/host simulation | Implemented |
| Command and fabric replay pipeline | Implemented (M132) |
| Execution-plan validation and replay | Implemented |
| Tensor-level execution (advisory/planned output) | Simulated — no real hardware |

### 5.4 Validation and regression coverage

| Capability | Status |
|------------|--------|
| Golden regression baselines | Implemented (M133) |
| Schema compatibility regression | Implemented (M134) |
| Hostile-input regression suite (37 negative fixtures) | Implemented (M135) |
| Local full-regression runner (`make regression`) | Implemented (M136) — 9 steps |
| CPU-only GitHub Actions CI | Implemented (M137) |
| CUDA signoff plan and self-hosted runner example | Implemented (M138) |
| Fuzz and loader smoke tests | Implemented (`make fuzz-smoke`) |
| Documentation lint and link checker | Implemented (M149) |

### 5.5 Documentation coverage

| Capability | Status |
|------------|--------|
| End-to-end tiny fixture demo (`demo_tiny_att1.sh`) | Implemented (M140) — 14 PASS |
| ATT-1 Reference Manual | Complete (M146) |
| AIMU Intrinsics and Operations Reference Manual | Complete (M147) |
| Documentation index | Complete (`docs/INDEX.md`) |
| Release readiness checklist | Current (`docs/RELEASE_READINESS.md`) |
| External review package checklist | Current (`docs/EXTERNAL_REVIEW_PACKAGE.md`) |
| CUDA validation policy | Current (`docs/CUDA_VALIDATION_PLAN.md`) |
| Milestone operation log (M0–M150) | Current (`docs/OPERATION_LOG.md`) |

---

## 6. Documentation Status

| Document | State | Notes |
|----------|-------|-------|
| [README.md](../README.md) | Current | Build, test, quick-start, docs map |
| [docs/INDEX.md](INDEX.md) | Current | Full document index |
| [docs/ATT1_REFERENCE_MANUAL.md](ATT1_REFERENCE_MANUAL.md) | Complete (M146) | C API, .att1 format, runtime semantics |
| [docs/AIMU_INTRINSICS_OPERATIONS_REFERENCE.md](AIMU_INTRINSICS_OPERATIONS_REFERENCE.md) | Complete (M147) | AIMU command set, MMIO, DMA, fabric |
| [docs/RELEASE_READINESS.md](RELEASE_READINESS.md) | Current (M150) | Pre-release gate checklist |
| [docs/EXTERNAL_REVIEW_PACKAGE.md](EXTERNAL_REVIEW_PACKAGE.md) | Current (M150) | Outside-review package guide |
| [docs/testing.md](testing.md) | Current (M149) | All test layers documented |
| [docs/CUDA_VALIDATION_PLAN.md](CUDA_VALIDATION_PLAN.md) | Current | CUDA signoff policy |
| [docs/OPERATION_LOG.md](OPERATION_LOG.md) | Current (M150) | Full milestone history |
| [docs/aimu_architecture.md](aimu_architecture.md) | Current (M148) | AIMU architectural overview |
| [docs/tensor_placement_report.md](tensor_placement_report.md) | Current | Placement schema reference |
| [docs/tensor_execution_plan.md](tensor_execution_plan.md) | Current | Execution plan schema reference |
| [docs/aimu_fabric_routing.md](aimu_fabric_routing.md) | Current | Fabric route report reference |
| [docs/aimu_pcie_prototype_review.md](aimu_pcie_prototype_review.md) | Current | PCIe prototype advisory |
| [docs/aimu_pcie_command_requirements.md](aimu_pcie_command_requirements.md) | Current | PCIe command requirements |
| [docs/aimu_register_map.md](aimu_register_map.md) | Current | AIMU register map reference |

All 16 required documents verified present by `make docs-check`.

---

## 7. Artifact Policy

| Category | Policy |
|----------|--------|
| Tiny deterministic test fixtures | Allowed — `compiler/fixtures/`, `models/dummy/` |
| Public model weights (Llama, Mistral, SmolLM2, etc.) | **Must stay external** — never commit |
| Generated large `.att1` files from public models | **External only** |
| Large generated binaries | **External only** |
| Source model directories | **External only** |
| Compiler-generated `.pyc` / `__pycache__` | **Never commit** — enforced by CI and `make docs-check` |
| Local absolute paths in committed docs | **Prohibited** — warned by `make docs-check` |

Only files tracked by `git ls-files` are included in a review archive.
Generate a safe review archive with:

```sh
git archive --format=tar.gz --prefix=att1/ HEAD \
    -o att1-review-$(git rev-parse --short HEAD).tar.gz
```

---

## 8. Known Limitations

Reviewers must be aware of these intentional boundaries before any evaluation.

| Limitation | Detail |
|------------|--------|
| No real PCIe endpoint | All AIMU simulation is in-process userspace only |
| No Linux kernel driver | `src/aimu_mmio.c` is a userspace emulator; no kernel module |
| No FPGA RTL | FPGA feasibility is advisory documentation only |
| No production ASIC | Phase 1 is software simulation only; no tape-out |
| No real hardware fabric | Fabric routing is planner-level simulation only |
| No hardware-backed AIMU tensor math | EXEC ops are simulated control flow; no AIMU silicon |
| Tensor-level placement execution | Advisory/simulated output — no hardware mapping |
| CUDA signoff is manual | CPU CI does not validate CUDA kernels |
| Large public model artifacts are external | No public model weights in repository |
| Performance numbers limited | All benchmarks use tiny 4-dimension dummy fixture only |
| KV-MMU is software simulated | No hardware page-table walker |

---

## 9. Outside-Review Recommendation

Reviewers should focus on the following areas:

| Area | Where to look |
|------|---------------|
| C11 API ownership and lifetime contracts | [docs/api_ownership_review.md](api_ownership_review.md), `include/att1_*.h` |
| Hostile-input validation approach | `compiler/test_hostile_inputs.py`, `compiler/fixtures/hostile/`, `tests/fuzz_model_loader.c` |
| `.att1` format and converter assumptions | [docs/model_format.md](model_format.md), `src/model_loader.c`, `include/att1_model.h` |
| Backend no-silent-fallback behavior | `src/backend_cpu_f32.c`, `src/backend_cpu_q8.c`, `src/backend_cpu_q4.c`, `src/backend_cuda.c` |
| q4/q8 tolerance policy and correctness approach | [docs/quantization.md](quantization.md), `tests/test_backend.c` |
| AIMU control-plane model | `include/att1_aimu_cmdq.h`, `src/aimu_cmdq.c`, `src/aimu_host.c` |
| MMIO emulator and replay stack | `src/aimu_mmio.c`, `src/aimu_userspace.c`, `compiler/replay_*.py` |
| Documentation clarity and completeness | Start at [docs/INDEX.md](INDEX.md) |

---

## 10. Review Decision

> **READY FOR LIMITED OUTSIDE TECHNICAL REVIEW**

ATT-1 at commit `afdc02b` (M149, tagged `milestone-149-docs-lint-checker`) is
ready for limited outside technical review of the software, simulator,
documentation, and validation infrastructure, subject to the following
labeled conditions:

| Condition | Label |
|-----------|-------|
| CUDA kernels | Not validated by CI — manual RTX 3090 signoff required |
| Real hardware | Not implemented — software simulator only |
| Public model weights | External — not included in repository |
| Production deployment | Not applicable — research/simulation package |
| Patent disclosure | Not included — engineering architecture language only |

Reviewers should inspect the areas listed in §9 and confirm they understand
§8 (Known Limitations) before drawing conclusions about hardware capability
or production readiness.

---

## 11. Next Milestone Proposals

| Milestone | Working title | Scope |
|-----------|---------------|-------|
| M151 | API opacity and refactor plan | Migrate `int`-returning init functions to `att1_status_t`; opacify `att1_kv_mmu` struct; resolve alias duplicates in `att1_status.h` |
| M152 | Deeper fuzzing and coverage expansion | libFuzzer / AFL++ integration; coverage measurement; expand hostile-input fixture set beyond 37 |
| M153 | Release package dry-run | `git archive` tarball verification; reviewer quick-start validation on clean VM; archive artifact inspection checklist |
| M154 | External review response log | Structured log for tracking outside reviewer questions, findings, and responses |
| M155 | Public small-model demo policy | Define and document policy for enabling opt-in public SmolLM2-135M or equivalent demo with external weight download |
| M156 | Self-hosted CUDA runner decision | Decide whether to activate `cuda-self-hosted.example.yml`; document RTX 3090 runner setup; activate or formally defer |
| M157 | Tensor-level execution simulator next slice | Advance AIMU EXEC simulation from control-flow-only to partial tensor-math validation for a single op (e.g. RMSNORM) |
