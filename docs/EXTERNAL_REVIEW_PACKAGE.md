# ATT-1 External Reviewer Package Checklist (M145)

This document describes what is included in an ATT-1 outside review, what is
excluded, how to validate the package before sharing, and what reviewers should
focus on.

The review package is the Git repository in its current state on `master`.
No additional bundling or archive step is required for a standard review.

---

## 1. Review Package Purpose

| Item | Status |
|------|--------|
| ATT-1 tiny-fixture end-to-end demo | Included — `./tools/demo_tiny_att1.sh` |
| C11 runtime API review | Included — `include/` and `src/` |
| Model converter and planning tool review | Included — `compiler/` |
| Control-plane simulator review | Included — `src/aimu_*.c`, `include/att1_aimu_*.h` |
| AIMU command/MMIO/DMA/fabric replay review | Included — `tools/att1-aimu-*.c`, `compiler/replay_*.py` |
| Regression and schema validation review | Included — `tests/`, `compiler/check_*.py` |
| Production model deployment | **Not included** — no public model weights |
| Hardware implementation | **Not included** — software simulator only |
| Patent claim disclosure | **Not included** — engineering architecture language only |
| Real PCIe or MMIO access | **Not included** — userspace emulator only |
| Linux kernel driver | **Not included** — not implemented |
| FPGA RTL | **Not included** — not implemented |

---

## 2. Included Materials

The following are tracked in Git and present in the repository:

| Material | Location |
|----------|----------|
| C11 runtime source | `src/`, `include/` |
| Simulator subsystems (fabric, tile, AIMU) | `simulator/`, `src/aimu_*.c` |
| Developer tools (inspect, size, bench, replay) | `tools/` |
| Model converter and planning pipeline | `compiler/` |
| Unit, integration, smoke, and regression tests | `tests/` |
| Tiny deterministic fixture models | `models/dummy/`, `models/tok_meta/`, etc. |
| Planning and schema fixture files | `compiler/fixtures/` |
| Golden regression baselines | `compiler/fixtures/golden/` |
| Hostile-input test fixtures | `compiler/fixtures/hostile/` |
| Schema compatibility fixtures | `compiler/fixtures/schema_compat/` |
| Tiny demo script | `tools/demo_tiny_att1.sh` |
| Documentation index | `docs/INDEX.md` |
| README | `README.md` |
| Release readiness checklist | `docs/RELEASE_READINESS.md` |
| Release candidate checkpoint | `docs/RELEASE_CANDIDATE_M150.md` |
| Testing guide | `docs/testing.md` |
| CUDA validation policy | `docs/CUDA_VALIDATION_PLAN.md` |
| API ownership review | `docs/api_ownership_review.md` |
| Operation log (M0–current) | `docs/OPERATION_LOG.md` |
| CPU-only CI workflow | `.github/workflows/ci.yml` |
| Example programs | `examples/` |
| Architecture diagram | `docs/ATT1-ARCH.jpg` |

---

## 3. Excluded Materials

The following are **not** included in the review package and must remain
outside the Git repository:

| Excluded material | Reason |
|-------------------|--------|
| Public model weights (e.g. SmolLM2-135M safetensors) | Too large; license and redistribution constraints |
| Generated large `.att1` artifacts from real models | Derived from excluded weights |
| Local model directories (`~/models/`, `/data/`, etc.) | External; local paths must not be committed |
| CUDA benchmark claims without manual CUDA signoff | Unvalidated on CPU-only CI |
| Patent claim drafts | Must stay outside version control |
| Private invention disclosures | Must stay outside version control |
| Hardware implementation details beyond architectural docs | Not implemented |
| Secrets, API keys, tokens | Must never be committed |
| Local absolute paths in committed files | Violates portability policy |

---

## 4. Required Pre-Review Validation

Run all of the following on a clean checkout before sharing the package.
All commands must exit 0.

```sh
# 1. Baseline build and test (CPU-only, no CUDA)
make clean && make && make test

# 2. Full regression (schema, hostile-input, golden, pipeline smoke, docs-check)
make regression

# 3. Documentation lint and link checker
make docs-check

# 4. ASAN build and test
make clean && make test-asan

# 5. UBSAN build and test
make clean && make test-ubsan

# 6. Loader and schema fuzz/smoke
make fuzz-smoke

# 7. End-to-end tiny fixture demo
./tools/demo_tiny_att1.sh

# 8. Python cache artifact check (must print "No tracked Python cache artifacts")
git ls-files | grep -E '(__pycache__|\.pyc$|\.pyo$)' || echo "No tracked Python cache artifacts"
```

Expected baselines:

| Check | Expected result |
|-------|-----------------|
| `make test` | 781 PASS 0 FAIL |
| `make regression` | All steps PASS (9 steps) |
| `make docs-check` | PASS (0 errors) |
| `make test-asan` | 781 PASS 0 FAIL (ASAN) |
| `make test-ubsan` | 781 PASS 0 FAIL (UBSAN) |
| `make fuzz-smoke` | fuzz_loader 17/17 PASS; fuzz_json 40/40 PASS |
| `./tools/demo_tiny_att1.sh` | 14/14 PASS; `ATT-1 tiny demo: PASS` |
| Cache artifact check | No tracked Python cache artifacts |

---

## 5. Optional CUDA Validation

CPU CI does **not** validate CUDA kernels. CUDA paths compile cleanly in
CPU-only builds behind `#ifdef CUDA_ENABLED` guards and skip their test
bodies on hosts without a GPU.

**CUDA signoff is manual, on a CUDA-capable host:**

```sh
make clean && make CUDA=1 && make test CUDA=1
python3 compiler/run_full_regression.py --cuda --report-json cuda_signoff_$(date +%Y%m%d).json
```

Mark CUDA status clearly in any review communication:

| Status | Meaning |
|--------|---------|
| `not validated` | CUDA path not reviewed; CPU-only signoff only |
| `pending` | CUDA build clean; awaiting manual CUDA signoff |
| `validated on CUDA-capable host` | `make test CUDA=1` and `run_full_regression.py --cuda` passed |

See `docs/CUDA_VALIDATION_PLAN.md` for full policy and signoff commands.

---

## 6. Reviewer Quick-Start

```sh
# 1. Clone
git clone git@github.com:rickenator/ATT-1.git
cd ATT-1

# 2. Install prerequisites (Debian/Ubuntu)
sudo apt-get install build-essential python3 make

# 3. Build and test
make clean && make && make test

# 4. Run the tiny demo
./tools/demo_tiny_att1.sh

# 5. Read the documentation index
# Open docs/INDEX.md for the full document map.

# 6. Read the release readiness checklist
# Open docs/RELEASE_READINESS.md for the full pre-release gate list.
```

No CUDA, no network access beyond the initial clone, no public model
downloads, and no external model weights are required for any of the above.

---

## 7. Suggested Reviewer Focus Areas

| Area | Where to look |
|------|---------------|
| C11 API ownership and lifetime | `docs/api_ownership_review.md`, `include/att1_*.h` |
| Hostile-input validation | `compiler/check_hostile_inputs.py`, `compiler/fixtures/hostile/`, `tests/fuzz_model_loader.c` |
| `.att1` format and versioning | `docs/model_format.md`, `src/model_loader.c`, `include/att1_model.h` |
| q4/q8/f32 backend correctness approach | `docs/quantization.md`, `src/backend_cpu_*.c`, `tests/test_backend.c` |
| Placement report schemas | `docs/tensor_placement_report.md`, `compiler/validate_tensor_placement_report.py` |
| Command/control-plane simulator | `include/att1_aimu_cmdq.h`, `src/aimu_cmdq.c`, `include/att1_aimu_host.h`, `src/aimu_host.c` |
| MMIO emulator path | `include/att1_aimu_mmio.h`, `src/aimu_mmio.c`, `include/att1_aimu_userspace.h`, `src/aimu_userspace.c` |
| Fabric route planner and simulator | `compiler/map_commands_to_fabric_routes.py`, `compiler/replay_fabric_routes.py`, `docs/aimu_fabric_routing.md` |
| Regression coverage | `tests/`, `compiler/check_golden_regressions.py`, `compiler/check_schema_compat.py`, `compiler/test_hostile_inputs.py` |
| Documentation clarity | `docs/INDEX.md` (start here), then per-topic docs |
| Schema compatibility policy | `docs/schema_compatibility.md` |
| CUDA signoff status | `docs/CUDA_VALIDATION_PLAN.md` |

---

## 8. Known Limitations

Reviewers should be aware of the following limitations:

| Limitation | Notes |
|------------|-------|
| No real PCIe hardware | All AIMU simulation is in-process userspace |
| No FPGA RTL | FPGA feasibility is advisory-only (`docs/fpga_feasibility.md`) |
| No Linux kernel driver | Userspace MMIO emulator only |
| No tensor-level placement against real hardware | Planning pipeline is advisory/simulation |
| No production ASIC | Not planned within this project scope |
| No performance claims for large real models | All perf numbers from tiny dummy fixture only |
| CUDA validation requires manual signoff on a CUDA-capable host | CPU CI does not validate CUDA kernels |
| Public model artifacts are external | No public model weights in repo |
| Fabric routing is planner-level only | No physical interconnect |
| KV-MMU is software simulated | No hardware page-table walker |

---

## 9. Artifact Inspection Checklist

Before sharing the repository or a tarball, verify the following:

```sh
# No large unexpected files
git ls-files | xargs wc -c 2>/dev/null | sort -rn | head -20

# No Python cache files
git ls-files | grep -E '(__pycache__|\.pyc$|\.pyo$)' || echo "OK"

# No absolute home or system paths in committed text files
git grep -l '/home/' -- '*.py' '*.sh' '*.md' '*.c' '*.h' 2>/dev/null || echo "OK"

# No generated model artifacts from public weights
git ls-files models/ | grep -v dummy | grep -v tok_meta | grep -v shard_meta \
  | grep -v converted_stub | grep -v real_tiny | grep -v q4_tiny \
  | grep -v m6[0-9] | grep -v m8[0-9] | echo "OK (check manually if non-empty)"

# No secrets or API keys
git grep -i 'api_key\|token\s*=\|password\s*=\|secret\s*=' -- '*.py' '*.sh' '*.c' '*.h' 2>/dev/null || echo "OK"

# No .att1 files larger than the tiny fixtures
find models/ -name '*.att1' -size +1M 2>/dev/null | head -5 || echo "OK: no large .att1 files"
```

All checks must produce `OK` or an empty result before sharing.

---

## 10. Safe Sharing Notes

- Use **engineering architecture language** only. Describe what the system
  does and how it works. Do not use language that makes legal performance or
  invention claims.
- **No patent claim language.** Do not include phrases such as "we claim",
  "novel method for", or "patent-pending". Keep claim drafts and invention
  disclosures completely outside the Git repository and any shared archive.
- **Do not share private model artifacts.** If you have locally generated
  `.att1` files from public model weights, exclude them from any shared
  tarball.
- **Generate shared archives from a clean Git state:**

  ```sh
  git archive --format=tar.gz --prefix=att1/ HEAD -o att1-review-$(git rev-parse --short HEAD).tar.gz
  ```

  This ensures only committed files are included, no local build artifacts,
  no ignored files, and no `.pyc` cache files.

- **Verify the archive before sending:**

  ```sh
  tar -tzf att1-review-*.tar.gz | grep -E '(__pycache__|\.pyc$|\.pyo$)' || echo "OK: no cache in archive"
  tar -tzf att1-review-*.tar.gz | grep '\.att1$'
  # Verify only tiny fixture .att1 files appear
  ```

---

## 11. Reference Manuals

The following formal reference documents are complete and available in `docs/`:

| Document | Milestone | File | Description |
|----------|-----------|------|-------------|
| ATT-1 Reference Manual | M146 | [ATT1_REFERENCE_MANUAL.md](ATT1_REFERENCE_MANUAL.md) | Complete C API reference, `.att1` format specification, runtime semantics, backend selection matrix, KV-cache MMU design, quantization pipeline |
| AIMU Intrinsics and Operations Reference Manual | M147 | [AIMU_INTRINSICS_OPERATIONS_REFERENCE.md](AIMU_INTRINSICS_OPERATIONS_REFERENCE.md) | AIMU command set reference, MMIO register map, fabric packet format, DMA protocol, execution phases, simulator correspondence |

Both manuals are included in the review package and cross-reference each other.

---

## Cross-References

| Related document | Purpose |
|------------------|---------|
| `docs/RELEASE_READINESS.md` | Full pre-release gate checklist |
| `docs/testing.md` | Testing guide: `make test`, regression, sanitizers, CI |
| `docs/CUDA_VALIDATION_PLAN.md` | CUDA signoff policy and manual commands |
| `docs/INDEX.md` | Full documentation map |
| `docs/schema_compatibility.md` | Schema versioning and hostile-input policy |
| `docs/api_ownership_review.md` | C API ownership, lifetime, const-correctness |
| `docs/OPERATION_LOG.md` | Full milestone history |
| `tools/demo_tiny_att1.sh` | Tiny fixture end-to-end demo |
| `.github/workflows/ci.yml` | CPU-only CI definition |
