# ATT-1 Testing Guide

This document describes the three testing levels used by the ATT-1 project,
and how they relate to CI and local development.

---

## 1. C unit and integration tests (`make test`)

All C source-level validation is driven by the Makefile:

```sh
make clean && make && make test
```

This compiles every test binary listed in `TEST_NAMES` and runs them in
order. A single binary failure aborts the run.

**Baseline**: 781 PASS 0 FAIL (CPU-only).

CUDA test targets (`cuda_*`, `cuda_q8_*`, etc.) are compiled with
`#ifdef ATT1_ENABLE_CUDA` guards and skip their test bodies cleanly when
built without `CUDA=1`. They do not fail in CPU-only CI.

---

## 2. Local full-regression runner (`make regression`)

M136 adds a deterministic Python orchestrator that runs all ATT-1
validation layers in a stable order:

| Step | What runs |
|------|-----------|
| 1 | `make clean` |
| 2 | `make` (or `make CUDA=1` with `--cuda`) |
| 3 | `make test` (or `make test CUDA=1`) |
| 4 | M133 golden regression baselines (`compiler/check_golden_regressions.py`) |
| 5 | M134 schema compatibility suite (`compiler/test_schema_compat.py`) |
| 6 | M135 hostile-input regression suite (`compiler/test_hostile_inputs.py`) |
| 7 | M132 integrated pipeline smoke (`compiler/run_execution_replay_pipeline.py`) |
| 8 | Git-tracked Python cache artifact check |

```sh
# CPU-only (default)
make regression

# or directly
python3 compiler/run_full_regression.py

# skip the build steps on a fresh build
python3 compiler/run_full_regression.py --no-build

# write a machine-readable JSON report
python3 compiler/run_full_regression.py --report-json regression_report.json
```

`make regression` is the **stronger** local check and should be run before
pushing to a shared branch.

---

## 3. CI (`git push` / pull request)

GitHub Actions runs `.github/workflows/ci.yml` on every push and pull
request. The CI pipeline is **CPU-only** and does the following:

1. `make clean`
2. `make`
3. `make test`
4. `python3 compiler/run_full_regression.py --no-build` (M133–M135 Python layers + pipeline smoke)
5. `git ls-files | grep -E '(__pycache__|\.pyc$|\.pyo$)'` — fail if any hit

### What CI validates

- All C11 binaries compile cleanly (GCC, Ubuntu, no CUDA).
- All C unit and integration tests pass.
- M133 golden regression baselines are stable.
- M134 schema compatibility fixtures all pass.
- M135 hostile-input fixtures all rejected correctly.
- M132 pipeline smoke exits 0 (binary-absent → warn, not fail).
- No `__pycache__` or `.pyc` files are git-tracked.

### What CI does NOT validate

| Area | Reason |
|------|--------|
| CUDA kernel correctness | No NVIDIA driver or GPU in the runner |
| CUDA q4 / q8 / f32 runtime | Same — CUDA=0 by default |
| Public model inference accuracy | Model weights are external and not committed |
| PCIe / MMIO hardware access | No real hardware; emulator is userspace-only |
| CUDA signoff | Manual — see below |

---

## 4. CUDA signoff (manual, CUDA-capable host)

CUDA validation requires a physical GPU and is not run in CI. To perform
a full CUDA signoff on a capable host:

```sh
python3 compiler/run_full_regression.py --cuda
```

This replaces steps 1–3 with `make CUDA=1` and `make test CUDA=1` and
marks the report as a CUDA signoff. The output JSON includes
`"cuda": true`.

For the full CUDA validation policy, signoff report template, paths to
test, expected skip behavior, and self-hosted runner plan, see
[docs/CUDA_VALIDATION_PLAN.md](CUDA_VALIDATION_PLAN.md).

An inactive example self-hosted workflow is provided at
`.github/workflows/cuda-self-hosted.example.yml`.

---

## 5. Difference between `make test`, `make regression`, and CI

| | `make test` | `make regression` | CI |
|---|---|---|---|
| C binaries | ✓ | ✓ | ✓ |
| Golden regressions (M133) | ✗ | ✓ | ✓ |
| Schema compat (M134) | ✗ | ✓ | ✓ |
| Hostile inputs (M135) | ✗ | ✓ | ✓ |
| Pipeline smoke (M132) | ✗ | ✓ | ✓ |
| Cache artifact check | ✗ | ✓ | ✓ |
| Docs lint/link check (M149) | ✗ | ✓ | ✓ |
| CUDA tests | optional | optional (`--cuda`) | ✗ |
| JSON report | ✗ | optional | ✗ |
| Runs automatically | ✗ | ✗ | ✓ (on push/PR) |

---

## 6. Before sharing or releasing

Before sending the repository to an external reviewer or pushing a release
tag, run the full outside-review checklist in
[docs/RELEASE_READINESS.md](RELEASE_READINESS.md).

The quick pre-release sequence is:

```sh
make clean && make && make test   # 781 PASS 0 FAIL
make regression                   # all 5 layers PASS
git ls-files | grep -E '(__pycache__|\.pyc$|\.pyo$)' || echo "PASS: no cache artifacts"
```

---

## 7. Sanitizer builds (local hardening, M142)

AddressSanitizer (ASAN) and UndefinedBehaviorSanitizer (UBSAN) targets are
provided for local hardening. They are **not** part of normal CI and are
**not** CUDA validation.

### Purpose

These tools catch ownership, lifetime, bounds, and undefined-behavior bugs
at runtime that are invisible to the normal compiler:

| Tool | Catches |
|------|---------|
| ASAN | Heap/stack buffer overflows, use-after-free, double-free, memory leaks |
| UBSAN | Signed overflow, misaligned pointer deref, null deref, invalid casts, shift errors |

### Usage

```sh
# AddressSanitizer — build and run all C tests
make clean && make test-asan

# UndefinedBehaviorSanitizer — build and run all C tests
make clean && make test-ubsan

# Build both sanitizer variants (does not run tests)
make sanitizer

# Remove sanitizer build directories
make clean-sanitizer
```

Sanitizer artifacts live in `build-asan/` and `build-ubsan/` and are
completely separate from the normal `build/` directory. Running
`make clean` does **not** remove them; use `make clean-sanitizer`.

### Flags applied

| Target | Extra flags |
|--------|------------|
| `test-asan` | `-fsanitize=address -fno-omit-frame-pointer -g` |
| `test-ubsan` | `-fsanitize=undefined -fno-omit-frame-pointer -g` |

Both targets force `CUDA=0`. Sanitizer flags are never applied to CUDA
builds.

### Interpreting failures

If a sanitizer reports a violation:

- **ASAN** prints a stack trace starting with `==ERROR: AddressSanitizer:`.
  The frame addresses can be symbolised with `addr2line` or by building with
  `-g` (already included).
- **UBSAN** prints `runtime error:` with a file/line reference. Fix the
  reported undefined behavior before the next commit.

Sanitizer failures that do not reproduce in the normal build indicate a
real latent bug. Fix before merging.

### Relationship to CI

| | Normal CI | Sanitizer targets |
|---|---|---|
| Runs automatically on push | ✓ | ✗ — local only |
| CUDA required | ✗ | ✗ |
| Separate build dir | ✗ | ✓ (`build-asan/`, `build-ubsan/`) |
| Part of `make regression` | ✗ | ✗ |
| Required to pass before release | recommended | recommended |

---

## 8. Fuzz/smoke harness (local hardening, M143)

### Purpose

The fuzz/smoke harness validates that hostile or malformed inputs are
**correctly rejected** by the binary model loader and the JSON schema
validators.  It is distinct from sanitizer builds:

| Concern | Sanitizer builds (§7) | Fuzz/smoke (§8) |
|---|---|---|
| Catches memory safety bugs | ✓ | ✗ |
| Catches incorrect acceptance of malformed input | ✗ | ✓ |
| Randomised / long-running | ✗ | ✗ (deterministic only) |

The harness is CPU-only, deterministic, and completes in under 10 seconds.

### Usage

```sh
# Run both harnesses (recommended before a release)
make fuzz-smoke

# C binary loader harness only
make fuzz-loader

# Python JSON schema mutation harness only
make fuzz-json
```

### C binary loader harness (`fuzz-loader`)

Source: `tests/fuzz_model_loader.c`
Binary: `build/fuzz_model_loader`

Builds minimal ATT-1 binary blobs in memory, writes them to `/tmp`, and
calls `att1_model_load()` on each.  Seed corpus (all inline — no external
seed files required):

| Case | Expected outcome |
|------|-----------------|
| Minimal valid V1 (no tensors) | accepted (status 0) |
| `models/dummy/model.att1`     | accepted (status 0) |
| Empty file                    | rejected |
| 7-byte truncation (below magic) | rejected |
| 40-byte truncation (incomplete header) | rejected |
| Bad magic bytes               | rejected |
| Version = 0                   | rejected |
| Version = 99 (future)         | rejected |
| V1 header_size mismatch       | rejected |
| config_size = 0               | rejected |
| config_size too large         | rejected |
| config_offset = UINT64_MAX    | rejected |
| data_offset = UINT64_MAX      | rejected |
| tensor_count = UINT64_MAX     | rejected |
| shard_size without shard_offset | rejected |
| All-zeros file                | rejected |

### Python JSON schema harness (`fuzz-json`)

Source: `compiler/fuzz_json_schemas.py`

Drives `compiler/check_hostile_inputs.py` (M135) against:

1. **All 27 hostile fixtures** in `compiler/fixtures/hostile/` — each must
   be rejected (non-zero exit).
2. **Valid baseline fixtures** (`placement_report_valid.json`,
   `exec_plan_valid_tiny.json`) — each must pass (exit 0).
3. **Inline-generated mutation seeds** (truncated JSON, wrong types,
   negative counts, future versions, deeply nested garbage, etc.) — each
   must be rejected.

### Adding new seeds

- **C loader seeds**: add a mutation block in `tests/fuzz_model_loader.c`
  using the `RUN()` macro.
- **JSON seeds**: add a new entry to `_MUTATIONS` in
  `compiler/fuzz_json_schemas.py`, or add a new JSON file to
  `compiler/fixtures/hostile/`.

### Longer fuzz runs

Integration with coverage-guided fuzzers (libFuzzer, AFL++) is planned as
future work.  The current harness provides a fast deterministic smoke screen
only.

### Relationship to CI

| | Normal CI | Fuzz/smoke |
|---|---|---|
| Runs automatically on push | ✓ | ✗ — local only |
| CUDA required | ✗ | ✗ |
| Part of `make test` | ✗ | ✗ |
| Part of `make regression` | ✗ | ✗ |
| Required to pass before release | recommended | recommended |

---

## 9. Documentation lint and link checker (`make docs-check`, M149)

### Purpose

`compiler/check_docs.py` validates the documentation tree without
executing inference, accessing the network, or requiring CUDA.  It catches
broken internal Markdown links, missing required documents, stale status
claims, and basic milestone/status consistency errors.

### Usage

```sh
# Basic run — exits 0 on success, 1 on any error
make docs-check

# With anchor validation warnings
make docs-check DOCS_CHECK_ARGS="--warn-anchors"

# Write a machine-readable JSON report
make docs-check DOCS_CHECK_ARGS="--report-json /tmp/docs_report.json"

# Or invoke directly
python3 compiler/check_docs.py
python3 compiler/check_docs.py --report-json report.json
```

### What is checked

| Check | Description |
|-------|-------------|
| Internal Markdown links | For every `[text](url-or-path)` in a `.md` file, verify the target file exists. External `http://` links are not followed (no network). |
| Required documents | All 17 key documents (README.md, docs/INDEX.md, both reference manuals, release candidate checkpoint, testing, release readiness, CUDA validation, OPERATION_LOG, AIMU architecture, PCIe, command requirements, register map, fabric routing, tensor placement/execution docs) must exist on disk. |
| Tracked cache artifacts | `git ls-files` must not return any `__pycache__`, `.pyc`, or `.pyo` files. |
| Absolute local paths | Warns when `/home/…` or `/usr/export/…` absolute paths appear in documentation. |
| Stale "future manual" claims | Errors if non-historical docs still use obsolete section titles or describe completed milestones as pending. OPERATION_LOG.md is excluded as a historical record. |
| Milestone/status consistency | OPERATION_LOG must contain a Milestone 151 entry and a Milestone 150 complete entry. CUDA signoff must be described as manual on a CUDA-capable host. |

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | All checks pass (warnings may appear) |
| 1 | One or more lint/link failures detected |
| 2 | Tool/parser error (unexpected exception) |

### Relationship to CI and regression

| | Normal CI | `make regression` | `make docs-check` |
|---|---|---|---|
| Runs automatically on push | ✓ | ✗ | ✗ |
| CUDA required | ✗ | ✗ | ✗ |
| Part of `make test` | — | — | ✗ — separate |
| Part of `make regression` | ✗ (CI step) | ✓ — step 9 | — |
| Network required | ✗ | ✗ | ✗ |
| Required to pass before release | — | ✓ | ✓ |
