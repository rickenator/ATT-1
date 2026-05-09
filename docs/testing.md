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
| RTX 3090 signoff | Manual — see below |

---

## 4. CUDA signoff (manual, RTX 3090-class host)

CUDA validation requires a physical GPU and is not run in CI. To perform
a full CUDA signoff on a capable host:

```sh
python3 compiler/run_full_regression.py --cuda
```

This replaces steps 1–3 with `make CUDA=1` and `make test CUDA=1` and
marks the report as a CUDA signoff. The output JSON includes
`"cuda": true`.

A future self-hosted GitHub Actions runner with an RTX 3090 could trigger
this automatically on a dedicated workflow, but that is not configured here.

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
| CUDA tests | optional | optional (`--cuda`) | ✗ |
| JSON report | ✗ | optional | ✗ |
| Runs automatically | ✗ | ✗ | ✓ (on push/PR) |
