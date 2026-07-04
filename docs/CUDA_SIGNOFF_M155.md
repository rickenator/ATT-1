# ATT-1 CUDA Signoff Report - M155

**Milestone:** M155 CUDA signoff closure
**Date:** 2026-07-04
**Host:** `192.168.1.240`
**GPU:** NVIDIA GeForce RTX 3090, 24 GiB VRAM
**Driver / runtime reported by `nvidia-smi`:** driver 580.159.03, CUDA 13.0
**CUDA compiler:** nvcc 12.0.140
**Source commit:** `70b1113` (`origin/master`) plus the M155 regression-runner fix in `compiler/run_full_regression.py`

## Summary

CUDA validation is complete on the RTX 3090 host for the M155 closeout.
The CUDA f32, q8, and q4 paths built, linked, and executed under `CUDA=1`.
The backend matrix reported all CUDA rows passing:

```text
backend_matrix: 28/28 passed, 0 skipped, 0 failed
```

The full CUDA regression runner completed successfully after fixing the M155
runner issue where the final fuzz-smoke step did not preserve `CUDA=1`.

## Commands

```sh
make clean && make CUDA=1 && make test CUDA=1
python3 compiler/run_full_regression.py --cuda --report-json /tmp/att1_m155_cuda_signoff_20260704_1535.json
```

## Results

| Check | Result |
|---|---|
| `make CUDA=1` | PASS |
| `make test CUDA=1` | PASS |
| CUDA f32 tests | PASS |
| CUDA q8 tests | PASS |
| CUDA q4 tests | PASS |
| Backend matrix | 28/28 passed, 0 skipped, 0 failed |
| `aimu_mmio_regression` | 126 PASS, 0 FAIL |
| `run_full_regression.py --cuda` | PASS |
| JSON report | `/tmp/att1_m155_cuda_signoff_20260704_1535.json` on the RTX 3090 host |

Regression summary:

```text
ATT-1 M136+M149+M152 Full Regression Summary
  Mode   : CUDA
  Overall: PASS
  Elapsed: 45.7s

  make clean                         PASS
  make CUDA=1                        PASS
  make test CUDA=1                   PASS
  golden regressions (M133)          PASS
  schema compatibility (M134)        PASS
  hostile-input regression (M135)    PASS
  pipeline smoke (M132)              PASS
  cache artifact check               PASS
  docs lint/link check (M149)        PASS
  fuzz smoke/coverage (M152)         PASS

Result: ALL STEPS PASSED
```

## M155 Fix

The first CUDA regression run exposed a runner-mode bug: after a CUDA build,
`compiler/run_full_regression.py --cuda` invoked plain `make fuzz-smoke`.
That target relinked `backend_cuda.o` without CUDA libraries and failed with
undefined cuBLAS/CUDA references. The runner now invokes:

```sh
make fuzz-smoke CUDA=1
```

when `--cuda` is active. This keeps the final fuzz-smoke step in the same
build mode as the rest of the CUDA signoff.

## Closeout Decision

M155 is closed. CUDA is now a signed-off RTX 3090 baseline for Phase 2
hardware-comparison work. The Phase 2 acceptance tolerances remain:

- f32: exact/reference-equivalent within existing test policy
- q8: <= 0.15 logit tolerance
- q4: <= 0.35 logit tolerance

Next milestone: M156 Phase 1 to Phase 2 gap audit.
