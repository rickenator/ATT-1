# ATT-1 CUDA Validation Plan (M138)

This document defines the CUDA validation policy, required manual signoff
steps, paths to test, expected skip behavior, self-hosted runner plan, and
signoff report template for ATT-1 CUDA paths.

---

## 1. CUDA Validation Policy

### 1.1 Default CI is CPU-only

The GitHub Actions workflow (`.github/workflows/ci.yml`) is **CPU-only**.
It does not install CUDA, does not require an NVIDIA driver, and does not
validate CUDA kernel correctness. See [testing.md](testing.md) for the
full comparison of testing levels.

### 1.2 CUDA validation is manual

CUDA validation is performed manually on a physical CUDA-capable host
by Rick. A future optional self-hosted runner (§5) could automate this,
but it is not required.

### 1.3 Milestone completion wording

Milestones that touch CUDA code use one of two status states:

| State | Meaning |
|-------|---------|
| **CPU validation complete; CUDA signoff pending** | All CPU-only tests pass; CUDA paths have not yet been validated on hardware |
| **CUDA validation complete on CUDA-capable host** | Manual signoff performed; see signoff report (§7) |

Any milestone that adds or modifies a CUDA kernel, CUDA test, CUDA backend
path, or CUDA-facing API is **not fully complete** until Rick performs the
manual CUDA signoff (§2).

### 1.4 No silent CUDA fallback

If `CUDA=1` is requested and the CUDA runtime is absent or the GPU is
unavailable, the build or tests must fail explicitly — never silently fall
back to CPU. The `no_cuda_dep` compile-time guards in the test suite
enforce the inverse: CPU-only builds must not contain CUDA symbols.

---

## 2. Required Manual CUDA Signoff Commands

Run these on a CUDA-capable host. Record output in a signoff report (§7).

### 2.1 Full CUDA build and test suite

```sh
make clean && make CUDA=1 && make test CUDA=1
```

Expected: all tests that have CUDA implementations execute their CUDA
bodies. Tests that are CPU-only still pass. `no_cuda_dep` guards in
`test_aimu_exec` and `test_aimu_mmio_regression` print PASS noting that
CUDA was intentionally enabled.

### 2.2 Full regression runner with CUDA flag

```sh
python3 compiler/run_full_regression.py --cuda
```

This replaces steps 1–3 with `make CUDA=1` / `make test CUDA=1` and
marks the JSON report with `"cuda": true`.

To save a timestamped signoff report:

```sh
python3 compiler/run_full_regression.py --cuda \
    --report-json "cuda_signoff_$(date +%Y%m%d_%H%M%S).json"
```

### 2.3 CUDA bench smoke commands

These require external model files (not committed to the repo). Run only
when the relevant model is available on the host.

```sh
# f32 single-tile
./build/att1-bench \
    --model models/dummy/model.att1 \
    --prompt hello --tokens 8 \
    --mode single --backend cuda

# f32 cluster (2 tiles)
./build/att1-bench \
    --model models/dummy/model.att1 \
    --prompt hello --tokens 8 \
    --mode cluster --tiles 2 --backend cuda

# q8 single-tile (requires real_tiny_q8 model)
./build/att1-bench \
    --model models/real_tiny_q8/model.att1 \
    --tokenizer external \
    --input-token-ids 1,3,5 --tokens 2 \
    --mode single --backend cuda-q8

# q4 single-tile (requires q4_tiny model)
./build/att1-bench \
    --model models/q4_tiny/model.att1 \
    --prompt hello --tokens 8 \
    --mode single --backend cuda-q4

# q4 cluster (2 tiles)
./build/att1-bench \
    --model models/q4_tiny/model.att1 \
    --prompt hello --tokens 8 \
    --mode cluster --tiles 2 --backend cuda-q4
```

---

## 3. CUDA Paths to Validate

### 3.1 f32 CUDA paths

| Path | Test target | Notes |
|------|-------------|-------|
| CUDA f32 matmul | `cuda_matmul` | core dense kernel |
| CUDA f32 norm (RMSNorm) | `cuda_norm` | |
| CUDA f32 RoPE | `cuda_rope` | |
| CUDA f32 attention | `cuda_attention` | KV-cache path |
| CUDA f32 FFN (SwiGLU) | `cuda_ffn` | |
| CUDA f32 inference | `cuda_infer` | single-tile end-to-end |
| CUDA f32 cluster inference | `cuda_cluster` | multi-tile |
| CUDA f32 benchmark | `cuda_bench` | throughput smoke |

### 3.2 q8 CUDA paths

| Path | Test target | Notes |
|------|-------------|-------|
| CUDA q8 matmul | `cuda_matmul` (q8 variant) | |
| CUDA q8 inference | `q8_cluster` | |
| CUDA q8 cluster | `cuda_q8_cluster` | |

### 3.3 q4 CUDA paths

| Path | Test target | Notes |
|------|-------------|-------|
| CUDA q4 matmul | `cuda_matmul_q4` | |
| CUDA q4 inference | `cuda_infer_q4` | |
| CUDA q4 cluster | `cuda_cluster_infer_q4` | |

### 3.4 Backend matrix

Run `test_backend` with `CUDA=1` to confirm the backend selection matrix
correctly routes `--backend cuda`, `--backend cuda-q8`, and
`--backend cuda-q4` to their respective CUDA dispatch functions.

### 3.5 q4/q8 comparison

Where a CPU golden output exists, compare CUDA output against it using
`compiler/check_golden_regressions.py` to detect numerical divergence.

---

## 4. Expected Skip Behavior

### 4.1 CPU-only builds

- CUDA test targets compile (they are always in `TEST_NAMES`).
- When built without `CUDA=1`, the CUDA test bodies are compiled out via
  `#ifdef ATT1_ENABLE_CUDA` guards and the tests pass by skipping.
- `no_cuda_dep` guards confirm no CUDA symbols leaked into CPU binaries.

### 4.2 CUDA unavailable at runtime

- If `CUDA=1` was specified but the CUDA runtime library is absent, the
  linker will fail with an undefined symbol error. This is intentional:
  no silent CPU fallback.
- If the GPU is absent but the runtime is installed, CUDA device
  enumeration returns 0 devices; tests that require a GPU should report
  `SKIP` or `UNSUPPORTED`, not silently pass on CPU.

### 4.3 No silent fallback policy

The `att1_backend` dispatch table must never silently route a CUDA request
to a CPU backend. Unsupported configurations must return
`ATT1_UNSUPPORTED` or equivalent — not a best-effort CPU result.

---

## 5. Self-Hosted Runner Plan

This section documents the **optional future** setup for automating CUDA
signoff via a dedicated GitHub Actions self-hosted runner.

### 5.1 Host requirements

| Item | Requirement |
|------|-------------|
| GPU | Any CUDA-capable GPU (RTX 3090 or equivalent used for initial signoff) |
| OS | Ubuntu 22.04 LTS or 24.04 LTS |
| NVIDIA driver | 535+ recommended |
| CUDA toolkit | 12.x |
| RAM | 32 GiB+ |
| Disk | 100 GiB+ free (build artifacts + model files) |

### 5.2 Runner labels

Register the runner with these labels so the CUDA workflow targets it
exclusively and CPU CI never runs on it:

```
self-hosted
linux
x64
cuda
rtx3090
```

### 5.3 Runner isolation rules

- The self-hosted runner must **not** be the same host as the CPU CI
  runner. Use `runs-on: [self-hosted, linux, cuda, rtx3090]` in the
  CUDA workflow, not `ubuntu-latest`.
- Public model weights must **not** be checked into the repo. The runner
  host may have models pre-staged at a local path not tracked by Git.
- The workflow must not expose secrets (API keys, tokens) unless
  explicitly needed.
- The CUDA workflow should not run on every push to avoid unnecessary GPU
  wear. Trigger on tags, manual dispatch, or a dedicated branch.

### 5.4 Suggested trigger strategy

```yaml
on:
  workflow_dispatch:          # manual trigger
  push:
    tags:
      - 'cuda-signoff-*'      # push a tag to trigger CUDA CI
```

See the example workflow in
[`.github/workflows/cuda-self-hosted.example.yml`](../.github/workflows/cuda-self-hosted.example.yml).

---

## 6. Example Self-Hosted Workflow

The file `.github/workflows/cuda-self-hosted.example.yml` is an **inactive
example** — rename it to `.yml` and configure the runner before enabling.

Key differences from CPU CI:

| Feature | CPU CI (`ci.yml`) | CUDA example |
|---------|-------------------|--------------|
| Runner | `ubuntu-latest` | `[self-hosted, linux, cuda, rtx3090]` |
| Build | `make` | `make CUDA=1` |
| Tests | `make test` | `make test CUDA=1` |
| Regression | `--no-build` | `--cuda` |
| Trigger | push / PR | `workflow_dispatch` / `cuda-signoff-*` tag |
| CUDA required | no | yes |

---

## 7. CUDA Signoff Report Template

Copy and fill in this template when performing a manual CUDA signoff.
Store the completed report locally; do not commit it to the repo unless
it contains no sensitive host information.

```
ATT-1 CUDA Signoff Report
=========================
Date          :
Host          :
GPU           :
Driver version:
CUDA version  :
Git commit    :
Branch        :

Commands run
------------
make clean && make CUDA=1 && make test CUDA=1
python3 compiler/run_full_regression.py --cuda --report-json cuda_signoff.json

make test CUDA=1 summary
------------------------
  [paste output here]

run_full_regression.py --cuda summary
--------------------------------------
  [paste summary table here]

Bench smoke (if models available)
----------------------------------
  cuda f32 single   :
  cuda f32 cluster  :
  cuda-q8 single    :
  cuda-q4 single    :
  cuda-q4 cluster   :

Result
------
  Overall          : PASS / FAIL
  Notes            :

Signed off by: Rick
```

---

## 8. Non-Goals

- No CUDA installation in hosted GitHub Actions runners.
- No GPU provisioning requirement.
- No benchmark performance guarantee.
- No new CUDA kernels added by this milestone.
- No Linux kernel driver.
- No FPGA RTL.
- No patent claim language.
