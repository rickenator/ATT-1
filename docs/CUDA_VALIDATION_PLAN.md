# CUDA Validation Plan

## Purpose

Validate Milestone 12 CUDA backend skeleton on real CUDA hardware (local CUDA-capable machine or rented H100-class cloud instance) without expanding scope into CUDA kernels, cuBLAS integration, or full transformer inference.

## Minimum Machine Requirements

- Linux host.
- NVIDIA GPU visible to the OS.
- CUDA toolkit installed.
- Working `nvcc` in `PATH`, or a documented explicit compiler/toolkit path.
- Enough disk and RAM for full ATT-1 build and test runs.

## Preflight Commands

Run and capture output before validation:

```sh
nvidia-smi
nvcc --version
gcc --version
make --version
git status
```

## Required Validation Commands

Run in repo root, in this order:

```sh
make clean && make && make test
make clean && make CUDA=1
make test CUDA=1
```

## Required CLI Smoke Tests

```sh
./build/att1-bench --model models/dummy/model.att1 --prompt hello --tokens 8 --mode single --backend cpu-f32
./build/att1-bench --model models/dummy/model.att1 --prompt hello --tokens 8 --mode single --backend cuda
./build/att1-q8-bench --backend cpu-q8
./build/att1-q8-bench --backend cuda
```

## Expected Results

- CPU path passes everywhere.
- CUDA backend create/destroy succeeds when CUDA is present and healthy.
- CUDA alloc/free/sync/copy helpers pass smoke checks if implemented in the current skeleton.
- CUDA operator kernels are not expected yet.

## Failure Triage

- CUDA toolkit missing.
- NVIDIA driver/runtime mismatch.
- `nvcc` missing from `PATH`.
- `CUDA=1` accidentally required for normal non-CUDA build.
- CLI behavior drift from docs.

## Artifact Collection

Save these artifacts for review:

- Full command transcript.
- Full `make` and `make test` output logs.
- `nvidia-smi` output.
- Git commit hash used for validation.

## Cloud Cost Discipline

- Provision instance.
- Clone repo.
- Run validation steps.
- Collect logs and artifacts.
- Stop/destroy instance immediately after completion.

## Post-Validation Milestone Target

After successful CUDA-host validation, advance to:

- Milestone 14: CUDA matmul/cuBLAS prototype, still no full transformer inference.