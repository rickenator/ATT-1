# ATT-1 Release and Outside-Review Readiness Checklist (M139)

Use this checklist before sharing the repository with any external reviewer,
publishing a release tag, or initiating a formal code review. Work through
every section and resolve any open items.

---

## 1. Repository Hygiene

Run each check and confirm output is clean.

```sh
# 1a. No tracked Python cache artifacts
git ls-files | grep -E '(__pycache__|\.pyc$|\.pyo$)' \
    && echo "FAIL: tracked cache artifacts found" \
    || echo "PASS: no tracked cache artifacts"

# 1b. No generated public model artifacts in the index
git ls-files | grep -E '\.(att1|bin|pt|safetensors|gguf)$' \
    | grep -v '^models/dummy/' \
    && echo "FAIL: unexpected binary artifacts tracked" \
    || echo "PASS: only intentional tiny fixtures found"

# 1c. No temp or editor files
git ls-files | grep -E '\.(swp|swo|orig|bak|tmp)$' \
    && echo "FAIL: temp/editor files tracked" \
    || echo "PASS"

# 1d. No large accidental binaries (anything > 1 MiB not in models/dummy/)
git ls-files | xargs -I{} sh -c \
    'size=$(wc -c < "{}"); [ "$size" -gt 1048576 ] && echo "LARGE: {} ($size bytes)"' \
    || true
```

Expected passing state:

| Check | Expected result |
|-------|----------------|
| `__pycache__` / `.pyc` / `.pyo` | 0 matches |
| Public model binary formats | 0 matches outside `models/dummy/` |
| Temp/editor files | 0 matches |
| Accidental large binaries | 0 matches outside `models/dummy/` |

Tiny fixture files inside `models/dummy/` are intentional and exempt.

---

## 2. Build and Test Status

### 2.1 CPU-only build and test (required before any release)

```sh
make clean && make && make test
```

Baseline: **781 PASS 0 FAIL**.

### 2.2 Full regression runner (required before any release)

```sh
make regression
```

Or equivalently:

```sh
python3 compiler/run_full_regression.py
```

All nine post-build layers must report PASS:

| Layer | Tool |
|-------|------|
| M133 golden regressions | `compiler/check_golden_regressions.py` |
| M134 schema compatibility | `compiler/test_schema_compat.py` |
| M135 hostile-input regression | `compiler/test_hostile_inputs.py` |
| M132 pipeline smoke | `compiler/run_execution_replay_pipeline.py` |
| Cache artifact check | `git ls-files` grep |
| M149 docs lint/link check | `compiler/check_docs.py` |

### 2.3 GitHub Actions CI status

| Workflow | Runner | Required |
|----------|--------|----------|
| `ci.yml` (push / pull_request) | ubuntu-latest, CPU-only | Yes |
| `cuda-self-hosted.example.yml` | self-hosted RTX 3090 | Optional/manual |

CI must be green on `master` before any release tag is pushed.

### 2.4 CUDA signoff status

See [CUDA_VALIDATION_PLAN.md](CUDA_VALIDATION_PLAN.md) for full details.

Current status column to use in milestone entries:

| Phrase | Meaning |
|--------|---------|
| `CPU validation complete; CUDA signoff pending` | CPU CI passed; no GPU validation yet |
| `CUDA validation complete on RTX 3090` | Manual signoff completed (see signoff report) |

---

## 3. CUDA Validation Policy Summary

- CPU CI (`ci.yml`) is **not** CUDA validation.
- CUDA signoff is **manual** on a physical RTX 3090-class host.
- A self-hosted runner (`.github/workflows/cuda-self-hosted.example.yml`)
  is available as an inactive example; rename to `.yml` and configure to
  activate.
- Any milestone that adds or changes a CUDA kernel, CUDA backend path,
  or CUDA-facing API is **not complete** until the manual signoff is done.
- No silent CPU fallback when `CUDA=1` is specified.

Quick command for manual CUDA signoff:

```sh
make clean && make CUDA=1 && make test CUDA=1
python3 compiler/run_full_regression.py --cuda
```

---

## 4. Artifact Policy

| Category | Policy |
|----------|--------|
| Tiny deterministic test fixtures | Allowed in `compiler/fixtures/` and `models/dummy/` |
| Public model weights (Llama, Mistral, etc.) | Must stay outside Git; never commit |
| Generated `.att1` files from real models | Outside Git only |
| Large generated binaries | Outside Git only |
| Source model directories | Outside Git only |
| Tokenizer/vocab assets from public models | External only, unless < 1 KiB synthetic fixture |
| Compiler-generated `.pyc` / `__pycache__` | Never commit |

Enforcement: M136 full-regression runner and CI both check `git ls-files`
for tracked cache artifacts and fail if any are found.

---

## 5. Security and Hostile-Input Policy

| Boundary | Policy |
|----------|--------|
| `.att1` binary loader | Treats all input as hostile; validates magic, version, size fields before use |
| Schema validators (M134) | Reject malformed, missing-field, and future-version inputs |
| Hostile-input regression (M135) | 37 negative fixtures must fail validation cleanly |
| Backend dispatch | No silent fallback to a different backend or dtype |
| CUDA dispatch | No silent CPU fallback when CUDA is requested |
| Unsupported dtypes | Return `ATT1_UNSUPPORTED`; never silently use a different dtype |

Before any release, confirm:

```sh
python3 compiler/test_hostile_inputs.py   # 37 PASS 0 FAIL
python3 compiler/test_schema_compat.py    # 31 PASS 0 FAIL
```

---

## 6. Schema and Version Status

The following schemas are validated by the M134/M135 regression suites:

| Schema | Covered since | Notes |
|--------|--------------|-------|
| `.att1` model format | M12 | Magic + version + section fields |
| Tensor placement report | M98/M99 | `placement_report` JSON |
| Command plan | M109 | `command_plan` JSON |
| Fabric route report | M115/M116 | `fabric_route_report` JSON |
| Tensor execution plan | M125/M128 | `execution_plan` JSON |
| Integrated pipeline report | M132 | `pipeline_report` JSON |

Version policy: future version values (e.g. version=999) must be rejected
by validators; unknown *optional* fields emit `W_UNKNOWN_FIELD` (exit 0)
in default mode and are rejected in `--strict` mode.

---

## 7. Current Capability Summary

| Capability | Status |
|------------|--------|
| CPU f32 single-tile inference | Implemented |
| CPU q8 quantized inference | Implemented |
| CPU q4 quantized inference | Implemented |
| CUDA f32 single-tile inference | Implemented (manual signoff pending) |
| CUDA q8 inference | Implemented (manual signoff pending) |
| CUDA q4 inference | Implemented (manual signoff pending) |
| Single-tile inference pipeline | Implemented |
| Multi-tile cluster inference | Implemented (simulated fabric) |
| Pre-tokenized input (`--input-token-ids`) | Implemented |
| Source text inference (`--prompt`) | Implemented |
| Tensor placement reports | Implemented (M98/M99) |
| Placement advisory / scenario comparison | Implemented |
| AIMU command queue simulation | Implemented |
| AIMU MMIO userspace emulator | Implemented |
| AIMU device/DMA/host simulation | Implemented |
| Command and fabric replay | Implemented (M132) |
| Golden regression baselines | Implemented (M133) |
| Schema compatibility regression | Implemented (M134) |
| Hostile-input regression suite | Implemented (M135) |
| Local full-regression runner | Implemented (M136) |
| CPU-only GitHub Actions CI | Implemented (M137) |
| CUDA signoff plan and self-hosted runner example | Implemented (M138) |

---

## 8. Known Non-Goals and Boundaries

These are intentional **out-of-scope** items for all current and near-term
milestones. Do not implement or claim them in documentation.

| Non-goal | Notes |
|----------|-------|
| Production ASIC design or tape-out | Phase 1 is simulation only |
| Real PCIe endpoint or PCIe driver | No actual hardware |
| Linux kernel driver | No kernel module |
| FPGA RTL or synthesis | No HDL |
| Public cloud deployment | No hosted inference |
| Mobile / Android / Vulkan / OpenCL targets | Not planned |
| Patent claim language in repo docs | Claims drafts stay private |
| Public model weights in Git | Stay external |
| Real MMIO hardware access | Userspace emulator only |

---

## 9. Patent-Sensitive Handling

ATT-1 documentation should describe **architecture, simulator behavior,
and API contracts only**. Follow these rules:

- Use **engineering wording**: "the simulator implements", "the tile
  emulates", "the command queue models".
- Do **not** use claim language: "the invention", "claim 1", "wherein",
  "a method comprising", "novel approach to".
- Do **not** include unpublished implementation claims in any file that
  is committed to a public or shared repository.
- Keep invention disclosures, patent application drafts, and attorney
  communications in a **private** location entirely separate from this repo.
- If a doc comment describes a novel algorithm or architecture, phrase it
  as a **design note**, not a claim.

Quick review before sharing:

```sh
grep -rn --include="*.md" --include="*.h" --include="*.c" \
    -iE '(claim [0-9]|wherein|the invention|novel method|patent pending)' \
    docs/ include/ src/ compiler/ \
    && echo "REVIEW: possible claim language found" \
    || echo "PASS: no obvious claim language"
```

---

## 10. Outside-Review Checklist

Run through this list before sending the repository to any external party.

- [ ] `make clean && make && make test` — **781 PASS 0 FAIL**
- [ ] `make regression` — all nine layers PASS
- [ ] `make docs-check` — PASS (0 errors)
- [ ] GitHub Actions CI is green on `master`
- [ ] No tracked `__pycache__` / `.pyc` / `.pyo` (see §1)
- [ ] No public model artifacts in the index (see §1)
- [ ] No temp or editor files in the index (see §1)
- [ ] `README.md` points to safe documentation
- [ ] `docs/OPERATION_LOG.md` current milestone entry is accurate
- [ ] CUDA signoff status is accurately labeled in all recent milestone entries
- [ ] No patent-claim wording in any committed doc (see §9 grep)
- [ ] `docs/CUDA_VALIDATION_PLAN.md` reflects current signoff state
- [ ] `docs/RELEASE_CANDIDATE_M150.md` review decision is current
- [ ] `docs/RELEASE_READINESS.md` (this file) is current

---

## 11. Recommended Next Milestones

Milestones M140–M150 are complete. See `docs/OPERATION_LOG.md` for their full entries.

| Milestone | Working title | Scope |
|-----------|---------------|-------|
| M150 | Release candidate checkpoint | Release candidate summary doc, validation baselines, review decision — **current milestone** |
| M151 | API opacity and refactor plan | Migrate `int`-returning init functions to `att1_status_t`; opacify `att1_kv_mmu` struct; resolve alias duplicates in `att1_status.h` |
| M152 | Deeper fuzzing and coverage expansion | libFuzzer / AFL++ integration; coverage measurement; expand hostile-input fixture set beyond 37 |
| M153 | Release package dry-run | `git archive` tarball verification; reviewer quick-start validation on clean VM |
| M154 | External review response log | Structured log for tracking outside reviewer questions, findings, and responses |
| M155 | Public small-model demo policy | Define opt-in public SmolLM2-135M demo with external weight download |
| M156 | Self-hosted CUDA runner decision | Decide whether to activate `cuda-self-hosted.example.yml` or formally defer |
| M157 | Tensor-level execution simulator next slice | Advance AIMU EXEC simulation from control-flow-only to partial tensor-math validation |
