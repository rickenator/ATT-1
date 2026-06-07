# ATT-1 Trace Diff Tool — M94

`compiler/trace_diff.py` compares two `att1-bench` output files and reports
execution, counter, and numerical differences between backends, shard plans,
or modes.

---

## Overview

`att1-bench` writes structured `key=value` counters to stdout after each run.
`trace_diff.py` reads two such output files (called A and B) and produces a
side-by-side comparison with a `same` / `DIFF` / `MISSING` tag per field.

The tool is pure Python, has no external dependencies, and requires only the
`att1-bench` output files — no live bench re-run is needed.

---

## Usage

```sh
# compare the same file to itself (no differences expected)
python3 compiler/trace_diff.py build/run_a.txt build/run_a.txt

# compare cpu-f32 vs cpu-q8 single-tile runs
python3 compiler/trace_diff.py build/f32_single.txt build/q8_single.txt

# cross-mode: single vs cluster, with JSON report
python3 compiler/trace_diff.py build/single.txt build/cluster.txt \
    --report-json build/diff_single_vs_cluster.json

# cross-backend: f32 vs cuda, with JSON
python3 compiler/trace_diff.py build/f32.txt build/cuda.txt \
    --report-json build/diff_f32_vs_cuda.json
```

---

## Input format

Each input file must be a valid `att1-bench` output: a sequence of
`key=value` lines plus optional `layer[N].*` and `tile[N].*` counter lines.
A file with no parseable `key=value` content causes the tool to exit
non-zero with an error message.

Example bench output (abbreviated):

```
mode=single
shard_plan=runtime
backend=cpu-f32
tokenizer=external
prompt_tokens=3
generated_tokens=1
last_token=0
tokens_decoded=1
token_time_us_total=28
logits_bytes_produced=64
fabric_packets_sent=0
kv_appends=2
kv_key_reads=2
kv_value_reads=2
tile_layer_executions=2
layer[0].executions=1 time_us=14 kv_appends=1
layer[1].executions=1 time_us=14 kv_appends=1
tile[0].layers=2 activation_bytes_sent=0 logits_bytes=64
```

---

## Output format

```
# ATT-1 trace diff (M94)
file_a: build/f32_single.txt
file_b: build/q8_single.txt

  field                                    a                    b                    status
  --------------------------------------------------------------------------------------
  mode                                     single               single               same
  shard_plan                               runtime              runtime              same
  backend                                  cpu-f32              cpu-q8               DIFF
  tokenizer                                external             external             same
  generated_tokens                         1                    1                    same
  last_token                               0                    3                    DIFF
  token_time_us_total                      28                   31                   DIFF
  logits_bytes_produced                    64                   64                   same
  fabric_packets_sent                      0                    0                    same
  kv_appends                               2                    2                    same
  ...

differences: 3  missing: 0
result: pass
report: ok
```

---

## Status values per field

| Tag       | Meaning |
|-----------|---------|
| `same`    | field present in both files with equal values |
| `DIFF`    | field present in both files but values differ |
| `MISSING` | field absent in one file |

Fields absent from **both** files are silently skipped.

---

## Exit codes

| Exit code | Meaning |
|-----------|---------|
| `0`       | both files parsed successfully; diff report written |
| `1`       | one or both files could not be opened or contained no parseable fields |

Differences between the two files do **not** cause a non-zero exit.  The tool
is a reporting instrument; pass/fail judgement is the caller's responsibility.

---

## JSON report (`--report-json`)

```json
{
  "file_a": "build/f32_single.txt",
  "file_b": "build/q8_single.txt",
  "differences": [
    {"field": "backend",             "a": "cpu-f32", "b": "cpu-q8", "status": "diff"},
    {"field": "last_token",          "a": "0",       "b": "3",      "status": "diff"},
    {"field": "token_time_us_total", "a": "28",      "b": "31",     "status": "diff"}
  ],
  "n_diff": 3,
  "n_missing": 0,
  "result": "pass"
}
```

---

## Fields compared

### Identity fields (string comparison)

`mode`, `shard_plan`, `backend`, `tokenizer`

### Header fields (run configuration)

`prompt_tokens`, `requested_tokens`, `benchmark_tokens`, `generated_tokens`, `last_token`

### Trace counter fields

`tokens_decoded`, `token_time_us_total`, `token_time_us_max`, `layer_time_us_total`,
`activation_bytes_sent`, `logits_bytes_produced`, `fabric_packets_sent`,
`fabric_packets_received`, `fabric_payload_bytes_sent`, `fabric_payload_bytes_received`,
`kv_appends`, `kv_key_reads`, `kv_value_reads`, `tile_layer_executions`

### Shard-meta summary fields

`shard_meta`, `shard_meta_count`, `shard_meta_assigned`, `shard_meta_unassigned`,
`shard_meta_tiles`, `shard_meta_aimus`, `shard_meta_dtype_f32`, `shard_meta_dtype_q8`

### Per-layer counters

`layer[N].executions`, `layer[N].time_us`, `layer[N].kv_appends`

### Per-tile counters

`tile[N].layers`, `tile[N].activation_bytes_sent`, `tile[N].logits_bytes`

---

## Notes on timing fields

`token_time_us_total`, `token_time_us_max`, `layer_time_us_total`, and per-layer
`time_us` fields are CPU process-time measurements from the `clock()` source.
They are expected to differ between runs even on the same input.  `trace_diff.py`
reports timing differences as `DIFF` but does not treat them specially; the
caller decides whether a timing delta is significant.

---

## Smoke test (`test_bench_smoke.c`, `check_trace_diff_smoke`)

`check_trace_diff_smoke` in `tests/test_bench_smoke.c` exercises the tool
in three scenarios using the checked-in `real_tiny_f32` and `real_tiny_q8`
fixtures:

1. **Identical** — same file diffed against itself.  Verifies `differences: 0  missing: 0`.
2. **Cross-backend** — `cpu-f32` vs `cpu-q8`.  Verifies `DIFF` for the
   `backend` field and valid JSON report.
3. **Malformed** — file with no `key=value` content.  Verifies non-zero exit.

Python-skippable: skipped silently when Python 3 is not available.
