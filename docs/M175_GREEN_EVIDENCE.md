# M175 Green Evidence Packet

**Status:** Open. M175 remains HOLD until this packet exists and passes.

This document defines the evidence packet that turns the M175 FPGA gate from
HOLD to GO. It is part of M175; it does not create a new milestone and it does
not renumber M176-M181.

## Required Inputs

| Input | Policy |
|---|---|
| Source model directory | Local filesystem path only; target class is SmolLM2-135M or equivalent 100M-200M parameter decoder model |
| f32 `.att1` artifact | Local generated artifact; do not commit public-model artifacts |
| q8 `.att1` artifact | Local generated artifact; do not commit public-model artifacts |
| Long-context token file | Local pretokenized input; M172 requires at least 128 prompt tokens unless using explicit tiny-fixture smoke mode outside the green packet |
| Placement report | M98/M100-style placement report for the same model/artifact family |
| Fabric route report | M115/M117-style route report with activation routes |
| Host-access decision | One local Markdown note selecting VFIO, vendor XDMA userspace, or LitePCIe userspace bridge |
| Minimum FPGA scope | One local Markdown note listing the BAR0, queue, completion, DMA, counter, and replay paths allowed for M176-M181 |
| Trace packet | One local JSON/Markdown packet with production-like or partner-style trace fields and explicit pass/fail criteria |

## Recommended Starting Model

Start with `HuggingFaceTB/SmolLM2-135M`, not SmolLM3-3B. The M171 validator
intentionally accepts the 100M-200M parameter band so the first green packet is
small enough to convert, run, inspect, and repeat locally without turning M175
into a large-model operations project.

Source: <https://huggingface.co/HuggingFaceTB/SmolLM2-135M>

Local source snapshot for the M175 work is expected at
`~/Models/SmolLM2-135M`. The initial local snapshot was cloned from the source
above at commit `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; keep that model
directory outside the ATT-1 repository.

SmolLM3-3B is useful as a later scale-up target, but it is outside the current
M175 green-packet band. Do not relax the M171 size gate until the 135M-class
packet passes and the conversion/report pipeline is proven end to end.

## Runner

Use `compiler/run_m175_green_packet.py` to run the M171-M174 validators and
bind their outputs to the host-access, FPGA-scope, and trace-packet evidence:

```sh
python3 compiler/run_m175_green_packet.py \
  --model-dir ~/Models/SmolLM2-135M \
  --att1-f32 ~/Models/att1/SmolLM2-135M/model_f32.att1 \
  --att1-q8 ~/Models/att1/SmolLM2-135M/model_q8.att1 \
  --tokens-file ~/Models/att1/SmolLM2-135M/long_prompt.ids \
  --placement-report ~/Models/att1/SmolLM2-135M/placement_report.json \
  --route-report ~/Models/att1/SmolLM2-135M/fabric_routes.json \
  --host-access-decision ~/Models/att1/SmolLM2-135M/host_access_decision.md \
  --fpga-scope ~/Models/att1/SmolLM2-135M/min_fpga_scope.md \
  --trace-packet ~/Models/att1/SmolLM2-135M/trace_packet.json \
  --out-dir ~/Models/att1/SmolLM2-135M/m175_green_packet
```

The runner writes:

| Output | Purpose |
|---|---|
| `m171_two_tile.json` | Real two-tile q8/f32 validation report |
| `m172_beachhead.json` | Beachhead latency, memory movement, KV, and fabric metrics |
| `m173_capacity.json` | Capacity-budget and KV-page decision |
| `m174_activation_precision.json` | f32-vs-bf16 activation packet decision |
| `m175_green_manifest.json` | Packet manifest tying paths, reports, stdout/stderr logs, and pass/fail state together |

## Current State

The source model has been cloned locally and f32/q8 artifacts have been emitted
under `~/Models/att1/SmolLM2-135M`. See
[M175_SMOLLM2_ARTIFACT_RECORD.md](M175_SMOLLM2_ARTIFACT_RECORD.md). SmolLM2-135M
uses grouped-query attention (`num_attention_heads=9`,
`num_key_value_heads=3`); the converter expands the grouped K/V projection
heads into legacy full-head `.att1` tensors so the existing v1/v2 format and
runtime can load the artifacts without a format break.

The green packet still cannot pass honestly until the long-context token file,
placement report, route report, host-access decision, minimum FPGA scope note,
and trace packet exist and the runner produces a passing manifest.

## Gate Rule

M175 may flip from HOLD to GO only when:

1. `m175_green_manifest.json` reports `result: pass`.
2. The manifest references real external-model M171-M174 reports, not
   `--allow-tiny-fixture` smoke outputs.
3. The host-access decision selects exactly one userspace path.
4. The FPGA scope excludes tensor math and custom kernel-driver work.
5. The trace packet has explicit pass/fail criteria.
6. `python3 compiler/check_docs.py`, `git diff --check`, `make test`, and
   `make regression` pass after the evidence is recorded.
