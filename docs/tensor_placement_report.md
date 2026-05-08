# Tensor-Level Placement Report Schema (Milestone 98)

This document defines the canonical schema for ATT-1 tensor-level AIMU
placement reports.  A placement report is a structured, serializable record
that describes how model tensors (or tensor slices) are assigned to tiles and
what the estimated memory, bandwidth, and fabric traffic implications are.

Placement reports are **advisory** in M98.  They describe a planned placement
without changing runtime execution.  Future milestones consume this schema:

| Milestone | Role of placement report |
|---|---|
| M99 | Validator input — checks §3 records against placement rules |
| M100 | `att1-size --placement PATH` consumes §2 tile summaries for per-tile capacity table |
| M101 | Advisory proposal tool produces §1–§3 from a model config and a placement policy |
| M102 | Opt-in CPU execution uses §3 records to route matmul/attention ops |

---

## 1. Report Header

The report header identifies the model, configuration, and planning targets
that were used to produce the placement.

### 1.1 Key-value text format

```
report_version          = 1
model_name              = <string>
artifact_path           = <path or empty>
config_source           = <"model_header" | "cli" | "manual" | "unknown">
dtype                   = <"f32" | "q8" | "q4">
quantization_family     = <"none" | "per_row_q8" | "per_group_q4">
n_layers                = <int>
d_model                 = <int>
n_heads                 = <int>
n_kv_heads              = <int>
ffn_hidden              = <int>
vocab_size              = <int>
tile_count              = <int>
tile_memory_mib         = <int or "unset">
target_context_length   = <int>
target_sessions         = <int>
target_tokens_per_sec   = <float or "unset">
fabric_gib_sec          = <float or "unset">
placement_policy        = <"layer_wise" | "head_wise" | "vocab_split" | "mixed" | "unknown">
report_timestamp        = <ISO-8601 or "unknown">
```

### 1.2 JSON shape

```json
{
  "report_version": 1,
  "header": {
    "model_name": "",
    "artifact_path": "",
    "config_source": "model_header",
    "dtype": "f32",
    "quantization_family": "none",
    "n_layers": 0,
    "d_model": 0,
    "n_heads": 0,
    "n_kv_heads": 0,
    "ffn_hidden": 0,
    "vocab_size": 0,
    "tile_count": 0,
    "tile_memory_mib": null,
    "target_context_length": 0,
    "target_sessions": 1,
    "target_tokens_per_sec": null,
    "fabric_gib_sec": null,
    "placement_policy": "layer_wise",
    "report_timestamp": ""
  }
}
```

**Field rules:**

| Field | Required | Notes |
|---|---|---|
| `report_version` | Yes | Must be `1` for M98 schema |
| `model_name` | Yes | Empty string is valid (unknown model) |
| `dtype` | Yes | Must be one of `f32`, `q8`, `q4` |
| `quantization_family` | Yes | Must be consistent with `dtype` |
| `tile_count` | Yes | Must be ≥ 1 |
| `tile_memory_mib` | No | `null`/`"unset"` if not known |
| `target_context_length` | Yes | Must be ≥ 1 |
| `target_sessions` | Yes | Must be ≥ 1 |
| `target_tokens_per_sec` | No | `null`/`"unset"` if not targeted |
| `fabric_gib_sec` | No | `null`/`"unset"` if not targeted |
| `placement_policy` | Yes | See §3 for policy definitions |

---

## 2. Tile/AIMU Summary Records

One record per tile.  The `tiles` list is ordered by `tile_id`.

### 2.1 Key-value text format

One block per tile, separated by blank lines:

```
[tile 0]
tile_id                          = 0
aimu_id                          = 0
layer_range_start                = 0
layer_range_end                  = 15
assigned_tensor_count            = 126
assigned_tensor_slice_count      = 0
model_bytes                      = 1234567890
kv_bytes                         = 16777216
activation_bytes_per_token       = 65536
logits_bytes_per_token           = 0
fabric_payload_bytes_per_token   = 131072
fabric_packets_per_token         = 16
memory_utilization_percent       = 72.4
capacity_status                  = PASS
bandwidth_status                 = PASS
```

`layer_range_start` / `layer_range_end` are `"n/a"` when layer-wise placement
does not apply (e.g., for vocab-split tiles or activation-only tiles).

`logits_bytes_per_token` is nonzero only for tiles that hold an lm_head slice.

### 2.2 JSON shape

```json
{
  "tiles": [
    {
      "tile_id": 0,
      "aimu_id": 0,
      "layer_range_start": 0,
      "layer_range_end": 15,
      "assigned_tensor_count": 126,
      "assigned_tensor_slice_count": 0,
      "model_bytes": 1234567890,
      "kv_bytes": 16777216,
      "activation_bytes_per_token": 65536,
      "logits_bytes_per_token": 0,
      "fabric_payload_bytes_per_token": 131072,
      "fabric_packets_per_token": 16,
      "memory_utilization_percent": 72.4,
      "capacity_status": "PASS",
      "bandwidth_status": "PASS"
    }
  ]
}
```

**Status thresholds** (same as M96 `att1-size`):

| Utilization | `capacity_status` |
|---|---|
| < 80 % | `PASS` |
| 80 – 100 % | `WARN` |
| > 100 % | `FAIL` |
| tile_memory_mib unset | `UNKNOWN` |

Bandwidth thresholds mirror capacity thresholds against the
`fabric_gib_sec` target.

**Byte accounting rules:**

- `model_bytes`: sum of `(slice_end - slice_start) × element_size` for all
  weight placement records on this tile, plus q8/q4 scale and zero-point
  storage.
- `kv_bytes`: KV placement records on this tile at `target_context_length ×
  target_sessions`.
- `activation_bytes_per_token`: peak intermediate activation buffer needed on
  this tile per decode step.  For layer-wise placement: `d_model × 4` (f32)
  or `d_model × 1` (q8).  For row-split projections: partial output vector.
- `logits_bytes_per_token`: `(slice_end - slice_start) × 4` for lm_head slice
  records on this tile; 0 otherwise.
- `fabric_payload_bytes_per_token`: activation bytes broadcast to this tile
  plus partial result bytes sent from this tile per decode step.
- `fabric_packets_per_token`: number of fabric packets expected per decode step,
  based on `fabric_payload_bytes_per_token` divided by the MTU (default 512
  bytes per packet, same as existing simulator).

---

## 3. Tensor Placement Records

One record per tensor or per tensor slice.  The `tensors` list is ordered by
`tensor_id`, then by `slice_start` within that tensor.

### 3.1 Tensor categories

| Code | Name | Examples |
|---|---|---|
| `embedding` | Token embedding table | `tok_embeddings.weight` |
| `attention_q` | Query projection | `layers.N.attention.wq.weight` |
| `attention_k` | Key projection | `layers.N.attention.wk.weight` |
| `attention_v` | Value projection | `layers.N.attention.wv.weight` |
| `attention_o` | Output projection | `layers.N.attention.wo.weight` |
| `ffn_gate` | FFN gate (SwiGLU) | `layers.N.feed_forward.w_gate.weight` |
| `ffn_up` | FFN up projection | `layers.N.feed_forward.w_up.weight` |
| `ffn_down` | FFN down projection | `layers.N.feed_forward.w_down.weight` |
| `norm` | RMSNorm weight | `layers.N.attention_norm.weight`, `output_norm.weight` |
| `lm_head` | Output logit projection | `output.weight` |
| `kv_cache` | KV paged memory | (AIMU-local KV region) |
| `activation` | Intermediate activation | Token embedding vector, attention output |
| `other` | Unclassified | Any tensor not matching above |

### 3.2 Replication policies

| Policy | Meaning |
|---|---|
| `unique` | Exactly one copy across all tiles |
| `replicated` | Identical copy on every tile that uses it; checksums must match |
| `partial` | Disjoint slice on each tile; slices union to full tensor |

### 3.3 Reduction behaviors

| Code | Meaning |
|---|---|
| `none` | No reduction needed (whole-tensor or row-lookup result) |
| `sum` | Partial outputs are summed element-wise |
| `max` | Partial outputs are reduced by element-wise maximum |
| `concat` | Partial outputs are concatenated in `slice_start` order |

### 3.4 Key-value text format

```
[tensor 0]
tensor_id                = 0
tensor_name              = tok_embeddings.weight
tensor_category          = embedding
layer                    = n/a
source_shape             = [32000, 4096]
placed_shape             = [32000, 4096]
dtype                    = f32
quantization             = none
scale_bytes              = 0
packed_bytes             = 524288000
owner_tile               = 0
owner_aimu               = 0
slice_axis               = none
slice_start              = n/a
slice_end                = n/a
replication_policy       = unique
routing_requirements     = none
reduction_behavior       = none
checksum                 = 0x0000000000000000
placement_status         = placed
```

When a tensor is split, multiple records share the same `tensor_id` and
`tensor_name`, with different `owner_tile`, `slice_start`, and `slice_end`:

```
[tensor 4 slice 0]
tensor_id                = 4
tensor_name              = layers.0.attention.wq.weight
tensor_category          = attention_q
layer                    = 0
source_shape             = [4096, 4096]
placed_shape             = [2048, 4096]
dtype                    = q8
quantization             = per_row_q8
scale_bytes              = 8192
packed_bytes             = 2097152
owner_tile               = 0
owner_aimu               = 0
slice_axis               = 0
slice_start              = 0
slice_end                = 2048
replication_policy       = partial
routing_requirements     = broadcast_activation
reduction_behavior       = sum
checksum                 = 0x0000000000000000
placement_status         = placed

[tensor 4 slice 1]
tensor_id                = 4
tensor_name              = layers.0.attention.wq.weight
tensor_category          = attention_q
layer                    = 0
source_shape             = [4096, 4096]
placed_shape             = [2048, 4096]
dtype                    = q8
quantization             = per_row_q8
scale_bytes              = 8192
packed_bytes             = 2097152
owner_tile               = 1
owner_aimu               = 1
slice_axis               = 0
slice_start              = 2048
slice_end                = 4096
replication_policy       = partial
routing_requirements     = broadcast_activation
reduction_behavior       = sum
checksum                 = 0x0000000000000000
placement_status         = placed
```

### 3.5 JSON shape

```json
{
  "tensors": [
    {
      "tensor_id": 0,
      "tensor_name": "tok_embeddings.weight",
      "tensor_category": "embedding",
      "layer": null,
      "source_shape": [32000, 4096],
      "placed_shape": [32000, 4096],
      "dtype": "f32",
      "quantization": "none",
      "quantization_group_size": null,
      "scale_bytes": 0,
      "packed_bytes": 524288000,
      "owner_tile": 0,
      "owner_aimu": 0,
      "slice_axis": null,
      "slice_start": null,
      "slice_end": null,
      "replication_policy": "unique",
      "routing_requirements": "none",
      "reduction_behavior": "none",
      "checksum": "0x0000000000000000",
      "placement_status": "placed"
    }
  ]
}
```

**`placement_status` values:**

| Value | Meaning |
|---|---|
| `placed` | Fully assigned to a tile |
| `unplaced` | Not yet assigned |
| `partial` | Some slices placed, some unplaced |
| `overflow` | Placed but tile capacity exceeded |
| `invalid` | Placement record fails a validation rule |

**`routing_requirements` values:**

| Value | Meaning |
|---|---|
| `none` | No special routing; activation flows in layer-pipeline order |
| `broadcast_activation` | Full activation vector must be broadcast to this tile before op |
| `fixed_path` | Activation must route to exactly this tile (KV owner) |
| `lookup_route` | Token ID lookup must be routed to the tile holding that vocab row |
| `concat_return` | Partial result must be returned and concatenated in slice order |

---

## 4. Validation Report Fields

The placement report carries a `validation` section that summarises any rule
violations detected.  Validation rules are defined in
[shard_metadata.md §13.5](shard_metadata.md).

### 4.1 Key-value text format

```
[validation]
total_tensors_checked          = 42
total_tensor_slices_checked    = 84
validation_pass                = false
missing_tensor_coverage        = 0
overlapping_slices             = 0
invalid_tile_ids               = 0
dtype_quant_mismatch           = 0
q4_group_alignment_violations  = 0
incomplete_reductions          = 1
unsupported_replication        = 0
capacity_overflow_tiles        = 0
bandwidth_overflow_tiles       = 0
total_warnings                 = 1
total_failures                 = 0
```

`validation_pass` is `true` only when all failure counts are zero.
Warnings are non-zero counts of `WARN`-level conditions; they do not set
`validation_pass = false`.

### 4.2 JSON shape

```json
{
  "validation": {
    "total_tensors_checked": 42,
    "total_tensor_slices_checked": 84,
    "validation_pass": false,
    "missing_tensor_coverage": 0,
    "overlapping_slices": 0,
    "invalid_tile_ids": 0,
    "dtype_quant_mismatch": 0,
    "q4_group_alignment_violations": 0,
    "incomplete_reductions": 1,
    "unsupported_replication": 0,
    "capacity_overflow_tiles": 0,
    "bandwidth_overflow_tiles": 0,
    "total_warnings": 1,
    "total_failures": 0,
    "violations": [
      {
        "rule": 8,
        "severity": "warning",
        "tensor_id": 12,
        "tensor_name": "layers.5.attention.wo.weight",
        "slice_start": null,
        "slice_end": null,
        "message": "Split tensor has no explicit reduction_behavior; defaulting to sum"
      }
    ]
  }
}
```

**Violation severity levels:**

| Severity | Meaning |
|---|---|
| `info` | Informational; no action required |
| `warning` | Non-blocking; placement is technically valid but suboptimal |
| `error` | Blocking; placement fails a required validation rule |

---

## 5. Human-Readable Output Format

The human-readable report is emitted by tools that produce placement reports
(M101) and consumed by the validator (M99) when `--text` is requested.

### 5.1 Tile summary table

```
Tile placement summary (4 tiles, layer-wise)
============================================================
 Tile  Layers    Tensors  Slices  Model MiB   KV MiB  Util%  Cap  BW
    0    0–15       126       0    1177.4      16.0   72.4%  PASS PASS
    1   16–31       126       0    1177.4      16.0   72.4%  PASS PASS
    2   32–47       126       0    1177.4      16.0   72.4%  PASS PASS
    3   48–63       126       0    1177.4      16.0   72.4%  PASS PASS
============================================================
Total model: 4709.7 MiB    Total KV: 64.0 MiB    All tiles: PASS
```

### 5.2 Overloaded tile report

Emitted when any tile has `capacity_status = WARN` or `FAIL`:

```
Overloaded tiles:
  Tile 2: model 1820.3 MiB + KV 32.0 MiB = 1852.3 MiB / 1536 MiB (120.6%)  FAIL
```

### 5.3 Largest tensor report

```
Largest tensors by placed bytes:
  1. output.weight           (lm_head,    tile 0, f32,     524.3 MiB)
  2. tok_embeddings.weight   (embedding,  tile 0, f32,     524.3 MiB)
  3. layers.0.wq.weight      (attn_q,     tile 0, q8,       16.0 MiB)
  ...
```

### 5.4 Placement warnings

```
Placement warnings:
  [W] layers.5.attention.wo.weight (tensor_id=12): split tensor has no explicit
      reduction_behavior — will default to sum. Set reduction_behavior=sum to suppress.
```

### 5.5 Suggested remediation

Remediation suggestions are produced when `capacity_status` or
`bandwidth_status` is `WARN` or `FAIL`, or when validation warnings are
present.  Each suggestion is tagged with its trigger condition:

```
Suggested remediation:
  [capacity-fail]   Increase tile count (currently 4 → try 8).
  [capacity-fail]   Split lm_head across all tiles (vocab split).
  [capacity-fail]   Convert model to q4 to reduce model_bytes by ~50%.
  [capacity-warn]   Reduce target context length or sessions.
  [capacity-warn]   Increase tile_memory_mib target.
  [bandwidth-warn]  Increase fabric_gib_sec target or reduce activation traffic
                    by using head-wise placement to keep QKV local.
```

Suggestions are advisory only.  They do not modify the placement.

---

## 6. JSON Schema Summary

The complete JSON document shape for a placement report:

```json
{
  "report_version": 1,
  "header": { /* §1.2 */ },
  "tiles": [ /* §2.2 — one object per tile */ ],
  "tensors": [ /* §3.5 — one object per tensor or tensor slice */ ],
  "validation": { /* §4.2 */ },
  "warnings": [
    {
      "category": "capacity",
      "tile_id": 2,
      "tensor_id": null,
      "message": "Tile 2 memory utilization 120.6% exceeds 100%"
    }
  ],
  "failures": [
    {
      "category": "validation",
      "rule": 3,
      "tile_id": null,
      "tensor_id": 7,
      "tensor_name": "layers.3.attention.wq.weight",
      "message": "Slices [0,2048) and [1024,4096) overlap"
    }
  ],
  "remediation": [
    {
      "trigger": "capacity-fail",
      "suggestion": "Split lm_head across all tiles (vocab split)"
    }
  ]
}
```

**Stability rules for JSON keys:**

1. All integer byte counts are JSON integers (not strings).
2. All status fields (`capacity_status`, `bandwidth_status`, `validation_pass`,
   `placement_status`, severity) are JSON strings from the fixed vocabularies
   defined in §2, §3, and §4.
3. `null` is used for optional absent values (not `0` or `""`).
4. Lists (`tiles`, `tensors`, `warnings`, `failures`, `remediation`) are always
   present, even if empty.
5. `report_version` must be checked before parsing any other field; a future
   schema version increment requires a migration note in this document.
6. Key names use `snake_case` throughout.
7. No floating-point values in `tiles` or `tensors` except
   `memory_utilization_percent`.  All bandwidth figures are integer bytes.

---

## 7. Relationship to Existing Tools

### 7.1 M96 `att1-size` estimator

`att1-size` currently computes per-tile estimates using an even-split
heuristic (total model bytes ÷ tile count).  The tensor-level placement report
schema provides the basis for replacing that heuristic with per-tile exact
byte counts:

| `att1-size` field | Current source | Future source (M100) |
|---|---|---|
| `model_bytes_per_tile` | `total_params × dtype_bytes / n_tiles` | Sum of `packed_bytes + scale_bytes` from §3 records for this tile |
| `kv_bytes` | `kv_bytes(max_ctx) / n_tiles` | Sum of KV placement records (`tensor_category=kv_cache`) on this tile |
| `tile_capacity_status` | Heuristic utilization vs `--tile-memory-mib` | `capacity_status` from §2 tile summary record |
| `fabric_bandwidth_status` | Heuristic fabric bytes vs `--fabric-gib-sec` | `bandwidth_status` from §2 tile summary record |

The M100 `--placement PATH` flag will accept a placement report JSON and
replace the heuristic fields with §2 tile summary records.

### 7.2 Shard metadata records

The existing 120-byte shard metadata records (shard_metadata.md §3) are the
per-file source of truth for tensor-to-tile assignment.  The tensor-level
placement report is a derived reporting layer on top of the metadata records:

| Shard metadata field | Maps to report field |
|---|---|
| `tensor_id` | `tensors[*].tensor_id` |
| `tile_id` | `tensors[*].owner_tile` |
| `owner_aimu` | `tensors[*].owner_aimu` |
| `dtype` | `tensors[*].dtype` |
| `quantization` | `tensors[*].quantization` |
| `replication_policy` | `tensors[*].replication_policy` |
| `routing_requirements` | `tensors[*].routing_requirements` |
| `reduction_behavior` | `tensors[*].reduction_behavior` |
| `checksum` | `tensors[*].checksum` |
| *(M97 extension)* `slice_axis` | `tensors[*].slice_axis` |
| *(M97 extension)* `slice_start` | `tensors[*].slice_start` |
| *(M97 extension)* `slice_end` | `tensors[*].slice_end` |
| *(M97 extension)* `tensor_category` | `tensors[*].tensor_category` |

The M99 validator reads shard metadata records and produces a placement report
(§3–§4) as output.  It does not modify the `.att1` binary artifact.

### 7.3 M99 validator

The M99 validator (`compiler/validate_tensor_placement.py`) will:

1. Read a placement report JSON (§1–§3) or accept a `.att1` file and extract
   the shard metadata section.
2. Apply validation rules from shard_metadata.md §13.5 (checks 1–12).
3. Produce a new placement report JSON with the `validation` section (§4)
   populated.
4. Exit 0 if `validation_pass = true`; exit 1 if any errors; exit 2 if
   warnings only.

### 7.4 M100 estimator integration

The M100 `att1-size --placement PATH` option will:

1. Parse the placement report header (§1) to validate the model config matches
   the current model.
2. Read the tile summary records (§2) and emit a per-tile capacity table using
   exact `model_bytes` and `kv_bytes` from §2 instead of the even-split
   heuristic.
3. Propagate `capacity_status` and `bandwidth_status` from §2 directly into
   the `att1-size` `[tile_capacity_estimate]` section.
4. Emit a `placement_source = tensor_level` line in the output to distinguish
   heuristic vs exact estimates.

---

## 8. Non-Goals for M98

- No execution scheduling change.
- No binary format change (the 200-byte extended record remains a M97 proposal).
- No new C source files.
- No CUDA change.
- No q4/q8/f32 tolerance or behavior change.
- No `compiler/tensor_placement.py` implementation (skeleton only — M101).
- No patent claim language.
- No PCIe register map.

---

## Related Documents

- [shard_metadata.md](shard_metadata.md) — shard metadata binary format and
  validation rules (§13.5 rules referenced by §4 above)
- [aimu_architecture.md](aimu_architecture.md) — AIMU placement architecture
  and fabric routing implications (§9)
- [cluster_inference.md](cluster_inference.md) — cluster sharding and
  activation routing
- [backend.md](backend.md) — backend vtable and operator set
- [kv_mmu.md](kv_mmu.md) — KV-MMU paged session memory

---

## 9. Placement Report Validator (Milestone 99)

`compiler/validate_tensor_placement_report.py` is the reference validator for
placement reports produced according to this schema.

### 9.1 Usage

```
python3 compiler/validate_tensor_placement_report.py \
    --report PATH [--report-json PATH] [--strict]
```

| Option | Description |
|---|---|
| `--report PATH` | Required. Path to the placement report JSON file to validate. |
| `--report-json PATH` | Optional. Write a machine-readable JSON validation summary to PATH. |
| `--strict` | Optional. Promote warnings to errors (validation fails on any issue). |

### 9.2 Exit Codes

| Code | Meaning |
|---|---|
| `0` | Validation passed — zero errors detected (warnings may be present). |
| `1` | Validation failed — one or more errors detected. |
| `2` | Report could not be parsed — malformed JSON or file not found. |

### 9.3 Checks Performed

The validator applies the following checks in order:

**Header checks (§1):**

- `report_version` present and in the supported set (`{1}`).
- `dtype` present and one of `f32`, `q8`, `q4`, `bf16`.
- `quantization_family` present and recognised.
- `tile_count` present, a positive integer.
- `target_context_length` ≥ 1 if present.
- `target_sessions` ≥ 1 if present.
- `tile_memory_mib` > 0 if present.
- `fabric_gib_sec` > 0 if present.

**Tile record checks (§2):**

- Each tile has a non-null `tile_id` integer.
- All `tile_id` values are unique.
- All `tile_id` values are within `[0, tile_count)`.
- `model_bytes`, `kv_bytes`, `activation_bytes_per_token`, `logits_bytes_per_token`,
  `fabric_payload_bytes_per_token` are non-negative integers.
- `memory_utilization_percent` is a non-negative number.
- `capacity_status` is one of `PASS`, `WARN`, `FAIL`, `UNKNOWN`.
- Consistency: if `capacity_status = PASS` and `memory_utilization_percent > 100`,
  that is an error.
- `bandwidth_status` is one of `PASS`, `WARN`, `FAIL`, `UNKNOWN`.

**Tensor record checks (§3):**

- Each record has `tensor_id` or `tensor_name`.
- `tensor_category` is recognised (or `other`).
- `source_shape` and `placed_shape` are lists of positive integers.
- `dtype` is recognised.
- `quantization = per_group_q4` requires a valid `quantization_group_size`.
- `packed_bytes`, `scale_bytes` are non-negative integers if present.
- `owner_tile` is present, an integer, within `[0, tile_count)`, and present
  in the `tiles` list.
- `slice_axis` is a valid axis for the tensor rank.
- `slice_start` and `slice_end` define a non-empty range within the tensor
  dimension.
- For q4 tensors: `slice_start` and `slice_end` must be aligned to `quantization_group_size`.
- `replication_policy` is `unique`, `replicated`, or `partial`.
- `reduction_behavior` is `none`, `sum`, `max`, or `concat`.
- Duplicate `unique` placement records for the same `tensor_id` are errors.

**Coverage and overlap checks:**

- Per `tensor_id`: overlapping slice ranges across tiles are errors (unless
  all records have `replication_policy = replicated`).
- Split tensors with `reduction_behavior = none` for all slices produce a
  warning (rule 8).
- Gaps in slice coverage for fully-placed tensors produce a warning (rule 3).

### 9.4 Fixture Files

| File | Expected outcome |
|---|---|
| `compiler/fixtures/placement_report_valid.json` | exit 0; `validation: pass` |
| `compiler/fixtures/placement_report_invalid_tile_id.json` | exit 1; `owner_tile=99` error |
| `compiler/fixtures/placement_report_capacity_overflow.json` | exit 1; utilization 204800% with `PASS` status |
| `compiler/fixtures/placement_report_overlapping_slices.json` | exit 1; slices `[0,48)` and `[32,64)` overlap |
| `compiler/fixtures/placement_report_q4_bad_align.json` | exit 1; `slice_start=60` not aligned to group_size=32 |

### 9.5 JSON Output Format

When `--report-json PATH` is used, the output file contains:

```json
{
  "status": "pass",
  "total_warnings": 0,
  "total_errors": 0,
  "warnings": [],
  "failures": [],
  "report_path": "compiler/fixtures/placement_report_valid.json",
  "tile_count_header": 2,
  "tensor_record_count": 5,
  "tile_record_count": 2
}
```

The `status` field is `"pass"` or `"fail"`.  `total_errors` and
`total_warnings` are non-negative integers.  `warnings` and `failures` are
lists of violation records (same schema as §4.2 `violations`).

---

## 10. Report Emitter: att1-size --placement-report-json (Milestone 100)

`att1-size` (the sizing/estimation tool) can emit a tensor-level placement
report JSON file via `--placement-report-json PATH`.

### 10.1 Usage

```
./build/att1-size --preset tiny-dummy \
    [--tiles N] [--context N] [--dtype f32|q8|q4] \
    [--tile-memory-mib N] [--sessions N] \
    [--target-tokens-per-sec N] [--fabric-gib-sec N] \
    --placement-report-json PATH
```

Works with all three input modes: `--preset`, `--config`, and manual shape
arguments.  The `--placement-report-json` option is independent of `--json`
(the existing machine-readable scaling report) — both can be used together.

### 10.2 Report Content

The emitted JSON follows the M98 schema (§1–§8 above) and passes the M99
validator (`compiler/validate_tensor_placement_report.py`).

**Header (§1):** model name, config source (`preset`/`config`/`config`),
dtype (f16 is mapped to f32 since the validator does not accept f16),
quantization family, shape summary, tile count, context length, sessions,
tile memory capacity and fabric bandwidth if provided.

**Tile records (§2):** one record per tile.  Uses layer-wise placement:
tile `t` owns `ceil(n_layers / tile_count)` consecutive layers.  Tile 0
holds `tok_embeddings`; the last tile holds `output_norm` and `lm_head`.
Computes `model_bytes` (dtype-aware), `kv_bytes` (f32, per session),
`activation_bytes_per_token`, `logits_bytes_per_token`,
`fabric_payload_bytes_per_token` and `fabric_packets_per_token`.
`capacity_status` and `bandwidth_status` are `UNKNOWN` when no tile memory
or fabric bandwidth limit is provided; otherwise `PASS` / `WARN` / `FAIL`
as per the §2 thresholds.

**Tensor records (§3):** one record per major weight tensor per tile:
`tok_embeddings`, per-layer `wq`, `wk`, `wv`, `wo`, `w_gate`, `w_up`,
`w_down`, `attention_norm`, `ffn_norm`, plus `output.norm` and `output.weight`
(lm_head) on the last tile.  All tensors use `replication_policy = unique`
(layer-wise, no row-slicing).  `slice_start = 0`, `slice_end = dim0` (the
full first dimension), so q4 group-size alignment rules are always satisfied.
Norm vectors are always emitted as `dtype = f32` regardless of model dtype.

**Warnings / failures (§4):** capacity overflow and bandwidth overflow tiles
are recorded in `failures[]`; marginal tiles (WARN) in `warnings[]`.
Synthetic/non-executable presets generate a warning entry.

### 10.3 Passes M99 Validator

- `--preset tiny-dummy`: validator exits 0, `validation: pass` (0 errors,
  0 warnings with no capacity/bandwidth limits specified).
- `--preset gpt-oss-120b-shape --tiles 8 --context 8192 --dtype q4`:
  validator exits 0, `validation: pass` (1 synthetic warning, 0 errors).
- Reports with capacity overflow tiles are valid JSON; the failures array
  is populated but the report itself is well-formed.

### 10.4 Non-goals for M100

- No execution scheduling change.
- No binary format change.
- No CUDA change.
- No q4/q8/f32 tolerance change.
- The emitter uses architectural estimates, not measured tensor sizes;
  it does not read or write `.att1` model files.
- No cross-tile tensor slicing (all placements are full-tensor, layer-wise).
