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

---

## 11. Placement Proposal Tool (Milestone 101)

`compiler/propose_tensor_placement.py` reads an M98/M100 placement report JSON
and produces an advisory remediation report.  It does not modify the report or
any `.att1` binary artifact.

### 11.1 Usage

```
python3 compiler/propose_tensor_placement.py \
    --report PATH [--report-json PATH]
```

| Option | Description |
|---|---|
| `--report PATH` | Required. Path to M98/M100 placement report JSON. |
| `--report-json PATH` | Optional. Write machine-readable advisory JSON to PATH. |

### 11.2 Exit Codes

| Code | Meaning |
|---|---|
| `0` | Analysis complete; advisory status is `ok` or `warn`. |
| `1` | Analysis complete; advisory status is `fail` — actionable failures found. |
| `2` | Report could not be parsed (malformed JSON, missing required field, or file not found). |

### 11.3 Detection Categories

The tool detects the following conditions and maps each to a
`[trigger]` tag in the output:

| Trigger | Condition |
|---|---|
| `capacity-fail` | One or more tiles have `capacity_status = FAIL`. |
| `capacity-warn` | One or more tiles have `capacity_status = WARN` (no FAIL). |
| `bandwidth-fail` | One or more tiles have `bandwidth_status = FAIL`. |
| `bandwidth-warn` | One or more tiles have `bandwidth_status = WARN` (no FAIL). |
| `kv-pressure` | KV bytes exceed 20% (warn) or 40% (fail) of tile memory. |
| `logits-traffic` | Logits traffic is heavy and bandwidth is constrained. |
| `load-imbalance` | Tile utilization spread exceeds 20% of the mean. |
| `oversized-tensor` | A single tensor's `packed_bytes` exceeds tile memory. |
| `q4-alignment` | Q4 group alignment warnings present in source report. |
| `model-note` | Report contains synthetic/non-executable model notes. |

### 11.4 Proposal Types

For each detected condition the tool proposes one or more of:

- **Increase tile count** — required minimum tile count estimated as
  `ceil(total_model_bytes / (tile_memory_bytes × 0.8))`.
- **Increase tile memory** — required per-tile memory estimated as
  `ceil(total_model_bytes / tile_count / 0.8)` MiB.
- **Convert dtype** — f32 → q8 (~4× reduction) or q4 (~8× reduction);
  q8 → q4 (~2× reduction); q4 already minimal.
- **Split lm_head** — vocab split to distribute logit projection memory.
- **Split tok_embeddings** — embedding split to reduce tile 0 pressure.
- **Redistribute layers** — rebalance layer ranges when utilization is uneven.
- **Reduce context length** — halving context reduces KV bytes by ~50%.
- **Reduce session count** — each session multiplies KV bytes linearly.
- **Increase fabric bandwidth** — with specific `fabric_gib_sec` estimate.
- **Switch to head-wise placement** — keeps QKV traffic tile-local.

Each proposal is tagged with its trigger and includes a detail string with
current and recommended values where applicable.

### 11.5 Advisory Status Levels

| Status | Meaning |
|---|---|
| `ok` | No capacity/bandwidth FAIL; no major remediation required. |
| `warn` | WARN-level issues only; minor remediation suggested. |
| `fail` | One or more tiles FAIL or a critical issue detected. |

### 11.6 Human-Readable Output

```
advisory: fail
proposal_count: 6
next_action: Add more tiles (current 8, minimum 9) or increase tile_memory_mib to at least 17466 MiB.

Proposals:
  [capacity-fail]  Increase tile count: current 8, recommended minimum 9
  [capacity-fail]  Increase tile memory: current 16384 MiB per tile, recommended minimum 17466 MiB
  [capacity-fail]  Split lm_head across all tiles to distribute logit projection memory.
  [capacity-fail]  Split tok_embeddings across tiles to reduce tile 0 memory pressure.
  [kv-pressure]  Excessive KV cache pressure: 56.2% of tile memory is KV; reduce context length or session count
  [model-note]  Report contains synthetic/non-executable model notes: validate with a real .att1 artifact before production use
```

### 11.7 JSON Output Format

When `--report-json PATH` is used:

```json
{
  "status": "fail",
  "proposal_count": 6,
  "next_action": "Add more tiles ...",
  "analysis": {
    "total_model_bytes": 135291068416,
    "total_kv_bytes": 75161927680,
    "tile_count": 8,
    "tile_memory_mib": 16384,
    "capacity_fail_tiles": 8,
    "capacity_warn_tiles": 0,
    "bandwidth_fail_tiles": 0,
    "bandwidth_warn_tiles": 0,
    "max_utilization_pct": 106.7,
    "min_utilization_pct": 104.5,
    "mean_utilization_pct": 105.3,
    "imbalanced_tiles": false,
    "kv_pressure_max_frac": 0.5618,
    "logits_heavy": false,
    "oversized_tensor_count": 0,
    "has_synthetic_note": true,
    "q4_alignment_issues": false,
    "dtype": "q4"
  },
  "proposals": [
    {
      "trigger": "capacity-fail",
      "action": "Increase tile count",
      "detail": "current 8, recommended minimum 9"
    }
  ]
}
```

### 11.8 Non-Goals for M101

- No execution scheduling change.
- No binary format change.
- No CUDA change.
- No q4/q8/f32 tolerance or behavior change.
- Proposals are advisory only; the tool does not modify the placement report
  or generate a new one.
- No automatic remediation or placement rebalancing.

---

## 12. Placement Scenario Comparison Tool (Milestone 102)

`compiler/propose_tensor_scenarios.py` reads an M100 placement report JSON
and generates candidate placement/capacity scenarios for AIMU tile planning.
It compares tile SKU sizes, tile counts, context lengths, and session counts
without changing runtime execution.

### 12.1 Usage

```
python3 compiler/propose_tensor_scenarios.py --report PATH [OPTIONS]
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--report PATH` | (required) | M98/M100 placement report JSON |
| `--tile-memory-gib LIST` | 16,32,64,128 | Comma-separated tile memory sizes in GiB |
| `--tile-count LIST` | derived | Comma-separated tile counts |
| `--context LIST` | from report | Comma-separated context lengths |
| `--sessions LIST` | from report | Comma-separated session counts |
| `--target-tokens-per-sec N` | — | Target decode rate for fabric estimate |
| `--fabric-gib-sec LIST` | from report | Fabric bandwidth targets in GiB/sec |
| `--report-json PATH` | — | Write machine-readable JSON output |

### 12.2 Exit Codes

| Code | Meaning |
|---|---|
| 0 | At least one PASS capacity scenario found |
| 1 | No PASS capacity scenario found among all candidates |
| 2 | Report could not be parsed (malformed JSON, missing fields, or I/O error) |

### 12.3 Scenario Metrics

Each generated scenario includes:

| Field | Description |
|---|---|
| `tile_count` | Number of AIMU tiles |
| `tile_memory_gib` | Per-tile memory in GiB |
| `context_length` | Context window tokens |
| `sessions` | Concurrent session count |
| `model_bytes_per_tile` | Model weight bytes distributed to this tile |
| `kv_bytes_per_tile` | KV cache bytes at given context/sessions |
| `total_bytes_per_tile` | Sum of model + KV bytes |
| `utilization_percent` | `total_bytes / tile_memory × 100` |
| `capacity_status` | `PASS` (≤ 80 %), `WARN` (80–100 %), `FAIL` (> 100 %) |
| `estimated_fabric_gib_sec` | Estimated fabric bandwidth at target TPS |
| `fabric_status` | `PASS / WARN / FAIL / UNKN` relative to fabric target |
| `recommendation_score` | Higher is better; FAIL capacity = 0 |

### 12.4 Ranking Criteria

Scenarios are sorted by `recommendation_score` (descending):

1. `capacity_status == PASS` first, then `WARN`, then `FAIL`
2. Lower tile count preferred (lower cross-tile fabric traffic)
3. Smaller tile memory SKU preferred (lower cost)
4. Lower KV pressure (lower kv_fraction)

### 12.5 Human-Readable Output

The tool prints:

1. **Original report summary** — dtype, tile count, model/KV bytes, context,
   sessions
2. **Scenario comparison table** — up to 30 rows, sorted by
   `recommendation_score`; the recommended row is marked `<-- recommended`
3. **Recommended scenario** — expanded detail for the top-ranked PASS (or WARN)
   scenario
4. **Notes** — SKU economics and fabric scaling guidance

### 12.6 JSON Output Format (`--report-json`)

```json
{
  "original": {
    "model_name": "",
    "dtype": "q4",
    "tile_count": 8,
    "tile_memory_mib": 16384,
    "context_length": 8192,
    "sessions": 1,
    "total_model_bytes": 135291068416,
    "total_kv_bytes": 75161927680
  },
  "scenarios": [ { ... }, ... ],
  "recommendation": { ... },
  "pass_count": 12,
  "total_count": 32
}
```

Each entry in `"scenarios"` contains all fields from §12.3.

### 12.7 Non-Goals for M102

- No execution scheduling change.
- No binary format change.
- No CUDA change.
- No model weight download or external network access.
- Scenarios are advisory only; the tool does not modify the placement report
  or generate a new one.
- Fabric bandwidth estimates are linear approximations only; they do not
  account for packet batching, latency, or topology.

---

## 13. Placement-Report-to-Command-Plan Mapper (Milestone 109)

The M109 mapper (`compiler/map_placement_to_commands.py`) consumes an M98/M100
placement report JSON and emits a deterministic simulated AIMU command plan
suitable for the M105 command queue / M103 command packet model.  The tool
does **not** execute inference, change runtime behaviour, or access real
PCIe/MMIO registers.

### 13.1 Usage

```
python3 compiler/map_placement_to_commands.py \
    --report <PLACEMENT_JSON> \
    [--model-id ID] \
    [--session-id ID] \
    [--plan-json PATH] \
    [--strict]
```

| Flag | Default | Description |
|---|---|---|
| `--report` | (required) | Path to M98/M100 placement report JSON |
| `--model-id` | `""` (uses `model_name` from header) | Model identifier in every command |
| `--session-id` | `session_0` | Session identifier in every command |
| `--plan-json` | None | Write command plan JSON to PATH |
| `--strict` | off | Reject if any tile has `capacity_status=FAIL` or any tensor has `placement_status != placed` |

### 13.2 Exit Codes

| Code | Meaning |
|---|---|
| 0 | Command plan generated successfully (warnings may be present) |
| 1 | Validation error — placement report problems prevented command generation |
| 2 | Parse error — malformed JSON or missing required top-level fields |

### 13.3 Command Plan Structure

For each tile (sorted by `tile_id`):

1. **`LOAD_TENSOR_TILE`** — one per placed tensor assigned to this tile (sorted
   by `tensor_id`, then `slice_start`).
2. **`VALIDATE_TENSOR`** — one per placed tensor; depends on the corresponding
   `LOAD_TENSOR_TILE` command via `fence_id`.
3. **`TILE_BARRIER`** — one per tile; depends on the last `VALIDATE_TENSOR`
   command on that tile.

After all tiles:

4. **`QUERY_COUNTERS`** — read command-queue counters; depends on the last
   `TILE_BARRIER`.
5. **`TRACE_SNAPSHOT`** — capture unified trace snapshot; depends on the last
   `TILE_BARRIER`.

Command IDs are deterministic sequential integers starting from 1.

### 13.4 Plan Header Fields

| Field | Description |
|---|---|
| `command_plan_version` | Schema version; currently `1` |
| `source_report_path` | Absolute path of the input placement report |
| `model_name` | From `header.model_name` |
| `model_id` | From `--model-id` or `model_name` |
| `session_id` | From `--session-id` |
| `tile_count` | From `header.tile_count` |
| `tensor_count` | Number of placed tensors emitted as `LOAD_TENSOR_TILE` commands |
| `command_count` | Total number of commands in the plan |
| `status` | `ok` |

### 13.5 Command Record Fields

| Field | Notes |
|---|---|
| `command_id` | Monotonically increasing from 1 |
| `command_type` | One of the five types above |
| `tile_id` / `aimu_id` | Owner tile |
| `session_id` / `model_id` | Passed through from CLI |
| `tensor_id` / `tensor_name` | Present for LOAD and VALIDATE; `null` otherwise |
| `dtype` | `f32`, `q8`, or `q4` |
| `quantization_group_size` | Group size for `q4` tensors; `null` for others |
| `packed_bytes` / `scale_bytes` / `total_bytes` | From tensor record |
| `src_descriptor` | `host_buf:tensor_N` for LOAD; tile local addr for VALIDATE |
| `dst_descriptor` | `tileN_local:tensor_N` for LOAD; `null` for VALIDATE |
| `fence_id` | Command ID this command depends on; 0 = no dependency |
| `expected_status` | `ATT1_AIMU_ERR_OK` |
| `checksum` | From tensor record |
| `notes` | Human-readable: category, layer, q4 group_size/packed/scale bytes |

### 13.6 Validation Rules

Exits 1 if: tile_id out of range; tensor_name/tensor_id missing; dtype
unsupported; q4 group_size missing or not in {32, 64}; placement_status !=
placed in --strict mode; capacity_status=FAIL in --strict mode.

Exits 2 if: JSON parse error; report_version missing/unsupported; header or
tensors section missing; tile_count invalid.

### 13.7 Summary Fields

commands_by_type, commands_by_tile, total_tensor_bytes, f32/q8/q4 tensor
counts, capacity_failures_observed, warnings_observed.

### 13.8 Fixtures

| Fixture | Purpose |
|---|---|
| `placement_report_valid.json` | 5-tensor 2-tile f32; 14 commands |
| `placement_report_q4_tiny.json` | 2-tensor 1-tile q4 (group 32 and 64) |
| `placement_report_capacity_fail.json` | tile capacity_status=FAIL |
| `placement_report_missing_tensor_id.json` | tensor missing tensor_id |

### 13.9 Non-Goals for M109

No C ABI, binary format, inference behaviour, CUDA, PCIe/MMIO, or kernel
driver change.  The command plan is advisory and simulated only.

---

## 14. Fabric Route Report Schema (Milestone 115)

This section documents how fabric route reports relate to the placement report
schema defined in §1–§9.  The full route report schema is specified in
`docs/aimu_fabric_routing.md` §11.

### 14.1 From Placement Records to Routes

Each tensor record in the `tensors` array (§3) contributes to zero or more
routes in the fabric route report via the following mapping:

| Placement field | Route field | Notes |
|---|---|---|
| `routing_requirement = "local"` | no route emitted | tensor does not cross tile boundaries |
| `routing_requirement = "broadcast"` | `route_type = ACTIVATION_BROADCAST` | full activation sent to all slice owners |
| `routing_requirement = "reduce"` | `route_type = PARTIAL_REDUCE` | partial results sent to aggregator |
| `routing_requirement = "unicast"` | `route_type = ACTIVATION_SEND` | single-target activation send |
| `routing_requirement = "none"` | no route emitted | replicated tensor; no fabric traffic |
| `reduction_behavior = "sum"` | `reduction_behavior = sum` in route | maps to `FABRIC_REDUCE` `op_param_0=0` |
| `reduction_behavior = "concat"` | `reduction_behavior = concat` in route | maps to `FABRIC_REDUCE` `op_param_0=1` |
| `owner_tile` | `source_tile` | tile that sends the activation or partial result |
| `slice_start`, `slice_end` | `payload_bytes` estimate | `(slice_end - slice_start) × element_size` |

### 14.2 Payload Byte Estimates

The route report's `payload_bytes` for each route is estimated from the
placement report as follows:

| Route type | Payload bytes formula |
|---|---|
| `ACTIVATION_SEND` / `ACTIVATION_BROADCAST` | `d_model × 4` (f32 activation vector) |
| `PARTIAL_REDUCE` (row-split) | `(slice_end - slice_start) × 4` per sending tile |
| `LOGITS_REDUCE` (vocab-split) | `(slice_end - slice_start) × 4` per sending tile |
| `TILE_BARRIER` | `0` (no payload) |
| `CONTROL_ACK` | `8` (8-byte acknowledgment token) |
| `KV_TRANSFER` | `n_kv_heads × head_dim × 2 × 4` per page |

For q8 partial results, `element_size = 4` (activations are kept in f32;
only weights are quantized).  For q4, same as q8 for activations.

### 14.3 Packet Count Estimate

The `packet_count_estimate` in the route report header equals:

```
sum over all routes of route.packet_count_estimate
```

Where per-route `packet_count_estimate` is:

- `ACTIVATION_SEND` → `1`
- `ACTIVATION_BROADCAST` → `len(destination_tiles)`
- `PARTIAL_REDUCE` / `LOGITS_REDUCE` → `1` per route record (one record per
  sending tile)
- `TILE_BARRIER` → `len(destination_tiles)` (one token per participant)
- `CONTROL_ACK` → `1`

For the two-tile layer-wise baseline: `1` (`ACTIVATION_SEND`) + `2`
(`TILE_BARRIER` tokens) = `3` total packets per decode step.

### 14.4 Fabric Bandwidth Estimate

The route report's `payload_bytes_estimate` is used to cross-check against the
placement report's `bandwidth_status` field (§2):

```
estimated_bw_gib_sec = payload_bytes_estimate × target_tokens_per_sec / 2^30
```

If `estimated_bw_gib_sec > fabric_gib_sec` (from the placement report header),
the route report emits a `BANDWIDTH_OVERFLOW` warning (diagnostic code F10).

### 14.5 Fixture

`compiler/fixtures/fabric_route_report_tiny.json` — reference route report for
the `placement_report_valid.json` baseline (2-tile, 2-layer, f32, layer-wise,
d_model=64):

- 3 routes: 2 × `ACTIVATION_SEND` + 1 × `TILE_BARRIER`
- `payload_bytes_estimate = 512` (2 × 256 B; barrier has 0 payload)
- `packet_count_estimate = 3`
- 0 warnings, 0 failures, `status = pass`

### 14.6 Tool: `compiler/map_placement_to_fabric_routes.py`

See `docs/aimu_fabric_routing.md` §11.9 for the full tool specification.

**Summary:**

- Input: M109 command plan JSON (`--plan PATH`)
- Output: fabric route report JSON (`--report-json PATH`) or human-readable text
- Exit 0: pass or warn; Exit 1: fail; Exit 2: parse error
- `--strict`: treats warnings as failures

### 14.7 Non-Goals for M115

No C ABI, binary format, inference behaviour, CUDA, PCIe/MMIO, kernel driver,
fabric execution, or runtime scheduling change.  The route report is advisory
and simulated only.
