# ATT-1 Schema Compatibility Policy

_Milestone 134 — schema and version compatibility regression suite._

This document defines the version compatibility rules for all ATT-1
planning and control schema documents.  The pipeline-input schemas below are
enforced by `compiler/check_schema_compat.py` and tested by
`compiler/test_schema_compat.py`; the frozen diagnostic replay-report schemas
added at M159 are documented here and regression-checked by the C/Python smoke
tests because they are tool outputs rather than compiler-ingested inputs.

---

## 1. Covered Schemas

| Schema | Version Field | Current Version | Source Milestone |
|--------|---------------|-----------------|-----------------|
| Placement report | `report_version` | 1 | M98/M99 |
| Command plan | `command_plan_version` | 1 | M109 |
| Fabric route report | `header.route_report_version` | 1 | M115/M116 |
| Command replay report | `replay_report_version` | 1 | M113/M159 |
| MMIO command replay report | `mmio_replay_report_version` | 1 | M122/M159 |
| Fabric replay report | `fabric_replay_report_version` | 1 | M123/M159 |
| Execution plan | `execution_plan_version` | 1 | M125/M128 |
| Pipeline report | `pipeline_version` | 132 | M132 |

---

## 2. Version Compatibility Rules

### 2.1 Current version — accepted

A document whose version field equals the current supported version is
accepted without error.  Warnings may still be emitted for optional-field
issues (see §4), but the tool exits 0.

### 2.2 Future version — rejected

A document with a version number _greater than_ the highest supported
version is rejected with error code `E_VERSION_UNSUPPORTED`.  The error
message names the unsupported value and lists the supported versions:

```
[ERROR  ] E_VERSION_UNSUPPORTED: 'report_version' value 99 is not
           supported; supported versions: [1]
```

Rationale: accepting unknown future versions silently risks consuming
documents with a different field structure, leading to silent data loss
or incorrect analysis.

### 2.3 Missing version field — rejected

A document that omits the version field entirely is rejected with error
code `E_MISSING_VERSION`.  All schema documents require an explicit
version field.

### 2.4 Wrong type for version field — rejected

The version field must be a JSON integer.  A string such as `"one"` is
rejected with `E_TYPE_MISMATCH`.

---

## 3. Required Field Policy

Each schema specifies a minimum set of required top-level fields.
A document missing any required field is rejected with `E_MISSING_FIELD`.

| Schema | Required fields |
|--------|----------------|
| Placement report | `report_version`, `header`, `tiles` |
| Command plan | `command_plan_version`, `header`, `commands` |
| Fabric route report | `header` (with `route_report_version`), `routes` |
| Execution plan | `execution_plan_version`, `commands` |
| Pipeline report | `pipeline_version`, `final_status` |

List-type fields (`tiles`, `commands`, `routes`) must be JSON arrays.  An
object `{}` in place of an array is rejected with `E_TYPE_MISMATCH`.

---

## 4. Unknown Optional Field Policy

ATT-1 schemas are designed for forward compatibility: producers may add
new optional fields without breaking existing consumers.

| Mode | Behaviour |
|------|-----------|
| Default (`--check`) | Unknown fields produce a `W_UNKNOWN_FIELD` warning; tool exits 0 |
| Strict (`--strict`) | Unknown fields are still warnings (exit 0); they are reported prominently |

Unknown fields are **never** treated as errors.  If a future tool version
adds a new field, older consumers continue to function while noting the
drift.

---

## 5. Type and Value Validation

### 5.1 Integer fields

Fields declared as integers (e.g. `tile_count`, `command_count`,
`route_count`) must be JSON numbers with no fractional part.  A string
value such as `"six"` is rejected with `E_TYPE_MISMATCH`.

### 5.2 Count consistency

Declared count fields must equal the actual length of the corresponding
list:

| Field | List |
|-------|------|
| `header.command_count` | `commands` array in command plan |
| `header.route_count` | `routes` array in fabric route report |
| `command_count` | `commands` array in execution plan |

A mismatch is rejected with `E_COUNT_MISMATCH`.

### 5.3 Negative byte counts

Byte-count fields in tile records (`model_bytes`, `kv_bytes`,
`activation_bytes_per_token`) must not be negative.  A negative value is
rejected with `E_NEGATIVE_BYTES`.

### 5.4 Status enum values

Status fields must contain a value from the declared enum:

| Field context | Valid values |
|---------------|--------------|
| Execution plan `status` | `pass`, `ok`, `warn`, `fail` |
| Pipeline stage statuses | `pass`, `ok`, `warn`, `fail`, `unknown`, `skip` |
| Pipeline `final_status` (required) | same set |
| Tile `capacity_status`, `bandwidth_status` | `PASS`, `WARN`, `FAIL`, `UNKNOWN` |

An out-of-enum value is rejected with `E_INVALID_STATUS`.

---

## 6. Cross-Schema Consistency

When both a placement report and a command plan are available (checked
via `--schema cross`), the checker verifies:

- `placement.header.tile_count` must equal `command_plan.header.tile_count`

A mismatch is rejected with `E_CROSS_TILE_COUNT`.  This guards against
a command plan generated from a different placement than the one currently
in use.

If a fabric route report is also provided (`--route-report`), its
`header.tile_count` must also match the placement.

---

## 7. Deprecated and Legacy Fields

There are currently no deprecated fields in any ATT-1 planning schema.
When a field is deprecated in a future milestone:

1. The field is added to a `_DEPRECATED_FIELDS` set in `check_schema_compat.py`.
2. Its presence is recorded as a `W_DEPRECATED_FIELD` warning.
3. The tool continues to accept the document (exit 0).
4. The deprecation period and expected removal milestone are noted in the
   warning message.

---

## 8. Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All checks passed (warnings may be present) |
| 1 | One or more compatibility failures (errors) |
| 2 | Parse/input error (malformed JSON, missing file, unknown schema type) |

---

## 9. Stable Key Names

The following fields are considered stable: their names and semantics will
not change without a version bump in the corresponding schema version field.

**Placement report (v1)**
`report_version`, `header.tile_count`, `header.model_name`,
`tiles[].tile_id`, `tiles[].model_bytes`, `tiles[].kv_bytes`,
`tiles[].capacity_status`, `tiles[].bandwidth_status`

**Command plan (v1)**
`command_plan_version`, `header.tile_count`, `header.command_count`,
`header.status`, `commands[].command_id`, `commands[].command_type`,
`commands[].tile_id`, `commands[].fence_id`

**Fabric route report (v1)**
`header.route_report_version`, `header.tile_count`, `header.route_count`,
`header.fabric_policy`, `header.status`,
`routes[].route_id`, `routes[].route_type`, `routes[].payload_bytes`

**Command replay report (v1)**
`replay_report_version`, `plan_path`, `model_id`, `session_id`, `tile_count`,
`command_count`, `commands_replayed`, `completions_seen`, `failed_commands`,
`unsupported_commands`, `dma_validations`, `doorbell_count`, `fence_final`,
`trace_event_count`, `commands_by_type`, `commands_by_tile`, `status`, `notes`

**MMIO command replay report (v1)**
`mmio_replay_report_version`, `emulator`, `milestone`, `plan_path`,
`bar0_file`, `model_id`, `session_id`, `tile_count`, `command_count`,
`commands_replayed`, `completions_seen`, `failed_commands`,
`unsupported_commands`, `dma_validations`, `doorbell_count`,
`mmio_doorbell_count`, `fence_final`, `trace_event_count`, `device_id`,
`register_map_version`, `mmio_tile_count`, `commands_by_type`,
`commands_by_tile`, `status`, `notes`

**Fabric replay report (v1)**
`fabric_replay_report_version`, `route_report_path`, `route_count`,
`routes_replayed`, `routes_failed`, `tile_count`, `aggregate_packets_sent`,
`aggregate_packets_received`, `aggregate_payload_bytes_sent`,
`aggregate_payload_bytes_received`, `reductions_started`,
`reductions_completed`, `barriers_started`, `barriers_completed`,
`trace_events`, `required_fabric_gib_sec`, `fabric_status`, `status`, `notes`,
`tiles[].tile_id`, `tiles[].packets_sent`, `tiles[].packets_received`,
`tiles[].payload_bytes_sent`, `tiles[].payload_bytes_received`,
`tiles[].reductions_started`, `tiles[].reductions_completed`,
`tiles[].barriers_started`, `tiles[].barriers_completed`,
`tiles[].trace_events`, `tiles[].route_failures`

**Execution plan (v1)**
`execution_plan_version`, `tile_count`, `command_count`, `status`,
`commands[].plan_command_id`, `commands[].command_type`,
`commands[].tile_id`, `commands[].fence_id`

**Pipeline report (v132)**
`pipeline_version`, `execution_plan_validation_status`,
`command_plan_status`, `mmio_replay_status`, `fabric_route_status`,
`fabric_replay_status`, `fabric_simulation_status`, `final_status`

---

## 10. Adding a New Schema Version

When a schema version must be incremented (e.g. a required field is renamed
or removed):

1. Update the `SUPPORTED_*_VERSIONS` constant in `check_schema_compat.py`.
2. Update the corresponding `COMMAND_PLAN_VERSION` / `SUPPORTED_REPORT_VERSIONS`
   / etc. constant in the production tool.
3. Create a new `*_v<N>_valid.json` fixture in
   `compiler/fixtures/schema_compat/`.
4. Add a `*_vN_future_version.json` fixture for the next unsupported version
   (current + 1).
5. Run `python3 compiler/check_golden_regressions.py --update-golden` if
   tool output changes.
6. Update this document.

---

## 11. Hostile-Input Policy  _(M135)_

The hostile-input regression suite (`compiler/check_hostile_inputs.py` and
`compiler/test_hostile_inputs.py`) enforces the following additional rules
beyond version and type compatibility:

### 11.1 Malformed input must fail clearly

Every hostile input must produce **exit code 1** with a specific, named
error code identifying the failing field or constraint.  Silent acceptance
of invalid input is a regression.

### 11.2 Negative counts and impossible dimensions

The following are always errors (exit 1):

| Field | Rule | Error code |
|-------|------|------------|
| `placement.header.tile_count` | must be ≥ 0 | `E_NEGATIVE_TILE_COUNT` |
| `pipeline.tile_count` | must be ≥ 0 | `E_NEGATIVE_TILE_COUNT` |
| `placement.tiles[i].model_bytes` / `kv_bytes` | must be ≥ 0 | `E_NEGATIVE_BYTES` |
| `command_plan.commands[i].packed_bytes` / `total_bytes` | must be ≥ 0 | `E_NEGATIVE_BYTE_COUNT` |

### 11.3 Duplicate identifiers

Duplicate identifiers within the same document are always errors:

| Field | Error code |
|-------|------------|
| `placement.tiles[i].tile_id` | `E_DUPLICATE_ID` |
| `command_plan.commands[i].command_id` | `E_DUPLICATE_ID` |
| `fabric_route.routes[i].route_id` | `E_DUPLICATE_ID` |
| `execution_plan.commands[i].plan_command_id` | `E_DUPLICATE_ID` |

### 11.4 Invalid enum values

| Field | Allowed values | Error code |
|-------|---------------|------------|
| `placement.tiles[i].capacity_status` | `PASS`, `WARN`, `FAIL`, `UNKNOWN` | `E_INVALID_STATUS` |
| `command_plan.commands[i].command_type` | ATT-1 M109 command type set | `E_UNKNOWN_TYPE` |
| `command_plan.commands[i].dtype` | `f32`, `q8`, `q4`, `bf16`, `i32` | `E_UNSUPPORTED_DTYPE` |
| `fabric_route.routes[i].route_type` | ATT-1 M116 route type set | `E_UNKNOWN_TYPE` |
| `fabric_route.routes[i].ordering_policy` | `ordered`, `unordered`, `barriered` | `E_INVALID_ORDERING` |
| `execution_plan.commands[i].execution_phase` | ATT-1 M128 phase set | `E_UNKNOWN_TYPE` |
| `execution_plan.commands[i].expected_status` | ATT-1 M128 status set | `E_UNKNOWN_TYPE` |

### 11.5 Missing required references

| Condition | Error code |
|-----------|------------|
| `command_plan`: tensor command type with `tensor_name: null` | `E_MISSING_TENSOR_NAME` |
| `placement.tensors[i].owner_tile` not in tile_id set | `E_NONEXISTENT_OWNER` |
| `placement.tensors[i].quantization == "q4"` and `quantization_group_size` is null | `E_Q4_MISSING_GROUP_SIZE` |
| `execution_plan`: `LOAD_TENSOR_TILE` with empty `output_buffers` | `E_MISSING_OUTPUT_BUFFER` |
| `execution_plan`: `EXEC_MATMUL`/`EXEC_RMSNORM` with empty `tensor_dependencies` | `E_MISSING_TENSOR_DEP` |

### 11.6 Fabric route semantic rules

| Condition | Error code |
|-----------|------------|
| Data route (`ACTIVATION_SEND`, etc.) with `payload_bytes == 0` | `E_ZERO_PAYLOAD` |
| Reduction route (`PARTIAL_REDUCE`, `LOGITS_REDUCE`) without explicit `reduction_behavior` | `E_MISSING_REDUCTION` |

### 11.7 Count consistency

| Condition | Error code |
|-----------|------------|
| `placement.header.tile_count` ≠ `len(tiles)` | `E_TILE_COUNT_MISMATCH` |
| `pipeline.commands_replayed` > `pipeline.command_count` | `E_INCONSISTENT_COUNTS` |

### 11.8 No silent fallback

Tools must not fall back silently when a hostile input is detected.  If a
tool succeeds (exit 0) on a fixture that is listed as hostile, the test
suite reports a regression (`FAIL`).

### 11.9 Fixture conventions

- Hostile fixtures live in `compiler/fixtures/hostile/`.
- Each fixture is named after the schema type and the specific violation,
  e.g. `placement_duplicate_tile_id.json`.
- Each fixture is valid JSON but semantically hostile — it triggers exactly
  the error(s) described by its name.
- Valid golden fixtures live in `compiler/fixtures/` and must continue to
  return exit 0 from `check_hostile_inputs.py`.

---

## 12. Compatibility Contract (Milestone 159)

Milestone 159 freezes the M107 in-process DMA descriptor model and the
M113/M122/M123 replay-report schemas at **v1.0**.

### 12.1 What consumers must accept

- Consumers of replay reports must accept version `1` for
  `replay_report_version`, `mmio_replay_report_version`, and
  `fabric_replay_report_version`.
- Consumers must continue to accept unknown optional fields in these reports.
  Additive metadata is forward-compatible and must not cause rejection when the
  version field remains `1`.
- Hardware implementations are **not required** to ingest these diagnostic JSON
  reports directly; however, any bridge, conformance harness, or log-ingestion
  path that does consume them must apply the same compatibility rules.
- DMA-descriptor producers and consumers inside the software simulator stack
  must accept the exact v1.0 `att1_aimu_dma_desc` layout, enum values, flag-bit
  assignments, validation rules, and counter-name set frozen in
  `docs/aimu_register_map.md` §15.7 and `include/att1_aimu_dma.h`.

### 12.2 What consumers may reject

- Replay reports may be rejected if the version field is missing, is not a JSON
  integer, or names a future/unsupported version.
- Replay reports may also be rejected when a required stable key is missing or
  has the wrong JSON type for the declared version.
- DMA descriptors may be rejected when they violate any frozen M107 validation
  rule: invalid direction/dtype, `byte_length` out of range, unsupported flag
  bits, invalid Q4 group size or payload multiple, misalignment, region miss,
  overflow, or D2D overlap.
- Because the DMA simulator model has no embedded per-descriptor version field
  in v1.0, consumers may reject any out-of-band attempt to reinterpret the
  struct with a different size, field order, or enum/flag numbering while still
  claiming Milestone-159 compatibility.

### 12.3 Version negotiation

- The three replay-report schemas negotiate by explicit version field: producer
  writes version `1`; consumer checks that the field exists, is an integer, and
  is within the supported set before trusting any other stable key names.
- Future additive replay-report extensions should keep the version at `1` when
  old consumers can safely ignore the new fields; any rename/removal or type
  change to a frozen stable key requires a version bump.
- The DMA descriptor simulator model negotiates by frozen-interface contract
  rather than an inline version field. In practice, software/hardware stacks use
  the Stage-1 freeze set together: register map v1.0 (M157), command/completion
  schema v1.0 (M158), DMA simulator model v1.0 and replay schemas v1.0 (M159),
  and fabric/barrier semantics v1.0 (M160). A future incompatible DMA-model
  revision therefore requires a new documented versioned contract and matching
  conformance coverage before it can be claimed as supported.
