# ATT-1 Schema Compatibility Policy

_Milestone 134 — schema and version compatibility regression suite._

This document defines the version compatibility rules for all ATT-1
planning and control schema documents.  These rules are enforced by
`compiler/check_schema_compat.py` and tested by
`compiler/test_schema_compat.py`.

---

## 1. Covered Schemas

| Schema | Version Field | Current Version | Source Milestone |
|--------|---------------|-----------------|-----------------|
| Placement report | `report_version` | 1 | M98/M99 |
| Command plan | `command_plan_version` | 1 | M109 |
| Fabric route report | `header.route_report_version` | 1 | M115/M116 |
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
