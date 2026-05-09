#!/usr/bin/env bash
# ATT-1 M144: public demo script for tiny fixtures.
#
# Exercises the ATT-1 binary loader, inference bench, and planning/control-plane
# pipeline end-to-end using only checked-in tiny/dummy fixtures.
#
# Requirements:
#   - CPU-only (no CUDA, no GPU)
#   - No network access
#   - No public model downloads
#   - No external model weights
#   - Temporary files go to /tmp/att1-demo-XXXXXX (cleaned up by default)
#
# Usage:
#   ./tools/demo_tiny_att1.sh [--skip-build] [--keep-temp] [--verbose]
#
# Exit codes:
#   0  all demo steps passed
#   1  one or more demo steps failed
#
# This demo uses the tiny 4-d dummy model (models/dummy/model.att1).
# It is NOT representative of real inference performance.

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SKIP_BUILD=0
KEEP_TEMP=0
VERBOSE=0

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        --keep-temp)  KEEP_TEMP=1  ;;
        --verbose)    VERBOSE=1    ;;
        --help|-h)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            echo "Usage: $0 [--skip-build] [--keep-temp] [--verbose]" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Locate repo root (script lives in tools/)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Temp directory
# ---------------------------------------------------------------------------
TMPDIR_DEMO="$(mktemp -d /tmp/att1-demo-XXXXXX)"
echo "Temp directory: $TMPDIR_DEMO"

cleanup() {
    if [[ "$KEEP_TEMP" -eq 0 ]]; then
        rm -rf "$TMPDIR_DEMO"
    else
        echo "Temp files kept in: $TMPDIR_DEMO"
    fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS_COUNT=0
FAIL_COUNT=0

section() {
    echo ""
    echo "=== $* ==="
}

run_step() {
    local label="$1"; shift
    if [[ "$VERBOSE" -eq 1 ]]; then
        echo "  $ $*"
        if "$@"; then
            echo "  → OK"
            PASS_COUNT=$(( PASS_COUNT + 1 ))
        else
            echo "  → FAIL: $label" >&2
            FAIL_COUNT=$(( FAIL_COUNT + 1 ))
        fi
    else
        if "$@" >/dev/null 2>&1; then
            echo "  OK  $label"
            PASS_COUNT=$(( PASS_COUNT + 1 ))
        else
            echo "  FAIL $label" >&2
            FAIL_COUNT=$(( FAIL_COUNT + 1 ))
        fi
    fi
}

run_step_stdout() {
    # Like run_step but shows stdout even in non-verbose mode (for informational steps)
    local label="$1"; shift
    if [[ "$VERBOSE" -eq 1 ]]; then
        echo "  $ $*"
    fi
    if "$@" 2>/dev/null; then
        echo "  OK  $label"
        PASS_COUNT=$(( PASS_COUNT + 1 ))
    else
        echo "  FAIL $label" >&2
        FAIL_COUNT=$(( FAIL_COUNT + 1 ))
    fi
}

# ---------------------------------------------------------------------------
# Key paths
# ---------------------------------------------------------------------------
MODEL="models/dummy/model.att1"
BUILD="build"
INSPECT="$BUILD/att1-inspect"
BENCH="$BUILD/att1-bench"
SIZE="$BUILD/att1-size"

PLACEMENT_JSON="$TMPDIR_DEMO/placement.json"
CMD_PLAN_JSON="$TMPDIR_DEMO/cmd_plan.json"
ROUTE_REPORT_JSON="$TMPDIR_DEMO/route_report.json"

# ---------------------------------------------------------------------------
# Step 0: Build
# ---------------------------------------------------------------------------
if [[ "$SKIP_BUILD" -eq 0 ]]; then
    section "Step 0: Build (CUDA=0)"
    echo "  \$ make clean && make CUDA=0"
    if [[ "$VERBOSE" -eq 1 ]]; then
        make clean CUDA=0 && make CUDA=0
    else
        make clean CUDA=0 >/dev/null 2>&1 && make CUDA=0 >/dev/null 2>&1
    fi
    echo "  OK  build"
    PASS_COUNT=$(( PASS_COUNT + 1 ))
else
    section "Step 0: Build skipped (--skip-build)"
fi

# Verify required binaries exist
for bin in "$INSPECT" "$BENCH" "$SIZE"; do
    if [[ ! -x "$bin" ]]; then
        echo "FAIL: required binary not found or not executable: $bin" >&2
        echo "      Run without --skip-build or run 'make CUDA=0' first." >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Step 1: att1-inspect on dummy model
# ---------------------------------------------------------------------------
section "Step 1: att1-inspect (dummy model)"
run_step_stdout "att1-inspect dummy model" \
    "$INSPECT" "$MODEL"

# ---------------------------------------------------------------------------
# Step 2: att1-size capacity planning
# ---------------------------------------------------------------------------
section "Step 2: att1-size capacity planning (tiny-dummy)"
run_step_stdout "att1-size --preset tiny-dummy" \
    "$SIZE" --preset tiny-dummy

# ---------------------------------------------------------------------------
# Step 3: att1-bench cpu-f32 single
# ---------------------------------------------------------------------------
section "Step 3: att1-bench cpu-f32 single"
run_step "bench cpu-f32 single (4 tokens)" \
    "$BENCH" --model "$MODEL" --tokens 4 --mode single \
             --backend cpu-f32 --prompt "hi"

# ---------------------------------------------------------------------------
# Step 4: att1-bench cpu-f32 cluster (2 tiles)
# ---------------------------------------------------------------------------
section "Step 4: att1-bench cpu-f32 cluster (2 tiles)"
run_step "bench cpu-f32 cluster 2-tile (4 tokens)" \
    "$BENCH" --model "$MODEL" --tokens 4 --mode cluster \
             --backend cpu-f32 --tiles 2 --prompt "hi"

# ---------------------------------------------------------------------------
# Step 5: att1-bench cpu-q8 single
# ---------------------------------------------------------------------------
section "Step 5: att1-bench cpu-q8 single"
run_step "bench cpu-q8 single (4 tokens)" \
    "$BENCH" --model "$MODEL" --tokens 4 --mode single \
             --backend cpu-q8 --prompt "hi"

# ---------------------------------------------------------------------------
# Step 6: Placement report generation
# ---------------------------------------------------------------------------
section "Step 6: Placement report generation (tiny-dummy)"
run_step "att1-size generate placement report" \
    "$SIZE" --preset tiny-dummy \
            --placement-report-json "$PLACEMENT_JSON"

# ---------------------------------------------------------------------------
# Step 7: Placement schema validation
# ---------------------------------------------------------------------------
section "Step 7: Placement schema validation"
run_step "check_schema_compat placement" \
    python3 compiler/check_schema_compat.py \
        --schema placement \
        --input "$PLACEMENT_JSON"

# ---------------------------------------------------------------------------
# Step 8: Placement advisory
# ---------------------------------------------------------------------------
section "Step 8: Placement advisory"
run_step "propose_tensor_placement advisory" \
    python3 compiler/propose_tensor_placement.py \
        --report "$PLACEMENT_JSON"

# ---------------------------------------------------------------------------
# Step 9: Command-plan mapping
# ---------------------------------------------------------------------------
section "Step 9: Command-plan mapping (placement → AIMU commands)"
run_step "map_placement_to_commands" \
    python3 compiler/map_placement_to_commands.py \
        --report "$PLACEMENT_JSON" \
        --plan-json "$CMD_PLAN_JSON"

# ---------------------------------------------------------------------------
# Step 10: Fabric-route mapping
# ---------------------------------------------------------------------------
section "Step 10: Fabric-route mapping (commands → routes)"
run_step "map_commands_to_fabric_routes" \
    python3 compiler/map_commands_to_fabric_routes.py \
        --plan "$CMD_PLAN_JSON" \
        --route-report-json "$ROUTE_REPORT_JSON"

# ---------------------------------------------------------------------------
# Step 11: Fabric-route validation
# ---------------------------------------------------------------------------
section "Step 11: Fabric-route validation"
run_step "validate_fabric_routes" \
    python3 compiler/validate_fabric_routes.py \
        --report "$ROUTE_REPORT_JSON"

# ---------------------------------------------------------------------------
# Step 12: Fabric replay / bandwidth simulation
# ---------------------------------------------------------------------------
section "Step 12: Fabric replay / bandwidth simulation"
run_step "replay_fabric_routes" \
    python3 compiler/replay_fabric_routes.py \
        --route-report "$ROUTE_REPORT_JSON"

# ---------------------------------------------------------------------------
# Step 13: Integrated execution/replay pipeline (M132)
# ---------------------------------------------------------------------------
section "Step 13: Integrated execution/replay pipeline (exec_plan_valid_tiny)"
EXEC_PLAN="compiler/fixtures/exec_plan_valid_tiny.json"
PIPELINE_WORKDIR="$TMPDIR_DEMO/pipeline"
mkdir -p "$PIPELINE_WORKDIR"
run_step "run_execution_replay_pipeline (6-stage)" \
    python3 compiler/run_execution_replay_pipeline.py \
        --execution-plan "$EXEC_PLAN" \
        --workdir "$PIPELINE_WORKDIR"

# ---------------------------------------------------------------------------
# Step 14: Hostile-input check (binary loader fuzz smoke)
# ---------------------------------------------------------------------------
section "Step 14: Binary loader fuzz smoke"
if [[ -x "$BUILD/fuzz_model_loader" ]]; then
    run_step "fuzz_model_loader (17 cases)" \
        "$BUILD/fuzz_model_loader"
else
    echo "  SKIP fuzz_model_loader not built; run 'make fuzz-loader' first"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  Demo steps attempted : $(( PASS_COUNT + FAIL_COUNT ))"
echo "  PASS                 : $PASS_COUNT"
echo "  FAIL                 : $FAIL_COUNT"
echo "========================================"
echo ""
echo "NOTE: This demo uses the tiny 4-d dummy fixture only."
echo "      Output does not represent real model inference performance."
echo ""
if [[ "$FAIL_COUNT" -gt 0 ]]; then
    echo "ATT-1 tiny demo: FAIL"
    exit 1
fi
echo "ATT-1 tiny demo: PASS"
exit 0
