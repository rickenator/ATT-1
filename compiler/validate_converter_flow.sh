#!/usr/bin/env bash
# M44: End-to-end validation of converter-generated .att1 artifacts with shard
# metadata.  Exercises the full pipeline:
#
#   1. Generate converted stub (Python converter)
#   2. Generate shard plan report (text + JSON)
#   3. Inspect artifact (att1-inspect)
#   4. Bench with --shard-plan runtime
#   5. Bench with --shard-plan metadata
#   6. Compare deterministic last_token output
#
# Prerequisites:  run `make` first to build the C tools.
# Not invoked by `make test` — run manually as a developer validation step.
#
# Usage:
#   bash compiler/validate_converter_flow.sh
#   bash compiler/validate_converter_flow.sh --keep   # keep build/m44_* artifacts

set -euo pipefail

KEEP=0
if [[ "${1:-}" == "--keep" ]]; then KEEP=1; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

OUTDIR="build/m44_validation"
mkdir -p "$OUTDIR"

STUB="$OUTDIR/converted_stub_meta.att1"
CONFIG="compiler/fixtures/tiny_llama_config.json"

echo "========================================"
echo " ATT-1 M44 Converter Validation Flow"
echo "========================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Generate converted stub with shard metadata
# ---------------------------------------------------------------------------
echo "--- Step 1: Generate converted stub ---"
python3 compiler/convert_llama_to_att1.py \
    --config "$CONFIG" \
    --tiles 2 --shard-meta \
    --output "$STUB"
echo "  artifact: $STUB ($(wc -c < "$STUB") bytes)"
echo ""

# ---------------------------------------------------------------------------
# Step 2: Generate shard plan report (text + JSON)
# ---------------------------------------------------------------------------
echo "--- Step 2: Shard plan report ---"
python3 compiler/convert_llama_to_att1.py \
    --config "$CONFIG" \
    --tiles 2 --shard-meta \
    --report \
    --report-json "$OUTDIR/report.json"
echo ""
echo "  JSON report: $OUTDIR/report.json"
echo ""

# ---------------------------------------------------------------------------
# Step 3: Inspect artifact
# ---------------------------------------------------------------------------
echo "--- Step 3: Inspect artifact ---"
./build/att1-inspect "$STUB" | tee "$OUTDIR/inspect.txt"
echo ""

# Validate key inspect fields
for field in "tensor_count=21" "n_tiles=2" "shard_meta: 21 records" \
             "shard_meta_tiles=2" "shard_meta_assigned=21" \
             "shard_meta_unassigned=0"; do
    if ! grep -qF "$field" "$OUTDIR/inspect.txt"; then
        echo "FAIL: inspect output missing: $field"
        exit 1
    fi
done
echo "  inspect fields: OK"
echo ""

# ---------------------------------------------------------------------------
# Step 4: Bench with --shard-plan runtime
# ---------------------------------------------------------------------------
echo "--- Step 4: Bench (shard-plan runtime) ---"
./build/att1-bench \
    --model "$STUB" \
    --prompt hello --tokens 8 \
    --mode cluster --tiles 2 \
    --shard-plan runtime --backend cpu-f32 \
    | tee "$OUTDIR/bench_runtime.txt"
echo ""

# ---------------------------------------------------------------------------
# Step 5: Bench with --shard-plan metadata
# ---------------------------------------------------------------------------
echo "--- Step 5: Bench (shard-plan metadata) ---"
./build/att1-bench \
    --model "$STUB" \
    --prompt hello --tokens 8 \
    --mode cluster --tiles 2 \
    --shard-plan metadata --backend cpu-f32 \
    | tee "$OUTDIR/bench_metadata.txt"
echo ""

# ---------------------------------------------------------------------------
# Step 6: Compare last_token (deterministic check)
# ---------------------------------------------------------------------------
echo "--- Step 6: Compare last_token (runtime vs metadata) ---"

LAST_RT=$(grep '^last_token=' "$OUTDIR/bench_runtime.txt"  | head -1)
LAST_MD=$(grep '^last_token=' "$OUTDIR/bench_metadata.txt" | head -1)

if [[ "$LAST_RT" != "$LAST_MD" ]]; then
    echo "FAIL: last_token mismatch"
    echo "  runtime  : $LAST_RT"
    echo "  metadata : $LAST_MD"
    exit 1
fi

LOGITS_RT=$(grep '^logits_bytes_produced=' "$OUTDIR/bench_runtime.txt"  | head -1)
LOGITS_MD=$(grep '^logits_bytes_produced=' "$OUTDIR/bench_metadata.txt" | head -1)

if [[ "$LOGITS_RT" != "$LOGITS_MD" ]]; then
    echo "FAIL: logits_bytes_produced mismatch"
    echo "  runtime  : $LOGITS_RT"
    echo "  metadata : $LOGITS_MD"
    exit 1
fi

PKTS_RT=$(grep '^fabric_packets_sent=' "$OUTDIR/bench_runtime.txt"  | head -1)
PKTS_MD=$(grep '^fabric_packets_sent=' "$OUTDIR/bench_metadata.txt" | head -1)

if [[ "$PKTS_RT" != "$PKTS_MD" ]]; then
    echo "FAIL: fabric_packets_sent mismatch"
    echo "  runtime  : $PKTS_RT"
    echo "  metadata : $PKTS_MD"
    exit 1
fi

echo "  runtime  : $LAST_RT  $LOGITS_RT  $PKTS_RT"
echo "  metadata : $LAST_MD  $LOGITS_MD  $PKTS_MD"
echo "  outputs agree: OK"
echo ""

if [[ "$KEEP" -eq 0 ]]; then
    rm -rf "$OUTDIR"
fi

echo "========================================"
echo " M44 validation passed."
echo "========================================"
