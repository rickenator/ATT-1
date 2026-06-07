#!/usr/bin/env python3
"""
replay_command_plan_via_mmio.py — M122: command-plan replay via M121 userspace MMIO emulator

Reads a M109 command-plan JSON, validates the plan structure, then drives the
compiled att1-aimu-mmio-replay C binary (which replays the plan through the M121
att1_aimu_userspace MMIO emulator) and forwards the result.

This is a userspace emulator replay, NOT real PCIe/hardware.
The existing M113 in-process replay path (att1-aimu-replay) is unaffected.

Usage:
    python3 compiler/replay_command_plan_via_mmio.py \\
        --plan PATH [--bar0-file PATH] [--tiles N]
        [--tile-memory-mib N] [--kv-memory-mib N]
        [--strict] [--report-json PATH]

Exit codes:
    0  all commands replayed successfully
    1  one or more commands failed or strict-mode violation
    2  JSON parse / missing-field error (detected before invoking C binary)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMMAND_PLAN_VERSION = 1

SUPPORTED_CMD_TYPES: frozenset[str] = frozenset({
    "LOAD_TENSOR_TILE",
    "VALIDATE_TENSOR",
    "TILE_BARRIER",
    "QUERY_COUNTERS",
    "TRACE_SNAPSHOT",
    "RESET_TILE",
    "NOP",
    "EXEC_MATMUL",
    "EXEC_SOFTMAX",
    "EXEC_RMSNORM",
    "EXEC_ROPE",
    "EXEC_ATTENTION",
    "EXEC_FFN",
    "KV_APPEND",
    "KV_READ",
    "FABRIC_SEND",
    "FABRIC_REDUCE",
})

REQUIRED_TOP_KEYS: tuple[str, ...] = (
    "command_plan_version",
    "header",
    "commands",
    "summary",
)

REQUIRED_HEADER_KEYS: tuple[str, ...] = (
    "tile_count",
    "command_count",
)

REQUIRED_CMD_KEYS: tuple[str, ...] = (
    "command_id",
    "command_type",
    "tile_id",
    "fence_id",
    "expected_status",
)

SUPPORTED_DTYPES: frozenset[str] = frozenset({"f32", "q8", "q4"})
VALID_Q4_GROUP_SIZES: frozenset[int] = frozenset({32, 64})

# Default location of the compiled C binary (M122).
_REPO_ROOT = Path(__file__).parent.parent
_DEFAULT_BINARY = _REPO_ROOT / "build" / "att1-aimu-mmio-replay"


# ---------------------------------------------------------------------------
# Validation helpers  (identical policy to replay_aimu_command_plan.py)
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    """Raised for plan structure or field errors (exit code 2)."""


def _validate_plan(plan: dict[str, Any], strict: bool) -> list[str]:
    """
    Validate the parsed plan dict.

    Returns a list of warning strings (non-fatal in non-strict mode).
    Raises ValidationError for fatal structural problems.
    """
    warnings: list[str] = []

    for key in REQUIRED_TOP_KEYS:
        if key not in plan:
            raise ValidationError(f"missing required top-level key: '{key}'")

    version = plan["command_plan_version"]
    if version != COMMAND_PLAN_VERSION:
        raise ValidationError(
            f"unsupported command_plan_version={version} (expected {COMMAND_PLAN_VERSION})"
        )

    header = plan["header"]
    if not isinstance(header, dict):
        raise ValidationError("'header' must be an object")

    for key in REQUIRED_HEADER_KEYS:
        if key not in header:
            raise ValidationError(f"missing required header key: '{key}'")

    tile_count: int = header["tile_count"]
    if not isinstance(tile_count, int) or tile_count <= 0:
        raise ValidationError(
            f"header.tile_count must be a positive integer, got {tile_count!r}"
        )
    if tile_count > 16:
        raise ValidationError(
            f"header.tile_count={tile_count} exceeds M121 emulator limit of 16"
        )

    header_status = header.get("status", "ok")
    if strict and header_status not in ("pass", "ok"):
        raise ValidationError(
            f"strict mode: header status is '{header_status}' (expected pass/ok)"
        )

    commands = plan["commands"]
    if not isinstance(commands, list):
        raise ValidationError("'commands' must be an array")
    if len(commands) == 0:
        raise ValidationError("'commands' array is empty")

    prev_command_id = 0
    for idx, cmd in enumerate(commands):
        if not isinstance(cmd, dict):
            raise ValidationError(f"commands[{idx}] is not an object")

        for key in REQUIRED_CMD_KEYS:
            if key not in cmd:
                raise ValidationError(
                    f"commands[{idx}] missing required key: '{key}'"
                )

        ctype = cmd["command_type"]
        if ctype not in SUPPORTED_CMD_TYPES:
            msg = f"commands[{idx}] unknown command_type: '{ctype}'"
            if strict:
                raise ValidationError(msg)
            warnings.append(f"WARNING: {msg}")

        tile_id = cmd["tile_id"]
        if not isinstance(tile_id, int) or tile_id < 0 or tile_id >= tile_count:
            raise ValidationError(
                f"commands[{idx}] tile_id={tile_id!r} out of range [0, {tile_count})"
            )

        cid = cmd["command_id"]
        if not isinstance(cid, int) or cid <= 0:
            raise ValidationError(
                f"commands[{idx}] command_id must be a positive integer, got {cid!r}"
            )
        if cid <= prev_command_id:
            raise ValidationError(
                f"commands[{idx}] command_id={cid} is not strictly increasing "
                f"(previous={prev_command_id})"
            )
        prev_command_id = cid

        dtype = cmd.get("dtype")
        if dtype is not None and dtype not in SUPPORTED_DTYPES:
            msg = f"commands[{idx}] unsupported dtype='{dtype}'"
            if strict:
                raise ValidationError(msg)
            warnings.append(f"WARNING: {msg}")

        if dtype == "q4":
            qgs = cmd.get("quantization_group_size")
            if qgs is not None and qgs not in VALID_Q4_GROUP_SIZES:
                msg = (
                    f"commands[{idx}] q4 quantization_group_size={qgs} "
                    f"not in {sorted(VALID_Q4_GROUP_SIZES)}"
                )
                if strict:
                    raise ValidationError(msg)
                warnings.append(f"WARNING: {msg}")

        fence_id = cmd["fence_id"]
        if not isinstance(fence_id, int) or fence_id < 0:
            raise ValidationError(
                f"commands[{idx}] fence_id must be a non-negative integer, got {fence_id!r}"
            )

    return warnings


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="M122: AIMU command-plan replay via M121 userspace MMIO emulator",
    )
    parser.add_argument(
        "--plan", required=True, metavar="PATH",
        help="M109 command-plan JSON file",
    )
    parser.add_argument(
        "--bar0-file", default=None, metavar="PATH",
        help="Path for BAR0 register-file backing (mmap'd file; optional)",
    )
    parser.add_argument(
        "--tiles", type=int, default=None, metavar="N",
        help="Override tile count from plan header (1-16)",
    )
    parser.add_argument(
        "--tile-memory-mib", type=int, default=None, metavar="N",
        help="Tile memory capacity in MiB (1-256; default 32)",
    )
    parser.add_argument(
        "--kv-memory-mib", type=int, default=None, metavar="N",
        help="KV cache capacity per tile in MiB (default 8)",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Fail on any validation warning, unsupported op, or bad plan status",
    )
    parser.add_argument(
        "--report-json", default=None, metavar="PATH",
        help="Write replay report as JSON to this path",
    )
    args = parser.parse_args()

    plan_path = args.plan

    # ── Load and parse JSON ──────────────────────────────────────────────
    try:
        with open(plan_path, "r", encoding="utf-8") as fh:
            plan = json.load(fh)
    except FileNotFoundError:
        print(f"mmio-replay: plan file not found: {plan_path}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"mmio-replay: JSON parse error in {plan_path}: {exc}", file=sys.stderr)
        return 2

    # ── Validate plan structure ──────────────────────────────────────────
    try:
        warnings = _validate_plan(plan, args.strict)
    except ValidationError as exc:
        print(f"mmio-replay: validation error: {exc}", file=sys.stderr)
        return 2

    for w in warnings:
        print(w, file=sys.stderr)

    # ── Locate and invoke C binary ───────────────────────────────────────
    binary = _DEFAULT_BINARY
    if not binary.exists():
        alt = Path("build") / "att1-aimu-mmio-replay"
        if alt.exists():
            binary = alt
        else:
            print(
                f"mmio-replay: compiled binary not found at {binary}.\n"
                f"             Run 'make' to build the project first.",
                file=sys.stderr,
            )
            return 2

    cmd: list[str] = [str(binary), "--plan", plan_path]
    if args.bar0_file:
        cmd.extend(["--bar0-file", args.bar0_file])
    if args.tiles is not None:
        cmd.extend(["--tiles", str(args.tiles)])
    if args.tile_memory_mib is not None:
        cmd.extend(["--tile-memory-mib", str(args.tile_memory_mib)])
    if args.kv_memory_mib is not None:
        cmd.extend(["--kv-memory-mib", str(args.kv_memory_mib)])
    if args.strict:
        cmd.append("--strict")
    if args.report_json:
        cmd.extend(["--report-json", args.report_json])

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except OSError as exc:
        print(f"mmio-replay: failed to run {binary}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
