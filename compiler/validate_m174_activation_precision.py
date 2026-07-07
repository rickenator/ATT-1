#!/usr/bin/env python3
"""
ATT-1 M174 activation precision study.

Consumes an M115/M117 fabric route report and compares f32 versus bf16
encoding for inter-tile activation payloads.  Bandwidth savings come from the
route metadata; numerical impact is a deterministic bf16 round-trip study over
representative activation values.

Exit codes:
  0 — bf16 activation packets satisfy savings and error thresholds
  1 — one or more decision thresholds failed
  2 — malformed input or invalid CLI arguments
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

_ACTIVATION_ROUTE_TYPES = {
    "ACTIVATION_SEND",
    "ACTIVATION_BROADCAST",
}
_DEFAULT_SAMPLE_COUNT = 4096


def _die(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(2)


def _load_report(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        _die(f"cannot parse route report: {exc}")

    if not isinstance(data, dict):
        _die("route report root must be an object")
    if not isinstance(data.get("header"), dict):
        _die("route report missing object header")
    if not isinstance(data.get("routes"), list):
        _die("route report missing routes array")
    return data


def _f32_to_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _round_f32_to_bf16(value: float) -> float:
    bits = _f32_to_bits(value)
    # Round-to-nearest-even before dropping the low 16 mantissa bits.
    lsb = (bits >> 16) & 1
    rounded = bits + 0x7FFF + lsb
    return _bits_to_f32(rounded & 0xFFFF0000)


def _activation_samples(count: int) -> list[float]:
    if count <= 0:
        _die("--sample-count must be positive")
    values: list[float] = []
    for i in range(count):
        # Deterministic mix: bounded transformer-like activations plus a few
        # small-magnitude values where relative error is most visible.
        x = i + 1
        value = (
            math.sin(x * 0.017) * 1.75
            + math.cos(x * 0.071) * 0.65
            + ((i % 23) - 11) * 0.013
        )
        if (i % 97) == 0:
            value *= 0.03125
        values.append(value)
    return values


def _numeric_study(sample_count: int) -> dict[str, Any]:
    samples = _activation_samples(sample_count)
    errors = []
    rel_errors = []
    for value in samples:
        rounded = _round_f32_to_bf16(value)
        err = abs(rounded - value)
        errors.append(err)
        denom = max(abs(value), 1.0e-6)
        rel_errors.append(err / denom)

    mse = sum(err * err for err in errors) / len(errors)
    return {
        "sample_count": sample_count,
        "max_abs_error": max(errors),
        "rms_abs_error": math.sqrt(mse),
        "max_relative_error": max(rel_errors),
        "mean_abs_error": sum(errors) / len(errors),
    }


def _route_study(report: dict[str, Any], packet_overhead_bytes: int) -> dict[str, Any]:
    if packet_overhead_bytes < 0:
        _die("--packet-overhead-bytes must be non-negative")

    activation_rows = []
    f32_payload = 0
    bf16_payload = 0
    f32_wire = 0
    bf16_wire = 0

    for route in report["routes"]:
        if not isinstance(route, dict):
            _die("routes entries must be objects")
        route_type = str(route.get("route_type", ""))
        payload_type = str(route.get("payload_type", ""))
        if (
            route_type not in _ACTIVATION_ROUTE_TYPES
            and payload_type != "activation"
        ):
            continue

        payload_bytes = route.get("payload_bytes", 0)
        packet_count = route.get("packet_count_estimate", 1)
        if not isinstance(payload_bytes, int) or payload_bytes < 0:
            _die("activation payload_bytes must be non-negative integers")
        if not isinstance(packet_count, int) or packet_count < 0:
            _die("activation packet_count_estimate must be non-negative integers")

        packets = max(packet_count, 1)
        bf16_bytes = (payload_bytes + 1) // 2
        f32_total = payload_bytes + packets * packet_overhead_bytes
        bf16_total = bf16_bytes + packets * packet_overhead_bytes

        f32_payload += payload_bytes
        bf16_payload += bf16_bytes
        f32_wire += f32_total
        bf16_wire += bf16_total
        activation_rows.append(
            {
                "route_id": route.get("route_id", 0),
                "route_type": route_type,
                "f32_payload_bytes": payload_bytes,
                "bf16_payload_bytes": bf16_bytes,
                "f32_wire_bytes": f32_total,
                "bf16_wire_bytes": bf16_total,
                "wire_savings_percent": (
                    100.0 * (f32_total - bf16_total) / f32_total
                    if f32_total > 0 else 0.0
                ),
            }
        )

    if not activation_rows:
        _die("route report contains no activation routes")

    payload_savings = (
        100.0 * (f32_payload - bf16_payload) / f32_payload
        if f32_payload > 0 else 0.0
    )
    wire_savings = (
        100.0 * (f32_wire - bf16_wire) / f32_wire
        if f32_wire > 0 else 0.0
    )

    return {
        "activation_route_count": len(activation_rows),
        "f32_payload_bytes_per_token": f32_payload,
        "bf16_payload_bytes_per_token": bf16_payload,
        "f32_wire_bytes_per_token": f32_wire,
        "bf16_wire_bytes_per_token": bf16_wire,
        "payload_savings_percent": payload_savings,
        "wire_savings_percent": wire_savings,
        "routes": activation_rows,
    }


def _render_text(report: dict[str, Any], out: dict[str, Any]) -> str:
    header = report["header"]
    traffic = out["traffic"]
    numeric = out["numeric"]
    decision = out["decision"]
    lines = [
        "# ATT-1 M174 activation precision study",
        f"model_name: {header.get('model_name', '')}",
        f"route_count: {header.get('route_count', len(report.get('routes', [])))}",
        f"activation_route_count: {traffic['activation_route_count']}",
        (
            "f32_activation_payload_bytes_per_token: "
            f"{traffic['f32_payload_bytes_per_token']}"
        ),
        (
            "bf16_activation_payload_bytes_per_token: "
            f"{traffic['bf16_payload_bytes_per_token']}"
        ),
        f"payload_savings_percent: {traffic['payload_savings_percent']:.3f}",
        f"wire_savings_percent: {traffic['wire_savings_percent']:.3f}",
        f"max_abs_error: {numeric['max_abs_error']:.8f}",
        f"rms_abs_error: {numeric['rms_abs_error']:.8f}",
        f"max_relative_error: {numeric['max_relative_error']:.8f}",
        f"selected_activation_precision: {decision['selected_activation_precision']}",
        "result: " + out["result"],
        "report: ok",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate M174 f32-vs-bf16 activation packet decision.",
    )
    ap.add_argument("--route-report", required=True, help="Fabric route report JSON.")
    ap.add_argument(
        "--packet-overhead-bytes",
        type=int,
        default=64,
        help="Per-packet wire overhead used for wire-byte savings.",
    )
    ap.add_argument(
        "--sample-count",
        type=int,
        default=_DEFAULT_SAMPLE_COUNT,
        help="Deterministic activation sample count for bf16 error study.",
    )
    ap.add_argument(
        "--min-payload-savings-percent",
        type=float,
        default=45.0,
        help="Required f32-to-bf16 activation payload savings.",
    )
    ap.add_argument(
        "--max-abs-error",
        type=float,
        default=0.02,
        help="Maximum allowed absolute bf16 round-trip error.",
    )
    ap.add_argument(
        "--max-rms-error",
        type=float,
        default=0.004,
        help="Maximum allowed RMS bf16 round-trip error.",
    )
    ap.add_argument(
        "--report-json",
        default=None,
        help="Optional machine-readable report output path.",
    )
    args = ap.parse_args()

    if args.min_payload_savings_percent < 0.0:
        _die("--min-payload-savings-percent must be non-negative")
    if args.max_abs_error <= 0.0:
        _die("--max-abs-error must be positive")
    if args.max_rms_error <= 0.0:
        _die("--max-rms-error must be positive")

    report = _load_report(Path(args.route_report))
    traffic = _route_study(report, args.packet_overhead_bytes)
    numeric = _numeric_study(args.sample_count)

    failures = []
    if traffic["payload_savings_percent"] < args.min_payload_savings_percent:
        failures.append("payload_savings_below_threshold")
    if numeric["max_abs_error"] > args.max_abs_error:
        failures.append("max_abs_error_above_threshold")
    if numeric["rms_abs_error"] > args.max_rms_error:
        failures.append("rms_abs_error_above_threshold")

    result = "pass" if not failures else "fail"
    decision = {
        "selected_activation_precision": "bf16" if result == "pass" else "f32",
        "f32_reference_precision": "f32",
        "rationale": (
            "bf16 halves activation payload bytes while deterministic round-trip "
            "error remains below the configured q8/q4 credibility margins."
        ),
        "thresholds": {
            "min_payload_savings_percent": args.min_payload_savings_percent,
            "max_abs_error": args.max_abs_error,
            "max_rms_error": args.max_rms_error,
        },
    }

    out = {
        "m174_activation_precision_report_version": 1,
        "source_route_report": args.route_report,
        "traffic": traffic,
        "numeric": numeric,
        "decision": decision,
        "failures": failures,
        "result": result,
    }

    print(_render_text(report, out))
    if args.report_json is not None:
        try:
            with Path(args.report_json).open("w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=2)
                fh.write("\n")
        except OSError as exc:
            _die(f"cannot write report JSON: {exc}")

    sys.exit(0 if result == "pass" else 1)


if __name__ == "__main__":
    main()
