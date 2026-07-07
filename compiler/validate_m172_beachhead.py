#!/usr/bin/env python3
"""
ATT-1 M172 beachhead workload and baseline metrics harness.

Runs a deterministic two-tile q8 cluster decode workload multiple times and
summarizes the hard metrics needed before the M175 FPGA gate:

- latency stability across repeated runs
- memory movement per generated token
- KV pressure per generated token
- optional cost-per-token comparison against a CUDA baseline from M155

The script does not download models or require CUDA. Public model artifacts
remain external; checked-in tiny fixtures may exercise the harness only with
--allow-tiny-fixture.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from typing import Any


_MOVEMENT_KEYS = (
    "activation_bytes_sent",
    "logits_bytes_produced",
    "fabric_payload_bytes_sent",
    "fabric_payload_bytes_received",
)

_KV_KEYS = ("kv_appends", "kv_key_reads", "kv_value_reads")


def _is_url_like(path: str | None) -> bool:
    if path is None:
        return False
    lowered = str(path).lower()
    return "://" in lowered or lowered.startswith(("hf:", "hf://"))


def _require_local_file(label: str, path: str) -> None:
    if not path:
        raise ValueError(f"{label} path is required")
    if _is_url_like(path):
        raise ValueError(f"{label} must be a local filesystem path: {path!r}")
    if not os.path.isfile(path):
        raise ValueError(f"{label} file not found: {path!r}")


def _parse_token_file(path: str) -> list[int]:
    tokens: list[int] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh.read().replace(",", "\n").splitlines():
            text = raw.strip()
            if not text:
                continue
            tokens.append(int(text, 10))
    return tokens


def _parse_kv_stdout(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key:
            values[key] = value
    return values


def _int(row: dict[str, Any], key: str) -> int:
    value = row.get(key)
    if value is None:
        raise ValueError(f"missing {key}")
    return int(str(value), 10)


def _run_bench(args: argparse.Namespace, run_index: int) -> dict[str, Any]:
    cmd = [
        args.bench,
        "--model", args.att1_q8,
        "--tokenizer", "external",
        "--tokens-file", args.tokens_file,
        "--tokens", str(args.tokens),
        "--mode", "cluster",
        "--tiles", str(args.tiles),
        "--backend", "cpu-q8",
        "--shard-plan", "runtime",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout_sec)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "run_index": run_index,
            "status": "fail",
            "error": str(exc),
        }

    kv = _parse_kv_stdout(proc.stdout)
    row: dict[str, Any] = {
        "run_index": run_index,
        "status": "pass" if proc.returncode == 0 else "fail",
        "error": None if proc.returncode == 0 else "\n".join(
            part for part in (proc.stdout, proc.stderr) if part
        ).strip(),
    }
    row.update(kv)
    return row


def _validate_row(row: dict[str, Any], tokens: int, tiles: int) -> list[str]:
    errors: list[str] = []
    if row.get("status") != "pass":
        return [f"run {row.get('run_index')} failed"]
    expected = {
        "backend": "cpu-q8",
        "mode": "cluster",
        "shard_plan": "runtime",
        "tokenizer": "external",
        "tiles": str(tiles),
    }
    for key, value in expected.items():
        if row.get(key) != value:
            errors.append(
                f"run {row.get('run_index')} expected {key}={value}, got {row.get(key)!r}"
            )
    try:
        if _int(row, "generated_tokens") != tokens:
            errors.append(f"run {row.get('run_index')} generated token mismatch")
        if _int(row, "fabric_packets_sent") <= 0:
            errors.append(f"run {row.get('run_index')} has no fabric packets")
        if _int(row, "fabric_payload_bytes_sent") <= 0:
            errors.append(f"run {row.get('run_index')} has no fabric payload")
        if _int(row, "kv_appends") <= 0:
            errors.append(f"run {row.get('run_index')} has no KV appends")
        if _int(row, "token_time_us_total") <= 0:
            errors.append(f"run {row.get('run_index')} has zero token time")
    except (TypeError, ValueError) as exc:
        errors.append(f"run {row.get('run_index')} parse failure: {exc}")
    return errors


def _summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    per_token_us = [
        _int(row, "token_time_us_total") / max(1, _int(row, "generated_tokens"))
        for row in rows
    ]
    movement_per_token = [
        sum(_int(row, key) for key in _MOVEMENT_KEYS) /
        max(1, _int(row, "generated_tokens"))
        for row in rows
    ]
    kv_ops_per_token = [
        sum(_int(row, key) for key in _KV_KEYS) /
        max(1, _int(row, "generated_tokens"))
        for row in rows
    ]

    avg_us = statistics.fmean(per_token_us)
    min_us = min(per_token_us)
    max_us = max(per_token_us)
    jitter_pct = 0.0 if avg_us == 0.0 else ((max_us - min_us) / avg_us) * 100.0
    cost_per_million = (
        None if args.dollars_per_hour is None
        else (args.dollars_per_hour * avg_us / 3600.0)
    )

    summary: dict[str, Any] = {
        "runs": len(rows),
        "tokens_per_run": args.tokens,
        "prompt_tokens": len(_parse_token_file(args.tokens_file)),
        "latency_us_per_token_avg": avg_us,
        "latency_us_per_token_min": min_us,
        "latency_us_per_token_max": max_us,
        "latency_jitter_pct": jitter_pct,
        "memory_movement_bytes_per_token_avg": statistics.fmean(movement_per_token),
        "kv_ops_per_token_avg": statistics.fmean(kv_ops_per_token),
        "fabric_packets_per_token_avg": statistics.fmean(
            _int(row, "fabric_packets_sent") / max(1, _int(row, "generated_tokens"))
            for row in rows
        ),
        "cost_per_million_tokens_usd": cost_per_million,
    }

    if args.cuda_baseline_us_per_token is not None:
        summary["cuda_baseline_us_per_token"] = args.cuda_baseline_us_per_token
        summary["latency_vs_cuda_baseline_ratio"] = (
            avg_us / args.cuda_baseline_us_per_token
        )
        if args.cuda_dollars_per_hour is not None:
            cuda_cost = args.cuda_dollars_per_hour * args.cuda_baseline_us_per_token / 3600.0
            summary["cuda_cost_per_million_tokens_usd"] = cuda_cost
            summary["cost_vs_cuda_baseline_ratio"] = (
                None if cost_per_million is None or cuda_cost == 0.0
                else cost_per_million / cuda_cost
            )

    return summary


def run_validation(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    try:
        _require_local_file("q8 ATT-1 artifact", args.att1_q8)
        _require_local_file("token IDs", args.tokens_file)
        _require_local_file("att1-bench", args.bench)
        if args.tiles != 2:
            raise ValueError("M172 beachhead requires --tiles 2")
        if args.runs < 2:
            raise ValueError("--runs must be >= 2 for stability measurement")
        token_ids = _parse_token_file(args.tokens_file)
        if len(token_ids) < args.min_prompt_tokens:
            raise ValueError(
                f"beachhead prompt has {len(token_ids)} tokens, "
                f"below --min-prompt-tokens {args.min_prompt_tokens}"
            )
        if not args.allow_tiny_fixture and args.min_prompt_tokens < 128:
            raise ValueError("real M172 runs must use --min-prompt-tokens >= 128")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return {}, 1

    rows = [_run_bench(args, i + 1) for i in range(args.runs)]
    failures: list[str] = []
    for row in rows:
        failures.extend(_validate_row(row, args.tokens, args.tiles))

    summary: dict[str, Any] = {}
    if not failures:
        summary = _summarize(rows, args)

    if summary and summary["latency_jitter_pct"] > args.max_jitter_pct:
        failures.append(
            "latency jitter %.2f%% exceeds %.2f%%" %
            (summary["latency_jitter_pct"], args.max_jitter_pct)
        )

    report = {
        "m172_beachhead_report_version": 1,
        "workload": {
            "name": args.workload_name,
            "artifact_path": args.att1_q8,
            "tokens_file": args.tokens_file,
            "prompt_tokens": len(token_ids),
            "generated_tokens_per_run": args.tokens,
            "tiles": args.tiles,
            "backend": "cpu-q8",
            "shard_plan": "runtime",
            "allow_tiny_fixture": bool(args.allow_tiny_fixture),
        },
        "metric_definitions": {
            "latency_us_per_token": "token_time_us_total / generated_tokens",
            "latency_jitter_pct": "(max per-token latency - min per-token latency) / avg * 100",
            "memory_movement_bytes_per_token": "activation_bytes_sent + logits_bytes_produced + fabric_payload_bytes_sent + fabric_payload_bytes_received, divided by generated tokens",
            "kv_ops_per_token": "kv_appends + kv_key_reads + kv_value_reads, divided by generated tokens",
            "cost_per_million_tokens_usd": "dollars_per_hour * avg_us_per_token / 3600",
        },
        "runs": rows,
        "summary": summary,
        "failures": failures,
        "result": "pass" if not failures else "fail",
    }

    print("# ATT-1 M172 beachhead baseline")
    print(f"workload: {args.workload_name}")
    print(f"artifact_path: {args.att1_q8}")
    print(f"tokens_file: {args.tokens_file}")
    print(f"prompt_tokens: {len(token_ids)}")
    print(f"generated_tokens_per_run: {args.tokens}")
    print(f"runs: {args.runs}")
    for row in rows:
        print(_format_run(row))
    if summary:
        print("latency_us_per_token_avg: %.3f" % summary["latency_us_per_token_avg"])
        print("latency_jitter_pct: %.3f" % summary["latency_jitter_pct"])
        print(
            "memory_movement_bytes_per_token_avg: %.3f" %
            summary["memory_movement_bytes_per_token_avg"]
        )
        print("kv_ops_per_token_avg: %.3f" % summary["kv_ops_per_token_avg"])
        print("fabric_packets_per_token_avg: %.3f" % summary["fabric_packets_per_token_avg"])
        if summary.get("cost_per_million_tokens_usd") is not None:
            print(
                "cost_per_million_tokens_usd: %.8f" %
                summary["cost_per_million_tokens_usd"]
            )
        if summary.get("latency_vs_cuda_baseline_ratio") is not None:
            print(
                "latency_vs_cuda_baseline_ratio: %.6f" %
                summary["latency_vs_cuda_baseline_ratio"]
            )
    for failure in failures:
        print(f"failure: {failure}")
    print(f"result: {report['result']}")
    print("report: ok")

    if args.report_json:
        try:
            out_dir = os.path.dirname(args.report_json)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.report_json, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2)
                fh.write("\n")
        except OSError as exc:
            print(
                f"error: cannot write report JSON {args.report_json!r}: {exc}",
                file=sys.stderr,
            )
            return report, 1

    return report, 0 if not failures else 2


def _format_run(row: dict[str, Any]) -> str:
    if row.get("status") != "pass":
        return "run: index=%s status=fail error=%s" % (
            row.get("run_index"), " ".join(str(row.get("error")).split())
        )
    fields = [
        "run:",
        f"index={row.get('run_index')}",
        f"backend={row.get('backend')}",
        f"mode={row.get('mode')}",
        f"generated_tokens={row.get('generated_tokens')}",
        f"token_time_us_total={row.get('token_time_us_total')}",
        f"fabric_packets_sent={row.get('fabric_packets_sent')}",
        f"fabric_payload_bytes_sent={row.get('fabric_payload_bytes_sent')}",
        f"kv_appends={row.get('kv_appends')}",
        f"status={row.get('status')}",
    ]
    return " ".join(fields)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the M172 deterministic beachhead workload baseline."
    )
    parser.add_argument("--att1-q8", required=True, metavar="PATH")
    parser.add_argument("--tokens-file", required=True, metavar="PATH")
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--tiles", type=int, default=2)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--min-prompt-tokens", type=int, default=128)
    parser.add_argument("--max-jitter-pct", type=float, default=10000.0)
    parser.add_argument("--workload-name", default="m172_beachhead_decode")
    parser.add_argument(
        "--bench", default=os.path.join("build", "att1-bench"), metavar="PATH"
    )
    parser.add_argument("--report-json", default=None, metavar="PATH")
    parser.add_argument("--dollars-per-hour", type=float, default=None)
    parser.add_argument("--cuda-baseline-us-per-token", type=float, default=None)
    parser.add_argument("--cuda-dollars-per-hour", type=float, default=None)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument(
        "--allow-tiny-fixture",
        action="store_true",
        help="Permit shorter checked-in fixture prompts for CI smoke coverage.",
    )
    args = parser.parse_args()

    if args.tokens < 1:
        print("error: --tokens must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.min_prompt_tokens < 1:
        print("error: --min-prompt-tokens must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.dollars_per_hour is not None and args.dollars_per_hour < 0.0:
        print("error: --dollars-per-hour must be non-negative", file=sys.stderr)
        sys.exit(1)
    if (
        args.cuda_baseline_us_per_token is not None and
        args.cuda_baseline_us_per_token <= 0.0
    ):
        print("error: --cuda-baseline-us-per-token must be > 0", file=sys.stderr)
        sys.exit(1)

    _, exit_code = run_validation(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
