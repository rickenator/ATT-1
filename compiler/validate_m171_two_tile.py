#!/usr/bin/env python3
"""
ATT-1 M171 two-tile real-model validation harness.

Runs a local converted f32 reference artifact and q8 primary artifact through
the two-tile cluster path using pre-tokenized input.  Public model weights and
generated public .att1 files must stay outside Git; this script only consumes
local filesystem paths and writes an optional local JSON report.

For CI/smoke coverage, --allow-tiny-fixture permits the checked-in tiny
fixtures to exercise the same control flow without pretending they satisfy
the SmolLM2-135M-class size requirement.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any


_MIN_REAL_PARAMS = 100_000_000
_MAX_REAL_PARAMS = 200_000_000


def _is_url_like(path: str | None) -> bool:
    if path is None:
        return False
    lowered = str(path).lower()
    return "://" in lowered or lowered.startswith(("hf:", "hf://"))


def _require_local_path(label: str, path: str, *, directory: bool = False) -> None:
    if not path:
        raise ValueError(f"{label} path is required")
    if _is_url_like(path):
        raise ValueError(f"{label} must be a local filesystem path: {path!r}")
    exists = os.path.isdir(path) if directory else os.path.isfile(path)
    if not exists:
        kind = "directory" if directory else "file"
        raise ValueError(f"{label} {kind} not found: {path!r}")


def _resolve(cfg: dict[str, Any], aliases: tuple[str, ...]) -> int:
    for key in aliases:
        value = cfg.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    raise ValueError(f"config missing required field: {aliases[0]}")


def _load_config(model_dir: str) -> tuple[dict[str, Any], dict[str, int], int]:
    config_path = os.path.join(model_dir, "config.json")
    _require_local_path("model config", config_path)
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    resolved = {
        "vocab_size": _resolve(cfg, ("vocab_size",)),
        "n_layers": _resolve(cfg, ("n_layers", "num_hidden_layers")),
        "n_heads": _resolve(cfg, ("n_heads", "num_attention_heads")),
        "d_model": _resolve(cfg, ("d_model", "hidden_size")),
        "d_ff": _resolve(cfg, ("d_ff", "intermediate_size")),
        "max_seq_len": _resolve(cfg, ("max_seq_len", "max_position_embeddings")),
    }
    params = _estimate_llama_params(resolved)
    return cfg, resolved, params


def _estimate_llama_params(c: dict[str, int]) -> int:
    d_model = c["d_model"]
    d_ff = c["d_ff"]
    n_layers = c["n_layers"]
    vocab = c["vocab_size"]

    embedding = vocab * d_model
    per_layer = (
        (4 * d_model * d_model) +
        (3 * d_model * d_ff) +
        (2 * d_model)
    )
    output_norm = d_model
    output = vocab * d_model
    return int(embedding + (n_layers * per_layer) + output_norm + output)


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


def _as_int(row: dict[str, Any], key: str) -> int:
    value = row.get(key)
    if value is None:
        raise ValueError(f"missing {key}")
    return int(str(value), 10)


def _run_cluster(
    bench: str,
    artifact: str,
    backend: str,
    tokens_file: str,
    tokens: int,
    tiles: int,
) -> dict[str, Any]:
    cmd = [
        bench,
        "--model", artifact,
        "--tokenizer", "external",
        "--tokens-file", tokens_file,
        "--tokens", str(tokens),
        "--mode", "cluster",
        "--tiles", str(tiles),
        "--backend", backend,
        "--shard-plan", "runtime",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "artifact_path": artifact,
            "backend": backend,
            "mode": "cluster",
            "shard_plan": "runtime",
            "status": "fail",
            "error": str(exc),
        }

    kv = _parse_kv_stdout(proc.stdout)
    row: dict[str, Any] = {
        "artifact_path": artifact,
        "requested_backend": backend,
        "backend": kv.get("backend", backend),
        "mode": kv.get("mode", "cluster"),
        "shard_plan": kv.get("shard_plan", "runtime"),
        "generated_tokens": kv.get("generated_tokens"),
        "last_token": kv.get("last_token"),
        "token_time_us_total": kv.get("token_time_us_total"),
        "fabric_packets_sent": kv.get("fabric_packets_sent"),
        "fabric_payload_bytes_sent": kv.get("fabric_payload_bytes_sent"),
        "kv_appends": kv.get("kv_appends"),
        "status": "pass" if proc.returncode == 0 else "fail",
        "error": None if proc.returncode == 0 else "\n".join(
            part for part in (proc.stdout, proc.stderr) if part
        ).strip(),
    }
    return row


def _validate_row(row: dict[str, Any], backend: str, tokens: int) -> list[str]:
    errors: list[str] = []
    if row.get("status") != "pass":
        errors.append(f"{backend} cluster run failed")
        return errors
    if row.get("backend") != backend:
        errors.append(f"{backend} run reported backend={row.get('backend')!r}")
    if row.get("mode") != "cluster":
        errors.append(f"{backend} run did not report mode=cluster")
    if row.get("shard_plan") != "runtime":
        errors.append(f"{backend} run did not report shard_plan=runtime")
    try:
        if _as_int(row, "generated_tokens") != tokens:
            errors.append(f"{backend} generated_tokens mismatch")
        if _as_int(row, "fabric_packets_sent") <= 0:
            errors.append(f"{backend} fabric_packets_sent is zero")
        if _as_int(row, "fabric_payload_bytes_sent") <= 0:
            errors.append(f"{backend} fabric_payload_bytes_sent is zero")
        if _as_int(row, "kv_appends") <= 0:
            errors.append(f"{backend} kv_appends is zero")
        _as_int(row, "last_token")
        _as_int(row, "token_time_us_total")
    except (TypeError, ValueError) as exc:
        errors.append(f"{backend} report parse failure: {exc}")
    return errors


def run_validation(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    try:
        _require_local_path("source model", args.model_dir, directory=True)
        _require_local_path("f32 ATT-1 artifact", args.att1_f32)
        _require_local_path("q8 ATT-1 artifact", args.att1_q8)
        _require_local_path("token IDs", args.tokens_file)
        _require_local_path("att1-bench", args.bench)
        if args.tiles != 2:
            raise ValueError("M171 requires --tiles 2")
        _, resolved, estimated_params = _load_config(args.model_dir)
        if not args.allow_tiny_fixture:
            if not (_MIN_REAL_PARAMS <= estimated_params <= _MAX_REAL_PARAMS):
                raise ValueError(
                    "source model is not in the M171 SmolLM2-135M class "
                    f"({estimated_params} estimated params)"
                )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return {}, 1

    q8_row = _run_cluster(
        args.bench, args.att1_q8, "cpu-q8", args.tokens_file, args.tokens, args.tiles
    )
    f32_row = _run_cluster(
        args.bench, args.att1_f32, "cpu-f32", args.tokens_file, args.tokens, args.tiles
    )

    failures = []
    failures.extend(_validate_row(q8_row, "cpu-q8", args.tokens))
    failures.extend(_validate_row(f32_row, "cpu-f32", args.tokens))

    q8_last = q8_row.get("last_token")
    f32_last = f32_row.get("last_token")
    report = {
        "m171_two_tile_report_version": 1,
        "source_model_path": args.model_dir,
        "model_config": resolved,
        "estimated_parameter_count": estimated_params,
        "allow_tiny_fixture": bool(args.allow_tiny_fixture),
        "tokens_file": args.tokens_file,
        "tokens": args.tokens,
        "tiles": args.tiles,
        "primary": q8_row,
        "reference": f32_row,
        "last_token_match": q8_last == f32_last,
        "result": "pass" if not failures else "fail",
        "failures": failures,
    }

    print("# ATT-1 M171 two-tile validation")
    print(f"source_model_path: {args.model_dir}")
    print(f"estimated_parameter_count: {estimated_params}")
    print(f"tokens_file: {args.tokens_file}")
    print(f"tokens: {args.tokens}")
    print(f"tiles: {args.tiles}")
    print(_format_row("primary", q8_row))
    print(_format_row("reference", f32_row))
    print(f"last_token_match: {'yes' if report['last_token_match'] else 'no'}")
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


def _format_row(label: str, row: dict[str, Any]) -> str:
    fields = [
        f"{label}:",
        f"backend={row.get('backend')}",
        f"mode={row.get('mode')}",
        f"shard_plan={row.get('shard_plan')}",
        f"generated_tokens={row.get('generated_tokens')}",
        f"last_token={row.get('last_token')}",
        f"fabric_packets_sent={row.get('fabric_packets_sent')}",
        f"fabric_payload_bytes_sent={row.get('fabric_payload_bytes_sent')}",
        f"kv_appends={row.get('kv_appends')}",
        f"token_time_us_total={row.get('token_time_us_total')}",
        f"status={row.get('status')}",
    ]
    if row.get("error"):
        fields.append("error=" + " ".join(str(row["error"]).split()))
    return " ".join(fields)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate M171 q8/f32 local artifacts through the two-tile cluster path."
    )
    parser.add_argument("--model-dir", required=True, metavar="DIR")
    parser.add_argument("--att1-f32", required=True, metavar="PATH")
    parser.add_argument("--att1-q8", required=True, metavar="PATH")
    parser.add_argument("--tokens-file", required=True, metavar="PATH")
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--tiles", type=int, default=2)
    parser.add_argument(
        "--bench", default=os.path.join("build", "att1-bench"), metavar="PATH"
    )
    parser.add_argument("--report-json", default=None, metavar="PATH")
    parser.add_argument(
        "--allow-tiny-fixture",
        action="store_true",
        help="Permit checked-in tiny fixtures to exercise the M171 harness.",
    )
    args = parser.parse_args()

    if args.tokens < 1:
        print("error: --tokens must be >= 1", file=sys.stderr)
        sys.exit(1)

    _, exit_code = run_validation(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
