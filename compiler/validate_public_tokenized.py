#!/usr/bin/env python3
"""
ATT-1 public-model tokenized end-to-end validator (Milestone 71).

Uses the local Hugging Face tokenizer helper from M59 to create a token IDs
file, then runs local converted f32/q8 ATT-1 artifacts through the M58
pretokenized input path across CPU and optional CUDA backend smoke paths.

No network access is attempted.  Public model files, tokenizer assets,
generated token ID files, and generated .att1 artifacts are expected to live
outside Git.
"""

import argparse
import json
import os
import subprocess
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from validate_public_backends import (  # noqa: E402
    _build_cases,
    _format_run,
    _require_local_path,
    _run_bench,
)


def _read_prompt(args):
    if args.prompt_text is not None:
        return args.prompt_text
    try:
        with open(args.prompt_file, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError as exc:
        raise ValueError(f"cannot read prompt file {args.prompt_file!r}: {exc}") from exc


def _read_token_ids(path):
    ids = []
    with open(path, "r", encoding="ascii") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                ids.append(int(line))
            except ValueError:
                raise ValueError(
                    f"{path!r} line {lineno}: not an integer token ID: {line!r}"
                ) from None
    if not ids:
        raise ValueError(f"{path!r}: tokenizer produced no token IDs")
    return ids


def _run_tokenizer(args):
    helper = os.path.join(_THIS_DIR, "tokenize_hf.py")
    cmd = [
        sys.executable,
        helper,
        "--tokenizer", args.tokenizer_dir,
        "--out", args.tokens_file,
        "--json-out", args.tokenizer_json,
        "--add-special-tokens", args.add_special_tokens,
    ]
    if args.prompt_text is not None:
        cmd += ["--text", args.prompt_text]
    else:
        cmd += ["--text-file", args.prompt_file]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc), 1

    if result.returncode != 0:
        msg = "\n".join([result.stdout, result.stderr]).strip()
        return msg or f"tokenize_hf.py exited {result.returncode}", result.returncode
    return None, 0


def _validate_paths(args):
    _require_local_path("source model", args.model_dir, directory=True)
    _require_local_path("tokenizer assets", args.tokenizer_dir, directory=True)
    _require_local_path("f32 ATT-1 artifact", args.att1_f32)
    _require_local_path("q8 ATT-1 artifact", args.att1_q8)
    _require_local_path("att1-bench", args.bench)
    if args.prompt_file is not None:
        _require_local_path("prompt text", args.prompt_file)


def run_validation(args):
    try:
        _validate_paths(args)
        prompt_text = _read_prompt(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return {}, 1

    tok_error, tok_rc = _run_tokenizer(args)
    if tok_error:
        print(f"error: tokenization failed: {tok_error}", file=sys.stderr)
        return {}, 2 if tok_rc == 2 else 1

    try:
        token_ids = _read_token_ids(args.tokens_file)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return {}, 1

    rows = []
    for artifact, backend, mode in _build_cases(args.att1_f32, args.att1_q8):
        rows.append(
            _run_bench(
                args.bench,
                artifact,
                backend,
                mode,
                args.tokens_file,
                args.tokens,
                args.tiles,
            )
        )

    failures = [row for row in rows if row["status"] == "fail"]
    overall = "pass" if not failures else "fail"
    report = {
        "source_model_path": args.model_dir,
        "tokenizer_path": args.tokenizer_dir,
        "prompt_text": prompt_text,
        "token_ids": token_ids,
        "token_count": len(token_ids),
        "tokens_file": args.tokens_file,
        "generated_tokens_requested": args.tokens,
        "tiles": args.tiles,
        "runs": rows,
        "result": overall,
    }

    print("# ATT-1 public tokenized end-to-end validation")
    print(f"source_model_path: {args.model_dir}")
    print(f"tokenizer_path: {args.tokenizer_dir}")
    print(f"prompt_text: {prompt_text}")
    print(f"token_ids: {token_ids}")
    print(f"token_count: {len(token_ids)}")
    print(f"tokens_file: {args.tokens_file}")
    print(f"generated_tokens_requested: {args.tokens}")
    print(f"tiles: {args.tiles}")
    for row in rows:
        print(_format_run(row))
    print(f"result: {overall}")
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

    return report, (0 if overall == "pass" else 2)


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize local prompt text and validate local ATT-1 artifacts."
    )
    parser.add_argument(
        "--model-dir", required=True, metavar="DIR",
        help="Local source model directory; provenance only"
    )
    parser.add_argument(
        "--tokenizer-dir", required=True, metavar="DIR",
        help="Local tokenizer asset directory used by compiler/tokenize_hf.py"
    )
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt-text", metavar="TEXT", help="Prompt text")
    prompt.add_argument("--prompt-file", metavar="PATH", help="UTF-8 prompt file")
    parser.add_argument("--att1-f32", required=True, metavar="PATH")
    parser.add_argument("--att1-q8", required=True, metavar="PATH")
    parser.add_argument(
        "--tokens-file", required=True, metavar="PATH",
        help="Output token IDs file, later passed to att1-bench --tokens-file"
    )
    parser.add_argument(
        "--tokenizer-json", required=True, metavar="PATH",
        help="Output JSON metadata from compiler/tokenize_hf.py"
    )
    parser.add_argument(
        "--add-special-tokens", default="true", choices=["true", "false"],
        help="Forwarded to tokenize_hf.py (default: true)"
    )
    parser.add_argument(
        "--tokens", type=int, default=1,
        help="Generated token count per bench run (default: 1)"
    )
    parser.add_argument(
        "--tiles", type=int, default=2,
        help="Cluster tile count (default: 2)"
    )
    parser.add_argument(
        "--bench", default=os.path.join("build", "att1-bench"), metavar="PATH",
        help="Path to att1-bench (default: build/att1-bench)"
    )
    parser.add_argument("--report-json", default=None, metavar="PATH")
    args = parser.parse_args()

    if args.tokens < 1:
        print("error: --tokens must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.tiles < 1:
        print("error: --tiles must be >= 1", file=sys.stderr)
        sys.exit(1)

    _, exit_code = run_validation(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
