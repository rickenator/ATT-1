#!/usr/bin/env python3
"""ATT-1 Hugging Face tokenizer helper (Milestone 59).

Converts text to pretokenized token IDs using a local Hugging Face tokenizer
directory.  Outputs IDs in formats accepted by att1-bench --tokenizer external:

  --input-token-ids  accepts comma-separated IDs printed to stdout by default
  --tokens-file      accepts one-ID-per-line files written with --out PATH

No network access.  All tokenizer assets must be present locally (the
tokenizer.json in the specified directory).  This script is under compiler/
and is never imported by or required by the C runtime or make test.  It is
covered by Python-skippable smoke tests only.

Dependencies (one of):
  pip install tokenizers        # HuggingFace Rust tokenizers library (fast)
  pip install transformers      # HuggingFace transformers (heavier)

Usage:
    # print comma-separated IDs to stdout
    python3 compiler/tokenize_hf.py --tokenizer PATH --text "hello world"

    # read text from file
    python3 compiler/tokenize_hf.py --tokenizer PATH --text-file prompt.txt

    # write one-ID-per-line file for att1-bench --tokens-file
    python3 compiler/tokenize_hf.py --tokenizer PATH --text "hi" --out ids.txt

    # write JSON metadata file
    python3 compiler/tokenize_hf.py --tokenizer PATH --text "hi" \\
        --json-out out.json

    # skip BOS/EOS special tokens
    python3 compiler/tokenize_hf.py --tokenizer PATH --text "hi" \\
        --add-special-tokens false

    # report tokenization wall-clock time to stderr
    python3 compiler/tokenize_hf.py --tokenizer PATH --text "hi" --timing

    # full pipeline: tokenize, then bench with external IDs
    IDS=$(python3 compiler/tokenize_hf.py --tokenizer PATH --text "hi" \\
            --add-special-tokens false)
    ./build/att1-bench --model MODEL.att1 --tokens 4 --mode single \\
        --tokenizer external --input-token-ids "$IDS"

    # or write to file first
    python3 compiler/tokenize_hf.py --tokenizer PATH --text "hi" \\
        --out build/ids.txt --add-special-tokens false
    ./build/att1-bench --model MODEL.att1 --tokens 4 --mode single \\
        --tokenizer external --tokens-file build/ids.txt

Exit codes:
    0  success
    1  argument or path error (missing directory, missing text file, etc.)
    2  missing dependency: neither tokenizers nor transformers is installed
    3  tokenization error (bad input, tokenizer internal error)
"""

import argparse
import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# Tokenizer backends
# ---------------------------------------------------------------------------

def _try_tokenizers(tok_json_path, text, add_special_tokens):
    """Attempt tokenization via the `tokenizers` library.

    Returns a list of int token IDs on success, or None if the library is not
    installed.  Raises RuntimeError on tokenization failure.
    """
    try:
        from tokenizers import Tokenizer  # noqa: PLC0415
    except ImportError:
        return None

    tok = Tokenizer.from_file(tok_json_path)
    try:
        enc = tok.encode(text, add_special_tokens=add_special_tokens)
    except TypeError:
        # Older tokenizers versions do not support the add_special_tokens kwarg.
        enc = tok.encode(text)
    return list(enc.ids)


def _try_transformers(tok_dir, text, add_special_tokens):
    """Attempt tokenization via the `transformers` library.

    Returns a list of int token IDs on success, or None if the library is not
    installed.  Raises RuntimeError on tokenization failure.
    local_files_only=True prevents any network access.
    """
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415
    except ImportError:
        return None

    tok = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)
    enc = tok(text, add_special_tokens=add_special_tokens,
              return_tensors=None)
    ids = enc["input_ids"]
    # Some transformers versions return a list-of-lists for single inputs.
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(int(x) for x in ids)


def _tokenize(tok_json_path, tok_dir, text, add_special_tokens):
    """Return a list of integer token IDs for *text*.

    Tries the `tokenizers` library first (faster, no heavy dependencies), then
    falls back to `transformers`.  Exits with code 2 if neither is available,
    or code 3 on tokenization error.
    """
    # --- tokenizers library ---
    try:
        ids = _try_tokenizers(tok_json_path, text, add_special_tokens)
        if ids is not None:
            return ids
    except Exception as exc:
        print(f"error: tokenization failed (tokenizers): {exc}",
              file=sys.stderr)
        sys.exit(3)

    # --- transformers library ---
    try:
        ids = _try_transformers(tok_dir, text, add_special_tokens)
        if ids is not None:
            return ids
    except Exception as exc:
        print(f"error: tokenization failed (transformers): {exc}",
              file=sys.stderr)
        sys.exit(3)

    # --- neither available ---
    print(
        "error: neither 'tokenizers' nor 'transformers' package is installed.",
        file=sys.stderr,
    )
    print("Install one of the following and retry:", file=sys.stderr)
    print("  pip install tokenizers        # recommended (faster)",
          file=sys.stderr)
    print("  pip install transformers      # heavier, also works",
          file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _ids_to_csv(ids):
    """Return token IDs as a comma-separated string."""
    return ",".join(str(i) for i in ids)


def _write_ids_file(path, ids):
    """Write one token ID per line to *path* (for att1-bench --tokens-file)."""
    with open(path, "w", encoding="ascii") as fh:
        for tok_id in ids:
            fh.write(f"{tok_id}\n")


def _write_json_out(path, text, ids, add_special_tokens, tok_dir,
                    elapsed_s=None):
    """Write a JSON metadata record to *path*."""
    obj = {
        "tokenizer_path": tok_dir,
        "add_special_tokens": add_special_tokens,
        "input_text_len": len(text),
        # Embed full text only for short inputs to keep the file manageable.
        "input_text": text if len(text) <= 256 else None,
        "token_count": len(ids),
        "token_ids": ids,
    }
    if elapsed_s is not None:
        obj["elapsed_s"] = round(elapsed_s, 9)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="tokenize_hf.py",
        description=(
            "Tokenize text with a local HuggingFace tokenizer for ATT-1."
        ),
    )
    p.add_argument(
        "--tokenizer", required=True, metavar="PATH",
        help="Path to directory containing tokenizer.json",
    )
    text_group = p.add_mutually_exclusive_group(required=True)
    text_group.add_argument(
        "--text", metavar="TEXT",
        help="Text string to tokenize",
    )
    text_group.add_argument(
        "--text-file", metavar="PATH",
        help="Path to a UTF-8 text file whose content is tokenized",
    )
    p.add_argument(
        "--out", metavar="PATH",
        help="Write one token ID per line to PATH (for --tokens-file)",
    )
    p.add_argument(
        "--json-out", metavar="PATH",
        help="Write JSON metadata record to PATH",
    )
    p.add_argument(
        "--add-special-tokens", default="true",
        choices=["true", "false"],
        help="Add BOS/EOS special tokens (default: true)",
    )
    p.add_argument(
        "--timing", action="store_true",
        help="Print tokenization wall-clock time to stderr",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    # --- validate tokenizer directory ---
    tok_dir = args.tokenizer
    if not os.path.isdir(tok_dir):
        print(f"error: tokenizer directory not found: {tok_dir}",
              file=sys.stderr)
        sys.exit(1)

    tok_json = os.path.join(tok_dir, "tokenizer.json")
    if not os.path.isfile(tok_json):
        print(f"error: tokenizer.json not found in {tok_dir}",
              file=sys.stderr)
        sys.exit(1)

    # --- load text ---
    if args.text_file is not None:
        if not os.path.isfile(args.text_file):
            print(f"error: text file not found: {args.text_file}",
                  file=sys.stderr)
            sys.exit(1)
        try:
            with open(args.text_file, "r", encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:
            print(f"error: cannot read text file: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        text = args.text

    add_special = (args.add_special_tokens == "true")

    # --- tokenize ---
    t0 = time.monotonic() if args.timing else None
    ids = _tokenize(tok_json, tok_dir, text, add_special)
    elapsed = (time.monotonic() - t0) if (t0 is not None) else None

    # --- output ---
    # Print comma-separated IDs to stdout; usable directly with
    #   att1-bench --tokenizer external --input-token-ids "$(tokenize_hf.py ...)"
    print(_ids_to_csv(ids))

    if args.out is not None:
        try:
            _write_ids_file(args.out, ids)
        except OSError as exc:
            print(f"error: cannot write output file: {exc}", file=sys.stderr)
            sys.exit(1)

    if args.json_out is not None:
        try:
            _write_json_out(args.json_out, text, ids, add_special,
                            tok_dir, elapsed)
        except OSError as exc:
            print(f"error: cannot write JSON output: {exc}", file=sys.stderr)
            sys.exit(1)

    if args.timing and elapsed is not None:
        print(f"elapsed_s={elapsed:.6f}", file=sys.stderr)


if __name__ == "__main__":
    main()
