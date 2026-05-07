#!/usr/bin/env python3
"""
ATT-1 tokenizer asset scanner (Milestone 52).

Scans a directory for tokenizer asset files, parses available metadata,
and reports tokenizer type, vocabulary size, special-token IDs, byte-fallback
status, normalization/pretokenizer info, and per-file checksums.

No external dependencies.  Does not import or execute tokenizer logic.
Does not modify any files.

Supported asset files:
    tokenizer.json          — Hugging Face tokenizers JSON (BPE, Unigram, …)
    tokenizer_config.json   — HF tokenizer class and special-token config
    special_tokens_map.json — BOS/EOS/PAD/UNK string → token string mapping
    tokenizer.model         — detected but reported as unsupported (M52)

Tokenizer type codes:
    bpe_json        tokenizer.json present with model.type BPE or WordPiece
    sentencepiece   tokenizer.model present without tokenizer.json
    byte            no tokenizer assets present; inferred byte-level default
    unknown         assets present but type could not be determined

Usage:
    # scan a directory of tokenizer assets
    python3 compiler/scan_tokenizer.py path/to/tokenizer/dir

    # cross-check vocab_size against a model config.json
    python3 compiler/scan_tokenizer.py path/to/tokenizer/dir \\
        --config compiler/fixtures/tiny_llama/config.json

    # JSON output
    python3 compiler/scan_tokenizer.py path/to/tokenizer/dir --json

    # suppress per-asset detail
    python3 compiler/scan_tokenizer.py path/to/tokenizer/dir --no-detail

Exit codes:
    0  scan ok (no fatal errors; warnings may appear)
    1  fatal scan error (file unreadable, malformed JSON, etc.)
    2  validation warning (vocab_size mismatch between tokenizer and config)
"""

import argparse
import hashlib
import json
import os
import sys

# ---------------------------------------------------------------------------
# Known asset filenames
# ---------------------------------------------------------------------------

_ASSET_NAMES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
]

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TokenizerScanError(Exception):
    """Fatal error during tokenizer asset scan."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: str) -> str:
    """Return the hex SHA-256 digest of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_json_file(path: str) -> dict:
    """Load and return a JSON object from a file.

    Raises TokenizerScanError on I/O or parse failure.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except OSError as exc:
        raise TokenizerScanError(f"could not read {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise TokenizerScanError(
            f"{path}: not valid UTF-8: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise TokenizerScanError(
            f"{path}: JSON parse error: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise TokenizerScanError(
            f"{path}: JSON root must be an object, got "
            f"{type(data).__name__}"
        )
    return data


def _resolve_special_token_id(token_str, vocab: dict) -> int:
    """Look up a token string in a vocab dict; return -1 if not found."""
    if token_str is None:
        return -1
    # token_str may arrive as a plain string or as a dict with a "content" key
    # (HF tokenizer_config.json sometimes uses the latter for complex specials).
    if isinstance(token_str, dict):
        token_str = token_str.get("content") or token_str.get("__type__")
    if not isinstance(token_str, str):
        return -1
    return vocab.get(token_str, -1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_tokenizer_dir(dirpath: str, config_path: str = None) -> dict:
    """Scan a directory for tokenizer asset files.

    Parameters:
        dirpath     Path to a directory containing tokenizer assets.
        config_path Optional path to a model config.json whose vocab_size
                    will be cross-checked against the tokenizer vocabulary.

    Returns a dict with keys:
        dir                  str
        assets               dict {filename: "present"|"absent"|"unsupported"}
        asset_hashes         dict {filename: sha256_hex}  (present files only)
        tokenizer_type       str  "bpe_json"|"sentencepiece"|"byte"|"unknown"
        vocab_size           int or None
        bos_id               int  (-1 = not found)
        eos_id               int  (-1 = not found)
        pad_id               int  (-1 = not found)
        unk_id               int  (-1 = not found)
        byte_fallback        bool or None
        normalizer           str or None  (type name from tokenizer.json)
        pretokenizer         str or None  (type name from tokenizer.json)
        config_vocab_size    int or None  (only if config_path given)
        vocab_size_match     bool or None (True/False, or None if no config)
        warnings             list[str]
        errors               list[str]   (non-fatal issues)

    Raises TokenizerScanError for fatal I/O or parse errors.
    """
    if not os.path.isdir(dirpath):
        raise TokenizerScanError(f"tokenizer directory not found: {dirpath}")

    assets       = {}
    asset_hashes = {}
    warnings     = []
    errors       = []

    # ----------------------------------------------------------------
    # Discover asset files
    # ----------------------------------------------------------------
    for fname in _ASSET_NAMES:
        fpath = os.path.join(dirpath, fname)
        if os.path.isfile(fpath):
            assets[fname] = "present"
            asset_hashes[fname] = _sha256_file(fpath)
        else:
            assets[fname] = "absent"

    # tokenizer.model is detected but its binary format is unsupported in M52.
    if assets.get("tokenizer.model") == "present":
        assets["tokenizer.model"] = "unsupported"
        errors.append(
            "tokenizer.model detected but binary SentencePiece format is "
            "not yet supported (M52); use tokenizer.json if available"
        )

    # ----------------------------------------------------------------
    # Parse tokenizer.json
    # ----------------------------------------------------------------
    tok_json_data  = None
    vocab          = {}
    vocab_size     = None
    byte_fallback  = None
    normalizer     = None
    pretokenizer   = None
    model_type_str = None

    if assets.get("tokenizer.json") == "present":
        tok_json_path = os.path.join(dirpath, "tokenizer.json")
        tok_json_data = _load_json_file(tok_json_path)  # raises on failure

        model_obj = tok_json_data.get("model")
        if not isinstance(model_obj, dict):
            errors.append(
                "tokenizer.json: missing or non-object 'model' key; "
                "cannot determine tokenizer type or vocabulary"
            )
        else:
            model_type_str = model_obj.get("type")
            vocab_raw      = model_obj.get("vocab")
            if isinstance(vocab_raw, dict):
                vocab      = vocab_raw
                vocab_size = len(vocab)
            else:
                errors.append(
                    "tokenizer.json: model.vocab is missing or not an object"
                )
            byte_fallback = model_obj.get("byte_fallback")  # may be None

        # normalizer
        norm_obj = tok_json_data.get("normalizer")
        if isinstance(norm_obj, dict):
            normalizer = norm_obj.get("type") or "present"

        # pre_tokenizer
        pre_obj = tok_json_data.get("pre_tokenizer")
        if isinstance(pre_obj, dict):
            pretokenizer = pre_obj.get("type") or "present"

    # ----------------------------------------------------------------
    # Parse tokenizer_config.json
    # ----------------------------------------------------------------
    bos_str = eos_str = pad_str = unk_str = None

    if assets.get("tokenizer_config.json") == "present":
        cfg_path = os.path.join(dirpath, "tokenizer_config.json")
        cfg_data = _load_json_file(cfg_path)  # raises on failure

        bos_str = cfg_data.get("bos_token")
        eos_str = cfg_data.get("eos_token")
        pad_str = cfg_data.get("pad_token")
        unk_str = cfg_data.get("unk_token")

        # vocab_size in tokenizer_config.json may differ from actual vocab length
        cfg_vocab_size = cfg_data.get("vocab_size")
        if cfg_vocab_size is not None:
            if vocab_size is None:
                vocab_size = int(cfg_vocab_size)
            elif int(cfg_vocab_size) != vocab_size:
                errors.append(
                    f"tokenizer_config.json vocab_size ({cfg_vocab_size}) "
                    f"does not match tokenizer.json model.vocab length "
                    f"({vocab_size})"
                )

    # ----------------------------------------------------------------
    # Parse special_tokens_map.json (supplement, prefer tokenizer_config.json)
    # ----------------------------------------------------------------
    if assets.get("special_tokens_map.json") == "present":
        stm_path = os.path.join(dirpath, "special_tokens_map.json")
        stm_data = _load_json_file(stm_path)  # raises on failure

        # Only fill in what tokenizer_config.json didn't provide
        if bos_str is None:
            bos_str = stm_data.get("bos_token")
        if eos_str is None:
            eos_str = stm_data.get("eos_token")
        if pad_str is None:
            pad_str = stm_data.get("pad_token")
        if unk_str is None:
            unk_str = stm_data.get("unk_token")

    # ----------------------------------------------------------------
    # Resolve special-token IDs from vocab
    # ----------------------------------------------------------------
    bos_id = _resolve_special_token_id(bos_str, vocab)
    eos_id = _resolve_special_token_id(eos_str, vocab)
    pad_id = _resolve_special_token_id(pad_str, vocab)
    unk_id = _resolve_special_token_id(unk_str, vocab)

    # ----------------------------------------------------------------
    # Determine tokenizer type
    # ----------------------------------------------------------------
    if assets.get("tokenizer.json") == "present":
        if model_type_str in ("BPE", "WordPiece", "Unigram"):
            tokenizer_type = "bpe_json"
        elif model_type_str is not None:
            tokenizer_type = "bpe_json"  # still JSON-based, type may vary
            warnings.append(
                f"tokenizer.json model.type is {model_type_str!r}; "
                "treating as bpe_json"
            )
        else:
            tokenizer_type = "unknown"
            errors.append(
                "tokenizer.json: could not determine model type from "
                "'model.type'"
            )
    elif assets.get("tokenizer.model") == "unsupported":
        tokenizer_type = "sentencepiece"
    elif all(assets[k] == "absent" for k in _ASSET_NAMES):
        tokenizer_type = "byte"
        warnings.append(
            "no tokenizer asset files found; defaulting to byte tokenizer"
        )
    else:
        tokenizer_type = "unknown"

    # ----------------------------------------------------------------
    # Cross-check with model config.json if given
    # ----------------------------------------------------------------
    config_vocab_size = None
    vocab_size_match  = None

    if config_path is not None:
        if not os.path.isfile(config_path):
            errors.append(f"config file not found: {config_path}")
        else:
            try:
                cfg = _load_json_file(config_path)
                config_vocab_size = (
                    cfg.get("vocab_size")
                    or cfg.get("n_vocab")
                )
                if config_vocab_size is not None:
                    config_vocab_size = int(config_vocab_size)
                    if vocab_size is not None:
                        vocab_size_match = (vocab_size == config_vocab_size)
                        if not vocab_size_match:
                            warnings.append(
                                f"vocab_size mismatch: tokenizer has "
                                f"{vocab_size} tokens, config has "
                                f"{config_vocab_size}"
                            )
            except TokenizerScanError as exc:
                errors.append(f"could not parse config: {exc}")

    return {
        "dir":               dirpath,
        "assets":            assets,
        "asset_hashes":      asset_hashes,
        "tokenizer_type":    tokenizer_type,
        "vocab_size":        vocab_size,
        "bos_id":            bos_id,
        "eos_id":            eos_id,
        "pad_id":            pad_id,
        "unk_id":            unk_id,
        "byte_fallback":     byte_fallback,
        "normalizer":        normalizer,
        "pretokenizer":      pretokenizer,
        "config_vocab_size": config_vocab_size,
        "vocab_size_match":  vocab_size_match,
        "warnings":          warnings,
        "errors":            errors,
    }


def format_tokenizer_report(result: dict, show_detail: bool = True) -> str:
    """Render a human-readable tokenizer scan report."""
    lines = []
    lines.append(f"tokenizer_dir={result['dir']}")
    lines.append(f"tokenizer_type={result['tokenizer_type']}")
    lines.append(
        f"vocab_size={result['vocab_size'] if result['vocab_size'] is not None else 'unknown'}"
    )
    lines.append(f"bos_id={result['bos_id']}")
    lines.append(f"eos_id={result['eos_id']}")
    lines.append(f"pad_id={result['pad_id']}")
    lines.append(f"unk_id={result['unk_id']}")
    lines.append(
        f"byte_fallback={'true' if result['byte_fallback'] else ('false' if result['byte_fallback'] is False else 'unknown')}"
    )
    lines.append(
        f"normalizer={result['normalizer'] if result['normalizer'] else 'none'}"
    )
    lines.append(
        f"pretokenizer={result['pretokenizer'] if result['pretokenizer'] else 'none'}"
    )

    if result["config_vocab_size"] is not None:
        lines.append(f"config_vocab_size={result['config_vocab_size']}")
        if result["vocab_size_match"] is True:
            lines.append("vocab_size_match=yes")
        elif result["vocab_size_match"] is False:
            lines.append("vocab_size_match=no")

    if show_detail:
        lines.append("")
        for fname in _ASSET_NAMES:
            status = result["assets"].get(fname, "absent")
            h      = result["asset_hashes"].get(fname, "")
            if h:
                lines.append(f"  {fname:<40} {status}  sha256={h[:16]}…")
            else:
                lines.append(f"  {fname:<40} {status}")

    if result["errors"]:
        lines.append("")
        for err in result["errors"]:
            lines.append(f"  error: {err}")

    if result["warnings"]:
        lines.append("")
        for w in result["warnings"]:
            lines.append(f"  warning: {w}")

    lines.append("")
    n_errors = len(result["errors"])
    if n_errors:
        lines.append(f"scan: {n_errors} error(s)")
    else:
        lines.append("scan: ok")

    # vocab_size_match=no is the fatal exit-2 condition; note it clearly
    if result["vocab_size_match"] is False:
        lines.append("vocab_mismatch: fatal")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="ATT-1 tokenizer asset scanner (M52)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dirpath",
        help="Path to directory containing tokenizer asset files",
    )
    parser.add_argument(
        "--config", metavar="PATH", default=None,
        help="Path to model config.json for vocab_size cross-check",
    )
    parser.add_argument(
        "--json", action="store_true", dest="output_json",
        help="Write JSON to stdout instead of human-readable text",
    )
    parser.add_argument(
        "--no-detail", action="store_true",
        help="Suppress per-asset file listing",
    )
    args = parser.parse_args()

    try:
        result = scan_tokenizer_dir(args.dirpath, config_path=args.config)
    except TokenizerScanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.output_json:
        print(json.dumps(result, indent=2))
    else:
        print(format_tokenizer_report(result, show_detail=not args.no_detail))

    if result["errors"]:
        sys.exit(1)
    if result["vocab_size_match"] is False:
        sys.exit(2)


if __name__ == "__main__":
    main()
