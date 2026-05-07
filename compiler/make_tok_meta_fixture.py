#!/usr/bin/env python3
"""
ATT-1 tokenizer metadata fixture generator (Milestone 54).

Writes models/tok_meta/model.att1: a version-2 .att1 file with a minimal
1-tensor model and a valid 96-byte tokenizer metadata section.

Model config matches compiler/fixtures/tiny_tokenizer (vocab_size=16).

The asset_hash field is set to the canonical SHA-256 of the
compiler/fixtures/tiny_tokenizer asset files (precomputed).

Usage:
    python3 compiler/make_tok_meta_fixture.py [output_path]
"""

import os
import struct
import sys

# ---------------------------------------------------------------------------
# .att1 wire format constants
# ---------------------------------------------------------------------------

MAGIC       = b"ATT1MODL"
VERSION_2   = 2
HEADER_SIZE = 96          # v2 extended header
CONFIG_SIZE = 36
DESC_SIZE   = 128
DTYPE_F32   = 1

# ---------------------------------------------------------------------------
# Tokenizer metadata section constants (matches att1_tok_meta.h)
# ---------------------------------------------------------------------------

TOK_META_SIZE           = 96
TOK_META_SCHEMA_VERSION = 1

TOK_TYPE_BPE_JSON  = 2

TOK_NORM_NONE      = 1
TOK_PRETOK_BYTE_LEVEL = 2

TOK_HASH_SHA256    = 1

# Canonical asset hash for compiler/fixtures/tiny_tokenizer (precomputed by
# compiler/scan_tokenizer.py _canonical_asset_hash).
TOK_ASSET_HASH_HEX = (
    "da38c00aa0b62d2699c46cb2d1ccadce"
    "14b36815d784265680b78a7a31ab1034"
)

# ---------------------------------------------------------------------------
# Model config for the fixture
# ---------------------------------------------------------------------------

CONFIG = {
    "vocab_size":   16,
    "n_layers":      2,
    "n_heads":       2,
    "d_model":       8,
    "d_ff":         16,
    "max_seq_len": 128,
    "rope_dim":      4,
    "n_tiles":       1,
    "shard_count":   0,
}

# ---------------------------------------------------------------------------
# Single tensor: tok_embeddings.weight  [16, 8]  f32
# ---------------------------------------------------------------------------

TENSOR_NAME   = "tok_embeddings.weight"
TENSOR_SHAPE  = [16, 8]
TENSOR_NBYTES = 16 * 8 * 4   # 512 bytes

def tensor_data():
    """Deterministic f32 payload: each element = 0.0001 * index."""
    count = 16 * 8
    return struct.pack(f"<{count}f", *[i * 0.0001 for i in range(count)])

# ---------------------------------------------------------------------------
# File layout (all offsets are absolute)
# ---------------------------------------------------------------------------
#   header          @   0  (96 bytes)
#   config          @  96  (36 bytes)
#   tensor desc     @ 132  (128 bytes)
#   tensor data     @ 260  (512 bytes)
#   tok_meta        @ 772  (96 bytes)
#   EOF             @ 868

LAYOUT = {
    "config_offset":    96,
    "desc_offset":      132,
    "data_offset":      260,
    "data_size":        TENSOR_NBYTES,
    "shard_offset":     0,
    "shard_size":       0,
    "tok_meta_offset":  772,
    "tok_meta_size":    TOK_META_SIZE,
    "file_size":        868,
}


def pack_u32le(v):
    return struct.pack("<I", v)


def pack_i32le(v):
    return struct.pack("<i", v)


def pack_u64le(v):
    return struct.pack("<Q", v)


def build_header():
    """96-byte v2 header."""
    h  = MAGIC
    h += pack_u32le(VERSION_2)
    h += pack_u32le(HEADER_SIZE)
    h += pack_u64le(LAYOUT["config_offset"])
    h += pack_u64le(CONFIG_SIZE)
    h += pack_u64le(LAYOUT["desc_offset"])
    h += pack_u64le(1)                          # tensor_count
    h += pack_u64le(LAYOUT["data_offset"])
    h += pack_u64le(LAYOUT["data_size"])
    h += pack_u64le(LAYOUT["shard_offset"])
    h += pack_u64le(LAYOUT["shard_size"])
    h += pack_u64le(LAYOUT["tok_meta_offset"])  # new in v2
    h += pack_u64le(LAYOUT["tok_meta_size"])    # new in v2
    assert len(h) == HEADER_SIZE, f"header len={len(h)}"
    return h


def build_config():
    """36-byte config section (9 × u32)."""
    c = b""
    for key in ("vocab_size", "n_layers", "n_heads", "d_model", "d_ff",
                "max_seq_len", "rope_dim", "n_tiles", "shard_count"):
        c += pack_u32le(CONFIG[key])
    assert len(c) == CONFIG_SIZE
    return c


def build_tensor_desc():
    """128-byte tensor descriptor."""
    name = TENSOR_NAME.encode("ascii")
    name = name + b"\x00" * (64 - len(name))
    ndims  = len(TENSOR_SHAPE)
    shape  = TENSOR_SHAPE + [1] * (4 - ndims)
    desc   = name
    desc  += pack_u32le(DTYPE_F32)           # dtype
    desc  += pack_u32le(ndims)               # ndims
    for s in shape:
        desc += pack_u64le(s)               # shape[4]
    desc  += pack_u64le(0)                  # offset relative to data section
    desc  += pack_u64le(TENSOR_NBYTES)      # nbytes
    desc  += pack_u32le(0)                  # shard_id
    desc  += pack_u32le(0)                  # flags
    assert len(desc) == DESC_SIZE, f"desc len={len(desc)}"
    return desc


def build_tok_meta():
    """96-byte tokenizer metadata section."""
    asset_hash = bytes.fromhex(TOK_ASSET_HASH_HEX)
    assert len(asset_hash) == 32

    sec  = pack_u32le(TOK_META_SCHEMA_VERSION)   # @  0
    sec += pack_u32le(TOK_TYPE_BPE_JSON)          # @  4
    sec += pack_u32le(CONFIG["vocab_size"])        # @  8
    sec += pack_i32le(1)    # bos_token_id         @ 12
    sec += pack_i32le(2)    # eos_token_id         @ 16
    sec += pack_i32le(-1)   # pad_token_id (absent)@ 20
    sec += pack_i32le(0)    # unk_token_id         @ 24
    sec += pack_u32le(0)    # byte_fallback=false  @ 28
    sec += pack_u32le(TOK_NORM_NONE)              # @ 32
    sec += pack_u32le(TOK_PRETOK_BYTE_LEVEL)      # @ 36
    sec += pack_u32le(TOK_HASH_SHA256)            # @ 40
    sec += pack_u32le(0)    # flags (reserved=0)  @ 44
    sec += asset_hash                             # @ 48  (32 bytes)
    sec += pack_u64le(0)    # asset_offset=0      @ 80
    sec += pack_u64le(0)    # asset_size=0        @ 88
    assert len(sec) == TOK_META_SIZE, f"tok_meta len={len(sec)}"
    return sec


def build_model():
    header  = build_header()
    config  = build_config()
    desc    = build_tensor_desc()
    data    = tensor_data()
    tok_sec = build_tok_meta()

    blob = header + config + desc + data + tok_sec
    assert len(blob) == LAYOUT["file_size"], \
        f"file_size={len(blob)} expected={LAYOUT['file_size']}"
    return blob


def main():
    out_path = (sys.argv[1]
                if len(sys.argv) > 1
                else "models/tok_meta/model.att1")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    blob = build_model()
    with open(out_path, "wb") as fh:
        fh.write(blob)
    print(f"wrote {len(blob)} bytes → {out_path}")


if __name__ == "__main__":
    main()
