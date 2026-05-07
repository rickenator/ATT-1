#!/usr/bin/env python3
"""
ATT-1 q4 fixture generator (Milestone 77).

Writes models/q4_tiny/model.att1: a minimal ATT-1 v1 model where all 2D
weight tensors (projections + output.weight) are stored as grouped q4 with
group_size=32.  1D tensors (RMSNorm weights) and tok_embeddings.weight remain
float32, matching the inference pipeline convention.

Model configuration
-------------------
  vocab_size   = 256
  n_layers     = 2
  n_heads      = 2
  d_model      = 32   (cols divisible by group_size=32 ✓)
  d_ff         = 64   (cols=32 or 64, both divisible by group_size=32 ✓)
  max_seq_len  = 8
  rope_dim     = 8
  n_tiles      = 1
  shard_count  = 0

Tensor shapes
-------------
  tok_embeddings.weight           [256, 32]  f32
  layers.N.attention_norm.weight  [32]       f32
  layers.N.attention.wq.weight    [32, 32]   q4  g32
  layers.N.attention.wk.weight    [32, 32]   q4  g32
  layers.N.attention.wv.weight    [32, 32]   q4  g32
  layers.N.attention.wo.weight    [32, 32]   q4  g32
  layers.N.ffn_norm.weight        [32]       f32
  layers.N.ffn.w_gate.weight      [32, 64]   q4  g32
  layers.N.ffn.w_up.weight        [32, 64]   q4  g32
  layers.N.ffn.w_down.weight      [64, 32]   q4  g32
  output_norm.weight              [32]       f32
  output.weight                   [32, 256]  q4  g32

Usage:
    python3 compiler/make_q4_fixture.py [output_path]
"""

import math
import os
import struct
import sys

# ---------------------------------------------------------------------------
# Wire format constants
# ---------------------------------------------------------------------------

MAGIC           = b"ATT1MODL"
VERSION         = 1
HEADER_SIZE     = 80
CONFIG_SIZE     = 36
DESC_SIZE       = 128
DTYPE_F32       = 1
DTYPE_Q4        = 3
Q4_GROUP_SIZE   = 32
Q4_FLAGS_MASK   = 0xFF   # group_size stored in flags[7:0]


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "vocab_size":   256,
    "n_layers":       2,
    "n_heads":        2,
    "d_model":       32,
    "d_ff":          64,
    "max_seq_len":    8,
    "rope_dim":       8,
    "n_tiles":        1,
    "shard_count":    0,
}


# ---------------------------------------------------------------------------
# Deterministic weight generators
# ---------------------------------------------------------------------------

def f32_weights(tensor_index, count):
    """Deterministic f32 values: base + small linear gradient."""
    base = (tensor_index + 1) * 0.01
    return [base + i * 0.001 for i in range(count)]


def q4_float_weights(tensor_index, rows, cols):
    """
    Deterministic float weights that will quantize to non-trivial q4 values.
    Uses a sine wave to guarantee |max| > 0 in every group.
    """
    base = (tensor_index + 1) * 0.1
    result = []
    for r in range(rows):
        for c in range(cols):
            result.append(math.sin(base + (r * cols + c) * 0.05) * 0.7)
    return result


# ---------------------------------------------------------------------------
# Q4 quantization helper
# ---------------------------------------------------------------------------

def q4_quantize(float_weights, rows, cols, group_size=32):
    """
    Quantize a row-major float matrix to the ATT-1 q4 wire format:
      packed layout : rows * cols // 2  bytes  (nibbles, low=even, high=odd)
      scale layout  : rows * (cols // group_size) * 4  bytes  (float32)

    Signed int4 range: [-7, 7].  Scale = max_abs / 7.0 (or 1.0 for zero rows).
    Nibble packing: packed[i//2] low nibble = element i (even),
                    packed[i//2] high nibble = element i+1 (odd).
    """
    assert cols % 2 == 0, "cols must be even"
    assert cols % group_size == 0, "cols must be divisible by group_size"

    n_groups_per_row = cols // group_size
    packed = bytearray(rows * cols // 2)
    scales = []

    for r in range(rows):
        for g in range(n_groups_per_row):
            start = r * cols + g * group_size
            group = float_weights[start : start + group_size]
            max_abs = max(abs(v) for v in group)
            scale = max_abs / 7.0 if max_abs != 0.0 else 1.0
            scales.append(scale)

            base_packed = r * (cols // 2) + g * (group_size // 2)
            int4_vals = [max(-7, min(7, int(round(v / scale)))) for v in group]

            for j in range(0, group_size, 2):
                lo = int4_vals[j] & 0x0F
                hi = int4_vals[j + 1] & 0x0F
                packed[base_packed + j // 2] = lo | (hi << 4)

    scale_bytes = struct.pack(f"<{len(scales)}f", *scales)
    return bytes(packed), scale_bytes


# ---------------------------------------------------------------------------
# Tensor builders
# ---------------------------------------------------------------------------

def make_f32_tensor(name, shape, tensor_index):
    count = 1
    for d in shape:
        count *= d
    data = struct.pack(f"<{count}f", *f32_weights(tensor_index, count))
    return {"name": name, "dtype": DTYPE_F32, "shape": shape, "data": data}


def make_q4_tensor(name, shape, tensor_index, group_size=Q4_GROUP_SIZE):
    assert len(shape) == 2, "q4 tensors must be 2D"
    rows, cols = shape
    floats = q4_float_weights(tensor_index, rows, cols)
    packed, scale_bytes = q4_quantize(floats, rows, cols, group_size)
    data = packed + scale_bytes
    return {
        "name": name,
        "dtype": DTYPE_Q4,
        "shape": shape,
        "data": data,
        "flags": group_size & Q4_FLAGS_MASK,
    }


# ---------------------------------------------------------------------------
# Descriptor builder
# ---------------------------------------------------------------------------

def descriptor(tensor, offset):
    name_bytes = tensor["name"].encode("ascii")
    if len(name_bytes) >= 64:
        raise ValueError(f"tensor name too long: {tensor['name']!r}")
    name_bytes = name_bytes + b"\x00" * (64 - len(name_bytes))

    shape = list(tensor["shape"]) + [1] * (4 - len(tensor["shape"]))
    flags = tensor.get("flags", 0)

    return struct.pack(
        "<64sIIQQQQQQII",
        name_bytes,
        tensor["dtype"],
        len(tensor["shape"]),
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        offset,
        len(tensor["data"]),
        0,          # shard_id
        flags,
    )


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model():
    cfg = CONFIG
    d   = cfg["d_model"]
    ff  = cfg["d_ff"]
    v   = cfg["vocab_size"]

    tensors = []
    idx = 0

    tensors.append(make_f32_tensor("tok_embeddings.weight", [v, d], idx)); idx += 1

    for layer in range(cfg["n_layers"]):
        p = f"layers.{layer}"
        tensors.append(make_f32_tensor(f"{p}.attention_norm.weight", [d], idx)); idx += 1
        tensors.append(make_q4_tensor(f"{p}.attention.wq.weight",    [d, d],  idx)); idx += 1
        tensors.append(make_q4_tensor(f"{p}.attention.wk.weight",    [d, d],  idx)); idx += 1
        tensors.append(make_q4_tensor(f"{p}.attention.wv.weight",    [d, d],  idx)); idx += 1
        tensors.append(make_q4_tensor(f"{p}.attention.wo.weight",    [d, d],  idx)); idx += 1
        tensors.append(make_f32_tensor(f"{p}.ffn_norm.weight",       [d],     idx)); idx += 1
        tensors.append(make_q4_tensor(f"{p}.ffn.w_gate.weight",      [d, ff], idx)); idx += 1
        tensors.append(make_q4_tensor(f"{p}.ffn.w_up.weight",        [d, ff], idx)); idx += 1
        tensors.append(make_q4_tensor(f"{p}.ffn.w_down.weight",      [ff, d], idx)); idx += 1

    tensors.append(make_f32_tensor("output_norm.weight", [d], idx)); idx += 1
    tensors.append(make_q4_tensor("output.weight",       [d, v], idx));  idx += 1

    config_offset = HEADER_SIZE
    desc_offset   = config_offset + CONFIG_SIZE
    data_offset   = desc_offset + len(tensors) * DESC_SIZE

    data_blob = bytearray()
    desc_blob = bytearray()
    offset = 0
    for tensor in tensors:
        desc_blob += descriptor(tensor, offset)
        data_blob += tensor["data"]
        offset += len(tensor["data"])

    header = struct.pack(
        "<8sIIQQQQQQQQ",
        MAGIC,
        VERSION,
        HEADER_SIZE,
        config_offset,
        CONFIG_SIZE,
        desc_offset,
        len(tensors),
        data_offset,
        len(data_blob),
        0,   # shard_offset
        0,   # shard_size
    )

    config_blob = struct.pack(
        "<IIIIIIIII",
        cfg["vocab_size"],
        cfg["n_layers"],
        cfg["n_heads"],
        cfg["d_model"],
        cfg["d_ff"],
        cfg["max_seq_len"],
        cfg["rope_dim"],
        cfg["n_tiles"],
        cfg["shard_count"],
    )

    return header + config_blob + bytes(desc_blob) + bytes(data_blob)


# ---------------------------------------------------------------------------
# Expected-nbytes self-check (mirrors att1_tensor_nbytes_expected in C)
# ---------------------------------------------------------------------------

def expected_q4_nbytes(rows, cols, group_size):
    packed_bytes = rows * cols // 2
    n_groups     = rows * (cols // group_size)
    scale_bytes  = n_groups * 4
    return packed_bytes + scale_bytes


def self_check(tensors):
    for t in tensors:
        if t["dtype"] == DTYPE_Q4:
            rows, cols = t["shape"]
            gs = t.get("flags", 0) & Q4_FLAGS_MASK
            if gs == 0:
                gs = 32  # default
            expected = expected_q4_nbytes(rows, cols, gs)
            actual   = len(t["data"])
            if actual != expected:
                raise AssertionError(
                    f"{t['name']}: expected nbytes={expected} got {actual}"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "models/q4_tiny/model.att1"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    blob = build_model()

    # Parse back to run self-check
    cfg = CONFIG
    d, ff, v = cfg["d_model"], cfg["d_ff"], cfg["vocab_size"]
    tensors = []
    idx = 0
    tensors.append(make_f32_tensor("tok_embeddings.weight", [v, d], idx)); idx += 1
    for layer in range(cfg["n_layers"]):
        p = f"layers.{layer}"
        tensors.append(make_f32_tensor(f"{p}.attention_norm.weight", [d], idx)); idx += 1
        tensors.append(make_q4_tensor(f"{p}.attention.wq.weight",    [d, d],  idx)); idx += 1
        tensors.append(make_q4_tensor(f"{p}.attention.wk.weight",    [d, d],  idx)); idx += 1
        tensors.append(make_q4_tensor(f"{p}.attention.wv.weight",    [d, d],  idx)); idx += 1
        tensors.append(make_q4_tensor(f"{p}.attention.wo.weight",    [d, d],  idx)); idx += 1
        tensors.append(make_f32_tensor(f"{p}.ffn_norm.weight",       [d],     idx)); idx += 1
        tensors.append(make_q4_tensor(f"{p}.ffn.w_gate.weight",      [d, ff], idx)); idx += 1
        tensors.append(make_q4_tensor(f"{p}.ffn.w_up.weight",        [d, ff], idx)); idx += 1
        tensors.append(make_q4_tensor(f"{p}.ffn.w_down.weight",      [ff, d], idx)); idx += 1
    tensors.append(make_f32_tensor("output_norm.weight", [d], idx)); idx += 1
    tensors.append(make_q4_tensor("output.weight",       [d, v], idx)); idx += 1
    self_check(tensors)

    with open(out_path, "wb") as f:
        f.write(blob)
    print(f"wrote {len(blob)} bytes to {out_path}")


if __name__ == "__main__":
    main()
