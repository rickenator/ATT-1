#!/usr/bin/env python3
"""
Generator for compiler/fixtures/m67_bf16_llama_2l.safetensors (M67 fixture).

Produces a .safetensors file with BF16-encoded seeded pseudo-random weights for
the Milestone 67 BF16 coercion tests.  Uses the same tiny model dimensions as
the M61 fixture (vocab=16, d=8, d_ff=16, n_layers=2, n_heads=2) but stores
all weights as BF16 dtype.

BF16 encoding: each F32 value is rounded to BF16 by dropping the low 16 bits
(truncation towards zero in the mantissa).  This is exactly the byte
representation used by PyTorch's .bfloat16() cast on CPU.

RNG: Python standard library random.gauss(0, WEIGHT_SCALE) with seed=67.
1-D norm/scale tensors: initialised to 1.0.
2-D projection/embedding tensors: seeded Gaussian N(0, WEIGHT_SCALE=0.1).

Usage:
    python3 compiler/fixtures/make_m67_bf16_fixture.py
    python3 compiler/fixtures/make_m67_bf16_fixture.py \\
        --output compiler/fixtures/m67_bf16_llama_2l.safetensors
"""

import argparse
import json
import os
import random
import struct

_VOCAB        = 16
_DMODEL       = 8
_DFF          = 16
_NLAYERS      = 2
_WEIGHT_SCALE = 0.1
_SEED         = 67


def _product(shape):
    r = 1
    for d in shape:
        r *= d
    return r


def _f32_to_bf16_bytes(f):
    """Convert a single Python float to 2 BF16 bytes (little-endian).

    BF16 is identical to the top 16 bits of IEEE 754 float32.  Conversion:
      1. Pack f as 4-byte little-endian F32.
      2. Discard the low 2 bytes (mantissa bits 15-0).
      3. Return the high 2 bytes (sign + exponent + mantissa bits 22-16).

    This is equivalent to PyTorch's bfloat16 truncation cast on CPU.
    """
    packed = struct.pack("<f", f)
    # packed[0]=bits7-0, packed[1]=bits15-8, packed[2]=bits23-16, packed[3]=bits31-24
    # BF16 little-endian stores the high 16 bits of F32:
    #   bf16_low  = packed[2]   (F32 bits 23-16)
    #   bf16_high = packed[3]   (F32 bits 31-24)
    return bytes([packed[2], packed[3]])


def _tensor_list():
    """Return list of (name, shape) pairs in HF LLaMA key order."""
    tensors = [
        ("model.embed_tokens.weight", [_VOCAB, _DMODEL]),
    ]
    for layer in range(_NLAYERS):
        tensors += [
            (f"model.layers.{layer}.input_layernorm.weight",          [_DMODEL]),
            (f"model.layers.{layer}.self_attn.q_proj.weight",         [_DMODEL, _DMODEL]),
            (f"model.layers.{layer}.self_attn.k_proj.weight",         [_DMODEL, _DMODEL]),
            (f"model.layers.{layer}.self_attn.v_proj.weight",         [_DMODEL, _DMODEL]),
            (f"model.layers.{layer}.self_attn.o_proj.weight",         [_DMODEL, _DMODEL]),
            (f"model.layers.{layer}.post_attention_layernorm.weight",  [_DMODEL]),
            (f"model.layers.{layer}.mlp.gate_proj.weight",            [_DFF, _DMODEL]),
            (f"model.layers.{layer}.mlp.up_proj.weight",              [_DFF, _DMODEL]),
            (f"model.layers.{layer}.mlp.down_proj.weight",            [_DMODEL, _DFF]),
        ]
    tensors += [
        ("model.norm.weight",  [_DMODEL]),
        ("lm_head.weight",     [_VOCAB, _DMODEL]),
    ]
    return tensors


def _is_norm_tensor(name):
    """Return True for 1-D norm/scale tensors (initialised to 1.0)."""
    return ("layernorm" in name) or (name == "model.norm.weight")


def make_safetensors_bytes(rng=None):
    """Return the raw bytes of a BF16 tiny .safetensors file.

    Parameters:
        rng  random.Random instance (default: seeded with _SEED=67)

    Returns:
        bytes
    """
    if rng is None:
        rng = random.Random(_SEED)

    tensors = _tensor_list()

    # Assign contiguous data offsets (BF16 = 2 bytes per element)
    offset = 0
    entries = {}
    payload_parts = []

    for name, shape in tensors:
        n      = _product(shape)
        nbytes = n * 2  # BF16: 2 bytes per element

        if _is_norm_tensor(name):
            values = [1.0] * n
        else:
            values = [rng.gauss(0.0, _WEIGHT_SCALE) for _ in range(n)]

        packed = b"".join(_f32_to_bf16_bytes(v) for v in values)
        assert len(packed) == nbytes

        entries[name] = {
            "dtype":        "BF16",
            "shape":        shape,
            "data_offsets": [offset, offset + nbytes],
        }
        payload_parts.append(packed)
        offset += nbytes

    # Build JSON header with __metadata__ first
    header_obj = {
        "__metadata__": {"format": "pt", "m67": "seeded_random_bf16"},
    }
    header_obj.update(entries)
    header_bytes = json.dumps(header_obj, separators=(",", ":")).encode("utf-8")

    prefix = struct.pack("<Q", len(header_bytes))
    data   = b"".join(payload_parts)
    return prefix + header_bytes + data


def main():
    default_out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "m67_bf16_llama_2l.safetensors",
    )

    parser = argparse.ArgumentParser(
        description="Generate m67_bf16_llama_2l.safetensors fixture (M67)"
    )
    parser.add_argument(
        "--output", default=default_out,
        help="Output path (default: compiler/fixtures/m67_bf16_llama_2l.safetensors)",
    )
    args = parser.parse_args()

    blob = make_safetensors_bytes()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(blob)

    tensors   = _tensor_list()
    data_size = sum(_product(s) * 2 for _, s in tensors)  # BF16: 2 bytes
    print(f"wrote {len(blob)} bytes -> {args.output}")
    print(f"  tensors   : {len(tensors)}")
    print(f"  data bytes: {data_size}")
    print(f"  dtype     : BF16")
    print(f"  seed      : {_SEED}")


if __name__ == "__main__":
    main()
