#!/usr/bin/env python3
"""
Generator for compiler/fixtures/m61_llama_2l.safetensors (M61 fixture).

Produces a .safetensors file with seeded pseudo-random F32 weights for the
Milestone 61 source-model comparison harness.  Uses the same tiny model
dimensions as the M46 fixture (vocab=16, d=8, d_ff=16, n_layers=2, n_heads=2)
but with non-trivial weights so numerical comparisons are meaningful.

RNG: Python standard library random.gauss(0, WEIGHT_SCALE) with seed=61.
1-D norm/scale tensors: initialised to 1.0.
2-D projection/embedding tensors: seeded Gaussian N(0, WEIGHT_SCALE=0.1).

Usage:
    python3 compiler/fixtures/make_m61_fixture.py
    python3 compiler/fixtures/make_m61_fixture.py \\
        --output compiler/fixtures/m61_llama_2l.safetensors
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
_SEED         = 61


def _product(shape):
    r = 1
    for d in shape:
        r *= d
    return r


def _tensor_list():
    """Return list of (name, shape) pairs in HF LLaMA key order."""
    tensors = [
        ("model.embed_tokens.weight", [_VOCAB, _DMODEL]),
    ]
    for layer in range(_NLAYERS):
        tensors += [
            (f"model.layers.{layer}.input_layernorm.weight",       [_DMODEL]),
            (f"model.layers.{layer}.self_attn.q_proj.weight",      [_DMODEL, _DMODEL]),
            (f"model.layers.{layer}.self_attn.k_proj.weight",      [_DMODEL, _DMODEL]),
            (f"model.layers.{layer}.self_attn.v_proj.weight",      [_DMODEL, _DMODEL]),
            (f"model.layers.{layer}.self_attn.o_proj.weight",      [_DMODEL, _DMODEL]),
            (f"model.layers.{layer}.post_attention_layernorm.weight", [_DMODEL]),
            (f"model.layers.{layer}.mlp.gate_proj.weight",         [_DFF, _DMODEL]),
            (f"model.layers.{layer}.mlp.up_proj.weight",           [_DFF, _DMODEL]),
            (f"model.layers.{layer}.mlp.down_proj.weight",         [_DMODEL, _DFF]),
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
    """Return the raw bytes of a seeded-random tiny .safetensors file.

    Parameters:
        rng  random.Random instance (default: seeded with _SEED=61)

    Returns:
        bytes
    """
    if rng is None:
        rng = random.Random(_SEED)

    tensors = _tensor_list()

    # Assign contiguous data offsets (F32 = 4 bytes per element)
    offset = 0
    entries = {}
    payload_parts = []

    for name, shape in tensors:
        n      = _product(shape)
        nbytes = n * 4

        if _is_norm_tensor(name):
            values = [1.0] * n
        else:
            values = [rng.gauss(0.0, _WEIGHT_SCALE) for _ in range(n)]

        packed = struct.pack(f"<{n}f", *values)
        assert len(packed) == nbytes

        entries[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        payload_parts.append(packed)
        offset += nbytes

    # Build JSON header with __metadata__ first
    header_obj = {"__metadata__": {"format": "pt", "m61": "seeded_random"}}
    header_obj.update(entries)
    header_bytes = json.dumps(header_obj, separators=(",", ":")).encode("utf-8")

    prefix = struct.pack("<Q", len(header_bytes))
    data   = b"".join(payload_parts)
    return prefix + header_bytes + data


def main():
    default_out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "m61_llama_2l.safetensors",
    )

    parser = argparse.ArgumentParser(
        description="Generate m61_llama_2l.safetensors fixture (M61)"
    )
    parser.add_argument(
        "--output", default=default_out,
        help="Output path (default: compiler/fixtures/m61_llama_2l.safetensors)",
    )
    args = parser.parse_args()

    blob = make_safetensors_bytes()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(blob)

    tensors   = _tensor_list()
    data_size = sum(_product(s) * 4 for _, s in tensors)
    print(f"wrote {len(blob)} bytes → {args.output}")
    print(f"  tensors   : {len(tensors)}")
    print(f"  data bytes: {data_size}")
    print(f"  seed      : {_SEED}")


if __name__ == "__main__":
    main()
