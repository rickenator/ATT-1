#!/usr/bin/env python3
"""
Generator for compiler/fixtures/m63_llama_2l.safetensors (M63 fixture).

Produces a .safetensors file with seeded pseudo-random F32 weights for the
Milestone 63 larger-tiny-model validation.  Uses larger dimensions than the
M61 fixture:
    vocab_size=64, d_model=32, d_ff=64, n_layers=2, n_heads=4, max_seq_len=32

The smaller weight scale (0.02 vs 0.1 in M61) prevents logit saturation in
the wider model where matrix products accumulate more terms.

RNG: Python standard library random.gauss(0, WEIGHT_SCALE) with seed=63.
1-D norm/scale tensors: initialised to 1.0.
2-D projection/embedding tensors: seeded Gaussian N(0, WEIGHT_SCALE=0.02).

Usage:
    python3 compiler/fixtures/make_m63_fixture.py
    python3 compiler/fixtures/make_m63_fixture.py \\
        --output compiler/fixtures/m63_llama_2l.safetensors
"""

import argparse
import json
import os
import random
import struct

_VOCAB        = 64
_DMODEL       = 32
_DFF          = 64
_NLAYERS      = 2
_WEIGHT_SCALE = 0.02
_SEED         = 63

_DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "m63_llama_2l.safetensors",
)


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
    """Return the raw bytes of a seeded-random .safetensors file.

    Parameters:
        rng  random.Random instance (default: seeded with _SEED=63)

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
    header_obj = {"__metadata__": {"format": "pt", "m63": "seeded_random"}}
    for name, shape in tensors:
        header_obj[name] = entries[name]

    header_bytes = json.dumps(header_obj, separators=(",", ":")).encode("utf-8")

    # Pad header to 8-byte boundary (safetensors spec)
    pad = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * pad

    header_len = len(header_bytes)
    result = struct.pack("<Q", header_len) + header_bytes
    for part in payload_parts:
        result += part

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate M63 larger-tiny safetensors fixture"
    )
    parser.add_argument(
        "--output", default=_DEFAULT_OUT,
        help=f"Output path (default: {_DEFAULT_OUT})"
    )
    parser.add_argument(
        "--seed", type=int, default=_SEED,
        help=f"RNG seed (default: {_SEED})"
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data = make_safetensors_bytes(rng)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as fh:
        fh.write(data)

    tensors = _tensor_list()
    print(f"written: {args.output}")
    print(f"  size:    {len(data)} bytes")
    print(f"  tensors: {len(tensors)}")
    print(f"  config:  vocab={_VOCAB} d_model={_DMODEL} d_ff={_DFF}"
          f" n_layers={_NLAYERS} seed={args.seed}")


if __name__ == "__main__":
    main()
