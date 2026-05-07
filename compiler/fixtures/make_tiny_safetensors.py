#!/usr/bin/env python3
"""
Generator for compiler/fixtures/tiny_llama_2l.safetensors (M46 fixture).

Produces a minimal but valid .safetensors file with realistic LLaMA-style
tensor names and shapes, using zero-valued F32 weights.

Dimensions:
  vocab_size=16, d_model=8, d_ff=16, n_heads=2, n_layers=2

Tensors (21 total, matching HF LLaMA key conventions):
  model.embed_tokens.weight
  model.layers.{L}.input_layernorm.weight       for L in 0..1
  model.layers.{L}.self_attn.q_proj.weight      for L in 0..1
  model.layers.{L}.self_attn.k_proj.weight      for L in 0..1
  model.layers.{L}.self_attn.v_proj.weight      for L in 0..1
  model.layers.{L}.self_attn.o_proj.weight      for L in 0..1
  model.layers.{L}.post_attention_layernorm.weight  for L in 0..1
  model.layers.{L}.mlp.gate_proj.weight         for L in 0..1
  model.layers.{L}.mlp.up_proj.weight           for L in 0..1
  model.layers.{L}.mlp.down_proj.weight         for L in 0..1
  model.norm.weight
  lm_head.weight

Usage:
    python3 compiler/fixtures/make_tiny_safetensors.py
    python3 compiler/fixtures/make_tiny_safetensors.py \
        --output compiler/fixtures/tiny_llama_2l.safetensors
"""

import argparse
import json
import os
import struct

_VOCAB   = 16
_DMODEL  = 8
_DFF     = 16
_NLAYERS = 2

# Tensor name → shape, using PyTorch/HF layout conventions:
#   projection weights stored [out_features, in_features]
#   gate_proj / up_proj: [d_ff, d_model]
#   down_proj:           [d_model, d_ff]
#   lm_head:             [vocab_size, d_model]
def _tensor_list():
    tensors = [
        ("model.embed_tokens.weight", [_VOCAB, _DMODEL]),
    ]
    for layer in range(_NLAYERS):
        tensors += [
            (f"model.layers.{layer}.input_layernorm.weight",
             [_DMODEL]),
            (f"model.layers.{layer}.self_attn.q_proj.weight",
             [_DMODEL, _DMODEL]),
            (f"model.layers.{layer}.self_attn.k_proj.weight",
             [_DMODEL, _DMODEL]),
            (f"model.layers.{layer}.self_attn.v_proj.weight",
             [_DMODEL, _DMODEL]),
            (f"model.layers.{layer}.self_attn.o_proj.weight",
             [_DMODEL, _DMODEL]),
            (f"model.layers.{layer}.post_attention_layernorm.weight",
             [_DMODEL]),
            (f"model.layers.{layer}.mlp.gate_proj.weight",
             [_DFF, _DMODEL]),
            (f"model.layers.{layer}.mlp.up_proj.weight",
             [_DFF, _DMODEL]),
            (f"model.layers.{layer}.mlp.down_proj.weight",
             [_DMODEL, _DFF]),
        ]
    tensors += [
        ("model.norm.weight",  [_DMODEL]),
        ("lm_head.weight",     [_VOCAB, _DMODEL]),
    ]
    return tensors


def _product(shape):
    r = 1
    for d in shape:
        r *= d
    return r


def make_safetensors_bytes():
    """Return the raw bytes of a valid tiny .safetensors file."""
    tensors = _tensor_list()

    # Assign contiguous data offsets (F32 = 4 bytes per element)
    offset = 0
    entries = {}
    for name, shape in tensors:
        nbytes = _product(shape) * 4
        entries[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    total_data = offset

    # Build JSON header with __metadata__ first
    header_obj = {"__metadata__": {"format": "pt"}}
    header_obj.update(entries)
    header_bytes = json.dumps(header_obj, separators=(",", ":")).encode("utf-8")

    # Assemble: 8-byte LE length prefix + JSON + zero data
    prefix = struct.pack("<Q", len(header_bytes))
    data   = b"\x00" * total_data
    return prefix + header_bytes + data


def main():
    parser = argparse.ArgumentParser(
        description="Generate tiny_llama_2l.safetensors fixture (M46)"
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(__file__), "tiny_llama_2l.safetensors"
        ),
        help="Output path (default: compiler/fixtures/tiny_llama_2l.safetensors)",
    )
    args = parser.parse_args()

    blob = make_safetensors_bytes()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(blob)

    tensors = _tensor_list()
    data_bytes = sum(_product(s) * 4 for _, s in tensors)
    print(f"wrote {len(blob)} bytes → {args.output}")
    print(f"  tensors   : {len(tensors)}")
    print(f"  data bytes: {data_bytes}")
    print(f"  header len: {len(blob) - 8 - data_bytes}")


if __name__ == "__main__":
    main()
