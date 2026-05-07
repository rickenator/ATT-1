#!/usr/bin/env python3
"""
Generate a tiny deterministic .att1 fixture with a valid shard metadata section.

Offline tool for developers.  Run once to regenerate models/shard_meta/model.att1;
the output is checked into the repository so `make test` needs no Python.

Usage:
    python3 compiler/make_shard_meta_fixture.py [OUTPUT]

Default output: models/shard_meta/model.att1

The generated model is identical to models/dummy/model.att1 except that the
header's shard_metadata_offset and shard_metadata_size fields are non-zero and
a valid 21-record shard metadata section is appended after the tensor data.

Tile assignment scheme (single-tile, n_tiles=1):
  - All tensors → tile_id=0, owner_aimu=0
  - replication_policy=0 (none), reduction_behavior=0 (none)
  - dtype=f32, quantization=none
  - dependency_graph: zero (no static dependency encoding yet)
  - allowed_ops: 0 (unconstrained; enforcement deferred to Phase 2)
  - checksum: 0 (no verification; CRC-64 deferred)
  - _reserved: 0

All values are deterministic and match the existing dummy model tensor names,
shapes, and data so that inference tests that load the dummy model continue to
pass if pointed at this fixture.
"""

import os
import struct
import sys

# ---------------------------------------------------------------------------
# Format constants (must match att1_model.h and att1_shard_meta.h)
# ---------------------------------------------------------------------------

MAGIC        = b"ATT1MODL"
VERSION      = 1
HEADER_SIZE  = 80
CONFIG_SIZE  = 36   # 9 × uint32
DESC_SIZE    = 128
SHARD_REC_SZ = 120  # ATT1_SHARD_META_RECORD_SIZE

DTYPE_F32    = 1

# Shard-record field codes
QUANT_NONE   = 0
REPL_NONE    = 0
REDUCE_NONE  = 0

# ---------------------------------------------------------------------------
# Model definition — identical config to models/dummy/model.att1
# ---------------------------------------------------------------------------

_CFG = dict(
    vocab_size   = 256,
    n_layers     = 2,
    n_heads      = 2,
    d_model      = 4,
    d_ff         = 8,
    max_seq_len  = 8,
    rope_dim     = 2,
    n_tiles      = 1,
    shard_count  = 0,
)


def _tensor_list() -> list[tuple[str, list[int]]]:
    """Ordered (name, shape) pairs matching the C runtime's expected tensor names."""
    d  = _CFG["d_model"]
    v  = _CFG["vocab_size"]
    ff = _CFG["d_ff"]
    plan: list[tuple[str, list[int]]] = [("tok_embeddings.weight", [v, d])]
    for layer in range(_CFG["n_layers"]):
        pfx = f"layers.{layer}"
        plan += [
            (f"{pfx}.attention_norm.weight", [d]),
            (f"{pfx}.attention.wq.weight",   [d, d]),
            (f"{pfx}.attention.wk.weight",   [d, d]),
            (f"{pfx}.attention.wv.weight",   [d, d]),
            (f"{pfx}.attention.wo.weight",   [d, d]),
            (f"{pfx}.ffn_norm.weight",       [d]),
            (f"{pfx}.ffn.w_gate.weight",     [d, ff]),
            (f"{pfx}.ffn.w_up.weight",       [d, ff]),
            (f"{pfx}.ffn.w_down.weight",     [ff, d]),
        ]
    plan += [
        ("output_norm.weight", [d]),
        ("output.weight",      [d, v]),
    ]
    return plan


def _padded(shape: list[int]) -> list[int]:
    """Pad shape to exactly 4 elements with 1s (matching the C parser expectation)."""
    return shape + [1] * (4 - len(shape))


def _elem_count(shape: list[int]) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count


def _tensor_data(tensor_index: int, count: int) -> bytes:
    """Deterministic synthetic float32 values matching the dummy model generator."""
    base = (tensor_index + 1) * 0.01
    return b"".join(struct.pack("<f", base + i * 0.001) for i in range(count))


def _descriptor(name: str, shape: list[int], rel_offset: int, nbytes: int) -> bytes:
    """Build one 128-byte tensor descriptor."""
    name_b = name.encode("ascii")
    assert len(name_b) < 64, f"tensor name too long: {name!r}"
    name_b = name_b + b"\x00" * (64 - len(name_b))
    ps = _padded(shape)
    return struct.pack(
        "<64sIIQQQQQQII",
        name_b,
        DTYPE_F32,
        len(shape),          # ndims
        ps[0], ps[1], ps[2], ps[3],
        rel_offset,          # offset within data section
        nbytes,
        0,                   # shard_id
        0,                   # flags
    )


def _shard_record(
    tensor_index: int,
    shape: list[int],
    byte_offset: int,
) -> bytes:
    """
    Build one 120-byte shard metadata record.

    Wire layout (all LE):
      @  0  tensor_id           u32
      @  4  tile_id             u32
      @  8  byte_offset         u64
      @ 16  shape[4]            u64[4]  (32 bytes)
      @ 48  dtype               u32
      @ 52  quantization        u32
      @ 56  owner_aimu          u32
      @ 60  replication_policy  u32
      @ 64  dependency_graph[8] u32[8]  (32 bytes)
      @ 96  allowed_ops         u32
      @100  routing_requirements u32
      @104  reduction_behavior  u32
      @108  _reserved           u32   (must be zero)
      @112  checksum            u64
    """
    ps = _padded(shape)

    # Single-tile model: all tensors on tile 0, owner_aimu 0.
    tile_id    = tensor_index % _CFG["n_tiles"]   # 0 when n_tiles=1
    owner_aimu = tile_id

    return struct.pack(
        "<IIQ4QIIII8IIIIIQ",
        tensor_index,   # tensor_id
        tile_id,
        byte_offset,
        ps[0], ps[1], ps[2], ps[3],   # shape[4]
        DTYPE_F32,
        QUANT_NONE,
        owner_aimu,
        REPL_NONE,
        0, 0, 0, 0, 0, 0, 0, 0,       # dependency_graph[8]
        0,                             # allowed_ops
        0,                             # routing_requirements
        REDUCE_NONE,
        0,                             # _reserved
        0,                             # checksum
    )


def build() -> bytes:
    """Build and return the complete .att1 binary with shard metadata."""
    tensors = _tensor_list()
    n_tensors = len(tensors)

    # --- compute section sizes and offsets ---
    config_offset  = HEADER_SIZE
    desc_offset    = config_offset + CONFIG_SIZE
    data_offset    = desc_offset + n_tensors * DESC_SIZE

    # Compute per-tensor data sizes and cumulative offsets
    nbytes_list: list[int] = []
    rel_offsets: list[int] = []
    cur = 0
    for _name, shape in tensors:
        nb = _elem_count(shape) * 4  # float32
        rel_offsets.append(cur)
        nbytes_list.append(nb)
        cur += nb
    data_size = cur

    shard_offset = data_offset + data_size
    shard_size   = n_tensors * SHARD_REC_SZ

    # --- header (80 bytes) ---
    header = struct.pack(
        "<8sIIQQQQQQQQ",
        MAGIC,
        VERSION,
        HEADER_SIZE,
        config_offset,
        CONFIG_SIZE,
        desc_offset,
        n_tensors,
        data_offset,
        data_size,
        shard_offset,
        shard_size,
    )
    assert len(header) == HEADER_SIZE

    # --- config (36 bytes) ---
    config_blob = struct.pack(
        "<IIIIIIIII",
        _CFG["vocab_size"],
        _CFG["n_layers"],
        _CFG["n_heads"],
        _CFG["d_model"],
        _CFG["d_ff"],
        _CFG["max_seq_len"],
        _CFG["rope_dim"],
        _CFG["n_tiles"],
        _CFG["shard_count"],
    )
    assert len(config_blob) == CONFIG_SIZE

    # --- tensor descriptors and data ---
    desc_blob = bytearray()
    data_blob = bytearray()
    for i, (name, shape) in enumerate(tensors):
        nb = nbytes_list[i]
        desc_blob += _descriptor(name, shape, rel_offsets[i], nb)
        data_blob += _tensor_data(i, _elem_count(shape))

    assert len(desc_blob) == n_tensors * DESC_SIZE
    assert len(data_blob) == data_size

    # --- shard metadata section ---
    shard_blob = bytearray()
    for i, (_, shape) in enumerate(tensors):
        shard_blob += _shard_record(i, shape, rel_offsets[i])

    assert len(shard_blob) == shard_size
    assert len(shard_blob) % SHARD_REC_SZ == 0

    payload = (
        header
        + config_blob
        + bytes(desc_blob)
        + bytes(data_blob)
        + bytes(shard_blob)
    )

    expected = HEADER_SIZE + CONFIG_SIZE + n_tensors * DESC_SIZE + data_size + shard_size
    assert len(payload) == expected, f"{len(payload)} != {expected}"
    return payload


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "models/shard_meta/model.att1"
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    data = build()

    with open(out_path, "wb") as f:
        f.write(data)

    tensors = _tensor_list()
    n = len(tensors)
    shard_size = n * SHARD_REC_SZ
    print(f"wrote {len(data)} bytes → {out_path}")
    print(f"  {n} tensors, shard_meta: {shard_size} bytes ({n} × {SHARD_REC_SZ})")
    for i, (name, shape) in enumerate(tensors):
        shape_str = " × ".join(str(d) for d in shape)
        print(f"  [{i:2d}] {name:<48} tile=0  shape={shape_str}")


if __name__ == "__main__":
    main()
