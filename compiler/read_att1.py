#!/usr/bin/env python3
"""
ATT-1 binary model reader for Python tooling (Milestone 61).

Reads an .att1 v1 or v2 model file using the Python standard library only
(struct module, no numpy, no external dependencies).

Public API
----------
read_att1_model(path) -> Att1Model
    Parse the binary file and return an Att1Model with config and tensors.

Att1Model.find_tensor(name) -> Att1Tensor | None
    Look up a tensor by ATT-1 name.

Att1Tensor.f32_values -> tuple[float]
    Decoded float32 values (flat, row-major).  Only for DTYPE_F32 tensors.

Att1Tensor.q8_int_values -> tuple[int]
    Row-major int8 quantised values (as signed Python ints).
    Only for DTYPE_Q8 tensors.

Att1Tensor.q8_scales -> tuple[float]
    Per-row float32 scales.  len == shape[0] for DTYPE_Q8 tensors.

Att1Tensor.dequantize() -> tuple[float]
    Reconstruct float32 values from q8 int8 values and per-row scales.

Exit codes (when used as a CLI):
    0  success
    1  read error
"""

import os
import struct
import sys
from collections import namedtuple

# ---------------------------------------------------------------------------
# ATT-1 binary format constants  (must match include/att1_model.h)
# ---------------------------------------------------------------------------

_MAGIC            = b"ATT1MODL"
_MAGIC_SIZE       = 8
_HEADER_SIZE_V1   = 80
_HEADER_SIZE_V2   = 96
_CONFIG_SIZE      = 36
_TENSOR_DESC_SIZE = 128
_NAME_SIZE        = 64
_MAX_DIMS         = 4

DTYPE_F32 = 1
DTYPE_Q8  = 2


class Att1ReadError(Exception):
    """Fatal error while reading an .att1 file."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _u32(data, offset):
    return struct.unpack_from("<I", data, offset)[0]


def _u64(data, offset):
    return struct.unpack_from("<Q", data, offset)[0]


def _product(shape):
    r = 1
    for d in shape:
        r *= d
    return r


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

class Att1Config:
    """Parsed ATT-1 model configuration block."""
    __slots__ = ("vocab_size", "n_layers", "n_heads", "d_model",
                 "d_ff", "max_seq_len", "rope_dim", "n_tiles", "shard_count")

    def __repr__(self):
        return (
            f"Att1Config(vocab_size={self.vocab_size}, n_layers={self.n_layers}, "
            f"n_heads={self.n_heads}, d_model={self.d_model}, d_ff={self.d_ff}, "
            f"max_seq_len={self.max_seq_len}, rope_dim={self.rope_dim})"
        )


class Att1Tensor:
    """Parsed ATT-1 tensor descriptor with decoded payload."""
    __slots__ = (
        "name", "dtype", "ndims", "shape",
        "_raw",       # bytes: raw payload blob
        "_data_base", # int: byte offset into file data where tensor data starts
    )

    def __init__(self, name, dtype, ndims, shape, raw_payload):
        self.name  = name
        self.dtype = dtype
        self.ndims = ndims
        self.shape = shape    # list[int]
        self._raw  = raw_payload

    # -- F32 -------------------------------------------------------------------

    @property
    def f32_values(self):
        """Flat row-major float32 values.  Only valid for DTYPE_F32."""
        if self.dtype != DTYPE_F32:
            raise Att1ReadError(
                f"f32_values called on non-F32 tensor {self.name!r}"
            )
        n = _product(self.shape)
        return struct.unpack(f"<{n}f", self._raw[:n * 4])

    # -- Q8 -------------------------------------------------------------------

    @property
    def q8_rows(self):
        """Number of rows (shape[0]) for DTYPE_Q8 tensors."""
        if self.dtype != DTYPE_Q8:
            raise Att1ReadError(f"q8_rows called on non-Q8 tensor {self.name!r}")
        return self.shape[0]

    @property
    def q8_cols(self):
        """Number of columns (shape[1]) for DTYPE_Q8 tensors."""
        if self.dtype != DTYPE_Q8:
            raise Att1ReadError(f"q8_cols called on non-Q8 tensor {self.name!r}")
        return self.shape[1]

    @property
    def q8_int_values(self):
        """Flat signed int8 quantised values.  Only valid for DTYPE_Q8."""
        if self.dtype != DTYPE_Q8:
            raise Att1ReadError(
                f"q8_int_values called on non-Q8 tensor {self.name!r}"
            )
        rows = self.shape[0]
        cols = self.shape[1]
        n    = rows * cols
        return struct.unpack(f"<{n}b", self._raw[:n])

    @property
    def q8_scales(self):
        """Per-row float32 scales.  Only valid for DTYPE_Q8."""
        if self.dtype != DTYPE_Q8:
            raise Att1ReadError(
                f"q8_scales called on non-Q8 tensor {self.name!r}"
            )
        rows   = self.shape[0]
        cols   = self.shape[1]
        scale_off = rows * cols
        return struct.unpack(f"<{rows}f", self._raw[scale_off: scale_off + rows * 4])

    def dequantize(self):
        """Reconstruct flat float32 values from q8 int8 + per-row scales."""
        if self.dtype != DTYPE_Q8:
            raise Att1ReadError(
                f"dequantize called on non-Q8 tensor {self.name!r}"
            )
        rows    = self.shape[0]
        cols    = self.shape[1]
        ints    = self.q8_int_values
        scales  = self.q8_scales
        result  = []
        for row in range(rows):
            s = scales[row]
            for col in range(cols):
                result.append(float(ints[row * cols + col]) * s)
        return tuple(result)

    def __repr__(self):
        shape_s = "×".join(str(d) for d in self.shape)
        dtype_s = "f32" if self.dtype == DTYPE_F32 else "q8"
        return f"Att1Tensor({self.name!r}, dtype={dtype_s}, shape=[{shape_s}])"


class Att1Model:
    """Parsed ATT-1 binary model."""
    __slots__ = ("path", "version", "config", "tensors", "_by_name")

    def __init__(self, path, version, config, tensors):
        self.path     = path
        self.version  = version
        self.config   = config
        self.tensors  = tensors          # list[Att1Tensor]
        self._by_name = {t.name: t for t in tensors}

    def find_tensor(self, name):
        """Return Att1Tensor or None."""
        return self._by_name.get(name)

    def __repr__(self):
        return (
            f"Att1Model({os.path.basename(self.path)!r}, "
            f"v{self.version}, tensors={len(self.tensors)}, "
            f"{self.config})"
        )


# ---------------------------------------------------------------------------
# Public reader
# ---------------------------------------------------------------------------

def read_att1_model(path):
    """Parse an ATT-1 binary model file.

    Parameters:
        path  str — path to .att1 file

    Returns:
        Att1Model

    Raises:
        Att1ReadError  on any format or I/O error
    """
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        raise Att1ReadError(f"cannot read {path!r}: {exc}") from exc

    file_size = len(data)

    # ---- magic ----
    if file_size < _HEADER_SIZE_V1:
        raise Att1ReadError("file too small for header")
    if data[:_MAGIC_SIZE] != _MAGIC:
        raise Att1ReadError("bad magic (not an ATT-1 file)")

    version     = _u32(data, 8)
    header_size = _u32(data, 12)

    if version == 1:
        if header_size != _HEADER_SIZE_V1:
            raise Att1ReadError(f"v1 header_size must be {_HEADER_SIZE_V1}, got {header_size}")
    elif version == 2:
        if header_size != _HEADER_SIZE_V2:
            raise Att1ReadError(f"v2 header_size must be {_HEADER_SIZE_V2}, got {header_size}")
        if file_size < _HEADER_SIZE_V2:
            raise Att1ReadError("file too small for v2 header")
    else:
        raise Att1ReadError(f"unsupported ATT-1 version {version}")

    config_offset  = _u64(data, 16)
    config_size    = _u64(data, 24)
    desc_offset    = _u64(data, 32)
    tensor_count   = _u64(data, 40)
    data_offset    = _u64(data, 48)
    data_size      = _u64(data, 56)

    # ---- config ----
    if config_size != _CONFIG_SIZE:
        raise Att1ReadError(f"config_size must be {_CONFIG_SIZE}, got {config_size}")
    if config_offset + config_size > file_size:
        raise Att1ReadError("config section out of range")

    cfg = Att1Config()
    p   = config_offset
    cfg.vocab_size  = _u32(data, p)
    cfg.n_layers    = _u32(data, p + 4)
    cfg.n_heads     = _u32(data, p + 8)
    cfg.d_model     = _u32(data, p + 12)
    cfg.d_ff        = _u32(data, p + 16)
    cfg.max_seq_len = _u32(data, p + 20)
    cfg.rope_dim    = _u32(data, p + 24)
    cfg.n_tiles     = _u32(data, p + 28)
    cfg.shard_count = _u32(data, p + 32)

    # ---- tensor descriptors ----
    desc_bytes = tensor_count * _TENSOR_DESC_SIZE
    if desc_offset + desc_bytes > file_size:
        raise Att1ReadError("tensor descriptor section out of range")
    if data_offset + data_size > file_size:
        raise Att1ReadError("tensor data section out of range")

    tensors = []
    for i in range(tensor_count):
        dp    = desc_offset + i * _TENSOR_DESC_SIZE
        name  = data[dp: dp + _NAME_SIZE].rstrip(b"\x00").decode("ascii", errors="replace")
        dtype = _u32(data, dp + 64)
        ndims = _u32(data, dp + 68)
        if ndims == 0 or ndims > _MAX_DIMS:
            raise Att1ReadError(f"tensor {i} has invalid ndims {ndims}")
        shape = [
            _u64(data, dp + 72 + j * 8)
            for j in range(ndims)
        ]
        t_offset = _u64(data, dp + 104)
        t_nbytes = _u64(data, dp + 112)

        if t_offset + t_nbytes > data_size:
            raise Att1ReadError(
                f"tensor {name!r} data out of range "
                f"(offset={t_offset}, nbytes={t_nbytes}, data_size={data_size})"
            )

        raw = data[data_offset + t_offset: data_offset + t_offset + t_nbytes]
        tensors.append(Att1Tensor(name, dtype, ndims, shape, raw))

    return Att1Model(path, version, cfg, tensors)


# ---------------------------------------------------------------------------
# CLI (diagnostic)
# ---------------------------------------------------------------------------

def _cli_main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ATT-1 binary model reader (M61 diagnostic)"
    )
    parser.add_argument("path", help="Path to .att1 model file")
    parser.add_argument("--tensor", metavar="NAME",
                        help="Show decoded values for one tensor")
    args = parser.parse_args()

    try:
        model = read_att1_model(args.path)
    except Att1ReadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(model)
    print(f"  tensors ({len(model.tensors)}):")
    for t in model.tensors:
        print(f"    {t}")

    if args.tensor:
        t = model.find_tensor(args.tensor)
        if t is None:
            print(f"tensor not found: {args.tensor!r}", file=sys.stderr)
            sys.exit(1)
        if t.dtype == DTYPE_F32:
            vals = t.f32_values
        elif t.dtype == DTYPE_Q8:
            vals = t.dequantize()
        else:
            print(f"unsupported dtype {t.dtype}", file=sys.stderr)
            sys.exit(1)
        n = min(8, len(vals))
        print(f"\n  {t.name}: first {n} values = {[f'{v:.5f}' for v in vals[:n]]}")


if __name__ == "__main__":
    _cli_main()
