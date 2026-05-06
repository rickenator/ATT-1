# ATT-1 Binary Model Format

Milestone 6 defines a small little-endian binary format for simulator model
fixtures. It supports tiny LLaMA-style decoder metadata and future shard
metadata offsets, but it is not connected to inference or the tile runtime yet.

## Header

All integer fields are little-endian.

```text
magic[8]                 "ATT1MODL"
version                  uint32, currently 1
header_size              uint32, currently 80
config_offset            uint64
config_size              uint64, currently 36
tensor_desc_offset       uint64
tensor_count             uint64
tensor_data_offset       uint64
tensor_data_size         uint64
shard_metadata_offset    uint64, zero when absent
shard_metadata_size      uint64, zero when absent
```

## Config

The config section contains nine `uint32` values:

```text
vocab_size
n_layers
n_heads
d_model
d_ff
max_seq_len
rope_dim
n_tiles
shard_count
```

## Tensor Descriptor

Each descriptor is 128 bytes:

```text
name[64]
dtype        uint32, 1 = float32
ndims        uint32
shape[4]     uint64
offset       uint64, relative to tensor_data_offset
nbytes       uint64
shard_id     uint32
flags        uint32
```

Only float32 tensors are supported in Milestone 6. The loader rejects unknown
dtypes, `ndims > 4`, zero dimensions, descriptor ranges outside the file, data
ranges outside the file, and tensor byte ranges extending past the tensor data
section.

## Dummy Model

`compiler/make_dummy_model.py` writes a deterministic model. With no arguments
it writes:

```text
models/dummy/model.att1
```

The dummy config is:

```text
vocab_size=256
n_layers=2
n_heads=2
d_model=4
d_ff=8
max_seq_len=8
rope_dim=2
n_tiles=1
shard_count=0
```
