#!/usr/bin/env python3
import os
import struct
import sys

MAGIC = b"ATT1MODL"
VERSION = 1
HEADER_SIZE = 80
CONFIG_SIZE = 36
DESC_SIZE = 128
DTYPE_F32 = 1


def tensor_values(tensor_index, count):
    base = (tensor_index + 1) * 0.01
    return [base + (i * 0.001) for i in range(count)]


def make_tensor(name, shape, tensor_index):
    count = 1
    for dim in shape:
        count *= dim
    data = b"".join(struct.pack("<f", value) for value in tensor_values(tensor_index, count))
    return {
        "name": name,
        "shape": shape,
        "data": data,
    }


def descriptor(tensor, offset):
    name = tensor["name"].encode("ascii")
    if len(name) >= 64:
        raise ValueError("tensor name too long")
    name = name + (b"\x00" * (64 - len(name)))
    shape = list(tensor["shape"]) + [1] * (4 - len(tensor["shape"]))
    return struct.pack(
        "<64sIIQQQQQQII",
        name,
        DTYPE_F32,
        len(tensor["shape"]),
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        offset,
        len(tensor["data"]),
        0,
        0,
    )


def build_model():
    config = {
        "vocab_size": 256,
        "n_layers": 2,
        "n_heads": 2,
        "d_model": 4,
        "d_ff": 8,
        "max_seq_len": 8,
        "rope_dim": 2,
        "n_tiles": 1,
        "shard_count": 0,
    }
    tensors = [make_tensor("tok_embeddings.weight", [256, 4], 0)]
    index = 1
    for layer in range(config["n_layers"]):
        prefix = f"layers.{layer}"
        tensors.extend(
            [
                make_tensor(f"{prefix}.attention_norm.weight", [4], index),
                make_tensor(f"{prefix}.attention.wq.weight", [4, 4], index + 1),
                make_tensor(f"{prefix}.attention.wk.weight", [4, 4], index + 2),
                make_tensor(f"{prefix}.attention.wv.weight", [4, 4], index + 3),
                make_tensor(f"{prefix}.attention.wo.weight", [4, 4], index + 4),
                make_tensor(f"{prefix}.ffn_norm.weight", [4], index + 5),
                make_tensor(f"{prefix}.ffn.w_gate.weight", [4, 8], index + 6),
                make_tensor(f"{prefix}.ffn.w_up.weight", [4, 8], index + 7),
                make_tensor(f"{prefix}.ffn.w_down.weight", [8, 4], index + 8),
            ]
        )
        index += 9
    tensors.extend(
        [
            make_tensor("output_norm.weight", [4], index),
            make_tensor("output.weight", [4, 256], index + 1),
        ]
    )

    config_offset = HEADER_SIZE
    desc_offset = config_offset + CONFIG_SIZE
    data_offset = desc_offset + (len(tensors) * DESC_SIZE)
    data_blob = bytearray()
    desc_blob = bytearray()
    offset = 0
    for i, tensor in enumerate(tensors):
        del i
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
        0,
        0,
    )
    config_blob = struct.pack(
        "<IIIIIIIII",
        config["vocab_size"],
        config["n_layers"],
        config["n_heads"],
        config["d_model"],
        config["d_ff"],
        config["max_seq_len"],
        config["rope_dim"],
        config["n_tiles"],
        config["shard_count"],
    )
    return header + config_blob + bytes(desc_blob) + bytes(data_blob)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "models/dummy/model.att1"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(build_model())


if __name__ == "__main__":
    main()
