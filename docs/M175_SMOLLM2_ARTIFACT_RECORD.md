# M175 SmolLM2 Artifact Record

**Status:** f32 and q8 `.att1` artifacts emitted locally.

This record captures the first M175 green-packet artifact delivery using the
local SmolLM2-135M source snapshot. It records local paths and validation
results only; public model weights and generated public-model `.att1` files
remain outside the ATT-1 repository.

## Source Snapshot

| Field | Value |
|---|---|
| Source model | `HuggingFaceTB/SmolLM2-135M` |
| Source link | <https://huggingface.co/HuggingFaceTB/SmolLM2-135M> |
| Local model directory | `~/Models/SmolLM2-135M` |
| Snapshot commit | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` |
| LFS object | `80521b4028 * model.safetensors` |
| Source tensor file | `model.safetensors` |

## Source Scan

`compiler/scan_safetensors.py` was run against the local safetensors file with
the local `config.json`.

Result:

```text
scan: ok
llama_check: ok
llama_missing: 0
```

The source weights are present, readable, and complete from ATT-1's current
LLaMA tensor-shape scanner's point of view.

## Compatibility Scan

`compiler/check_llama_compat.py --model-dir ~/Models/SmolLM2-135M` was run
before generating f32/q8 `.att1` files.

Key facts:

| Field | Value |
|---|---|
| `model_type` | `llama` |
| `vocab_size` | `49152` |
| `n_layers` | `30` |
| `n_heads` | `9` |
| `n_kv_heads` | `3` |
| `d_model` | `576` |
| `d_ff` | `1536` |
| `max_seq_len` | `8192` |
| Source dtype | `BF16` |
| Estimated params | `176132160` |
| Estimated f32 artifact | `671.9 MB` |
| Estimated q8 artifact | `330.2 MB` |

Initial result:

```text
gqa_detected              yes
warning: GQA: num_key_value_heads=3, num_attention_heads=9 (ratio 3:1); converter expands K/V projection heads to legacy ATT-1 MHA-shaped tensors
compat: pass
```

## GQA Import Decision

SmolLM2-135M uses grouped-query attention: 9 query heads and 3 K/V heads. The
current v1/v2 `.att1` model configuration stores `n_heads`, but does not store
`n_kv_heads`, and the runtime attention path stores one K/V vector per query
head.

For this M175 artifact delivery, the converter expands each grouped K/V
projection head into the legacy full-head layout. This preserves the existing
`.att1` binary format and runtime while trading additional K/V weight storage
for a directly loadable artifact:

- source K/V shape: `[192,576]`
- emitted K/V shape: `[576,576]`
- expansion: each source K/V head is repeated across its three query heads

Native `n_kv_heads` in the `.att1` format remains a future efficiency
improvement, not a blocker for this local M175 artifact.

## Local Artifacts

| Artifact | Path | Size | SHA-256 |
|---|---|---:|---|
| f32 reference | `~/Models/att1/SmolLM2-135M/model_f32.att1` | 672 MB | `a0311864fbc541c0f45d27820ad617fcfab17ba3700fa3134863c20998885532` |
| q8 primary | `~/Models/att1/SmolLM2-135M/model_q8.att1` | 250 MB | `6dc72ed5d72805446580ea667e30b09c5d10b06811bfedeb14f5d024fe175e84` |
| f32 report | `~/Models/att1/SmolLM2-135M/model_f32_report.json` | local | n/a |
| q8 report | `~/Models/att1/SmolLM2-135M/model_q8_report.json` | local | n/a |
| f32 source compare | `~/Models/att1/SmolLM2-135M/source_compare_f32_report.json` | local | n/a |

Both artifacts were emitted with `--tiles 2 --shard-meta`.

## Validation

`att1-inspect` loads both artifacts successfully:

```text
vocab_size=49152
n_layers=30
n_heads=9
d_model=576
d_ff=1536
max_seq_len=8192
rope_dim=64
n_tiles=2
tensor_count=273
shard_meta_plan_matching=30
shard_meta_plan_mismatch=0
```

The f32 static source comparison checks all 273 mapped tensors and reports exact
agreement after GQA expansion:

```text
f32_tensors_checked: 273
f32_max_abs_error:   0.000e+00
f32_max_rel_error:   0.000e+00
result:              pass
```

The optional Python forward path in `compare_att1_to_source.py` is not yet
updated for the expanded legacy-GQA representation and reports a reshape error
after the static comparison succeeds. That is not an artifact-loader failure.

## Remaining M175 Work

The `.att1` f32/q8 artifacts now exist. The M175 green packet still requires a
long-context token file, placement report, route report, host-access decision,
minimum FPGA scope note, and trace packet before
`compiler/run_m175_green_packet.py` can honestly produce a passing manifest.
