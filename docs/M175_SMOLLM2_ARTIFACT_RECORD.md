# M175 SmolLM2 Artifact Record

**Status:** Blocked before `.att1` emission.

This record captures the first M175 green-packet attempt using the local
SmolLM2-135M source snapshot. It is intentionally a local artifact-readiness
record, not a generated model artifact. Public model weights and generated
public-model `.att1` files remain outside the ATT-1 repository.

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

Result:

```text
gqa_detected              yes
required_change: GQA: num_key_value_heads=3, num_attention_heads=9 (ratio 3:1); GQA requires format change (n_kv_heads in att1_model_config), converter change, and runtime attention change (M68)
compat: fail
```

## Artifact Decision

No f32 or q8 `.att1` artifacts were emitted for this model. Forcing conversion
would create an artifact that the current runtime cannot interpret correctly:
the current ATT-1 model configuration stores `n_heads`, but does not store
`n_kv_heads`, and the runtime attention path assumes full multi-head K/V
projection rather than grouped-query attention.

The correct next implementation step is GQA support, not artifact generation.

Required work:

1. Extend the `.att1` model configuration in a versioned way so it can represent
   `n_kv_heads` without breaking older fixtures.
2. Update the LLaMA converter to emit K/V tensor metadata and weights using the
   source model's grouped-query shapes.
3. Update runtime attention and KV-cache handling so query heads map onto the
   smaller K/V head set.
4. Add source-comparison and two-tile validation coverage for a GQA model before
   recording M171/M172 green-packet reports.

Until that work lands, M175 remains HOLD and the green packet cannot honestly
pass for SmolLM2-135M.
