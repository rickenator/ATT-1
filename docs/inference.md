# ATT-1 Single-Tile Tiny Inference

Milestone 7 adds a deterministic single-tile decode path for the ATT-1 dummy
binary model. It is intentionally small and does not use the tile runtime,
fabric, quantization, sharding, tokenizer files, or PCIe concepts.

## Tokenizer

The tokenizer is byte-level:

```text
token_id 0..255 == byte value 0x00..0xff
```

Invalid token ids greater than 255 fail to decode.

## Sampler

The sampler is greedy argmax over float32 logits. Ties resolve to the lowest
token id. NaN logits are rejected.

## Decode Path

For each input token:

1. Load token embedding from `tok_embeddings.weight`.
2. Run every decoder layer with the Milestone 2 local transformer block.
3. Use one simple local KV cache per layer.
4. Apply `output_norm.weight`.
5. Project with `output.weight` to `vocab_size` logits.
6. Greedily select the next token.

Batch size is 1. The logits length is exactly `vocab_size`.

## Prompt Policy

`att1_infer_generate` requires a non-empty prompt. Empty prompts are rejected.

## Example

```sh
make
./build/run_tiny_llm
```
