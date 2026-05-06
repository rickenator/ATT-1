# ATT-1 Cluster Inference

Milestone 8 adds a synchronous layer-sharded decode path across simulated
tensor tiles. It uses the existing float32 transformer block, local KV cache,
model loader, greedy sampler, byte tokenizer, and fabric simulator.

## Sharding Policy

Each transformer layer belongs to exactly one tile. Tensors are not split
across tiles. When the model has no shard metadata, layers are assigned as
contiguous ranges, balanced by tile id. More tiles than layers is allowed;
extra tiles receive and forward activations without running layers.

Models with nonzero `shard_count` are rejected for now because the Milestone 6
format reserves shard metadata but does not define a parsed metadata table yet.

## Fabric Path

The cluster fabric contains one endpoint per compute tile plus one host
endpoint. For `N` compute tiles:

```text
compute tiles: 0..N-1
host endpoint: N
```

The host sends the embedded prompt activation to tile 0 as an `ACTIVATION`
packet. Each tile receives one activation, runs its assigned layer range, and
sends the activation to the next tile. The last tile applies the output norm,
projects logits, and sends a `LOGITS` packet to the host endpoint.

Activation payload size is `d_model * sizeof(float)`. Logit payload size is
`vocab_size * sizeof(float)`. Fabric sends copy header and payload into queue
storage, so callers retain ownership of their buffers.

## Limits

Milestone 8 does not implement tensor-parallel sharding, quantization, PCIe, or
asynchronous tile worker execution for inference. Batch size remains 1 and the
sampler remains greedy argmax.

## Example

```sh
make
./build/run_cluster_llm
```
