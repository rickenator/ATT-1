# ATT-1 tracing and benchmarking

Milestone 9 adds optional simulator instrumentation without changing inference
outputs. A caller may attach an `att1_trace_t` to single-tile inference or
cluster inference. Passing no trace object keeps inference on the existing path.

## Trace scope

The trace object records:

- tokens decoded
- per-token CPU time in microseconds
- per-layer CPU time in microseconds
- activation bytes sent between tiles
- logits bytes produced
- fabric packet and payload byte counters
- local KV append/read counters used by the decode path
- tile layer execution counters

Timing uses the portable C `clock()` source, so values are process CPU time, not
hardware wall-clock timing. The counters are intended for deterministic simulator
smoke tests and relative scaling checks, not cycle-accurate performance claims.

## Inference attachment

Single tile:

```c
att1_trace_t *trace = NULL;
att1_infer_t *infer = NULL;

att1_trace_create(model.config.n_layers, 1u, &trace);
att1_infer_create(&model, &infer);
att1_infer_set_trace(infer, trace);
```

Cluster:

```c
att1_trace_t *trace = NULL;
att1_cluster_infer_t *infer = NULL;

att1_trace_create(model.config.n_layers, tile_count, &trace);
att1_cluster_infer_create(&model, &config, &infer);
att1_cluster_infer_set_trace(infer, trace);
```

The trace object is caller-owned. Destroy the inference context before or after
destroying the trace only after no further decode call will use the trace.

## Benchmark tool

`build/att1-bench` runs the loaded dummy model in either single-tile or cluster
mode and prints human-readable `key=value` counters.

```sh
./build/att1-bench --model models/dummy/model.att1 --prompt ATT1 --tokens 4 --mode single
./build/att1-bench --model models/dummy/model.att1 --prompt ATT1 --tokens 4 --mode cluster
```

The benchmark proves trace plumbing and decode behavior. It is not a throughput
claim for the future ASIC.

## Size estimator

`build/att1-size` prints synthetic memory and scaling reports:

```sh
./build/att1-size --preset tiny-dummy
./build/att1-size --preset gpt-oss-120b-shape
```

The `gpt-oss-120b-shape` preset is synthetic/non-executable. It estimates model
bytes, per-tile bytes, KV bytes per session, and activation bytes per token for
architecture planning only.

## Benchmark fields: prefill vs decode split (M95)

`att1-bench` snapshots the trace counters at the boundary between the prefill
phase (feeding prompt tokens) and the decode phase (generating new tokens).
The following fields are appended after the existing counters in every run.

### Phase-split counter fields

| Field | Description |
|-------|-------------|
| `prompt_tokens` | Number of prompt tokens fed (byte count for byte tokenizer; ID count for external) |
| `decode_tokens` | Number of new tokens generated |
| `prefill_time_us_total` | CPU time used during prefill (microseconds) |
| `decode_time_us_total` | CPU time used during decode (microseconds) |
| `prefill_kv_appends` | KV cache appends during prefill |
| `decode_kv_appends` | KV cache appends during decode |
| `prefill_kv_reads` | KV key+value cache reads during prefill |
| `decode_kv_reads` | KV key+value cache reads during decode |
| `prefill_logits_bytes` | Logits bytes produced during prefill |
| `decode_logits_bytes` | Logits bytes produced during decode |
| `prefill_fabric_packets` | Fabric packets sent during prefill (cluster mode only) |
| `decode_fabric_packets` | Fabric packets sent during decode (cluster mode only) |

All existing fields (`tokens_decoded`, `token_time_us_total`, etc.) remain
present for compatibility.

### Notes

- Timing fields are CPU process time from `clock()`, not wall-clock time.
- In single-tile mode, `prefill_fabric_packets` and `decode_fabric_packets` are
  omitted (fabric is unused in single-tile mode).
- If `decode_tokens=0` (requested tokens capped to zero by `max_seq_len`), all
  decode-phase fields will be zero.
- `kv_reads` is the sum of `kv_key_reads + kv_value_reads` for that phase.
