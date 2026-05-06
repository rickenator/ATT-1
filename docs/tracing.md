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
