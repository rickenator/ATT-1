# Future Memory Tiers for ATT-1/AIMU

This note is for later hardware-design work, especially the post-M175 period
when the active question becomes whether AIMU should remain an SRAM/HBM-shaped
accelerator or deliberately exploit cheaper, slower, more sequential memory
tiers.

## Carmack Note: Sequential Flash for Inference

On July 6, 2026, John Carmack posted an argument that AI inference may be a
strong fit for nontraditional memory systems because model-weight access is
much more deterministic than game-rendering memory access. The core idea is
that inference weights do not need true low-latency random access if the
accelerator can tolerate millisecond-scale cold-start behavior and sustain the
needed continuous read bandwidth once the stream is moving.

The linked post: <https://x.com/ID_AA_Carmack/status/2074248758422864226>

Key points from the post, paraphrased:

- NAND flash is dramatically cheaper per GB than HBM, leaving room for a
  specialized wide flash interface and controller before the cost curve looks
  like HBM.
- A very narrow programming model could expose flash as a pipelined transfer
  source for full 16 KiB-or-larger pages into accelerator-managed scratchpad.
- A more compatible model could make the device look like random-access memory,
  but with fragile performance: sequential reads are fast, while nonsequential
  behavior falls off by orders of magnitude.
- The RAM-like facade has a migration advantage because existing cache
  hierarchies and software can run first, albeit slowly, then be incrementally
  tuned for sequential performance.
- If scratchpad cannot hold a full layer, data duplication across the
  sequential medium may be worthwhile, analogous to old optical-drive layout
  tricks used to avoid seeks.
- A trace-capture/remapping approach, similar in spirit to CUDA graph capture,
  might automatically linearize access, but explicit programmer or agent work
  managing scratchpad ring buffers may be lower risk.
- A split memory system with some HBM and some flash is likely less elegant
  than a uniform memory system, but may make much larger inference models
  affordable.
- Training is harder because even linearized writes wear flash quickly; a
  high-latency, massively parallel DRAM tier may be a better training-side
  cost reduction than flash.

## Relevance to ATT-1/AIMU

ATT-1 already leans toward the premise behind this argument. The `.att1`
artifact format, shard metadata, placement reports, command plans, fabric route
reports, and replay tooling all treat inference as a schedulable dataflow
problem rather than an opaque random-memory workload. That makes AIMU a better
candidate for explicit memory-tier orchestration than a conventional GPU
programming model would be.

The current AIMU direction assumes local tile memory plus host-controlled shard
residency. A future flash-backed tier would make that residency model more
important, not less. `LOAD_TENSOR_TILE`, DMA descriptors, endpoint-owned
memory, and the "weights are transferred once, then stay resident" rule become
the control-plane foundation for a larger hierarchy:

1. Hot tile scratchpad/SRAM for active layer fragments, KV page working sets,
   and partial reductions.
2. HBM or commodity DRAM for medium-latency reusable shards and activations.
3. Flash-backed sequential weight storage for cold or streaming model weights.

The main architectural question is whether AIMU should expose flash-like memory
as explicit streams into scratchpad or as a RAM-compatible tier with severe
sequentiality requirements. The explicit stream model better matches command
plans and gives the scheduler clear responsibility. The RAM-compatible model
would be easier to bring up and easier to update in place, but it risks hiding
performance cliffs until too late.

For ATT-1, the lower-risk path is probably to model both:

- Add a planner-level "sequential tier" model first: page size, cold latency,
  sustained bandwidth, queue depth, erase/write policy, and per-tile scratchpad
  ring size.
- Extend placement/scenario reports to mark tensors or tensor slices as
  scratchpad-resident, HBM/DRAM-resident, or sequential-tier streamed.
- Extend execution plans so `LOAD_TENSOR_TILE` can represent scheduled page
  streams, prefetch windows, duplicate sequential layouts, and page reuse.
- Use trace/replay tooling to identify when the normal cluster path violates
  sequential assumptions, before any hardware protocol is chosen.

The most interesting overlap with ATT-1 is the page/ring-buffer problem. If a
layer's weights cannot fit in scratchpad, AIMU can either:

- stream sub-layer tiles in computation order,
- duplicate weight pages in multiple sequential positions to avoid expensive
  jumps,
- or change tensor placement so layers are sliced around page-stream order
  instead of only around arithmetic balance.

This directly affects M172-M174 style work: memory movement, latency stability,
capacity envelopes, and activation precision cannot be final hardware metrics
unless the memory-tier assumption is explicit.

## Deferred Hardware Questions

- What page size should a sequential tier expose: 16 KiB, 64 KiB, 256 KiB, or
  larger?
- Is a RAM-compatible facade useful enough to justify the risk of accidental
  random access?
- Should the command queue gain an explicit stream/prefetch command, or should
  `LOAD_TENSOR_TILE` grow flags that describe sequential-tier transfers?
- How large must tile scratchpad rings be to keep matmul/attention execution
  fed from sequential pages?
- Can placement reports cheaply predict duplicate-page overhead for layers
  that do not fit in scratchpad?
- Should `.att1` artifacts eventually support physical or logical page layout
  hints for sequential media?
- Is flash only an inference-tier candidate, with training left to HBM/DRAM
  tiers because of write endurance?

## Near-Term Use

Do not let this note change current Phase 2 implementation scope. The immediate
M172-M175 path should continue to measure the existing emulated endpoint,
define baseline workloads, validate capacity envelopes, and make the FPGA gate
decision. This document should become active again when hardware memory
architecture is the main design battlefield.
