# ATT-1 Tile Runtime

Milestone 5 adds a pthread-backed simulated tile runtime. Each tile owns local
state and a bounded command queue. Worker threads process explicit commands and
use the existing fabric simulator for activation send/receive placeholders.

## Lifecycle Policy

- Commands before `att1_runtime_start` fail with `ATT1_ERR_INVALID`.
- Starting an already-started runtime returns `ATT1_ERR_ALREADY_STARTED`.
- Stopping an already-stopped runtime succeeds as a no-op.
- `att1_runtime_destroy` calls stop before releasing resources.
- A `SHUTDOWN` command causes the addressed worker to exit.

## Commands

- `LOAD_MODEL_SHARD`: records the shard ID in tile-local state.
- `RUN_LAYER_RANGE`: records the requested layer range and increments counters.
- `SEND_ACTIVATION`: sends an activation packet through the fabric.
- `RECV_ACTIVATION`: attempts a nonblocking fabric receive.
- `SYNC_BARRIER`: enters a runtime barrier across all tiles.
- `SHUTDOWN`: exits the worker loop.

Real model loading, transformer execution, and quantization remain out of scope.

## Queues

Each tile command queue is bounded. If a queue is full,
`att1_runtime_send_command` returns `ATT1_ERR_QUEUE_FULL`.
