# ATT-1 Fabric Simulator

Milestone 4 adds a packet fabric abstraction for simulated tensor tiles. The
fabric is a deterministic single-threaded simulator, not a tile runtime.

## Queues

Each tile has one fixed-capacity inbound queue. Sending a packet copies the
packet header and payload into the destination queue. The caller retains
ownership of the original payload buffer.

If a destination queue is full, send returns `ATT1_ERR_QUEUE_FULL` and the
packet is not enqueued. Receive is nonblocking and returns
`ATT1_ERR_QUEUE_EMPTY` when no packet is available.

## Payloads

Phase 1 uses a fixed maximum payload size configured at fabric creation. Receive
copies payload bytes into caller-owned storage and removes the queue entry only
after the caller's output buffer is validated.

## Broadcast

Broadcast excludes the sender. Passing `NULL` as the group broadcasts to all
tiles except the source tile. Passing a group broadcasts to listed tiles except
the source tile. Broadcast is preflighted: if any destination is invalid or
full, no destination receives the packet.

## Barrier

The barrier API models one single-generation participant set. The first arrival
defines the participants; the barrier completes only when every participant has
arrived, then resets for reuse.

## Counters

The simulator records packet sends, receives, broadcast packets, payload bytes,
queue-full errors, invalid packets, empty receives, barrier arrivals, and
barrier completions.
