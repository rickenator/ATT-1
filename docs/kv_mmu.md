# ATT-1 Paged KV-MMU

Milestone 3 adds a simulator for a future hardware-shaped KV-cache MMU. It is
separate from the Milestone 2 local contiguous KV cache.

## Lifecycle

As of M151, `att1_kv_mmu` is an opaque heap-owned handle:

```c
att1_kv_mmu *mmu = NULL;
att1_status_t st = att1_kv_mmu_create(&config, &mmu);
/* ... use mmu ... */
att1_kv_mmu_destroy(mmu);
```

The public header exposes configuration, counter, and page-reference value
types, but not the live page/session storage layout.

## Address Shape

KV entries are addressed by:

- `session_id`
- `layer_id`
- `head_id`
- token `position`

The MMU maps token positions to fixed-size logical pages:

```text
logical_page = position / page_tokens
token_slot   = position % page_tokens
```

Positions are valid in the range `0..max_positions - 1`.
Creation rejects configurations where `max_positions` exceeds
`max_pages * page_tokens`.

Each physical page stores float32 keys and values as:

```text
[token_slot][head_id][head_dim]
```

## Page Behavior

Pages are allocated on append when a session/layer/logical-page mapping is
missing. Positions append sequentially per `session_id` and `layer_id`,
starting at position 0. Appending position 5 before positions 0..4 is rejected.

Appending an already populated `session_id` / `layer_id` / token `position`
fails. The API appends a full token worth of heads, so duplicate appends do not
overwrite individual heads.

Reads and range copies require all requested token positions to be present.
Missing pages, missing token slots, invalid sessions, invalid layers, invalid
heads, positions beyond `max_positions`, and invalid ranges fail cleanly.

## Trace Counters

The simulator records page hits, page misses, page allocations, append
operations, read operations, range copy operations, and errors. These counters
are intended to support future tile memory experiments without implementing the
fabric or runtime yet.
