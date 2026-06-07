#ifndef ATT1_KV_MMU_H
#define ATT1_KV_MMU_H

#include "att1_status.h"

#include <stddef.h>
#include <stdint.h>

typedef struct att1_kv_mmu_config {
    size_t max_sessions;
    size_t max_pages;
    size_t num_layers;
    size_t num_heads;
    size_t head_dim;
    size_t page_tokens;
    size_t max_positions;
} att1_kv_mmu_config;

typedef struct att1_kv_mmu_counters {
    uint64_t page_hits;
    uint64_t page_misses;
    uint64_t page_allocations;
    uint64_t append_ops;
    uint64_t read_ops;
    uint64_t range_copy_ops;
    uint64_t errors;
} att1_kv_mmu_counters;

typedef struct att1_kv_mmu_page_ref {
    uint64_t physical_page;
    uint64_t session_id;
    size_t layer_id;
    size_t logical_page;
    size_t page_tokens;
} att1_kv_mmu_page_ref;

typedef struct att1_kv_mmu att1_kv_mmu;

/*
 * Create a fixed-capacity paged KV-MMU simulator.
 *
 * Pages are allocated on demand and store float32 K/V data as
 * [token_in_page][head][head_dim]. This is separate from the Milestone 2 local
 * KV cache and intentionally does not model the fabric or tile runtime.
 *
 * The returned handle is opaque and must be released with
 * att1_kv_mmu_destroy. The handle is not thread-safe.
 */
att1_status_t att1_kv_mmu_create(const att1_kv_mmu_config *config,
                                 att1_kv_mmu **out_mmu);

/*
 * Release all sessions, pages, and page storage.
 *
 * Passing NULL is allowed.
 */
void att1_kv_mmu_destroy(att1_kv_mmu *mmu);

/*
 * Create or destroy a hardware-visible KV address space for a session ID.
 *
 * Destroying a session releases all pages owned by that session.
 */
att1_status_t att1_kv_mmu_create_session(att1_kv_mmu *mmu,
                                         uint64_t session_id);
att1_status_t att1_kv_mmu_destroy_session(att1_kv_mmu *mmu,
                                          uint64_t session_id);

/*
 * Append one token position for one session and layer.
 *
 * key and value contain num_heads * head_dim float32 values. position selects
 * the logical token position and determines the fixed-size logical page.
 * Positions must append sequentially per session/layer starting at 0. Gaps and
 * duplicate appends are rejected; no overwrite occurs.
 */
att1_status_t att1_kv_mmu_append(att1_kv_mmu *mmu,
                                 uint64_t session_id,
                                 size_t layer_id,
                                 size_t position,
                                 const float *key,
                                 const float *value);

/*
 * Read one session/layer/head/position vector into caller-owned buffers.
 *
 * out_key and out_value must each have head_dim elements. Either pointer may
 * be NULL to skip copying that stream.
 */
att1_status_t att1_kv_mmu_read(att1_kv_mmu *mmu,
                               uint64_t session_id,
                               size_t layer_id,
                               size_t head_id,
                               size_t position,
                               float *out_key,
                               float *out_value);

/*
 * Copy a contiguous token range for one head in token order.
 *
 * out_keys and out_values must each have position_count * head_dim elements.
 * Either pointer may be NULL to skip that stream.
 */
att1_status_t att1_kv_mmu_copy_range(att1_kv_mmu *mmu,
                                     uint64_t session_id,
                                     size_t layer_id,
                                     size_t head_id,
                                     size_t start_position,
                                     size_t position_count,
                                     float *out_keys,
                                     float *out_values);

/*
 * Look up the page containing one logical token position.
 *
 * This records a hit or miss in the trace counters. It does not allocate.
 */
att1_status_t att1_kv_mmu_lookup_page(att1_kv_mmu *mmu,
                                      uint64_t session_id,
                                      size_t layer_id,
                                      size_t position,
                                      att1_kv_mmu_page_ref *out_page);

void att1_kv_mmu_get_counters(const att1_kv_mmu *mmu,
                              att1_kv_mmu_counters *out_counters);
void att1_kv_mmu_reset_counters(att1_kv_mmu *mmu);

#endif
