#ifndef ATT1_CLUSTER_INFER_H
#define ATT1_CLUSTER_INFER_H

#include "att1_fabric.h"
#include "att1_kv_cache.h"
#include "att1_model.h"
#include "att1_shard.h"

#include <stddef.h>
#include <stdint.h>

typedef struct att1_cluster_infer_config {
    size_t tile_count;
    size_t fabric_queue_capacity;
    size_t fabric_max_payload_bytes;
} att1_cluster_infer_config;

typedef struct att1_cluster_tile_counters {
    uint64_t activations_received;
    uint64_t activations_sent;
    uint64_t layers_run;
    uint64_t logits_sent;
    uint32_t layer_start;
    uint32_t layer_end;
} att1_cluster_tile_counters;

typedef struct att1_cluster_infer {
    const att1_model *model;
    att1_shard_plan shard_plan;
    att1_fabric fabric;
    uint32_t host_tile_id;
    att1_kv_cache *layer_kv;
    float *hidden;
    float *next_hidden;
    float *norm;
    float *logits;
    att1_cluster_tile_counters *tile_counters;
    size_t position;
} att1_cluster_infer;

/*
 * Initialize a synchronous multi-tile layer-sharded inference context.
 *
 * Compute tiles are numbered 0..tile_count-1. The host is modeled as fabric
 * endpoint tile_count and receives the final LOGITS packet. Each transformer
 * layer belongs to exactly one compute tile; no tensor-parallel sharding is
 * performed in Milestone 8.
 */
int att1_cluster_infer_init(att1_cluster_infer *infer,
                            const att1_model *model,
                            const att1_cluster_infer_config *config);

void att1_cluster_infer_free(att1_cluster_infer *infer);

int att1_cluster_infer_decode_token(att1_cluster_infer *infer,
                                    uint32_t token_id,
                                    uint32_t *out_token);

int att1_cluster_infer_generate(att1_cluster_infer *infer,
                                const unsigned char *prompt,
                                size_t prompt_bytes,
                                size_t generated_token_count,
                                uint32_t *out_tokens,
                                size_t out_token_capacity,
                                size_t *out_token_count);

int att1_cluster_infer_get_tile_counters(
    const att1_cluster_infer *infer,
    uint32_t tile_id,
    att1_cluster_tile_counters *out_counters);

#endif
