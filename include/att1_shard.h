#ifndef ATT1_SHARD_H
#define ATT1_SHARD_H

#include "att1_model.h"
#include "att1_status.h"

#include <stddef.h>
#include <stdint.h>

typedef struct att1_layer_shard {
    uint32_t tile_id;
    uint32_t layer_start;
    uint32_t layer_end;
} att1_layer_shard;

typedef struct att1_shard_plan {
    size_t tile_count;
    size_t layer_count;
    att1_layer_shard *tiles;
    uint32_t *layer_to_tile;
} att1_shard_plan;

/*
 * Build a contiguous layer-to-tile plan.
 *
 * Milestone 8 supports layer sharding only: every transformer layer is assigned
 * to exactly one tile, and no tensor is split across tiles. If the model has no
 * shard metadata, layers are divided as evenly as possible by tile id.
 */
att1_status_t att1_shard_plan_build(att1_shard_plan *plan,
                                    const att1_model *model,
                                    size_t tile_count);

void att1_shard_plan_free(att1_shard_plan *plan);

const att1_layer_shard *att1_shard_plan_tile(
    const att1_shard_plan *plan,
    uint32_t tile_id);

att1_status_t att1_shard_plan_layer_tile(const att1_shard_plan *plan,
                                         uint32_t layer_id,
                                         uint32_t *out_tile_id);

#endif
