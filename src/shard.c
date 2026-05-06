#include "att1_shard.h"

#include <stdlib.h>
#include <string.h>

att1_status_t att1_shard_plan_build(att1_shard_plan *plan,
                                    const att1_model *model,
                                    size_t tile_count)
{
    size_t tile = 0u;
    uint32_t layer = 0u;
    uint32_t next_layer = 0u;

    if ((plan == NULL) || (model == NULL) ||
        (tile_count == 0u) ||
        (tile_count > (size_t)UINT32_MAX) ||
        (model->config.n_layers == 0u)) {
        return ATT1_ERR_INVALID_ARG;
    }

    if (model->config.shard_count != 0u) {
        return ATT1_ERR_UNSUPPORTED;
    }

    memset(plan, 0, sizeof(*plan));
    plan->tiles = calloc(tile_count, sizeof(*plan->tiles));
    plan->layer_to_tile = calloc(model->config.n_layers,
                                 sizeof(*plan->layer_to_tile));
    if ((plan->tiles == NULL) || (plan->layer_to_tile == NULL)) {
        att1_shard_plan_free(plan);
        return ATT1_ERR_OOM;
    }

    plan->tile_count = tile_count;
    plan->layer_count = model->config.n_layers;

    for (tile = 0u; tile < tile_count; tile++) {
        const uint32_t remaining_layers = model->config.n_layers - next_layer;
        const size_t remaining_tiles = tile_count - tile;
        const uint32_t count = (uint32_t)((remaining_layers +
                                           remaining_tiles - 1u) /
                                          remaining_tiles);

        plan->tiles[tile].tile_id = (uint32_t)tile;
        plan->tiles[tile].layer_start = next_layer;
        plan->tiles[tile].layer_end = next_layer + count;
        next_layer += count;
    }

    for (tile = 0u; tile < tile_count; tile++) {
        for (layer = plan->tiles[tile].layer_start;
             layer < plan->tiles[tile].layer_end;
             layer++) {
            if (layer >= model->config.n_layers) {
                att1_shard_plan_free(plan);
                return ATT1_ERR_STATE;
            }
            plan->layer_to_tile[layer] = (uint32_t)tile;
        }
    }

    return ATT1_OK;
}

void att1_shard_plan_free(att1_shard_plan *plan)
{
    if (plan == NULL) {
        return;
    }

    free(plan->tiles);
    free(plan->layer_to_tile);
    memset(plan, 0, sizeof(*plan));
}

const att1_layer_shard *att1_shard_plan_tile(
    const att1_shard_plan *plan,
    uint32_t tile_id)
{
    if ((plan == NULL) || (plan->tiles == NULL) ||
        ((size_t)tile_id >= plan->tile_count)) {
        return NULL;
    }

    return &plan->tiles[tile_id];
}

att1_status_t att1_shard_plan_layer_tile(const att1_shard_plan *plan,
                                         uint32_t layer_id,
                                         uint32_t *out_tile_id)
{
    if ((plan == NULL) || (plan->layer_to_tile == NULL) ||
        (out_tile_id == NULL) ||
        ((size_t)layer_id >= plan->layer_count)) {
        return ATT1_ERR_INVALID_ARG;
    }

    if ((size_t)plan->layer_to_tile[layer_id] >= plan->tile_count) {
        return ATT1_ERR_STATE;
    }

    *out_tile_id = plan->layer_to_tile[layer_id];
    return ATT1_OK;
}
