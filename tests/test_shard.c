#include "att1_shard.h"

#include <stdio.h>
#include <string.h>

static int check_plan_covers_once(const att1_shard_plan *plan, size_t layers)
{
    unsigned char seen[16];
    size_t tile = 0u;
    size_t layer = 0u;

    if (layers > sizeof(seen)) {
        return -1;
    }

    memset(seen, 0, sizeof(seen));
    for (tile = 0u; tile < plan->tile_count; tile++) {
        const att1_layer_shard *shard = att1_shard_plan_tile(
            plan,
            (uint32_t)tile);

        if ((shard == NULL) ||
            (shard->tile_id != tile) ||
            (shard->layer_start > shard->layer_end) ||
            (shard->layer_end > layers)) {
            return -1;
        }

        for (layer = shard->layer_start; layer < shard->layer_end; layer++) {
            uint32_t owner = 99u;

            if (seen[layer] != 0u) {
                return -1;
            }
            seen[layer] = 1u;

            if ((att1_shard_plan_layer_tile(plan,
                                            (uint32_t)layer,
                                            &owner) != 0) ||
                (owner != tile)) {
                return -1;
            }
        }
    }

    for (layer = 0u; layer < layers; layer++) {
        if (seen[layer] == 0u) {
            return -1;
        }
    }

    return 0;
}

int main(void)
{
    att1_model model;
    att1_shard_plan plan;

    memset(&model, 0, sizeof(model));
    model.config.n_layers = 5u;

    if (att1_shard_plan_build(&plan, &model, 2u) != 0) {
        fputs("failed to build 5-layer/2-tile shard plan\n", stderr);
        return 1;
    }

    if ((check_plan_covers_once(&plan, 5u) != 0) ||
        (plan.tiles[0].layer_start != 0u) ||
        (plan.tiles[0].layer_end != 3u) ||
        (plan.tiles[1].layer_start != 3u) ||
        (plan.tiles[1].layer_end != 5u)) {
        fputs("5-layer/2-tile shard coverage failed\n", stderr);
        att1_shard_plan_free(&plan);
        return 1;
    }
    att1_shard_plan_free(&plan);

    model.config.n_layers = 8u;
    if (att1_shard_plan_build(&plan, &model, 4u) != 0) {
        fputs("failed to build 8-layer/4-tile shard plan\n", stderr);
        return 1;
    }

    if ((check_plan_covers_once(&plan, 8u) != 0) ||
        (plan.tiles[0].layer_start != 0u) ||
        (plan.tiles[0].layer_end != 2u) ||
        (plan.tiles[1].layer_start != 2u) ||
        (plan.tiles[1].layer_end != 4u) ||
        (plan.tiles[2].layer_start != 4u) ||
        (plan.tiles[2].layer_end != 6u) ||
        (plan.tiles[3].layer_start != 6u) ||
        (plan.tiles[3].layer_end != 8u)) {
        fputs("8-layer/4-tile shard layout failed\n", stderr);
        att1_shard_plan_free(&plan);
        return 1;
    }
    att1_shard_plan_free(&plan);

    model.config.n_layers = 2u;
    if (att1_shard_plan_build(&plan, &model, 4u) != 0) {
        fputs("more tiles than layers should be allowed\n", stderr);
        return 1;
    }

    if ((check_plan_covers_once(&plan, 2u) != 0) ||
        (plan.tiles[2].layer_start != plan.tiles[2].layer_end) ||
        (plan.tiles[3].layer_start != plan.tiles[3].layer_end)) {
        fputs("more-tiles-than-layers shard policy failed\n", stderr);
        att1_shard_plan_free(&plan);
        return 1;
    }
    att1_shard_plan_free(&plan);

    if (att1_shard_plan_build(&plan, &model, 0u) != ATT1_ERR_INVALID_ARG) {
        fputs("zero tile shard plan should fail\n", stderr);
        att1_shard_plan_free(&plan);
        return 1;
    }

    model.config.shard_count = 1u;
    if (att1_shard_plan_build(&plan, &model, 2u) != ATT1_ERR_UNSUPPORTED) {
        fputs("reserved shard metadata should fail until parsed\n", stderr);
        att1_shard_plan_free(&plan);
        return 1;
    }

    puts("shard test passed");
    return 0;
}
