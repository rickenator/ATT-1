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

/* -------------------------------------------------------------------------
 * att1_shard_plan_from_meta — convert a validated meta plan to shard plan
 * ---------------------------------------------------------------------- */

att1_status_t att1_shard_plan_from_meta(const att1_meta_plan *proposed,
                                        uint32_t              n_layers,
                                        size_t                tile_count,
                                        att1_shard_plan      *out)
{
    uint32_t i;

    if ((proposed == NULL) || (out == NULL) ||
        (tile_count == 0u) || (n_layers == 0u)) {
        return ATT1_ERR_INVALID_ARG;
    }

    /* Every layer must be covered. */
    if (proposed->count != n_layers) {
        return ATT1_ERR_INVALID_ARG;
    }

    /* No conflicting tile assignments within any layer. */
    if (proposed->conflict > 0u) {
        return ATT1_ERR_INVALID_ARG;
    }

    memset(out, 0, sizeof(*out));

    if (n_layers > (uint32_t)(SIZE_MAX / sizeof(*out->layer_to_tile))) {
        return ATT1_ERR_OOM;
    }

    out->layer_to_tile = calloc(n_layers, sizeof(*out->layer_to_tile));
    out->tiles         = calloc(tile_count, sizeof(*out->tiles));
    if ((out->layer_to_tile == NULL) || (out->tiles == NULL)) {
        att1_shard_plan_free(out);
        return ATT1_ERR_OOM;
    }

    out->layer_count = n_layers;
    out->tile_count  = tile_count;

    /* Pre-populate tiles with their tile_id and sentinel layer_start. */
    for (i = 0u; i < (uint32_t)tile_count; i++) {
        out->tiles[i].tile_id     = i;
        out->tiles[i].layer_start = n_layers; /* sentinel: no layers yet */
        out->tiles[i].layer_end   = 0u;
    }

    /* Fill layer_to_tile from proposed entries (already sorted by layer_id). */
    for (i = 0u; i < proposed->count; i++) {
        const uint32_t layer_id = proposed->entries[i].layer_id;
        const uint32_t tile_id  = proposed->entries[i].tile_id;
        att1_layer_shard *shard;

        /* Entries must cover layers 0..n_layers-1 with no gaps. */
        if (layer_id != i) {
            att1_shard_plan_free(out);
            return ATT1_ERR_INVALID_ARG;
        }

        if ((size_t)tile_id >= tile_count) {
            att1_shard_plan_free(out);
            return ATT1_ERR_INVALID_ARG;
        }

        out->layer_to_tile[layer_id] = tile_id;

        shard = &out->tiles[tile_id];
        if (shard->layer_start == n_layers) {
            /* First layer for this tile. */
            shard->layer_start = layer_id;
            shard->layer_end   = layer_id + 1u;
        } else if (shard->layer_end == layer_id) {
            /* Contiguous extension. */
            shard->layer_end = layer_id + 1u;
        } else {
            /* Non-contiguous: this tile already has a range that doesn't
               extend naturally to layer_id. */
            att1_shard_plan_free(out);
            return ATT1_ERR_INVALID_ARG;
        }
    }

    /* Fix up tiles with no assigned layers: layer_start = layer_end = 0. */
    for (i = 0u; i < (uint32_t)tile_count; i++) {
        if (out->tiles[i].layer_start == n_layers) {
            out->tiles[i].layer_start = 0u;
            out->tiles[i].layer_end   = 0u;
        }
    }

    return ATT1_OK;
}

/* -------------------------------------------------------------------------
 * Metadata-driven proposed shard plan (Milestone 39)
 * ---------------------------------------------------------------------- */

/*
 * Extract the layer index N from a tensor name of the form "layers.N.xxx".
 * Returns 1 on success and writes N to *out_layer.
 * Returns 0 when the name does not follow this pattern.
 */
static int parse_layer_id(const char *name, uint32_t *out_layer)
{
    const char *p = name;
    uint32_t    layer = 0u;

    if (strncmp(p, "layers.", 7) != 0) {
        return 0;
    }
    p += 7;

    if ((*p < '0') || (*p > '9')) {
        return 0;
    }

    while ((*p >= '0') && (*p <= '9')) {
        layer = layer * 10u + (uint32_t)((unsigned char)*p - '0');
        p++;
    }

    if (*p != '.') {
        return 0;
    }

    *out_layer = layer;
    return 1;
}

att1_status_t att1_meta_plan_build(const att1_model *model,
                                   att1_meta_plan   *out)
{
    const att1_shard_meta *meta;
    uint32_t               n_layers;
    uint32_t               i;
    uint32_t               extra    = 0u;
    uint32_t               conflict = 0u;
    uint32_t               entry_count = 0u;
    att1_meta_plan_entry  *entries  = NULL;

    /* per-layer tracking */
    struct layer_slot {
        int      seen;
        uint32_t tile_id;
        int      conflict;
    } *slots = NULL;

    if ((model == NULL) || (out == NULL)) {
        return ATT1_ERR_INVALID_ARG;
    }

    memset(out, 0, sizeof(*out));

    meta     = &model->shard_meta;
    n_layers = model->config.n_layers;

    if ((meta->count == 0u) || (n_layers == 0u)) {
        return ATT1_OK;
    }

    slots = calloc(n_layers, sizeof(*slots));
    if (slots == NULL) {
        return ATT1_ERR_OOM;
    }

    /* Pass 1: categorise each record */
    for (i = 0u; i < (uint32_t)meta->count; i++) {
        const att1_shard_meta_record *rec = &meta->records[i];
        const char *name;
        uint32_t    layer_id = 0u;

        if (rec->tensor_id >= (uint32_t)model->tensor_count) {
            extra++;
            continue;
        }

        name = model->tensors[rec->tensor_id].name;

        if (!parse_layer_id(name, &layer_id) || (layer_id >= n_layers)) {
            extra++;
            continue;
        }

        if (rec->tile_id == ATT1_SHARD_TILE_UNASSIGNED) {
            continue;  /* unassigned; does not contribute a placement */
        }

        if (!slots[layer_id].seen) {
            slots[layer_id].seen    = 1;
            slots[layer_id].tile_id = rec->tile_id;
        } else if ((slots[layer_id].tile_id != rec->tile_id) &&
                   !slots[layer_id].conflict) {
            slots[layer_id].conflict = 1;
            conflict++;
        }
    }

    /* Count covered layers */
    for (i = 0u; i < n_layers; i++) {
        if (slots[i].seen) {
            entry_count++;
        }
    }

    if (entry_count > 0u) {
        if (entry_count > (uint32_t)(SIZE_MAX / sizeof(*entries))) {
            free(slots);
            return ATT1_ERR_OOM;
        }

        entries = calloc(entry_count, sizeof(*entries));
        if (entries == NULL) {
            free(slots);
            return ATT1_ERR_OOM;
        }

        /* Pass 2: fill entries in layer order */
        {
            uint32_t j = 0u;

            for (i = 0u; i < n_layers; i++) {
                if (slots[i].seen) {
                    entries[j].layer_id = i;
                    entries[j].tile_id  = slots[i].tile_id;
                    j++;
                }
            }
        }
    }

    free(slots);

    out->entries  = entries;
    out->count    = entry_count;
    out->extra    = extra;
    out->conflict = conflict;

    return ATT1_OK;
}

void att1_meta_plan_free(att1_meta_plan *plan)
{
    if (plan == NULL) {
        return;
    }
    free(plan->entries);
    memset(plan, 0, sizeof(*plan));
}

att1_status_t att1_meta_plan_compare(const att1_meta_plan  *proposed,
                                     const att1_shard_plan *runtime,
                                     att1_meta_plan_diff   *out)
{
    uint32_t layer_id;

    if ((proposed == NULL) || (runtime == NULL) || (out == NULL)) {
        return ATT1_ERR_INVALID_ARG;
    }

    memset(out, 0, sizeof(*out));
    out->extra    = proposed->extra;
    out->conflict = proposed->conflict;

    for (layer_id = 0u; layer_id < (uint32_t)runtime->layer_count; layer_id++) {
        uint32_t p;
        const att1_meta_plan_entry *entry = NULL;

        /* Linear search in proposed entries (sorted by layer_id). */
        for (p = 0u; p < proposed->count; p++) {
            if (proposed->entries[p].layer_id == layer_id) {
                entry = &proposed->entries[p];
                break;
            }
        }

        if (entry == NULL) {
            out->missing++;
        } else {
            uint32_t runtime_tile = 0u;

            (void)att1_shard_plan_layer_tile(runtime, layer_id, &runtime_tile);

            if (entry->tile_id == runtime_tile) {
                out->matching++;
            } else {
                out->mismatch++;
            }
        }
    }

    return ATT1_OK;
}
