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

/* -------------------------------------------------------------------------
 * Metadata-driven proposed shard plan (Milestone 39)
 *
 * Advisory only.  Never used for scheduling.  Derived from shard metadata
 * when present; the runtime plan (att1_shard_plan) is always used for
 * actual inference.
 * ---------------------------------------------------------------------- */

/*
 * One proposed layer→tile assignment derived from shard metadata.
 */
typedef struct att1_meta_plan_entry {
    uint32_t layer_id;  /* transformer layer index */
    uint32_t tile_id;   /* proposed tile (first encountered on conflict) */
} att1_meta_plan_entry;

/*
 * Proposed shard plan built from shard metadata tensor placement records.
 * count == 0 when no metadata is present or no layer tensors are covered.
 */
typedef struct att1_meta_plan {
    att1_meta_plan_entry *entries;   /* one per covered layer, sorted by layer_id */
    uint32_t              count;     /* number of covered layers */
    uint32_t              extra;     /* records that don't map to any layer */
    uint32_t              conflict;  /* layers with conflicting tile_id values */
} att1_meta_plan;

/*
 * Comparison result between a proposed metadata plan and the runtime plan.
 */
typedef struct att1_meta_plan_diff {
    uint32_t matching;   /* layers with identical tile assignment */
    uint32_t mismatch;   /* layers where proposed tile != runtime tile */
    uint32_t missing;    /* layers in runtime plan absent from proposed */
    uint32_t extra;      /* metadata records not mapping to any layer */
    uint32_t conflict;   /* layers with conflicting metadata tile assignments */
} att1_meta_plan_diff;

/*
 * Derive a proposed shard plan from model shard metadata.
 *
 * Each tensor name of the form "layers.N.anything" contributes to the
 * proposed tile assignment for layer N.  Tensors with
 * tile_id == ATT1_SHARD_TILE_UNASSIGNED are skipped.  When multiple tensors
 * in the same layer carry different tile_id values, the layer is counted in
 * out->conflict and the first encountered tile_id is used.
 *
 * Returns ATT1_OK on success (out->count may be 0 for absent metadata).
 * Returns ATT1_ERR_OOM on allocation failure.
 * Returns ATT1_ERR_INVALID_ARG if model or out is NULL.
 * Caller must call att1_meta_plan_free() when done.
 */
att1_status_t att1_meta_plan_build(const att1_model *model,
                                   att1_meta_plan   *out);

void att1_meta_plan_free(att1_meta_plan *plan);

/*
 * Compare a metadata-derived proposed plan against the runtime shard plan.
 * Populates out; the extra and conflict fields are copied from proposed.
 *
 * Returns ATT1_OK always.
 * Returns ATT1_ERR_INVALID_ARG if any pointer is NULL.
 */
att1_status_t att1_meta_plan_compare(const att1_meta_plan  *proposed,
                                     const att1_shard_plan *runtime,
                                     att1_meta_plan_diff   *out);

#endif
