#include "att1_shard_meta.h"
#include "att1_model.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t shard_read_u32le(const unsigned char *p)
{
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8u) |
           ((uint32_t)p[2] << 16u) |
           ((uint32_t)p[3] << 24u);
}

static uint64_t shard_read_u64le(const unsigned char *p)
{
    return ((uint64_t)p[0]) |
           ((uint64_t)p[1] << 8u) |
           ((uint64_t)p[2] << 16u) |
           ((uint64_t)p[3] << 24u) |
           ((uint64_t)p[4] << 32u) |
           ((uint64_t)p[5] << 40u) |
           ((uint64_t)p[6] << 48u) |
           ((uint64_t)p[7] << 56u);
}

static int shard_dtype_valid(uint32_t dtype)
{
    return (dtype == ATT1_SHARD_DTYPE_F32) || (dtype == ATT1_SHARD_DTYPE_Q8);
}

static int shard_quant_valid(uint32_t quant)
{
    return (quant == ATT1_SHARD_QUANT_NONE) ||
           (quant == ATT1_SHARD_QUANT_PER_ROW_Q8);
}

static int shard_repl_valid(uint32_t repl)
{
    return (repl == ATT1_SHARD_REPL_NONE) ||
           (repl == ATT1_SHARD_REPL_READ) ||
           (repl == ATT1_SHARD_REPL_WRITE_BCAST);
}

static int shard_reduce_valid(uint32_t reduce)
{
    return (reduce == ATT1_SHARD_REDUCE_NONE) ||
           (reduce == ATT1_SHARD_REDUCE_SUM) ||
           (reduce == ATT1_SHARD_REDUCE_MAX) ||
           (reduce == ATT1_SHARD_REDUCE_CONCAT);
}

att1_status_t att1_shard_meta_parse(const unsigned char            *section_data,
                                     uint64_t                        section_size,
                                     uint64_t                        tensor_count,
                                     const struct att1_model_tensor *tensors,
                                     att1_shard_meta                *out_meta)
{
    att1_shard_meta_record *records = NULL;
    uint64_t record_count = 0u;
    uint64_t i = 0u;
    uint32_t j = 0u;

    if ((section_data == NULL) || (out_meta == NULL)) {
        return ATT1_ERR_INVALID_ARG;
    }

    out_meta->records = NULL;
    out_meta->count   = 0u;

    /* Section size must be an exact multiple of the record wire size. */
    if ((section_size % ATT1_SHARD_META_RECORD_SIZE) != 0u) {
        return ATT1_ERR_BAD_FORMAT;
    }

    record_count = section_size / ATT1_SHARD_META_RECORD_SIZE;

    /* Record count must equal the number of tensor descriptors. */
    if (record_count != tensor_count) {
        return ATT1_ERR_BAD_FORMAT;
    }

    if (tensor_count == 0u) {
        return ATT1_OK;
    }

    if (tensor_count > (uint64_t)(SIZE_MAX / sizeof(*records))) {
        return ATT1_ERR_OOM;
    }

    records = calloc((size_t)tensor_count, sizeof(*records));
    if (records == NULL) {
        return ATT1_ERR_OOM;
    }

    for (i = 0u; i < tensor_count; i++) {
        const unsigned char    *p   = section_data +
                                      (i * ATT1_SHARD_META_RECORD_SIZE);
        att1_shard_meta_record *rec = &records[i];
        uint32_t                reserved = 0u;

        rec->tensor_id          = shard_read_u32le(&p[0]);
        rec->tile_id            = shard_read_u32le(&p[4]);
        rec->byte_offset        = shard_read_u64le(&p[8]);
        for (j = 0u; j < 4u; j++) {
            rec->shape[j] = shard_read_u64le(&p[16u + ((size_t)j * 8u)]);
        }
        rec->dtype              = shard_read_u32le(&p[48]);
        rec->quantization       = shard_read_u32le(&p[52]);
        rec->owner_aimu         = shard_read_u32le(&p[56]);
        rec->replication_policy = shard_read_u32le(&p[60]);
        for (j = 0u; j < 8u; j++) {
            rec->dependency_graph[j] = shard_read_u32le(&p[64u +
                                                           ((size_t)j * 4u)]);
        }
        rec->allowed_ops           = shard_read_u32le(&p[96]);
        rec->routing_requirements  = shard_read_u32le(&p[100]);
        rec->reduction_behavior    = shard_read_u32le(&p[104]);
        reserved                   = shard_read_u32le(&p[108]);
        rec->checksum              = shard_read_u64le(&p[112]);

        /* tensor_id values must be 0, 1, ..., tensor_count-1 in order. */
        if (rec->tensor_id != (uint32_t)i) {
            free(records);
            return ATT1_ERR_BAD_FORMAT;
        }

        /* _reserved must be zero to prevent future field aliasing. */
        if (reserved != 0u) {
            free(records);
            return ATT1_ERR_BAD_FORMAT;
        }

        /* Validate enum-like fields. */
        if (!shard_dtype_valid(rec->dtype)) {
            free(records);
            return ATT1_ERR_BAD_FORMAT;
        }

        if (!shard_quant_valid(rec->quantization)) {
            free(records);
            return ATT1_ERR_BAD_FORMAT;
        }

        if (!shard_repl_valid(rec->replication_policy)) {
            free(records);
            return ATT1_ERR_BAD_FORMAT;
        }

        if (!shard_reduce_valid(rec->reduction_behavior)) {
            free(records);
            return ATT1_ERR_BAD_FORMAT;
        }

        /*
         * Cross-validate shape against the parsed tensor descriptor.
         * tensors is non-NULL when tensor_count > 0 (checked above).
         */
        if (tensors != NULL) {
            uint32_t d = 0u;

            for (d = 0u; d < ATT1_MODEL_MAX_DIMS; d++) {
                if (rec->shape[d] != tensors[i].shape[d]) {
                    free(records);
                    return ATT1_ERR_BAD_FORMAT;
                }
            }
        }

        /*
         * Checksum: zero means no verification (permitted for development
         * artifacts and models without a pre-computed checksum).
         * Non-zero checksum verification (CRC-64/ECMA-182) is deferred
         * to a later milestone.
         */
    }

    out_meta->records = records;
    out_meta->count   = tensor_count;
    return ATT1_OK;
}

void att1_shard_meta_free(att1_shard_meta *meta)
{
    if (meta == NULL) {
        return;
    }

    free(meta->records);
    meta->records = NULL;
    meta->count   = 0u;
}

void att1_shard_meta_summarize(const att1_shard_meta   *meta,
                               att1_shard_meta_summary *out)
{
    uint64_t i = 0u;

    if (out == NULL) {
        return;
    }

    memset(out, 0, sizeof(*out));

    if ((meta == NULL) || (meta->count == 0u)) {
        return;
    }

    out->count = meta->count;

    for (i = 0u; i < meta->count; i++) {
        const att1_shard_meta_record *rec = &meta->records[i];
        uint64_t j = 0u;
        int seen = 0;

        /* assigned vs unassigned */
        if (rec->tile_id == ATT1_SHARD_TILE_UNASSIGNED) {
            out->unassigned++;
        } else {
            out->assigned++;

            /* unique tile: check if this tile_id appeared in earlier records */
            seen = 0;
            for (j = 0u; j < i; j++) {
                if (meta->records[j].tile_id == rec->tile_id) {
                    seen = 1;
                    break;
                }
            }
            if (!seen) {
                out->unique_tiles++;
            }
        }

        /* unique aimu: check if this owner_aimu appeared in earlier records */
        seen = 0;
        for (j = 0u; j < i; j++) {
            if (meta->records[j].owner_aimu == rec->owner_aimu) {
                seen = 1;
                break;
            }
        }
        if (!seen) {
            out->unique_aimus++;
        }

        /* dtype breakdown */
        if (rec->dtype == ATT1_SHARD_DTYPE_F32) {
            out->dtype_f32++;
        } else if (rec->dtype == ATT1_SHARD_DTYPE_Q8) {
            out->dtype_q8++;
        }
    }
}

/* -------------------------------------------------------------------------
 * Consistency validation
 * ---------------------------------------------------------------------- */

static int violation_append(att1_shard_meta_validation *v,
                            uint32_t                    tensor_id,
                            const char                 *field,
                            const char                 *desc)
{
    att1_shard_meta_violation *newvec = NULL;
    uint64_t                   new_count = 0u;

    if (v->count >= (uint64_t)(SIZE_MAX / sizeof(*newvec))) {
        return -1;
    }

    new_count = v->count + 1u;
    newvec = realloc(v->violations,
                     (size_t)new_count * sizeof(*newvec));
    if (newvec == NULL) {
        return -1;
    }

    v->violations = newvec;
    v->violations[v->count].tensor_id = tensor_id;

    strncpy(v->violations[v->count].field, field,
            ATT1_SHARD_META_VIOLATION_FIELD_SIZE - 1u);
    v->violations[v->count].field[ATT1_SHARD_META_VIOLATION_FIELD_SIZE - 1u]
        = '\0';

    strncpy(v->violations[v->count].description, desc,
            ATT1_SHARD_META_VIOLATION_DESC_SIZE - 1u);
    v->violations[v->count].description[ATT1_SHARD_META_VIOLATION_DESC_SIZE - 1u]
        = '\0';

    v->count = new_count;
    return 0;
}

att1_status_t att1_shard_meta_validate(
        const att1_shard_meta           *meta,
        const struct att1_model_config  *config,
        const struct att1_model_tensor  *tensors,
        uint64_t                         tensor_count,
        att1_shard_meta_validation      *out)
{
    uint64_t i = 0u;
    char     desc[ATT1_SHARD_META_VIOLATION_DESC_SIZE];

    if (out == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }

    out->violations = NULL;
    out->count      = 0u;

    /* Nothing to validate when metadata is absent. */
    if ((meta == NULL) || (meta->count == 0u) || (config == NULL)) {
        return ATT1_OK;
    }

    for (i = 0u; i < meta->count; i++) {
        const att1_shard_meta_record *rec = &meta->records[i];

        /* ── tile_id within [0, n_tiles) ─────────────────────────────── */
        if ((rec->tile_id != ATT1_SHARD_TILE_UNASSIGNED) &&
            (config->n_tiles > 0u) &&
            (rec->tile_id >= config->n_tiles)) {
            (void)snprintf(desc, sizeof(desc),
                           "tile_id %u out of range (n_tiles=%u)",
                           rec->tile_id, config->n_tiles);
            if (violation_append(out, rec->tensor_id, "tile_id", desc) != 0) {
                att1_shard_meta_validation_free(out);
                return ATT1_ERR_OOM;
            }
        }

        /* ── owner_aimu within [0, n_tiles) ──────────────────────────── */
        if ((config->n_tiles > 0u) &&
            (rec->owner_aimu >= config->n_tiles)) {
            (void)snprintf(desc, sizeof(desc),
                           "owner_aimu %u out of range (n_tiles=%u)",
                           rec->owner_aimu, config->n_tiles);
            if (violation_append(out, rec->tensor_id, "owner_aimu", desc) != 0) {
                att1_shard_meta_validation_free(out);
                return ATT1_ERR_OOM;
            }
        }

        /* ── per-tensor descriptor checks ────────────────────────────── */
        if ((tensors != NULL) && (rec->tensor_id < (uint32_t)tensor_count)) {
            const att1_model_tensor *t = &tensors[rec->tensor_id];
            int dtype_ok = 0;

            /* dtype consistency */
            if (rec->dtype == ATT1_SHARD_DTYPE_F32) {
                dtype_ok = (t->dtype == (uint32_t)ATT1_MODEL_DTYPE_F32);
            }
            /* ATT1_SHARD_DTYPE_Q8: no matching model dtype exists yet */

            if (!dtype_ok) {
                (void)snprintf(desc, sizeof(desc),
                               "dtype mismatch: shard=%u tensor=%u",
                               rec->dtype, t->dtype);
                if (violation_append(out, rec->tensor_id, "dtype", desc) != 0) {
                    att1_shard_meta_validation_free(out);
                    return ATT1_ERR_OOM;
                }
            }

            /* byte_offset must mirror tensor descriptor offset (version=1) */
            if (rec->byte_offset != t->offset) {
                (void)snprintf(desc, sizeof(desc),
                               "byte_offset %" PRIu64 " does not match tensor offset %" PRIu64,
                               rec->byte_offset, t->offset);
                if (violation_append(out, rec->tensor_id, "byte_offset", desc) != 0) {
                    att1_shard_meta_validation_free(out);
                    return ATT1_ERR_OOM;
                }
            }
        }
    }

    return ATT1_OK;
}

void att1_shard_meta_validation_free(att1_shard_meta_validation *v)
{
    if (v == NULL) {
        return;
    }

    free(v->violations);
    v->violations = NULL;
    v->count      = 0u;
}
