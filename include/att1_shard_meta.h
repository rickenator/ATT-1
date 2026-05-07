#ifndef ATT1_SHARD_META_H
#define ATT1_SHARD_META_H

#include "att1_status.h"

#include <stddef.h>
#include <stdint.h>

/* Wire size of one shard metadata record (bytes, little-endian). */
#define ATT1_SHARD_META_RECORD_SIZE 120u

/* tile_id sentinel: tensor not yet assigned to any AIMU tile. */
#define ATT1_SHARD_TILE_UNASSIGNED 0xFFFFFFFFu

/* dtype codes (mirror att1_model_dtype numbering) */
#define ATT1_SHARD_DTYPE_F32  1u
#define ATT1_SHARD_DTYPE_Q8   2u

/* quantization codes */
#define ATT1_SHARD_QUANT_NONE        0u
#define ATT1_SHARD_QUANT_PER_ROW_Q8  1u

/* replication_policy codes */
#define ATT1_SHARD_REPL_NONE         0u
#define ATT1_SHARD_REPL_READ         1u
#define ATT1_SHARD_REPL_WRITE_BCAST  2u

/* reduction_behavior codes */
#define ATT1_SHARD_REDUCE_NONE    0u
#define ATT1_SHARD_REDUCE_SUM     1u
#define ATT1_SHARD_REDUCE_MAX     2u
#define ATT1_SHARD_REDUCE_CONCAT  3u

/*
 * Parsed representation of one 120-byte shard metadata record.
 *
 * Wire layout (all little-endian):
 *   @  0  tensor_id          u32
 *   @  4  tile_id            u32
 *   @  8  byte_offset        u64
 *   @ 16  shape[4]           u64[4]  (32 bytes)
 *   @ 48  dtype              u32
 *   @ 52  quantization       u32
 *   @ 56  owner_aimu         u32
 *   @ 60  replication_policy u32
 *   @ 64  dependency_graph[8] u32[8] (32 bytes)
 *   @ 96  allowed_ops        u32
 *   @100  routing_requirements u32
 *   @104  reduction_behavior u32
 *   @108  _reserved          u32     (must be zero; not stored)
 *   @112  checksum           u64
 *   ────────────────────────────────
 *         Total              120 bytes
 *
 * The _reserved field is validated (must be zero) but is not stored.
 * A zero checksum means no verification; non-zero checksum is accepted
 * but not yet verified (CRC-64 implementation deferred to a later milestone).
 */
typedef struct att1_shard_meta_record {
    uint32_t tensor_id;
    uint32_t tile_id;
    uint64_t byte_offset;
    uint64_t shape[4];
    uint32_t dtype;
    uint32_t quantization;
    uint32_t owner_aimu;
    uint32_t replication_policy;
    uint32_t dependency_graph[8];
    uint32_t allowed_ops;
    uint32_t routing_requirements;
    uint32_t reduction_behavior;
    uint64_t checksum;
} att1_shard_meta_record;

/*
 * Container for a parsed shard metadata section.
 * count == 0 and records == NULL when no shard metadata is present.
 */
typedef struct att1_shard_meta {
    att1_shard_meta_record *records;
    uint64_t                count;
} att1_shard_meta;

/*
 * Forward declaration; full definition in att1_model.h.
 * Used only as a const pointer parameter for cross-validation.
 */
struct att1_model_tensor;

/*
 * Parse and validate a raw shard metadata section.
 *
 *   section_data   First byte of the shard metadata section in the file.
 *   section_size   Byte length of the section.
 *   tensor_count   Number of tensor descriptors in the model.
 *   tensors        Parsed tensor descriptors (for shape cross-validation).
 *   out_meta       Caller-allocated output; populated on ATT1_OK.
 *
 * Returns ATT1_OK on success.
 * Returns ATT1_ERR_BAD_FORMAT for any structural or field violation.
 * Returns ATT1_ERR_OOM on allocation failure.
 */
att1_status_t att1_shard_meta_parse(const unsigned char            *section_data,
                                     uint64_t                        section_size,
                                     uint64_t                        tensor_count,
                                     const struct att1_model_tensor *tensors,
                                     att1_shard_meta                *out_meta);

void att1_shard_meta_free(att1_shard_meta *meta);

#endif /* ATT1_SHARD_META_H */
