/*
 * aimu_device.c  —  AIMU device discovery and tile capability simulator (M106)
 *
 * In-process implementation.  See include/att1_aimu_device.h for the public
 * API and docs/aimu_register_map.md for the register-map specification.
 */

#include "att1_aimu_device.h"

#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Internal defaults
 * ---------------------------------------------------------------------- */

#define DEFAULT_TILE_MEMORY_BYTES   (UINT64_C(1) << 30)   /* 1 GiB */
#define DEFAULT_TILE_KV_BYTES       (UINT64_C(256) << 20) /* 256 MiB */
#define DEFAULT_MAX_SESSIONS        4u

/* -------------------------------------------------------------------------
 * Lifecycle
 * ---------------------------------------------------------------------- */

att1_status_t att1_aimu_device_create(const att1_aimu_device_config *config,
                                       att1_aimu_device             **out)
{
    att1_aimu_device_config cfg;
    att1_aimu_device       *dev;
    size_t                  i;

    if (out == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }
    *out = NULL;

    /* Build effective config from caller or defaults. */
    if (config == NULL) {
        memset(&cfg, 0, sizeof(cfg));
        cfg.tile_count      = 1u;
        cfg.tile_memory_bytes = DEFAULT_TILE_MEMORY_BYTES;
        cfg.tile_kv_bytes     = DEFAULT_TILE_KV_BYTES;
        cfg.supported_dtypes  = ATT1_AIMU_DTYPE_ALL;
        cfg.supported_ops     = ATT1_AIMU_OP_ALL;
        cfg.feature_flags     = ATT1_AIMU_DEVICE_DEFAULT_FEATURES;
        cfg.version.major     = ATT1_AIMU_DEVICE_VERSION_MAJOR;
        cfg.version.minor     = ATT1_AIMU_DEVICE_VERSION_MINOR;
        cfg.version.patch     = ATT1_AIMU_DEVICE_VERSION_PATCH;
        cfg.version.build     = ATT1_AIMU_DEVICE_VERSION_BUILD;
    } else {
        cfg = *config;

        /* Apply zero-means-default rules. */
        if (cfg.tile_count == 0u) {
            cfg.tile_count = 1u;
        }
        if (cfg.tile_memory_bytes == 0u) {
            cfg.tile_memory_bytes = DEFAULT_TILE_MEMORY_BYTES;
        }
        if (cfg.tile_kv_bytes == 0u) {
            cfg.tile_kv_bytes = DEFAULT_TILE_KV_BYTES;
        }
        if (cfg.supported_dtypes == 0u) {
            cfg.supported_dtypes = ATT1_AIMU_DTYPE_ALL;
        }
        if (cfg.supported_ops == 0u) {
            cfg.supported_ops = ATT1_AIMU_OP_ALL;
        }
        if (cfg.feature_flags == 0u) {
            cfg.feature_flags = ATT1_AIMU_DEVICE_DEFAULT_FEATURES;
        }
        /* Version: if all fields are zero use the default. */
        if (cfg.version.major == 0u && cfg.version.minor == 0u &&
            cfg.version.patch == 0u && cfg.version.build == 0u) {
            cfg.version.major = ATT1_AIMU_DEVICE_VERSION_MAJOR;
            cfg.version.minor = ATT1_AIMU_DEVICE_VERSION_MINOR;
            cfg.version.patch = ATT1_AIMU_DEVICE_VERSION_PATCH;
            cfg.version.build = ATT1_AIMU_DEVICE_VERSION_BUILD;
        }
    }

    /* Validate. */
    if (cfg.tile_count == 0u || cfg.tile_count > ATT1_AIMU_DEVICE_MAX_TILES) {
        return ATT1_ERR_INVALID_ARG;
    }

    dev = (att1_aimu_device *)calloc(1u, sizeof(*dev));
    if (dev == NULL) {
        return ATT1_ERR_OOM;
    }

    dev->magic                = ATT1_AIMU_DEVICE_MAGIC;
    dev->register_map_version = ATT1_AIMU_REGISTER_MAP_VERSION;
    dev->version              = cfg.version;
    dev->feature_flags        = cfg.feature_flags;
    dev->tile_count           = cfg.tile_count;
    dev->global_status        = 0u;
    dev->global_error         = 0u;
    dev->reset_count          = 0u;
    dev->cmdq                 = NULL;

    for (i = 0u; i < cfg.tile_count; i++) {
        att1_aimu_tile_info *t = &dev->tiles[i];
        t->tile_id                 = (uint8_t)i;
        t->state                   = ATT1_AIMU_TILE_IDLE;
        t->memory_capacity_bytes   = cfg.tile_memory_bytes;
        t->kv_capacity_bytes       = cfg.tile_kv_bytes;
        t->supported_dtypes        = cfg.supported_dtypes;
        t->supported_ops           = cfg.supported_ops;
        t->memory_used_bytes       = 0u;
        t->kv_used_bytes           = 0u;
        t->max_sessions            = DEFAULT_MAX_SESSIONS;
        t->error_code              = 0u;
        t->reset_count             = 0u;

        /*
         * Fabric link mask: tile i links to all other tiles in the device.
         * E.g. for 4 tiles, tile 0 has mask 0b1110.
         */
        {
            uint16_t mask = 0u;
            size_t   j;
            for (j = 0u; j < cfg.tile_count; j++) {
                if (j != i) {
                    mask |= (uint16_t)(1u << j);
                }
            }
            t->fabric_link_mask = mask;
        }
    }

    *out = dev;
    return ATT1_OK;
}

void att1_aimu_device_destroy(att1_aimu_device *dev)
{
    if (dev == NULL) {
        return;
    }
    /* Poison magic to catch use-after-free. */
    dev->magic = 0u;
    free(dev);
}

/* -------------------------------------------------------------------------
 * Device-level queries
 * ---------------------------------------------------------------------- */

att1_status_t att1_aimu_device_query_info(const att1_aimu_device *dev,
                                           att1_aimu_device_info  *info)
{
    if (dev == NULL || info == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }
    info->register_map_version = dev->register_map_version;
    info->version              = dev->version;
    info->feature_flags        = dev->feature_flags;
    info->tile_count           = dev->tile_count;
    info->global_status        = dev->global_status;
    info->global_error         = dev->global_error;
    return ATT1_OK;
}

size_t att1_aimu_device_tile_count(const att1_aimu_device *dev)
{
    if (dev == NULL) {
        return 0u;
    }
    return dev->tile_count;
}

/* -------------------------------------------------------------------------
 * Tile-level queries
 * ---------------------------------------------------------------------- */

att1_status_t att1_aimu_device_query_tile(const att1_aimu_device *dev,
                                           uint8_t                 tile_id,
                                           att1_aimu_tile_info    *info)
{
    if (dev == NULL || info == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }
    if ((size_t)tile_id >= dev->tile_count) {
        return ATT1_ERR_INVALID_ARG;
    }
    *info = dev->tiles[tile_id];
    return ATT1_OK;
}

att1_status_t att1_aimu_device_validate_tile_id(const att1_aimu_device *dev,
                                                 uint8_t                 tile_id)
{
    if (dev == NULL || (size_t)tile_id >= dev->tile_count) {
        return ATT1_ERR_INVALID_ARG;
    }
    return ATT1_OK;
}

int att1_aimu_device_supports_dtype(const att1_aimu_device *dev,
                                     uint32_t                dtype_bit)
{
    size_t i;
    if (dev == NULL) {
        return 0;
    }
    for (i = 0u; i < dev->tile_count; i++) {
        if (!(dev->tiles[i].supported_dtypes & dtype_bit)) {
            return 0;
        }
    }
    return 1;
}

int att1_aimu_device_supports_op(const att1_aimu_device *dev,
                                  uint32_t                op_bit)
{
    size_t i;
    if (dev == NULL) {
        return 0;
    }
    for (i = 0u; i < dev->tile_count; i++) {
        if (!(dev->tiles[i].supported_ops & op_bit)) {
            return 0;
        }
    }
    return 1;
}

int att1_aimu_device_tile_supports_dtype(const att1_aimu_device *dev,
                                          uint8_t                 tile_id,
                                          uint32_t                dtype_bit)
{
    if (dev == NULL || (size_t)tile_id >= dev->tile_count) {
        return 0;
    }
    return (dev->tiles[tile_id].supported_dtypes & dtype_bit) ? 1 : 0;
}

int att1_aimu_device_tile_supports_op(const att1_aimu_device *dev,
                                       uint8_t                 tile_id,
                                       uint32_t                op_bit)
{
    if (dev == NULL || (size_t)tile_id >= dev->tile_count) {
        return 0;
    }
    return (dev->tiles[tile_id].supported_ops & op_bit) ? 1 : 0;
}

/* -------------------------------------------------------------------------
 * Reset
 * ---------------------------------------------------------------------- */

att1_status_t att1_aimu_device_reset(att1_aimu_device *dev)
{
    size_t i;
    if (dev == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }
    dev->global_status = 0u;
    dev->global_error  = 0u;
    dev->reset_count++;

    for (i = 0u; i < dev->tile_count; i++) {
        att1_aimu_tile_info *t = &dev->tiles[i];
        t->memory_used_bytes = 0u;
        t->kv_used_bytes     = 0u;
        t->error_code        = 0u;
        t->state             = ATT1_AIMU_TILE_IDLE;
        /* Note: reset_count on each tile is NOT touched by device reset;
         * only att1_aimu_device_reset_tile increments the per-tile counter. */
    }
    return ATT1_OK;
}

att1_status_t att1_aimu_device_reset_tile(att1_aimu_device *dev,
                                           uint8_t           tile_id)
{
    att1_aimu_tile_info *t;
    if (dev == NULL || (size_t)tile_id >= dev->tile_count) {
        return ATT1_ERR_INVALID_ARG;
    }
    t = &dev->tiles[tile_id];
    t->memory_used_bytes = 0u;
    t->kv_used_bytes     = 0u;
    t->error_code        = 0u;
    t->state             = ATT1_AIMU_TILE_IDLE;
    t->reset_count++;
    return ATT1_OK;
}

/* -------------------------------------------------------------------------
 * Counter snapshot
 * ---------------------------------------------------------------------- */

att1_status_t att1_aimu_device_snapshot_counters(
        const att1_aimu_device       *dev,
        att1_aimu_cmdq_counters      *out)
{
    if (dev == NULL || out == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }
    if (dev->cmdq == NULL) {
        return ATT1_ERR_STATE;
    }
    return att1_aimu_cmdq_get_counters(dev->cmdq, out);
}

/* -------------------------------------------------------------------------
 * Command-queue attachment
 * ---------------------------------------------------------------------- */

att1_status_t att1_aimu_device_attach_cmdq(att1_aimu_device *dev,
                                            att1_aimu_cmdq   *q)
{
    if (dev == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }
    dev->cmdq = q;
    return ATT1_OK;
}

/* -------------------------------------------------------------------------
 * Name helpers
 * ---------------------------------------------------------------------- */

const char *att1_aimu_dtype_name(uint32_t dtype_bit)
{
    switch (dtype_bit) {
    case ATT1_AIMU_DTYPE_F32:   return "F32";
    case ATT1_AIMU_DTYPE_Q8:    return "Q8";
    case ATT1_AIMU_DTYPE_Q4:    return "Q4";
    default:                     return "UNKNOWN";
    }
}

const char *att1_aimu_op_name(uint32_t op_bit)
{
    switch (op_bit) {
    case ATT1_AIMU_OP_MATMUL:        return "MATMUL";
    case ATT1_AIMU_OP_RMSNORM:       return "RMSNORM";
    case ATT1_AIMU_OP_ROPE:          return "ROPE";
    case ATT1_AIMU_OP_ATTENTION:     return "ATTENTION";
    case ATT1_AIMU_OP_FFN:           return "FFN";
    case ATT1_AIMU_OP_KV_APPEND:     return "KV_APPEND";
    case ATT1_AIMU_OP_KV_READ:       return "KV_READ";
    case ATT1_AIMU_OP_FABRIC_SEND:   return "FABRIC_SEND";
    case ATT1_AIMU_OP_FABRIC_REDUCE: return "FABRIC_REDUCE";
    default:                          return "UNKNOWN";
    }
}

const char *att1_aimu_feat_name(uint64_t feat_bit)
{
    switch (feat_bit) {
    case ATT1_AIMU_FEAT_CMD_RING:        return "CMD_RING";
    case ATT1_AIMU_FEAT_COMP_RING:       return "COMP_RING";
    case ATT1_AIMU_FEAT_DMA:             return "DMA";
    case ATT1_AIMU_FEAT_FABRIC:          return "FABRIC";
    case ATT1_AIMU_FEAT_TRACE:           return "TRACE";
    case ATT1_AIMU_FEAT_COUNTERS:        return "COUNTERS";
    case ATT1_AIMU_FEAT_MSI_X:           return "MSI_X";
    case ATT1_AIMU_FEAT_MULTI_SESSION:   return "MULTI_SESSION";
    case ATT1_AIMU_FEAT_FENCE:           return "FENCE";
    case ATT1_AIMU_FEAT_KV_MMU:          return "KV_MMU";
    case ATT1_AIMU_FEAT_PLACEMENT_AWARE: return "PLACEMENT_AWARE";
    default:                              return "UNKNOWN";
    }
}

const char *att1_aimu_tile_state_name(att1_aimu_tile_state state)
{
    switch (state) {
    case ATT1_AIMU_TILE_IDLE:   return "IDLE";
    case ATT1_AIMU_TILE_ACTIVE: return "ACTIVE";
    case ATT1_AIMU_TILE_ERROR:  return "ERROR";
    case ATT1_AIMU_TILE_RESET:  return "RESET";
    default:                     return "UNKNOWN";
    }
}
