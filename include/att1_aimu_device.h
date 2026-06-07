/*
 * att1_aimu_device.h  —  AIMU device discovery and tile capability simulator (M106)
 *
 * In-process simulator for the device probe / tile enumeration phase defined
 * in M104 (docs/aimu_register_map.md §2–§3).  It models:
 *
 *   • Per-device version / feature registers  (BAR0 0x0000–0x003F)
 *   • Per-tile capability registers           (BAR0 0x8000+N×0x800)
 *   • Tile status / error / reset registers
 *   • Device-level global status and error
 *
 * This is NOT a real PCIe driver or MMIO accessor.  All register state lives
 * in heap-allocated C structs so that M105 command-queue sessions and future
 * milestones can validate placement reports against the discovered capability
 * of each simulated tile.
 */

#ifndef ATT1_AIMU_DEVICE_H
#define ATT1_AIMU_DEVICE_H

#include "att1_status.h"
#include "att1_aimu_cmdq.h"      /* att1_aimu_cmdq_counters */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Compile-time limits
 * ---------------------------------------------------------------------- */

/** Maximum number of tiles in a simulated device. */
#define ATT1_AIMU_DEVICE_MAX_TILES      16u

/** Magic written into every att1_aimu_device to detect use-after-free. */
#define ATT1_AIMU_DEVICE_MAGIC          UINT32_C(0xA771DE7C)

/* -------------------------------------------------------------------------
 * Version fields   (mirrors M104 §2: DEVICE_VERSION, REGISTER_MAP_VERSION)
 * ---------------------------------------------------------------------- */

/** Default device version components. */
#define ATT1_AIMU_DEVICE_VERSION_MAJOR  0u
#define ATT1_AIMU_DEVICE_VERSION_MINOR  1u
#define ATT1_AIMU_DEVICE_VERSION_PATCH  0u
#define ATT1_AIMU_DEVICE_VERSION_BUILD  0u

/** Register-map version implemented by this simulator. */
#define ATT1_AIMU_REGISTER_MAP_VERSION  UINT32_C(0x00010000)  /* v1.0 */

/* -------------------------------------------------------------------------
 * Feature flags   (M104 §2: FEATURE_FLAGS_LOW/HIGH)
 *
 * Each constant is a bit position within the 64-bit feature word.
 * ---------------------------------------------------------------------- */

#define ATT1_AIMU_FEAT_CMD_RING         (UINT64_C(1) << 0)  /**< command ring present     */
#define ATT1_AIMU_FEAT_COMP_RING        (UINT64_C(1) << 1)  /**< completion ring present  */
#define ATT1_AIMU_FEAT_DMA              (UINT64_C(1) << 2)  /**< DMA engine present       */
#define ATT1_AIMU_FEAT_FABRIC           (UINT64_C(1) << 3)  /**< fabric link present      */
#define ATT1_AIMU_FEAT_TRACE            (UINT64_C(1) << 4)  /**< trace buffer present     */
#define ATT1_AIMU_FEAT_COUNTERS         (UINT64_C(1) << 5)  /**< hardware counters        */
#define ATT1_AIMU_FEAT_MSI_X            (UINT64_C(1) << 6)  /**< MSI-X interrupts         */
#define ATT1_AIMU_FEAT_MULTI_SESSION    (UINT64_C(1) << 7)  /**< multiple inference sessions */
#define ATT1_AIMU_FEAT_FENCE            (UINT64_C(1) << 8)  /**< cross-tile fences        */
#define ATT1_AIMU_FEAT_KV_MMU           (UINT64_C(1) << 9)  /**< KV-cache MMU             */
#define ATT1_AIMU_FEAT_PLACEMENT_AWARE  (UINT64_C(1) << 10) /**< placement-report aware   */

/** Feature flags set by default in the simulator. */
#define ATT1_AIMU_DEVICE_DEFAULT_FEATURES \
    (ATT1_AIMU_FEAT_CMD_RING | ATT1_AIMU_FEAT_COMP_RING | \
     ATT1_AIMU_FEAT_TRACE    | ATT1_AIMU_FEAT_COUNTERS  | \
     ATT1_AIMU_FEAT_MULTI_SESSION | ATT1_AIMU_FEAT_FENCE | \
     ATT1_AIMU_FEAT_KV_MMU   | ATT1_AIMU_FEAT_PLACEMENT_AWARE)

/* -------------------------------------------------------------------------
 * Supported dtype bitmask   (M104 §3: SUPPORTED_DTYPES)
 * ---------------------------------------------------------------------- */

#define ATT1_AIMU_DTYPE_F32     (UINT32_C(1) << 0)  /**< 32-bit float                 */
#define ATT1_AIMU_DTYPE_Q8      (UINT32_C(1) << 1)  /**< 8-bit symmetric quantization */
#define ATT1_AIMU_DTYPE_Q4      (UINT32_C(1) << 2)  /**< 4-bit grouped quantization   */

/** All dtypes supported (default). */
#define ATT1_AIMU_DTYPE_ALL \
    (ATT1_AIMU_DTYPE_F32 | ATT1_AIMU_DTYPE_Q8 | ATT1_AIMU_DTYPE_Q4)

/* -------------------------------------------------------------------------
 * Supported op bitmask   (M104 §3: SUPPORTED_OPS_LOW/HIGH)
 * ---------------------------------------------------------------------- */

#define ATT1_AIMU_OP_MATMUL         (UINT32_C(1) << 0)
#define ATT1_AIMU_OP_RMSNORM        (UINT32_C(1) << 1)
#define ATT1_AIMU_OP_ROPE           (UINT32_C(1) << 2)
#define ATT1_AIMU_OP_ATTENTION      (UINT32_C(1) << 3)
#define ATT1_AIMU_OP_FFN            (UINT32_C(1) << 4)
#define ATT1_AIMU_OP_KV_APPEND      (UINT32_C(1) << 5)
#define ATT1_AIMU_OP_KV_READ        (UINT32_C(1) << 6)
#define ATT1_AIMU_OP_FABRIC_SEND    (UINT32_C(1) << 7)
#define ATT1_AIMU_OP_FABRIC_REDUCE  (UINT32_C(1) << 8)

/** All ops supported (default). */
#define ATT1_AIMU_OP_ALL \
    (ATT1_AIMU_OP_MATMUL       | ATT1_AIMU_OP_RMSNORM    | \
     ATT1_AIMU_OP_ROPE         | ATT1_AIMU_OP_ATTENTION  | \
     ATT1_AIMU_OP_FFN          | ATT1_AIMU_OP_KV_APPEND  | \
     ATT1_AIMU_OP_KV_READ      | ATT1_AIMU_OP_FABRIC_SEND | \
     ATT1_AIMU_OP_FABRIC_REDUCE)

/* -------------------------------------------------------------------------
 * Tile state   (M104 §3: TILE_STATUS.state field)
 * ---------------------------------------------------------------------- */

typedef enum att1_aimu_tile_state {
    ATT1_AIMU_TILE_IDLE         = 0,  /**< ready, no active session            */
    ATT1_AIMU_TILE_ACTIVE       = 1,  /**< executing one or more sessions      */
    ATT1_AIMU_TILE_ERROR        = 2,  /**< halted on error; needs RESET_TILE   */
    ATT1_AIMU_TILE_RESET        = 3   /**< reset in progress                   */
} att1_aimu_tile_state;

/* -------------------------------------------------------------------------
 * Per-tile capability / info   (M104 §3: per-tile register window)
 * ---------------------------------------------------------------------- */

typedef struct att1_aimu_tile_info {
    uint8_t             tile_id;

    att1_aimu_tile_state state;

    /** Total model-weight memory available on this tile (bytes). */
    uint64_t            memory_capacity_bytes;
    /** KV-cache memory available on this tile (bytes). */
    uint64_t            kv_capacity_bytes;

    /** ATT1_AIMU_DTYPE_* bitmask. */
    uint32_t            supported_dtypes;

    /**
     * ATT1_AIMU_OP_* bitmask.
     * Maps to M104 SUPPORTED_OPS_LOW (bits 0–31) and SUPPORTED_OPS_HIGH
     * (bits 32–63); this simulator merges them into a single 32-bit field
     * since only 9 ops are defined.
     */
    uint32_t            supported_ops;

    /** Fabric link bitmask: bit N=1 if this tile has a link to tile N. */
    uint16_t            fabric_link_mask;

    /** Maximum concurrent inference sessions on this tile. */
    uint8_t             max_sessions;

    /** Per-tile error code (att1_aimu_result from att1_aimu_cmdq.h). */
    uint8_t             error_code;

    /** Memory currently allocated (bytes); updated by reset. */
    uint64_t            memory_used_bytes;
    /** KV memory currently used (bytes); updated by reset. */
    uint64_t            kv_used_bytes;

    /** Per-tile reset counter (incremented each RESET_TILE). */
    uint32_t            reset_count;
} att1_aimu_tile_info;

/* -------------------------------------------------------------------------
 * Device version   (M104 §2: DEVICE_VERSION)
 * ---------------------------------------------------------------------- */

typedef struct att1_aimu_device_version {
    uint8_t     major;
    uint8_t     minor;
    uint8_t     patch;
    uint8_t     build;
} att1_aimu_device_version;

/* -------------------------------------------------------------------------
 * Device info snapshot   (flat view returned by att1_aimu_device_query_info)
 * ---------------------------------------------------------------------- */

typedef struct att1_aimu_device_info {
    uint32_t                    register_map_version;  /**< M104 REGISTER_MAP_VERSION */
    att1_aimu_device_version    version;               /**< M104 DEVICE_VERSION       */
    uint64_t                    feature_flags;         /**< ATT1_AIMU_FEAT_* bitmask  */
    size_t                      tile_count;
    uint32_t                    global_status;         /**< 0 = OK, non-zero = error  */
    uint32_t                    global_error;          /**< error detail bits         */
} att1_aimu_device_info;

/* -------------------------------------------------------------------------
 * Device configuration   (passed to att1_aimu_device_create)
 * ---------------------------------------------------------------------- */

typedef struct att1_aimu_device_config {
    /** Number of tiles to simulate (1–ATT1_AIMU_DEVICE_MAX_TILES). */
    size_t      tile_count;

    /**
     * Per-tile model-weight memory (bytes).  Applied uniformly to all tiles.
     * 0 → default (1 GiB).
     */
    uint64_t    tile_memory_bytes;

    /**
     * Per-tile KV-cache memory (bytes).  Applied uniformly to all tiles.
     * 0 → default (256 MiB).
     */
    uint64_t    tile_kv_bytes;

    /** ATT1_AIMU_DTYPE_* bitmask.  0 → ATT1_AIMU_DTYPE_ALL. */
    uint32_t    supported_dtypes;

    /** ATT1_AIMU_OP_* bitmask.  0 → ATT1_AIMU_OP_ALL. */
    uint32_t    supported_ops;

    /** ATT1_AIMU_FEAT_* bitmask.  0 → ATT1_AIMU_DEVICE_DEFAULT_FEATURES. */
    uint64_t    feature_flags;

    /** Device version.  All-zero → use default (0.1.0.0). */
    att1_aimu_device_version version;
} att1_aimu_device_config;

/* -------------------------------------------------------------------------
 * Device object
 * ---------------------------------------------------------------------- */

typedef struct att1_aimu_device {
    uint32_t                    magic;
    uint32_t                    register_map_version;
    att1_aimu_device_version    version;
    uint64_t                    feature_flags;

    size_t                      tile_count;
    att1_aimu_tile_info         tiles[ATT1_AIMU_DEVICE_MAX_TILES];

    /** 0 = device OK; non-zero bits indicate global error conditions. */
    uint32_t                    global_status;

    /**
     * Error detail bits (last error recorded by reset or status check).
     * Cleared by att1_aimu_device_reset().
     */
    uint32_t                    global_error;

    /** Total device-level reset count. */
    uint32_t                    reset_count;

    /**
     * Optional attached command queue.  NULL if not attached.
     * The device does NOT own the queue; callers manage its lifetime.
     */
    att1_aimu_cmdq             *cmdq;
} att1_aimu_device;

/* -------------------------------------------------------------------------
 * Lifecycle
 * ---------------------------------------------------------------------- */

/**
 * att1_aimu_device_create
 *
 * Allocate and initialise a simulated AIMU device.  Pass NULL for config to
 * use defaults (1 tile, 1 GiB model memory, 256 MiB KV memory, all dtypes,
 * all ops, default features, version 0.1.0.0).
 *
 * Returns ATT1_OK and writes *out on success.
 * Returns ATT1_ERR_INVALID_ARG if out is NULL or config has invalid fields.
 * Returns ATT1_ERR_OOM if allocation fails.
 */
att1_status_t att1_aimu_device_create(const att1_aimu_device_config *config,
                                       att1_aimu_device             **out);

/**
 * att1_aimu_device_destroy
 *
 * Free all resources held by the simulated device.  Safe to call with NULL.
 * Does NOT destroy any attached command queue.
 */
void att1_aimu_device_destroy(att1_aimu_device *dev);

/* -------------------------------------------------------------------------
 * Device-level queries
 * ---------------------------------------------------------------------- */

/**
 * att1_aimu_device_query_info
 *
 * Fill *info with a snapshot of device-level registers.
 * Returns ATT1_ERR_INVALID_ARG if dev or info is NULL.
 */
att1_status_t att1_aimu_device_query_info(const att1_aimu_device   *dev,
                                           att1_aimu_device_info    *info);

/**
 * att1_aimu_device_tile_count
 *
 * Return the number of simulated tiles.  Returns 0 on NULL input.
 */
size_t att1_aimu_device_tile_count(const att1_aimu_device *dev);

/* -------------------------------------------------------------------------
 * Tile-level queries
 * ---------------------------------------------------------------------- */

/**
 * att1_aimu_device_query_tile
 *
 * Fill *info with a snapshot of per-tile registers for tile tile_id.
 * Returns ATT1_ERR_INVALID_ARG if dev or info is NULL or tile_id is
 * out of range.
 */
att1_status_t att1_aimu_device_query_tile(const att1_aimu_device *dev,
                                           uint8_t                 tile_id,
                                           att1_aimu_tile_info    *info);

/**
 * att1_aimu_device_validate_tile_id
 *
 * Return ATT1_OK if tile_id < dev->tile_count, ATT1_ERR_INVALID_ARG
 * otherwise.  Useful as a quick pre-flight check before submission.
 */
att1_status_t att1_aimu_device_validate_tile_id(const att1_aimu_device *dev,
                                                 uint8_t                 tile_id);

/**
 * att1_aimu_device_supports_dtype
 *
 * Return 1 if the device (all tiles) supports the given dtype bit
 * (ATT1_AIMU_DTYPE_*), 0 otherwise.  Returns 0 on NULL device.
 */
int att1_aimu_device_supports_dtype(const att1_aimu_device *dev,
                                     uint32_t                dtype_bit);

/**
 * att1_aimu_device_supports_op
 *
 * Return 1 if the device (all tiles) supports the given op bit
 * (ATT1_AIMU_OP_*), 0 otherwise.  Returns 0 on NULL device.
 */
int att1_aimu_device_supports_op(const att1_aimu_device *dev,
                                  uint32_t                op_bit);

/**
 * att1_aimu_device_tile_supports_dtype
 *
 * Return 1 if the specific tile supports the given dtype bit.
 * Returns 0 on NULL device or out-of-range tile_id.
 */
int att1_aimu_device_tile_supports_dtype(const att1_aimu_device *dev,
                                          uint8_t                 tile_id,
                                          uint32_t                dtype_bit);

/**
 * att1_aimu_device_tile_supports_op
 *
 * Return 1 if the specific tile supports the given op bit.
 * Returns 0 on NULL device or out-of-range tile_id.
 */
int att1_aimu_device_tile_supports_op(const att1_aimu_device *dev,
                                       uint8_t                 tile_id,
                                       uint32_t                op_bit);

/* -------------------------------------------------------------------------
 * Reset
 * ---------------------------------------------------------------------- */

/**
 * att1_aimu_device_reset
 *
 * Reset the device: clear global_status, global_error, and all per-tile
 * used-bytes fields and error codes.  Increments dev->reset_count.
 * Does NOT free or re-initialise the attached command queue.
 *
 * Returns ATT1_ERR_INVALID_ARG if dev is NULL.
 */
att1_status_t att1_aimu_device_reset(att1_aimu_device *dev);

/**
 * att1_aimu_device_reset_tile
 *
 * Reset a single tile: clear that tile's memory_used_bytes, kv_used_bytes,
 * error_code, and set state to IDLE.  Increments tile->reset_count.
 * Does NOT affect other tiles or device-level fields.
 *
 * Returns ATT1_ERR_INVALID_ARG if dev is NULL or tile_id out of range.
 */
att1_status_t att1_aimu_device_reset_tile(att1_aimu_device *dev,
                                           uint8_t           tile_id);

/* -------------------------------------------------------------------------
 * Counter snapshot
 * ---------------------------------------------------------------------- */

/**
 * att1_aimu_device_snapshot_counters
 *
 * Copy command-queue counters from the attached cmdq into *out.
 * Returns ATT1_ERR_STATE if no cmdq is attached (dev->cmdq == NULL).
 * Returns ATT1_ERR_INVALID_ARG if dev or out is NULL.
 */
att1_status_t att1_aimu_device_snapshot_counters(
        const att1_aimu_device       *dev,
        att1_aimu_cmdq_counters      *out);

/* -------------------------------------------------------------------------
 * Command-queue attachment
 * ---------------------------------------------------------------------- */

/**
 * att1_aimu_device_attach_cmdq
 *
 * Associate a command queue with this device.  The device does NOT take
 * ownership; the caller must ensure the queue outlives the device (or
 * detach before destroy).  Pass NULL to detach.
 *
 * Returns ATT1_ERR_INVALID_ARG if dev is NULL.
 */
att1_status_t att1_aimu_device_attach_cmdq(att1_aimu_device *dev,
                                            att1_aimu_cmdq   *q);

/* -------------------------------------------------------------------------
 * Name helpers
 * ---------------------------------------------------------------------- */

/**
 * att1_aimu_dtype_name
 *
 * Return a stable ASCII name for a dtype bit, e.g. "F32", "Q8", "Q4".
 * Returns "UNKNOWN" for unrecognised values.
 */
const char *att1_aimu_dtype_name(uint32_t dtype_bit);

/**
 * att1_aimu_op_name
 *
 * Return a stable ASCII name for an op bit, e.g. "MATMUL", "ATTENTION".
 * Returns "UNKNOWN" for unrecognised values.
 */
const char *att1_aimu_op_name(uint32_t op_bit);

/**
 * att1_aimu_feat_name
 *
 * Return a stable ASCII name for a feature flag bit, e.g. "CMD_RING".
 * Returns "UNKNOWN" for unrecognised values.
 */
const char *att1_aimu_feat_name(uint64_t feat_bit);

/**
 * att1_aimu_tile_state_name
 *
 * Return a stable ASCII name for a tile state, e.g. "IDLE", "ACTIVE".
 * Returns "UNKNOWN" for unrecognised values.
 */
const char *att1_aimu_tile_state_name(att1_aimu_tile_state state);

#ifdef __cplusplus
}
#endif

#endif /* ATT1_AIMU_DEVICE_H */
