/*
 * att1_aimu_userspace.h  —  AIMU userspace MMIO emulator workflow (M121)
 *
 * Exposes the M111 AIMU MMIO/register-file simulator through a userspace
 * workflow.  Host-side code can probe, enumerate tiles, submit commands,
 * drain completions, and snapshot counters through an external interface
 * without implementing a Linux kernel driver or real PCIe hardware.
 *
 * The mmap-backed BAR0 file
 * --------------------------
 * When att1_aimu_userspace_open() is called with a non-NULL bar0_path, the
 * implementation creates or opens the file, ftruncates it to 64 KiB, and
 * mmap's it shared.  After every state-changing operation the MMIO register
 * array is copied into the mmap'd buffer so the file reflects current state.
 *
 * Reads and writes through the API route through M111 MMIO simulator
 * semantics (RO enforcement, WO side-effects, RW1C clear-on-write) — NOT
 * raw unchecked memory mutation of the mmap buffer.
 *
 * If bar0_path is NULL the emulator runs entirely in memory without creating
 * or flushing any file (useful for tests that do not need external access).
 *
 * Memory guardrail
 * -----------------
 * Tile memory capacity is stored as metadata in TILE_MEMORY_CAPACITY_*
 * registers only.  No buffer of tile_memory_bytes size is allocated.
 * DMA descriptors are validated descriptor-only; no huge backing buffers
 * are needed.
 *
 * This is NOT a real PCIe endpoint, MMIO accessor, DMA engine, interrupt
 * controller, or kernel driver.  All state lives in owned M112
 * att1_aimu_host sub-simulators.
 */

#ifndef ATT1_AIMU_USERSPACE_H
#define ATT1_AIMU_USERSPACE_H

#include "att1_status.h"
#include "att1_aimu_host.h"   /* att1_aimu_host, att1_aimu_host_config,
                                  att1_aimu_host_summary,
                                  att1_aimu_host_tile_info */
#include "att1_aimu_cmdq.h"   /* att1_aimu_cmd, att1_aimu_cmd_type */
#include "att1_aimu_dma.h"    /* att1_aimu_dma_desc */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Constants
 * ====================================================================== */

#define ATT1_AIMU_USERSPACE_MAGIC       UINT32_C(0xA771A121)

/** Maximum bar0_path length including NUL terminator. */
#define ATT1_AIMU_USERSPACE_PATH_MAX    512u

/** Default tile count when config.tile_count == 0. */
#define ATT1_AIMU_USERSPACE_DEFAULT_TILES   4u

/** Default tile memory in bytes when config.tile_memory_bytes == 0.
 *  Stored as register metadata only; no buffer is allocated. */
#define ATT1_AIMU_USERSPACE_DEFAULT_TILE_MEM_BYTES  \
    (UINT64_C(32) * UINT64_C(1024) * UINT64_C(1024))   /* 32 MiB */

/** Default KV memory in bytes when config.tile_kv_bytes == 0.
 *  Stored as register metadata only; no buffer is allocated. */
#define ATT1_AIMU_USERSPACE_DEFAULT_KV_MEM_BYTES    \
    (UINT64_C(8) * UINT64_C(1024) * UINT64_C(1024))    /* 8 MiB */

/** Maximum allowed tile_memory_bytes to guard against accidental
 *  gigabyte-scale allocations.  The limit is checked in open(); the value
 *  is stored in registers only, never malloc'd. */
#define ATT1_AIMU_USERSPACE_MAX_TILE_MEM_BYTES  \
    (UINT64_C(256) * UINT64_C(1024) * UINT64_C(1024))  /* 256 MiB register limit */

/* =========================================================================
 * Configuration
 * ====================================================================== */

/**
 * att1_aimu_userspace_config
 *
 * Configuration passed to att1_aimu_userspace_open().
 * Fields that are zero use their defaults (documented per-field).
 */
typedef struct att1_aimu_userspace_config {
    /** Number of simulated AIMU tiles (1–16; 0 → default 4). */
    size_t   tile_count;

    /**
     * Per-tile model-weight memory capacity in bytes.
     * Stored as register metadata only; 0 → 32 MiB.
     * Values > ATT1_AIMU_USERSPACE_MAX_TILE_MEM_BYTES are rejected.
     */
    uint64_t tile_memory_bytes;

    /**
     * Per-tile KV-cache memory capacity in bytes.
     * Stored as register metadata only; 0 → 8 MiB.
     */
    uint64_t tile_kv_bytes;

    /** ATT1_AIMU_DTYPE_* bitmask (0 → all dtypes). */
    uint32_t supported_dtypes;

    /** ATT1_AIMU_OP_* bitmask (0 → all ops). */
    uint32_t supported_ops;

    /** Command ring depth (0 → default). */
    size_t   cmd_ring_depth;
} att1_aimu_userspace_config;

/* =========================================================================
 * Emulator object
 * ====================================================================== */

/**
 * att1_aimu_userspace
 *
 * Opaque handle for the userspace MMIO emulator.  Allocate with
 * att1_aimu_userspace_open(); free with att1_aimu_userspace_close().
 *
 * The struct is public so tests can inspect sub-fields, but callers should
 * treat it as opaque and use the API functions below.
 */
typedef struct att1_aimu_userspace {
    uint32_t            magic;

    /** Owned M112 host harness (contains device, cmdq, dma, trace, mmio). */
    att1_aimu_host     *host;

    /** File descriptor for the BAR0 mmap file; -1 if no file was opened. */
    int                 bar0_fd;

    /** mmap'd buffer; NULL if no file was opened or mmap failed. */
    void               *bar0_map;

    /** Path passed to open(); empty string if no file. */
    char                bar0_path[ATT1_AIMU_USERSPACE_PATH_MAX];

    /** Set to 1 after att1_aimu_userspace_probe() succeeds. */
    int                 probed;

    /** Set to 1 after att1_aimu_userspace_setup_cmdq() succeeds. */
    int                 cmdq_ready;
} att1_aimu_userspace;

/* =========================================================================
 * Lifecycle
 * ====================================================================== */

/**
 * att1_aimu_userspace_open
 *
 * Create the userspace AIMU emulator:
 *   1. Validate config (tile_count, tile_memory_bytes).
 *   2. Create the M112 att1_aimu_host with the given config.
 *   3. If bar0_path is non-NULL: create/open the file, ftruncate to 64 KiB,
 *      mmap MAP_SHARED, and flush initial register state to the file.
 *
 * Does NOT call probe/enumerate/setup_cmdq automatically.  Call those
 * separately to follow the explicit step sequence.
 *
 * @param config     Configuration; NULL → all defaults.
 * @param bar0_path  Path for the mmap'd BAR0 file; NULL → in-memory only.
 * @param out        Receives the allocated emulator handle on success.
 *
 * Returns ATT1_OK on success.
 * Returns ATT1_ERR_INVALID_ARG if out is NULL, tile_count > 16, or
 *   tile_memory_bytes > ATT1_AIMU_USERSPACE_MAX_TILE_MEM_BYTES.
 * Returns ATT1_ERR_OOM if any sub-allocation fails.
 * Returns ATT1_ERR_INVALID_ARG if the BAR0 file cannot be opened/mapped.
 */
att1_status_t att1_aimu_userspace_open(
        const att1_aimu_userspace_config *config,
        const char                       *bar0_path,
        att1_aimu_userspace             **out);

/**
 * att1_aimu_userspace_close
 *
 * Unmap the BAR0 file (if any), close the file descriptor, destroy the
 * owned M112 host harness, and free the emulator handle.
 * Safe to call with NULL.
 */
void att1_aimu_userspace_close(att1_aimu_userspace *u);

/* =========================================================================
 * Control-plane flow
 * ====================================================================== */

/**
 * att1_aimu_userspace_probe
 *
 * Probe the simulated device:
 *   - Synchronise MMIO register file from all attached simulators.
 *   - Read DEVICE_ID, REGISTER_MAP_VERSION, GLOBAL_STATUS, TILE_COUNT.
 *   - Store result in u->host->probe and set u->probed = 1.
 *   - Flush BAR0 file.
 *
 * @param out  Optional; receives the probe result.  May be NULL.
 *
 * Returns ATT1_ERR_INVALID_ARG if u is NULL.
 */
att1_status_t att1_aimu_userspace_probe(
        att1_aimu_userspace             *u,
        att1_aimu_host_probe_result     *out);

/**
 * att1_aimu_userspace_enumerate_tiles
 *
 * Enumerate per-tile registers for all tiles discovered during probe.
 * Writes up to *count entries into infos[]; updates *count with the actual
 * number written.  Flushes BAR0 file.
 *
 * Returns ATT1_ERR_STATE if probe has not been called.
 * Returns ATT1_ERR_INVALID_ARG if u, infos, or count is NULL.
 */
att1_status_t att1_aimu_userspace_enumerate_tiles(
        att1_aimu_userspace      *u,
        att1_aimu_host_tile_info *infos,
        size_t                   *count);

/**
 * att1_aimu_userspace_setup_cmdq
 *
 * Attach the command queue to the device and MMIO simulator.
 * Sets u->cmdq_ready = 1 on success.  Flushes BAR0 file.
 *
 * Returns ATT1_ERR_STATE if probe has not been called.
 * Returns ATT1_ERR_INVALID_ARG if u is NULL.
 */
att1_status_t att1_aimu_userspace_setup_cmdq(att1_aimu_userspace *u);

/**
 * att1_aimu_userspace_read32
 *
 * Read a 32-bit register value at BAR0 offset via M111 MMIO semantics.
 * WO registers read as 0; reserved offsets read as 0xDEADBEEF.
 *
 * Returns ATT1_ERR_INVALID_ARG if u or out is NULL, or offset is out of
 *   range or misaligned.
 */
att1_status_t att1_aimu_userspace_read32(att1_aimu_userspace *u,
                                          uint32_t             offset,
                                          uint32_t            *out);

/**
 * att1_aimu_userspace_write32
 *
 * Write a 32-bit value to BAR0 offset via M111 MMIO semantics (RO
 * enforcement, WO side-effects, RW1C clear-on-write).  Flushes BAR0 file
 * after a successful write.
 *
 * Returns ATT1_ERR_UNSUPPORTED if the register is read-only.
 * Returns ATT1_ERR_INVALID_ARG if u is NULL, or offset is out of range or
 *   misaligned.
 */
att1_status_t att1_aimu_userspace_write32(att1_aimu_userspace *u,
                                           uint32_t             offset,
                                           uint32_t             value);

/**
 * att1_aimu_userspace_validate_dma
 *
 * Validate a DMA descriptor through the M107 DMA simulator.
 * Does not submit; does not increment counters.
 *
 * Returns ATT1_ERR_INVALID_ARG if u or desc is NULL.
 */
att1_status_t att1_aimu_userspace_validate_dma(
        att1_aimu_userspace      *u,
        const att1_aimu_dma_desc *desc);

/**
 * att1_aimu_userspace_submit_cmd
 *
 * Submit a command to the owned M105 command queue.
 * Flushes BAR0 file after submission.
 *
 * Returns ATT1_ERR_STATE if cmdq is not ready.
 * Returns ATT1_ERR_INVALID_ARG if u or cmd is NULL.
 */
att1_status_t att1_aimu_userspace_submit_cmd(att1_aimu_userspace *u,
                                              att1_aimu_cmd       *cmd);

/**
 * att1_aimu_userspace_ring_doorbell
 *
 * Write ATT1_MMIO_CQ_DOORBELL via MMIO (triggers doorbell side-effect).
 * Flushes BAR0 file.
 *
 * Returns ATT1_ERR_STATE if cmdq is not ready.
 * Returns ATT1_ERR_INVALID_ARG if u is NULL.
 */
att1_status_t att1_aimu_userspace_ring_doorbell(att1_aimu_userspace *u);

/**
 * att1_aimu_userspace_drain
 *
 * Dispatch all pending commands (simulated AIMU side).
 * Flushes BAR0 file after drain.
 *
 * Returns ATT1_ERR_INVALID_ARG if u is NULL.
 */
att1_status_t att1_aimu_userspace_drain(att1_aimu_userspace *u);

/**
 * att1_aimu_userspace_snapshot
 *
 * Trigger a trace/counter snapshot via att1_aimu_host_snapshot_counters
 * and synchronise the MMIO register file.
 * Flushes BAR0 file.
 *
 * Returns ATT1_ERR_INVALID_ARG if u is NULL.
 */
att1_status_t att1_aimu_userspace_snapshot(att1_aimu_userspace *u);

/**
 * att1_aimu_userspace_get_summary
 *
 * Aggregate counters from MMIO, command queue, DMA, and trace into a flat
 * att1_aimu_host_summary struct.
 *
 * Returns ATT1_ERR_INVALID_ARG if u or out is NULL.
 */
att1_status_t att1_aimu_userspace_get_summary(att1_aimu_userspace    *u,
                                               att1_aimu_host_summary *out);

/* =========================================================================
 * Utility
 * ====================================================================== */

/**
 * att1_aimu_userspace_flush_bar0
 *
 * Copy the current MMIO register array into the mmap'd BAR0 buffer and
 * call msync().  No-op if no BAR0 file is open.
 *
 * Returns ATT1_OK always (msync errors are silently ignored in emulator
 * mode).
 */
att1_status_t att1_aimu_userspace_flush_bar0(att1_aimu_userspace *u);

/**
 * att1_aimu_userspace_print_summary
 *
 * Print a human-readable summary of the emulator state to fp.
 *
 * Returns ATT1_ERR_INVALID_ARG if u or fp is NULL.
 */
att1_status_t att1_aimu_userspace_print_summary(const att1_aimu_userspace *u,
                                                 FILE                      *fp);

#ifdef __cplusplus
}
#endif

#endif /* ATT1_AIMU_USERSPACE_H */
