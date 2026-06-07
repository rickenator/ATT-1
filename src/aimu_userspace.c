/*
 * aimu_userspace.c  —  AIMU userspace MMIO emulator workflow (M121)
 *
 * Implements att1_aimu_userspace: wraps the M112 att1_aimu_host harness and
 * optionally exposes the 64 KiB BAR0 register window through a mmap-backed
 * file so external tools can observe register state.
 *
 * Design notes
 * ------------
 *   - All register reads/writes route through att1_aimu_mmio_read32() /
 *     att1_aimu_mmio_write32() — NOT through raw mmap buffer mutation.
 *   - The mmap'd file is a read-only snapshot updated by flush_bar0() after
 *     every state-changing operation.
 *   - Tile memory capacity is stored in TILE_MEMORY_CAPACITY_* registers
 *     only; no buffer of that size is ever malloc'd.
 *   - DMA descriptors are validated only; no huge data buffers move.
 *   - No inference, CUDA, tokenizer, or .att1 binary format behaviour is
 *     changed by this module.
 */

/* Enable POSIX.1-2001 for mmap, ftruncate, msync */
#define _POSIX_C_SOURCE 200112L

#include "att1_aimu_userspace.h"
#include "att1_aimu_mmio.h"
#include "att1_aimu_host.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* POSIX headers for mmap */
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* =========================================================================
 * Internal helpers
 * ====================================================================== */

/*
 * flush_bar0
 *
 * Copy the current MMIO backing array to the mmap'd file buffer and msync.
 * No-op if bar0_map is NULL.
 */
static void flush_bar0(att1_aimu_userspace *u)
{
    if (!u || !u->bar0_map || u->bar0_map == MAP_FAILED) return;
    (void)att1_aimu_mmio_sync(u->host->mmio);
    memcpy(u->bar0_map, u->host->mmio->regs, ATT1_AIMU_MMIO_BAR0_SIZE);
    (void)msync(u->bar0_map, ATT1_AIMU_MMIO_BAR0_SIZE, MS_SYNC);
}

/* =========================================================================
 * Lifecycle
 * ====================================================================== */

att1_status_t att1_aimu_userspace_open(
        const att1_aimu_userspace_config *config,
        const char                       *bar0_path,
        att1_aimu_userspace             **out)
{
    if (!out) return ATT1_ERR_INVALID_ARG;

    /* Apply defaults */
    att1_aimu_userspace_config cfg;
    if (config) {
        cfg = *config;
    } else {
        memset(&cfg, 0, sizeof(cfg));
    }
    if (cfg.tile_count == 0) cfg.tile_count = ATT1_AIMU_USERSPACE_DEFAULT_TILES;
    if (cfg.tile_memory_bytes == 0)
        cfg.tile_memory_bytes = ATT1_AIMU_USERSPACE_DEFAULT_TILE_MEM_BYTES;
    if (cfg.tile_kv_bytes == 0)
        cfg.tile_kv_bytes = ATT1_AIMU_USERSPACE_DEFAULT_KV_MEM_BYTES;

    /* Validate */
    if (cfg.tile_count > 16u) return ATT1_ERR_INVALID_ARG;
    if (cfg.tile_memory_bytes > ATT1_AIMU_USERSPACE_MAX_TILE_MEM_BYTES)
        return ATT1_ERR_INVALID_ARG;

    att1_aimu_userspace *u = calloc(1, sizeof(att1_aimu_userspace));
    if (!u) return ATT1_ERR_OOM;

    u->magic    = ATT1_AIMU_USERSPACE_MAGIC;
    u->bar0_fd  = -1;
    u->bar0_map = NULL;

    /* Build M112 host config */
    att1_aimu_host_config hcfg;
    memset(&hcfg, 0, sizeof(hcfg));
    hcfg.tile_count        = cfg.tile_count;
    hcfg.tile_memory_bytes = cfg.tile_memory_bytes;
    hcfg.tile_kv_bytes     = cfg.tile_kv_bytes;
    hcfg.supported_dtypes  = cfg.supported_dtypes;
    hcfg.supported_ops     = cfg.supported_ops;
    hcfg.cmd_ring_depth    = cfg.cmd_ring_depth;

    att1_status_t rc = att1_aimu_host_create(&hcfg, &u->host);
    if (rc != ATT1_OK) {
        free(u);
        return rc;
    }

    /* Open mmap'd BAR0 file if requested */
    if (bar0_path && bar0_path[0] != '\0') {
        size_t plen = strlen(bar0_path);
        if (plen >= ATT1_AIMU_USERSPACE_PATH_MAX) {
            att1_aimu_host_destroy(u->host);
            free(u);
            return ATT1_ERR_INVALID_ARG;
        }
        memcpy(u->bar0_path, bar0_path, plen + 1u);

        u->bar0_fd = open(bar0_path, O_RDWR | O_CREAT, (mode_t)0600);
        if (u->bar0_fd < 0) {
            att1_aimu_host_destroy(u->host);
            free(u);
            return ATT1_ERR_INVALID_ARG;
        }
        if (ftruncate(u->bar0_fd, (off_t)ATT1_AIMU_MMIO_BAR0_SIZE) != 0) {
            close(u->bar0_fd);
            att1_aimu_host_destroy(u->host);
            free(u);
            return ATT1_ERR_INVALID_ARG;
        }
        u->bar0_map = mmap(NULL, ATT1_AIMU_MMIO_BAR0_SIZE,
                           PROT_READ | PROT_WRITE, MAP_SHARED,
                           u->bar0_fd, (off_t)0);
        if (u->bar0_map == MAP_FAILED) {
            u->bar0_map = NULL;
            close(u->bar0_fd);
            u->bar0_fd = -1;
            att1_aimu_host_destroy(u->host);
            free(u);
            return ATT1_ERR_INVALID_ARG;
        }
    }

    /* Flush initial register state to the mmap'd file */
    (void)att1_aimu_mmio_sync(u->host->mmio);
    flush_bar0(u);

    *out = u;
    return ATT1_OK;
}

void att1_aimu_userspace_close(att1_aimu_userspace *u)
{
    if (!u) return;
    if (u->magic != ATT1_AIMU_USERSPACE_MAGIC) return;

    if (u->bar0_map && u->bar0_map != MAP_FAILED) {
        flush_bar0(u);
        munmap(u->bar0_map, ATT1_AIMU_MMIO_BAR0_SIZE);
        u->bar0_map = NULL;
    }
    if (u->bar0_fd >= 0) {
        close(u->bar0_fd);
        u->bar0_fd = -1;
    }
    att1_aimu_host_destroy(u->host);
    u->host = NULL;
    u->magic = 0u;
    free(u);
}

/* =========================================================================
 * Control-plane flow
 * ====================================================================== */

att1_status_t att1_aimu_userspace_probe(
        att1_aimu_userspace         *u,
        att1_aimu_host_probe_result *out)
{
    if (!u || u->magic != ATT1_AIMU_USERSPACE_MAGIC) return ATT1_ERR_INVALID_ARG;

    att1_aimu_host_probe_result tmp;
    att1_status_t rc = att1_aimu_host_probe_device(u->host, &tmp);
    if (rc != ATT1_OK) return rc;

    u->probed = 1;
    if (out) *out = tmp;
    flush_bar0(u);
    return ATT1_OK;
}

att1_status_t att1_aimu_userspace_enumerate_tiles(
        att1_aimu_userspace      *u,
        att1_aimu_host_tile_info *infos,
        size_t                   *count)
{
    if (!u || u->magic != ATT1_AIMU_USERSPACE_MAGIC) return ATT1_ERR_INVALID_ARG;
    if (!infos || !count) return ATT1_ERR_INVALID_ARG;

    att1_status_t rc = att1_aimu_host_enumerate_tiles(u->host, infos, count);
    if (rc != ATT1_OK) return rc;

    flush_bar0(u);
    return ATT1_OK;
}

att1_status_t att1_aimu_userspace_setup_cmdq(att1_aimu_userspace *u)
{
    if (!u || u->magic != ATT1_AIMU_USERSPACE_MAGIC) return ATT1_ERR_INVALID_ARG;

    att1_status_t rc = att1_aimu_host_setup_cmdq(u->host);
    if (rc != ATT1_OK) return rc;

    u->cmdq_ready = 1;
    flush_bar0(u);
    return ATT1_OK;
}

att1_status_t att1_aimu_userspace_read32(att1_aimu_userspace *u,
                                          uint32_t             offset,
                                          uint32_t            *out)
{
    if (!u || u->magic != ATT1_AIMU_USERSPACE_MAGIC) return ATT1_ERR_INVALID_ARG;
    if (!out) return ATT1_ERR_INVALID_ARG;
    return att1_aimu_mmio_read32(u->host->mmio, offset, out);
}

att1_status_t att1_aimu_userspace_write32(att1_aimu_userspace *u,
                                           uint32_t             offset,
                                           uint32_t             value)
{
    if (!u || u->magic != ATT1_AIMU_USERSPACE_MAGIC) return ATT1_ERR_INVALID_ARG;

    att1_status_t rc = att1_aimu_mmio_write32(u->host->mmio, offset, value);
    if (rc != ATT1_OK) return rc;

    flush_bar0(u);
    return ATT1_OK;
}

att1_status_t att1_aimu_userspace_validate_dma(
        att1_aimu_userspace      *u,
        const att1_aimu_dma_desc *desc)
{
    if (!u || u->magic != ATT1_AIMU_USERSPACE_MAGIC) return ATT1_ERR_INVALID_ARG;
    if (!desc) return ATT1_ERR_INVALID_ARG;
    return att1_aimu_host_validate_dma(u->host, desc);
}

att1_status_t att1_aimu_userspace_submit_cmd(att1_aimu_userspace *u,
                                              att1_aimu_cmd       *cmd)
{
    if (!u || u->magic != ATT1_AIMU_USERSPACE_MAGIC) return ATT1_ERR_INVALID_ARG;
    if (!cmd) return ATT1_ERR_INVALID_ARG;

    att1_status_t rc = att1_aimu_host_submit_cmd(u->host, cmd);
    if (rc != ATT1_OK) return rc;

    flush_bar0(u);
    return ATT1_OK;
}

att1_status_t att1_aimu_userspace_ring_doorbell(att1_aimu_userspace *u)
{
    if (!u || u->magic != ATT1_AIMU_USERSPACE_MAGIC) return ATT1_ERR_INVALID_ARG;

    att1_status_t rc = att1_aimu_host_ring_doorbell(u->host);
    if (rc != ATT1_OK) return rc;

    flush_bar0(u);
    return ATT1_OK;
}

att1_status_t att1_aimu_userspace_drain(att1_aimu_userspace *u)
{
    if (!u || u->magic != ATT1_AIMU_USERSPACE_MAGIC) return ATT1_ERR_INVALID_ARG;

    att1_status_t rc = att1_aimu_host_drain(u->host);
    if (rc != ATT1_OK) return rc;

    flush_bar0(u);
    return ATT1_OK;
}

att1_status_t att1_aimu_userspace_snapshot(att1_aimu_userspace *u)
{
    if (!u || u->magic != ATT1_AIMU_USERSPACE_MAGIC) return ATT1_ERR_INVALID_ARG;

    att1_status_t rc = att1_aimu_host_snapshot_counters(u->host);
    if (rc != ATT1_OK) return rc;

    flush_bar0(u);
    return ATT1_OK;
}

att1_status_t att1_aimu_userspace_get_summary(att1_aimu_userspace    *u,
                                               att1_aimu_host_summary *out)
{
    if (!u || u->magic != ATT1_AIMU_USERSPACE_MAGIC) return ATT1_ERR_INVALID_ARG;
    if (!out) return ATT1_ERR_INVALID_ARG;
    return att1_aimu_host_get_summary(u->host, out);
}

/* =========================================================================
 * Utility
 * ====================================================================== */

att1_status_t att1_aimu_userspace_flush_bar0(att1_aimu_userspace *u)
{
    if (!u || u->magic != ATT1_AIMU_USERSPACE_MAGIC) return ATT1_ERR_INVALID_ARG;
    flush_bar0(u);
    return ATT1_OK;
}

att1_status_t att1_aimu_userspace_print_summary(const att1_aimu_userspace *u,
                                                 FILE                      *fp)
{
    if (!u || !fp) return ATT1_ERR_INVALID_ARG;

    att1_aimu_host_summary sum;
    memset(&sum, 0, sizeof(sum));
    (void)att1_aimu_host_get_summary(u->host, &sum);

    fprintf(fp, "device_id=0x%08X\n",          sum.device_id);
    fprintf(fp, "register_map_version=0x%08X\n", sum.register_map_version);
    fprintf(fp, "tile_count=%zu\n",              sum.tile_count);
    fprintf(fp, "commands_submitted=%" PRIu64 "\n", sum.commands_submitted);
    fprintf(fp, "commands_completed=%" PRIu64 "\n", sum.commands_completed);
    fprintf(fp, "commands_failed=%" PRIu64 "\n",    sum.commands_failed);
    fprintf(fp, "dma_submitted=%" PRIu64 "\n",      sum.dma_submitted);
    fprintf(fp, "dma_completed=%" PRIu64 "\n",      sum.dma_completed);
    fprintf(fp, "doorbell_count=%u\n",           sum.doorbell_count);
    fprintf(fp, "fence_value=%" PRIu64 "\n",         sum.fence_value);

    const char *status_str = "pass";
    if (sum.status != ATT1_AIMU_OK && sum.status != ATT1_AIMU_PENDING) status_str = "fail";
    if (sum.commands_failed > 0u) status_str = "fail";
    fprintf(fp, "status=%s\n", status_str);

    return ATT1_OK;
}
