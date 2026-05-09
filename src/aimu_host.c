/*
 * aimu_host.c  —  AIMU control-plane integration harness (M112)
 *
 * Wires M105 (cmdq) + M106 (device) + M107 (DMA) + M108 (trace) +
 * M111 (MMIO) into one deterministic in-process host-to-AIMU flow.
 */

#include "att1_aimu_host.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

/* =========================================================================
 * Internal defaults
 * ====================================================================== */

#define DEFAULT_TILE_COUNT      4
#define DEFAULT_CMD_RING_DEPTH  64
#define DEFAULT_COMP_RING_DEPTH 64

/* =========================================================================
 * Lifecycle
 * ====================================================================== */

att1_status_t
att1_aimu_host_create(const att1_aimu_host_config *config,
                      att1_aimu_host             **out)
{
    if (!out) {
        return ATT1_ERR_INVALID_ARG;
    }

    /* Resolve configuration ------------------------------------------------ */
    att1_aimu_host_config cfg;
    if (config) {
        cfg = *config;
    } else {
        memset(&cfg, 0, sizeof(cfg));
    }
    if (cfg.tile_count == 0) {
        cfg.tile_count = DEFAULT_TILE_COUNT;
    }
    if (cfg.cmd_ring_depth == 0) {
        cfg.cmd_ring_depth = DEFAULT_CMD_RING_DEPTH;
    }

    /* Allocate host context ------------------------------------------------ */
    att1_aimu_host *h = calloc(1, sizeof(*h));
    if (!h) {
        return ATT1_ERR_OOM;
    }

    /* Create device -------------------------------------------------------- */
    att1_aimu_device_config dev_cfg;
    memset(&dev_cfg, 0, sizeof(dev_cfg));
    dev_cfg.tile_count        = cfg.tile_count;
    dev_cfg.tile_memory_bytes = cfg.tile_memory_bytes;
    dev_cfg.tile_kv_bytes     = cfg.tile_kv_bytes;
    dev_cfg.supported_dtypes  = cfg.supported_dtypes;
    dev_cfg.supported_ops     = cfg.supported_ops;

    att1_status_t s = att1_aimu_device_create(&dev_cfg, &h->device);
    if (s != ATT1_OK) {
        free(h);
        return s;
    }

    /* Create cmdq ---------------------------------------------------------- */
    att1_aimu_cmdq_config q_cfg;
    memset(&q_cfg, 0, sizeof(q_cfg));
    q_cfg.tile_count       = cfg.tile_count;
    q_cfg.cmd_ring_depth   = cfg.cmd_ring_depth;
    q_cfg.comp_ring_depth  = cfg.cmd_ring_depth; /* keep symmetric */

    s = att1_aimu_cmdq_create(&q_cfg, &h->cmdq);
    if (s != ATT1_OK) {
        att1_aimu_device_destroy(h->device);
        free(h);
        return s;
    }

    /* Create DMA simulator ------------------------------------------------- */
    s = att1_aimu_dma_create(&h->dma);
    if (s != ATT1_OK) {
        att1_aimu_cmdq_destroy(h->cmdq);
        att1_aimu_device_destroy(h->device);
        free(h);
        return s;
    }

    /* Create trace --------------------------------------------------------- */
    s = att1_aimu_trace_create(&h->trace);
    if (s != ATT1_OK) {
        att1_aimu_dma_destroy(h->dma);
        att1_aimu_cmdq_destroy(h->cmdq);
        att1_aimu_device_destroy(h->device);
        free(h);
        return s;
    }

    /* Create MMIO ---------------------------------------------------------- */
    s = att1_aimu_mmio_create(&h->mmio);
    if (s != ATT1_OK) {
        att1_aimu_trace_destroy(h->trace);
        att1_aimu_dma_destroy(h->dma);
        att1_aimu_cmdq_destroy(h->cmdq);
        att1_aimu_device_destroy(h->device);
        free(h);
        return s;
    }

    /* Wire MMIO to device and trace (cmdq attached lazily in setup_cmdq) --- */
    s = att1_aimu_mmio_attach_device(h->mmio, h->device);
    if (s != ATT1_OK) {
        att1_aimu_mmio_destroy(h->mmio);
        att1_aimu_trace_destroy(h->trace);
        att1_aimu_dma_destroy(h->dma);
        att1_aimu_cmdq_destroy(h->cmdq);
        att1_aimu_device_destroy(h->device);
        free(h);
        return s;
    }

    s = att1_aimu_mmio_attach_trace(h->mmio, h->trace);
    if (s != ATT1_OK) {
        att1_aimu_mmio_destroy(h->mmio);
        att1_aimu_trace_destroy(h->trace);
        att1_aimu_dma_destroy(h->dma);
        att1_aimu_cmdq_destroy(h->cmdq);
        att1_aimu_device_destroy(h->device);
        free(h);
        return s;
    }

    /* Initial MMIO sync ---------------------------------------------------- */
    att1_aimu_mmio_sync(h->mmio);

    h->magic      = ATT1_AIMU_HOST_MAGIC;
    h->last_error = ATT1_AIMU_OK;
    h->probed     = 0;
    h->cmdq_ready = 0;

    *out = h;
    return ATT1_OK;
}

void
att1_aimu_host_destroy(att1_aimu_host *h)
{
    if (!h) {
        return;
    }
    att1_aimu_mmio_destroy(h->mmio);
    att1_aimu_trace_destroy(h->trace);
    att1_aimu_dma_destroy(h->dma);
    att1_aimu_cmdq_destroy(h->cmdq);
    att1_aimu_device_destroy(h->device);
    h->magic = 0;
    free(h);
}

/* =========================================================================
 * Control-plane flow
 * ====================================================================== */

att1_status_t
att1_aimu_host_probe_device(att1_aimu_host             *h,
                             att1_aimu_host_probe_result *out)
{
    if (!h) {
        return ATT1_ERR_INVALID_ARG;
    }

    /* Sync from all attached sub-simulators */
    att1_aimu_mmio_sync(h->mmio);

    /* Read identity registers */
    uint32_t device_id = 0;
    att1_status_t s = att1_aimu_mmio_read32(h->mmio,
                                             ATT1_MMIO_DEVICE_ID, &device_id);
    if (s != ATT1_OK) {
        return s;
    }

    uint32_t reg_map_ver = 0;
    s = att1_aimu_mmio_read32(h->mmio,
                               ATT1_MMIO_REGISTER_MAP_VERSION, &reg_map_ver);
    if (s != ATT1_OK) {
        return s;
    }

    uint32_t tile_count_reg = 0;
    s = att1_aimu_mmio_read32(h->mmio, ATT1_MMIO_TILE_COUNT, &tile_count_reg);
    if (s != ATT1_OK) {
        return s;
    }

    uint32_t global_status = 0;
    s = att1_aimu_mmio_read32(h->mmio,
                               ATT1_MMIO_GLOBAL_STATUS, &global_status);
    if (s != ATT1_OK) {
        return s;
    }

    /* Check DEVICE_READY */
    if (!(global_status & ATT1_MMIO_GSTAT_DEVICE_READY)) {
        return ATT1_ERR_STATE;
    }

    h->probe.device_id            = device_id;
    h->probe.register_map_version = reg_map_ver;
    h->probe.global_status        = global_status;
    h->probe.tile_count           = (size_t)tile_count_reg;
    h->probed = 1;

    if (out) {
        *out = h->probe;
    }
    return ATT1_OK;
}

att1_status_t
att1_aimu_host_enumerate_tiles(att1_aimu_host           *h,
                                att1_aimu_host_tile_info *infos,
                                size_t                   *count)
{
    if (!h || !infos || !count) {
        return ATT1_ERR_INVALID_ARG;
    }
    if (!h->probed) {
        return ATT1_ERR_STATE;
    }

    size_t n = h->probe.tile_count;
    if (n > *count) {
        n = *count;
    }

    for (size_t i = 0; i < n; i++) {
        att1_aimu_tile_info ti;
        att1_status_t s = att1_aimu_device_query_tile(h->device,
                                                       (uint8_t)i, &ti);
        if (s != ATT1_OK) {
            *count = i;
            return s;
        }
        infos[i].tile_id               = ti.tile_id;
        infos[i].memory_capacity_bytes = ti.memory_capacity_bytes;
        infos[i].kv_capacity_bytes     = ti.kv_capacity_bytes;
        infos[i].supported_dtypes      = ti.supported_dtypes;
        infos[i].supported_ops         = ti.supported_ops;
        infos[i].state                 = (uint8_t)ti.state;
    }

    *count = n;
    return ATT1_OK;
}

att1_status_t
att1_aimu_host_setup_cmdq(att1_aimu_host *h)
{
    if (!h) {
        return ATT1_ERR_INVALID_ARG;
    }
    if (!h->probed) {
        return ATT1_ERR_STATE;
    }

    /* Attach cmdq to device */
    att1_status_t s = att1_aimu_device_attach_cmdq(h->device, h->cmdq);
    if (s != ATT1_OK) {
        return s;
    }

    /* Attach cmdq to MMIO */
    s = att1_aimu_mmio_attach_cmdq(h->mmio, h->cmdq);
    if (s != ATT1_OK) {
        return s;
    }

    /* Re-sync so CQ registers are populated */
    s = att1_aimu_mmio_sync(h->mmio);
    if (s != ATT1_OK) {
        return s;
    }

    h->cmdq_ready = 1;
    return ATT1_OK;
}

att1_status_t
att1_aimu_host_validate_dma(att1_aimu_host           *h,
                              const att1_aimu_dma_desc *desc)
{
    if (!h || !desc) {
        return ATT1_ERR_INVALID_ARG;
    }
    return att1_aimu_dma_validate(h->dma, desc);
}

att1_status_t
att1_aimu_host_submit_cmd(att1_aimu_host *h,
                           att1_aimu_cmd  *cmd)
{
    if (!h || !cmd) {
        return ATT1_ERR_INVALID_ARG;
    }
    if (!h->cmdq_ready) {
        return ATT1_ERR_STATE;
    }
    return att1_aimu_cmdq_submit(h->cmdq, cmd);
}

att1_status_t
att1_aimu_host_ring_doorbell(att1_aimu_host *h)
{
    if (!h) {
        return ATT1_ERR_INVALID_ARG;
    }
    if (!h->cmdq_ready) {
        return ATT1_ERR_STATE;
    }
    /* Write the current cmd_tail as the doorbell value */
    uint32_t tail = (uint32_t)h->cmdq->cmd_tail;
    return att1_aimu_mmio_write32(h->mmio, ATT1_MMIO_CQ_DOORBELL, tail);
}

att1_status_t
att1_aimu_host_process_one(att1_aimu_host *h)
{
    if (!h) {
        return ATT1_ERR_INVALID_ARG;
    }
    return att1_aimu_cmdq_dispatch_one(h->cmdq);
}

att1_status_t
att1_aimu_host_drain(att1_aimu_host *h)
{
    if (!h) {
        return ATT1_ERR_INVALID_ARG;
    }
    return att1_aimu_cmdq_dispatch_all(h->cmdq);
}

att1_status_t
att1_aimu_host_read_completion(att1_aimu_host       *h,
                                att1_aimu_completion *out)
{
    if (!h || !out) {
        return ATT1_ERR_INVALID_ARG;
    }
    return att1_aimu_cmdq_poll_completion(h->cmdq, out);
}

att1_status_t
att1_aimu_host_snapshot_counters(att1_aimu_host *h)
{
    if (!h) {
        return ATT1_ERR_INVALID_ARG;
    }

    /* Snapshot all available sub-simulators (DMA is always present) */
    att1_status_t s = att1_aimu_trace_snapshot_all(
            h->trace,
            h->cmdq_ready ? h->cmdq : NULL,
            h->device,
            h->dma);
    if (s != ATT1_OK) {
        return s;
    }

    /* Sync MMIO counters/trace registers from updated sources */
    return att1_aimu_mmio_sync(h->mmio);
}

att1_status_t
att1_aimu_host_get_summary(att1_aimu_host         *h,
                            att1_aimu_host_summary *out)
{
    if (!h || !out) {
        return ATT1_ERR_INVALID_ARG;
    }

    memset(out, 0, sizeof(*out));

    /* Device identity from cached probe */
    out->device_id            = h->probe.device_id;
    out->register_map_version = h->probe.register_map_version;
    out->tile_count           = h->probe.tile_count;

    /* CQ counters */
    if (h->cmdq_ready) {
        att1_aimu_cmdq_counters cq_cnt;
        att1_status_t s = att1_aimu_cmdq_get_counters(h->cmdq, &cq_cnt);
        if (s == ATT1_OK) {
            out->commands_submitted = cq_cnt.commands_submitted;
            out->commands_completed = cq_cnt.commands_completed;
            out->commands_failed    = cq_cnt.commands_failed;
            out->fence_value        = cq_cnt.fence_value;
        }
    }

    /* DMA counters */
    att1_aimu_dma_counters dma_cnt;
    att1_status_t s = att1_aimu_dma_get_counters(h->dma, &dma_cnt);
    if (s == ATT1_OK) {
        out->dma_submitted = dma_cnt.dma_submitted;
        out->dma_completed = dma_cnt.dma_completed;
        out->dma_failed    = dma_cnt.dma_failed;
    }

    /* MMIO interaction counters */
    out->doorbell_count         = h->mmio->doorbell_write_count;
    out->snapshot_trigger_count = h->mmio->snapshot_trigger_count;

    /* Trace metadata from last snapshot */
    att1_aimu_trace_snapshot snap;
    s = att1_aimu_trace_get_snapshot(h->trace, &snap);
    if (s == ATT1_OK) {
        out->trace_event_count  = snap.meta.event_count;
        out->trace_snapshot_id  = snap.meta.snapshot_id;
        out->trace_status       = snap.meta.status;
    }

    out->status = h->last_error;
    return ATT1_OK;
}

att1_status_t
att1_aimu_host_reset(att1_aimu_host *h)
{
    if (!h) {
        return ATT1_ERR_INVALID_ARG;
    }

    att1_status_t s;

    s = att1_aimu_device_reset(h->device);
    if (s != ATT1_OK) {
        return s;
    }

    s = att1_aimu_mmio_reset(h->mmio);
    if (s != ATT1_OK) {
        return s;
    }

    s = att1_aimu_trace_reset(h->trace);
    if (s != ATT1_OK) {
        return s;
    }

    s = att1_aimu_dma_reset_counters(h->dma);
    if (s != ATT1_OK) {
        return s;
    }

    /* Re-sync MMIO from freshly-reset sub-simulators */
    s = att1_aimu_mmio_sync(h->mmio);

    h->probed     = 0;
    h->cmdq_ready = 0;
    h->last_error = ATT1_AIMU_OK;
    memset(&h->probe, 0, sizeof(h->probe));

    return s;
}

att1_status_t
att1_aimu_host_render(const att1_aimu_host *h, FILE *fp)
{
    if (!h || !fp) {
        return ATT1_ERR_INVALID_ARG;
    }
    return att1_aimu_mmio_render(h->mmio, fp);
}
