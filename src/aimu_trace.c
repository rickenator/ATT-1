/*
 * aimu_trace.c  —  AIMU unified trace/counter snapshot (M108)
 *
 * Implements att1_aimu_trace_create / destroy, individual and combined
 * counter snapshot functions, reset, render, and name helpers.
 *
 * No inference, backend, tokenizer, or CUDA behaviour is touched.
 */

#include "att1_aimu_trace.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* -------------------------------------------------------------------------
 * Lifecycle
 * ---------------------------------------------------------------------- */

att1_status_t att1_aimu_trace_create(att1_aimu_trace **out)
{
    if (!out)
        return ATT1_ERR_INVALID_ARG;

    att1_aimu_trace *t = calloc(1u, sizeof(*t));
    if (!t)
        return ATT1_ERR_OOM;

    t->magic                        = ATT1_AIMU_TRACE_MAGIC;
    t->snapshot.meta.trace_version  = ATT1_AIMU_TRACE_VERSION;
    t->snapshot.meta.status         = ATT1_AIMU_TRACE_STATUS_EMPTY;

    *out = t;
    return ATT1_OK;
}

void att1_aimu_trace_destroy(att1_aimu_trace *t)
{
    if (!t)
        return;
    t->magic = 0u;
    free(t);
}

/* -------------------------------------------------------------------------
 * Individual counter snapshots
 * ---------------------------------------------------------------------- */

att1_status_t att1_aimu_trace_snapshot_cmdq(att1_aimu_trace      *t,
                                              const att1_aimu_cmdq *q)
{
    if (!t || !q)
        return ATT1_ERR_INVALID_ARG;

    const att1_aimu_cmdq_counters  *src = &q->counters;
    att1_aimu_trace_cmdq_counters  *dst = &t->snapshot.cmdq;

    dst->commands_submitted  = src->commands_submitted;
    dst->commands_completed  = src->commands_completed;
    dst->commands_failed     = src->commands_failed;
    dst->queue_full_count    = src->queue_full_count;
    dst->unsupported_commands = src->unsupported_commands;
    dst->fence_value         = src->fence_value;

    return ATT1_OK;
}

att1_status_t att1_aimu_trace_snapshot_device(att1_aimu_trace        *t,
                                               const att1_aimu_device *dev)
{
    if (!t || !dev)
        return ATT1_ERR_INVALID_ARG;

    att1_aimu_trace_device_counters *dst = &t->snapshot.device;

    dst->device_resets = (uint64_t)dev->reset_count;

    uint64_t tile_resets = 0u;
    uint64_t tile_errors = 0u;

    for (size_t i = 0u; i < dev->tile_count; i++) {
        tile_resets += (uint64_t)dev->tiles[i].reset_count;
        if (dev->tiles[i].state == ATT1_AIMU_TILE_ERROR)
            tile_errors++;
    }

    dst->tile_resets = tile_resets;
    dst->tile_errors = tile_errors;

    t->snapshot.meta.tile_count = (uint32_t)dev->tile_count;

    return ATT1_OK;
}

att1_status_t att1_aimu_trace_snapshot_dma(att1_aimu_trace     *t,
                                            const att1_aimu_dma *sim)
{
    if (!t || !sim)
        return ATT1_ERR_INVALID_ARG;

    const att1_aimu_dma_counters  *src = &sim->counters;
    att1_aimu_trace_dma_counters  *dst = &t->snapshot.dma;

    dst->dma_submitted            = src->dma_submitted;
    dst->dma_completed            = src->dma_completed;
    dst->dma_failed               = src->dma_failed;
    dst->bytes_host_to_device     = src->bytes_host_to_device;
    dst->bytes_device_to_host     = src->bytes_device_to_host;
    dst->bytes_device_to_device   = src->bytes_device_to_device;
    dst->alignment_failures       = src->alignment_failures;
    dst->range_failures           = src->range_failures;
    dst->unsupported_flags        = src->unsupported_flags;

    return ATT1_OK;
}

/* -------------------------------------------------------------------------
 * Combined snapshot
 * ---------------------------------------------------------------------- */

att1_status_t att1_aimu_trace_snapshot_all(att1_aimu_trace        *t,
                                            const att1_aimu_cmdq   *q,
                                            const att1_aimu_device *dev,
                                            const att1_aimu_dma    *dma)
{
    if (!t)
        return ATT1_ERR_INVALID_ARG;

    if (q)   att1_aimu_trace_snapshot_cmdq(t, q);
    if (dev) att1_aimu_trace_snapshot_device(t, dev);
    if (dma) att1_aimu_trace_snapshot_dma(t, dma);

    t->snapshot.meta.snapshot_id++;
    t->snapshot.meta.status =
        (q && dev && dma) ? ATT1_AIMU_TRACE_STATUS_OK
                          : ATT1_AIMU_TRACE_STATUS_PARTIAL;

    return ATT1_OK;
}

/* -------------------------------------------------------------------------
 * Accessors
 * ---------------------------------------------------------------------- */

att1_status_t att1_aimu_trace_get_snapshot(const att1_aimu_trace    *t,
                                            att1_aimu_trace_snapshot *out)
{
    if (!t || !out)
        return ATT1_ERR_INVALID_ARG;

    *out = t->snapshot;
    return ATT1_OK;
}

/* -------------------------------------------------------------------------
 * Reset
 * ---------------------------------------------------------------------- */

att1_status_t att1_aimu_trace_reset(att1_aimu_trace *t)
{
    if (!t)
        return ATT1_ERR_INVALID_ARG;

    memset(&t->snapshot, 0, sizeof(t->snapshot));
    t->snapshot.meta.trace_version = ATT1_AIMU_TRACE_VERSION;
    t->snapshot.meta.status        = ATT1_AIMU_TRACE_STATUS_EMPTY;

    return ATT1_OK;
}

/* -------------------------------------------------------------------------
 * Rendering
 * ---------------------------------------------------------------------- */

att1_status_t att1_aimu_trace_render(const att1_aimu_trace_snapshot *snap,
                                      FILE                           *f)
{
    if (!snap || !f)
        return ATT1_ERR_INVALID_ARG;

    const att1_aimu_trace_meta             *m = &snap->meta;
    const att1_aimu_trace_cmdq_counters    *c = &snap->cmdq;
    const att1_aimu_trace_device_counters  *d = &snap->device;
    const att1_aimu_trace_dma_counters     *a = &snap->dma;
    const att1_aimu_trace_fabric_counters  *b = &snap->fabric;

    fprintf(f, "--- AIMU Trace Snapshot ---\n");
    fprintf(f, "trace_version=0x%08X  snapshot_id=%u  device_id=%u"
               "  tile_count=%u\n",
            (unsigned)m->trace_version, (unsigned)m->snapshot_id,
            (unsigned)m->device_id,     (unsigned)m->tile_count);
    fprintf(f, "event_count=%llu  dropped_events=%llu  status=%s\n",
            (unsigned long long)m->event_count,
            (unsigned long long)m->dropped_events,
            att1_aimu_trace_status_name(m->status));

    fprintf(f, "[cmdq]\n");
    fprintf(f, "  commands_submitted=%llu  commands_completed=%llu"
               "  commands_failed=%llu\n",
            (unsigned long long)c->commands_submitted,
            (unsigned long long)c->commands_completed,
            (unsigned long long)c->commands_failed);
    fprintf(f, "  queue_full_count=%llu  unsupported_commands=%llu"
               "  fence_value=%llu\n",
            (unsigned long long)c->queue_full_count,
            (unsigned long long)c->unsupported_commands,
            (unsigned long long)c->fence_value);

    fprintf(f, "[device]\n");
    fprintf(f, "  device_resets=%llu  tile_resets=%llu  tile_errors=%llu\n",
            (unsigned long long)d->device_resets,
            (unsigned long long)d->tile_resets,
            (unsigned long long)d->tile_errors);

    fprintf(f, "[dma]\n");
    fprintf(f, "  dma_submitted=%llu  dma_completed=%llu  dma_failed=%llu\n",
            (unsigned long long)a->dma_submitted,
            (unsigned long long)a->dma_completed,
            (unsigned long long)a->dma_failed);
    fprintf(f, "  bytes_h2d=%llu  bytes_d2h=%llu  bytes_d2d=%llu\n",
            (unsigned long long)a->bytes_host_to_device,
            (unsigned long long)a->bytes_device_to_host,
            (unsigned long long)a->bytes_device_to_device);
    fprintf(f, "  alignment_failures=%llu  range_failures=%llu"
               "  unsupported_flags=%llu\n",
            (unsigned long long)a->alignment_failures,
            (unsigned long long)a->range_failures,
            (unsigned long long)a->unsupported_flags);

    fprintf(f, "[fabric] (placeholder)\n");
    fprintf(f, "  packets_sent=%llu  packets_received=%llu\n",
            (unsigned long long)b->packets_sent,
            (unsigned long long)b->packets_received);
    fprintf(f, "  payload_bytes_sent=%llu  payload_bytes_received=%llu"
               "  congestion_events=%llu\n",
            (unsigned long long)b->payload_bytes_sent,
            (unsigned long long)b->payload_bytes_received,
            (unsigned long long)b->congestion_events);
    fprintf(f, "---\n");

    return ATT1_OK;
}

/* -------------------------------------------------------------------------
 * Name helpers
 * ---------------------------------------------------------------------- */

const char *att1_aimu_trace_status_name(uint32_t status)
{
    switch (status) {
    case ATT1_AIMU_TRACE_STATUS_OK:       return "OK";
    case ATT1_AIMU_TRACE_STATUS_PARTIAL:  return "PARTIAL";
    case ATT1_AIMU_TRACE_STATUS_EMPTY:    return "EMPTY";
    default:                               return "UNKNOWN";
    }
}
