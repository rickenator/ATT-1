/*
 * aimu_mmio.c  —  AIMU MMIO/register-file simulator (M111)
 *
 * In-process implementation of the 64 KiB BAR0 register file described in
 * M104 (docs/aimu_register_map.md).
 *
 * Design
 * ------
 * The backing store is a flat uint32_t array indexed by (offset / 4).  A
 * per-offset access-type function enforces RO/RW/RW1C/WO semantics.
 * Reserved offsets return 0xDEADBEEF on read and silently discard writes.
 *
 * WO registers (GLOBAL_CONTROL, RESET_CONTROL, CQ_DOORBELL,
 * TILE_RESET_CONTROL) are not stored; reads return 0.  Writes trigger
 * side-effects against attached M105–M108 simulators.
 *
 * COUNTER_SNAPSHOT_CONTROL is RW; the SNAP_NOW bit (bit 0) is
 * self-clearing: it triggers att1_aimu_trace_snapshot_all() and refreshes
 * counter register cells, but is not persisted in the backing store.
 *
 * No ATT-1 inference, backend, tokenizer, CUDA, or binary-format behaviour
 * is changed by this module.
 */

#include "att1_aimu_mmio.h"
#include "att1_aimu_device.h"
#include "att1_aimu_cmdq.h"
#include "att1_aimu_trace.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* =========================================================================
 * Internal helpers
 * ====================================================================== */

typedef enum {
    ACCESS_RO,        /* read-only; write → ATT1_ERR_UNSUPPORTED */
    ACCESS_RW,        /* read-write */
    ACCESS_RW1C,      /* read-write-one-to-clear */
    ACCESS_WO,        /* write-only; read returns 0 */
    ACCESS_RESERVED   /* unassigned; read → 0xDEADBEEF, write discarded */
} reg_access_t;

/*
 * mmio_access_type
 *
 * Return the access type for a 4-byte-aligned BAR0 offset.
 * Caller has already validated alignment and range.
 */
static reg_access_t mmio_access_type(uint32_t off)
{
    /* ---- Global device registers (M104 §2, base 0x0000) ---- */
    switch (off) {
        case ATT1_MMIO_DEVICE_ID:                return ACCESS_RO;
        case ATT1_MMIO_DEVICE_VERSION:           return ACCESS_RO;
        case ATT1_MMIO_REGISTER_MAP_VERSION:     return ACCESS_RO;
        case ATT1_MMIO_FEATURE_FLAGS_LOW:        return ACCESS_RO;
        case ATT1_MMIO_FEATURE_FLAGS_HIGH:       return ACCESS_RO;
        case ATT1_MMIO_TILE_COUNT:               return ACCESS_RO;
        case ATT1_MMIO_COMMAND_QUEUE_COUNT:      return ACCESS_RO;
        case ATT1_MMIO_INTERRUPT_STATUS:         return ACCESS_RW1C;
        case ATT1_MMIO_INTERRUPT_ENABLE:         return ACCESS_RW;
        case ATT1_MMIO_GLOBAL_STATUS:            return ACCESS_RO;
        case ATT1_MMIO_GLOBAL_CONTROL:           return ACCESS_WO;
        case ATT1_MMIO_RESET_CONTROL:            return ACCESS_WO;
        case ATT1_MMIO_ERROR_STATUS:             return ACCESS_RW1C;
        case ATT1_MMIO_ERROR_DETAIL:             return ACCESS_RO;
        case ATT1_MMIO_TRACE_CONTROL:            return ACCESS_RW;
        case ATT1_MMIO_COUNTER_SNAPSHOT_CONTROL: return ACCESS_RW;
        default: break;
    }

    /* ---- Command queue registers (M104 §4, base 0x1000) ---- */
    switch (off) {
        case ATT1_MMIO_CQ_BASE_ADDR_LOW:       return ACCESS_RW;
        case ATT1_MMIO_CQ_BASE_ADDR_HIGH:      return ACCESS_RW;
        case ATT1_MMIO_CQ_SIZE:                return ACCESS_RW;
        case ATT1_MMIO_CQ_HEAD:                return ACCESS_RO;
        case ATT1_MMIO_CQ_TAIL:                return ACCESS_RW;
        case ATT1_MMIO_CQ_DOORBELL:            return ACCESS_WO;
        case ATT1_MMIO_CQ_STATUS:              return ACCESS_RO;
        case ATT1_MMIO_CQ_ERROR:               return ACCESS_RW1C;
        case ATT1_MMIO_CQ_FENCE_VALUE:         return ACCESS_RO;
        case ATT1_MMIO_CQ_COMPLETION_ADDR_LOW: return ACCESS_RW;
        case ATT1_MMIO_CQ_COMPLETION_ADDR_HIGH:return ACCESS_RW;
        case ATT1_MMIO_CQ_COMPLETION_SIZE:     return ACCESS_RW;
        default: break;
    }

    /* ---- DMA registers (M104 §5, base 0x2000) ---- */
    switch (off) {
        case ATT1_MMIO_DMA_CONTROL:         return ACCESS_RW;
        case ATT1_MMIO_DMA_STATUS:          return ACCESS_RO;
        case ATT1_MMIO_DMA_ERROR_STATUS:    return ACCESS_RW1C;
        case ATT1_MMIO_DMA_RING_BASE_LOW:   return ACCESS_RW;
        case ATT1_MMIO_DMA_RING_BASE_HIGH:  return ACCESS_RW;
        case ATT1_MMIO_DMA_RING_SIZE:       return ACCESS_RW;
        default: break;
    }

    /* ---- Fabric registers (M104 §6, base 0x3000) ---- */
    switch (off) {
        case ATT1_MMIO_FABRIC_STATUS:             return ACCESS_RO;
        case ATT1_MMIO_FABRIC_CONTROL:            return ACCESS_RW;
        case ATT1_MMIO_FABRIC_ROUTE_BASE_LOW:     return ACCESS_RW;
        case ATT1_MMIO_FABRIC_ROUTE_BASE_HIGH:    return ACCESS_RW;
        case ATT1_MMIO_FABRIC_ROUTE_SIZE:         return ACCESS_RW;
        case ATT1_MMIO_FABRIC_PKT_CTR_LOW:        return ACCESS_RO;
        case ATT1_MMIO_FABRIC_PKT_CTR_HIGH:       return ACCESS_RO;
        case ATT1_MMIO_FABRIC_PAYLOAD_BYTES_LOW:  return ACCESS_RO;
        case ATT1_MMIO_FABRIC_PAYLOAD_BYTES_HIGH: return ACCESS_RO;
        case ATT1_MMIO_FABRIC_CONGESTION_CTR:     return ACCESS_RO;
        case ATT1_MMIO_FABRIC_ERROR_STATUS:       return ACCESS_RW1C;
        default: break;
    }

    /* ---- Counter registers (M104 §7, base 0x4000): all RO ---- */
    if (off >= ATT1_MMIO_CNT_BASE && off < ATT1_MMIO_CNT_END) {
        return ACCESS_RO;
    }

    /* ---- Trace/debug registers (M104 §8, base 0x5000) ---- */
    switch (off) {
        case ATT1_MMIO_TRACE_BUFFER_BASE_LOW:  return ACCESS_RW;
        case ATT1_MMIO_TRACE_BUFFER_BASE_HIGH: return ACCESS_RW;
        case ATT1_MMIO_TRACE_BUFFER_SIZE:      return ACCESS_RW;
        case ATT1_MMIO_TRACE_WRITE_PTR:        return ACCESS_RO;
        case ATT1_MMIO_TRACE_FLAGS:            return ACCESS_RW;
        case ATT1_MMIO_TRACE_DROPPED_EVENTS:   return ACCESS_RO;
        case ATT1_MMIO_TRACE_SNAPSHOT_CONTROL: return ACCESS_RW;
        default: break;
    }

    /* ---- Per-tile register windows (M104 §3, 0x8000–0xFFFF) ---- */
    if (off >= UINT32_C(0x8000)) {
        uint32_t tile_off = (off - UINT32_C(0x8000)) % UINT32_C(0x800);
        switch (tile_off) {
            case ATT1_MMIO_TILE_ID:                   return ACCESS_RO;
            case ATT1_MMIO_TILE_STATUS:               return ACCESS_RO;
            case ATT1_MMIO_TILE_FEATURE_FLAGS:        return ACCESS_RO;
            case ATT1_MMIO_TILE_MEMORY_CAPACITY_LOW:  return ACCESS_RO;
            case ATT1_MMIO_TILE_MEMORY_CAPACITY_HIGH: return ACCESS_RO;
            case ATT1_MMIO_TILE_MEMORY_USED_LOW:      return ACCESS_RO;
            case ATT1_MMIO_TILE_MEMORY_USED_HIGH:     return ACCESS_RO;
            case ATT1_MMIO_TILE_KV_CAPACITY_LOW:      return ACCESS_RO;
            case ATT1_MMIO_TILE_KV_CAPACITY_HIGH:     return ACCESS_RO;
            case ATT1_MMIO_TILE_KV_USED_LOW:          return ACCESS_RO;
            case ATT1_MMIO_TILE_KV_USED_HIGH:         return ACCESS_RO;
            case ATT1_MMIO_TILE_SUPPORTED_DTYPES:     return ACCESS_RO;
            case ATT1_MMIO_TILE_SUPPORTED_OPS_LOW:    return ACCESS_RO;
            case ATT1_MMIO_TILE_SUPPORTED_OPS_HIGH:   return ACCESS_RO;
            case ATT1_MMIO_TILE_FABRIC_LINK_MASK:     return ACCESS_RO;
            case ATT1_MMIO_TILE_ERROR_STATUS:         return ACCESS_RW1C;
            case ATT1_MMIO_TILE_RESET_CONTROL:        return ACCESS_WO;
            default:                                  return ACCESS_RESERVED;
        }
    }

    return ACCESS_RESERVED;
}

/*
 * mmio_set_defaults
 *
 * Populate architectural-default register cells.  Called at create-time and
 * after SOFT_RESET_ALL.
 */
static void mmio_set_defaults(att1_aimu_mmio *m)
{
    m->regs[ATT1_MMIO_DEVICE_ID            / 4u] = ATT1_MMIO_DEVICE_ID_DEFAULT;
    m->regs[ATT1_MMIO_REGISTER_MAP_VERSION / 4u] = ATT1_AIMU_REGISTER_MAP_VERSION;
    m->regs[ATT1_MMIO_GLOBAL_STATUS        / 4u] = ATT1_MMIO_GSTAT_DEVICE_READY;
    m->regs[ATT1_MMIO_DEVICE_VERSION       / 4u] =
        ((uint32_t)ATT1_AIMU_DEVICE_VERSION_MAJOR << 24u) |
        ((uint32_t)ATT1_AIMU_DEVICE_VERSION_MINOR << 16u) |
        ((uint32_t)ATT1_AIMU_DEVICE_VERSION_PATCH <<  8u) |
         (uint32_t)ATT1_AIMU_DEVICE_VERSION_BUILD;
}

/*
 * sync_from_device
 *
 * Copy M106 device/tile state into RO register cells.
 */
static void sync_from_device(att1_aimu_mmio *m)
{
    att1_aimu_device *dev = m->device;
    if (!dev) return;

    att1_aimu_device_info info;
    if (att1_aimu_device_query_info(dev, &info) != ATT1_OK) return;

    /* Global device registers */
    m->regs[ATT1_MMIO_DEVICE_ID            / 4u] = ATT1_MMIO_DEVICE_ID_DEFAULT;
    m->regs[ATT1_MMIO_DEVICE_VERSION       / 4u] =
        ((uint32_t)info.version.major << 24u) |
        ((uint32_t)info.version.minor << 16u) |
        ((uint32_t)info.version.patch <<  8u) |
         (uint32_t)info.version.build;
    m->regs[ATT1_MMIO_REGISTER_MAP_VERSION / 4u] = info.register_map_version;
    m->regs[ATT1_MMIO_FEATURE_FLAGS_LOW    / 4u] = (uint32_t)(info.feature_flags & UINT32_MAX);
    m->regs[ATT1_MMIO_FEATURE_FLAGS_HIGH   / 4u] = (uint32_t)(info.feature_flags >> 32u);
    m->regs[ATT1_MMIO_TILE_COUNT           / 4u] = (uint32_t)info.tile_count;
    m->regs[ATT1_MMIO_COMMAND_QUEUE_COUNT  / 4u] = (uint32_t)info.tile_count;

    /* GLOBAL_STATUS: derive from tile states */
    uint32_t gstat = ATT1_MMIO_GSTAT_DEVICE_READY;
    for (size_t i = 0; i < dev->tile_count; i++) {
        if (dev->tiles[i].state == ATT1_AIMU_TILE_ACTIVE)
            gstat |= ATT1_MMIO_GSTAT_ANY_TILE_ACTIVE;
        if (dev->tiles[i].state == ATT1_AIMU_TILE_ERROR)
            gstat |= ATT1_MMIO_GSTAT_ANY_TILE_ERROR;
    }
    /* Preserve TRACE_ACTIVE if it was previously set via GLOBAL_CONTROL */
    gstat |= (m->regs[ATT1_MMIO_GLOBAL_STATUS / 4u] & ATT1_MMIO_GSTAT_TRACE_ACTIVE);
    m->regs[ATT1_MMIO_GLOBAL_STATUS / 4u] = gstat;

    /* Per-tile register windows */
    for (size_t n = 0; n < dev->tile_count && n < ATT1_AIMU_DEVICE_MAX_TILES; n++) {
        const att1_aimu_tile_info *ti = &dev->tiles[n];
        uint32_t b = ATT1_MMIO_BASE_TILE(n);

        m->regs[(b + ATT1_MMIO_TILE_ID)                   / 4u] = ti->tile_id;
        m->regs[(b + ATT1_MMIO_TILE_STATUS)               / 4u] = (uint32_t)ti->state & 0x3u;
        m->regs[(b + ATT1_MMIO_TILE_FEATURE_FLAGS)        / 4u] =
            (uint32_t)(info.feature_flags & UINT32_MAX);

        m->regs[(b + ATT1_MMIO_TILE_MEMORY_CAPACITY_LOW)  / 4u] =
            (uint32_t)(ti->memory_capacity_bytes & UINT32_MAX);
        m->regs[(b + ATT1_MMIO_TILE_MEMORY_CAPACITY_HIGH) / 4u] =
            (uint32_t)(ti->memory_capacity_bytes >> 32u);

        m->regs[(b + ATT1_MMIO_TILE_MEMORY_USED_LOW)      / 4u] =
            (uint32_t)(ti->memory_used_bytes & UINT32_MAX);
        m->regs[(b + ATT1_MMIO_TILE_MEMORY_USED_HIGH)     / 4u] =
            (uint32_t)(ti->memory_used_bytes >> 32u);

        m->regs[(b + ATT1_MMIO_TILE_KV_CAPACITY_LOW)      / 4u] =
            (uint32_t)(ti->kv_capacity_bytes & UINT32_MAX);
        m->regs[(b + ATT1_MMIO_TILE_KV_CAPACITY_HIGH)     / 4u] =
            (uint32_t)(ti->kv_capacity_bytes >> 32u);

        m->regs[(b + ATT1_MMIO_TILE_KV_USED_LOW)          / 4u] =
            (uint32_t)(ti->kv_used_bytes & UINT32_MAX);
        m->regs[(b + ATT1_MMIO_TILE_KV_USED_HIGH)         / 4u] =
            (uint32_t)(ti->kv_used_bytes >> 32u);

        m->regs[(b + ATT1_MMIO_TILE_SUPPORTED_DTYPES)     / 4u] = ti->supported_dtypes;
        m->regs[(b + ATT1_MMIO_TILE_SUPPORTED_OPS_LOW)    / 4u] = ti->supported_ops;
        m->regs[(b + ATT1_MMIO_TILE_SUPPORTED_OPS_HIGH)   / 4u] = 0u;
        m->regs[(b + ATT1_MMIO_TILE_FABRIC_LINK_MASK)     / 4u] =
            (uint32_t)ti->fabric_link_mask;

        /* TILE_ERROR_STATUS: error_code in bits[7:0] */
        m->regs[(b + ATT1_MMIO_TILE_ERROR_STATUS)         / 4u] =
            (uint32_t)ti->error_code;
    }
}

/*
 * sync_from_cmdq
 *
 * Copy M105 command-queue live state into CQ register cells.
 */
static void sync_from_cmdq(att1_aimu_mmio *m)
{
    att1_aimu_cmdq *q = m->cmdq;
    if (!q) return;

    m->regs[ATT1_MMIO_CQ_HEAD        / 4u] = (uint32_t)q->cmd_head & 0xFFFFu;
    m->regs[ATT1_MMIO_CQ_TAIL        / 4u] = (uint32_t)q->cmd_tail & 0xFFFFu;
    m->regs[ATT1_MMIO_CQ_FENCE_VALUE / 4u] = (uint32_t)(q->counters.fence_value & 0xFFFFu);

    /* CQ_STATUS.queue_state bits[7:4]: 0=idle if head==tail, 1=processing */
    m->regs[ATT1_MMIO_CQ_STATUS / 4u] =
        (q->cmd_head == q->cmd_tail) ? 0u : UINT32_C(0x10);

    /* Counter register cells (cmdq-sourced subset) */
    m->regs[ATT1_MMIO_CNT_CMD_ISSUED_LO    / 4u] =
        (uint32_t)(q->counters.commands_submitted & UINT32_MAX);
    m->regs[ATT1_MMIO_CNT_CMD_ISSUED_HI    / 4u] =
        (uint32_t)(q->counters.commands_submitted >> 32u);
    m->regs[ATT1_MMIO_CNT_CMD_COMPLETED_LO / 4u] =
        (uint32_t)(q->counters.commands_completed & UINT32_MAX);
    m->regs[ATT1_MMIO_CNT_CMD_COMPLETED_HI / 4u] =
        (uint32_t)(q->counters.commands_completed >> 32u);
    m->regs[ATT1_MMIO_CNT_STALL_QUEUE_FULL_LO / 4u] =
        (uint32_t)(q->counters.queue_full_count & UINT32_MAX);
    m->regs[ATT1_MMIO_CNT_STALL_QUEUE_FULL_HI / 4u] =
        (uint32_t)(q->counters.queue_full_count >> 32u);
}

/*
 * sync_from_trace
 *
 * Copy the most recent M108 trace snapshot into counter register cells.
 * Counter cells from the trace snapshot override those from sync_from_cmdq.
 */
static void sync_from_trace(att1_aimu_mmio *m)
{
    att1_aimu_trace *tr = m->trace;
    if (!tr) return;
    if (tr->snapshot.meta.status == ATT1_AIMU_TRACE_STATUS_EMPTY) return;

    const att1_aimu_trace_cmdq_counters   *cmdq   = &tr->snapshot.cmdq;
    const att1_aimu_trace_dma_counters    *dma    = &tr->snapshot.dma;
    const att1_aimu_trace_fabric_counters *fabric = &tr->snapshot.fabric;
    const att1_aimu_trace_device_counters *device = &tr->snapshot.device;

#define WR64(lo_off, val) \
    m->regs[(lo_off)       / 4u] = (uint32_t)((val) & UINT32_MAX); \
    m->regs[((lo_off) + 4u) / 4u] = (uint32_t)((val) >> 32u)

    WR64(ATT1_MMIO_CNT_CMD_ISSUED_LO,       cmdq->commands_submitted);
    WR64(ATT1_MMIO_CNT_CMD_COMPLETED_LO,    cmdq->commands_completed);
    WR64(ATT1_MMIO_CNT_STALL_QUEUE_FULL_LO, cmdq->queue_full_count);

    uint64_t errors_total = cmdq->commands_failed + device->tile_errors;
    WR64(ATT1_MMIO_CNT_ERRORS_TOTAL_LO, errors_total);

    WR64(ATT1_MMIO_CNT_TENSOR_BYTES_RD_LO,  dma->bytes_host_to_device);
    WR64(ATT1_MMIO_CNT_TENSOR_BYTES_WR_LO,  dma->bytes_device_to_host);

    WR64(ATT1_MMIO_CNT_FABRIC_PKT_SENT_LO,  fabric->packets_sent);
    WR64(ATT1_MMIO_CNT_FABRIC_PKT_RECV_LO,  fabric->packets_received);

    /* Also refresh fabric section RO registers */
    WR64(ATT1_MMIO_FABRIC_PKT_CTR_LOW,       fabric->packets_sent);
    WR64(ATT1_MMIO_FABRIC_PAYLOAD_BYTES_LOW,  fabric->payload_bytes_sent);
    m->regs[ATT1_MMIO_FABRIC_CONGESTION_CTR / 4u] =
        (uint32_t)(fabric->congestion_events & UINT32_MAX);

#undef WR64
}

/* =========================================================================
 * WO register side-effect dispatcher
 * ====================================================================== */

static att1_status_t mmio_wo_write(att1_aimu_mmio *m,
                                    uint32_t        off,
                                    uint32_t        value)
{
    /* Per-tile TILE_RESET_CONTROL */
    if (off >= UINT32_C(0x8000)) {
        uint32_t rel      = off - UINT32_C(0x8000);
        uint32_t tile_n   = rel / UINT32_C(0x800);
        uint32_t tile_off = rel % UINT32_C(0x800);
        if (tile_off == ATT1_MMIO_TILE_RESET_CONTROL) {
            (void)value; /* all bits are reset-mode triggers; handled as full reset */
            if (m->device && tile_n < (uint32_t)m->device->tile_count)
                (void)att1_aimu_device_reset_tile(m->device, (uint8_t)tile_n);
            sync_from_device(m);
        }
        return ATT1_OK;
    }

    switch (off) {
        case ATT1_MMIO_GLOBAL_CONTROL:
            if (value & ATT1_MMIO_GCTRL_ENABLE_DEVICE)
                m->regs[ATT1_MMIO_GLOBAL_STATUS / 4u] |= ATT1_MMIO_GSTAT_DEVICE_READY;
            if (value & ATT1_MMIO_GCTRL_DISABLE_DEVICE)
                m->regs[ATT1_MMIO_GLOBAL_STATUS / 4u] &= ~ATT1_MMIO_GSTAT_DEVICE_READY;
            if (value & ATT1_MMIO_GCTRL_ENABLE_TRACE)
                m->regs[ATT1_MMIO_GLOBAL_STATUS / 4u] |= ATT1_MMIO_GSTAT_TRACE_ACTIVE;
            if (value & ATT1_MMIO_GCTRL_DISABLE_TRACE)
                m->regs[ATT1_MMIO_GLOBAL_STATUS / 4u] &= ~ATT1_MMIO_GSTAT_TRACE_ACTIVE;
            break;

        case ATT1_MMIO_RESET_CONTROL:
            if (value & ATT1_MMIO_RSTCTL_SOFT_RESET_ALL) {
                if (m->device) (void)att1_aimu_device_reset(m->device);
                mmio_set_defaults(m);
                sync_from_device(m);
                sync_from_cmdq(m);
                sync_from_trace(m);
            }
            if (value & ATT1_MMIO_RSTCTL_RESET_COUNTERS) {
                memset(&m->regs[ATT1_MMIO_CNT_BASE / 4u], 0,
                       (size_t)(ATT1_MMIO_CNT_END - ATT1_MMIO_CNT_BASE));
                if (m->trace) (void)att1_aimu_trace_reset(m->trace);
            }
            if (value & ATT1_MMIO_RSTCTL_RESET_TRACE) {
                m->regs[ATT1_MMIO_TRACE_WRITE_PTR      / 4u] = 0u;
                m->regs[ATT1_MMIO_TRACE_DROPPED_EVENTS / 4u] = 0u;
            }
            /* RESET_FABRIC: placeholder; no-op */
            break;

        case ATT1_MMIO_CQ_DOORBELL:
            m->doorbell_write_count++;
            m->regs[ATT1_MMIO_CQ_TAIL / 4u] = value & UINT32_C(0xFFFF);
            break;

        default:
            break;
    }
    return ATT1_OK;
}

/* =========================================================================
 * Public API
 * ====================================================================== */

att1_status_t att1_aimu_mmio_create(att1_aimu_mmio **out)
{
    if (!out) return ATT1_ERR_INVALID_ARG;

    att1_aimu_mmio *m = calloc(1, sizeof(att1_aimu_mmio));
    if (!m) return ATT1_ERR_OOM;

    m->magic = ATT1_AIMU_MMIO_MAGIC;
    mmio_set_defaults(m);

    *out = m;
    return ATT1_OK;
}

void att1_aimu_mmio_destroy(att1_aimu_mmio *m)
{
    if (!m) return;
    m->magic = 0u;
    free(m);
}

/* -------------------------------------------------------------------------
 * Register access
 * ---------------------------------------------------------------------- */

att1_status_t att1_aimu_mmio_read32(att1_aimu_mmio *m,
                                     uint32_t        offset,
                                     uint32_t       *out)
{
    if (!m || !out)                             return ATT1_ERR_INVALID_ARG;
    if (m->magic != ATT1_AIMU_MMIO_MAGIC)       return ATT1_ERR_INVALID_ARG;
    if (offset % 4u)                            return ATT1_ERR_INVALID_ARG;
    if (offset >= ATT1_AIMU_MMIO_BAR0_SIZE)     return ATT1_ERR_INVALID_ARG;

    reg_access_t at = mmio_access_type(offset);
    switch (at) {
        case ACCESS_WO:
            *out = 0u;
            return ATT1_OK;
        case ACCESS_RESERVED:
            *out = ATT1_AIMU_MMIO_RESERVED_RD;
            return ATT1_OK;
        default:
            *out = m->regs[offset / 4u];
            return ATT1_OK;
    }
}

att1_status_t att1_aimu_mmio_write32(att1_aimu_mmio *m,
                                      uint32_t        offset,
                                      uint32_t        value)
{
    if (!m)                                     return ATT1_ERR_INVALID_ARG;
    if (m->magic != ATT1_AIMU_MMIO_MAGIC)       return ATT1_ERR_INVALID_ARG;
    if (offset % 4u)                            return ATT1_ERR_INVALID_ARG;
    if (offset >= ATT1_AIMU_MMIO_BAR0_SIZE)     return ATT1_ERR_INVALID_ARG;

    reg_access_t at = mmio_access_type(offset);
    switch (at) {
        case ACCESS_RO:
            return ATT1_ERR_UNSUPPORTED;

        case ACCESS_RW:
            /* COUNTER_SNAPSHOT_CONTROL: SNAP_NOW is self-clearing */
            if (offset == ATT1_MMIO_COUNTER_SNAPSHOT_CONTROL) {
                if (value & ATT1_MMIO_SNAP_NOW) {
                    m->snapshot_trigger_count++;
                    if (m->trace) {
                        (void)att1_aimu_trace_snapshot_all(
                            m->trace, m->cmdq, m->device, NULL);
                        sync_from_trace(m);
                    } else if (m->cmdq) {
                        sync_from_cmdq(m);
                    }
                    m->regs[offset / 4u] = value & ~ATT1_MMIO_SNAP_NOW;
                } else {
                    m->regs[offset / 4u] = value;
                }
            } else {
                m->regs[offset / 4u] = value;
            }
            return ATT1_OK;

        case ACCESS_RW1C:
            m->regs[offset / 4u] &= ~value;
            return ATT1_OK;

        case ACCESS_WO:
            return mmio_wo_write(m, offset, value);

        case ACCESS_RESERVED:
            /* Silently discard */
            return ATT1_OK;
    }
    return ATT1_OK; /* unreachable */
}

att1_status_t att1_aimu_mmio_read64(att1_aimu_mmio *m,
                                     uint32_t        offset,
                                     uint64_t       *out)
{
    if (!m || !out)                         return ATT1_ERR_INVALID_ARG;
    if (offset % 8u)                        return ATT1_ERR_INVALID_ARG;
    if (offset + 4u >= ATT1_AIMU_MMIO_BAR0_SIZE) return ATT1_ERR_INVALID_ARG;

    uint32_t lo, hi;
    att1_status_t rc;

    rc = att1_aimu_mmio_read32(m, offset,      &lo); if (rc) return rc;
    rc = att1_aimu_mmio_read32(m, offset + 4u, &hi); if (rc) return rc;

    *out = ((uint64_t)hi << 32u) | (uint64_t)lo;
    return ATT1_OK;
}

att1_status_t att1_aimu_mmio_write64(att1_aimu_mmio *m,
                                      uint32_t        offset,
                                      uint64_t        value)
{
    if (!m)                                  return ATT1_ERR_INVALID_ARG;
    if (offset % 8u)                         return ATT1_ERR_INVALID_ARG;
    if (offset + 4u >= ATT1_AIMU_MMIO_BAR0_SIZE) return ATT1_ERR_INVALID_ARG;

    att1_status_t rc;
    rc = att1_aimu_mmio_write32(m, offset,      (uint32_t)(value & UINT32_MAX));
    if (rc) return rc;
    rc = att1_aimu_mmio_write32(m, offset + 4u, (uint32_t)(value >> 32u));
    return rc;
}

/* -------------------------------------------------------------------------
 * Attachment
 * ---------------------------------------------------------------------- */

att1_status_t att1_aimu_mmio_attach_device(att1_aimu_mmio  *m,
                                            att1_aimu_device *dev)
{
    if (!m) return ATT1_ERR_INVALID_ARG;
    m->device = dev;
    return ATT1_OK;
}

att1_status_t att1_aimu_mmio_attach_cmdq(att1_aimu_mmio *m,
                                          att1_aimu_cmdq *cq)
{
    if (!m) return ATT1_ERR_INVALID_ARG;
    m->cmdq = cq;
    return ATT1_OK;
}

att1_status_t att1_aimu_mmio_attach_trace(att1_aimu_mmio *m,
                                           att1_aimu_trace *tr)
{
    if (!m) return ATT1_ERR_INVALID_ARG;
    m->trace = tr;
    return ATT1_OK;
}

/* -------------------------------------------------------------------------
 * Synchronisation
 * ---------------------------------------------------------------------- */

att1_status_t att1_aimu_mmio_sync(att1_aimu_mmio *m)
{
    if (!m) return ATT1_ERR_INVALID_ARG;
    mmio_set_defaults(m);
    sync_from_device(m);
    sync_from_cmdq(m);
    sync_from_trace(m);
    return ATT1_OK;
}

/* -------------------------------------------------------------------------
 * Reset
 * ---------------------------------------------------------------------- */

att1_status_t att1_aimu_mmio_reset(att1_aimu_mmio *m)
{
    if (!m) return ATT1_ERR_INVALID_ARG;
    memset(m->regs, 0, sizeof(m->regs));
    m->doorbell_write_count    = 0u;
    m->snapshot_trigger_count  = 0u;
    mmio_set_defaults(m);
    return ATT1_OK;
}

/* -------------------------------------------------------------------------
 * Rendering / diagnostics
 * ---------------------------------------------------------------------- */

att1_status_t att1_aimu_mmio_render(const att1_aimu_mmio *m, FILE *fp)
{
    if (!m || !fp) return ATT1_ERR_INVALID_ARG;

    fprintf(fp, "[att1_aimu_mmio]\n");
    fprintf(fp, "  magic=0x%08X\n", m->magic);
    fprintf(fp, "  doorbell_write_count=%u\n", m->doorbell_write_count);
    fprintf(fp, "  snapshot_trigger_count=%u\n", m->snapshot_trigger_count);

    /* Global device registers */
    fprintf(fp, "  DEVICE_ID=0x%08X\n",
            m->regs[ATT1_MMIO_DEVICE_ID / 4u]);
    fprintf(fp, "  DEVICE_VERSION=0x%08X\n",
            m->regs[ATT1_MMIO_DEVICE_VERSION / 4u]);
    fprintf(fp, "  REGISTER_MAP_VERSION=0x%08X\n",
            m->regs[ATT1_MMIO_REGISTER_MAP_VERSION / 4u]);
    fprintf(fp, "  FEATURE_FLAGS_LOW=0x%08X\n",
            m->regs[ATT1_MMIO_FEATURE_FLAGS_LOW / 4u]);
    fprintf(fp, "  TILE_COUNT=%u\n",
            m->regs[ATT1_MMIO_TILE_COUNT / 4u]);
    fprintf(fp, "  GLOBAL_STATUS=0x%08X\n",
            m->regs[ATT1_MMIO_GLOBAL_STATUS / 4u]);
    fprintf(fp, "  ERROR_STATUS=0x%08X\n",
            m->regs[ATT1_MMIO_ERROR_STATUS / 4u]);
    fprintf(fp, "  TRACE_CONTROL=0x%08X\n",
            m->regs[ATT1_MMIO_TRACE_CONTROL / 4u]);
    fprintf(fp, "  COUNTER_SNAPSHOT_CONTROL=0x%08X\n",
            m->regs[ATT1_MMIO_COUNTER_SNAPSHOT_CONTROL / 4u]);

    /* CQ registers */
    fprintf(fp, "  CQ_HEAD=%u  CQ_TAIL=%u  CQ_FENCE_VALUE=%u\n",
            m->regs[ATT1_MMIO_CQ_HEAD        / 4u],
            m->regs[ATT1_MMIO_CQ_TAIL        / 4u],
            m->regs[ATT1_MMIO_CQ_FENCE_VALUE / 4u]);

    /* Counter summary */
    uint64_t issued =
        (uint64_t)m->regs[ATT1_MMIO_CNT_CMD_ISSUED_LO    / 4u] |
        ((uint64_t)m->regs[ATT1_MMIO_CNT_CMD_ISSUED_HI   / 4u] << 32u);
    uint64_t completed =
        (uint64_t)m->regs[ATT1_MMIO_CNT_CMD_COMPLETED_LO / 4u] |
        ((uint64_t)m->regs[ATT1_MMIO_CNT_CMD_COMPLETED_HI/ 4u] << 32u);
    fprintf(fp, "  CNT_CMD_ISSUED=%llu  CNT_CMD_COMPLETED=%llu\n",
            (unsigned long long)issued, (unsigned long long)completed);

    /* Per-tile summary */
    uint32_t tile_count = m->regs[ATT1_MMIO_TILE_COUNT / 4u];
    for (uint32_t n = 0; n < tile_count && n < ATT1_AIMU_DEVICE_MAX_TILES; n++) {
        uint32_t b = ATT1_MMIO_BASE_TILE(n);
        uint64_t cap =
            (uint64_t)m->regs[(b + ATT1_MMIO_TILE_MEMORY_CAPACITY_LOW)  / 4u] |
            ((uint64_t)m->regs[(b + ATT1_MMIO_TILE_MEMORY_CAPACITY_HIGH) / 4u] << 32u);
        fprintf(fp, "  tile[%u]: status=0x%02X  dtypes=0x%08X  mem_cap=%llu B\n",
                n,
                m->regs[(b + ATT1_MMIO_TILE_STATUS)           / 4u],
                m->regs[(b + ATT1_MMIO_TILE_SUPPORTED_DTYPES) / 4u],
                (unsigned long long)cap);
    }

    return ATT1_OK;
}

const char *att1_aimu_mmio_err_name(att1_status_t s)
{
    switch (s) {
        case ATT1_OK:                   return "ATT1_OK";
        case ATT1_ERR_INVALID_ARG:      return "ATT1_ERR_INVALID_ARG";
        case ATT1_ERR_OOM:              return "ATT1_ERR_OOM";
        case ATT1_ERR_IO:               return "ATT1_ERR_IO";
        case ATT1_ERR_BAD_FORMAT:       return "ATT1_ERR_BAD_FORMAT";
        case ATT1_ERR_NOT_FOUND:        return "ATT1_ERR_NOT_FOUND";
        case ATT1_ERR_SHAPE:            return "ATT1_ERR_SHAPE";
        case ATT1_ERR_QUEUE_FULL:       return "ATT1_ERR_QUEUE_FULL";
        case ATT1_ERR_STATE:            return "ATT1_ERR_STATE";
        case ATT1_ERR_UNSUPPORTED:      return "ATT1_ERR_UNSUPPORTED";
        case ATT1_ERR_QUEUE_EMPTY:      return "ATT1_ERR_QUEUE_EMPTY";
        case ATT1_ERR_TIMEOUT:          return "ATT1_ERR_TIMEOUT";
        case ATT1_ERR_ALREADY_STARTED:  return "ATT1_ERR_ALREADY_STARTED";
        default:                        return "ATT1_UNKNOWN";
    }
}
