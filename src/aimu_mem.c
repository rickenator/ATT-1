/*
 * aimu_mem.c  —  AIMU tile memory allocator simulator (M124)
 *
 * Implements att1_aimu_mem: a simulated per-tile address-space allocator.
 * All allocation records are O(count) C structs; no backing buffer of
 * capacity_bytes is ever allocated.
 *
 * Allocator algorithm: first-fit linear scan over the address space.
 *   1. Sort live allocations by base_address.
 *   2. Scan gaps between consecutive allocations (and before the first /
 *      after the last) for a gap large enough after alignment rounding.
 *
 * No inference, CUDA, tokenizer, or .att1 binary format behaviour is
 * changed by this module.
 */

#include "att1_aimu_mem.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <inttypes.h>

/* =========================================================================
 * Internal helpers
 * ====================================================================== */

static int is_pow2(uint64_t v)
{
    return (v != 0) && ((v & (v - 1)) == 0);
}

static uint64_t align_up(uint64_t addr, uint64_t align)
{
    /* align must be power of two */
    return (addr + align - 1) & ~(align - 1);
}

/* Validate magic and not-NULL. */
static int mem_valid(const att1_aimu_mem *m)
{
    return m && m->magic == ATT1_AIMU_MEM_MAGIC;
}

/*
 * collect_live_sorted
 *
 * Fill out[0..n) with pointers to live allocations sorted by base_address.
 * Returns the count of live allocations.  Caller must provide a buffer of
 * at least ATT1_AIMU_MEM_MAX_ALLOCS pointers.
 */
static uint32_t collect_live_sorted(const att1_aimu_mem        *m,
                                    const att1_aimu_mem_alloc **out)
{
    uint32_t n = 0;
    for (uint32_t i = 0; i < m->alloc_count; i++) {
        if (m->allocs[i].live) {
            out[n++] = &m->allocs[i];
        }
    }
    /* Insertion sort — alloc counts are small */
    for (uint32_t i = 1; i < n; i++) {
        const att1_aimu_mem_alloc *key = out[i];
        int32_t j = (int32_t)i - 1;
        while (j >= 0 && out[j]->base_address > key->base_address) {
            out[j + 1] = out[j];
            j--;
        }
        out[j + 1] = key;
    }
    return n;
}

/* =========================================================================
 * Lifecycle
 * ====================================================================== */

att1_status_t att1_aimu_mem_create(uint32_t        tile_id,
                                   uint64_t        capacity_bytes,
                                   att1_aimu_mem **out)
{
    if (!out)                return ATT1_ERR_INVALID_ARG;
    if (capacity_bytes == 0) return ATT1_ERR_INVALID_ARG;

    att1_aimu_mem *m = calloc(1, sizeof(att1_aimu_mem));
    if (!m) return ATT1_ERR_OOM;

    m->magic          = ATT1_AIMU_MEM_MAGIC;
    m->tile_id        = tile_id;
    m->capacity_bytes = capacity_bytes;
    m->next_id        = 1u;
    m->alloc_count    = 0u;

    *out = m;
    return ATT1_OK;
}

void att1_aimu_mem_destroy(att1_aimu_mem *m)
{
    if (!m) return;
    m->magic = 0u;
    free(m);
}

/* =========================================================================
 * Allocation
 * ====================================================================== */

att1_status_t att1_aimu_mem_alloc_range(att1_aimu_mem             *m,
                                        att1_aimu_mem_region_type  region_type,
                                        uint64_t                   byte_size,
                                        uint64_t                   alignment,
                                        uint8_t                    dtype,
                                        uint32_t                   flags,
                                        const char                *name,
                                        uint64_t                  *addr_out,
                                        uint32_t                  *id_out)
{
    if (!mem_valid(m))                          return ATT1_ERR_INVALID_ARG;
    if (byte_size == 0)                         return ATT1_ERR_INVALID_ARG;
    if ((int)region_type < 0 ||
        region_type >= ATT1_AIMU_MEM_REGION_COUNT) return ATT1_ERR_INVALID_ARG;
    if (flags & ~ATT1_AIMU_MEM_FLAG_ALL_KNOWN) return ATT1_ERR_INVALID_ARG;

    /* Apply minimum alignment */
    if (alignment == 0) alignment = ATT1_AIMU_MEM_MIN_ALIGN;
    if (!is_pow2(alignment))               return ATT1_ERR_INVALID_ARG;
    if (alignment < ATT1_AIMU_MEM_MIN_ALIGN) return ATT1_ERR_INVALID_ARG;
    if (alignment > ATT1_AIMU_MEM_MAX_ALIGN) return ATT1_ERR_INVALID_ARG;

    /* Allocation table full? */
    if (m->alloc_count >= ATT1_AIMU_MEM_MAX_ALLOCS) return ATT1_ERR_STATE;

    /* Overflow-safe capacity check: if byte_size > capacity_bytes, reject */
    if (byte_size > m->capacity_bytes)     return ATT1_ERR_OOM;

    /* Collect live allocations sorted by base_address */
    const att1_aimu_mem_alloc *sorted[ATT1_AIMU_MEM_MAX_ALLOCS];
    uint32_t live_count = collect_live_sorted(m, sorted);

    /* First-fit scan: find a gap large enough for align_up(candidate) + byte_size */
    uint64_t found_addr = UINT64_MAX;
    uint64_t search_start = ATT1_AIMU_MEM_BASE;

    for (uint32_t i = 0; i <= live_count; i++) {
        /* candidate: first aligned address in this gap */
        uint64_t candidate = align_up(search_start, alignment);

        /* overflow check */
        if (candidate < search_start) break;

        /* end of this gap */
        uint64_t gap_end;
        if (i < live_count) {
            gap_end = sorted[i]->base_address;
        } else {
            gap_end = m->capacity_bytes;
        }

        if (candidate > gap_end) {
            /* aligned start is past the gap end — no room here */
            if (i < live_count)
                search_start = sorted[i]->base_address + sorted[i]->byte_size;
            continue;
        }

        /* available bytes in this gap from candidate */
        uint64_t available = gap_end - candidate;
        if (available >= byte_size) {
            found_addr = candidate;
            break;
        }

        if (i < live_count)
            search_start = sorted[i]->base_address + sorted[i]->byte_size;
    }

    if (found_addr == UINT64_MAX) return ATT1_ERR_OOM;

    /* Write the record */
    att1_aimu_mem_alloc *rec = &m->allocs[m->alloc_count];
    memset(rec, 0, sizeof(*rec));
    rec->alloc_id    = m->next_id++;
    rec->tile_id     = m->tile_id;
    rec->region_type = region_type;
    rec->dtype       = dtype;
    rec->flags       = flags;
    rec->base_address = found_addr;
    rec->byte_size   = byte_size;
    rec->alignment   = alignment;
    rec->live        = 1;

    if (name) {
        size_t n = strlen(name);
        if (n >= ATT1_AIMU_MEM_NAME_MAX) n = ATT1_AIMU_MEM_NAME_MAX - 1u;
        memcpy(rec->name, name, n);
        rec->name[n] = '\0';
    }

    m->alloc_count++;

    if (addr_out) *addr_out = found_addr;
    if (id_out)   *id_out   = rec->alloc_id;

    return ATT1_OK;
}

att1_status_t att1_aimu_mem_free(att1_aimu_mem *m, uint32_t alloc_id)
{
    if (!mem_valid(m))          return ATT1_ERR_INVALID_ARG;
    if (alloc_id == 0 ||
        alloc_id == ATT1_AIMU_MEM_INVALID_ID) return ATT1_ERR_NOT_FOUND;

    for (uint32_t i = 0; i < m->alloc_count; i++) {
        if (m->allocs[i].alloc_id == alloc_id) {
            if (!m->allocs[i].live) return ATT1_ERR_STATE; /* double free */
            m->allocs[i].live = 0;
            return ATT1_OK;
        }
    }
    return ATT1_ERR_NOT_FOUND;
}

/* =========================================================================
 * Queries
 * ====================================================================== */

att1_status_t att1_aimu_mem_query_by_id(const att1_aimu_mem        *m,
                                        uint32_t                    alloc_id,
                                        const att1_aimu_mem_alloc **out)
{
    if (!mem_valid(m) || !out) return ATT1_ERR_INVALID_ARG;

    for (uint32_t i = 0; i < m->alloc_count; i++) {
        if (m->allocs[i].alloc_id == alloc_id && m->allocs[i].live) {
            *out = &m->allocs[i];
            return ATT1_OK;
        }
    }
    return ATT1_ERR_NOT_FOUND;
}

att1_status_t att1_aimu_mem_query_by_address(const att1_aimu_mem        *m,
                                             uint64_t                    addr,
                                             const att1_aimu_mem_alloc **out)
{
    if (!mem_valid(m) || !out) return ATT1_ERR_INVALID_ARG;

    for (uint32_t i = 0; i < m->alloc_count; i++) {
        if (!m->allocs[i].live) continue;
        uint64_t base = m->allocs[i].base_address;
        uint64_t end  = base + m->allocs[i].byte_size;
        if (addr >= base && addr < end) {
            *out = &m->allocs[i];
            return ATT1_OK;
        }
    }
    return ATT1_ERR_NOT_FOUND;
}

int att1_aimu_mem_range_valid(const att1_aimu_mem *m,
                              uint64_t             addr,
                              uint64_t             size)
{
    if (!mem_valid(m) || size == 0) return 0;

    /* Overflow-safe end address */
    if (addr > UINT64_MAX - size) return 0;
    uint64_t end = addr + size;

    for (uint32_t i = 0; i < m->alloc_count; i++) {
        if (!m->allocs[i].live) continue;
        uint64_t base  = m->allocs[i].base_address;
        uint64_t limit = base + m->allocs[i].byte_size;
        if (addr >= base && end <= limit) return 1;
    }
    return 0;
}

/* =========================================================================
 * Accounting
 * ====================================================================== */

att1_status_t att1_aimu_mem_get_frag(const att1_aimu_mem *m,
                                     att1_aimu_mem_frag  *out)
{
    if (!mem_valid(m) || !out) return ATT1_ERR_INVALID_ARG;

    const att1_aimu_mem_alloc *sorted[ATT1_AIMU_MEM_MAX_ALLOCS];
    uint32_t live_count = collect_live_sorted(m, sorted);

    uint64_t used = 0;
    for (uint32_t i = 0; i < live_count; i++) {
        used += sorted[i]->byte_size;
    }

    uint64_t free_bytes = (m->capacity_bytes > used)
                          ? (m->capacity_bytes - used)
                          : 0;

    /* Largest free block: scan gaps */
    uint64_t largest  = 0;
    uint64_t scan_pos = ATT1_AIMU_MEM_BASE;

    for (uint32_t i = 0; i <= live_count; i++) {
        uint64_t gap_end = (i < live_count)
                           ? sorted[i]->base_address
                           : m->capacity_bytes;

        if (gap_end > scan_pos) {
            uint64_t gap_size = gap_end - scan_pos;
            if (gap_size > largest) largest = gap_size;
        }

        if (i < live_count)
            scan_pos = sorted[i]->base_address + sorted[i]->byte_size;
    }

    out->capacity_bytes     = m->capacity_bytes;
    out->used_bytes         = used;
    out->free_bytes         = free_bytes;
    out->largest_free_block = largest;
    out->allocation_count   = live_count;

    /* Fragmentation estimate: portion of free space that is fragmented */
    if (free_bytes == 0) {
        out->fragmentation_pct = 0;
    } else if (largest >= free_bytes) {
        out->fragmentation_pct = 0;
    } else {
        uint64_t fragmented = free_bytes - largest;
        uint64_t pct = (fragmented * 100u) / free_bytes;
        out->fragmentation_pct = (pct > 100) ? 100u : (uint32_t)pct;
    }

    return ATT1_OK;
}

/* =========================================================================
 * Reset
 * ====================================================================== */

att1_status_t att1_aimu_mem_reset(att1_aimu_mem *m)
{
    if (!mem_valid(m)) return ATT1_ERR_INVALID_ARG;

    memset(m->allocs, 0, sizeof(m->allocs));
    m->alloc_count = 0u;
    m->next_id     = 1u;
    return ATT1_OK;
}

/* =========================================================================
 * Debug render
 * ====================================================================== */

static const char *region_name(att1_aimu_mem_region_type t)
{
    switch (t) {
    case ATT1_AIMU_MEM_REGION_TENSOR:           return "TENSOR";
    case ATT1_AIMU_MEM_REGION_KV_CACHE:         return "KV_CACHE";
    case ATT1_AIMU_MEM_REGION_STAGING:          return "STAGING";
    case ATT1_AIMU_MEM_REGION_DMA_BUFFER:       return "DMA_BUFFER";
    case ATT1_AIMU_MEM_REGION_COMMAND_QUEUE:    return "COMMAND_QUEUE";
    case ATT1_AIMU_MEM_REGION_COMPLETION_QUEUE: return "COMPLETION_QUEUE";
    case ATT1_AIMU_MEM_REGION_TRACE_BUFFER:     return "TRACE_BUFFER";
    case ATT1_AIMU_MEM_REGION_FABRIC_BUFFER:    return "FABRIC_BUFFER";
    case ATT1_AIMU_MEM_REGION_RESERVED:         return "RESERVED";
    default:                                    return "UNKNOWN";
    }
}

att1_status_t att1_aimu_mem_render(const att1_aimu_mem *m, FILE *fp)
{
    if (!mem_valid(m) || !fp) return ATT1_ERR_INVALID_ARG;

    att1_aimu_mem_frag frag;
    (void)att1_aimu_mem_get_frag(m, &frag);

    fprintf(fp, "att1_aimu_mem tile_id=%" PRIu32
                "  capacity=%" PRIu64 " B\n",
            m->tile_id, m->capacity_bytes);
    fprintf(fp, "  used=%" PRIu64 " B  free=%" PRIu64 " B"
                "  largest_free=%" PRIu64 " B"
                "  allocs=%" PRIu32 "  frag=%" PRIu32 "%%\n",
            frag.used_bytes, frag.free_bytes,
            frag.largest_free_block,
            frag.allocation_count,
            frag.fragmentation_pct);

    for (uint32_t i = 0; i < m->alloc_count; i++) {
        const att1_aimu_mem_alloc *a = &m->allocs[i];
        if (!a->live) continue;
        fprintf(fp,
            "  [%4" PRIu32 "] %-16s  base=0x%016" PRIx64
            "  size=%" PRIu64 " B  align=%" PRIu64
            "  flags=0x%08" PRIx32 "  name=\"%s\"\n",
            a->alloc_id,
            region_name(a->region_type),
            a->base_address,
            a->byte_size,
            a->alignment,
            a->flags,
            a->name);
    }

    return ATT1_OK;
}
