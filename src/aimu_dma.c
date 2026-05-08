/*
 * aimu_dma.c  —  AIMU DMA descriptor simulator implementation (M107)
 */

#include "att1_aimu_dma.h"

#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Internal validation result codes
 * ====================================================================== */

typedef enum {
    DMA_VALID           = 0,
    DMA_ERR_NULL,
    DMA_ERR_DIRECTION,
    DMA_ERR_BYTE_LENGTH,
    DMA_ERR_FLAGS,
    DMA_ERR_DTYPE,
    DMA_ERR_Q4_ALIGN,
    DMA_ERR_ALIGNMENT,
    DMA_ERR_OVERFLOW,
    DMA_ERR_RANGE,
    DMA_ERR_D2D_OVERLAP
} dma_vresult;

/* =========================================================================
 * Internal helpers
 * ====================================================================== */

/* Returns 1 if [addr, addr+len) lies entirely within [r->base, r->base+r->size). */
static int region_contains(const att1_aimu_dma_region *r,
                            uint64_t addr, uint64_t len)
{
    if (len == 0u)    return 1;           /* degenerate — always contained */
    if (addr < r->base) return 0;
    uint64_t end_addr = addr + len;
    if (end_addr < addr) return 0;        /* wrapped — overflow */
    uint64_t reg_end = r->base + r->size;
    if (reg_end < r->base) return 0;      /* region itself overflowed */
    return (end_addr <= reg_end) ? 1 : 0;
}

/* Returns 1 if [addr, addr+len) is contained in at least one registered region. */
static int fits_any_region(const att1_aimu_dma_region *regions,
                            size_t count,
                            uint64_t addr, uint64_t len)
{
    for (size_t i = 0; i < count; i++) {
        if (region_contains(&regions[i], addr, len)) return 1;
    }
    return 0;
}

/* Returns 1 if the two address ranges overlap (conservative on overflow). */
static int ranges_overlap(uint64_t a_base, uint64_t a_len,
                           uint64_t b_base, uint64_t b_len)
{
    uint64_t a_end = a_base + a_len;
    uint64_t b_end = b_base + b_len;
    /* Treat wrapped ends as overlap — conservative and safe. */
    if (a_end < a_base || b_end < b_base) return 1;
    return !(a_end <= b_base || b_end <= a_base);
}

/* =========================================================================
 * Internal descriptor validation
 * ====================================================================== */

static dma_vresult validate_desc(const att1_aimu_dma      *sim,
                                  const att1_aimu_dma_desc *desc)
{
    if (!sim || !desc)
        return DMA_ERR_NULL;

    /* 1. Direction */
    if (desc->direction > (uint8_t)ATT1_AIMU_DMA_DEVICE_TO_DEVICE)
        return DMA_ERR_DIRECTION;

    /* 2. Byte length */
    if (desc->byte_length == 0u ||
        desc->byte_length > ATT1_AIMU_DMA_MAX_TRANSFER_BYTES)
        return DMA_ERR_BYTE_LENGTH;

    /* 3. Unknown flag bits */
    if (desc->flags & (uint16_t)(~ATT1_AIMU_DMA_FLAG_VALID_MASK))
        return DMA_ERR_FLAGS;

    /* 4. Dtype */
    if (desc->dtype != ATT1_AIMU_DMA_DTYPE_F32 &&
        desc->dtype != ATT1_AIMU_DMA_DTYPE_Q8  &&
        desc->dtype != ATT1_AIMU_DMA_DTYPE_Q4)
        return DMA_ERR_DTYPE;

    /* 5. Q4 group-size and payload alignment */
    if (desc->dtype == ATT1_AIMU_DMA_DTYPE_Q4) {
        uint8_t gs = desc->quant_group_size;
        if (gs != 32u && gs != 64u)
            return DMA_ERR_Q4_ALIGN;
        /* Each Q4 group is gs nibbles = gs/2 bytes. */
        uint32_t group_bytes = (uint32_t)gs / 2u;
        if (desc->byte_length % group_bytes != 0u)
            return DMA_ERR_Q4_ALIGN;
    }

    if (desc->direction == (uint8_t)ATT1_AIMU_DMA_HOST_TO_DEVICE ||
        desc->direction == (uint8_t)ATT1_AIMU_DMA_DEVICE_TO_HOST) {

        /* 6. Address alignment */
        if (desc->host_addr   % ATT1_AIMU_DMA_ALIGN_BYTES != 0u)
            return DMA_ERR_ALIGNMENT;
        if (desc->device_addr % ATT1_AIMU_DMA_ALIGN_BYTES != 0u)
            return DMA_ERR_ALIGNMENT;

        /* 7. Overflow */
        uint64_t h_end = desc->host_addr   + (uint64_t)desc->byte_length;
        uint64_t d_end = desc->device_addr + (uint64_t)desc->byte_length;
        if (h_end < desc->host_addr)   return DMA_ERR_OVERFLOW;
        if (d_end < desc->device_addr) return DMA_ERR_OVERFLOW;

        /* 8. Host region range (skipped when no regions registered) */
        if (sim->host_region_count > 0u) {
            if (!fits_any_region(sim->host_regions, sim->host_region_count,
                                 desc->host_addr, (uint64_t)desc->byte_length))
                return DMA_ERR_RANGE;
        }

        /* 8. Device region range */
        if (sim->device_region_count > 0u) {
            if (!fits_any_region(sim->device_regions, sim->device_region_count,
                                 desc->device_addr, (uint64_t)desc->byte_length))
                return DMA_ERR_RANGE;
        }

    } else {
        /* direction == ATT1_AIMU_DMA_DEVICE_TO_DEVICE */

        /* 6. Address alignment */
        if (desc->src_device_addr % ATT1_AIMU_DMA_ALIGN_BYTES != 0u)
            return DMA_ERR_ALIGNMENT;
        if (desc->dst_device_addr % ATT1_AIMU_DMA_ALIGN_BYTES != 0u)
            return DMA_ERR_ALIGNMENT;

        /* 7. Overflow */
        uint64_t s_end = desc->src_device_addr + (uint64_t)desc->byte_length;
        uint64_t d_end = desc->dst_device_addr + (uint64_t)desc->byte_length;
        if (s_end < desc->src_device_addr) return DMA_ERR_OVERFLOW;
        if (d_end < desc->dst_device_addr) return DMA_ERR_OVERFLOW;

        /* 9. Overlap */
        if (ranges_overlap(desc->src_device_addr, (uint64_t)desc->byte_length,
                           desc->dst_device_addr, (uint64_t)desc->byte_length))
            return DMA_ERR_D2D_OVERLAP;

        /* 8. Device region range */
        if (sim->device_region_count > 0u) {
            if (!fits_any_region(sim->device_regions, sim->device_region_count,
                                 desc->src_device_addr, (uint64_t)desc->byte_length))
                return DMA_ERR_RANGE;
            if (!fits_any_region(sim->device_regions, sim->device_region_count,
                                 desc->dst_device_addr, (uint64_t)desc->byte_length))
                return DMA_ERR_RANGE;
        }
    }

    return DMA_VALID;
}

/* =========================================================================
 * Lifecycle
 * ====================================================================== */

att1_status_t att1_aimu_dma_create(att1_aimu_dma **out)
{
    if (!out) return ATT1_ERR_INVALID_ARG;

    att1_aimu_dma *sim = (att1_aimu_dma *)calloc(1, sizeof(*sim));
    if (!sim) return ATT1_ERR_OOM;

    sim->magic = ATT1_AIMU_DMA_MAGIC;
    sim->next_descriptor_id = 1u;

    *out = sim;
    return ATT1_OK;
}

void att1_aimu_dma_destroy(att1_aimu_dma *sim)
{
    if (!sim) return;
    sim->magic = 0u;
    free(sim);
}

/* =========================================================================
 * Region registration
 * ====================================================================== */

att1_status_t att1_aimu_dma_register_host_region(att1_aimu_dma *sim,
                                                   uint64_t       base,
                                                   uint64_t       size)
{
    if (!sim || size == 0u) return ATT1_ERR_INVALID_ARG;
    if (base % ATT1_AIMU_DMA_ALIGN_BYTES != 0u) return ATT1_ERR_INVALID_ARG;
    if (sim->host_region_count >= ATT1_AIMU_DMA_MAX_HOST_REGIONS)
        return ATT1_ERR_STATE;

    sim->host_regions[sim->host_region_count].base = base;
    sim->host_regions[sim->host_region_count].size = size;
    sim->host_region_count++;
    return ATT1_OK;
}

att1_status_t att1_aimu_dma_register_device_region(att1_aimu_dma *sim,
                                                     uint64_t       base,
                                                     uint64_t       size)
{
    if (!sim || size == 0u) return ATT1_ERR_INVALID_ARG;
    if (base % ATT1_AIMU_DMA_ALIGN_BYTES != 0u) return ATT1_ERR_INVALID_ARG;
    if (sim->device_region_count >= ATT1_AIMU_DMA_MAX_DEVICE_REGIONS)
        return ATT1_ERR_STATE;

    sim->device_regions[sim->device_region_count].base = base;
    sim->device_regions[sim->device_region_count].size = size;
    sim->device_region_count++;
    return ATT1_OK;
}

/* =========================================================================
 * Public validate (no counter update)
 * ====================================================================== */

att1_status_t att1_aimu_dma_validate(const att1_aimu_dma      *sim,
                                      const att1_aimu_dma_desc *desc)
{
    return (validate_desc(sim, desc) == DMA_VALID) ? ATT1_OK
                                                    : ATT1_ERR_INVALID_ARG;
}

/* =========================================================================
 * Submit (validates + updates counters)
 * ====================================================================== */

att1_status_t att1_aimu_dma_submit(att1_aimu_dma            *sim,
                                    const att1_aimu_dma_desc *desc)
{
    if (!sim || !desc) return ATT1_ERR_INVALID_ARG;

    sim->counters.dma_submitted++;

    dma_vresult r = validate_desc(sim, desc);

    if (r != DMA_VALID) {
        sim->counters.dma_failed++;
        switch (r) {
        case DMA_ERR_ALIGNMENT:
            sim->counters.alignment_failures++;
            break;
        case DMA_ERR_OVERFLOW:
        case DMA_ERR_RANGE:
        case DMA_ERR_D2D_OVERLAP:
            sim->counters.range_failures++;
            break;
        case DMA_ERR_FLAGS:
            sim->counters.unsupported_flags++;
            break;
        default:
            /* direction / byte_length / dtype / q4_align:
             * counted in dma_failed only — no dedicated sub-counter. */
            break;
        }
        return ATT1_ERR_INVALID_ARG;
    }

    /* Transfer accepted */
    sim->counters.dma_completed++;
    switch ((att1_aimu_dma_direction)desc->direction) {
    case ATT1_AIMU_DMA_HOST_TO_DEVICE:
        sim->counters.bytes_host_to_device += (uint64_t)desc->byte_length;
        break;
    case ATT1_AIMU_DMA_DEVICE_TO_HOST:
        sim->counters.bytes_device_to_host += (uint64_t)desc->byte_length;
        break;
    case ATT1_AIMU_DMA_DEVICE_TO_DEVICE:
        sim->counters.bytes_device_to_device += (uint64_t)desc->byte_length;
        break;
    }

    return ATT1_OK;
}

/* =========================================================================
 * Counters
 * ====================================================================== */

att1_status_t att1_aimu_dma_get_counters(const att1_aimu_dma    *sim,
                                          att1_aimu_dma_counters *out)
{
    if (!sim || !out) return ATT1_ERR_INVALID_ARG;
    *out = sim->counters;
    return ATT1_OK;
}

att1_status_t att1_aimu_dma_reset_counters(att1_aimu_dma *sim)
{
    if (!sim) return ATT1_ERR_INVALID_ARG;
    memset(&sim->counters, 0, sizeof(sim->counters));
    return ATT1_OK;
}
