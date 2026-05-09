/*
 * test_aimu_mem.c  —  Tests for M124 AIMU tile memory allocator simulator
 *
 * All tests use small allocation sizes.  A tile may be configured with a
 * 64 GiB capacity; the test verifies that no large buffer is allocated by
 * running in under a second with negligible RAM usage.
 *
 * Covers:
 *  1.  test_create_destroy           — lifecycle, magic check
 *  2.  test_alloc_free_simple        — basic allocate then free
 *  3.  test_alignment_honored        — returned address is aligned
 *  4.  test_no_overlap               — multiple allocations do not overlap
 *  5.  test_out_of_capacity          — allocation beyond capacity rejected
 *  6.  test_zero_size_fails          — byte_size == 0 rejected
 *  7.  test_invalid_alignment        — non-power-of-two alignment rejected
 *  8.  test_unknown_region_type      — region_type >= COUNT rejected
 *  9.  test_double_free_fails        — second free of same id rejected
 * 10.  test_query_by_id              — query_by_id returns correct record
 * 11.  test_query_by_address         — query_by_address finds containing alloc
 * 12.  test_range_valid              — range_valid true/false scenarios
 * 13.  test_reset_clears             — reset clears all allocations
 * 14.  test_frag_deterministic       — fragmentation report reproducible
 * 15.  test_large_capacity_no_alloc  — 64 GiB capacity metadata, tiny alloc
 * 16.  test_no_cuda_dep              — compile-time guard
 */

#include "att1_aimu_mem.h"
#include "att1_status.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>

/* =========================================================================
 * Test harness
 * ====================================================================== */

static int g_pass = 0;
static int g_fail = 0;

#define EXPECT(cond, name) \
    do { \
        if (cond) { \
            printf("PASS: aimu_mem: " name "\n"); \
            g_pass++; \
        } else { \
            printf("FAIL: aimu_mem: " name "\n"); \
            g_fail++; \
        } \
    } while (0)

/* =========================================================================
 * 1. test_create_destroy
 * ====================================================================== */

static void test_create_destroy(void)
{
    att1_aimu_mem *m = NULL;

    /* NULL out pointer */
    att1_status_t s = att1_aimu_mem_create(0, 1024, NULL);
    EXPECT(s == ATT1_ERR_INVALID_ARG, "create: null out -> INVALID_ARG");

    /* zero capacity */
    s = att1_aimu_mem_create(0, 0, &m);
    EXPECT(s == ATT1_ERR_INVALID_ARG, "create: zero capacity -> INVALID_ARG");
    EXPECT(m == NULL, "create: zero capacity m stays NULL");

    /* valid create */
    s = att1_aimu_mem_create(3, 1024 * 1024, &m);
    EXPECT(s == ATT1_OK, "create: valid -> ATT1_OK");
    EXPECT(m != NULL, "create: valid m != NULL");
    EXPECT(m->magic == ATT1_AIMU_MEM_MAGIC, "create: magic set");
    EXPECT(m->tile_id == 3, "create: tile_id set");
    EXPECT(m->capacity_bytes == 1024 * 1024, "create: capacity_bytes set");
    EXPECT(m->alloc_count == 0, "create: alloc_count zero");

    /* destroy clears magic */
    att1_aimu_mem_destroy(m);

    /* destroy NULL is safe */
    att1_aimu_mem_destroy(NULL);
    EXPECT(1, "destroy: NULL safe");
}

/* =========================================================================
 * 2. test_alloc_free_simple
 * ====================================================================== */

static void test_alloc_free_simple(void)
{
    att1_aimu_mem *m = NULL;
    att1_aimu_mem_create(0, 4096, &m);

    uint64_t addr = 0;
    uint32_t id   = 0;
    att1_status_t s = att1_aimu_mem_alloc_range(
        m, ATT1_AIMU_MEM_REGION_TENSOR, 256, 0,
        ATT1_AIMU_MEM_DTYPE_F32, ATT1_AIMU_MEM_FLAG_NONE,
        "weight0", &addr, &id);

    EXPECT(s == ATT1_OK, "alloc_simple: ATT1_OK");
    EXPECT(id != ATT1_AIMU_MEM_INVALID_ID, "alloc_simple: id valid");
    EXPECT(addr < 4096, "alloc_simple: addr within capacity");

    /* free it */
    s = att1_aimu_mem_free(m, id);
    EXPECT(s == ATT1_OK, "free_simple: ATT1_OK");

    /* free unknown id */
    s = att1_aimu_mem_free(m, 9999u);
    EXPECT(s == ATT1_ERR_NOT_FOUND, "free: unknown id -> NOT_FOUND");

    att1_aimu_mem_destroy(m);
}

/* =========================================================================
 * 3. test_alignment_honored
 * ====================================================================== */

static void test_alignment_honored(void)
{
    att1_aimu_mem *m = NULL;
    att1_aimu_mem_create(0, 64 * 1024, &m);

    /* Request 4096-byte alignment */
    uint64_t addr = 0;
    att1_status_t s = att1_aimu_mem_alloc_range(
        m, ATT1_AIMU_MEM_REGION_STAGING, 512, 4096,
        ATT1_AIMU_MEM_DTYPE_NONE, ATT1_AIMU_MEM_FLAG_DMA_TARGET,
        NULL, &addr, NULL);

    EXPECT(s == ATT1_OK, "alignment: alloc ok");
    EXPECT((addr % 4096) == 0, "alignment: 4096-byte aligned");

    /* alignment 0 -> uses MIN_ALIGN */
    uint64_t addr2 = 0;
    s = att1_aimu_mem_alloc_range(
        m, ATT1_AIMU_MEM_REGION_TENSOR, 64, 0,
        ATT1_AIMU_MEM_DTYPE_F32, ATT1_AIMU_MEM_FLAG_NONE,
        NULL, &addr2, NULL);
    EXPECT(s == ATT1_OK, "alignment: default align ok");
    EXPECT((addr2 % ATT1_AIMU_MEM_MIN_ALIGN) == 0, "alignment: min_align honored");

    /* invalid alignment: non-power-of-two */
    s = att1_aimu_mem_alloc_range(
        m, ATT1_AIMU_MEM_REGION_TENSOR, 64, 100,
        ATT1_AIMU_MEM_DTYPE_F32, ATT1_AIMU_MEM_FLAG_NONE,
        NULL, NULL, NULL);
    EXPECT(s == ATT1_ERR_INVALID_ARG, "alignment: non-pow2 -> INVALID_ARG");

    /* alignment below minimum */
    s = att1_aimu_mem_alloc_range(
        m, ATT1_AIMU_MEM_REGION_TENSOR, 64, 8,
        ATT1_AIMU_MEM_DTYPE_F32, ATT1_AIMU_MEM_FLAG_NONE,
        NULL, NULL, NULL);
    EXPECT(s == ATT1_ERR_INVALID_ARG, "alignment: below min -> INVALID_ARG");

    /* alignment above maximum */
    s = att1_aimu_mem_alloc_range(
        m, ATT1_AIMU_MEM_REGION_TENSOR, 64,
        ATT1_AIMU_MEM_MAX_ALIGN * 2,
        ATT1_AIMU_MEM_DTYPE_F32, ATT1_AIMU_MEM_FLAG_NONE,
        NULL, NULL, NULL);
    EXPECT(s == ATT1_ERR_INVALID_ARG, "alignment: above max -> INVALID_ARG");

    att1_aimu_mem_destroy(m);
}

/* =========================================================================
 * 4. test_no_overlap
 * ====================================================================== */

static void test_no_overlap(void)
{
    att1_aimu_mem *m = NULL;
    att1_aimu_mem_create(0, 256 * 1024, &m);

    /* Allocate 8 blocks of 4096 bytes */
#define N_ALLOCS 8
    uint64_t addrs[N_ALLOCS];
    uint32_t ids[N_ALLOCS];
    const uint64_t block = 4096;

    for (int i = 0; i < N_ALLOCS; i++) {
        att1_status_t s = att1_aimu_mem_alloc_range(
            m, ATT1_AIMU_MEM_REGION_TENSOR, block, 0,
            ATT1_AIMU_MEM_DTYPE_F32, ATT1_AIMU_MEM_FLAG_NONE,
            NULL, &addrs[i], &ids[i]);
        EXPECT(s == ATT1_OK, "no_overlap: alloc ok");
    }

    /* Verify no pair overlaps: [a, a+block) ∩ [b, b+block) = ∅ */
    int overlap = 0;
    for (int i = 0; i < N_ALLOCS; i++) {
        for (int j = i + 1; j < N_ALLOCS; j++) {
            uint64_t a_end = addrs[i] + block;
            uint64_t b_end = addrs[j] + block;
            if (addrs[i] < b_end && addrs[j] < a_end) overlap = 1;
        }
    }
    EXPECT(!overlap, "no_overlap: no pair overlaps");

    att1_aimu_mem_destroy(m);
#undef N_ALLOCS
}

/* =========================================================================
 * 5. test_out_of_capacity
 * ====================================================================== */

static void test_out_of_capacity(void)
{
    att1_aimu_mem *m = NULL;
    att1_aimu_mem_create(0, 1024, &m);

    /* Allocate 1024 bytes — fills capacity exactly */
    uint32_t id = 0;
    att1_status_t s = att1_aimu_mem_alloc_range(
        m, ATT1_AIMU_MEM_REGION_TENSOR, 1024, 0,
        ATT1_AIMU_MEM_DTYPE_F32, ATT1_AIMU_MEM_FLAG_NONE,
        NULL, NULL, &id);
    EXPECT(s == ATT1_OK, "capacity: fill exact ok");

    /* One more byte should fail */
    s = att1_aimu_mem_alloc_range(
        m, ATT1_AIMU_MEM_REGION_TENSOR, 1, 0,
        ATT1_AIMU_MEM_DTYPE_F32, ATT1_AIMU_MEM_FLAG_NONE,
        NULL, NULL, NULL);
    EXPECT(s == ATT1_ERR_OOM, "capacity: over capacity -> OOM");

    att1_aimu_mem_destroy(m);
}

/* =========================================================================
 * 6. test_zero_size_fails
 * ====================================================================== */

static void test_zero_size_fails(void)
{
    att1_aimu_mem *m = NULL;
    att1_aimu_mem_create(0, 4096, &m);

    att1_status_t s = att1_aimu_mem_alloc_range(
        m, ATT1_AIMU_MEM_REGION_TENSOR, 0, 0,
        ATT1_AIMU_MEM_DTYPE_F32, ATT1_AIMU_MEM_FLAG_NONE,
        NULL, NULL, NULL);
    EXPECT(s == ATT1_ERR_INVALID_ARG, "zero_size: -> INVALID_ARG");

    att1_aimu_mem_destroy(m);
}

/* =========================================================================
 * 7. test_invalid_alignment (covered in alignment test above, also here)
 * ====================================================================== */

static void test_invalid_alignment(void)
{
    att1_aimu_mem *m = NULL;
    att1_aimu_mem_create(0, 4096, &m);

    /* alignment = 3 (not power of two) */
    att1_status_t s = att1_aimu_mem_alloc_range(
        m, ATT1_AIMU_MEM_REGION_TENSOR, 64, 3,
        ATT1_AIMU_MEM_DTYPE_F32, ATT1_AIMU_MEM_FLAG_NONE,
        NULL, NULL, NULL);
    EXPECT(s == ATT1_ERR_INVALID_ARG, "invalid_align: 3 -> INVALID_ARG");

    att1_aimu_mem_destroy(m);
}

/* =========================================================================
 * 8. test_unknown_region_type
 * ====================================================================== */

static void test_unknown_region_type(void)
{
    att1_aimu_mem *m = NULL;
    att1_aimu_mem_create(0, 4096, &m);

    att1_status_t s = att1_aimu_mem_alloc_range(
        m, (att1_aimu_mem_region_type)ATT1_AIMU_MEM_REGION_COUNT, 64, 0,
        ATT1_AIMU_MEM_DTYPE_NONE, ATT1_AIMU_MEM_FLAG_NONE,
        NULL, NULL, NULL);
    EXPECT(s == ATT1_ERR_INVALID_ARG, "unknown_region: COUNT -> INVALID_ARG");

    /* out-of-range positive */
    s = att1_aimu_mem_alloc_range(
        m, (att1_aimu_mem_region_type)255, 64, 0,
        ATT1_AIMU_MEM_DTYPE_NONE, ATT1_AIMU_MEM_FLAG_NONE,
        NULL, NULL, NULL);
    EXPECT(s == ATT1_ERR_INVALID_ARG, "unknown_region: 255 -> INVALID_ARG");

    att1_aimu_mem_destroy(m);
}

/* =========================================================================
 * 9. test_double_free_fails
 * ====================================================================== */

static void test_double_free_fails(void)
{
    att1_aimu_mem *m = NULL;
    att1_aimu_mem_create(0, 4096, &m);

    uint32_t id = 0;
    att1_aimu_mem_alloc_range(
        m, ATT1_AIMU_MEM_REGION_TENSOR, 128, 0,
        ATT1_AIMU_MEM_DTYPE_F32, ATT1_AIMU_MEM_FLAG_NONE,
        NULL, NULL, &id);

    att1_status_t s = att1_aimu_mem_free(m, id);
    EXPECT(s == ATT1_OK, "double_free: first free ok");

    s = att1_aimu_mem_free(m, id);
    EXPECT(s == ATT1_ERR_STATE, "double_free: second free -> ERR_STATE");

    att1_aimu_mem_destroy(m);
}

/* =========================================================================
 * 10. test_query_by_id
 * ====================================================================== */

static void test_query_by_id(void)
{
    att1_aimu_mem *m = NULL;
    att1_aimu_mem_create(0, 64 * 1024, &m);

    uint64_t addr = 0;
    uint32_t id   = 0;
    att1_aimu_mem_alloc_range(
        m, ATT1_AIMU_MEM_REGION_KV_CACHE, 1024, 0,
        ATT1_AIMU_MEM_DTYPE_Q8, ATT1_AIMU_MEM_FLAG_NONE,
        "kvcache0", &addr, &id);

    const att1_aimu_mem_alloc *rec = NULL;
    att1_status_t s = att1_aimu_mem_query_by_id(m, id, &rec);
    EXPECT(s == ATT1_OK, "query_id: ok");
    EXPECT(rec != NULL, "query_id: rec non-null");
    EXPECT(rec->alloc_id == id, "query_id: id matches");
    EXPECT(rec->base_address == addr, "query_id: addr matches");
    EXPECT(rec->byte_size == 1024, "query_id: size matches");
    EXPECT(rec->region_type == ATT1_AIMU_MEM_REGION_KV_CACHE, "query_id: region ok");
    EXPECT(strcmp(rec->name, "kvcache0") == 0, "query_id: name matches");

    /* query unknown id */
    s = att1_aimu_mem_query_by_id(m, 9999u, &rec);
    EXPECT(s == ATT1_ERR_NOT_FOUND, "query_id: unknown -> NOT_FOUND");

    att1_aimu_mem_destroy(m);
}

/* =========================================================================
 * 11. test_query_by_address
 * ====================================================================== */

static void test_query_by_address(void)
{
    att1_aimu_mem *m = NULL;
    att1_aimu_mem_create(0, 64 * 1024, &m);

    uint64_t addr = 0;
    uint32_t id   = 0;
    att1_aimu_mem_alloc_range(
        m, ATT1_AIMU_MEM_REGION_TRACE_BUFFER, 512, 0,
        ATT1_AIMU_MEM_DTYPE_NONE, ATT1_AIMU_MEM_FLAG_NONE,
        NULL, &addr, &id);

    const att1_aimu_mem_alloc *rec = NULL;

    /* query at exact base */
    att1_status_t s = att1_aimu_mem_query_by_address(m, addr, &rec);
    EXPECT(s == ATT1_OK, "query_addr: base addr found");
    EXPECT(rec->alloc_id == id, "query_addr: id matches");

    /* query at interior byte */
    s = att1_aimu_mem_query_by_address(m, addr + 255, &rec);
    EXPECT(s == ATT1_OK, "query_addr: interior found");

    /* query at end (exclusive) — should not find */
    s = att1_aimu_mem_query_by_address(m, addr + 512, &rec);
    EXPECT(s == ATT1_ERR_NOT_FOUND, "query_addr: past end -> NOT_FOUND");

    att1_aimu_mem_destroy(m);
}

/* =========================================================================
 * 12. test_range_valid
 * ====================================================================== */

static void test_range_valid(void)
{
    att1_aimu_mem *m = NULL;
    att1_aimu_mem_create(0, 64 * 1024, &m);

    uint64_t addr = 0;
    att1_aimu_mem_alloc_range(
        m, ATT1_AIMU_MEM_REGION_DMA_BUFFER, 256, 0,
        ATT1_AIMU_MEM_DTYPE_NONE, ATT1_AIMU_MEM_FLAG_DMA_TARGET,
        NULL, &addr, NULL);

    /* entire range valid */
    EXPECT(att1_aimu_mem_range_valid(m, addr, 256), "range_valid: exact range ok");

    /* sub-range valid */
    EXPECT(att1_aimu_mem_range_valid(m, addr + 64, 64), "range_valid: sub-range ok");

    /* range extends beyond allocation */
    EXPECT(!att1_aimu_mem_range_valid(m, addr, 257), "range_valid: over-end fail");

    /* range before allocation */
    if (addr > 0) {
        EXPECT(!att1_aimu_mem_range_valid(m, addr - 1, 1),
               "range_valid: before alloc fail");
    } else {
        EXPECT(1, "range_valid: skip before-alloc check (addr==0)");
    }

    /* size == 0 */
    EXPECT(!att1_aimu_mem_range_valid(m, addr, 0), "range_valid: size 0 fail");

    att1_aimu_mem_destroy(m);
}

/* =========================================================================
 * 13. test_reset_clears
 * ====================================================================== */

static void test_reset_clears(void)
{
    att1_aimu_mem *m = NULL;
    att1_aimu_mem_create(0, 64 * 1024, &m);

    uint32_t id = 0;
    att1_aimu_mem_alloc_range(
        m, ATT1_AIMU_MEM_REGION_TENSOR, 1024, 0,
        ATT1_AIMU_MEM_DTYPE_F32, ATT1_AIMU_MEM_FLAG_NONE,
        NULL, NULL, &id);

    att1_status_t s = att1_aimu_mem_reset(m);
    EXPECT(s == ATT1_OK, "reset: ok");
    EXPECT(m->alloc_count == 0, "reset: alloc_count zeroed");

    /* query for the old id should fail */
    const att1_aimu_mem_alloc *rec = NULL;
    s = att1_aimu_mem_query_by_id(m, id, &rec);
    EXPECT(s == ATT1_ERR_NOT_FOUND, "reset: old id not found");

    /* can allocate again after reset */
    uint32_t id2 = 0;
    s = att1_aimu_mem_alloc_range(
        m, ATT1_AIMU_MEM_REGION_TENSOR, 512, 0,
        ATT1_AIMU_MEM_DTYPE_F32, ATT1_AIMU_MEM_FLAG_NONE,
        NULL, NULL, &id2);
    EXPECT(s == ATT1_OK, "reset: re-alloc ok after reset");

    att1_aimu_mem_destroy(m);
}

/* =========================================================================
 * 14. test_frag_deterministic
 * ====================================================================== */

static void test_frag_deterministic(void)
{
    att1_aimu_mem *m = NULL;
    att1_aimu_mem_create(0, 16 * 1024, &m);

    /* Allocate 4 blocks, free the 2nd — creates a hole */
    uint32_t ids[4];
    for (int i = 0; i < 4; i++) {
        att1_aimu_mem_alloc_range(
            m, ATT1_AIMU_MEM_REGION_TENSOR, 1024, 0,
            ATT1_AIMU_MEM_DTYPE_F32, ATT1_AIMU_MEM_FLAG_NONE,
            NULL, NULL, &ids[i]);
    }
    att1_aimu_mem_free(m, ids[1]);

    att1_aimu_mem_frag f1, f2;
    att1_aimu_mem_get_frag(m, &f1);
    att1_aimu_mem_get_frag(m, &f2);

    EXPECT(f1.used_bytes == f2.used_bytes, "frag: deterministic used_bytes");
    EXPECT(f1.free_bytes == f2.free_bytes, "frag: deterministic free_bytes");
    EXPECT(f1.largest_free_block == f2.largest_free_block,
           "frag: deterministic largest_free_block");
    EXPECT(f1.fragmentation_pct == f2.fragmentation_pct,
           "frag: deterministic pct");

    /* used = 3 * 1024; free = capacity - used */
    EXPECT(f1.used_bytes == 3 * 1024, "frag: used_bytes correct");
    EXPECT(f1.free_bytes == 16 * 1024 - 3 * 1024, "frag: free_bytes correct");
    EXPECT(f1.allocation_count == 3, "frag: allocation_count 3");

    /* Render to /dev/null — must not crash */
    FILE *nul = fopen("/dev/null", "w");
    if (nul) {
        att1_aimu_mem_render(m, nul);
        fclose(nul);
    }
    EXPECT(1, "frag: render no crash");

    att1_aimu_mem_destroy(m);
}

/* =========================================================================
 * 15. test_large_capacity_no_alloc
 * ====================================================================== */

static void test_large_capacity_no_alloc(void)
{
    /* 64 GiB metadata capacity — must not allocate 64 GiB of RAM */
    const uint64_t gib64 = UINT64_C(64) * 1024 * 1024 * 1024;
    att1_aimu_mem *m = NULL;
    att1_status_t s = att1_aimu_mem_create(7, gib64, &m);
    EXPECT(s == ATT1_OK, "large_cap: create 64 GiB ok");
    EXPECT(m != NULL, "large_cap: m non-null");
    EXPECT(m->capacity_bytes == gib64, "large_cap: capacity_bytes matches");

    /* Allocate a small region — should succeed from the huge address space */
    uint64_t addr = 0;
    uint32_t id   = 0;
    s = att1_aimu_mem_alloc_range(
        m, ATT1_AIMU_MEM_REGION_TENSOR, 4096, 0,
        ATT1_AIMU_MEM_DTYPE_F32, ATT1_AIMU_MEM_FLAG_NONE,
        "tiny", &addr, &id);
    EXPECT(s == ATT1_OK, "large_cap: small alloc ok");
    EXPECT(addr < gib64, "large_cap: addr within capacity");

    att1_aimu_mem_frag frag;
    att1_aimu_mem_get_frag(m, &frag);
    EXPECT(frag.used_bytes == 4096, "large_cap: used_bytes 4096");
    EXPECT(frag.free_bytes == gib64 - 4096, "large_cap: free_bytes correct");
    /* fragmentation should be near 0 for a single allocation */
    EXPECT(frag.fragmentation_pct <= 1, "large_cap: frag near zero");

    att1_aimu_mem_destroy(m);
    EXPECT(1, "large_cap: destroy ok (no giant free)");
}

/* =========================================================================
 * 16. test_no_cuda_dep
 * ====================================================================== */

static void test_no_cuda_dep(void)
{
#ifdef ATT1_ENABLE_CUDA
    EXPECT(1, "no_cuda_dep: CUDA enabled (opt-in path ok)");
#else
    EXPECT(1, "no_cuda_dep: no CUDA dependency in default build");
#endif
}

/* =========================================================================
 * main
 * ====================================================================== */

int main(void)
{
    test_create_destroy();
    test_alloc_free_simple();
    test_alignment_honored();
    test_no_overlap();
    test_out_of_capacity();
    test_zero_size_fails();
    test_invalid_alignment();
    test_unknown_region_type();
    test_double_free_fails();
    test_query_by_id();
    test_query_by_address();
    test_range_valid();
    test_reset_clears();
    test_frag_deterministic();
    test_large_capacity_no_alloc();
    test_no_cuda_dep();

    printf("\naimu_mem: %d PASS  %d FAIL\n", g_pass, g_fail);
    return (g_fail > 0) ? 1 : 0;
}
