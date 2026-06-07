/*
 * test_aimu_dma.c  —  Unit tests for the AIMU DMA descriptor simulator (M107)
 */

#include "att1_aimu_dma.h"
#include "att1_status.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>

/* -------------------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------------- */

#define PASS(name) do { printf("PASS: aimu_dma: %s\n", (name)); } while (0)
#define FAIL(name) do { printf("FAIL: aimu_dma: %s\n", (name)); return 1; } while (0)
#define REQUIRE(cond, name) do { if (!(cond)) { FAIL(name); } } while (0)

/* Base aligned addresses used across tests. */
#define HOST_BASE   UINT64_C(0x0000000000010000)  /* 64-KiB, 64-aligned */
#define DEVICE_BASE UINT64_C(0x0000000080000000)  /* 2 GiB, 64-aligned */
#define HOST_SIZE   UINT64_C(0x0000000010000000)  /* 256 MiB */
#define DEVICE_SIZE UINT64_C(0x0000000040000000)  /* 1 GiB */

/*
 * Return a minimal valid host-to-device descriptor.
 * All addresses are 64-aligned; byte_length is 4 KiB; dtype is F32.
 */
static att1_aimu_dma_desc make_h2d(void)
{
    att1_aimu_dma_desc d;
    memset(&d, 0, sizeof(d));
    d.host_addr   = HOST_BASE;
    d.device_addr = DEVICE_BASE;
    d.byte_length = 4096u;
    d.direction   = (uint8_t)ATT1_AIMU_DMA_HOST_TO_DEVICE;
    d.dtype       = ATT1_AIMU_DMA_DTYPE_F32;
    return d;
}

static att1_aimu_dma_desc make_d2h(void)
{
    att1_aimu_dma_desc d;
    memset(&d, 0, sizeof(d));
    d.host_addr   = HOST_BASE;
    d.device_addr = DEVICE_BASE;
    d.byte_length = 4096u;
    d.direction   = (uint8_t)ATT1_AIMU_DMA_DEVICE_TO_HOST;
    d.dtype       = ATT1_AIMU_DMA_DTYPE_F32;
    return d;
}

static att1_aimu_dma_desc make_d2d(void)
{
    att1_aimu_dma_desc d;
    memset(&d, 0, sizeof(d));
    /* src and dst must not overlap; separate by at least byte_length */
    d.src_device_addr = DEVICE_BASE;
    d.dst_device_addr = DEVICE_BASE + UINT64_C(0x100000); /* +1 MiB */
    d.byte_length = 4096u;
    d.direction   = (uint8_t)ATT1_AIMU_DMA_DEVICE_TO_DEVICE;
    d.dtype       = ATT1_AIMU_DMA_DTYPE_F32;
    return d;
}

/* Helper: create simulator with host and device regions registered. */
static att1_aimu_dma *make_sim_with_regions(void)
{
    att1_aimu_dma *sim = NULL;
    att1_aimu_dma_create(&sim);
    att1_aimu_dma_register_host_region(sim, HOST_BASE, HOST_SIZE);
    att1_aimu_dma_register_device_region(sim, DEVICE_BASE, DEVICE_SIZE);
    return sim;
}

/* -------------------------------------------------------------------------
 * Tests
 * ---------------------------------------------------------------------- */

/* Lifecycle: create / destroy / double-free-safe */
static int test_create_destroy(void)
{
    att1_aimu_dma *sim = NULL;
    att1_status_t  st  = att1_aimu_dma_create(&sim);

    REQUIRE(st  == ATT1_OK,             "create: returns OK");
    REQUIRE(sim != NULL,                "create: out non-null");
    REQUIRE(sim->magic == ATT1_AIMU_DMA_MAGIC, "create: magic set");

    att1_aimu_dma_destroy(sim);
    att1_aimu_dma_destroy(NULL);  /* must not crash */

    PASS("create_destroy");
    return 0;
}

/* NULL out-pointer is rejected */
static int test_create_null_out(void)
{
    att1_status_t st = att1_aimu_dma_create(NULL);
    REQUIRE(st == ATT1_ERR_INVALID_ARG, "create_null_out: rejected");
    PASS("create_null_out");
    return 0;
}

/* Valid H2D descriptor passes validate and submit */
static int test_valid_h2d(void)
{
    att1_aimu_dma     *sim = NULL;
    att1_aimu_dma_desc d   = make_h2d();

    att1_aimu_dma_create(&sim);

    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_OK, "valid_h2d: validate OK");
    REQUIRE(att1_aimu_dma_submit(sim, &d)   == ATT1_OK, "valid_h2d: submit OK");

    att1_aimu_dma_destroy(sim);
    PASS("valid_h2d");
    return 0;
}

/* Valid D2H descriptor passes */
static int test_valid_d2h(void)
{
    att1_aimu_dma     *sim = NULL;
    att1_aimu_dma_desc d   = make_d2h();

    att1_aimu_dma_create(&sim);

    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_OK, "valid_d2h: validate OK");
    REQUIRE(att1_aimu_dma_submit(sim, &d)   == ATT1_OK, "valid_d2h: submit OK");

    att1_aimu_dma_destroy(sim);
    PASS("valid_d2h");
    return 0;
}

/* Valid D2D descriptor passes */
static int test_valid_d2d(void)
{
    att1_aimu_dma     *sim = NULL;
    att1_aimu_dma_desc d   = make_d2d();

    att1_aimu_dma_create(&sim);

    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_OK, "valid_d2d: validate OK");
    REQUIRE(att1_aimu_dma_submit(sim, &d)   == ATT1_OK, "valid_d2d: submit OK");

    att1_aimu_dma_destroy(sim);
    PASS("valid_d2d");
    return 0;
}

/* byte_length == 0 is rejected */
static int test_zero_byte_length(void)
{
    att1_aimu_dma     *sim = NULL;
    att1_aimu_dma_desc d   = make_h2d();

    att1_aimu_dma_create(&sim);
    d.byte_length = 0u;

    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_ERR_INVALID_ARG,
            "zero_byte_length: rejected");

    att1_aimu_dma_destroy(sim);
    PASS("zero_byte_length");
    return 0;
}

/* Exactly ATT1_AIMU_DMA_MAX_TRANSFER_BYTES passes; one byte over fails */
static int test_max_byte_length(void)
{
    att1_aimu_dma     *sim = NULL;
    att1_aimu_dma_desc d   = make_h2d();

    att1_aimu_dma_create(&sim);

    d.byte_length = ATT1_AIMU_DMA_MAX_TRANSFER_BYTES;
    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_OK,
            "max_byte_length: max passes");

    d.byte_length = ATT1_AIMU_DMA_MAX_TRANSFER_BYTES + 1u;
    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_ERR_INVALID_ARG,
            "max_byte_length: max+1 rejected");

    att1_aimu_dma_destroy(sim);
    PASS("max_byte_length");
    return 0;
}

/* Unaligned host address fails */
static int test_unaligned_host_addr(void)
{
    att1_aimu_dma     *sim = NULL;
    att1_aimu_dma_desc d   = make_h2d();

    att1_aimu_dma_create(&sim);

    d.host_addr = HOST_BASE + 1u;  /* misaligned by 1 byte */

    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_ERR_INVALID_ARG,
            "unaligned_host_addr: rejected");

    att1_aimu_dma_destroy(sim);
    PASS("unaligned_host_addr");
    return 0;
}

/* Unaligned device address fails */
static int test_unaligned_device_addr(void)
{
    att1_aimu_dma     *sim = NULL;
    att1_aimu_dma_desc d   = make_h2d();

    att1_aimu_dma_create(&sim);

    d.device_addr = DEVICE_BASE + 32u;  /* 32-byte aligned but not 64 */

    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_ERR_INVALID_ARG,
            "unaligned_device_addr: rejected");

    att1_aimu_dma_destroy(sim);
    PASS("unaligned_device_addr");
    return 0;
}

/* host_addr + byte_length wraps around uint64_t */
static int test_host_range_overflow(void)
{
    att1_aimu_dma     *sim = NULL;
    att1_aimu_dma_desc d   = make_h2d();

    att1_aimu_dma_create(&sim);

    /* UINT64_MAX - 63 is the highest 64-aligned address; adding 128 wraps */
    d.host_addr   = UINT64_MAX - UINT64_C(63);  /* 0xFFFFFFFFFFFFFFC0 */
    d.byte_length = 128u;

    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_ERR_INVALID_ARG,
            "host_range_overflow: rejected");

    att1_aimu_dma_destroy(sim);
    PASS("host_range_overflow");
    return 0;
}

/* device_addr + byte_length wraps around uint64_t */
static int test_device_range_overflow(void)
{
    att1_aimu_dma     *sim = NULL;
    att1_aimu_dma_desc d   = make_h2d();

    att1_aimu_dma_create(&sim);

    d.device_addr = UINT64_MAX - UINT64_C(63);
    d.byte_length = 128u;

    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_ERR_INVALID_ARG,
            "device_range_overflow: rejected");

    att1_aimu_dma_destroy(sim);
    PASS("device_range_overflow");
    return 0;
}

/* Transfer outside registered host region is rejected; inside passes */
static int test_range_check_host(void)
{
    att1_aimu_dma     *sim = make_sim_with_regions();
    att1_aimu_dma_desc d   = make_h2d();

    /* This transfer is inside the registered region — should pass. */
    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_OK,
            "range_check_host: inside region OK");

    /* Place host_addr just beyond the registered region. */
    d.host_addr = HOST_BASE + HOST_SIZE;   /* first address past the end */
    d.byte_length = 64u;
    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_ERR_INVALID_ARG,
            "range_check_host: outside region rejected");

    att1_aimu_dma_destroy(sim);
    PASS("range_check_host");
    return 0;
}

/* Transfer outside registered device region is rejected */
static int test_range_check_device(void)
{
    att1_aimu_dma     *sim = make_sim_with_regions();
    att1_aimu_dma_desc d   = make_h2d();

    /* Inside — should pass. */
    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_OK,
            "range_check_device: inside OK");

    /* Past the end of the device region. */
    d.device_addr = DEVICE_BASE + DEVICE_SIZE;
    d.byte_length = 64u;
    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_ERR_INVALID_ARG,
            "range_check_device: outside rejected");

    att1_aimu_dma_destroy(sim);
    PASS("range_check_device");
    return 0;
}

/* Unknown dtype is rejected */
static int test_unknown_dtype(void)
{
    att1_aimu_dma     *sim = NULL;
    att1_aimu_dma_desc d   = make_h2d();

    att1_aimu_dma_create(&sim);

    d.dtype = 0xFFu;
    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_ERR_INVALID_ARG,
            "unknown_dtype: rejected");

    att1_aimu_dma_destroy(sim);
    PASS("unknown_dtype");
    return 0;
}

/* Q4 payload not a multiple of group_bytes fails */
static int test_q4_misaligned_payload(void)
{
    att1_aimu_dma     *sim = NULL;
    att1_aimu_dma_desc d   = make_h2d();

    att1_aimu_dma_create(&sim);

    d.dtype            = ATT1_AIMU_DMA_DTYPE_Q4;
    d.quant_group_size = 32u;    /* group_bytes = 16 */
    d.byte_length      = 100u;   /* 100 % 16 != 0 */

    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_ERR_INVALID_ARG,
            "q4_misaligned_payload: rejected");

    /* Now align it — should pass. */
    d.byte_length = 96u;  /* 96 % 16 == 0 */
    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_OK,
            "q4_misaligned_payload: aligned passes");

    att1_aimu_dma_destroy(sim);
    PASS("q4_misaligned_payload");
    return 0;
}

/* Q4 with unsupported group_size (not 32 or 64) fails */
static int test_q4_bad_group_size(void)
{
    att1_aimu_dma     *sim = NULL;
    att1_aimu_dma_desc d   = make_h2d();

    att1_aimu_dma_create(&sim);

    d.dtype            = ATT1_AIMU_DMA_DTYPE_Q4;
    d.quant_group_size = 16u;    /* only 32 and 64 are valid */
    d.byte_length      = 64u;

    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_ERR_INVALID_ARG,
            "q4_bad_group_size: 16 rejected");

    /* group_size == 64: group_bytes = 32; byte_length must be multiple of 32. */
    d.quant_group_size = 64u;
    d.byte_length      = 64u;    /* 64 % 32 == 0 */
    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_OK,
            "q4_bad_group_size: 64-group passes");

    att1_aimu_dma_destroy(sim);
    PASS("q4_bad_group_size");
    return 0;
}

/* Unknown flag bits are rejected */
static int test_unknown_flags(void)
{
    att1_aimu_dma     *sim = NULL;
    att1_aimu_dma_desc d   = make_h2d();

    att1_aimu_dma_create(&sim);

    d.flags = (uint16_t)(ATT1_AIMU_DMA_FLAG_VALID_MASK | 0x0010u);  /* bit 4 invalid */

    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_ERR_INVALID_ARG,
            "unknown_flags: bit 4 rejected");

    /* All valid flags together should pass. */
    d.flags = ATT1_AIMU_DMA_FLAG_VALID_MASK;
    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_OK,
            "unknown_flags: all valid flags OK");

    att1_aimu_dma_destroy(sim);
    PASS("unknown_flags");
    return 0;
}

/* Counters update deterministically for H2D, D2H, and D2D transfers */
static int test_counters_basic(void)
{
    att1_aimu_dma         *sim = NULL;
    att1_aimu_dma_counters  c;
    att1_aimu_dma_desc      d;

    att1_aimu_dma_create(&sim);

    /* H2D: 1024 bytes */
    d = make_h2d(); d.byte_length = 1024u;
    att1_aimu_dma_submit(sim, &d);

    /* D2H: 2048 bytes */
    d = make_d2h(); d.byte_length = 2048u;
    att1_aimu_dma_submit(sim, &d);

    /* D2D: 512 bytes */
    d = make_d2d(); d.byte_length = 512u;
    att1_aimu_dma_submit(sim, &d);

    att1_aimu_dma_get_counters(sim, &c);

    REQUIRE(c.dma_submitted          == 3u, "counters: submitted == 3");
    REQUIRE(c.dma_completed          == 3u, "counters: completed == 3");
    REQUIRE(c.dma_failed             == 0u, "counters: failed == 0");
    REQUIRE(c.bytes_host_to_device   == 1024u, "counters: h2d bytes");
    REQUIRE(c.bytes_device_to_host   == 2048u, "counters: d2h bytes");
    REQUIRE(c.bytes_device_to_device == 512u,  "counters: d2d bytes");

    att1_aimu_dma_destroy(sim);
    PASS("counters_basic");
    return 0;
}

/* Failure counters update correctly */
static int test_counters_failures(void)
{
    att1_aimu_dma         *sim = make_sim_with_regions();
    att1_aimu_dma_counters  c;
    att1_aimu_dma_desc      d;

    /* alignment failure */
    d = make_h2d(); d.host_addr = HOST_BASE + 1u;
    att1_aimu_dma_submit(sim, &d);

    /* range failure (outside registered host region) */
    d = make_h2d(); d.host_addr = HOST_BASE + HOST_SIZE;
    att1_aimu_dma_submit(sim, &d);

    /* unsupported_flags failure */
    d = make_h2d(); d.flags = 0xFF00u;
    att1_aimu_dma_submit(sim, &d);

    att1_aimu_dma_get_counters(sim, &c);

    REQUIRE(c.dma_submitted       == 3u, "fail_counters: submitted == 3");
    REQUIRE(c.dma_completed       == 0u, "fail_counters: completed == 0");
    REQUIRE(c.dma_failed          == 3u, "fail_counters: failed == 3");
    REQUIRE(c.alignment_failures  == 1u, "fail_counters: alignment == 1");
    REQUIRE(c.range_failures      == 1u, "fail_counters: range == 1");
    REQUIRE(c.unsupported_flags   == 1u, "fail_counters: unsupported_flags == 1");

    att1_aimu_dma_destroy(sim);
    PASS("counters_failures");
    return 0;
}

/* Overlapping D2D src/dst ranges fail and increment range_failures */
static int test_d2d_overlap(void)
{
    att1_aimu_dma         *sim = NULL;
    att1_aimu_dma_counters  c;
    att1_aimu_dma_desc      d;

    att1_aimu_dma_create(&sim);

    d = make_d2d();
    /* src=[0x1000, 0x2000), dst=[0x1040, 0x2040) — overlaps */
    d.src_device_addr = UINT64_C(0x1000);
    d.dst_device_addr = UINT64_C(0x1040);
    d.byte_length     = 4096u;

    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_ERR_INVALID_ARG,
            "d2d_overlap: validate rejected");
    att1_aimu_dma_submit(sim, &d);

    att1_aimu_dma_get_counters(sim, &c);
    REQUIRE(c.range_failures == 1u, "d2d_overlap: range_failures == 1");

    /* Non-overlapping: src=[0x1000, 0x2000), dst=[0x2000, 0x3000) — adjacent */
    d.src_device_addr = UINT64_C(0x1000);
    d.dst_device_addr = UINT64_C(0x2000);
    d.byte_length     = 4096u;
    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_OK,
            "d2d_overlap: adjacent ranges OK");

    att1_aimu_dma_destroy(sim);
    PASS("d2d_overlap");
    return 0;
}

/* validate() does not increment any counters */
static int test_validate_no_counters(void)
{
    att1_aimu_dma         *sim = NULL;
    att1_aimu_dma_counters  c;
    att1_aimu_dma_desc      d;

    att1_aimu_dma_create(&sim);

    /* Valid descriptor — validate only, no submit. */
    d = make_h2d();
    att1_aimu_dma_validate(sim, &d);

    /* Invalid descriptor — validate only, no submit. */
    d.byte_length = 0u;
    att1_aimu_dma_validate(sim, &d);

    att1_aimu_dma_get_counters(sim, &c);
    REQUIRE(c.dma_submitted == 0u, "validate_no_counters: submitted stays 0");
    REQUIRE(c.dma_failed    == 0u, "validate_no_counters: failed stays 0");

    att1_aimu_dma_destroy(sim);
    PASS("validate_no_counters");
    return 0;
}

/* NULL safety for all public API functions */
static int test_null_safety(void)
{
    att1_aimu_dma_desc     d;
    att1_aimu_dma_counters c;
    memset(&d, 0, sizeof(d));
    memset(&c, 0, sizeof(c));

    REQUIRE(att1_aimu_dma_validate(NULL, &d)  == ATT1_ERR_INVALID_ARG,
            "null: validate NULL sim");
    REQUIRE(att1_aimu_dma_validate((att1_aimu_dma *)&c, NULL) == ATT1_ERR_INVALID_ARG,
            "null: validate NULL desc");
    REQUIRE(att1_aimu_dma_submit(NULL, &d)    == ATT1_ERR_INVALID_ARG,
            "null: submit NULL sim");
    REQUIRE(att1_aimu_dma_submit((att1_aimu_dma *)&c, NULL) == ATT1_ERR_INVALID_ARG,
            "null: submit NULL desc");
    REQUIRE(att1_aimu_dma_get_counters(NULL, &c) == ATT1_ERR_INVALID_ARG,
            "null: get_counters NULL sim");
    REQUIRE(att1_aimu_dma_get_counters((att1_aimu_dma *)&c, NULL) == ATT1_ERR_INVALID_ARG,
            "null: get_counters NULL out");
    REQUIRE(att1_aimu_dma_reset_counters(NULL) == ATT1_ERR_INVALID_ARG,
            "null: reset_counters NULL sim");
    REQUIRE(att1_aimu_dma_register_host_region(NULL, HOST_BASE, 64u) == ATT1_ERR_INVALID_ARG,
            "null: register_host NULL sim");
    REQUIRE(att1_aimu_dma_register_device_region(NULL, DEVICE_BASE, 64u) == ATT1_ERR_INVALID_ARG,
            "null: register_device NULL sim");

    PASS("null_safety");
    return 0;
}

/* Invalid direction value is rejected */
static int test_invalid_direction(void)
{
    att1_aimu_dma     *sim = NULL;
    att1_aimu_dma_desc d   = make_h2d();

    att1_aimu_dma_create(&sim);

    d.direction = 3u;  /* valid range is 0–2 */
    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_ERR_INVALID_ARG,
            "invalid_direction: rejected");

    att1_aimu_dma_destroy(sim);
    PASS("invalid_direction");
    return 0;
}

/* Region registration and range checking work together */
static int test_region_registration(void)
{
    att1_aimu_dma     *sim = NULL;
    att1_aimu_dma_desc d;
    att1_status_t      st;

    att1_aimu_dma_create(&sim);

    /* Before registration: any aligned address is accepted. */
    d = make_h2d();
    d.host_addr   = UINT64_C(0xDEAD0000);
    d.device_addr = UINT64_C(0xBEEF0000);
    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_OK,
            "region_reg: pre-register permissive");

    /* Register regions. */
    st = att1_aimu_dma_register_host_region(sim, HOST_BASE, HOST_SIZE);
    REQUIRE(st == ATT1_OK, "region_reg: host region registered");

    st = att1_aimu_dma_register_device_region(sim, DEVICE_BASE, DEVICE_SIZE);
    REQUIRE(st == ATT1_OK, "region_reg: device region registered");

    /* Transfer within registered regions passes. */
    d = make_h2d();
    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_OK,
            "region_reg: within regions OK");

    /* Host address outside region fails. */
    d.host_addr = UINT64_C(0xDEAD0000);
    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_ERR_INVALID_ARG,
            "region_reg: host outside fails");

    att1_aimu_dma_destroy(sim);
    PASS("region_registration");
    return 0;
}

/* Valid Q4 descriptor passes */
static int test_q4_valid(void)
{
    att1_aimu_dma     *sim = NULL;
    att1_aimu_dma_desc d   = make_h2d();

    att1_aimu_dma_create(&sim);

    /* group_size=32: group_bytes=16; byte_length must be multiple of 16 */
    d.dtype            = ATT1_AIMU_DMA_DTYPE_Q4;
    d.quant_group_size = 32u;
    d.byte_length      = 4096u;  /* 4096 % 16 == 0 */

    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_OK,
            "q4_valid: group32 passes");

    /* group_size=64: group_bytes=32 */
    d.quant_group_size = 64u;
    d.byte_length      = 4096u;  /* 4096 % 32 == 0 */
    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_OK,
            "q4_valid: group64 passes");

    att1_aimu_dma_destroy(sim);
    PASS("q4_valid");
    return 0;
}

/* Q8 descriptor passes (quant_group_size is not checked for Q8) */
static int test_q8_valid(void)
{
    att1_aimu_dma     *sim = NULL;
    att1_aimu_dma_desc d   = make_h2d();

    att1_aimu_dma_create(&sim);

    d.dtype            = ATT1_AIMU_DMA_DTYPE_Q8;
    d.quant_group_size = 0u;
    d.byte_length      = 4096u;

    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_OK,
            "q8_valid: passes");

    att1_aimu_dma_destroy(sim);
    PASS("q8_valid");
    return 0;
}

/* reset_counters zeroes everything */
static int test_reset_counters(void)
{
    att1_aimu_dma         *sim = NULL;
    att1_aimu_dma_counters  c;
    att1_aimu_dma_desc      d;

    att1_aimu_dma_create(&sim);

    d = make_h2d();
    att1_aimu_dma_submit(sim, &d);

    att1_aimu_dma_get_counters(sim, &c);
    REQUIRE(c.dma_submitted == 1u, "reset_counters: before reset == 1");

    att1_aimu_dma_reset_counters(sim);
    att1_aimu_dma_get_counters(sim, &c);
    REQUIRE(c.dma_submitted        == 0u, "reset_counters: submitted cleared");
    REQUIRE(c.dma_completed        == 0u, "reset_counters: completed cleared");
    REQUIRE(c.bytes_host_to_device == 0u, "reset_counters: h2d bytes cleared");

    att1_aimu_dma_destroy(sim);
    PASS("reset_counters");
    return 0;
}

/* D2D unaligned src_device_addr fails */
static int test_d2d_unaligned_src(void)
{
    att1_aimu_dma     *sim = NULL;
    att1_aimu_dma_desc d   = make_d2d();

    att1_aimu_dma_create(&sim);

    d.src_device_addr = DEVICE_BASE + 1u;  /* not 64-aligned */

    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_ERR_INVALID_ARG,
            "d2d_unaligned_src: rejected");

    att1_aimu_dma_destroy(sim);
    PASS("d2d_unaligned_src");
    return 0;
}

/* D2D src_device_addr overflow */
static int test_d2d_src_overflow(void)
{
    att1_aimu_dma     *sim = NULL;
    att1_aimu_dma_desc d   = make_d2d();

    att1_aimu_dma_create(&sim);

    d.src_device_addr = UINT64_MAX - UINT64_C(63);  /* 64-aligned, near max */
    d.byte_length     = 128u;  /* wraps */

    REQUIRE(att1_aimu_dma_validate(sim, &d) == ATT1_ERR_INVALID_ARG,
            "d2d_src_overflow: rejected");

    att1_aimu_dma_destroy(sim);
    PASS("d2d_src_overflow");
    return 0;
}

/* Descriptor size static assert: just being compiled proves it passes. */
static int test_desc_size(void)
{
    /* If sizeof(att1_aimu_dma_desc) != 64, the static assert in the header
     * produces a compilation error, so this test cannot reach here. */
    REQUIRE(sizeof(att1_aimu_dma_desc) == 64u,
            "desc_size: 64 bytes");
    PASS("desc_size");
    return 0;
}

/* No CUDA symbol is required at link time */
static int test_no_cuda_dependency(void)
{
    /* If the test binary links (it must to run), there is no CUDA dependency. */
    PASS("no_cuda_dependency");
    return 0;
}

/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */

int main(void)
{
    int rc = 0;

    rc |= test_create_destroy();
    rc |= test_create_null_out();
    rc |= test_valid_h2d();
    rc |= test_valid_d2h();
    rc |= test_valid_d2d();
    rc |= test_zero_byte_length();
    rc |= test_max_byte_length();
    rc |= test_unaligned_host_addr();
    rc |= test_unaligned_device_addr();
    rc |= test_host_range_overflow();
    rc |= test_device_range_overflow();
    rc |= test_range_check_host();
    rc |= test_range_check_device();
    rc |= test_unknown_dtype();
    rc |= test_q4_misaligned_payload();
    rc |= test_q4_bad_group_size();
    rc |= test_unknown_flags();
    rc |= test_counters_basic();
    rc |= test_counters_failures();
    rc |= test_d2d_overlap();
    rc |= test_validate_no_counters();
    rc |= test_null_safety();
    rc |= test_invalid_direction();
    rc |= test_region_registration();
    rc |= test_q4_valid();
    rc |= test_q8_valid();
    rc |= test_reset_counters();
    rc |= test_d2d_unaligned_src();
    rc |= test_d2d_src_overflow();
    rc |= test_desc_size();
    rc |= test_no_cuda_dependency();

    return rc;
}
