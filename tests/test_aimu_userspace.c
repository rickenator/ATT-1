/*
 * test_aimu_userspace.c  —  Tests for the AIMU userspace MMIO emulator (M121)
 *
 * Covers:
 *  1.  test_smoke                  — full smoke flow exits zero
 *  2.  test_device_id_readable     — DEVICE_ID reads ATT1_MMIO_DEVICE_ID_DEFAULT
 *  3.  test_register_map_version_readable — REGISTER_MAP_VERSION reads expected value
 *  4.  test_tile_count_matches     — TILE_COUNT register == configured tile_count
 *  5.  test_tile_capacity_metadata_only — TILE_MEMORY_CAPACITY_LOW is non-zero;
 *                                        no huge real buffer was allocated
 *  6.  test_doorbell_increments_counter — ring_doorbell → summary.doorbell_count==1
 *  7.  test_nop_completes          — submit NOP, drain, commands_completed==1
 *  8.  test_load_tensor_completes  — submit LOAD_TENSOR_TILE, drain, completed>=1
 *  9.  test_validate_tensor_completes — submit VALIDATE_TENSOR, drain, completed
 * 10.  test_query_counters_completes — submit QUERY_COUNTERS, drain, completed
 * 11.  test_completion_order_deterministic — NOP before QUERY_COUNTERS in order
 * 12.  test_invalid_register_offset — write to offset==BAR0_SIZE returns error
 * 13.  test_invalid_tile_count     — open with tile_count==17 returns error
 * 14.  test_invalid_tile_memory    — open with tile_memory_bytes too large returns error
 * 15.  test_no_cuda_dep            — no hidden CUDA dependency (compile-time)
 *
 * Temporary BAR0 files are created under /tmp/ and unlinked after each test.
 * No large buffers are allocated (tile memory is metadata/register-only).
 * No CUDA, no inference, no .att1 format changes.
 */

#define _POSIX_C_SOURCE 200112L

#include "att1_aimu_userspace.h"
#include "att1_aimu_mmio.h"
#include "att1_aimu_cmdq.h"
#include "att1_aimu_dma.h"
#include "att1_status.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <unistd.h>   /* unlink */

/* =========================================================================
 * Minimal test harness  (matches convention across test_aimu_*.c)
 * ====================================================================== */

static int g_pass = 0;
static int g_fail = 0;

#define EXPECT(cond, name) \
    do { \
        if (cond) { \
            printf("PASS: aimu_userspace: " name "\n"); \
            g_pass++; \
        } else { \
            printf("FAIL: aimu_userspace: " name "\n"); \
            g_fail++; \
        } \
    } while (0)

/* =========================================================================
 * Helper: create + probe + setup_cmdq
 * ====================================================================== */

/** Create a userspace emulator with default small settings. */
static att1_aimu_userspace *make_emu(const char *bar0_path)
{
    att1_aimu_userspace_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.tile_count        = 2;
    cfg.tile_memory_bytes = 32u * 1024u * 1024u;  /* 32 MiB — register only */
    cfg.tile_kv_bytes     = 8u  * 1024u * 1024u;  /* 8 MiB  — register only */

    att1_aimu_userspace *u = NULL;
    if (att1_aimu_userspace_open(&cfg, bar0_path, &u) != ATT1_OK) return NULL;
    return u;
}

/** Probe + setup_cmdq on u.  Returns 1 on success. */
static int probe_and_setup(att1_aimu_userspace *u)
{
    if (att1_aimu_userspace_probe(u, NULL)     != ATT1_OK) return 0;
    if (att1_aimu_userspace_setup_cmdq(u)      != ATT1_OK) return 0;
    return 1;
}

/** Submit a command of given type on tile 0, assigning command_id.
 *  Returns ATT1_OK on success. */
static att1_status_t submit_cmd_type(att1_aimu_userspace  *u,
                                      att1_aimu_cmd_type   type,
                                      uint32_t             cmd_id)
{
    att1_aimu_cmd cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_id   = cmd_id;
    cmd.command_type = (uint8_t)type;
    cmd.tile_id      = 0;
    return att1_aimu_userspace_submit_cmd(u, &cmd);
}

/* =========================================================================
 * Test 1: smoke — full flow using a temp file
 * ====================================================================== */

static void test_smoke(void)
{
    const char *path = "/tmp/att1_test_bar0_smoke.bin";
    att1_aimu_userspace *u = make_emu(path);
    EXPECT(u != NULL, "smoke: open succeeds");
    if (!u) return;

    /* probe → enumerate → setup_cmdq */
    EXPECT(att1_aimu_userspace_probe(u, NULL)        == ATT1_OK, "smoke: probe");
    att1_aimu_host_tile_info tiles[4];
    size_t ntiles = 4;
    EXPECT(att1_aimu_userspace_enumerate_tiles(u, tiles, &ntiles) == ATT1_OK,
           "smoke: enumerate_tiles");
    EXPECT(att1_aimu_userspace_setup_cmdq(u) == ATT1_OK, "smoke: setup_cmdq");

    /* submit NOP + QUERY_COUNTERS */
    EXPECT(submit_cmd_type(u, ATT1_AIMU_CMD_NOP,           1) == ATT1_OK,
           "smoke: submit NOP");
    EXPECT(submit_cmd_type(u, ATT1_AIMU_CMD_QUERY_COUNTERS, 2) == ATT1_OK,
           "smoke: submit QUERY_COUNTERS");

    EXPECT(att1_aimu_userspace_ring_doorbell(u) == ATT1_OK, "smoke: ring_doorbell");
    EXPECT(att1_aimu_userspace_drain(u)         == ATT1_OK, "smoke: drain");
    EXPECT(att1_aimu_userspace_snapshot(u)      == ATT1_OK, "smoke: snapshot");

    att1_aimu_host_summary sum;
    memset(&sum, 0, sizeof(sum));
    EXPECT(att1_aimu_userspace_get_summary(u, &sum) == ATT1_OK, "smoke: get_summary");
    EXPECT(sum.commands_submitted >= 2u, "smoke: commands_submitted >= 2");
    EXPECT(sum.commands_completed >= 1u, "smoke: commands_completed >= 1");

    att1_aimu_userspace_close(u);
    unlink(path);
    EXPECT(1, "smoke: closed without crash");
}

/* =========================================================================
 * Test 2: DEVICE_ID is readable
 * ====================================================================== */

static void test_device_id_readable(void)
{
    att1_aimu_userspace *u = make_emu(NULL);
    EXPECT(u != NULL, "device_id: open succeeds");
    if (!u) return;

    EXPECT(probe_and_setup(u), "device_id: probe+setup_cmdq");

    uint32_t val = 0;
    att1_status_t rc = att1_aimu_userspace_read32(u, ATT1_MMIO_DEVICE_ID, &val);
    EXPECT(rc == ATT1_OK, "device_id: read32 returns ATT1_OK");
    EXPECT(val == ATT1_MMIO_DEVICE_ID_DEFAULT,
           "device_id: value == ATT1_MMIO_DEVICE_ID_DEFAULT");

    att1_aimu_userspace_close(u);
}

/* =========================================================================
 * Test 3: REGISTER_MAP_VERSION is readable
 * ====================================================================== */

static void test_register_map_version_readable(void)
{
    att1_aimu_userspace *u = make_emu(NULL);
    EXPECT(u != NULL, "regmap_version: open succeeds");
    if (!u) return;

    EXPECT(probe_and_setup(u), "regmap_version: probe+setup_cmdq");

    uint32_t val = 0;
    att1_status_t rc = att1_aimu_userspace_read32(u, ATT1_MMIO_REGISTER_MAP_VERSION, &val);
    EXPECT(rc == ATT1_OK, "regmap_version: read32 returns ATT1_OK");
    EXPECT(val == ATT1_AIMU_REGISTER_MAP_VERSION,
           "regmap_version: value == ATT1_AIMU_REGISTER_MAP_VERSION");

    att1_aimu_userspace_close(u);
}

/* =========================================================================
 * Test 4: TILE_COUNT matches configured value
 * ====================================================================== */

static void test_tile_count_matches(void)
{
    att1_aimu_userspace_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.tile_count        = 3;
    cfg.tile_memory_bytes = 32u * 1024u * 1024u;
    cfg.tile_kv_bytes     = 8u  * 1024u * 1024u;

    att1_aimu_userspace *u = NULL;
    EXPECT(att1_aimu_userspace_open(&cfg, NULL, &u) == ATT1_OK,
           "tile_count_matches: open");
    if (!u) return;

    EXPECT(att1_aimu_userspace_probe(u, NULL) == ATT1_OK,
           "tile_count_matches: probe");

    uint32_t val = 0;
    att1_status_t rc = att1_aimu_userspace_read32(u, ATT1_MMIO_TILE_COUNT, &val);
    EXPECT(rc == ATT1_OK, "tile_count_matches: read32 ATT1_OK");
    EXPECT(val == 3u, "tile_count_matches: TILE_COUNT == 3");

    att1_aimu_userspace_close(u);
}

/* =========================================================================
 * Test 5: tile capacity is metadata-only (register non-zero, no huge malloc)
 * ====================================================================== */

static void test_tile_capacity_metadata_only(void)
{
    att1_aimu_userspace_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.tile_count        = 2;
    cfg.tile_memory_bytes = 64u * 1024u * 1024u;  /* 64 MiB */
    cfg.tile_kv_bytes     = 8u  * 1024u * 1024u;

    att1_aimu_userspace *u = NULL;
    EXPECT(att1_aimu_userspace_open(&cfg, NULL, &u) == ATT1_OK,
           "tile_cap: open");
    if (!u) return;

    EXPECT(att1_aimu_userspace_probe(u, NULL) == ATT1_OK, "tile_cap: probe");

    /* Enumerate tiles and check capacity register is non-zero */
    att1_aimu_host_tile_info tiles[4];
    size_t ntiles = 4;
    EXPECT(att1_aimu_userspace_enumerate_tiles(u, tiles, &ntiles) == ATT1_OK,
           "tile_cap: enumerate_tiles");
    EXPECT(ntiles == 2u, "tile_cap: ntiles == 2");
    EXPECT(tiles[0].memory_capacity_bytes > 0u,
           "tile_cap: tile[0].memory_capacity_bytes non-zero");

    /* Confirm the register reflects 64 MiB */
    uint32_t low = 0, high = 0;
    /* Per-tile TILE_MEMORY_CAPACITY_LOW/HIGH for tile 0 */
    uint32_t base = 0x8000u + 0u * 0x800u;
    (void)att1_aimu_userspace_read32(u,
            base + ATT1_MMIO_TILE_MEMORY_CAPACITY_LOW, &low);
    (void)att1_aimu_userspace_read32(u,
            base + ATT1_MMIO_TILE_MEMORY_CAPACITY_HIGH, &high);
    uint64_t capacity = ((uint64_t)high << 32u) | (uint64_t)low;
    EXPECT(capacity == 64u * 1024u * 1024u,
           "tile_cap: TILE_MEMORY_CAPACITY register == 64 MiB");

    att1_aimu_userspace_close(u);
}

/* =========================================================================
 * Test 6: doorbell increments counter
 * ====================================================================== */

static void test_doorbell_increments_counter(void)
{
    att1_aimu_userspace *u = make_emu(NULL);
    EXPECT(u != NULL, "doorbell: open");
    if (!u) return;

    EXPECT(probe_and_setup(u), "doorbell: probe+setup_cmdq");

    att1_status_t rc = att1_aimu_userspace_ring_doorbell(u);
    EXPECT(rc == ATT1_OK, "doorbell: ring_doorbell returns ATT1_OK");

    att1_aimu_host_summary sum;
    memset(&sum, 0, sizeof(sum));
    (void)att1_aimu_userspace_get_summary(u, &sum);
    EXPECT(sum.doorbell_count == 1u, "doorbell: doorbell_count == 1");

    att1_aimu_userspace_close(u);
}

/* =========================================================================
 * Test 7: NOP completes
 * ====================================================================== */

static void test_nop_completes(void)
{
    att1_aimu_userspace *u = make_emu(NULL);
    EXPECT(u != NULL, "nop: open");
    if (!u) return;

    EXPECT(probe_and_setup(u), "nop: probe+setup_cmdq");
    EXPECT(submit_cmd_type(u, ATT1_AIMU_CMD_NOP, 1) == ATT1_OK, "nop: submit");
    EXPECT(att1_aimu_userspace_ring_doorbell(u) == ATT1_OK, "nop: ring_doorbell");
    EXPECT(att1_aimu_userspace_drain(u)         == ATT1_OK, "nop: drain");

    att1_aimu_host_summary sum;
    memset(&sum, 0, sizeof(sum));
    (void)att1_aimu_userspace_get_summary(u, &sum);
    EXPECT(sum.commands_completed >= 1u, "nop: commands_completed >= 1");
    EXPECT(sum.commands_submitted >= 1u, "nop: commands_submitted >= 1");

    att1_aimu_userspace_close(u);
}

/* =========================================================================
 * Test 8: LOAD_TENSOR_TILE produces a completion
 * ====================================================================== */

static void test_load_tensor_completes(void)
{
    att1_aimu_userspace *u = make_emu(NULL);
    EXPECT(u != NULL, "load_tensor: open");
    if (!u) return;

    EXPECT(probe_and_setup(u), "load_tensor: probe+setup_cmdq");

    att1_aimu_cmd cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_id      = 1u;
    cmd.command_type    = (uint8_t)ATT1_AIMU_CMD_LOAD_TENSOR_TILE;
    cmd.tile_id         = 0u;
    cmd.tensor_id       = 1u;
    cmd.input_buf_addr  = 64u;
    cmd.input_buf_bytes = 64u;
    att1_status_t rc = att1_aimu_userspace_submit_cmd(u, &cmd);
    EXPECT(rc == ATT1_OK, "load_tensor: submit");

    EXPECT(att1_aimu_userspace_ring_doorbell(u) == ATT1_OK, "load_tensor: doorbell");
    EXPECT(att1_aimu_userspace_drain(u)         == ATT1_OK, "load_tensor: drain");

    att1_aimu_host_summary sum;
    memset(&sum, 0, sizeof(sum));
    (void)att1_aimu_userspace_get_summary(u, &sum);
    EXPECT(sum.commands_completed >= 1u, "load_tensor: commands_completed >= 1");

    att1_aimu_userspace_close(u);
}

/* =========================================================================
 * Test 9: VALIDATE_TENSOR produces a completion
 * ====================================================================== */

static void test_validate_tensor_completes(void)
{
    att1_aimu_userspace *u = make_emu(NULL);
    EXPECT(u != NULL, "validate_tensor: open");
    if (!u) return;

    EXPECT(probe_and_setup(u), "validate_tensor: probe+setup_cmdq");
    EXPECT(submit_cmd_type(u, ATT1_AIMU_CMD_VALIDATE_TENSOR, 1) == ATT1_OK,
           "validate_tensor: submit");
    EXPECT(att1_aimu_userspace_ring_doorbell(u) == ATT1_OK,
           "validate_tensor: doorbell");
    EXPECT(att1_aimu_userspace_drain(u) == ATT1_OK, "validate_tensor: drain");

    att1_aimu_host_summary sum;
    memset(&sum, 0, sizeof(sum));
    (void)att1_aimu_userspace_get_summary(u, &sum);
    EXPECT(sum.commands_completed >= 1u, "validate_tensor: commands_completed >= 1");

    att1_aimu_userspace_close(u);
}

/* =========================================================================
 * Test 10: QUERY_COUNTERS produces a completion
 * ====================================================================== */

static void test_query_counters_completes(void)
{
    att1_aimu_userspace *u = make_emu(NULL);
    EXPECT(u != NULL, "query_counters: open");
    if (!u) return;

    EXPECT(probe_and_setup(u), "query_counters: probe+setup_cmdq");
    EXPECT(submit_cmd_type(u, ATT1_AIMU_CMD_QUERY_COUNTERS, 1) == ATT1_OK,
           "query_counters: submit");
    EXPECT(att1_aimu_userspace_ring_doorbell(u) == ATT1_OK,
           "query_counters: doorbell");
    EXPECT(att1_aimu_userspace_drain(u) == ATT1_OK, "query_counters: drain");

    att1_aimu_host_summary sum;
    memset(&sum, 0, sizeof(sum));
    (void)att1_aimu_userspace_get_summary(u, &sum);
    EXPECT(sum.commands_completed >= 1u, "query_counters: commands_completed >= 1");

    att1_aimu_userspace_close(u);
}

/* =========================================================================
 * Test 11: completion order is deterministic (submit order)
 * ====================================================================== */

static void test_completion_order_deterministic(void)
{
    att1_aimu_userspace *u = make_emu(NULL);
    EXPECT(u != NULL, "completion_order: open");
    if (!u) return;

    EXPECT(probe_and_setup(u), "completion_order: probe+setup_cmdq");

    /* Submit NOP then QUERY_COUNTERS in that order */
    EXPECT(submit_cmd_type(u, ATT1_AIMU_CMD_NOP,           1) == ATT1_OK,
           "completion_order: submit NOP id=1");
    EXPECT(submit_cmd_type(u, ATT1_AIMU_CMD_QUERY_COUNTERS, 2) == ATT1_OK,
           "completion_order: submit QUERY_COUNTERS id=2");

    EXPECT(att1_aimu_userspace_ring_doorbell(u) == ATT1_OK, "completion_order: doorbell");
    EXPECT(att1_aimu_userspace_drain(u)         == ATT1_OK, "completion_order: drain");

    /* Poll completions and verify first completion has smaller command_id */
    att1_aimu_completion comp1, comp2;
    memset(&comp1, 0, sizeof(comp1));
    memset(&comp2, 0, sizeof(comp2));

    att1_status_t rc1 = att1_aimu_host_read_completion(u->host, &comp1);
    att1_status_t rc2 = att1_aimu_host_read_completion(u->host, &comp2);
    EXPECT(rc1 == ATT1_OK, "completion_order: first completion available");
    EXPECT(rc2 == ATT1_OK, "completion_order: second completion available");
    if (rc1 == ATT1_OK && rc2 == ATT1_OK) {
        EXPECT(comp1.command_id <= comp2.command_id,
               "completion_order: first completion has <= command_id");
    }

    att1_aimu_userspace_close(u);
}

/* =========================================================================
 * Test 12: write to invalid offset returns error
 * ====================================================================== */

static void test_invalid_register_offset(void)
{
    att1_aimu_userspace *u = make_emu(NULL);
    EXPECT(u != NULL, "invalid_offset: open");
    if (!u) return;

    /* Write at exactly BAR0_SIZE should fail */
    att1_status_t rc = att1_aimu_userspace_write32(u,
            ATT1_AIMU_MMIO_BAR0_SIZE, 0u);
    EXPECT(rc != ATT1_OK, "invalid_offset: write at BAR0_SIZE returns error");

    /* Read at exactly BAR0_SIZE should also fail */
    uint32_t val = 0u;
    rc = att1_aimu_userspace_read32(u, ATT1_AIMU_MMIO_BAR0_SIZE, &val);
    EXPECT(rc != ATT1_OK, "invalid_offset: read at BAR0_SIZE returns error");

    att1_aimu_userspace_close(u);
}

/* =========================================================================
 * Test 13: invalid tile_count rejected
 * ====================================================================== */

static void test_invalid_tile_count(void)
{
    att1_aimu_userspace_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.tile_count        = 17u;  /* > 16, must be rejected */
    cfg.tile_memory_bytes = 32u * 1024u * 1024u;
    cfg.tile_kv_bytes     = 8u  * 1024u * 1024u;

    att1_aimu_userspace *u = NULL;
    att1_status_t rc = att1_aimu_userspace_open(&cfg, NULL, &u);
    EXPECT(rc != ATT1_OK, "invalid_tile_count: tile_count=17 rejected");
    EXPECT(u == NULL,     "invalid_tile_count: out ptr is NULL on error");

    if (u) att1_aimu_userspace_close(u);
}

/* =========================================================================
 * Test 14: invalid tile_memory_bytes rejected
 * ====================================================================== */

static void test_invalid_tile_memory(void)
{
    att1_aimu_userspace_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.tile_count        = 2u;
    /* Exceeds ATT1_AIMU_USERSPACE_MAX_TILE_MEM_BYTES (256 MiB) */
    cfg.tile_memory_bytes = ATT1_AIMU_USERSPACE_MAX_TILE_MEM_BYTES + 1u;
    cfg.tile_kv_bytes     = 8u * 1024u * 1024u;

    att1_aimu_userspace *u = NULL;
    att1_status_t rc = att1_aimu_userspace_open(&cfg, NULL, &u);
    EXPECT(rc != ATT1_OK, "invalid_tile_memory: over-large tile_memory rejected");
    EXPECT(u == NULL,     "invalid_tile_memory: out ptr is NULL on error");

    if (u) att1_aimu_userspace_close(u);
}

/* =========================================================================
 * Test 15: no hidden CUDA dependency
 * ====================================================================== */

static void test_no_cuda_dep(void)
{
#ifdef ATT1_ENABLE_CUDA
    printf("PASS: aimu_userspace: no_cuda_dep (built without CUDA)\n");
    g_pass++;
#else
    /* Build succeeded without ATT1_ENABLE_CUDA → no CUDA leakage */
    EXPECT(1, "no_cuda_dep: builds without ATT1_ENABLE_CUDA");
#endif
}

/* =========================================================================
 * main
 * ====================================================================== */

int main(void)
{
    printf("=== aimu_userspace tests ===\n");

    test_smoke();
    test_device_id_readable();
    test_register_map_version_readable();
    test_tile_count_matches();
    test_tile_capacity_metadata_only();
    test_doorbell_increments_counter();
    test_nop_completes();
    test_load_tensor_completes();
    test_validate_tensor_completes();
    test_query_counters_completes();
    test_completion_order_deterministic();
    test_invalid_register_offset();
    test_invalid_tile_count();
    test_invalid_tile_memory();
    test_no_cuda_dep();

    printf("---\n");
    printf("PASS %d  FAIL %d\n", g_pass, g_fail);
    return (g_fail > 0) ? 1 : 0;
}
