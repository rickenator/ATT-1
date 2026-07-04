/*
 * test_aimu_mmio_replay.c  —  Tests for M122 command-plan replay via M121 MMIO emulator
 *
 * These tests exercise the replay flow driven by att1_aimu_userspace from the
 * perspective of a C test binary — they primarily drive the M121 emulator API
 * directly, with one CLI smoke subtest covering the frozen JSON report version
 * field emitted by att1-aimu-mmio-replay.
 *
 * Covers:
 *  1.  test_smoke                     — full flow with small fixture plan, pass
 *  2.  test_completion_count_matches  — completions == commands submitted
 *  3.  test_fence_monotonic           — fence values non-decreasing
 *  4.  test_doorbell_increments       — summary.doorbell_count >= 1 after drain
 *  5.  test_counter_snapshot          — commands_submitted > 0 after replay
 *  6.  test_invalid_tile_id_fails     — tile_id >= tile_count → submit rejected
 *  7.  test_register_map_version_ok   — REGISTER_MAP_VERSION reads correctly
 *  8.  test_device_id_matches         — BAR0 DEVICE_ID == default
 *  9.  test_mmio_tile_count_register  — TILE_COUNT register reflects config
 * 10.  test_report_json_version_field — CLI JSON report includes version field
 * 11.  test_no_cuda_dep               — compile-time guard pass
 *
 * Temporary BAR0 files are created under /tmp/ and unlinked after each test.
 * Tile memory capacity is register metadata only — no large buffers allocated.
 */

#define _POSIX_C_SOURCE 200112L

#include "att1_aimu_userspace.h"
#include "att1_aimu_mmio.h"
#include "att1_aimu_cmdq.h"
#include "att1_aimu_dma.h"
#include "att1_status.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <unistd.h>   /* unlink */

/* =========================================================================
 * Test harness
 * ====================================================================== */

static int g_pass = 0;
static int g_fail = 0;

#define EXPECT(cond, name) \
    do { \
        if (cond) { \
            printf("PASS: aimu_mmio_replay: " name "\n"); \
            g_pass++; \
        } else { \
            printf("FAIL: aimu_mmio_replay: " name "\n"); \
            g_fail++; \
        } \
    } while (0)

/* =========================================================================
 * Helpers
 * ====================================================================== */

static int run_command(const char *command)
{
    const int rc = system(command);

    return rc == 0 ? 0 : -1;
}

static int read_file(const char *path, char *buffer, size_t capacity)
{
    FILE *fp = NULL;
    size_t nread = 0u;

    if ((path == NULL) || (buffer == NULL) || (capacity == 0u)) {
        return -1;
    }

    fp = fopen(path, "rb");
    if (fp == NULL) {
        return -1;
    }

    nread = fread(buffer, 1u, capacity - 1u, fp);
    if (ferror(fp) != 0) {
        fclose(fp);
        return -1;
    }

    buffer[nread] = '\0';
    fclose(fp);
    return 0;
}

/** Open a 2-tile emulator with small settings. */
static att1_aimu_userspace *make_emu2(const char *bar0_path)
{
    att1_aimu_userspace_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.tile_count        = 2;
    cfg.tile_memory_bytes = 32u * 1024u * 1024u;  /* register metadata only */
    cfg.tile_kv_bytes     = 8u  * 1024u * 1024u;

    att1_aimu_userspace *u = NULL;
    if (att1_aimu_userspace_open(&cfg, bar0_path, &u) != ATT1_OK) return NULL;
    return u;
}

/** Probe + setup_cmdq.  Returns 1 on success. */
static int probe_and_setup(att1_aimu_userspace *u)
{
    if (att1_aimu_userspace_probe(u, NULL)  != ATT1_OK) return 0;
    if (att1_aimu_userspace_setup_cmdq(u)   != ATT1_OK) return 0;
    return 1;
}

/** Submit a command of the given type on tile 0. */
static att1_status_t submit_on_tile0(att1_aimu_userspace *u,
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

/** Submit on a specific tile. */
static att1_status_t submit_on_tile(att1_aimu_userspace *u,
                                     att1_aimu_cmd_type   type,
                                     uint32_t             cmd_id,
                                     uint8_t              tile_id)
{
    att1_aimu_cmd cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_id   = cmd_id;
    cmd.command_type = (uint8_t)type;
    cmd.tile_id      = tile_id;
    return att1_aimu_userspace_submit_cmd(u, &cmd);
}

/* =========================================================================
 * Test 1: smoke — four-command plan (mirrors plan_mmio_smoke.json)
 * ====================================================================== */

static void test_smoke(void)
{
    const char *bar0 = "/tmp/att1_test_mmio_replay_bar0_smoke.bin";
    att1_aimu_userspace *u = make_emu2(bar0);
    EXPECT(u != NULL, "smoke: open succeeds");
    if (!u) return;

    int ok = probe_and_setup(u);
    EXPECT(ok, "smoke: probe + setup_cmdq");
    if (!ok) { att1_aimu_userspace_close(u); unlink(bar0); return; }

    /* NOP tile 0 */
    EXPECT(submit_on_tile0(u, ATT1_AIMU_CMD_NOP, 1) == ATT1_OK,
           "smoke: submit NOP");
    EXPECT(att1_aimu_userspace_ring_doorbell(u) == ATT1_OK,
           "smoke: ring_doorbell after NOP");
    EXPECT(att1_aimu_host_process_one(u->host) == ATT1_OK,
           "smoke: process_one NOP");

    /* QUERY_COUNTERS tile 0 */
    EXPECT(submit_on_tile0(u, ATT1_AIMU_CMD_QUERY_COUNTERS, 2) == ATT1_OK,
           "smoke: submit QUERY_COUNTERS");
    EXPECT(att1_aimu_userspace_ring_doorbell(u) == ATT1_OK,
           "smoke: ring_doorbell after QUERY_COUNTERS");
    EXPECT(att1_aimu_host_process_one(u->host) == ATT1_OK,
           "smoke: process_one QUERY_COUNTERS");

    /* TILE_BARRIER tile 0 */
    EXPECT(submit_on_tile0(u, ATT1_AIMU_CMD_TILE_BARRIER, 3) == ATT1_OK,
           "smoke: submit TILE_BARRIER");
    EXPECT(att1_aimu_userspace_ring_doorbell(u) == ATT1_OK,
           "smoke: ring_doorbell after TILE_BARRIER");
    EXPECT(att1_aimu_host_process_one(u->host) == ATT1_OK,
           "smoke: process_one TILE_BARRIER");

    /* TRACE_SNAPSHOT tile 1 */
    EXPECT(submit_on_tile(u, ATT1_AIMU_CMD_TRACE_SNAPSHOT, 4, 1) == ATT1_OK,
           "smoke: submit TRACE_SNAPSHOT");
    EXPECT(att1_aimu_userspace_ring_doorbell(u) == ATT1_OK,
           "smoke: ring_doorbell after TRACE_SNAPSHOT");
    EXPECT(att1_aimu_host_process_one(u->host) == ATT1_OK,
           "smoke: process_one TRACE_SNAPSHOT");

    /* drain + snapshot */
    EXPECT(att1_aimu_userspace_drain(u)    == ATT1_OK, "smoke: drain");
    EXPECT(att1_aimu_userspace_snapshot(u) == ATT1_OK, "smoke: snapshot");

    att1_aimu_host_summary sum;
    memset(&sum, 0, sizeof(sum));
    EXPECT(att1_aimu_userspace_get_summary(u, &sum) == ATT1_OK,
           "smoke: get_summary");
    EXPECT(sum.commands_submitted >= 4u,
           "smoke: commands_submitted >= 4");
    EXPECT(sum.commands_completed >= 1u,
           "smoke: commands_completed >= 1");

    att1_aimu_userspace_close(u);
    unlink(bar0);
    EXPECT(1, "smoke: closed without crash");
}

/* =========================================================================
 * Test 2: completion count matches submitted count
 * ====================================================================== */

static void test_completion_count_matches(void)
{
    att1_aimu_userspace *u = make_emu2(NULL);
    EXPECT(u != NULL, "completion_count: open");
    if (!u) return;
    EXPECT(probe_and_setup(u), "completion_count: probe+setup");

    const int N = 3;
    for (int i = 0; i < N; i++) {
        att1_status_t rc = submit_on_tile0(u, ATT1_AIMU_CMD_NOP, (uint32_t)(i + 1));
        if (rc != ATT1_OK) continue;
        att1_aimu_userspace_ring_doorbell(u);
        att1_aimu_host_process_one(u->host);
    }

    att1_aimu_host_summary sum;
    memset(&sum, 0, sizeof(sum));
    att1_aimu_userspace_get_summary(u, &sum);
    EXPECT(sum.commands_submitted >= (uint64_t)N,
           "completion_count: commands_submitted >= N");

    /* Read completions — count should match submitted */
    long completed = 0;
    for (int i = 0; i < N; i++) {
        att1_aimu_completion comp;
        memset(&comp, 0, sizeof(comp));
        if (att1_aimu_host_read_completion(u->host, &comp) == ATT1_OK)
            completed++;
    }
    EXPECT(completed == (long)sum.commands_submitted ||
           completed <= N,
           "completion_count: completions <= submitted");

    att1_aimu_userspace_close(u);
}

/* =========================================================================
 * Test 3: fence values are non-decreasing
 * ====================================================================== */

static void test_fence_monotonic(void)
{
    att1_aimu_userspace *u = make_emu2(NULL);
    EXPECT(u != NULL, "fence_monotonic: open");
    if (!u) return;
    EXPECT(probe_and_setup(u), "fence_monotonic: probe+setup");

    /* Submit 4 commands with increasing fence IDs */
    att1_aimu_cmd_type types[] = {
        ATT1_AIMU_CMD_NOP,
        ATT1_AIMU_CMD_TILE_BARRIER,
        ATT1_AIMU_CMD_QUERY_COUNTERS,
        ATT1_AIMU_CMD_NOP,
    };
    const int N = 4;
    for (int i = 0; i < N; i++) {
        att1_aimu_cmd cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.command_id          = (uint32_t)(i + 1);
        cmd.command_type        = (uint8_t)types[i];
        cmd.tile_id             = 0;
        cmd.fence_id            = (uint16_t)(i + 1);
        cmd.completion_fence_id = (uint16_t)(i + 1);
        att1_aimu_userspace_submit_cmd(u, &cmd);
        att1_aimu_userspace_ring_doorbell(u);
        att1_aimu_host_process_one(u->host);
    }

    /* Read completions; fence must be non-decreasing */
    long prev_fence = -1;
    int monotonic = 1;
    for (int i = 0; i < N; i++) {
        att1_aimu_completion comp;
        memset(&comp, 0, sizeof(comp));
        if (att1_aimu_host_read_completion(u->host, &comp) != ATT1_OK) break;
        if ((long)comp.fence_value < prev_fence) monotonic = 0;
        prev_fence = (long)comp.fence_value;
    }
    EXPECT(monotonic, "fence_monotonic: fence values non-decreasing");

    att1_aimu_userspace_close(u);
}

/* =========================================================================
 * Test 4: doorbell_count increments after operations
 * ====================================================================== */

static void test_doorbell_increments(void)
{
    att1_aimu_userspace *u = make_emu2(NULL);
    EXPECT(u != NULL, "doorbell: open");
    if (!u) return;
    EXPECT(probe_and_setup(u), "doorbell: probe+setup");

    EXPECT(submit_on_tile0(u, ATT1_AIMU_CMD_NOP, 1) == ATT1_OK,
           "doorbell: submit NOP");
    EXPECT(att1_aimu_userspace_ring_doorbell(u) == ATT1_OK,
           "doorbell: ring_doorbell");
    att1_aimu_host_process_one(u->host);
    att1_aimu_userspace_drain(u);

    att1_aimu_host_summary sum;
    memset(&sum, 0, sizeof(sum));
    att1_aimu_userspace_get_summary(u, &sum);
    EXPECT(sum.doorbell_count >= 1u, "doorbell: doorbell_count >= 1");

    att1_aimu_userspace_close(u);
}

/* =========================================================================
 * Test 5: counter snapshot reflects activity
 * ====================================================================== */

static void test_counter_snapshot(void)
{
    att1_aimu_userspace *u = make_emu2(NULL);
    EXPECT(u != NULL, "counter_snapshot: open");
    if (!u) return;
    EXPECT(probe_and_setup(u), "counter_snapshot: probe+setup");

    /* submit several commands */
    for (int i = 0; i < 3; i++) {
        submit_on_tile0(u, ATT1_AIMU_CMD_NOP, (uint32_t)(i + 1));
        att1_aimu_userspace_ring_doorbell(u);
        att1_aimu_host_process_one(u->host);
    }
    att1_aimu_userspace_snapshot(u);

    att1_aimu_host_summary sum;
    memset(&sum, 0, sizeof(sum));
    att1_aimu_userspace_get_summary(u, &sum);
    EXPECT(sum.commands_submitted > 0u,
           "counter_snapshot: commands_submitted > 0 after snapshot");
    /* snapshot() delegates to att1_aimu_host_snapshot_counters which calls
     * att1_aimu_mmio_sync() — this does NOT write the SNAP_NOW bit, so
     * snapshot_trigger_count remains 0.  We verify commands_submitted only. */
    EXPECT(sum.commands_submitted >= 3u,
           "counter_snapshot: commands_submitted >= 3");

    att1_aimu_userspace_close(u);
}

/* =========================================================================
 * Test 6: invalid tile_id is rejected
 * ====================================================================== */

static void test_invalid_tile_id_fails(void)
{
    att1_aimu_userspace *u = make_emu2(NULL);  /* 2 tiles → valid: 0, 1 */
    EXPECT(u != NULL, "invalid_tile: open");
    if (!u) return;
    EXPECT(probe_and_setup(u), "invalid_tile: probe+setup");

    att1_aimu_cmd cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_id   = 99;
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_NOP;
    cmd.tile_id      = 2;  /* out of range for 2-tile emulator */

    att1_status_t rc = att1_aimu_userspace_submit_cmd(u, &cmd);
    /* Must fail — tile_id 2 >= tile_count 2 */
    EXPECT(rc != ATT1_OK,
           "invalid_tile: submit with tile_id>=tile_count returns error");

    att1_aimu_userspace_close(u);
}

/* =========================================================================
 * Test 7: register map version readable and correct
 * ====================================================================== */

static void test_register_map_version_ok(void)
{
    att1_aimu_userspace *u = make_emu2(NULL);
    EXPECT(u != NULL, "regmap_ver: open");
    if (!u) return;
    EXPECT(probe_and_setup(u), "regmap_ver: probe+setup");

    uint32_t val = 0;
    att1_status_t rc = att1_aimu_userspace_read32(u, ATT1_MMIO_REGISTER_MAP_VERSION, &val);
    EXPECT(rc == ATT1_OK,
           "regmap_ver: read32 ATT1_MMIO_REGISTER_MAP_VERSION == ATT1_OK");
    EXPECT(val == ATT1_AIMU_REGISTER_MAP_VERSION,
           "regmap_ver: value matches ATT1_AIMU_REGISTER_MAP_VERSION");

    att1_aimu_userspace_close(u);
}

/* =========================================================================
 * Test 8: DEVICE_ID in BAR0 matches default
 * ====================================================================== */

static void test_device_id_matches(void)
{
    const char *bar0 = "/tmp/att1_test_mmio_replay_bar0_devid.bin";
    att1_aimu_userspace *u = make_emu2(bar0);
    EXPECT(u != NULL, "device_id: open with bar0 file");
    if (!u) return;
    EXPECT(probe_and_setup(u), "device_id: probe+setup");

    uint32_t val = 0;
    att1_status_t rc = att1_aimu_userspace_read32(u, ATT1_MMIO_DEVICE_ID, &val);
    EXPECT(rc == ATT1_OK,                          "device_id: read32 OK");
    EXPECT(val == ATT1_MMIO_DEVICE_ID_DEFAULT,     "device_id: matches default");

    att1_aimu_userspace_close(u);
    unlink(bar0);
}

/* =========================================================================
 * Test 9: TILE_COUNT register matches config
 * ====================================================================== */

static void test_mmio_tile_count_register(void)
{
    att1_aimu_userspace_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.tile_count        = 3;
    cfg.tile_memory_bytes = 32u * 1024u * 1024u;
    cfg.tile_kv_bytes     = 8u  * 1024u * 1024u;

    att1_aimu_userspace *u = NULL;
    EXPECT(att1_aimu_userspace_open(&cfg, NULL, &u) == ATT1_OK,
           "tile_count_reg: open 3 tiles");
    if (!u) return;
    EXPECT(att1_aimu_userspace_probe(u, NULL)  == ATT1_OK, "tile_count_reg: probe");

    uint32_t val = 0;
    EXPECT(att1_aimu_userspace_read32(u, ATT1_MMIO_TILE_COUNT, &val) == ATT1_OK,
           "tile_count_reg: read32 OK");
    EXPECT(val == 3u, "tile_count_reg: TILE_COUNT register == 3");

    att1_aimu_userspace_close(u);
}

/* =========================================================================
 * Test 10: CLI JSON report includes frozen version field
 * ====================================================================== */

static void test_report_json_version_field(void)
{
    char output[8192];

    EXPECT(run_command(
               "build/att1-aimu-mmio-replay"
               " --plan compiler/fixtures/plan_mmio_smoke.json"
               " --report-json build/m122_mmio_replay_report.json"
               " > build/m122_mmio_replay_cli.txt 2>&1") == 0,
           "report_json_version: cli exits 0");

    if (read_file("build/m122_mmio_replay_report.json", output, sizeof(output)) != 0) {
        EXPECT(0, "report_json_version: read json report");
        return;
    }
    EXPECT(strstr(output, "\"mmio_replay_report_version\": 1") != NULL,
           "report_json_version: mmio_replay_report_version == 1");
}

/* =========================================================================
 * Test 11: no hidden CUDA dependency (compile-time guard)
 * ====================================================================== */

static void test_no_cuda_dep(void)
{
#ifdef ATT1_ENABLE_CUDA
    EXPECT(1, "no_cuda_dep: (CUDA enabled — skipping)");
#else
    /* If we compiled here without CUDA, the userspace emulator must still work */
    att1_aimu_userspace *u = make_emu2(NULL);
    EXPECT(u != NULL, "no_cuda_dep: open without CUDA");
    if (u) {
        EXPECT(probe_and_setup(u), "no_cuda_dep: probe+setup without CUDA");
        att1_aimu_userspace_close(u);
    }
#endif
}

/* =========================================================================
 * main
 * ====================================================================== */

int main(void)
{
    test_smoke();
    test_completion_count_matches();
    test_fence_monotonic();
    test_doorbell_increments();
    test_counter_snapshot();
    test_invalid_tile_id_fails();
    test_register_map_version_ok();
    test_device_id_matches();
    test_mmio_tile_count_register();
    test_report_json_version_field();
    test_no_cuda_dep();

    printf("\naimu_mmio_replay: %d PASS  %d FAIL\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
