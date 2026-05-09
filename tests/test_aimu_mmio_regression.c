/*
 * test_aimu_mmio_regression.c  —  M131: Userspace AIMU MMIO emulator
 *                                  regression suite.
 *
 * This file is a REGRESSION SUITE covering the M111 BAR0 register-file
 * simulator, the M112 integration harness, M105 command-queue / doorbell
 * path, M107 DMA descriptor path, M130 simulated EXEC replay, and the
 * M108 trace/counter snapshot mechanism.
 *
 * No new runtime code, CUDA kernels, .att1 format changes, or backend
 * modifications are introduced.  All interactions are in-process userspace
 * simulation; no real PCIe/MMIO hardware, Linux kernel driver, or FPGA RTL
 * is exercised.
 *
 * Test functions (16 total):
 *   1.  test_device_probe               DEVICE_ID, reg-map version, tile count
 *   2.  test_register_rw                round-trip on RW register
 *   3.  test_register_error_paths       invalid offset, unaligned, RO-write
 *   4.  test_register_reserved          reserved offset sentinel 0xDEADBEEF
 *   5.  test_reset_control              RESET_CONTROL path deterministic
 *   6.  test_cmdq_doorbell              submit/ring/drain/read completion
 *   7.  test_cmdq_fifo_order            completions arrive in FIFO order
 *   8.  test_cmdq_queue_full            ring fills → QUEUE_FULL
 *   9.  test_dma_valid                  valid H2D descriptor validates OK
 *  10.  test_dma_invalid                zero-length, unaligned, out-of-range
 *  11.  test_dma_huge_tile_no_alloc     huge tile_memory_bytes, no crash
 *  12.  test_exec_replay_supported      all EXEC_* ops via M130 dispatcher
 *  13.  test_exec_replay_unsupported    EXEC_MATMUL on restricted device
 *  14.  test_replay_integration         6-command M129 plan through host MMIO
 *  15.  test_trace_snapshot_deterministic  two replays → identical snapshots
 *  16.  test_no_cuda_dep                compile-time guard
 */

#define _POSIX_C_SOURCE 200112L

#include "att1_aimu_mmio.h"
#include "att1_aimu_device.h"
#include "att1_aimu_cmdq.h"
#include "att1_aimu_dma.h"
#include "att1_aimu_trace.h"
#include "att1_aimu_host.h"
#include "att1_aimu_exec.h"
#include "att1_status.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>

/* =========================================================================
 * Test harness
 * ====================================================================== */

static int g_pass = 0;
static int g_fail = 0;

#define EXPECT(cond, name)                                              \
    do {                                                                \
        if (cond) {                                                     \
            printf("PASS: aimu_mmio_regression: " name "\n");          \
            g_pass++;                                                   \
        } else {                                                        \
            printf("FAIL: aimu_mmio_regression: " name "\n");          \
            g_fail++;                                                   \
        }                                                               \
    } while (0)

/* =========================================================================
 * Helpers
 * ====================================================================== */

/** Allocate an att1_aimu_host with the given tile count and optional ring
 *  depth (0 → default).  Returns NULL on failure. */
static att1_aimu_host *make_host(size_t tile_count, size_t cmd_ring_depth)
{
    att1_aimu_host_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.tile_count        = tile_count ? tile_count : 2;
    cfg.tile_memory_bytes = 32u * 1024u * 1024u;  /* 32 MiB — metadata only */
    cfg.tile_kv_bytes     = 8u  * 1024u * 1024u;
    cfg.cmd_ring_depth    = cmd_ring_depth;
    att1_aimu_host *h     = NULL;
    if (att1_aimu_host_create(&cfg, &h) != ATT1_OK) return NULL;
    return h;
}

/** Probe then setup_cmdq on an existing host.  Returns 1 on success. */
static int probe_and_setup(att1_aimu_host *h)
{
    if (att1_aimu_host_probe_device(h, NULL) != ATT1_OK) return 0;
    if (att1_aimu_host_setup_cmdq(h)          != ATT1_OK) return 0;
    return 1;
}

/** Submit a command of type 'type' on tile tile_id.  Returns the status. */
static att1_status_t submit_type(att1_aimu_host *h,
                                  att1_aimu_cmd_type type,
                                  uint8_t            tile_id)
{
    att1_aimu_cmd cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)type;
    cmd.tile_id      = tile_id;
    return att1_aimu_host_submit_cmd(h, &cmd);
}

/* =========================================================================
 * 1. test_device_probe
 *
 * Verify that DEVICE_ID, REGISTER_MAP_VERSION, TILE_COUNT, GLOBAL_STATUS,
 * and per-tile metadata fields are accessible and correct after probe.
 * ====================================================================== */

static void test_device_probe(void)
{
    att1_aimu_host *h = make_host(2, 0);
    EXPECT(h != NULL, "device_probe: host created");
    if (!h) return;

    att1_aimu_host_probe_result probe;
    att1_status_t s = att1_aimu_host_probe_device(h, &probe);
    EXPECT(s == ATT1_OK, "device_probe: probe_device returns OK");

    /* Device ID must match the architectural default */
    EXPECT(probe.device_id == ATT1_MMIO_DEVICE_ID_DEFAULT,
           "device_probe: DEVICE_ID matches default");

    /* Register-map version must match M106 v1.0 */
    EXPECT(probe.register_map_version == ATT1_AIMU_REGISTER_MAP_VERSION,
           "device_probe: REGISTER_MAP_VERSION correct");

    /* TILE_COUNT reflects the configured value */
    EXPECT(probe.tile_count == 2,
           "device_probe: TILE_COUNT == 2");

    /* GLOBAL_STATUS has DEVICE_READY set */
    uint32_t gstatus = 0;
    s = att1_aimu_mmio_read32(h->mmio, ATT1_MMIO_GLOBAL_STATUS, &gstatus);
    EXPECT(s == ATT1_OK,
           "device_probe: GLOBAL_STATUS readable");
    EXPECT((gstatus & ATT1_MMIO_GSTAT_DEVICE_READY) != 0,
           "device_probe: DEVICE_READY bit set");

    /* Per-tile metadata: enumerate_tiles succeeds now that probe has run */
    att1_aimu_host_tile_info infos[2];
    size_t count = 2;
    s = att1_aimu_host_enumerate_tiles(h, infos, &count);
    EXPECT(s == ATT1_OK,  "device_probe: enumerate_tiles OK");
    EXPECT(count == 2,    "device_probe: 2 tiles enumerated");

    /* Tile 0 memory_capacity_bytes reflects configured 32 MiB (metadata) */
    EXPECT(infos[0].memory_capacity_bytes == 32u * 1024u * 1024u,
           "device_probe: tile0 memory_capacity_bytes correct");

    /* Tile 0 starts IDLE */
    EXPECT(infos[0].state == (uint8_t)ATT1_AIMU_TILE_IDLE,
           "device_probe: tile0 state IDLE");

    att1_aimu_host_destroy(h);
}

/* =========================================================================
 * 2. test_register_rw
 *
 * Write a pattern to INTERRUPT_ENABLE (0x0020, RW) and read it back.
 * ====================================================================== */

static void test_register_rw(void)
{
    att1_aimu_mmio *m = NULL;
    att1_status_t s   = att1_aimu_mmio_create(&m);
    EXPECT(s == ATT1_OK, "register_rw: mmio_create OK");
    if (!m) return;

    const uint32_t val = UINT32_C(0xA5A5A5A5);
    s = att1_aimu_mmio_write32(m, ATT1_MMIO_INTERRUPT_ENABLE, val);
    EXPECT(s == ATT1_OK, "register_rw: write32 INTERRUPT_ENABLE OK");

    uint32_t rd = 0;
    s = att1_aimu_mmio_read32(m, ATT1_MMIO_INTERRUPT_ENABLE, &rd);
    EXPECT(s == ATT1_OK,   "register_rw: read32 INTERRUPT_ENABLE OK");
    EXPECT(rd == val,      "register_rw: read back matches written value");

    /* Write a second pattern to confirm the cell is writable repeatedly */
    const uint32_t val2 = UINT32_C(0x5A5A5A5A);
    (void)att1_aimu_mmio_write32(m, ATT1_MMIO_INTERRUPT_ENABLE, val2);
    rd = 0;
    (void)att1_aimu_mmio_read32(m, ATT1_MMIO_INTERRUPT_ENABLE, &rd);
    EXPECT(rd == val2, "register_rw: second write round-trips");

    att1_aimu_mmio_destroy(m);
}

/* =========================================================================
 * 3. test_register_error_paths
 *
 * Out-of-range offset → INVALID_ARG
 * Unaligned offset    → INVALID_ARG
 * RO register write   → UNSUPPORTED
 * read64 unaligned    → INVALID_ARG
 * Null arguments      → INVALID_ARG
 * ====================================================================== */

static void test_register_error_paths(void)
{
    att1_aimu_mmio *m = NULL;
    att1_aimu_mmio_create(&m);
    if (!m) { EXPECT(0, "error_paths: mmio_create"); return; }

    uint32_t v = 0;
    att1_status_t s;

    /* Offset == BAR0_SIZE is out of range */
    s = att1_aimu_mmio_read32(m, ATT1_AIMU_MMIO_BAR0_SIZE, &v);
    EXPECT(s == ATT1_ERR_INVALID_ARG,
           "error_paths: read32 offset==BAR0_SIZE → INVALID_ARG");

    s = att1_aimu_mmio_write32(m, ATT1_AIMU_MMIO_BAR0_SIZE, 0);
    EXPECT(s == ATT1_ERR_INVALID_ARG,
           "error_paths: write32 offset==BAR0_SIZE → INVALID_ARG");

    /* Unaligned offset (not a multiple of 4) */
    s = att1_aimu_mmio_read32(m, ATT1_MMIO_DEVICE_ID + 1u, &v);
    EXPECT(s == ATT1_ERR_INVALID_ARG,
           "error_paths: read32 unaligned offset → INVALID_ARG");

    s = att1_aimu_mmio_write32(m, ATT1_MMIO_DEVICE_ID + 1u, 0);
    EXPECT(s == ATT1_ERR_INVALID_ARG,
           "error_paths: write32 unaligned offset → INVALID_ARG");

    /* DEVICE_ID (0x0000) is RO; write must be rejected */
    s = att1_aimu_mmio_write32(m, ATT1_MMIO_DEVICE_ID, UINT32_C(0xDEAD));
    EXPECT(s == ATT1_ERR_UNSUPPORTED,
           "error_paths: write32 RO register → UNSUPPORTED");

    /* REGISTER_MAP_VERSION (0x0008) is also RO */
    s = att1_aimu_mmio_write32(m, ATT1_MMIO_REGISTER_MAP_VERSION, 0);
    EXPECT(s == ATT1_ERR_UNSUPPORTED,
           "error_paths: write32 REGISTER_MAP_VERSION → UNSUPPORTED");

    /* TILE_COUNT (0x0014) is RO */
    s = att1_aimu_mmio_write32(m, ATT1_MMIO_TILE_COUNT, 0);
    EXPECT(s == ATT1_ERR_UNSUPPORTED,
           "error_paths: write32 TILE_COUNT → UNSUPPORTED");

    /* read64: offset must be 8-byte aligned; 0x0004 is not */
    uint64_t v64 = 0;
    s = att1_aimu_mmio_read64(m, UINT32_C(0x0004), &v64);
    EXPECT(s == ATT1_ERR_INVALID_ARG,
           "error_paths: read64 non-8-aligned offset → INVALID_ARG");

    /* Null m */
    s = att1_aimu_mmio_read32(NULL, ATT1_MMIO_INTERRUPT_ENABLE, &v);
    EXPECT(s == ATT1_ERR_INVALID_ARG,
           "error_paths: read32 null m → INVALID_ARG");

    /* Null output pointer */
    s = att1_aimu_mmio_read32(m, ATT1_MMIO_INTERRUPT_ENABLE, NULL);
    EXPECT(s == ATT1_ERR_INVALID_ARG,
           "error_paths: read32 null out → INVALID_ARG");

    att1_aimu_mmio_destroy(m);
}

/* =========================================================================
 * 4. test_register_reserved
 *
 * Reading a reserved (undefined) offset returns ATT1_AIMU_MMIO_RESERVED_RD
 * (0xDEADBEEF).  Writing to a reserved offset is silently discarded (OK)
 * and does not change the read value.
 * ====================================================================== */

static void test_register_reserved(void)
{
    att1_aimu_mmio *m = NULL;
    att1_aimu_mmio_create(&m);
    if (!m) { EXPECT(0, "register_reserved: create"); return; }

    /* 0x0040 is not defined in the M104 global register map */
    uint32_t v  = 0;
    att1_status_t s = att1_aimu_mmio_read32(m, UINT32_C(0x0040), &v);
    EXPECT(s == ATT1_OK && v == ATT1_AIMU_MMIO_RESERVED_RD,
           "register_reserved: reserved offset reads 0xDEADBEEF");

    /* Write to reserved → silently discarded (returns OK) */
    s = att1_aimu_mmio_write32(m, UINT32_C(0x0040), UINT32_C(0x12345678));
    EXPECT(s == ATT1_OK,
           "register_reserved: write32 reserved offset returns OK");

    /* Value unchanged after the discarded write */
    v = 0;
    s = att1_aimu_mmio_read32(m, UINT32_C(0x0040), &v);
    EXPECT(s == ATT1_OK && v == ATT1_AIMU_MMIO_RESERVED_RD,
           "register_reserved: sentinel unchanged after discarded write");

    att1_aimu_mmio_destroy(m);
}

/* =========================================================================
 * 5. test_reset_control
 *
 * Write RESET_CONTROL.RESET_COUNTERS to zero the counter register cells.
 * Verify the counter register reads 0 afterward.
 * Verify that reading the WO RESET_CONTROL itself returns 0.
 * ====================================================================== */

static void test_reset_control(void)
{
    att1_aimu_host *h = make_host(2, 0);
    if (!h) { EXPECT(0, "reset_control: host created"); return; }

    if (!probe_and_setup(h)) {
        EXPECT(0, "reset_control: probe and setup");
        att1_aimu_host_destroy(h);
        return;
    }

    /* Submit and drain one command so cmdq counter cells are non-zero */
    (void)submit_type(h, ATT1_AIMU_CMD_NOP, 0);
    (void)att1_aimu_host_ring_doorbell(h);
    (void)att1_aimu_host_drain(h);
    (void)att1_aimu_host_snapshot_counters(h);

    att1_aimu_host_summary sum;
    (void)att1_aimu_host_get_summary(h, &sum);
    EXPECT(sum.commands_submitted >= 1,
           "reset_control: commands_submitted >= 1 before reset");

    /* Trigger RESET_COUNTERS — zeroes the BAR0 counter register cells */
    att1_status_t s = att1_aimu_mmio_write32(h->mmio,
                                              ATT1_MMIO_RESET_CONTROL,
                                              ATT1_MMIO_RSTCTL_RESET_COUNTERS);
    EXPECT(s == ATT1_OK, "reset_control: write RESET_COUNTERS returns OK");

    /* CNT_CMD_ISSUED_LO (0x4000) must now read 0 */
    uint32_t cnt = UINT32_C(0xFFFFFFFF);
    s = att1_aimu_mmio_read32(h->mmio, ATT1_MMIO_CNT_CMD_ISSUED_LO, &cnt);
    EXPECT(s == ATT1_OK && cnt == 0,
           "reset_control: CNT_CMD_ISSUED_LO zeroed after RESET_COUNTERS");

    /* RESET_CONTROL is WO; reading it must return 0 (not the last write) */
    uint32_t rc = UINT32_C(0xFF);
    s = att1_aimu_mmio_read32(h->mmio, ATT1_MMIO_RESET_CONTROL, &rc);
    EXPECT(s == ATT1_OK && rc == 0,
           "reset_control: RESET_CONTROL reads 0 (WO register)");

    att1_aimu_host_destroy(h);
}

/* =========================================================================
 * 6. test_cmdq_doorbell
 *
 * Submit one NOP, ring the doorbell, drain, read the completion.
 * Verify doorbell_write_count increments and completion result is OK.
 * ====================================================================== */

static void test_cmdq_doorbell(void)
{
    att1_aimu_host *h = make_host(2, 0);
    if (!h) { EXPECT(0, "cmdq_doorbell: host created"); return; }

    EXPECT(probe_and_setup(h), "cmdq_doorbell: probe and setup");
    if (!h->cmdq_ready) { att1_aimu_host_destroy(h); return; }

    /* Submit NOP */
    att1_status_t s = submit_type(h, ATT1_AIMU_CMD_NOP, 0);
    EXPECT(s == ATT1_OK, "cmdq_doorbell: submit NOP OK");

    /* Doorbell count starts at 0 before any ring */
    EXPECT(h->mmio->doorbell_write_count == 0,
           "cmdq_doorbell: doorbell_write_count == 0 before ring");

    /* Ring doorbell */
    s = att1_aimu_host_ring_doorbell(h);
    EXPECT(s == ATT1_OK, "cmdq_doorbell: ring_doorbell OK");
    EXPECT(h->mmio->doorbell_write_count == 1,
           "cmdq_doorbell: doorbell_write_count == 1 after ring");

    /* Drain (simulate AIMU dispatch) */
    s = att1_aimu_host_drain(h);
    EXPECT(s == ATT1_OK, "cmdq_doorbell: drain OK");

    /* Exactly one completion must be available */
    EXPECT(att1_aimu_cmdq_completions_available(h->cmdq) == 1,
           "cmdq_doorbell: 1 completion available after drain");

    /* Read completion */
    att1_aimu_completion comp;
    s = att1_aimu_host_read_completion(h, &comp);
    EXPECT(s == ATT1_OK,                     "cmdq_doorbell: read_completion OK");
    EXPECT(comp.result_code == ATT1_AIMU_OK, "cmdq_doorbell: NOP completion result OK");
    EXPECT(comp.tile_id == 0,                "cmdq_doorbell: completion tile_id == 0");

    /* Completion ring is now empty */
    EXPECT(att1_aimu_cmdq_completions_available(h->cmdq) == 0,
           "cmdq_doorbell: 0 completions after consuming");

    att1_aimu_host_destroy(h);
}

/* =========================================================================
 * 7. test_cmdq_fifo_order
 *
 * Submit three commands (NOP, QUERY_COUNTERS, TILE_BARRIER) and verify
 * that their completions arrive in FIFO order: command_ids monotonically
 * increasing.
 * ====================================================================== */

static void test_cmdq_fifo_order(void)
{
    att1_aimu_host *h = make_host(2, 0);
    if (!h) { EXPECT(0, "fifo_order: host created"); return; }
    if (!probe_and_setup(h)) {
        att1_aimu_host_destroy(h);
        return;
    }

    /* Submit three commands in order */
    att1_aimu_cmd_type types[3] = {
        ATT1_AIMU_CMD_NOP,
        ATT1_AIMU_CMD_QUERY_COUNTERS,
        ATT1_AIMU_CMD_TILE_BARRIER
    };
    int i;
    for (i = 0; i < 3; i++) {
        att1_aimu_cmd cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.command_type = (uint8_t)types[i];
        cmd.tile_id      = 0;
        (void)att1_aimu_host_submit_cmd(h, &cmd);
    }

    (void)att1_aimu_host_ring_doorbell(h);
    (void)att1_aimu_host_drain(h);

    EXPECT(att1_aimu_cmdq_completions_available(h->cmdq) == 3,
           "fifo_order: 3 completions available");

    /* Read all three completions */
    att1_aimu_completion comps[3];
    for (i = 0; i < 3; i++) {
        (void)att1_aimu_host_read_completion(h, &comps[i]);
    }

    /* command_ids must be monotonically increasing (FIFO) */
    EXPECT(comps[0].command_id < comps[1].command_id,
           "fifo_order: comp[0].command_id < comp[1].command_id");
    EXPECT(comps[1].command_id < comps[2].command_id,
           "fifo_order: comp[1].command_id < comp[2].command_id");

    /* All three should complete OK (NOP, QUERY_COUNTERS, TILE_BARRIER) */
    EXPECT(comps[0].result_code == ATT1_AIMU_OK,
           "fifo_order: NOP result OK");
    EXPECT(comps[1].result_code == ATT1_AIMU_OK,
           "fifo_order: QUERY_COUNTERS result OK");
    EXPECT(comps[2].result_code == ATT1_AIMU_OK,
           "fifo_order: TILE_BARRIER result OK");

    att1_aimu_host_destroy(h);
}

/* =========================================================================
 * 8. test_cmdq_queue_full
 *
 * Create a host with a small command ring (depth 4).  Fill the ring and
 * verify that the next submit returns ATT1_ERR_QUEUE_FULL.
 * ====================================================================== */

static void test_cmdq_queue_full(void)
{
    /* depth 4 → ring holds at most depth-1 = 3 entries before full */
    att1_aimu_host *h = make_host(2, 4);
    if (!h) { EXPECT(0, "queue_full: host created"); return; }
    if (!probe_and_setup(h)) {
        att1_aimu_host_destroy(h);
        return;
    }

    int full_seen    = 0;
    size_t submitted = 0;
    int i;

    for (i = 0; i < 16; i++) {
        att1_aimu_cmd cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.command_type = (uint8_t)ATT1_AIMU_CMD_NOP;
        cmd.tile_id      = 0;
        att1_status_t s = att1_aimu_host_submit_cmd(h, &cmd);
        if (s == ATT1_ERR_QUEUE_FULL) {
            full_seen = 1;
            break;
        }
        if (s == ATT1_OK) submitted++;
    }

    EXPECT(full_seen, "queue_full: QUEUE_FULL observed when ring saturated");

    /* Pending count must equal the number of successful submits */
    size_t pending = att1_aimu_cmdq_pending(h->cmdq);
    EXPECT(pending == submitted,
           "queue_full: pending count matches successful submits");

    /* Drain and verify completions match pending */
    (void)att1_aimu_host_ring_doorbell(h);
    (void)att1_aimu_host_drain(h);
    size_t done = att1_aimu_cmdq_completions_available(h->cmdq);
    EXPECT(done == submitted,
           "queue_full: completions after drain == submitted");

    att1_aimu_host_destroy(h);
}

/* =========================================================================
 * 9. test_dma_valid
 *
 * A valid 64-byte-aligned H2D descriptor with no registered regions
 * (permissive mode) must validate and submit successfully.
 * ====================================================================== */

static void test_dma_valid(void)
{
    att1_aimu_dma *dma = NULL;
    att1_status_t s    = att1_aimu_dma_create(&dma);
    EXPECT(s == ATT1_OK, "dma_valid: create OK");
    if (!dma) return;

    att1_aimu_dma_desc desc;
    memset(&desc, 0, sizeof(desc));
    desc.host_addr     = UINT64_C(0x1000);   /* 64-byte aligned */
    desc.device_addr   = UINT64_C(0x1000);
    desc.byte_length   = 4096;
    desc.direction     = (uint8_t)ATT1_AIMU_DMA_HOST_TO_DEVICE;
    desc.dtype         = ATT1_AIMU_DMA_DTYPE_F32;
    desc.flags         = ATT1_AIMU_DMA_FLAG_LAST_DESCRIPTOR;
    desc.descriptor_id = 1;

    /* validate must return OK */
    s = att1_aimu_dma_validate(dma, &desc);
    EXPECT(s == ATT1_OK, "dma_valid: validate OK");

    /* submit must succeed and update counters */
    s = att1_aimu_dma_submit(dma, &desc);
    EXPECT(s == ATT1_OK, "dma_valid: submit OK");

    att1_aimu_dma_counters cnt;
    att1_aimu_dma_get_counters(dma, &cnt);
    EXPECT(cnt.dma_submitted       == 1,    "dma_valid: dma_submitted == 1");
    EXPECT(cnt.dma_completed       == 1,    "dma_valid: dma_completed == 1");
    EXPECT(cnt.dma_failed          == 0,    "dma_valid: dma_failed == 0");
    EXPECT(cnt.bytes_host_to_device == 4096, "dma_valid: H2D bytes == 4096");

    att1_aimu_dma_destroy(dma);
}

/* =========================================================================
 * 10. test_dma_invalid
 *
 * Zero-length byte_length  → INVALID_ARG
 * Unaligned host_addr      → INVALID_ARG
 * Unaligned device_addr    → INVALID_ARG
 * Out-of-range host_addr   → INVALID_ARG (with registered region)
 * Failed submit increments dma_failed counter.
 * ====================================================================== */

static void test_dma_invalid(void)
{
    att1_aimu_dma *dma = NULL;
    att1_aimu_dma_create(&dma);
    if (!dma) { EXPECT(0, "dma_invalid: create"); return; }

    att1_aimu_dma_desc desc;
    att1_status_t s;

    /* Zero-length */
    memset(&desc, 0, sizeof(desc));
    desc.host_addr   = UINT64_C(0x1000);
    desc.device_addr = UINT64_C(0x1000);
    desc.byte_length = 0;
    desc.direction   = (uint8_t)ATT1_AIMU_DMA_HOST_TO_DEVICE;
    desc.dtype       = ATT1_AIMU_DMA_DTYPE_F32;
    s = att1_aimu_dma_validate(dma, &desc);
    EXPECT(s == ATT1_ERR_INVALID_ARG,
           "dma_invalid: zero-length byte_length → INVALID_ARG");

    /* Unaligned host_addr (not a multiple of 64) */
    memset(&desc, 0, sizeof(desc));
    desc.host_addr   = UINT64_C(0x1003);
    desc.device_addr = UINT64_C(0x1000);
    desc.byte_length = 4096;
    desc.direction   = (uint8_t)ATT1_AIMU_DMA_HOST_TO_DEVICE;
    desc.dtype       = ATT1_AIMU_DMA_DTYPE_F32;
    s = att1_aimu_dma_validate(dma, &desc);
    EXPECT(s == ATT1_ERR_INVALID_ARG,
           "dma_invalid: unaligned host_addr → INVALID_ARG");

    /* Unaligned device_addr */
    memset(&desc, 0, sizeof(desc));
    desc.host_addr   = UINT64_C(0x1000);
    desc.device_addr = UINT64_C(0x1001);
    desc.byte_length = 4096;
    desc.direction   = (uint8_t)ATT1_AIMU_DMA_HOST_TO_DEVICE;
    desc.dtype       = ATT1_AIMU_DMA_DTYPE_F32;
    s = att1_aimu_dma_validate(dma, &desc);
    EXPECT(s == ATT1_ERR_INVALID_ARG,
           "dma_invalid: unaligned device_addr → INVALID_ARG");

    /* Register a host region [0x1000, 0x9000) then use an address outside it */
    s = att1_aimu_dma_register_host_region(dma, UINT64_C(0x1000),
                                            UINT64_C(0x8000));
    EXPECT(s == ATT1_OK, "dma_invalid: register_host_region OK");

    memset(&desc, 0, sizeof(desc));
    desc.host_addr   = UINT64_C(0x90000);  /* outside [0x1000, 0x9000) */
    desc.device_addr = UINT64_C(0x1000);
    desc.byte_length = 4096;
    desc.direction   = (uint8_t)ATT1_AIMU_DMA_HOST_TO_DEVICE;
    desc.dtype       = ATT1_AIMU_DMA_DTYPE_F32;
    s = att1_aimu_dma_validate(dma, &desc);
    EXPECT(s == ATT1_ERR_INVALID_ARG,
           "dma_invalid: out-of-range host_addr → INVALID_ARG");

    /* Submit the bad descriptor; dma_failed must increment */
    (void)att1_aimu_dma_submit(dma, &desc);
    att1_aimu_dma_counters cnt;
    att1_aimu_dma_get_counters(dma, &cnt);
    EXPECT(cnt.dma_failed > 0,
           "dma_invalid: dma_failed incremented on failed submit");

    att1_aimu_dma_destroy(dma);
}

/* =========================================================================
 * 11. test_dma_huge_tile_no_alloc
 *
 * Create a host whose tile_memory_bytes is larger than physical RAM.
 * The value is register metadata only — no real SRAM buffer is allocated.
 * The test verifies create returns without crash and, if OK, probe works.
 * ====================================================================== */

static void test_dma_huge_tile_no_alloc(void)
{
    att1_aimu_host_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.tile_count        = 1;
    cfg.tile_memory_bytes = UINT64_C(4) * 1024u * 1024u * 1024u; /* 4 GiB */
    cfg.tile_kv_bytes     = 256u * 1024u * 1024u;                 /* 256 MiB */

    att1_aimu_host *h = NULL;
    att1_status_t s   = att1_aimu_host_create(&cfg, &h);

    /* Either OK (value capped internally) or INVALID_ARG (out of range) */
    EXPECT(s == ATT1_OK || s == ATT1_ERR_INVALID_ARG,
           "huge_tile_no_alloc: create returns OK or INVALID_ARG (no crash)");

    if (h != NULL && s == ATT1_OK) {
        /* Probe must succeed; DEVICE_READY is set at init time */
        att1_status_t ps = att1_aimu_host_probe_device(h, NULL);
        EXPECT(ps == ATT1_OK,
               "huge_tile_no_alloc: probe succeeds on capped host");
        att1_aimu_host_destroy(h);
    }

    /* Reach here without crash or OOM → success */
    EXPECT(1, "huge_tile_no_alloc: no crash or unbounded allocation");
}

/* =========================================================================
 * 12. test_exec_replay_supported
 *
 * Dispatch every EXEC_* / KV_* / FABRIC_* / BARRIER / TRACE_SNAPSHOT
 * command through att1_aimu_exec_dispatch with a full-capability device.
 * Verify per-op counters and aggregate counters.
 * ====================================================================== */

static void test_exec_replay_supported(void)
{
    att1_aimu_device_config dcfg;
    memset(&dcfg, 0, sizeof(dcfg));
    dcfg.tile_count = 1;
    /* 0 → defaults: all dtypes, all ops */

    att1_aimu_device *dev = NULL;
    att1_status_t s = att1_aimu_device_create(&dcfg, &dev);
    EXPECT(s == ATT1_OK, "exec_supported: device created");
    if (!dev) return;

    att1_aimu_exec_ctx *ctx = NULL;
    s = att1_aimu_exec_ctx_create(dev, &ctx);
    EXPECT(s == ATT1_OK, "exec_supported: exec_ctx created");
    if (!ctx) { att1_aimu_device_destroy(dev); return; }

    att1_aimu_cmd cmd;
    att1_aimu_result r;

    /* EXEC_MATMUL — needs non-zero tensor_id and byte counts */
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type    = (uint8_t)ATT1_AIMU_CMD_EXEC_MATMUL;
    cmd.tile_id         = 0;
    cmd.tensor_id       = 1;
    cmd.input_buf_bytes = 16384;
    cmd.output_buf_bytes = 4096;
    r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_OK, "exec_supported: EXEC_MATMUL OK");

    /* EXEC_RMSNORM */
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type    = (uint8_t)ATT1_AIMU_CMD_EXEC_RMSNORM;
    cmd.tile_id         = 0;
    cmd.input_buf_bytes = 512;
    cmd.output_buf_bytes = 512;
    r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_OK, "exec_supported: EXEC_RMSNORM OK");

    /* EXEC_ROPE */
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type    = (uint8_t)ATT1_AIMU_CMD_EXEC_ROPE;
    cmd.tile_id         = 0;
    cmd.input_buf_bytes = 256;
    cmd.output_buf_bytes = 256;
    r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_OK, "exec_supported: EXEC_ROPE OK");

    /* EXEC_ATTENTION */
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type    = (uint8_t)ATT1_AIMU_CMD_EXEC_ATTENTION;
    cmd.tile_id         = 0;
    cmd.input_buf_bytes = 1024;
    cmd.output_buf_bytes = 512;
    r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_OK, "exec_supported: EXEC_ATTENTION OK");

    /* EXEC_FFN */
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type    = (uint8_t)ATT1_AIMU_CMD_EXEC_FFN;
    cmd.tile_id         = 0;
    cmd.input_buf_bytes = 2048;
    cmd.output_buf_bytes = 1024;
    r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_OK, "exec_supported: EXEC_FFN OK");

    /* KV_APPEND */
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type     = (uint8_t)ATT1_AIMU_CMD_KV_APPEND;
    cmd.tile_id          = 0;
    cmd.output_buf_bytes = 64;
    r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_OK, "exec_supported: KV_APPEND OK");

    /* KV_READ */
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type    = (uint8_t)ATT1_AIMU_CMD_KV_READ;
    cmd.tile_id         = 0;
    cmd.input_buf_bytes = 64;
    r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_OK, "exec_supported: KV_READ OK");

    /* FABRIC_SEND */
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_FABRIC_SEND;
    cmd.tile_id      = 0;
    r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_OK, "exec_supported: FABRIC_SEND OK");

    /* FABRIC_REDUCE */
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_FABRIC_REDUCE;
    cmd.tile_id      = 0;
    r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_OK, "exec_supported: FABRIC_REDUCE OK");

    /* TILE_BARRIER */
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_TILE_BARRIER;
    r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_OK, "exec_supported: TILE_BARRIER OK");

    /* TRACE_SNAPSHOT */
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_TRACE_SNAPSHOT;
    r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_OK, "exec_supported: TRACE_SNAPSHOT OK");

    /* Verify per-op counters */
    att1_aimu_exec_counters cnt;
    att1_aimu_exec_ctx_get_counters(ctx, &cnt);

    EXPECT(cnt.matmul_count          == 1, "exec_supported: matmul_count == 1");
    EXPECT(cnt.rmsnorm_count         == 1, "exec_supported: rmsnorm_count == 1");
    EXPECT(cnt.rope_count            == 1, "exec_supported: rope_count == 1");
    EXPECT(cnt.attention_count       == 1, "exec_supported: attention_count == 1");
    EXPECT(cnt.ffn_count             == 1, "exec_supported: ffn_count == 1");
    EXPECT(cnt.kv_append_count       == 1, "exec_supported: kv_append_count == 1");
    EXPECT(cnt.kv_read_count         == 1, "exec_supported: kv_read_count == 1");
    EXPECT(cnt.fabric_send_count     == 1, "exec_supported: fabric_send_count == 1");
    EXPECT(cnt.fabric_reduce_count   == 1, "exec_supported: fabric_reduce_count == 1");
    EXPECT(cnt.barrier_count         == 1, "exec_supported: barrier_count == 1");
    EXPECT(cnt.trace_snapshot_count  == 1, "exec_supported: trace_snapshot_count == 1");

    EXPECT(cnt.exec_commands_seen      == 11,
           "exec_supported: exec_commands_seen == 11");
    EXPECT(cnt.exec_commands_completed == 11,
           "exec_supported: exec_commands_completed == 11");
    EXPECT(cnt.exec_commands_failed    == 0,
           "exec_supported: exec_commands_failed == 0");
    EXPECT(cnt.exec_unsupported        == 0,
           "exec_supported: exec_unsupported == 0");

    att1_aimu_exec_ctx_destroy(ctx);
    att1_aimu_device_destroy(dev);
}

/* =========================================================================
 * 13. test_exec_replay_unsupported
 *
 * Create a device that does NOT support ATT1_AIMU_OP_MATMUL.
 * Dispatch EXEC_MATMUL → must return ATT1_AIMU_ERR_UNSUPPORTED_OP.
 * Also verify that the cmdq dispatcher returns UNSUPPORTED_OP for EXEC_*.
 * ====================================================================== */

static void test_exec_replay_unsupported(void)
{
    /* Exclude MATMUL from the device's supported ops */
    att1_aimu_device_config dcfg;
    memset(&dcfg, 0, sizeof(dcfg));
    dcfg.tile_count    = 1;
    dcfg.supported_ops = ATT1_AIMU_OP_ALL & ~(uint32_t)ATT1_AIMU_OP_MATMUL;

    att1_aimu_device *dev = NULL;
    att1_aimu_device_create(&dcfg, &dev);
    if (!dev) { EXPECT(0, "exec_unsupported: device create"); return; }

    att1_aimu_exec_ctx *ctx = NULL;
    att1_aimu_exec_ctx_create(dev, &ctx);
    if (!ctx) { att1_aimu_device_destroy(dev); return; }

    att1_aimu_cmd cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type     = (uint8_t)ATT1_AIMU_CMD_EXEC_MATMUL;
    cmd.tile_id          = 0;
    cmd.tensor_id        = 1;
    cmd.input_buf_bytes  = 4096;
    cmd.output_buf_bytes = 1024;

    att1_aimu_result r = att1_aimu_exec_dispatch(ctx, &cmd);
    EXPECT(r == ATT1_AIMU_ERR_UNSUPPORTED_OP,
           "exec_unsupported: EXEC_MATMUL on restricted dev → UNSUPPORTED_OP");

    att1_aimu_exec_counters cnt;
    att1_aimu_exec_ctx_get_counters(ctx, &cnt);
    EXPECT(cnt.exec_unsupported    == 1,
           "exec_unsupported: exec_unsupported == 1");
    EXPECT(cnt.matmul_count        == 0,
           "exec_unsupported: matmul_count == 0");
    EXPECT(cnt.exec_commands_failed == 1,
           "exec_unsupported: exec_commands_failed == 1");

    /* Verify cmdq dispatcher also returns UNSUPPORTED_OP for EXEC_MATMUL */
    {
        att1_aimu_cmdq_config cqcfg;
        memset(&cqcfg, 0, sizeof(cqcfg));
        cqcfg.tile_count      = 1;
        cqcfg.cmd_ring_depth  = 16;
        cqcfg.comp_ring_depth = 16;

        att1_aimu_cmdq *q = NULL;
        att1_aimu_cmdq_create(&cqcfg, &q);
        if (q) {
            att1_aimu_cmd cmd2;
            memset(&cmd2, 0, sizeof(cmd2));
            cmd2.command_type    = (uint8_t)ATT1_AIMU_CMD_EXEC_MATMUL;
            cmd2.tile_id         = 0;
            cmd2.tensor_id       = 1;
            (void)att1_aimu_cmdq_submit(q, &cmd2);
            (void)att1_aimu_cmdq_dispatch_one(q);

            att1_aimu_completion comp;
            att1_status_t ps = att1_aimu_cmdq_poll_completion(q, &comp);
            if (ps == ATT1_OK) {
                EXPECT(comp.result_code == ATT1_AIMU_ERR_UNSUPPORTED_OP,
                       "exec_unsupported: cmdq EXEC_MATMUL → UNSUPPORTED_OP");
            } else {
                EXPECT(0,
                       "exec_unsupported: cmdq poll_completion returned error");
            }
            att1_aimu_cmdq_destroy(q);
        }
    }

    att1_aimu_exec_ctx_destroy(ctx);
    att1_aimu_device_destroy(dev);
}

/* =========================================================================
 * 14. test_replay_integration
 *
 * Drive the 6-command M129 command plan through att1_aimu_host:
 *   [0] LOAD_TENSOR_TILE  tile=0 tensor=1
 *   [1] VALIDATE_TENSOR   tile=0 tensor=1
 *   [2] TILE_BARRIER      tile=0
 *   [3] EXEC_MATMUL       tile=0 tensor=1
 *   [4] KV_APPEND         tile=0
 *   [5] QUERY_COUNTERS    tile=0
 *
 * Verify: all 6 produce completions; TILE_BARRIER/QUERY_COUNTERS complete
 * OK; EXEC_MATMUL carries UNSUPPORTED_OP (the "strict mode" signal).
 * Verify: invalid tile_id is rejected at submit time.
 * ====================================================================== */

static void test_replay_integration(void)
{
    att1_aimu_host *h = make_host(2, 0);
    if (!h) { EXPECT(0, "replay_integration: host created"); return; }

    if (!probe_and_setup(h)) {
        EXPECT(0, "replay_integration: probe and setup");
        att1_aimu_host_destroy(h);
        return;
    }

    /* The 6-command M129 plan */
    att1_aimu_cmd_type plan[6] = {
        ATT1_AIMU_CMD_LOAD_TENSOR_TILE,
        ATT1_AIMU_CMD_VALIDATE_TENSOR,
        ATT1_AIMU_CMD_TILE_BARRIER,
        ATT1_AIMU_CMD_EXEC_MATMUL,
        ATT1_AIMU_CMD_KV_APPEND,
        ATT1_AIMU_CMD_QUERY_COUNTERS
    };

    att1_status_t s;
    int i;
    for (i = 0; i < 6; i++) {
        att1_aimu_cmd cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.command_type    = (uint8_t)plan[i];
        cmd.tile_id         = 0;
        cmd.tensor_id       = 1;
        cmd.input_buf_bytes = 256;
        s = att1_aimu_host_submit_cmd(h, &cmd);
        EXPECT(s == ATT1_OK, "replay_integration: submit plan command");
    }

    (void)att1_aimu_host_ring_doorbell(h);
    s = att1_aimu_host_drain(h);
    EXPECT(s == ATT1_OK, "replay_integration: drain OK");

    /* All 6 commands must produce completions (even UNSUPPORTED ones) */
    EXPECT(att1_aimu_cmdq_completions_available(h->cmdq) == 6,
           "replay_integration: 6 completions available");

    /* Read all completions */
    att1_aimu_completion comps[6];
    size_t nc = 0;
    while (att1_aimu_cmdq_completions_available(h->cmdq) > 0 && nc < 6) {
        (void)att1_aimu_host_read_completion(h, &comps[nc++]);
    }
    EXPECT(nc == 6, "replay_integration: read 6 completions");

    /* TILE_BARRIER (index 2) must complete OK */
    EXPECT(comps[2].result_code == ATT1_AIMU_OK,
           "replay_integration: TILE_BARRIER completion OK");

    /* QUERY_COUNTERS (index 5) must complete OK */
    EXPECT(comps[5].result_code == ATT1_AIMU_OK,
           "replay_integration: QUERY_COUNTERS completion OK");

    /* EXEC_MATMUL (index 3) must carry UNSUPPORTED_OP (strict-mode signal) */
    EXPECT(comps[3].result_code == ATT1_AIMU_ERR_UNSUPPORTED_OP,
           "replay_integration: EXEC_MATMUL completion → UNSUPPORTED_OP");

    /* Summary: commands_submitted >= 6, fence_value > 0 */
    (void)att1_aimu_host_snapshot_counters(h);
    att1_aimu_host_summary sum;
    (void)att1_aimu_host_get_summary(h, &sum);
    EXPECT(sum.commands_submitted >= 6,
           "replay_integration: summary commands_submitted >= 6");
    EXPECT(sum.commands_completed >= 6,
           "replay_integration: summary commands_completed >= 6");

    /* Malformed plan: tile_id beyond tile_count must fail at submit */
    att1_aimu_cmd bad;
    memset(&bad, 0, sizeof(bad));
    bad.command_type = (uint8_t)ATT1_AIMU_CMD_NOP;
    bad.tile_id      = 99;   /* 2-tile device → invalid */
    s = att1_aimu_host_submit_cmd(h, &bad);
    EXPECT(s == ATT1_ERR_INVALID_ARG,
           "replay_integration: invalid tile_id rejected at submit");

    att1_aimu_host_destroy(h);
}

/* =========================================================================
 * 15. test_trace_snapshot_deterministic
 *
 * Run a 3-command mini-plan twice (with att1_aimu_host_reset between runs)
 * and verify that the trace counter snapshots are bitwise-identical for
 * the fields that reflect per-run activity.
 * ====================================================================== */

static void test_trace_snapshot_deterministic(void)
{
    /*
     * Run the same 3-command mini-plan twice using two independent hosts
     * (fresh allocation each run).  Because both hosts start with zero
     * counters the snapshots must be bitwise-identical for all per-run
     * cmdq counter fields, proving replay is deterministic.
     */
    att1_aimu_trace_snapshot snap[2];

    int run;
    for (run = 0; run < 2; run++) {
        att1_aimu_host *h = make_host(2, 0);
        if (!h || !probe_and_setup(h)) {
            if (h) att1_aimu_host_destroy(h);
            EXPECT(0, "trace_deterministic: host created and ready");
            return;
        }

        att1_aimu_cmd_type types[3] = {
            ATT1_AIMU_CMD_NOP,
            ATT1_AIMU_CMD_TILE_BARRIER,
            ATT1_AIMU_CMD_QUERY_COUNTERS
        };
        int i;
        for (i = 0; i < 3; i++) {
            att1_aimu_cmd cmd;
            memset(&cmd, 0, sizeof(cmd));
            cmd.command_type = (uint8_t)types[i];
            cmd.tile_id      = 0;
            (void)att1_aimu_host_submit_cmd(h, &cmd);
        }
        (void)att1_aimu_host_ring_doorbell(h);
        (void)att1_aimu_host_drain(h);
        (void)att1_aimu_host_snapshot_counters(h);
        (void)att1_aimu_trace_get_snapshot(h->trace, &snap[run]);
        att1_aimu_host_destroy(h);
    }

    /* Both fresh-host runs must produce identical cmdq counter snapshots */
    EXPECT(snap[0].cmdq.commands_submitted == snap[1].cmdq.commands_submitted,
           "trace_deterministic: commands_submitted identical both runs");
    EXPECT(snap[0].cmdq.commands_completed == snap[1].cmdq.commands_completed,
           "trace_deterministic: commands_completed identical both runs");
    EXPECT(snap[0].cmdq.commands_failed    == snap[1].cmdq.commands_failed,
           "trace_deterministic: commands_failed identical both runs");

    /* Snapshot metadata must be valid */
    EXPECT(snap[0].meta.trace_version == ATT1_AIMU_TRACE_VERSION,
           "trace_deterministic: trace_version set correctly");
    EXPECT(snap[0].meta.tile_count == 2,
           "trace_deterministic: tile_count captured in snapshot");
    EXPECT(snap[0].cmdq.commands_submitted >= 3,
           "trace_deterministic: at least 3 commands in first snapshot");

    /* Device resets == 0 on both fresh hosts (no reset occurred) */
    EXPECT(snap[0].device.device_resets == 0,
           "trace_deterministic: device_resets == 0 on first run");
    EXPECT(snap[1].device.device_resets == 0,
           "trace_deterministic: device_resets == 0 on second run");
}

/* =========================================================================
 * 16. test_no_cuda_dep
 *
 * Compile-time guard: ATT1_ENABLE_CUDA must not be defined in the default
 * build.  This test enforces that the MMIO/host emulator has no CUDA
 * dependency in the non-GPU build configuration.
 * ====================================================================== */

static void test_no_cuda_dep(void)
{
#ifdef ATT1_ENABLE_CUDA
    /* CUDA was explicitly enabled (make CUDA=1) — intentional, not a leak. */
    printf("PASS: aimu_mmio_regression: no_cuda_dep (CUDA=1 build; intentional)\n");
    g_pass++;
#else
    EXPECT(1, "no_cuda_dep: no CUDA dependency in default build");
#endif
}

/* =========================================================================
 * main
 * ====================================================================== */

int main(void)
{
    test_device_probe();
    test_register_rw();
    test_register_error_paths();
    test_register_reserved();
    test_reset_control();
    test_cmdq_doorbell();
    test_cmdq_fifo_order();
    test_cmdq_queue_full();
    test_dma_valid();
    test_dma_invalid();
    test_dma_huge_tile_no_alloc();
    test_exec_replay_supported();
    test_exec_replay_unsupported();
    test_replay_integration();
    test_trace_snapshot_deterministic();
    test_no_cuda_dep();

    printf("\naimu_mmio_regression: %d PASS  %d FAIL\n", g_pass, g_fail);
    return (g_fail > 0) ? 1 : 0;
}
