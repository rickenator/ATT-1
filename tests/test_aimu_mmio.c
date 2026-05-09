/*
 * test_aimu_mmio.c  —  Tests for the AIMU MMIO/register-file simulator (M111)
 *
 * Covers:
 *   1.  create/destroy lifecycle
 *   2.  DEVICE_ID reads ATT1_MMIO_DEVICE_ID_DEFAULT (no device attached)
 *   3.  REGISTER_MAP_VERSION reads ATT1_AIMU_REGISTER_MAP_VERSION
 *   4.  TILE_COUNT after attach_device + sync
 *   5.  Per-tile memory capacity after attach_device + sync
 *   6.  RO register write returns ATT1_ERR_UNSUPPORTED
 *   7.  RW register read/write round-trip
 *   8.  Reserved offset read returns ATT1_AIMU_MMIO_RESERVED_RD
 *   9.  Reserved offset write returns ATT1_OK (silently discarded)
 *  10.  Unaligned read/write returns ATT1_ERR_INVALID_ARG
 *  11.  64-bit read helper: read64 of a LOW/HIGH RW pair round-trips
 *  12.  RESET_CONTROL SOFT_RESET_ALL calls device reset and re-syncs
 *  13.  RESET_CONTROL RESET_COUNTERS zeroes counter register cells
 *  14.  CQ_DOORBELL write increments doorbell_write_count
 *  15.  COUNTER_SNAPSHOT_CONTROL SNAP_NOW increments snapshot_trigger_count
 *       and SNAP_NOW bit does not persist in backing store
 *  16.  Counter snapshot path: after SNAP_NOW, counter cells reflect cmdq
 *  17.  GLOBAL_CONTROL ENABLE_DEVICE/DISABLE_DEVICE toggles DEVICE_READY bit
 *  18.  TILE_RESET_CONTROL write calls device tile reset
 *  19.  RW1C INTERRUPT_STATUS clear behaviour
 *  20.  NULL / invalid args return ATT1_ERR_INVALID_ARG
 *  21.  Out-of-range offset fails clearly
 *  22.  No hidden CUDA dependency
 *
 * No ATT-1 inference, backend, tokenizer, CUDA, or binary-format behaviour
 * is changed by these tests.
 */

#include "att1_aimu_mmio.h"
#include "att1_aimu_device.h"
#include "att1_aimu_cmdq.h"
#include "att1_aimu_trace.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>

/* =========================================================================
 * Minimal test harness
 * ====================================================================== */

static int g_pass = 0;
static int g_fail = 0;

#define EXPECT(cond, name) \
    do { \
        if (cond) { \
            printf("PASS: aimu_mmio: " name "\n"); \
            g_pass++; \
        } else { \
            printf("FAIL: aimu_mmio: " name "\n"); \
            g_fail++; \
        } \
    } while (0)

/* =========================================================================
 * Helper: create a default device (4 tiles, 1 GiB each)
 * ====================================================================== */

static att1_aimu_device *make_device(size_t tile_count,
                                      uint64_t mem_bytes,
                                      uint64_t kv_bytes)
{
    att1_aimu_device_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.tile_count        = tile_count;
    cfg.tile_memory_bytes = mem_bytes;
    cfg.tile_kv_bytes     = kv_bytes;
    att1_aimu_device *dev = NULL;
    if (att1_aimu_device_create(&cfg, &dev) != ATT1_OK) return NULL;
    return dev;
}

/* =========================================================================
 * Test cases
 * ====================================================================== */

static void test_create_destroy(void)
{
    att1_aimu_mmio *m = NULL;

    /* create with NULL out → error */
    EXPECT(att1_aimu_mmio_create(NULL) == ATT1_ERR_INVALID_ARG,
           "create NULL out returns error");

    /* normal create */
    att1_status_t rc = att1_aimu_mmio_create(&m);
    EXPECT(rc == ATT1_OK && m != NULL, "create succeeds");
    EXPECT(m->magic == ATT1_AIMU_MMIO_MAGIC, "magic set correctly");

    /* destroy */
    att1_aimu_mmio_destroy(m);
    EXPECT(1, "destroy does not crash");

    /* destroy NULL is safe */
    att1_aimu_mmio_destroy(NULL);
    EXPECT(1, "destroy NULL is safe");
}

static void test_default_registers(void)
{
    att1_aimu_mmio *m = NULL;
    att1_aimu_mmio_create(&m);

    uint32_t val = 0;

    /* DEVICE_ID */
    att1_aimu_mmio_read32(m, ATT1_MMIO_DEVICE_ID, &val);
    EXPECT(val == ATT1_MMIO_DEVICE_ID_DEFAULT,
           "DEVICE_ID reads default (vendor=0xA771, class=0x0001)");

    /* REGISTER_MAP_VERSION */
    att1_aimu_mmio_read32(m, ATT1_MMIO_REGISTER_MAP_VERSION, &val);
    EXPECT(val == ATT1_AIMU_REGISTER_MAP_VERSION,
           "REGISTER_MAP_VERSION reads ATT1_AIMU_REGISTER_MAP_VERSION");

    /* GLOBAL_STATUS should have DEVICE_READY set */
    att1_aimu_mmio_read32(m, ATT1_MMIO_GLOBAL_STATUS, &val);
    EXPECT((val & ATT1_MMIO_GSTAT_DEVICE_READY) != 0,
           "GLOBAL_STATUS.DEVICE_READY set by default");

    att1_aimu_mmio_destroy(m);
}

static void test_tile_count_after_attach(void)
{
    att1_aimu_mmio  *m   = NULL;
    att1_aimu_device *dev = make_device(4, 0, 0);
    att1_aimu_mmio_create(&m);

    att1_aimu_mmio_attach_device(m, dev);
    att1_aimu_mmio_sync(m);

    uint32_t val = 0;
    att1_aimu_mmio_read32(m, ATT1_MMIO_TILE_COUNT, &val);
    EXPECT(val == 4u, "TILE_COUNT == 4 after attach + sync");

    att1_aimu_mmio_destroy(m);
    att1_aimu_device_destroy(dev);
}

static void test_tile_memory_capacity(void)
{
    const uint64_t MEM = UINT64_C(4) * 1024u * 1024u * 1024u; /* 4 GiB */

    att1_aimu_mmio  *m   = NULL;
    att1_aimu_device *dev = make_device(2, MEM, 0);
    att1_aimu_mmio_create(&m);
    att1_aimu_mmio_attach_device(m, dev);
    att1_aimu_mmio_sync(m);

    /* Read tile 0 memory capacity as two 32-bit halves and combine */
    uint32_t cap_lo = 0, cap_hi = 0;
    att1_status_t rc;
    rc = att1_aimu_mmio_read32(
        m, ATT1_MMIO_TILE_REG(0, ATT1_MMIO_TILE_MEMORY_CAPACITY_LOW), &cap_lo);
    EXPECT(rc == ATT1_OK, "tile[0] MEMORY_CAPACITY_LOW read returns ATT1_OK");
    rc = att1_aimu_mmio_read32(
        m, ATT1_MMIO_TILE_REG(0, ATT1_MMIO_TILE_MEMORY_CAPACITY_HIGH), &cap_hi);
    EXPECT(rc == ATT1_OK, "tile[0] MEMORY_CAPACITY_HIGH read returns ATT1_OK");
    uint64_t cap = ((uint64_t)cap_hi << 32u) | (uint64_t)cap_lo;
    EXPECT(cap == MEM,
           "tile[0] MEMORY_CAPACITY reads back 4 GiB");

    att1_aimu_mmio_destroy(m);
    att1_aimu_device_destroy(dev);
}

static void test_ro_write_rejected(void)
{
    att1_aimu_mmio *m = NULL;
    att1_aimu_mmio_create(&m);

    att1_status_t rc = att1_aimu_mmio_write32(m, ATT1_MMIO_DEVICE_ID, 0xDEADBEEFu);
    EXPECT(rc == ATT1_ERR_UNSUPPORTED,
           "write to RO DEVICE_ID returns ATT1_ERR_UNSUPPORTED");

    rc = att1_aimu_mmio_write32(m, ATT1_MMIO_GLOBAL_STATUS, 0xFFFFFFFFu);
    EXPECT(rc == ATT1_ERR_UNSUPPORTED,
           "write to RO GLOBAL_STATUS returns ATT1_ERR_UNSUPPORTED");

    att1_aimu_mmio_destroy(m);
}

static void test_rw_round_trip(void)
{
    att1_aimu_mmio *m = NULL;
    att1_aimu_mmio_create(&m);

    /* CQ_BASE_ADDR_LOW is RW */
    const uint32_t WVAL = UINT32_C(0x12345678);
    att1_status_t rc = att1_aimu_mmio_write32(m, ATT1_MMIO_CQ_BASE_ADDR_LOW, WVAL);
    EXPECT(rc == ATT1_OK, "RW write returns ATT1_OK");

    uint32_t val = 0;
    att1_aimu_mmio_read32(m, ATT1_MMIO_CQ_BASE_ADDR_LOW, &val);
    EXPECT(val == WVAL, "RW register read-back matches written value");

    /* INTERRUPT_ENABLE is RW */
    att1_aimu_mmio_write32(m, ATT1_MMIO_INTERRUPT_ENABLE, 0xFFu);
    att1_aimu_mmio_read32(m, ATT1_MMIO_INTERRUPT_ENABLE, &val);
    EXPECT(val == 0xFFu, "INTERRUPT_ENABLE RW preserves written value");

    att1_aimu_mmio_destroy(m);
}

static void test_reserved_offset(void)
{
    att1_aimu_mmio *m = NULL;
    att1_aimu_mmio_create(&m);

    /* 0x0040 is in the reserved range between global and CQ windows */
    uint32_t val = 0;
    att1_status_t rc = att1_aimu_mmio_read32(m, 0x0040u, &val);
    EXPECT(rc == ATT1_OK && val == ATT1_AIMU_MMIO_RESERVED_RD,
           "reserved offset read returns 0xDEADBEEF");

    rc = att1_aimu_mmio_write32(m, 0x0040u, 0xABCDEFu);
    EXPECT(rc == ATT1_OK, "reserved offset write returns ATT1_OK (discarded)");

    /* Value should still be 0xDEADBEEF after discarded write */
    att1_aimu_mmio_read32(m, 0x0040u, &val);
    EXPECT(val == ATT1_AIMU_MMIO_RESERVED_RD,
           "reserved offset still reads 0xDEADBEEF after discarded write");

    att1_aimu_mmio_destroy(m);
}

static void test_unaligned_access(void)
{
    att1_aimu_mmio *m = NULL;
    att1_aimu_mmio_create(&m);

    uint32_t val = 0;
    /* Unaligned read */
    att1_status_t rc = att1_aimu_mmio_read32(m, 0x0001u, &val);
    EXPECT(rc == ATT1_ERR_INVALID_ARG,
           "unaligned read32 (offset 0x0001) returns ATT1_ERR_INVALID_ARG");

    rc = att1_aimu_mmio_read32(m, 0x0002u, &val);
    EXPECT(rc == ATT1_ERR_INVALID_ARG,
           "unaligned read32 (offset 0x0002) returns ATT1_ERR_INVALID_ARG");

    /* Unaligned write */
    rc = att1_aimu_mmio_write32(m, 0x0003u, 0u);
    EXPECT(rc == ATT1_ERR_INVALID_ARG,
           "unaligned write32 (offset 0x0003) returns ATT1_ERR_INVALID_ARG");

    /* Unaligned read64 (not 8-byte aligned) */
    uint64_t v64 = 0;
    rc = att1_aimu_mmio_read64(m, 0x0004u, &v64);
    EXPECT(rc == ATT1_ERR_INVALID_ARG,
           "read64 offset 0x0004 (not 8-byte aligned) returns ATT1_ERR_INVALID_ARG");

    att1_aimu_mmio_destroy(m);
}

static void test_read64_write64(void)
{
    att1_aimu_mmio *m = NULL;
    att1_aimu_mmio_create(&m);

    /* CQ_BASE_ADDR_LOW/HIGH (0x1000/0x1004) is an RW LOW/HIGH pair, 8-byte aligned */
    const uint64_t WVAL = UINT64_C(0xDEADBEEF12345678);
    att1_status_t rc = att1_aimu_mmio_write64(m, ATT1_MMIO_CQ_BASE_ADDR_LOW, WVAL);
    EXPECT(rc == ATT1_OK, "write64 of CQ_BASE_ADDR returns ATT1_OK");

    uint64_t rval = 0;
    rc = att1_aimu_mmio_read64(m, ATT1_MMIO_CQ_BASE_ADDR_LOW, &rval);
    EXPECT(rc == ATT1_OK && rval == WVAL,
           "read64 of CQ_BASE_ADDR round-trips written 64-bit value");

    att1_aimu_mmio_destroy(m);
}

static void test_reset_control_soft_reset(void)
{
    att1_aimu_mmio  *m   = NULL;
    att1_aimu_device *dev = make_device(2, 0, 0);
    att1_aimu_mmio_create(&m);
    att1_aimu_mmio_attach_device(m, dev);
    att1_aimu_mmio_sync(m);

    uint32_t before_reset_count = dev->reset_count;

    /* Write RESET_CONTROL.SOFT_RESET_ALL */
    att1_status_t rc = att1_aimu_mmio_write32(
        m, ATT1_MMIO_RESET_CONTROL, ATT1_MMIO_RSTCTL_SOFT_RESET_ALL);
    EXPECT(rc == ATT1_OK, "RESET_CONTROL SOFT_RESET_ALL write returns ATT1_OK");

    /* Device reset_count should have incremented */
    EXPECT(dev->reset_count == before_reset_count + 1u,
           "SOFT_RESET_ALL increments device reset_count");

    /* GLOBAL_STATUS.DEVICE_READY should still be set after reset */
    uint32_t gstat = 0;
    att1_aimu_mmio_read32(m, ATT1_MMIO_GLOBAL_STATUS, &gstat);
    EXPECT((gstat & ATT1_MMIO_GSTAT_DEVICE_READY) != 0,
           "GLOBAL_STATUS.DEVICE_READY set after SOFT_RESET_ALL");

    /* RESET_CONTROL is WO: read returns 0 */
    uint32_t val = 0xFFFFu;
    att1_aimu_mmio_read32(m, ATT1_MMIO_RESET_CONTROL, &val);
    EXPECT(val == 0u, "RESET_CONTROL reads 0 (WO register)");

    att1_aimu_mmio_destroy(m);
    att1_aimu_device_destroy(dev);
}

static void test_reset_counters(void)
{
    att1_aimu_mmio  *m   = NULL;
    att1_aimu_device *dev = make_device(1, 0, 0);
    att1_aimu_cmdq  *q   = NULL;
    att1_aimu_mmio_create(&m);
    att1_aimu_cmdq_create(NULL, &q);
    att1_aimu_mmio_attach_device(m, dev);
    att1_aimu_mmio_attach_cmdq(m, q);
    att1_aimu_mmio_sync(m);

    /* Manually set a counter cell to a non-zero value */
    m->regs[ATT1_MMIO_CNT_CMD_ISSUED_LO / 4u] = 0xABCDu;

    /* RESET_COUNTERS should zero it */
    att1_aimu_mmio_write32(m, ATT1_MMIO_RESET_CONTROL,
                            ATT1_MMIO_RSTCTL_RESET_COUNTERS);

    uint32_t val = 0xFFFFu;
    att1_aimu_mmio_read32(m, ATT1_MMIO_CNT_CMD_ISSUED_LO, &val);
    EXPECT(val == 0u, "RESET_COUNTERS zeroes counter register cells");

    att1_aimu_mmio_destroy(m);
    att1_aimu_cmdq_destroy(q);
    att1_aimu_device_destroy(dev);
}

static void test_doorbell_write(void)
{
    att1_aimu_mmio *m = NULL;
    att1_aimu_mmio_create(&m);

    EXPECT(m->doorbell_write_count == 0u,
           "doorbell_write_count starts at 0");

    att1_aimu_mmio_write32(m, ATT1_MMIO_CQ_DOORBELL, 5u);
    EXPECT(m->doorbell_write_count == 1u,
           "CQ_DOORBELL write increments doorbell_write_count");

    att1_aimu_mmio_write32(m, ATT1_MMIO_CQ_DOORBELL, 10u);
    EXPECT(m->doorbell_write_count == 2u,
           "second CQ_DOORBELL write increments doorbell_write_count again");

    /* New tail should be reflected in CQ_TAIL */
    uint32_t tail = 0;
    att1_aimu_mmio_read32(m, ATT1_MMIO_CQ_TAIL, &tail);
    EXPECT(tail == 10u, "CQ_TAIL updated by CQ_DOORBELL write");

    /* CQ_DOORBELL is WO: read returns 0 */
    uint32_t v = 0xFFFFu;
    att1_aimu_mmio_read32(m, ATT1_MMIO_CQ_DOORBELL, &v);
    EXPECT(v == 0u, "CQ_DOORBELL reads 0 (WO register)");

    att1_aimu_mmio_destroy(m);
}

static void test_snapshot_control(void)
{
    att1_aimu_mmio  *m   = NULL;
    att1_aimu_device *dev = make_device(1, 0, 0);
    att1_aimu_cmdq  *q   = NULL;
    att1_aimu_trace *tr  = NULL;
    att1_aimu_mmio_create(&m);
    att1_aimu_cmdq_create(NULL, &q);
    att1_aimu_trace_create(&tr);
    att1_aimu_mmio_attach_device(m, dev);
    att1_aimu_mmio_attach_cmdq(m, q);
    att1_aimu_mmio_attach_trace(m, tr);
    att1_aimu_mmio_sync(m);

    EXPECT(m->snapshot_trigger_count == 0u,
           "snapshot_trigger_count starts at 0");

    /* Trigger snapshot via COUNTER_SNAPSHOT_CONTROL.SNAP_NOW */
    att1_aimu_mmio_write32(m, ATT1_MMIO_COUNTER_SNAPSHOT_CONTROL,
                            ATT1_MMIO_SNAP_NOW);
    EXPECT(m->snapshot_trigger_count == 1u,
           "SNAP_NOW write increments snapshot_trigger_count");

    /* SNAP_NOW bit should be self-clearing */
    uint32_t val = 0;
    att1_aimu_mmio_read32(m, ATT1_MMIO_COUNTER_SNAPSHOT_CONTROL, &val);
    EXPECT((val & ATT1_MMIO_SNAP_NOW) == 0u,
           "SNAP_NOW bit does not persist (self-clearing)");

    /* Second trigger */
    att1_aimu_mmio_write32(m, ATT1_MMIO_COUNTER_SNAPSHOT_CONTROL,
                            ATT1_MMIO_SNAP_NOW);
    EXPECT(m->snapshot_trigger_count == 2u,
           "second SNAP_NOW write increments snapshot_trigger_count again");

    att1_aimu_mmio_destroy(m);
    att1_aimu_trace_destroy(tr);
    att1_aimu_cmdq_destroy(q);
    att1_aimu_device_destroy(dev);
}

static void test_snapshot_reflects_counters(void)
{
    att1_aimu_mmio  *m   = NULL;
    att1_aimu_device *dev = make_device(1, 0, 0);
    att1_aimu_cmdq  *q   = NULL;
    att1_aimu_trace *tr  = NULL;
    att1_aimu_mmio_create(&m);
    att1_aimu_cmdq_create(NULL, &q);
    att1_aimu_trace_create(&tr);
    att1_aimu_mmio_attach_device(m, dev);
    att1_aimu_mmio_attach_cmdq(m, q);
    att1_aimu_mmio_attach_trace(m, tr);

    /* Submit a command to drive cmdq counters */
    att1_aimu_cmd cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = ATT1_AIMU_CMD_NOP;
    cmd.tile_id      = 0;
    att1_aimu_cmdq_submit(q, &cmd);
    att1_aimu_cmdq_dispatch_all(q);

    /* Trigger snapshot; counters should flow to counter registers */
    att1_aimu_mmio_write32(m, ATT1_MMIO_COUNTER_SNAPSHOT_CONTROL,
                            ATT1_MMIO_SNAP_NOW);

    uint64_t issued = 0;
    att1_aimu_mmio_read64(m, ATT1_MMIO_CNT_CMD_ISSUED_LO, &issued);
    EXPECT(issued == 1u,
           "CNT_CMD_ISSUED counter reflects submitted command after SNAP_NOW");

    att1_aimu_mmio_destroy(m);
    att1_aimu_trace_destroy(tr);
    att1_aimu_cmdq_destroy(q);
    att1_aimu_device_destroy(dev);
}

static void test_global_control(void)
{
    att1_aimu_mmio *m = NULL;
    att1_aimu_mmio_create(&m);

    /* Initially DEVICE_READY should be set */
    uint32_t gstat = 0;
    att1_aimu_mmio_read32(m, ATT1_MMIO_GLOBAL_STATUS, &gstat);
    EXPECT((gstat & ATT1_MMIO_GSTAT_DEVICE_READY) != 0u,
           "GLOBAL_STATUS.DEVICE_READY set initially");

    /* DISABLE_DEVICE clears DEVICE_READY */
    att1_aimu_mmio_write32(m, ATT1_MMIO_GLOBAL_CONTROL,
                            ATT1_MMIO_GCTRL_DISABLE_DEVICE);
    att1_aimu_mmio_read32(m, ATT1_MMIO_GLOBAL_STATUS, &gstat);
    EXPECT((gstat & ATT1_MMIO_GSTAT_DEVICE_READY) == 0u,
           "DISABLE_DEVICE clears GLOBAL_STATUS.DEVICE_READY");

    /* ENABLE_DEVICE restores DEVICE_READY */
    att1_aimu_mmio_write32(m, ATT1_MMIO_GLOBAL_CONTROL,
                            ATT1_MMIO_GCTRL_ENABLE_DEVICE);
    att1_aimu_mmio_read32(m, ATT1_MMIO_GLOBAL_STATUS, &gstat);
    EXPECT((gstat & ATT1_MMIO_GSTAT_DEVICE_READY) != 0u,
           "ENABLE_DEVICE restores GLOBAL_STATUS.DEVICE_READY");

    /* GLOBAL_CONTROL reads as 0 (WO) */
    uint32_t ctrl = 0xFFu;
    att1_aimu_mmio_read32(m, ATT1_MMIO_GLOBAL_CONTROL, &ctrl);
    EXPECT(ctrl == 0u, "GLOBAL_CONTROL reads 0 (WO register)");

    att1_aimu_mmio_destroy(m);
}

static void test_tile_reset_control(void)
{
    att1_aimu_mmio  *m   = NULL;
    att1_aimu_device *dev = make_device(2, 0, 0);
    att1_aimu_mmio_create(&m);
    att1_aimu_mmio_attach_device(m, dev);
    att1_aimu_mmio_sync(m);

    uint32_t before = dev->tiles[0].reset_count;

    /* Write TILE_RESET_CONTROL for tile 0 */
    att1_status_t rc = att1_aimu_mmio_write32(
        m,
        ATT1_MMIO_TILE_REG(0, ATT1_MMIO_TILE_RESET_CONTROL),
        UINT32_C(0x1));  /* RESET_TILE_SOFT */
    EXPECT(rc == ATT1_OK, "TILE_RESET_CONTROL write returns ATT1_OK");
    EXPECT(dev->tiles[0].reset_count == before + 1u,
           "TILE_RESET_CONTROL triggers tile reset in device simulator");

    /* TILE_RESET_CONTROL reads as 0 (WO) */
    uint32_t v = 0xFFu;
    att1_aimu_mmio_read32(m, ATT1_MMIO_TILE_REG(0, ATT1_MMIO_TILE_RESET_CONTROL), &v);
    EXPECT(v == 0u, "TILE_RESET_CONTROL reads 0 (WO register)");

    att1_aimu_mmio_destroy(m);
    att1_aimu_device_destroy(dev);
}

static void test_rw1c_interrupt_status(void)
{
    att1_aimu_mmio *m = NULL;
    att1_aimu_mmio_create(&m);

    /* Manually set some interrupt bits */
    m->regs[ATT1_MMIO_INTERRUPT_STATUS / 4u] = 0x07u;  /* bits 0,1,2 set */

    uint32_t val = 0;
    att1_aimu_mmio_read32(m, ATT1_MMIO_INTERRUPT_STATUS, &val);
    EXPECT(val == 0x07u, "INTERRUPT_STATUS reads back pre-set value");

    /* Write 1 to bit 1 to clear it; bits 0 and 2 should remain */
    att1_aimu_mmio_write32(m, ATT1_MMIO_INTERRUPT_STATUS, 0x02u);
    att1_aimu_mmio_read32(m, ATT1_MMIO_INTERRUPT_STATUS, &val);
    EXPECT(val == 0x05u, "RW1C: writing 0x02 to 0x07 yields 0x05");

    /* Writing 0 to RW1C should have no effect */
    att1_aimu_mmio_write32(m, ATT1_MMIO_INTERRUPT_STATUS, 0x00u);
    att1_aimu_mmio_read32(m, ATT1_MMIO_INTERRUPT_STATUS, &val);
    EXPECT(val == 0x05u, "RW1C: writing 0 does not clear any bit");

    att1_aimu_mmio_destroy(m);
}

static void test_null_invalid_args(void)
{
    att1_aimu_mmio *m = NULL;
    att1_aimu_mmio_create(&m);

    uint32_t val = 0;

    /* NULL m */
    EXPECT(att1_aimu_mmio_read32(NULL, 0x0000u, &val) == ATT1_ERR_INVALID_ARG,
           "read32 NULL m returns ATT1_ERR_INVALID_ARG");
    EXPECT(att1_aimu_mmio_write32(NULL, 0x0000u, 0u) == ATT1_ERR_INVALID_ARG,
           "write32 NULL m returns ATT1_ERR_INVALID_ARG");

    /* NULL out */
    EXPECT(att1_aimu_mmio_read32(m, 0x0000u, NULL) == ATT1_ERR_INVALID_ARG,
           "read32 NULL out returns ATT1_ERR_INVALID_ARG");

    /* NULL for sync */
    EXPECT(att1_aimu_mmio_sync(NULL) == ATT1_ERR_INVALID_ARG,
           "sync NULL returns ATT1_ERR_INVALID_ARG");

    /* NULL for reset */
    EXPECT(att1_aimu_mmio_reset(NULL) == ATT1_ERR_INVALID_ARG,
           "reset NULL returns ATT1_ERR_INVALID_ARG");

    /* NULL for attach_device */
    EXPECT(att1_aimu_mmio_attach_device(NULL, NULL) == ATT1_ERR_INVALID_ARG,
           "attach_device NULL m returns ATT1_ERR_INVALID_ARG");

    att1_aimu_mmio_destroy(m);
}

static void test_out_of_range_offset(void)
{
    att1_aimu_mmio *m = NULL;
    att1_aimu_mmio_create(&m);

    uint32_t val = 0;

    /* One past the end of BAR0 */
    EXPECT(att1_aimu_mmio_read32(m, ATT1_AIMU_MMIO_BAR0_SIZE, &val) ==
               ATT1_ERR_INVALID_ARG,
           "read32 at BAR0_SIZE returns ATT1_ERR_INVALID_ARG");

    EXPECT(att1_aimu_mmio_write32(m, ATT1_AIMU_MMIO_BAR0_SIZE, 0u) ==
               ATT1_ERR_INVALID_ARG,
           "write32 at BAR0_SIZE returns ATT1_ERR_INVALID_ARG");

    att1_aimu_mmio_destroy(m);
}

static void test_no_cuda_dependency(void)
{
    /*
     * This test confirms the module was compiled without ATT1_ENABLE_CUDA.
     * If CUDA were accidentally required, the macro would be defined.
     */
#ifdef ATT1_ENABLE_CUDA
    EXPECT(0, "no hidden CUDA dependency (ATT1_ENABLE_CUDA is NOT defined)");
#else
    EXPECT(1, "no hidden CUDA dependency (ATT1_ENABLE_CUDA is not defined)");
#endif
}

static void test_render(void)
{
    att1_aimu_mmio *m = NULL;
    att1_aimu_mmio_create(&m);

    att1_status_t rc = att1_aimu_mmio_render(m, stdout);
    EXPECT(rc == ATT1_OK, "render to stdout returns ATT1_OK");

    rc = att1_aimu_mmio_render(m, NULL);
    EXPECT(rc == ATT1_ERR_INVALID_ARG, "render NULL fp returns ATT1_ERR_INVALID_ARG");

    att1_aimu_mmio_destroy(m);
}

static void test_err_name(void)
{
    EXPECT(strcmp(att1_aimu_mmio_err_name(ATT1_OK),              "ATT1_OK")             == 0,
           "err_name ATT1_OK");
    EXPECT(strcmp(att1_aimu_mmio_err_name(ATT1_ERR_INVALID_ARG), "ATT1_ERR_INVALID_ARG") == 0,
           "err_name ATT1_ERR_INVALID_ARG");
    EXPECT(strcmp(att1_aimu_mmio_err_name(ATT1_ERR_UNSUPPORTED), "ATT1_ERR_UNSUPPORTED") == 0,
           "err_name ATT1_ERR_UNSUPPORTED");
    EXPECT(strcmp(att1_aimu_mmio_err_name((att1_status_t)-999),  "ATT1_UNKNOWN")         == 0,
           "err_name unknown value returns ATT1_UNKNOWN");
}

/* =========================================================================
 * main
 * ====================================================================== */

int main(void)
{
    test_create_destroy();
    test_default_registers();
    test_tile_count_after_attach();
    test_tile_memory_capacity();
    test_ro_write_rejected();
    test_rw_round_trip();
    test_reserved_offset();
    test_unaligned_access();
    test_read64_write64();
    test_reset_control_soft_reset();
    test_reset_counters();
    test_doorbell_write();
    test_snapshot_control();
    test_snapshot_reflects_counters();
    test_global_control();
    test_tile_reset_control();
    test_rw1c_interrupt_status();
    test_null_invalid_args();
    test_out_of_range_offset();
    test_no_cuda_dependency();
    test_render();
    test_err_name();

    if (g_fail > 0) {
        printf("aimu_mmio: %d passed, %d FAILED\n", g_pass, g_fail);
        return 1;
    }
    printf("aimu_mmio: %d passed, 0 failed\n", g_pass);
    return 0;
}
