/*
 * test_aimu_trace.c  —  Unit tests for the AIMU unified trace/counter
 *                        snapshot module (M108)
 */

#include "att1_aimu_trace.h"
#include "att1_aimu_cmdq.h"
#include "att1_aimu_device.h"
#include "att1_aimu_dma.h"
#include "att1_status.h"

#include <stdio.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------------- */

#define PASS(name) do { printf("PASS: aimu_trace: %s\n", (name)); } while (0)
#define FAIL(name) do { printf("FAIL: aimu_trace: %s\n", (name)); return 1; } while (0)
#define REQUIRE(cond, name) do { if (!(cond)) { FAIL(name); } } while (0)

/* Create a trace; assert success and return it. */
static att1_aimu_trace *make_trace(void)
{
    att1_aimu_trace *t = NULL;
    att1_status_t    st = att1_aimu_trace_create(&t);
    if (st != ATT1_OK || !t) {
        printf("FAIL: aimu_trace: helper make_trace() failed\n");
        return NULL;
    }
    return t;
}

/* Create a cmdq with default config (1 tile, default ring depths). */
static att1_aimu_cmdq *make_cmdq(void)
{
    att1_aimu_cmdq *q  = NULL;
    att1_aimu_cmdq_create(NULL, &q);
    return q;
}

/* Create a device with default config (1 tile). */
static att1_aimu_device *make_device(void)
{
    att1_aimu_device *dev = NULL;
    att1_aimu_device_create(NULL, &dev);
    return dev;
}

/* Create a DMA simulator with no registered regions (permissive mode). */
static att1_aimu_dma *make_dma(void)
{
    att1_aimu_dma *sim = NULL;
    att1_aimu_dma_create(&sim);
    return sim;
}

/* Submit a NOP command to q and dispatch it, returning 0 on success. */
static int submit_nop(att1_aimu_cmdq *q)
{
    att1_aimu_cmd cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_NOP;
    cmd.tile_id      = 0u;
    if (att1_aimu_cmdq_submit(q, &cmd) != ATT1_OK)
        return 1;
    if (att1_aimu_cmdq_dispatch_one(q) != ATT1_OK)
        return 1;
    return 0;
}

/* Submit an unsupported command (EXEC_MATMUL) and dispatch it. */
static int submit_unsupported(att1_aimu_cmdq *q)
{
    att1_aimu_cmd cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_EXEC_MATMUL;
    cmd.tile_id      = 0u;
    if (att1_aimu_cmdq_submit(q, &cmd) != ATT1_OK)
        return 1;
    /* Dispatch produces a completion with ERR_UNSUPPORTED_OP. */
    att1_aimu_cmdq_dispatch_one(q);
    return 0;
}

/* Build a valid H2D DMA descriptor (no region registration needed). */
static att1_aimu_dma_desc make_h2d_desc(void)
{
    att1_aimu_dma_desc d;
    memset(&d, 0, sizeof(d));
    d.host_addr   = UINT64_C(0x1000);       /* 64-aligned */
    d.device_addr = UINT64_C(0x80000000);   /* 64-aligned */
    d.byte_length = 4096u;
    d.direction   = (uint8_t)ATT1_AIMU_DMA_HOST_TO_DEVICE;
    d.dtype       = ATT1_AIMU_DMA_DTYPE_F32;
    return d;
}

/* -------------------------------------------------------------------------
 * Tests
 * ---------------------------------------------------------------------- */

/* Basic lifecycle: create / destroy / double-free-safe */
static int test_create_destroy(void)
{
    att1_aimu_trace *t  = NULL;
    att1_status_t    st = att1_aimu_trace_create(&t);

    REQUIRE(st == ATT1_OK,             "create_destroy: returns OK");
    REQUIRE(t  != NULL,                "create_destroy: out non-null");
    REQUIRE(t->magic == ATT1_AIMU_TRACE_MAGIC, "create_destroy: magic set");

    att1_aimu_trace_destroy(t);
    att1_aimu_trace_destroy(NULL);  /* must not crash */

    PASS("create_destroy");
    return 0;
}

/* NULL out pointer is rejected */
static int test_create_null_out(void)
{
    att1_status_t st = att1_aimu_trace_create(NULL);
    REQUIRE(st == ATT1_ERR_INVALID_ARG, "create_null_out: rejected");
    PASS("create_null_out");
    return 0;
}

/* After create, all snapshot counters are zero and status is EMPTY */
static int test_empty_snapshot_zeroed(void)
{
    att1_aimu_trace          *t    = make_trace();
    att1_aimu_trace_snapshot  snap;

    REQUIRE(t != NULL, "empty_snapshot_zeroed: trace created");
    REQUIRE(att1_aimu_trace_get_snapshot(t, &snap) == ATT1_OK,
            "empty_snapshot_zeroed: get_snapshot OK");

    REQUIRE(snap.meta.status         == ATT1_AIMU_TRACE_STATUS_EMPTY,
            "empty_snapshot_zeroed: status EMPTY");
    REQUIRE(snap.meta.snapshot_id    == 0u,
            "empty_snapshot_zeroed: snapshot_id 0");
    REQUIRE(snap.meta.trace_version  == ATT1_AIMU_TRACE_VERSION,
            "empty_snapshot_zeroed: trace_version set");

    REQUIRE(snap.cmdq.commands_submitted   == 0u, "empty: cmdq zeroed");
    REQUIRE(snap.device.device_resets      == 0u, "empty: device zeroed");
    REQUIRE(snap.dma.dma_submitted         == 0u, "empty: dma zeroed");
    REQUIRE(snap.fabric.packets_sent       == 0u, "empty: fabric zeroed");

    att1_aimu_trace_destroy(t);
    PASS("empty_snapshot_zeroed");
    return 0;
}

/* After reset, snapshot counters return to zero and status is EMPTY */
static int test_reset_clears_snapshot(void)
{
    att1_aimu_trace  *t   = make_trace();
    att1_aimu_cmdq   *q   = make_cmdq();
    att1_status_t     st;

    REQUIRE(t && q, "reset_clears: setup OK");
    submit_nop(q);

    att1_aimu_trace_snapshot_all(t, q, NULL, NULL);

    /* Verify something was recorded */
    REQUIRE(t->snapshot.cmdq.commands_submitted > 0u,
            "reset_clears: recorded before reset");

    st = att1_aimu_trace_reset(t);
    REQUIRE(st == ATT1_OK, "reset_clears: reset returns OK");
    REQUIRE(t->snapshot.cmdq.commands_submitted == 0u,
            "reset_clears: cmdq zeroed after reset");
    REQUIRE(t->snapshot.meta.status == ATT1_AIMU_TRACE_STATUS_EMPTY,
            "reset_clears: status EMPTY after reset");
    REQUIRE(t->snapshot.meta.snapshot_id == 0u,
            "reset_clears: snapshot_id 0 after reset");
    REQUIRE(t->snapshot.meta.trace_version == ATT1_AIMU_TRACE_VERSION,
            "reset_clears: version preserved after reset");

    att1_aimu_cmdq_destroy(q);
    att1_aimu_trace_destroy(t);
    PASS("reset_clears_snapshot");
    return 0;
}

/* snapshot_cmdq captures command counts correctly */
static int test_snapshot_cmdq_basic(void)
{
    att1_aimu_trace *t = make_trace();
    att1_aimu_cmdq  *q = make_cmdq();

    REQUIRE(t && q, "snapshot_cmdq_basic: setup OK");

    /* Submit and dispatch two NOPs. */
    REQUIRE(submit_nop(q) == 0, "snapshot_cmdq_basic: nop 1 OK");
    REQUIRE(submit_nop(q) == 0, "snapshot_cmdq_basic: nop 2 OK");

    REQUIRE(att1_aimu_trace_snapshot_cmdq(t, q) == ATT1_OK,
            "snapshot_cmdq_basic: returns OK");

    REQUIRE(t->snapshot.cmdq.commands_submitted  == 2u,
            "snapshot_cmdq_basic: submitted == 2");
    REQUIRE(t->snapshot.cmdq.commands_completed  == 2u,
            "snapshot_cmdq_basic: completed == 2");
    REQUIRE(t->snapshot.cmdq.commands_failed     == 0u,
            "snapshot_cmdq_basic: failed == 0");
    REQUIRE(t->snapshot.cmdq.unsupported_commands == 0u,
            "snapshot_cmdq_basic: unsupported == 0");

    att1_aimu_cmdq_destroy(q);
    att1_aimu_trace_destroy(t);
    PASS("snapshot_cmdq_basic");
    return 0;
}

/* snapshot_cmdq: NULL trace is rejected */
static int test_snapshot_cmdq_null_trace(void)
{
    att1_aimu_cmdq *q  = make_cmdq();
    att1_status_t   st = att1_aimu_trace_snapshot_cmdq(NULL, q);

    REQUIRE(st == ATT1_ERR_INVALID_ARG, "snapshot_cmdq_null_trace: rejected");

    att1_aimu_cmdq_destroy(q);
    PASS("snapshot_cmdq_null_trace");
    return 0;
}

/* snapshot_cmdq: NULL source is rejected */
static int test_snapshot_cmdq_null_src(void)
{
    att1_aimu_trace *t  = make_trace();
    att1_status_t    st = att1_aimu_trace_snapshot_cmdq(t, NULL);

    REQUIRE(st == ATT1_ERR_INVALID_ARG, "snapshot_cmdq_null_src: rejected");

    att1_aimu_trace_destroy(t);
    PASS("snapshot_cmdq_null_src");
    return 0;
}

/* snapshot_dma captures DMA counts correctly */
static int test_snapshot_dma_basic(void)
{
    att1_aimu_trace    *t   = make_trace();
    att1_aimu_dma      *sim = make_dma();
    att1_aimu_dma_desc  d   = make_h2d_desc();

    REQUIRE(t && sim, "snapshot_dma_basic: setup OK");

    /* Submit two valid H2D descriptors. */
    REQUIRE(att1_aimu_dma_submit(sim, &d) == ATT1_OK,
            "snapshot_dma_basic: submit 1 OK");
    REQUIRE(att1_aimu_dma_submit(sim, &d) == ATT1_OK,
            "snapshot_dma_basic: submit 2 OK");

    REQUIRE(att1_aimu_trace_snapshot_dma(t, sim) == ATT1_OK,
            "snapshot_dma_basic: returns OK");

    REQUIRE(t->snapshot.dma.dma_submitted        == 2u,
            "snapshot_dma_basic: submitted == 2");
    REQUIRE(t->snapshot.dma.dma_completed        == 2u,
            "snapshot_dma_basic: completed == 2");
    REQUIRE(t->snapshot.dma.dma_failed           == 0u,
            "snapshot_dma_basic: failed == 0");
    REQUIRE(t->snapshot.dma.bytes_host_to_device == 2u * 4096u,
            "snapshot_dma_basic: bytes_h2d correct");

    att1_aimu_dma_destroy(sim);
    att1_aimu_trace_destroy(t);
    PASS("snapshot_dma_basic");
    return 0;
}

/* snapshot_dma: NULL trace is rejected */
static int test_snapshot_dma_null_trace(void)
{
    att1_aimu_dma *sim = make_dma();
    att1_status_t  st  = att1_aimu_trace_snapshot_dma(NULL, sim);

    REQUIRE(st == ATT1_ERR_INVALID_ARG, "snapshot_dma_null_trace: rejected");

    att1_aimu_dma_destroy(sim);
    PASS("snapshot_dma_null_trace");
    return 0;
}

/* snapshot_dma: NULL source is rejected */
static int test_snapshot_dma_null_src(void)
{
    att1_aimu_trace *t  = make_trace();
    att1_status_t    st = att1_aimu_trace_snapshot_dma(t, NULL);

    REQUIRE(st == ATT1_ERR_INVALID_ARG, "snapshot_dma_null_src: rejected");

    att1_aimu_trace_destroy(t);
    PASS("snapshot_dma_null_src");
    return 0;
}

/* snapshot_device captures device reset count */
static int test_snapshot_device_basic(void)
{
    att1_aimu_trace  *t   = make_trace();
    att1_aimu_device *dev = make_device();

    REQUIRE(t && dev, "snapshot_device_basic: setup OK");

    /* Reset the device once — increments reset_count. */
    REQUIRE(att1_aimu_device_reset(dev) == ATT1_OK,
            "snapshot_device_basic: device reset OK");

    REQUIRE(att1_aimu_trace_snapshot_device(t, dev) == ATT1_OK,
            "snapshot_device_basic: returns OK");

    REQUIRE(t->snapshot.device.device_resets == 1u,
            "snapshot_device_basic: device_resets == 1");
    REQUIRE(t->snapshot.meta.tile_count      == 1u,
            "snapshot_device_basic: tile_count captured");

    att1_aimu_device_destroy(dev);
    att1_aimu_trace_destroy(t);
    PASS("snapshot_device_basic");
    return 0;
}

/* snapshot_device: NULL trace is rejected */
static int test_snapshot_device_null_trace(void)
{
    att1_aimu_device *dev = make_device();
    att1_status_t     st  = att1_aimu_trace_snapshot_device(NULL, dev);

    REQUIRE(st == ATT1_ERR_INVALID_ARG, "snapshot_device_null_trace: rejected");

    att1_aimu_device_destroy(dev);
    PASS("snapshot_device_null_trace");
    return 0;
}

/* snapshot_device: NULL source is rejected */
static int test_snapshot_device_null_src(void)
{
    att1_aimu_trace *t  = make_trace();
    att1_status_t    st = att1_aimu_trace_snapshot_device(t, NULL);

    REQUIRE(st == ATT1_ERR_INVALID_ARG, "snapshot_device_null_src: rejected");

    att1_aimu_trace_destroy(t);
    PASS("snapshot_device_null_src");
    return 0;
}

/* snapshot_all wires all three sources into one snapshot */
static int test_snapshot_all_combined(void)
{
    att1_aimu_trace  *t   = make_trace();
    att1_aimu_cmdq   *q   = make_cmdq();
    att1_aimu_device *dev = make_device();
    att1_aimu_dma    *sim = make_dma();
    att1_aimu_dma_desc d  = make_h2d_desc();

    REQUIRE(t && q && dev && sim, "snapshot_all_combined: setup OK");

    submit_nop(q);
    att1_aimu_device_reset(dev);
    att1_aimu_dma_submit(sim, &d);

    REQUIRE(att1_aimu_trace_snapshot_all(t, q, dev, sim) == ATT1_OK,
            "snapshot_all_combined: returns OK");

    REQUIRE(t->snapshot.meta.status == ATT1_AIMU_TRACE_STATUS_OK,
            "snapshot_all_combined: status OK");
    REQUIRE(t->snapshot.cmdq.commands_submitted  == 1u,
            "snapshot_all_combined: cmdq present");
    REQUIRE(t->snapshot.device.device_resets     == 1u,
            "snapshot_all_combined: device present");
    REQUIRE(t->snapshot.dma.dma_submitted        == 1u,
            "snapshot_all_combined: dma present");

    att1_aimu_dma_destroy(sim);
    att1_aimu_device_destroy(dev);
    att1_aimu_cmdq_destroy(q);
    att1_aimu_trace_destroy(t);
    PASS("snapshot_all_combined");
    return 0;
}

/* snapshot_all: NULL trace is rejected */
static int test_snapshot_all_null_trace(void)
{
    att1_status_t st = att1_aimu_trace_snapshot_all(NULL, NULL, NULL, NULL);
    REQUIRE(st == ATT1_ERR_INVALID_ARG, "snapshot_all_null_trace: rejected");
    PASS("snapshot_all_null_trace");
    return 0;
}

/* snapshot_all with some NULL sources → status PARTIAL */
static int test_snapshot_all_partial(void)
{
    att1_aimu_trace *t = make_trace();
    att1_aimu_cmdq  *q = make_cmdq();

    REQUIRE(t && q, "snapshot_all_partial: setup OK");
    submit_nop(q);

    REQUIRE(att1_aimu_trace_snapshot_all(t, q, NULL, NULL) == ATT1_OK,
            "snapshot_all_partial: returns OK");
    REQUIRE(t->snapshot.meta.status == ATT1_AIMU_TRACE_STATUS_PARTIAL,
            "snapshot_all_partial: status PARTIAL");
    REQUIRE(t->snapshot.cmdq.commands_submitted == 1u,
            "snapshot_all_partial: cmdq captured");
    REQUIRE(t->snapshot.device.device_resets == 0u,
            "snapshot_all_partial: device zeroed");
    REQUIRE(t->snapshot.dma.dma_submitted == 0u,
            "snapshot_all_partial: dma zeroed");

    att1_aimu_cmdq_destroy(q);
    att1_aimu_trace_destroy(t);
    PASS("snapshot_all_partial");
    return 0;
}

/* snapshot_all with all NULL sources → PARTIAL but no crash */
static int test_snapshot_all_null_sources(void)
{
    att1_aimu_trace *t  = make_trace();
    att1_status_t    st = att1_aimu_trace_snapshot_all(t, NULL, NULL, NULL);

    REQUIRE(st == ATT1_OK, "snapshot_all_null_sources: returns OK");
    REQUIRE(t->snapshot.meta.status     == ATT1_AIMU_TRACE_STATUS_PARTIAL,
            "snapshot_all_null_sources: status PARTIAL");
    REQUIRE(t->snapshot.meta.snapshot_id == 1u,
            "snapshot_all_null_sources: snapshot_id incremented");

    att1_aimu_trace_destroy(t);
    PASS("snapshot_all_null_sources");
    return 0;
}

/* snapshot_all increments snapshot_id on each call */
static int test_snapshot_id_increments(void)
{
    att1_aimu_trace *t = make_trace();

    REQUIRE(t, "snapshot_id_increments: setup OK");
    REQUIRE(t->snapshot.meta.snapshot_id == 0u, "snapshot_id: starts at 0");

    att1_aimu_trace_snapshot_all(t, NULL, NULL, NULL);
    REQUIRE(t->snapshot.meta.snapshot_id == 1u, "snapshot_id: after 1st == 1");

    att1_aimu_trace_snapshot_all(t, NULL, NULL, NULL);
    REQUIRE(t->snapshot.meta.snapshot_id == 2u, "snapshot_id: after 2nd == 2");

    att1_aimu_trace_destroy(t);
    PASS("snapshot_id_increments");
    return 0;
}

/* Same simulator state → identical snapshot on repeated calls */
static int test_snapshot_deterministic(void)
{
    att1_aimu_trace          *t   = make_trace();
    att1_aimu_cmdq           *q   = make_cmdq();
    att1_aimu_trace_snapshot  s1, s2;

    REQUIRE(t && q, "snapshot_deterministic: setup OK");
    submit_nop(q);

    att1_aimu_trace_snapshot_cmdq(t, q);
    REQUIRE(att1_aimu_trace_get_snapshot(t, &s1) == ATT1_OK,
            "snapshot_deterministic: first get OK");

    /* Take a second snapshot without changing the source. */
    att1_aimu_trace_snapshot_cmdq(t, q);
    REQUIRE(att1_aimu_trace_get_snapshot(t, &s2) == ATT1_OK,
            "snapshot_deterministic: second get OK");

    REQUIRE(s1.cmdq.commands_submitted == s2.cmdq.commands_submitted,
            "snapshot_deterministic: cmdq matches");
    REQUIRE(s1.cmdq.commands_completed == s2.cmdq.commands_completed,
            "snapshot_deterministic: completed matches");

    att1_aimu_cmdq_destroy(q);
    att1_aimu_trace_destroy(t);
    PASS("snapshot_deterministic");
    return 0;
}

/* snapshot_cmdq must not mutate source cmdq counters */
static int test_does_not_mutate_cmdq(void)
{
    att1_aimu_trace *t = make_trace();
    att1_aimu_cmdq  *q = make_cmdq();

    REQUIRE(t && q, "does_not_mutate_cmdq: setup OK");
    submit_nop(q);
    submit_nop(q);

    uint64_t before_submitted  = q->counters.commands_submitted;
    uint64_t before_completed  = q->counters.commands_completed;

    att1_aimu_trace_snapshot_cmdq(t, q);

    REQUIRE(q->counters.commands_submitted == before_submitted,
            "does_not_mutate_cmdq: submitted unchanged");
    REQUIRE(q->counters.commands_completed == before_completed,
            "does_not_mutate_cmdq: completed unchanged");

    att1_aimu_cmdq_destroy(q);
    att1_aimu_trace_destroy(t);
    PASS("does_not_mutate_cmdq");
    return 0;
}

/* snapshot_dma must not mutate source DMA counters */
static int test_does_not_mutate_dma(void)
{
    att1_aimu_trace    *t   = make_trace();
    att1_aimu_dma      *sim = make_dma();
    att1_aimu_dma_desc  d   = make_h2d_desc();

    REQUIRE(t && sim, "does_not_mutate_dma: setup OK");
    att1_aimu_dma_submit(sim, &d);

    uint64_t before_submitted = sim->counters.dma_submitted;
    uint64_t before_bytes     = sim->counters.bytes_host_to_device;

    att1_aimu_trace_snapshot_dma(t, sim);

    REQUIRE(sim->counters.dma_submitted          == before_submitted,
            "does_not_mutate_dma: dma_submitted unchanged");
    REQUIRE(sim->counters.bytes_host_to_device   == before_bytes,
            "does_not_mutate_dma: bytes_h2d unchanged");

    att1_aimu_dma_destroy(sim);
    att1_aimu_trace_destroy(t);
    PASS("does_not_mutate_dma");
    return 0;
}

/* Unsupported command appears in unsupported_commands counter */
static int test_unsupported_cmd_counted(void)
{
    att1_aimu_trace *t = make_trace();
    att1_aimu_cmdq  *q = make_cmdq();

    REQUIRE(t && q, "unsupported_cmd_counted: setup OK");
    REQUIRE(submit_unsupported(q) == 0,
            "unsupported_cmd_counted: submit OK");

    att1_aimu_trace_snapshot_cmdq(t, q);

    REQUIRE(t->snapshot.cmdq.unsupported_commands == 1u,
            "unsupported_cmd_counted: unsupported == 1");
    /* Unsupported ops produce a completion (so host can drain the ring),
     * so they count in commands_completed, not commands_failed. */
    REQUIRE(t->snapshot.cmdq.commands_completed   == 1u,
            "unsupported_cmd_counted: completed == 1");
    REQUIRE(t->snapshot.cmdq.commands_submitted   == 1u,
            "unsupported_cmd_counted: submitted == 1");

    att1_aimu_cmdq_destroy(q);
    att1_aimu_trace_destroy(t);
    PASS("unsupported_cmd_counted");
    return 0;
}

/* Tiles in ERROR state are counted in tile_errors */
static int test_tile_errors_counted(void)
{
    att1_aimu_device_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.tile_count = 3u;

    att1_aimu_device *dev = NULL;
    att1_aimu_device_create(&cfg, &dev);
    att1_aimu_trace  *t   = make_trace();

    REQUIRE(dev && t, "tile_errors_counted: setup OK");

    /* Force two tiles into the ERROR state by direct field write.
     * The device struct is concrete (not opaque), so this is valid in tests. */
    dev->tiles[0].state = ATT1_AIMU_TILE_ERROR;
    dev->tiles[2].state = ATT1_AIMU_TILE_ERROR;

    att1_aimu_trace_snapshot_device(t, dev);

    REQUIRE(t->snapshot.device.tile_errors == 2u,
            "tile_errors_counted: tile_errors == 2");

    att1_aimu_device_destroy(dev);
    att1_aimu_trace_destroy(t);
    PASS("tile_errors_counted");
    return 0;
}

/* tile_resets is the sum of all per-tile reset counts */
static int test_tile_resets_summed(void)
{
    att1_aimu_device_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.tile_count = 2u;

    att1_aimu_device *dev = NULL;
    att1_aimu_device_create(&cfg, &dev);
    att1_aimu_trace  *t   = make_trace();

    REQUIRE(dev && t, "tile_resets_summed: setup OK");

    /* Reset tile 0 twice, tile 1 once → sum = 3. */
    att1_aimu_device_reset_tile(dev, 0u);
    att1_aimu_device_reset_tile(dev, 0u);
    att1_aimu_device_reset_tile(dev, 1u);

    att1_aimu_trace_snapshot_device(t, dev);

    REQUIRE(t->snapshot.device.tile_resets == 3u,
            "tile_resets_summed: tile_resets == 3");

    att1_aimu_device_destroy(dev);
    att1_aimu_trace_destroy(t);
    PASS("tile_resets_summed");
    return 0;
}

/* Device-level reset count appears in device_resets */
static int test_device_resets_in_snapshot(void)
{
    att1_aimu_trace  *t   = make_trace();
    att1_aimu_device *dev = make_device();

    REQUIRE(t && dev, "device_resets_in_snapshot: setup OK");

    att1_aimu_device_reset(dev);
    att1_aimu_device_reset(dev);

    att1_aimu_trace_snapshot_device(t, dev);

    REQUIRE(t->snapshot.device.device_resets == 2u,
            "device_resets_in_snapshot: device_resets == 2");

    att1_aimu_device_destroy(dev);
    att1_aimu_trace_destroy(t);
    PASS("device_resets_in_snapshot");
    return 0;
}

/* Fabric placeholder counters are always zero */
static int test_fabric_placeholder_zero(void)
{
    att1_aimu_trace          *t = make_trace();
    att1_aimu_trace_snapshot  snap;

    REQUIRE(t, "fabric_placeholder_zero: setup OK");
    att1_aimu_trace_get_snapshot(t, &snap);

    REQUIRE(snap.fabric.packets_sent           == 0u, "fabric: packets_sent zero");
    REQUIRE(snap.fabric.packets_received       == 0u, "fabric: packets_received zero");
    REQUIRE(snap.fabric.payload_bytes_sent     == 0u, "fabric: payload_bytes_sent zero");
    REQUIRE(snap.fabric.payload_bytes_received == 0u, "fabric: payload_bytes_received zero");
    REQUIRE(snap.fabric.congestion_events      == 0u, "fabric: congestion_events zero");

    /* Also after snapshot_all with all three real sources. */
    att1_aimu_cmdq   *q   = make_cmdq();
    att1_aimu_device *dev = make_device();
    att1_aimu_dma    *sim = make_dma();

    REQUIRE(q && dev && sim, "fabric_placeholder_zero: simulators created");
    att1_aimu_trace_snapshot_all(t, q, dev, sim);
    att1_aimu_trace_get_snapshot(t, &snap);

    REQUIRE(snap.fabric.packets_sent == 0u, "fabric: still zero after snapshot_all");

    att1_aimu_dma_destroy(sim);
    att1_aimu_device_destroy(dev);
    att1_aimu_cmdq_destroy(q);
    att1_aimu_trace_destroy(t);
    PASS("fabric_placeholder_zero");
    return 0;
}

/* render writes output to a file and returns OK */
static int test_render_to_file(void)
{
    att1_aimu_trace          *t   = make_trace();
    att1_aimu_cmdq           *q   = make_cmdq();
    att1_aimu_dma            *sim = make_dma();
    att1_aimu_dma_desc        d   = make_h2d_desc();
    att1_aimu_trace_snapshot  snap;
    att1_status_t             st;
    FILE                     *f;

    REQUIRE(t && q && sim, "render_to_file: setup OK");
    submit_nop(q);
    att1_aimu_dma_submit(sim, &d);
    att1_aimu_trace_snapshot_all(t, q, NULL, sim);
    att1_aimu_trace_get_snapshot(t, &snap);

    f = tmpfile();
    REQUIRE(f != NULL, "render_to_file: tmpfile() OK");

    st = att1_aimu_trace_render(&snap, f);
    REQUIRE(st == ATT1_OK, "render_to_file: returns OK");

    /* Verify bytes were actually written. */
    REQUIRE(ftell(f) > 0, "render_to_file: bytes written");

    fclose(f);
    att1_aimu_dma_destroy(sim);
    att1_aimu_cmdq_destroy(q);
    att1_aimu_trace_destroy(t);
    PASS("render_to_file");
    return 0;
}

/* render rejects null arguments */
static int test_render_null_args(void)
{
    att1_aimu_trace_snapshot snap;
    memset(&snap, 0, sizeof(snap));

    REQUIRE(att1_aimu_trace_render(NULL, stdout) == ATT1_ERR_INVALID_ARG,
            "render_null_args: null snap rejected");
    REQUIRE(att1_aimu_trace_render(&snap, NULL)  == ATT1_ERR_INVALID_ARG,
            "render_null_args: null file rejected");
    PASS("render_null_args");
    return 0;
}

/* get_snapshot copies the current state correctly */
static int test_get_snapshot(void)
{
    att1_aimu_trace          *t   = make_trace();
    att1_aimu_cmdq           *q   = make_cmdq();
    att1_aimu_trace_snapshot  snap;

    REQUIRE(t && q, "get_snapshot: setup OK");
    submit_nop(q);
    att1_aimu_trace_snapshot_all(t, q, NULL, NULL);

    REQUIRE(att1_aimu_trace_get_snapshot(t, &snap) == ATT1_OK,
            "get_snapshot: returns OK");
    REQUIRE(snap.cmdq.commands_submitted == t->snapshot.cmdq.commands_submitted,
            "get_snapshot: cmdq matches");
    REQUIRE(snap.meta.snapshot_id == t->snapshot.meta.snapshot_id,
            "get_snapshot: snapshot_id matches");

    att1_aimu_cmdq_destroy(q);
    att1_aimu_trace_destroy(t);
    PASS("get_snapshot");
    return 0;
}

/* get_snapshot rejects null arguments */
static int test_get_snapshot_null(void)
{
    att1_aimu_trace         *t    = make_trace();
    att1_aimu_trace_snapshot snap;

    REQUIRE(t, "get_snapshot_null: setup OK");

    REQUIRE(att1_aimu_trace_get_snapshot(NULL, &snap) == ATT1_ERR_INVALID_ARG,
            "get_snapshot_null: null trace rejected");
    REQUIRE(att1_aimu_trace_get_snapshot(t, NULL)     == ATT1_ERR_INVALID_ARG,
            "get_snapshot_null: null out rejected");

    att1_aimu_trace_destroy(t);
    PASS("get_snapshot_null");
    return 0;
}

/* reset rejects null argument */
static int test_reset_null(void)
{
    att1_status_t st = att1_aimu_trace_reset(NULL);
    REQUIRE(st == ATT1_ERR_INVALID_ARG, "reset_null: rejected");
    PASS("reset_null");
    return 0;
}

/* att1_aimu_trace_status_name returns correct strings */
static int test_status_name_helpers(void)
{
    REQUIRE(strcmp(att1_aimu_trace_status_name(ATT1_AIMU_TRACE_STATUS_OK),
                   "OK") == 0,
            "status_name: OK");
    REQUIRE(strcmp(att1_aimu_trace_status_name(ATT1_AIMU_TRACE_STATUS_PARTIAL),
                   "PARTIAL") == 0,
            "status_name: PARTIAL");
    REQUIRE(strcmp(att1_aimu_trace_status_name(ATT1_AIMU_TRACE_STATUS_EMPTY),
                   "EMPTY") == 0,
            "status_name: EMPTY");
    REQUIRE(strcmp(att1_aimu_trace_status_name(0xDEADBEEFu),
                   "UNKNOWN") == 0,
            "status_name: UNKNOWN");
    PASS("status_name_helpers");
    return 0;
}

/* DMA failures are reflected in the snapshot */
static int test_dma_failures_reflected(void)
{
    att1_aimu_trace    *t   = make_trace();
    att1_aimu_dma      *sim = make_dma();
    att1_aimu_dma_desc  d   = make_h2d_desc();

    REQUIRE(t && sim, "dma_failures_reflected: setup OK");

    /* Bad alignment → alignment_failure */
    d.host_addr = 0x1001u;  /* misaligned */
    att1_aimu_dma_submit(sim, &d);

    att1_aimu_trace_snapshot_dma(t, sim);

    REQUIRE(t->snapshot.dma.dma_submitted      == 1u,
            "dma_failures_reflected: submitted == 1");
    REQUIRE(t->snapshot.dma.dma_failed         == 1u,
            "dma_failures_reflected: failed == 1");
    REQUIRE(t->snapshot.dma.alignment_failures == 1u,
            "dma_failures_reflected: alignment_failures == 1");

    att1_aimu_dma_destroy(sim);
    att1_aimu_trace_destroy(t);
    PASS("dma_failures_reflected");
    return 0;
}

/*
 * No hidden CUDA dependency: ensure the header and implementation compile
 * and function without ATT1_ENABLE_CUDA or any CUDA header.
 */
static int test_no_cuda_dependency(void)
{
    /*
     * If ATT1_ENABLE_CUDA were required, this translation unit (which does
     * not include any CUDA headers) would fail to compile.  Reaching this
     * point proves the trace module is CUDA-free.
     */
#ifdef ATT1_ENABLE_CUDA
    /* CUDA is opt-in; we only verify the trace API compiles cleanly. */
#endif
    att1_aimu_trace *t = make_trace();
    REQUIRE(t != NULL, "no_cuda_dependency: create OK without CUDA");
    att1_aimu_trace_destroy(t);
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
    rc |= test_empty_snapshot_zeroed();
    rc |= test_reset_clears_snapshot();
    rc |= test_snapshot_cmdq_basic();
    rc |= test_snapshot_cmdq_null_trace();
    rc |= test_snapshot_cmdq_null_src();
    rc |= test_snapshot_dma_basic();
    rc |= test_snapshot_dma_null_trace();
    rc |= test_snapshot_dma_null_src();
    rc |= test_snapshot_device_basic();
    rc |= test_snapshot_device_null_trace();
    rc |= test_snapshot_device_null_src();
    rc |= test_snapshot_all_combined();
    rc |= test_snapshot_all_null_trace();
    rc |= test_snapshot_all_partial();
    rc |= test_snapshot_all_null_sources();
    rc |= test_snapshot_id_increments();
    rc |= test_snapshot_deterministic();
    rc |= test_does_not_mutate_cmdq();
    rc |= test_does_not_mutate_dma();
    rc |= test_unsupported_cmd_counted();
    rc |= test_tile_errors_counted();
    rc |= test_tile_resets_summed();
    rc |= test_device_resets_in_snapshot();
    rc |= test_fabric_placeholder_zero();
    rc |= test_render_to_file();
    rc |= test_render_null_args();
    rc |= test_get_snapshot();
    rc |= test_get_snapshot_null();
    rc |= test_reset_null();
    rc |= test_status_name_helpers();
    rc |= test_dma_failures_reflected();
    rc |= test_no_cuda_dependency();

    return rc;
}
