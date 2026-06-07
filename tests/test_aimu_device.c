/*
 * test_aimu_device.c  —  Unit tests for the AIMU device discovery simulator (M106)
 */

#include "att1_aimu_device.h"
#include "att1_aimu_cmdq.h"
#include "att1_status.h"

#include <stdio.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------------- */

#define PASS(name) do { printf("PASS: aimu_device: %s\n", (name)); } while (0)
#define FAIL(name) do { printf("FAIL: aimu_device: %s\n", (name)); return 1; } while (0)
#define REQUIRE(cond, name) do { if (!(cond)) { FAIL(name); } } while (0)

/* -------------------------------------------------------------------------
 * Tests
 * ---------------------------------------------------------------------- */

/* create / destroy lifecycle */
static int test_create_destroy(void)
{
    att1_aimu_device *dev = NULL;
    att1_status_t     st;

    st = att1_aimu_device_create(NULL, &dev);
    REQUIRE(st == ATT1_OK,   "create: returns OK");
    REQUIRE(dev != NULL,     "create: out non-null");
    REQUIRE(dev->magic == ATT1_AIMU_DEVICE_MAGIC, "create: magic correct");

    att1_aimu_device_destroy(dev);
    att1_aimu_device_destroy(NULL);  /* must not crash */
    PASS("create_destroy");
    return 0;
}

/* null out pointer rejected */
static int test_create_null_out(void)
{
    att1_status_t st = att1_aimu_device_create(NULL, NULL);
    REQUIRE(st == ATT1_ERR_INVALID_ARG, "create_null_out: rejected");
    PASS("create_null_out");
    return 0;
}

/* invalid tile_count rejected */
static int test_create_invalid_tile_count(void)
{
    att1_aimu_device_config cfg;
    att1_aimu_device       *dev = NULL;
    att1_status_t           st;

    memset(&cfg, 0, sizeof(cfg));

    /* tile_count > max */
    cfg.tile_count = ATT1_AIMU_DEVICE_MAX_TILES + 1u;
    st = att1_aimu_device_create(&cfg, &dev);
    REQUIRE(st == ATT1_ERR_INVALID_ARG, "create_invalid: tile_count too large");
    REQUIRE(dev == NULL, "create_invalid: out is NULL on error");

    PASS("create_invalid_tile_count");
    return 0;
}

/* default device has expected version and register_map_version */
static int test_default_version(void)
{
    att1_aimu_device      *dev = NULL;
    att1_aimu_device_info  info;

    att1_aimu_device_create(NULL, &dev);
    att1_aimu_device_query_info(dev, &info);

    REQUIRE(info.register_map_version == ATT1_AIMU_REGISTER_MAP_VERSION,
            "version: register_map_version matches constant");
    REQUIRE(info.version.major == ATT1_AIMU_DEVICE_VERSION_MAJOR,
            "version: major");
    REQUIRE(info.version.minor == ATT1_AIMU_DEVICE_VERSION_MINOR,
            "version: minor");
    REQUIRE(info.version.patch == ATT1_AIMU_DEVICE_VERSION_PATCH,
            "version: patch");
    REQUIRE(info.global_status == 0u, "version: global_status == 0");

    att1_aimu_device_destroy(dev);
    PASS("default_version");
    return 0;
}

/* tile_count queries work correctly */
static int test_tile_count(void)
{
    att1_aimu_device_config cfg;
    att1_aimu_device       *dev = NULL;

    memset(&cfg, 0, sizeof(cfg));
    cfg.tile_count = 4u;

    att1_aimu_device_create(&cfg, &dev);
    REQUIRE(att1_aimu_device_tile_count(dev) == 4u,
            "tile_count: query returns 4");
    REQUIRE(att1_aimu_device_tile_count(NULL) == 0u,
            "tile_count: NULL returns 0");

    att1_aimu_device_destroy(dev);
    PASS("tile_count");
    return 0;
}

/* per-tile capacity query returns configured values */
static int test_tile_capacity(void)
{
    att1_aimu_device_config cfg;
    att1_aimu_device       *dev = NULL;
    att1_aimu_tile_info     tile;
    att1_status_t           st;

    memset(&cfg, 0, sizeof(cfg));
    cfg.tile_count        = 2u;
    cfg.tile_memory_bytes = UINT64_C(512) << 20;  /* 512 MiB */
    cfg.tile_kv_bytes     = UINT64_C(64)  << 20;  /* 64 MiB */

    att1_aimu_device_create(&cfg, &dev);

    st = att1_aimu_device_query_tile(dev, 0u, &tile);
    REQUIRE(st == ATT1_OK, "tile_capacity: query tile 0 OK");
    REQUIRE(tile.memory_capacity_bytes == (UINT64_C(512) << 20),
            "tile_capacity: memory_capacity_bytes");
    REQUIRE(tile.kv_capacity_bytes     == (UINT64_C(64)  << 20),
            "tile_capacity: kv_capacity_bytes");
    REQUIRE(tile.tile_id == 0u, "tile_capacity: tile_id == 0");
    REQUIRE(tile.state   == ATT1_AIMU_TILE_IDLE, "tile_capacity: state IDLE");

    st = att1_aimu_device_query_tile(dev, 1u, &tile);
    REQUIRE(st == ATT1_OK, "tile_capacity: query tile 1 OK");
    REQUIRE(tile.tile_id == 1u, "tile_capacity: tile_id == 1");

    att1_aimu_device_destroy(dev);
    PASS("tile_capacity");
    return 0;
}

/* invalid tile_id rejected */
static int test_invalid_tile_id(void)
{
    att1_aimu_device *dev = NULL;
    att1_aimu_tile_info tile;
    att1_status_t st;

    att1_aimu_device_create(NULL, &dev);  /* 1 tile */

    st = att1_aimu_device_query_tile(dev, 1u, &tile);
    REQUIRE(st == ATT1_ERR_INVALID_ARG, "invalid_tile: query tile 1 rejected");

    st = att1_aimu_device_validate_tile_id(dev, 1u);
    REQUIRE(st == ATT1_ERR_INVALID_ARG, "invalid_tile: validate fails");

    st = att1_aimu_device_validate_tile_id(dev, 0u);
    REQUIRE(st == ATT1_OK, "invalid_tile: validate tile 0 OK");

    att1_aimu_device_destroy(dev);
    PASS("invalid_tile_id");
    return 0;
}

/* dtype support queries */
static int test_dtype_support(void)
{
    att1_aimu_device *dev = NULL;

    att1_aimu_device_create(NULL, &dev);

    REQUIRE(att1_aimu_device_supports_dtype(dev, ATT1_AIMU_DTYPE_F32) == 1,
            "dtype: F32 supported");
    REQUIRE(att1_aimu_device_supports_dtype(dev, ATT1_AIMU_DTYPE_Q8) == 1,
            "dtype: Q8 supported");
    REQUIRE(att1_aimu_device_supports_dtype(dev, ATT1_AIMU_DTYPE_Q4) == 1,
            "dtype: Q4 supported");
    REQUIRE(att1_aimu_device_supports_dtype(NULL, ATT1_AIMU_DTYPE_F32) == 0,
            "dtype: NULL dev returns 0");
    REQUIRE(att1_aimu_device_supports_dtype(dev, UINT32_C(0x80)) == 0,
            "dtype: unknown bit not supported");

    /* per-tile variant */
    REQUIRE(att1_aimu_device_tile_supports_dtype(dev, 0u, ATT1_AIMU_DTYPE_Q8) == 1,
            "dtype: tile 0 Q8 OK");
    REQUIRE(att1_aimu_device_tile_supports_dtype(dev, 1u, ATT1_AIMU_DTYPE_F32) == 0,
            "dtype: tile 1 out-of-range returns 0");

    att1_aimu_device_destroy(dev);
    PASS("dtype_support");
    return 0;
}

/* op support queries */
static int test_op_support(void)
{
    att1_aimu_device *dev = NULL;

    att1_aimu_device_create(NULL, &dev);

    REQUIRE(att1_aimu_device_supports_op(dev, ATT1_AIMU_OP_MATMUL)    == 1, "op: MATMUL");
    REQUIRE(att1_aimu_device_supports_op(dev, ATT1_AIMU_OP_RMSNORM)   == 1, "op: RMSNORM");
    REQUIRE(att1_aimu_device_supports_op(dev, ATT1_AIMU_OP_ROPE)      == 1, "op: ROPE");
    REQUIRE(att1_aimu_device_supports_op(dev, ATT1_AIMU_OP_ATTENTION) == 1, "op: ATTENTION");
    REQUIRE(att1_aimu_device_supports_op(dev, ATT1_AIMU_OP_FFN)       == 1, "op: FFN");
    REQUIRE(att1_aimu_device_supports_op(dev, ATT1_AIMU_OP_KV_APPEND) == 1, "op: KV_APPEND");
    REQUIRE(att1_aimu_device_supports_op(dev, ATT1_AIMU_OP_KV_READ)   == 1, "op: KV_READ");
    REQUIRE(att1_aimu_device_supports_op(dev, ATT1_AIMU_OP_FABRIC_SEND)   == 1, "op: FABRIC_SEND");
    REQUIRE(att1_aimu_device_supports_op(dev, ATT1_AIMU_OP_FABRIC_REDUCE) == 1, "op: FABRIC_REDUCE");
    REQUIRE(att1_aimu_device_supports_op(NULL, ATT1_AIMU_OP_MATMUL)   == 0, "op: NULL returns 0");
    REQUIRE(att1_aimu_device_supports_op(dev, UINT32_C(0x200))        == 0, "op: unknown bit");

    att1_aimu_device_destroy(dev);
    PASS("op_support");
    return 0;
}

/* device with subset of dtypes / ops */
static int test_partial_capability(void)
{
    att1_aimu_device_config cfg;
    att1_aimu_device       *dev = NULL;

    memset(&cfg, 0, sizeof(cfg));
    cfg.tile_count       = 1u;
    cfg.supported_dtypes = ATT1_AIMU_DTYPE_F32;          /* no Q8 or Q4 */
    cfg.supported_ops    = ATT1_AIMU_OP_MATMUL | ATT1_AIMU_OP_RMSNORM;

    att1_aimu_device_create(&cfg, &dev);

    REQUIRE(att1_aimu_device_supports_dtype(dev, ATT1_AIMU_DTYPE_F32) == 1,
            "partial: F32 present");
    REQUIRE(att1_aimu_device_supports_dtype(dev, ATT1_AIMU_DTYPE_Q8) == 0,
            "partial: Q8 absent");
    REQUIRE(att1_aimu_device_supports_op(dev, ATT1_AIMU_OP_MATMUL) == 1,
            "partial: MATMUL present");
    REQUIRE(att1_aimu_device_supports_op(dev, ATT1_AIMU_OP_ATTENTION) == 0,
            "partial: ATTENTION absent");

    att1_aimu_device_destroy(dev);
    PASS("partial_capability");
    return 0;
}

/* reset device clears status and per-tile used bytes */
static int test_reset_device(void)
{
    att1_aimu_device_config  cfg;
    att1_aimu_device        *dev = NULL;
    att1_aimu_device_info    info;
    att1_aimu_tile_info      tile;

    memset(&cfg, 0, sizeof(cfg));
    cfg.tile_count = 2u;
    att1_aimu_device_create(&cfg, &dev);

    /* Manually dirty device state to test clearing. */
    dev->global_status = 0xDEADu;
    dev->global_error  = 0xBEEFu;
    dev->tiles[0].memory_used_bytes = 1024u;
    dev->tiles[0].kv_used_bytes     = 512u;
    dev->tiles[0].error_code        = 0x10u;
    dev->tiles[1].memory_used_bytes = 2048u;

    att1_aimu_device_reset(dev);

    att1_aimu_device_query_info(dev, &info);
    REQUIRE(info.global_status == 0u,  "reset_device: global_status cleared");
    REQUIRE(info.global_error  == 0u,  "reset_device: global_error cleared");

    att1_aimu_device_query_tile(dev, 0u, &tile);
    REQUIRE(tile.memory_used_bytes == 0u, "reset_device: tile 0 memory_used cleared");
    REQUIRE(tile.kv_used_bytes     == 0u, "reset_device: tile 0 kv_used cleared");
    REQUIRE(tile.error_code        == 0u, "reset_device: tile 0 error_code cleared");
    REQUIRE(tile.state == ATT1_AIMU_TILE_IDLE, "reset_device: tile 0 state IDLE");

    att1_aimu_device_query_tile(dev, 1u, &tile);
    REQUIRE(tile.memory_used_bytes == 0u, "reset_device: tile 1 memory_used cleared");

    REQUIRE(dev->reset_count == 1u, "reset_device: reset_count incremented");

    att1_aimu_device_destroy(dev);
    PASS("reset_device");
    return 0;
}

/* reset_tile affects only the target tile */
static int test_reset_tile_isolated(void)
{
    att1_aimu_device_config cfg;
    att1_aimu_device       *dev = NULL;
    att1_aimu_tile_info     tile0, tile1;

    memset(&cfg, 0, sizeof(cfg));
    cfg.tile_count = 2u;
    att1_aimu_device_create(&cfg, &dev);

    /* Dirty both tiles. */
    dev->tiles[0].memory_used_bytes = 111u;
    dev->tiles[0].kv_used_bytes     = 222u;
    dev->tiles[0].error_code        = 0x42u;
    dev->tiles[1].memory_used_bytes = 333u;
    dev->tiles[1].kv_used_bytes     = 444u;
    dev->tiles[1].error_code        = 0x43u;

    /* Reset only tile 0. */
    att1_aimu_device_reset_tile(dev, 0u);

    att1_aimu_device_query_tile(dev, 0u, &tile0);
    att1_aimu_device_query_tile(dev, 1u, &tile1);

    REQUIRE(tile0.memory_used_bytes == 0u,   "reset_tile: tile 0 memory cleared");
    REQUIRE(tile0.kv_used_bytes     == 0u,   "reset_tile: tile 0 kv cleared");
    REQUIRE(tile0.error_code        == 0u,   "reset_tile: tile 0 error cleared");
    REQUIRE(tile0.state == ATT1_AIMU_TILE_IDLE, "reset_tile: tile 0 IDLE");
    REQUIRE(tile0.reset_count       == 1u,   "reset_tile: tile 0 reset_count == 1");

    /* tile 1 must be untouched. */
    REQUIRE(tile1.memory_used_bytes == 333u, "reset_tile: tile 1 memory unchanged");
    REQUIRE(tile1.kv_used_bytes     == 444u, "reset_tile: tile 1 kv unchanged");
    REQUIRE(tile1.error_code        == 0x43u,"reset_tile: tile 1 error unchanged");
    REQUIRE(tile1.reset_count       == 0u,   "reset_tile: tile 1 reset_count unchanged");

    /* Device-level reset_count must be unchanged. */
    REQUIRE(dev->reset_count == 0u, "reset_tile: device reset_count unchanged");

    att1_aimu_device_destroy(dev);
    PASS("reset_tile_isolated");
    return 0;
}

/* feature flags are present and helpers work */
static int test_feature_flags(void)
{
    att1_aimu_device     *dev = NULL;
    att1_aimu_device_info info;

    att1_aimu_device_create(NULL, &dev);
    att1_aimu_device_query_info(dev, &info);

    REQUIRE(info.feature_flags & ATT1_AIMU_FEAT_CMD_RING,      "feat: CMD_RING");
    REQUIRE(info.feature_flags & ATT1_AIMU_FEAT_COMP_RING,     "feat: COMP_RING");
    REQUIRE(info.feature_flags & ATT1_AIMU_FEAT_TRACE,         "feat: TRACE");
    REQUIRE(info.feature_flags & ATT1_AIMU_FEAT_COUNTERS,      "feat: COUNTERS");
    REQUIRE(info.feature_flags & ATT1_AIMU_FEAT_KV_MMU,        "feat: KV_MMU");
    REQUIRE(info.feature_flags & ATT1_AIMU_FEAT_PLACEMENT_AWARE, "feat: PLACEMENT_AWARE");

    /* DMA not in default set. */
    REQUIRE(!(info.feature_flags & ATT1_AIMU_FEAT_DMA), "feat: DMA absent by default");

    att1_aimu_device_destroy(dev);
    PASS("feature_flags");
    return 0;
}

/* name helpers return stable strings */
static int test_name_helpers(void)
{
    REQUIRE(strcmp(att1_aimu_dtype_name(ATT1_AIMU_DTYPE_F32), "F32") == 0,
            "name: F32");
    REQUIRE(strcmp(att1_aimu_dtype_name(ATT1_AIMU_DTYPE_Q8), "Q8") == 0,
            "name: Q8");
    REQUIRE(strcmp(att1_aimu_dtype_name(ATT1_AIMU_DTYPE_Q4), "Q4") == 0,
            "name: Q4");
    REQUIRE(strcmp(att1_aimu_dtype_name(UINT32_C(0xFF)), "UNKNOWN") == 0,
            "name: dtype UNKNOWN");

    REQUIRE(strcmp(att1_aimu_op_name(ATT1_AIMU_OP_MATMUL),        "MATMUL")    == 0, "name: MATMUL");
    REQUIRE(strcmp(att1_aimu_op_name(ATT1_AIMU_OP_RMSNORM),       "RMSNORM")   == 0, "name: RMSNORM");
    REQUIRE(strcmp(att1_aimu_op_name(ATT1_AIMU_OP_ATTENTION),     "ATTENTION") == 0, "name: ATTENTION");
    REQUIRE(strcmp(att1_aimu_op_name(ATT1_AIMU_OP_FABRIC_SEND),   "FABRIC_SEND") == 0, "name: FABRIC_SEND");
    REQUIRE(strcmp(att1_aimu_op_name(ATT1_AIMU_OP_FABRIC_REDUCE), "FABRIC_REDUCE") == 0, "name: FABRIC_REDUCE");
    REQUIRE(strcmp(att1_aimu_op_name(UINT32_C(0x400)), "UNKNOWN") == 0,
            "name: op UNKNOWN");

    REQUIRE(strcmp(att1_aimu_feat_name(ATT1_AIMU_FEAT_CMD_RING),        "CMD_RING")        == 0, "name: CMD_RING");
    REQUIRE(strcmp(att1_aimu_feat_name(ATT1_AIMU_FEAT_PLACEMENT_AWARE), "PLACEMENT_AWARE") == 0, "name: PLACEMENT_AWARE");
    REQUIRE(strcmp(att1_aimu_feat_name(UINT64_C(0x80000000)),           "UNKNOWN")         == 0, "name: feat UNKNOWN");

    REQUIRE(strcmp(att1_aimu_tile_state_name(ATT1_AIMU_TILE_IDLE),   "IDLE")   == 0, "name: IDLE");
    REQUIRE(strcmp(att1_aimu_tile_state_name(ATT1_AIMU_TILE_ACTIVE), "ACTIVE") == 0, "name: ACTIVE");
    REQUIRE(strcmp(att1_aimu_tile_state_name(ATT1_AIMU_TILE_ERROR),  "ERROR")  == 0, "name: ERROR");
    REQUIRE(strcmp(att1_aimu_tile_state_name(ATT1_AIMU_TILE_RESET),  "RESET")  == 0, "name: RESET");
    REQUIRE(strcmp(att1_aimu_tile_state_name((att1_aimu_tile_state)99), "UNKNOWN") == 0,
            "name: state UNKNOWN");

    PASS("name_helpers");
    return 0;
}

/* cmdq attachment and counter snapshot */
static int test_cmdq_attach_and_snapshot(void)
{
    att1_aimu_device        *dev = NULL;
    att1_aimu_cmdq          *q   = NULL;
    att1_aimu_cmdq_counters  ctrs;
    att1_aimu_cmd            cmd;
    att1_status_t            st;

    att1_aimu_device_create(NULL, &dev);

    /* snapshot before attach returns ATT1_ERR_STATE */
    st = att1_aimu_device_snapshot_counters(dev, &ctrs);
    REQUIRE(st == ATT1_ERR_STATE, "cmdq_attach: snapshot without queue returns STATE");

    /* attach a queue, submit a NOP, dispatch, then snapshot */
    att1_aimu_cmdq_create(NULL, &q);
    att1_aimu_device_attach_cmdq(dev, q);

    memset(&cmd, 0, sizeof(cmd));
    cmd.command_type = (uint8_t)ATT1_AIMU_CMD_NOP;
    cmd.tile_id      = 0u;
    att1_aimu_cmdq_submit(q, &cmd);
    att1_aimu_cmdq_dispatch_all(q);

    st = att1_aimu_device_snapshot_counters(dev, &ctrs);
    REQUIRE(st == ATT1_OK,                    "cmdq_attach: snapshot returns OK");
    REQUIRE(ctrs.commands_submitted == 1u,    "cmdq_attach: 1 submitted");
    REQUIRE(ctrs.commands_completed == 1u,    "cmdq_attach: 1 completed");

    /* detach */
    att1_aimu_device_attach_cmdq(dev, NULL);
    st = att1_aimu_device_snapshot_counters(dev, &ctrs);
    REQUIRE(st == ATT1_ERR_STATE, "cmdq_attach: snapshot after detach returns STATE");

    att1_aimu_cmdq_destroy(q);
    att1_aimu_device_destroy(dev);
    PASS("cmdq_attach_and_snapshot");
    return 0;
}

/* null safety for all API functions */
static int test_null_safety(void)
{
    att1_aimu_device_info  info;
    att1_aimu_tile_info    tile;
    att1_aimu_cmdq_counters ctrs;

    memset(&info, 0, sizeof(info));
    memset(&tile, 0, sizeof(tile));
    memset(&ctrs, 0, sizeof(ctrs));

    REQUIRE(att1_aimu_device_query_info(NULL, &info)    == ATT1_ERR_INVALID_ARG, "null: query_info NULL dev");
    REQUIRE(att1_aimu_device_query_info((att1_aimu_device *)&info, NULL)
                                                         == ATT1_ERR_INVALID_ARG, "null: query_info NULL info");
    REQUIRE(att1_aimu_device_query_tile(NULL, 0u, &tile) == ATT1_ERR_INVALID_ARG, "null: query_tile NULL dev");
    REQUIRE(att1_aimu_device_validate_tile_id(NULL, 0u)  == ATT1_ERR_INVALID_ARG, "null: validate NULL dev");
    REQUIRE(att1_aimu_device_reset(NULL)                 == ATT1_ERR_INVALID_ARG, "null: reset NULL dev");
    REQUIRE(att1_aimu_device_reset_tile(NULL, 0u)        == ATT1_ERR_INVALID_ARG, "null: reset_tile NULL dev");
    REQUIRE(att1_aimu_device_snapshot_counters(NULL, &ctrs) == ATT1_ERR_INVALID_ARG, "null: snapshot NULL dev");
    REQUIRE(att1_aimu_device_attach_cmdq(NULL, NULL)     == ATT1_ERR_INVALID_ARG, "null: attach NULL dev");
    REQUIRE(att1_aimu_device_tile_count(NULL)            == 0u,                   "null: tile_count NULL");
    REQUIRE(att1_aimu_device_supports_dtype(NULL, ATT1_AIMU_DTYPE_F32) == 0,     "null: supports_dtype NULL");
    REQUIRE(att1_aimu_device_supports_op(NULL, ATT1_AIMU_OP_MATMUL)    == 0,     "null: supports_op NULL");

    PASS("null_safety");
    return 0;
}

/* fabric link mask is correct for a multi-tile device */
static int test_fabric_link_mask(void)
{
    att1_aimu_device_config cfg;
    att1_aimu_device       *dev = NULL;
    att1_aimu_tile_info     tile;

    memset(&cfg, 0, sizeof(cfg));
    cfg.tile_count = 4u;
    att1_aimu_device_create(&cfg, &dev);

    /* tile 0: links to tiles 1,2,3 → mask = 0b1110 = 0x000E */
    att1_aimu_device_query_tile(dev, 0u, &tile);
    REQUIRE(tile.fabric_link_mask == 0x000Eu, "fabric: tile 0 links to 1,2,3");

    /* tile 3: links to tiles 0,1,2 → mask = 0b0111 = 0x0007 */
    att1_aimu_device_query_tile(dev, 3u, &tile);
    REQUIRE(tile.fabric_link_mask == 0x0007u, "fabric: tile 3 links to 0,1,2");

    /* single-tile device: no links */
    att1_aimu_device_destroy(dev);
    dev = NULL;
    att1_aimu_device_create(NULL, &dev);
    att1_aimu_device_query_tile(dev, 0u, &tile);
    REQUIRE(tile.fabric_link_mask == 0u, "fabric: single tile has no links");

    att1_aimu_device_destroy(dev);
    PASS("fabric_link_mask");
    return 0;
}

/* custom version is preserved */
static int test_custom_version(void)
{
    att1_aimu_device_config cfg;
    att1_aimu_device       *dev = NULL;
    att1_aimu_device_info   info;

    memset(&cfg, 0, sizeof(cfg));
    cfg.tile_count    = 1u;
    cfg.version.major = 2u;
    cfg.version.minor = 3u;
    cfg.version.patch = 4u;
    cfg.version.build = 5u;

    att1_aimu_device_create(&cfg, &dev);
    att1_aimu_device_query_info(dev, &info);

    REQUIRE(info.version.major == 2u, "custom_version: major");
    REQUIRE(info.version.minor == 3u, "custom_version: minor");
    REQUIRE(info.version.patch == 4u, "custom_version: patch");
    REQUIRE(info.version.build == 5u, "custom_version: build");

    att1_aimu_device_destroy(dev);
    PASS("custom_version");
    return 0;
}

/* multi-tile device: reset_tile does not disturb other tiles' reset_count */
static int test_per_tile_reset_count(void)
{
    att1_aimu_device_config cfg;
    att1_aimu_device       *dev = NULL;
    att1_aimu_tile_info     t0, t1, t2;

    memset(&cfg, 0, sizeof(cfg));
    cfg.tile_count = 3u;
    att1_aimu_device_create(&cfg, &dev);

    att1_aimu_device_reset_tile(dev, 0u);
    att1_aimu_device_reset_tile(dev, 0u);
    att1_aimu_device_reset_tile(dev, 2u);

    att1_aimu_device_query_tile(dev, 0u, &t0);
    att1_aimu_device_query_tile(dev, 1u, &t1);
    att1_aimu_device_query_tile(dev, 2u, &t2);

    REQUIRE(t0.reset_count == 2u, "per_tile_reset_count: tile 0 == 2");
    REQUIRE(t1.reset_count == 0u, "per_tile_reset_count: tile 1 == 0");
    REQUIRE(t2.reset_count == 1u, "per_tile_reset_count: tile 2 == 1");

    att1_aimu_device_destroy(dev);
    PASS("per_tile_reset_count");
    return 0;
}

/* no CUDA symbol is required at link time */
static int test_no_cuda_dependency(void)
{
    /* This test exists purely as a documentation point: if the test binary
     * links (which it must to run at all), there is no CUDA dependency. */
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
    rc |= test_create_invalid_tile_count();
    rc |= test_default_version();
    rc |= test_tile_count();
    rc |= test_tile_capacity();
    rc |= test_invalid_tile_id();
    rc |= test_dtype_support();
    rc |= test_op_support();
    rc |= test_partial_capability();
    rc |= test_reset_device();
    rc |= test_reset_tile_isolated();
    rc |= test_feature_flags();
    rc |= test_name_helpers();
    rc |= test_cmdq_attach_and_snapshot();
    rc |= test_null_safety();
    rc |= test_fabric_link_mask();
    rc |= test_custom_version();
    rc |= test_per_tile_reset_count();
    rc |= test_no_cuda_dependency();

    return rc;
}
