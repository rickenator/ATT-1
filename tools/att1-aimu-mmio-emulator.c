/*
 * att1-aimu-mmio-emulator.c  —  AIMU userspace MMIO emulator tool (M121)
 *
 * Command-line tool that runs the M121 userspace AIMU MMIO emulator through
 * a deterministic smoke sequence:
 *
 *   probe → read DEVICE_ID → read REGISTER_MAP_VERSION → read TILE_COUNT
 *   → enumerate tile capacities → setup cmdq → submit NOP
 *   → submit LOAD_TENSOR_TILE (descriptor-only, no huge buffer)
 *   → submit VALIDATE_TENSOR → submit QUERY_COUNTERS
 *   → ring doorbell → drain completions → snapshot counters → print summary
 *
 * Options:
 *   --bar0-file PATH        Path for the mmap-backed BAR0 register file.
 *                           Contents reflect the 64 KiB BAR0 window after
 *                           each state-changing operation.
 *   --tiles N               Number of simulated AIMU tiles (1–16; default 4).
 *   --tile-memory-mib N     Per-tile memory in MiB (stored as register
 *                           metadata; default 32; max 256).
 *   --kv-memory-mib N       Per-tile KV memory in MiB (default 8).
 *   --run-smoke             Run the built-in smoke flow (default: on).
 *   --report-json PATH      Write JSON summary to PATH.
 *   --verbose               Print every operation step.
 *
 * This tool is NOT a real PCIe driver, MMIO accessor, or kernel module.
 * Tile memory capacity is stored in registers only; no huge buffers are
 * allocated.  No inference executes; no CUDA kernels run.
 *
 * Exit codes:
 *   0  smoke flow passed (status=pass)
 *   1  smoke flow failed or emulator error
 *   2  argument parse error
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
#include <inttypes.h>

/* =========================================================================
 * Helpers
 * ====================================================================== */

static void usage(FILE *fp)
{
    fprintf(fp,
        "Usage: att1-aimu-mmio-emulator [options]\n"
        "\n"
        "Options:\n"
        "  --bar0-file PATH       mmap-backed BAR0 register file path\n"
        "  --tiles N              simulated tile count (1-16; default 4)\n"
        "  --tile-memory-mib N    per-tile memory MiB, metadata only (default 32)\n"
        "  --kv-memory-mib N      per-tile KV memory MiB, metadata only (default 8)\n"
        "  --run-smoke            run built-in smoke flow (default)\n"
        "  --report-json PATH     write JSON summary to PATH\n"
        "  --verbose              print each operation step\n"
        "  --help                 show this message\n"
        "\n"
        "This tool is a userspace AIMU MMIO emulator (M121).\n"
        "Tile memory capacity is metadata only; no huge buffers are allocated.\n");
}

static void write_json_report(const char             *path,
                               att1_aimu_host_summary *sum,
                               const char             *status)
{
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "warning: cannot write JSON report to %s\n", path);
        return;
    }
    fprintf(fp, "{\n");
    fprintf(fp, "  \"emulator\": \"att1-aimu-mmio-emulator\",\n");
    fprintf(fp, "  \"milestone\": \"M121\",\n");
    fprintf(fp, "  \"device_id\": \"0x%08X\",\n",          sum->device_id);
    fprintf(fp, "  \"register_map_version\": \"0x%08X\",\n", sum->register_map_version);
    fprintf(fp, "  \"tile_count\": %zu,\n",                sum->tile_count);
    fprintf(fp, "  \"commands_submitted\": %" PRIu64 ",\n",  sum->commands_submitted);
    fprintf(fp, "  \"commands_completed\": %" PRIu64 ",\n",  sum->commands_completed);
    fprintf(fp, "  \"commands_failed\": %" PRIu64 ",\n",     sum->commands_failed);
    fprintf(fp, "  \"dma_submitted\": %" PRIu64 ",\n",       sum->dma_submitted);
    fprintf(fp, "  \"dma_completed\": %" PRIu64 ",\n",       sum->dma_completed);
    fprintf(fp, "  \"doorbell_count\": %u,\n",             sum->doorbell_count);
    fprintf(fp, "  \"fence_value\": %" PRIu64 ",\n",           sum->fence_value);
    fprintf(fp, "  \"status\": \"%s\"\n", status);
    fprintf(fp, "}\n");
    fclose(fp);
}

/* =========================================================================
 * Smoke flow
 * ====================================================================== */

static int run_smoke(att1_aimu_userspace *u, int verbose)
{
    att1_status_t rc;

    /* ---- Step 1: probe device ---- */
    att1_aimu_host_probe_result probe;
    rc = att1_aimu_userspace_probe(u, &probe);
    if (rc != ATT1_OK) {
        fprintf(stderr, "smoke: probe failed rc=%d\n", (int)rc);
        return 1;
    }
    if (verbose)
        printf("step=probe device_id=0x%08X tile_count=%zu\n",
               probe.device_id, probe.tile_count);

    /* ---- Step 2: read DEVICE_ID ---- */
    uint32_t did = 0;
    rc = att1_aimu_userspace_read32(u, ATT1_MMIO_DEVICE_ID, &did);
    if (rc != ATT1_OK) { fprintf(stderr, "smoke: read DEVICE_ID failed\n"); return 1; }
    if (verbose) printf("step=read_device_id value=0x%08X\n", did);
    if (did != ATT1_MMIO_DEVICE_ID_DEFAULT) {
        fprintf(stderr, "smoke: DEVICE_ID mismatch: got 0x%08X\n", did);
        return 1;
    }

    /* ---- Step 3: read REGISTER_MAP_VERSION ---- */
    uint32_t rmv = 0;
    rc = att1_aimu_userspace_read32(u, ATT1_MMIO_REGISTER_MAP_VERSION, &rmv);
    if (rc != ATT1_OK) { fprintf(stderr, "smoke: read REGISTER_MAP_VERSION failed\n"); return 1; }
    if (verbose) printf("step=read_register_map_version value=0x%08X\n", rmv);

    /* ---- Step 4: read TILE_COUNT ---- */
    uint32_t tc = 0;
    rc = att1_aimu_userspace_read32(u, ATT1_MMIO_TILE_COUNT, &tc);
    if (rc != ATT1_OK) { fprintf(stderr, "smoke: read TILE_COUNT failed\n"); return 1; }
    if (verbose) printf("step=read_tile_count value=%u\n", tc);

    /* ---- Step 5: enumerate tile capacities ---- */
    att1_aimu_host_tile_info tiles[16];
    size_t ntiles = 16;
    rc = att1_aimu_userspace_enumerate_tiles(u, tiles, &ntiles);
    if (rc != ATT1_OK) { fprintf(stderr, "smoke: enumerate_tiles failed\n"); return 1; }
    if (verbose) {
        for (size_t i = 0; i < ntiles; i++) {
            printf("step=tile tile_id=%u mem_bytes=%" PRIu64 " kv_bytes=%" PRIu64 "\n",
                   tiles[i].tile_id,
                   tiles[i].memory_capacity_bytes,
                   tiles[i].kv_capacity_bytes);
        }
    }

    /* ---- Step 6: setup command queue ---- */
    rc = att1_aimu_userspace_setup_cmdq(u);
    if (rc != ATT1_OK) { fprintf(stderr, "smoke: setup_cmdq failed rc=%d\n", (int)rc); return 1; }
    if (verbose) printf("step=setup_cmdq\n");

    uint32_t cmd_id = 1u;

    /* ---- Step 7: submit NOP ---- */
    {
        att1_aimu_cmd cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.command_id   = cmd_id++;
        cmd.command_type = (uint8_t)ATT1_AIMU_CMD_NOP;
        cmd.tile_id      = 0;
        rc = att1_aimu_userspace_submit_cmd(u, &cmd);
        if (rc != ATT1_OK) { fprintf(stderr, "smoke: submit NOP failed\n"); return 1; }
        if (verbose) printf("step=submit_nop cmd_id=%u\n", cmd.command_id);
    }

    /* ---- Step 8: submit LOAD_TENSOR_TILE with descriptor-only DMA ---- */
    {
        /* Validate a tiny descriptor (64-byte aligned addresses, small payload).
         * No actual large buffer is allocated — descriptor-only simulation. */
        att1_aimu_dma_desc desc;
        memset(&desc, 0, sizeof(desc));
        desc.host_addr      = 64u;   /* 64-byte aligned; simulated host address */
        desc.device_addr    = 64u;   /* 64-byte aligned; simulated device address */
        desc.byte_length    = 64u;   /* minimal payload */
        desc.dtype          = 0u;    /* F32 */
        desc.direction      = 0u;    /* host-to-device */
        desc.descriptor_id  = 1u;
        desc.command_id     = cmd_id;
        desc.tensor_id      = 1u;
        rc = att1_aimu_userspace_validate_dma(u, &desc);
        if (rc != ATT1_OK) {
            fprintf(stderr, "smoke: DMA descriptor validation failed rc=%d\n", (int)rc);
            return 1;
        }

        att1_aimu_cmd cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.command_id      = cmd_id++;
        cmd.command_type    = (uint8_t)ATT1_AIMU_CMD_LOAD_TENSOR_TILE;
        cmd.tile_id         = 0;
        cmd.tensor_id       = 1u;
        cmd.input_buf_addr  = desc.host_addr;
        cmd.input_buf_bytes = desc.byte_length;
        rc = att1_aimu_userspace_submit_cmd(u, &cmd);
        if (rc != ATT1_OK) { fprintf(stderr, "smoke: submit LOAD_TENSOR_TILE failed\n"); return 1; }
        if (verbose) printf("step=submit_load_tensor cmd_id=%u\n", cmd.command_id);
    }

    /* ---- Step 9: submit VALIDATE_TENSOR ---- */
    {
        att1_aimu_cmd cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.command_id   = cmd_id++;
        cmd.command_type = (uint8_t)ATT1_AIMU_CMD_VALIDATE_TENSOR;
        cmd.tile_id      = 0;
        cmd.tensor_id    = 1u;
        rc = att1_aimu_userspace_submit_cmd(u, &cmd);
        if (rc != ATT1_OK) { fprintf(stderr, "smoke: submit VALIDATE_TENSOR failed\n"); return 1; }
        if (verbose) printf("step=submit_validate_tensor cmd_id=%u\n", cmd.command_id);
    }

    /* ---- Step 10: submit QUERY_COUNTERS ---- */
    {
        att1_aimu_cmd cmd;
        memset(&cmd, 0, sizeof(cmd));
        cmd.command_id   = cmd_id++;
        cmd.command_type = (uint8_t)ATT1_AIMU_CMD_QUERY_COUNTERS;
        cmd.tile_id      = 0;
        rc = att1_aimu_userspace_submit_cmd(u, &cmd);
        if (rc != ATT1_OK) { fprintf(stderr, "smoke: submit QUERY_COUNTERS failed\n"); return 1; }
        if (verbose) printf("step=submit_query_counters cmd_id=%u\n", cmd.command_id);
    }

    /* ---- Step 11: ring doorbell ---- */
    rc = att1_aimu_userspace_ring_doorbell(u);
    if (rc != ATT1_OK) { fprintf(stderr, "smoke: ring_doorbell failed\n"); return 1; }
    if (verbose) printf("step=ring_doorbell\n");

    /* ---- Step 12: drain completions ---- */
    rc = att1_aimu_userspace_drain(u);
    if (rc != ATT1_OK) { fprintf(stderr, "smoke: drain failed\n"); return 1; }
    if (verbose) printf("step=drain\n");

    /* ---- Step 13: snapshot counters ---- */
    rc = att1_aimu_userspace_snapshot(u);
    if (rc != ATT1_OK) { fprintf(stderr, "smoke: snapshot failed\n"); return 1; }
    if (verbose) printf("step=snapshot\n");

    return 0;
}

/* =========================================================================
 * main
 * ====================================================================== */

int main(int argc, char **argv)
{
    const char *bar0_path    = NULL;
    const char *report_json  = NULL;
    size_t      tiles        = 0;       /* 0 → default */
    uint64_t    tile_mem_mib = 0;
    uint64_t    kv_mem_mib   = 0;
    int         verbose      = 0;
    int         do_smoke     = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bar0-file") == 0) {
            if (i + 1 >= argc) { fprintf(stderr, "--bar0-file requires PATH\n"); return 2; }
            bar0_path = argv[++i];
        } else if (strcmp(argv[i], "--tiles") == 0) {
            if (i + 1 >= argc) { fprintf(stderr, "--tiles requires N\n"); return 2; }
            long v = strtol(argv[++i], NULL, 10);
            if (v <= 0 || v > 16) {
                fprintf(stderr, "--tiles must be 1-16, got %s\n", argv[i]);
                return 2;
            }
            tiles = (size_t)v;
        } else if (strcmp(argv[i], "--tile-memory-mib") == 0) {
            if (i + 1 >= argc) { fprintf(stderr, "--tile-memory-mib requires N\n"); return 2; }
            long v = strtol(argv[++i], NULL, 10);
            if (v <= 0) {
                fprintf(stderr, "--tile-memory-mib must be positive, got %s\n", argv[i]);
                return 2;
            }
            tile_mem_mib = (uint64_t)v;
        } else if (strcmp(argv[i], "--kv-memory-mib") == 0) {
            if (i + 1 >= argc) { fprintf(stderr, "--kv-memory-mib requires N\n"); return 2; }
            long v = strtol(argv[++i], NULL, 10);
            if (v <= 0) {
                fprintf(stderr, "--kv-memory-mib must be positive, got %s\n", argv[i]);
                return 2;
            }
            kv_mem_mib = (uint64_t)v;
        } else if (strcmp(argv[i], "--run-smoke") == 0) {
            do_smoke = 1;
        } else if (strcmp(argv[i], "--report-json") == 0) {
            if (i + 1 >= argc) { fprintf(stderr, "--report-json requires PATH\n"); return 2; }
            report_json = argv[++i];
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        } else if (strcmp(argv[i], "--help") == 0 ||
                   strcmp(argv[i], "-h") == 0) {
            usage(stdout);
            return 0;
        } else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            usage(stderr);
            return 2;
        }
    }

    /* Build config */
    att1_aimu_userspace_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.tile_count        = tiles;
    cfg.tile_memory_bytes = tile_mem_mib
                            ? tile_mem_mib * 1024u * 1024u
                            : 0u;  /* 0 → default in open() */
    cfg.tile_kv_bytes     = kv_mem_mib
                            ? kv_mem_mib * 1024u * 1024u
                            : 0u;

    /* Validate tile_memory_mib against max */
    if (tile_mem_mib > 256u) {
        fprintf(stderr, "--tile-memory-mib max is 256, got %" PRIu64 "\n", tile_mem_mib);
        return 2;
    }

    att1_aimu_userspace *u = NULL;
    att1_status_t rc = att1_aimu_userspace_open(&cfg, bar0_path, &u);
    if (rc != ATT1_OK) {
        fprintf(stderr, "att1_aimu_userspace_open failed rc=%d\n", (int)rc);
        return 1;
    }

    int result = 0;
    if (do_smoke) {
        result = run_smoke(u, verbose);
    }

    att1_aimu_host_summary sum;
    memset(&sum, 0, sizeof(sum));
    (void)att1_aimu_userspace_get_summary(u, &sum);

    const char *status_str = "pass";
    if (result != 0 || sum.commands_failed > 0u) status_str = "fail";

    /* Print summary */
    printf("\n--- AIMU userspace emulator summary ---\n");
    (void)att1_aimu_userspace_print_summary(u, stdout);

    if (report_json) {
        write_json_report(report_json, &sum, status_str);
        if (verbose) printf("report_json=%s\n", report_json);
    }

    att1_aimu_userspace_close(u);
    return (result != 0) ? 1 : 0;
}
