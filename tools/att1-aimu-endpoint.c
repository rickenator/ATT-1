/*
 * att1-aimu-endpoint.c  —  M162 AIMU endpoint process skeleton.
 *
 * A separate process that owns tile memory, the frozen v1.0 register file
 * (M157), the command queue (M158), the DMA simulator (M159), and the
 * fabric bus (M160), and exposes them over a Unix domain socket using the
 * fixed-size protocol in att1_aimu_endpoint_protocol.h. This is the M93
 * §8.8 "out-of-process endpoint" substitution: the in-process conformance
 * simulator (M161) is reused unchanged as the backing implementation, so
 * queue semantics and counters are identical to the same-process path.
 *
 * This tool is a userspace simulator only: no real PCIe transaction layer,
 * no kernel driver, no cross-architecture byte-order handling. Tile memory
 * is metadata-scale only (see att1_aimu_conformance_config).
 *
 * Usage:
 *   att1-aimu-endpoint --socket PATH [options]
 *
 * Options:
 *   --socket PATH          Unix domain socket path to listen on (required).
 *   --tiles N               Simulated tile count (default 4).
 *   --tile-memory-mib N      Per-tile memory in MiB (default 1024).
 *   --kv-memory-mib N        Per-tile KV memory in MiB (default 256).
 *   --once                  Exit after the first client disconnects
 *                            (deterministic for tests/tools).
 *   --verbose               Print each accepted request.
 *   --help                  Show this message.
 *
 * Exit codes:
 *   0  clean shutdown (OP_SHUTDOWN received, or --once client finished)
 *   1  setup or fatal I/O error
 *   2  argument parse error
 */

#define _POSIX_C_SOURCE 200112L

#include "att1_aimu_conformance.h"
#include "att1_aimu_endpoint_protocol.h"
#include "att1_status.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

static void usage(FILE *fp)
{
    fprintf(fp,
        "Usage: att1-aimu-endpoint --socket PATH [options]\n"
        "\n"
        "Options:\n"
        "  --socket PATH          Unix domain socket path to listen on\n"
        "  --tiles N              simulated tile count (default 4)\n"
        "  --tile-memory-mib N    per-tile memory MiB, metadata only (default 1024)\n"
        "  --kv-memory-mib N      per-tile KV memory MiB, metadata only (default 256)\n"
        "  --once                 exit after first client disconnects\n"
        "  --verbose              print each accepted request\n"
        "  --help                 show this message\n"
        "\n"
        "This tool is a userspace AIMU endpoint process skeleton (M162).\n");
}

static att1_status_t dispatch_request(att1_aimu_conformance_endpoint *endpoint,
                                      const att1_aimu_endpoint_request *req,
                                      att1_aimu_endpoint_response *resp,
                                      int *out_shutdown)
{
    att1_status_t st = ATT1_OK;

    memset(resp, 0, sizeof(*resp));
    *out_shutdown = 0;

    switch ((att1_aimu_endpoint_op)req->op) {
    case ATT1_AIMU_ENDPOINT_OP_SYNC_MMIO:
        st = att1_aimu_conformance_sync_mmio(endpoint);
        break;
    case ATT1_AIMU_ENDPOINT_OP_SNAPSHOT_COUNTERS:
        st = att1_aimu_conformance_snapshot_counters(endpoint);
        break;
    case ATT1_AIMU_ENDPOINT_OP_MMIO_READ32:
        st = att1_aimu_conformance_mmio_read32(endpoint, req->offset, &resp->value32);
        break;
    case ATT1_AIMU_ENDPOINT_OP_MMIO_WRITE32:
        st = att1_aimu_conformance_mmio_write32(endpoint, req->offset, req->value32);
        break;
    case ATT1_AIMU_ENDPOINT_OP_MMIO_READ64:
        st = att1_aimu_conformance_mmio_read64(endpoint, req->offset, &resp->value64);
        break;
    case ATT1_AIMU_ENDPOINT_OP_MMIO_WRITE64:
        st = att1_aimu_conformance_mmio_write64(endpoint, req->offset, req->value64);
        break;
    case ATT1_AIMU_ENDPOINT_OP_CMD_SUBMIT: {
        att1_aimu_cmd cmd = req->cmd;
        st = att1_aimu_conformance_cmd_submit(endpoint, &cmd);
        resp->cmd = cmd;
        break;
    }
    case ATT1_AIMU_ENDPOINT_OP_CMD_DISPATCH_ONE:
        st = att1_aimu_conformance_cmd_dispatch_one(endpoint);
        break;
    case ATT1_AIMU_ENDPOINT_OP_CMD_DISPATCH_ALL:
        st = att1_aimu_conformance_cmd_dispatch_all(endpoint);
        break;
    case ATT1_AIMU_ENDPOINT_OP_CMD_POLL_COMPLETION:
        st = att1_aimu_conformance_cmd_poll_completion(endpoint, &resp->completion);
        break;
    case ATT1_AIMU_ENDPOINT_OP_CMD_GET_COUNTERS:
        st = att1_aimu_conformance_cmd_get_counters(endpoint, &resp->cmd_counters);
        break;
    case ATT1_AIMU_ENDPOINT_OP_DMA_VALIDATE:
        st = att1_aimu_conformance_dma_validate(endpoint, &req->dma_desc);
        break;
    case ATT1_AIMU_ENDPOINT_OP_DMA_SUBMIT:
        st = att1_aimu_conformance_dma_submit(endpoint, &req->dma_desc);
        break;
    case ATT1_AIMU_ENDPOINT_OP_DMA_GET_COUNTERS:
        st = att1_aimu_conformance_dma_get_counters(endpoint, &resp->dma_counters);
        break;
    case ATT1_AIMU_ENDPOINT_OP_FABRIC_SEND:
        st = att1_aimu_conformance_fabric_send(endpoint,
                                               req->source_tile,
                                               req->target_tile,
                                               req->packet_type,
                                               req->payload_bytes > 0u ? req->payload : NULL,
                                               req->payload_bytes,
                                               req->value64);
        break;
    case ATT1_AIMU_ENDPOINT_OP_FABRIC_BROADCAST:
        st = att1_aimu_conformance_fabric_broadcast(endpoint,
                                                    req->source_tile,
                                                    req->group_tiles,
                                                    req->group_count,
                                                    req->packet_type,
                                                    req->payload_bytes > 0u ? req->payload : NULL,
                                                    req->payload_bytes,
                                                    req->value64);
        break;
    case ATT1_AIMU_ENDPOINT_OP_FABRIC_RECEIVE: {
        size_t out_bytes = 0;
        uint32_t capacity = req->payload_capacity;
        if (capacity > ATT1_AIMU_ENDPOINT_MAX_PAYLOAD) {
            capacity = ATT1_AIMU_ENDPOINT_MAX_PAYLOAD;
        }
        st = att1_aimu_conformance_fabric_receive(endpoint,
                                                  req->tile_id,
                                                  &resp->fabric_packet,
                                                  resp->payload,
                                                  capacity,
                                                  &out_bytes);
        resp->payload_bytes = (uint32_t)out_bytes;
        break;
    }
    case ATT1_AIMU_ENDPOINT_OP_FABRIC_BARRIER_ARRIVE:
        st = att1_aimu_conformance_fabric_barrier_arrive(endpoint,
                                                         req->tile_id,
                                                         req->participants,
                                                         req->participant_count,
                                                         &resp->out_complete);
        break;
    case ATT1_AIMU_ENDPOINT_OP_FABRIC_GET_COUNTERS:
        st = att1_aimu_conformance_fabric_get_counters(endpoint, &resp->fabric_counters);
        break;
    case ATT1_AIMU_ENDPOINT_OP_TRACE_GET_SNAPSHOT:
        st = att1_aimu_conformance_trace_get_snapshot(endpoint, &resp->trace_snapshot);
        break;
    case ATT1_AIMU_ENDPOINT_OP_SHUTDOWN:
        *out_shutdown = 1;
        st = ATT1_OK;
        break;
    default:
        st = ATT1_ERR_UNSUPPORTED;
        break;
    }

    resp->status = (int32_t)st;
    return st;
}

static int serve_client(att1_aimu_conformance_endpoint *endpoint, int client_fd,
                        int verbose, int *out_shutdown)
{
    for (;;) {
        att1_aimu_endpoint_request req;
        att1_aimu_endpoint_response resp;
        int shutdown_requested = 0;
        att1_status_t io_st = att1_aimu_endpoint_recv_request(client_fd, &req);
        if (io_st != ATT1_OK) {
            /* Client disconnected (or short read); end this session. */
            return 0;
        }
        if (verbose) {
            fprintf(stderr, "att1-aimu-endpoint: op=%u\n", req.op);
        }
        (void)dispatch_request(endpoint, &req, &resp, &shutdown_requested);
        io_st = att1_aimu_endpoint_send_response(client_fd, &resp);
        if (io_st != ATT1_OK) {
            return -1;
        }
        if (shutdown_requested) {
            *out_shutdown = 1;
            return 0;
        }
    }
}

int main(int argc, char **argv)
{
    const char *socket_path = NULL;
    att1_aimu_conformance_config config;
    att1_aimu_conformance_endpoint *endpoint = NULL;
    int listen_fd = -1;
    int run_once = 0;
    int verbose = 0;
    int i;
    struct sockaddr_un addr;
    att1_status_t st;

    att1_aimu_conformance_default_config(&config);

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--socket") == 0 && i + 1 < argc) {
            socket_path = argv[++i];
        } else if (strcmp(argv[i], "--tiles") == 0 && i + 1 < argc) {
            config.tile_count = (size_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--tile-memory-mib") == 0 && i + 1 < argc) {
            config.tile_memory_bytes = (uint64_t)strtoull(argv[++i], NULL, 10) << 20;
        } else if (strcmp(argv[i], "--kv-memory-mib") == 0 && i + 1 < argc) {
            config.tile_kv_bytes = (uint64_t)strtoull(argv[++i], NULL, 10) << 20;
        } else if (strcmp(argv[i], "--once") == 0) {
            run_once = 1;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            usage(stdout);
            return 0;
        } else {
            fprintf(stderr, "att1-aimu-endpoint: unknown or malformed argument '%s'\n", argv[i]);
            usage(stderr);
            return 2;
        }
    }

    if (socket_path == NULL) {
        fprintf(stderr, "att1-aimu-endpoint: --socket PATH is required\n");
        usage(stderr);
        return 2;
    }

    if (strlen(socket_path) >= sizeof(addr.sun_path)) {
        fprintf(stderr, "att1-aimu-endpoint: socket path too long\n");
        return 1;
    }

    st = att1_aimu_conformance_inproc_create(&config, &endpoint);
    if (st != ATT1_OK) {
        fprintf(stderr, "att1-aimu-endpoint: failed to create endpoint (status=%d)\n", (int)st);
        return 1;
    }

    listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        fprintf(stderr, "att1-aimu-endpoint: socket() failed: %s\n", strerror(errno));
        att1_aimu_conformance_endpoint_destroy(endpoint);
        return 1;
    }

    unlink(socket_path);

    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1u);

    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        fprintf(stderr, "att1-aimu-endpoint: bind() failed: %s\n", strerror(errno));
        close(listen_fd);
        att1_aimu_conformance_endpoint_destroy(endpoint);
        return 1;
    }

    if (listen(listen_fd, 1) != 0) {
        fprintf(stderr, "att1-aimu-endpoint: listen() failed: %s\n", strerror(errno));
        close(listen_fd);
        unlink(socket_path);
        att1_aimu_conformance_endpoint_destroy(endpoint);
        return 1;
    }

    if (verbose) {
        fprintf(stderr, "att1-aimu-endpoint: listening on %s (tiles=%zu)\n",
                socket_path, config.tile_count);
    }

    for (;;) {
        int client_fd = accept(listen_fd, NULL, NULL);
        int shutdown_requested = 0;
        int rc;
        if (client_fd < 0) {
            if (errno == EINTR) {
                continue;
            }
            fprintf(stderr, "att1-aimu-endpoint: accept() failed: %s\n", strerror(errno));
            break;
        }
        rc = serve_client(endpoint, client_fd, verbose, &shutdown_requested);
        close(client_fd);
        if (rc != 0 || shutdown_requested || run_once) {
            break;
        }
    }

    close(listen_fd);
    unlink(socket_path);
    att1_aimu_conformance_endpoint_destroy(endpoint);
    return 0;
}
