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
 *   --tiles N               simulated tile count (default 4).
 *   --tile-memory-mib N      per-tile memory in MiB (default 1024).
 *   --kv-memory-mib N        per-tile KV memory in MiB (default 256).
 *   --once                  exit after the first client disconnects
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

#include "att1_aimu_cmdq.h"
#include "att1_aimu_conformance.h"
#include "att1_aimu_dma.h"
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

/*
 * Cross-process EXEC_* tensor-math execution (real fix for the M167
 * "known limitation": M166's exec hook resolves tensor operands via raw
 * host pointers embedded in the command packet, which are only valid in
 * the process that issued them; forwarding those pointer values verbatim
 * to this daemon's separate address space and dereferencing them here is
 * undefined behavior (it used to reproducibly crash the daemon).
 *
 * Instead, for the command types whose exec hook dereferences
 * `input_buf_addr`/`output_buf_addr`/the resident-tensor registry's
 * `host_addr` (LOAD_TENSOR_TILE, EXEC_MATMUL, EXEC_RMSNORM, EXEC_ROPE,
 * EXEC_FFN), CMD_SUBMIT now carries the *actual* operand bytes in the
 * request's generic `payload` field (already part of the frozen M162 wire
 * protocol, previously unused for this op): this daemon copies those bytes
 * into buffers it owns, rewrites the submitted command's address fields to
 * point at its own memory before handing the command to the wrapped
 * `att1_aimu_conformance_endpoint` (so every later `EXEC_*`/exec-hook
 * dereference is safely within this process), and — since the M158 frozen
 * command packet only carries two address fields and dispatch happens
 * asynchronously relative to submit — remembers the daemon-owned output
 * buffer keyed by `command_id` so `CMD_POLL_COMPLETION` can copy the real
 * result bytes back to the client in its response payload once the
 * command has actually executed.
 *
 * `LOAD_TENSOR_TILE` currently only supports this real transfer for
 * `dtype == ATT1_AIMU_DMA_DTYPE_F32`: the frozen packet only carries a
 * `tensor_id`/`dtype`/packed `dim0`/`dim1` (`op_param_1`), so an f32 weight
 * matrix (`dim0 * dim1` contiguous floats) is exactly reconstructible from
 * those fields, but the q8/q4 weight pointer instead references an
 * `att1_q8_matrix`/`att1_q4_matrix` *struct* with its own nested owned
 * pointers (`values`/`scales` or `packed`/`scales`, plus a `group_size` for
 * q4 that the frozen packet has nowhere to carry) — safely reconstructing
 * that would need wire-protocol changes beyond this fix's scope, so q8/q4
 * `LOAD_TENSOR_TILE` over the socket transport is cleanly rejected with
 * `ATT1_ERR_UNSUPPORTED` instead of dereferencing a foreign-process struct
 * pointer (turning the old crash into a clean, documented error).
 */
#define ATT1_AIMU_ENDPOINT_MAX_PENDING_XFERS 16u

typedef struct att1_aimu_endpoint_pending_xfer {
    int      used;
    uint32_t command_id;
    void    *input_ptr;
    void    *output_ptr;   /* == input_ptr for in-place ops (EXEC_ROPE) */
    uint32_t output_bytes;
    int      free_input;   /* 0 when output_ptr == input_ptr (free once) */
} att1_aimu_endpoint_pending_xfer;

static att1_aimu_endpoint_pending_xfer
    g_pending_xfers[ATT1_AIMU_ENDPOINT_MAX_PENDING_XFERS];

static int cmd_type_needs_buffer_xfer(uint8_t command_type)
{
    switch ((att1_aimu_cmd_type)command_type) {
    case ATT1_AIMU_CMD_EXEC_MATMUL:
    case ATT1_AIMU_CMD_EXEC_RMSNORM:
    case ATT1_AIMU_CMD_EXEC_ROPE:
    case ATT1_AIMU_CMD_EXEC_FFN:
        return 1;
    default:
        return 0;
    }
}

static void pending_xfer_release(att1_aimu_endpoint_pending_xfer *entry)
{
    if (entry->free_input) {
        free(entry->input_ptr);
    }
    free(entry->output_ptr);
    memset(entry, 0, sizeof(*entry));
}

static att1_aimu_endpoint_pending_xfer *pending_xfer_alloc_slot(void)
{
    size_t i;
    for (i = 0u; i < ATT1_AIMU_ENDPOINT_MAX_PENDING_XFERS; i++) {
        if (!g_pending_xfers[i].used) {
            return &g_pending_xfers[i];
        }
    }
    /* Table full (shouldn't happen: one command in flight at a time per
     * the only current caller, src/backend_pcie.c); reclaim the oldest
     * slot rather than leaking the new transfer silently. */
    pending_xfer_release(&g_pending_xfers[0]);
    return &g_pending_xfers[0];
}

static att1_aimu_endpoint_pending_xfer *pending_xfer_find(uint32_t command_id)
{
    size_t i;
    for (i = 0u; i < ATT1_AIMU_ENDPOINT_MAX_PENDING_XFERS; i++) {
        if (g_pending_xfers[i].used && g_pending_xfers[i].command_id == command_id) {
            return &g_pending_xfers[i];
        }
    }
    return NULL;
}

/*
 * dispatch_cmd_submit
 *
 * Handles ATT1_AIMU_ENDPOINT_OP_CMD_SUBMIT: rewrites LOAD_TENSOR_TILE (f32
 * only)/EXEC_MATMUL/EXEC_RMSNORM/EXEC_ROPE/EXEC_FFN operand addresses to
 * daemon-owned copies of the bytes carried in `req->payload` (see file
 * header) before submitting to the wrapped endpoint, so the exec hook
 * installed by src/aimu_conformance.c (M166) only ever dereferences memory
 * that is valid in this process.
 */
static att1_status_t dispatch_cmd_submit(att1_aimu_conformance_endpoint *endpoint,
                                         const att1_aimu_endpoint_request *req,
                                         att1_aimu_endpoint_response *resp)
{
    att1_aimu_cmd cmd = req->cmd;
    uint64_t orig_input_addr = cmd.input_buf_addr;
    uint64_t orig_output_addr = cmd.output_buf_addr;
    void *daemon_input = NULL;
    void *daemon_output = NULL;
    int in_place = 0;
    int needs_xfer = 0;
    att1_status_t st;

    if ((att1_aimu_cmd_type)cmd.command_type == ATT1_AIMU_CMD_LOAD_TENSOR_TILE) {
        uint32_t dim0 = cmd.op_param_1 >> 16;
        uint32_t dim1 = cmd.op_param_1 & 0xFFFFu;
        uint32_t bytes;

        if (cmd.dtype != ATT1_AIMU_DMA_DTYPE_F32) {
            resp->cmd = cmd;
            return ATT1_ERR_UNSUPPORTED;
        }
        bytes = dim0 * dim1 * 4u;
        if ((bytes == 0u) || (bytes != req->payload_bytes) ||
            (bytes > ATT1_AIMU_ENDPOINT_MAX_PAYLOAD)) {
            resp->cmd = cmd;
            return ATT1_ERR_INVALID_ARG;
        }
        daemon_input = malloc(bytes);
        if (daemon_input == NULL) {
            resp->cmd = cmd;
            return ATT1_ERR_OOM;
        }
        memcpy(daemon_input, req->payload, bytes);
        cmd.input_buf_addr = (uint64_t)(uintptr_t)daemon_input;
        /* Resident weight buffer: kept alive for the tensor's lifetime
         * (i.e. this daemon process's lifetime), matching the "no separate
         * device memory backing" residency model already documented for
         * M164/M166; not tracked in the pending-transfer table since there
         * is no result to read back. */
    } else if (cmd_type_needs_buffer_xfer(cmd.command_type)) {
        uint32_t bytes = cmd.input_buf_bytes;

        if ((bytes == 0u) || (bytes != req->payload_bytes) ||
            (bytes > ATT1_AIMU_ENDPOINT_MAX_PAYLOAD)) {
            resp->cmd = cmd;
            return ATT1_ERR_INVALID_ARG;
        }
        daemon_input = malloc(bytes);
        if (daemon_input == NULL) {
            resp->cmd = cmd;
            return ATT1_ERR_OOM;
        }
        memcpy(daemon_input, req->payload, bytes);
        cmd.input_buf_addr = (uint64_t)(uintptr_t)daemon_input;

        if (cmd.output_buf_bytes > 0u) {
            int same_addr = (orig_output_addr == orig_input_addr);
            if (same_addr && (cmd.output_buf_bytes != bytes)) {
                /* Same address but different byte counts can't be a
                 * genuine in-place op (e.g. EXEC_ROPE always reuses the
                 * same buffer at the same size); treat as caller error
                 * rather than silently allocating a second buffer at the
                 * same client-visible address. */
                free(daemon_input);
                resp->cmd = cmd;
                return ATT1_ERR_INVALID_ARG;
            }
            in_place = same_addr;
            if (in_place) {
                daemon_output = daemon_input;
            } else {
                daemon_output = malloc(cmd.output_buf_bytes);
                if (daemon_output == NULL) {
                    free(daemon_input);
                    resp->cmd = cmd;
                    return ATT1_ERR_OOM;
                }
            }
            cmd.output_buf_addr = (uint64_t)(uintptr_t)daemon_output;
            needs_xfer = 1;
        }
    }

    st = att1_aimu_conformance_cmd_submit(endpoint, &cmd);
    if (st == ATT1_OK && needs_xfer) {
        /* EXEC_MATMUL/RMSNORM/ROPE/FFN: remember the daemon-owned output
         * buffer so CMD_POLL_COMPLETION can copy the real result back. */
        att1_aimu_endpoint_pending_xfer *entry = pending_xfer_alloc_slot();
        entry->used = 1;
        entry->command_id = cmd.command_id;
        entry->input_ptr = daemon_input;
        entry->output_ptr = daemon_output;
        entry->output_bytes = cmd.output_buf_bytes;
        entry->free_input = !in_place;
    } else if ((att1_aimu_cmd_type)cmd.command_type == ATT1_AIMU_CMD_LOAD_TENSOR_TILE) {
        /* LOAD_TENSOR_TILE has no output to read back; on success the
         * resident weight buffer is intentionally kept alive (see comment
         * above), only freed here if submission itself failed. */
        if (st != ATT1_OK) {
            free(daemon_input);
        }
    } else if (daemon_input != NULL) {
        /* EXEC_* submit failed before a completion could be scheduled:
         * nothing to keep alive. */
        if ((daemon_output != NULL) && (daemon_output != daemon_input)) {
            free(daemon_output);
        }
        free(daemon_input);
    }

    /* Restore the client's own addresses in the echoed command so the
     * response never leaks a daemon-local pointer value to the client. */
    cmd.input_buf_addr = orig_input_addr;
    cmd.output_buf_addr = orig_output_addr;
    resp->cmd = cmd;
    return st;
}

/*
 * dispatch_cmd_poll_completion
 *
 * Handles ATT1_AIMU_ENDPOINT_OP_CMD_POLL_COMPLETION: after the wrapped
 * endpoint reports a completion, copies any daemon-owned result bytes
 * registered by dispatch_cmd_submit() (keyed by command_id) into the
 * response payload, then releases the daemon-owned buffers.
 */
static att1_status_t dispatch_cmd_poll_completion(att1_aimu_conformance_endpoint *endpoint,
                                                  att1_aimu_endpoint_response *resp)
{
    att1_status_t st = att1_aimu_conformance_cmd_poll_completion(endpoint, &resp->completion);
    if (st == ATT1_OK) {
        att1_aimu_endpoint_pending_xfer *entry =
                pending_xfer_find(resp->completion.command_id);
        if (entry != NULL) {
            if ((entry->output_bytes > 0u) &&
                (entry->output_bytes <= ATT1_AIMU_ENDPOINT_MAX_PAYLOAD)) {
                memcpy(resp->payload, entry->output_ptr, entry->output_bytes);
                resp->payload_bytes = entry->output_bytes;
            }
            pending_xfer_release(entry);
        }
    }
    return st;
}

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
        "  --cmd-ring-depth N     command ring depth, power of two (default 64)\n"
        "  --comp-ring-depth N    completion ring depth, power of two (default 64)\n"
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
    case ATT1_AIMU_ENDPOINT_OP_CMD_SUBMIT:
        st = dispatch_cmd_submit(endpoint, req, resp);
        break;
    case ATT1_AIMU_ENDPOINT_OP_CMD_DISPATCH_ONE:
        st = att1_aimu_conformance_cmd_dispatch_one(endpoint);
        break;
    case ATT1_AIMU_ENDPOINT_OP_CMD_DISPATCH_ALL:
        st = att1_aimu_conformance_cmd_dispatch_all(endpoint);
        break;
    case ATT1_AIMU_ENDPOINT_OP_CMD_POLL_COMPLETION:
        st = dispatch_cmd_poll_completion(endpoint, resp);
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
    case ATT1_AIMU_ENDPOINT_OP_KV_CREATE_SESSION:
        st = att1_aimu_conformance_kv_create_session(endpoint, req->kv_session_id);
        break;
    case ATT1_AIMU_ENDPOINT_OP_KV_DESTROY_SESSION:
        st = att1_aimu_conformance_kv_destroy_session(endpoint, req->kv_session_id);
        break;
    case ATT1_AIMU_ENDPOINT_OP_KV_APPEND: {
        const size_t half_payload = ATT1_AIMU_ENDPOINT_MAX_PAYLOAD / 2u;
        size_t key_count = req->kv_key_bytes / sizeof(float);
        size_t value_count = req->kv_value_bytes / sizeof(float);
        const float *key = req->kv_key_bytes > 0u
                ? (const float *)(const void *)req->payload
                : NULL;
        const float *value = req->kv_value_bytes > 0u
                ? (const float *)(const void *)(req->payload + half_payload)
                : NULL;
        st = att1_aimu_conformance_kv_append(endpoint,
                                             req->kv_session_id,
                                             req->kv_layer_id,
                                             req->kv_position,
                                             key,
                                             key_count,
                                             value,
                                             value_count);
        break;
    }
    case ATT1_AIMU_ENDPOINT_OP_KV_READ: {
        const size_t half_payload = ATT1_AIMU_ENDPOINT_MAX_PAYLOAD / 2u;
        size_t key_count = req->kv_key_bytes / sizeof(float);
        size_t value_count = req->kv_value_bytes / sizeof(float);
        float *out_key = key_count > 0u ? (float *)(void *)resp->payload : NULL;
        float *out_value = value_count > 0u
                ? (float *)(void *)(resp->payload + half_payload)
                : NULL;
        st = att1_aimu_conformance_kv_read(endpoint,
                                           req->kv_session_id,
                                           req->kv_layer_id,
                                           req->kv_head_id,
                                           req->kv_position,
                                           out_key,
                                           key_count,
                                           out_value,
                                           value_count);
        if (st == ATT1_OK) {
            resp->kv_key_bytes = req->kv_key_bytes;
            resp->kv_value_bytes = req->kv_value_bytes;
        }
        break;
    }
    case ATT1_AIMU_ENDPOINT_OP_KV_COPY_RANGE: {
        const size_t half_payload = ATT1_AIMU_ENDPOINT_MAX_PAYLOAD / 2u;
        size_t keys_count = req->kv_key_bytes / sizeof(float);
        size_t values_count = req->kv_value_bytes / sizeof(float);
        float *out_keys = keys_count > 0u ? (float *)(void *)resp->payload : NULL;
        float *out_values = values_count > 0u
                ? (float *)(void *)(resp->payload + half_payload)
                : NULL;
        st = att1_aimu_conformance_kv_copy_range(endpoint,
                                                 req->kv_session_id,
                                                 req->kv_layer_id,
                                                 req->kv_head_id,
                                                 req->kv_start_position,
                                                 req->kv_position_count,
                                                 out_keys,
                                                 keys_count,
                                                 out_values,
                                                 values_count);
        if (st == ATT1_OK) {
            resp->kv_key_bytes = req->kv_key_bytes;
            resp->kv_value_bytes = req->kv_value_bytes;
        }
        break;
    }
    case ATT1_AIMU_ENDPOINT_OP_KV_GET_COUNTERS:
        st = att1_aimu_conformance_kv_get_counters(endpoint, &resp->kv_counters);
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
        } else if (strcmp(argv[i], "--cmd-ring-depth") == 0 && i + 1 < argc) {
            config.cmd_ring_depth = (size_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--comp-ring-depth") == 0 && i + 1 < argc) {
            config.comp_ring_depth = (size_t)strtoul(argv[++i], NULL, 10);
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
