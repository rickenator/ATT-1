/*
 * test_aimu_endpoint_fault_injection.c  —  M168 fault injection and
 * hostile-endpoint testing.
 *
 * Extends the Phase 1 hostile-input tradition (M93 §8.7-6) to the M162
 * out-of-process transport boundary: queue-full, out-of-order/duplicate KV
 * appends, DMA descriptor rejection, endpoint crash mid-decode, and a
 * hostile mock daemon sending deliberately malformed completion/response
 * messages must all map to att1_status_t without undefined behavior on the
 * client side (tests/test_aimu_endpoint.c already covers the non-hostile
 * transport lifecycle; this suite is dedicated to hostile/faulty cases).
 */

#define _POSIX_C_SOURCE 200112L

#include "att1_aimu_conformance.h"
#include "att1_aimu_endpoint_client.h"
#include "att1_aimu_endpoint_protocol.h"

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define PASS(name) do { printf("PASS: aimu_endpoint_fault_injection: %s\n", (name)); } while (0)
#define FAIL(name) do { printf("FAIL: aimu_endpoint_fault_injection: %s\n", (name)); return 1; } while (0)
#define REQUIRE(cond, name) do { if (!(cond)) { FAIL(name); } } while (0)

static const char *g_socket_path = "build/test_aimu_endpoint_fault_injection.sock";

static pid_t spawn_daemon_argv(char *const argv[])
{
    pid_t pid = fork();
    if (pid < 0) {
        return -1;
    }
    if (pid == 0) {
        execv("./build/att1-aimu-endpoint", argv);
        _exit(127);
    }
    return pid;
}

static int connect_with_retry(const char *socket_path, att1_aimu_conformance_endpoint **out)
{
    int attempt;
    for (attempt = 0; attempt < 200; ++attempt) {
        if (att1_aimu_conformance_socket_connect(socket_path, out) == ATT1_OK) {
            return 0;
        }
        struct timespec ts = {0, 10 * 1000 * 1000L}; /* 10ms */
        nanosleep(&ts, NULL);
    }
    return -1;
}

static void make_cmd(att1_aimu_cmd *cmd, att1_aimu_cmd_type type, uint8_t tile_id)
{
    memset(cmd, 0, sizeof(*cmd));
    cmd->command_type = (uint8_t)type;
    cmd->tile_id = tile_id;
}

/*
 * Queue-full over the socket transport: spawn a daemon with a small,
 * explicit command-ring depth (via the M168 --cmd-ring-depth/
 * --comp-ring-depth CLI options) so the ring fills after a handful of
 * submits, then confirm ATT1_ERR_QUEUE_FULL is returned over the wire
 * (not a crash/hang) and the daemon remains usable afterward.
 */
static int test_queue_full_over_socket(void)
{
    att1_aimu_conformance_endpoint *endpoint = NULL;
    pid_t pid;
    int status = 0;
    att1_aimu_cmd cmd;
    att1_aimu_completion completion;
    char *argv[] = {
        (char *)"att1-aimu-endpoint", (char *)"--socket", (char *)g_socket_path,
        (char *)"--cmd-ring-depth", (char *)"4", (char *)"--comp-ring-depth", (char *)"4",
        (char *)"--once", NULL
    };

    unlink(g_socket_path);
    pid = spawn_daemon_argv(argv);
    REQUIRE(pid > 0, "queue_full: daemon spawn");
    REQUIRE(connect_with_retry(g_socket_path, &endpoint) == 0,
            "queue_full: client connect");

    make_cmd(&cmd, ATT1_AIMU_CMD_NOP, 0u);
    REQUIRE(att1_aimu_conformance_cmd_submit(endpoint, &cmd) == ATT1_OK,
            "queue_full: submit 1 over socket");
    REQUIRE(att1_aimu_conformance_cmd_submit(endpoint, &cmd) == ATT1_OK,
            "queue_full: submit 2 over socket");
    REQUIRE(att1_aimu_conformance_cmd_submit(endpoint, &cmd) == ATT1_OK,
            "queue_full: submit 3 over socket");
    REQUIRE(att1_aimu_conformance_cmd_submit(endpoint, &cmd) == ATT1_ERR_QUEUE_FULL,
            "queue_full: 4th submit rejected over socket");

    /* Daemon must still be alive and functional after the hostile submit. */
    REQUIRE(att1_aimu_conformance_cmd_dispatch_all(endpoint) == ATT1_OK,
            "queue_full: dispatch_all still works after queue-full rejection");
    memset(&completion, 0, sizeof(completion));
    REQUIRE(att1_aimu_conformance_cmd_poll_completion(endpoint, &completion) == ATT1_OK,
            "queue_full: poll_completion still works after queue-full rejection");

    att1_aimu_conformance_endpoint_destroy(endpoint);
    waitpid(pid, &status, 0);
    unlink(g_socket_path);
    PASS("queue_full_over_socket");
    return 0;
}

/*
 * Out-of-order and duplicate KV appends over the socket transport: M151's
 * att1_kv_mmu_append() (wrapped here through the M165 endpoint op) rejects
 * appends that skip ahead of the session's next expected position and
 * rejects re-appending an already-populated position; confirm both map to
 * a clean att1_status_t over the wire and the session stays usable.
 */
static int test_out_of_order_kv_append_over_socket(void)
{
    att1_aimu_conformance_endpoint *endpoint = NULL;
    pid_t pid;
    int status = 0;
    const uint64_t session_id = 42u;
    const size_t layer_id = 0u;
    const size_t num_heads = 4u;
    const size_t head_dim = 16u;
    const size_t per_position = num_heads * head_dim;
    float key[64];
    float value[64];
    size_t i;

    for (i = 0u; i < per_position; ++i) {
        key[i] = (float)(1u + i);
        value[i] = (float)(2u + i);
    }

    unlink(g_socket_path);
    pid = spawn_daemon_argv((char *const[]){
        (char *)"att1-aimu-endpoint", (char *)"--socket", (char *)g_socket_path,
        (char *)"--once", NULL
    });
    REQUIRE(pid > 0, "kv_ooo: daemon spawn");
    REQUIRE(connect_with_retry(g_socket_path, &endpoint) == 0,
            "kv_ooo: client connect");

    REQUIRE(att1_aimu_conformance_kv_create_session(endpoint, session_id) == ATT1_OK,
            "kv_ooo: create session over socket");

    /* Skip position 0: appending directly at position 5 must be rejected. */
    REQUIRE(att1_aimu_conformance_kv_append(endpoint,
                                            session_id,
                                            layer_id,
                                            5u,
                                            key,
                                            per_position,
                                            value,
                                            per_position) != ATT1_OK,
            "kv_ooo: out-of-order append rejected over socket");

    /* Session must still accept the correct next append (position 0). */
    REQUIRE(att1_aimu_conformance_kv_append(endpoint,
                                            session_id,
                                            layer_id,
                                            0u,
                                            key,
                                            per_position,
                                            value,
                                            per_position) == ATT1_OK,
            "kv_ooo: in-order append still succeeds after rejected out-of-order append");

    /* Duplicate append of the now-populated position 0 must be rejected. */
    REQUIRE(att1_aimu_conformance_kv_append(endpoint,
                                            session_id,
                                            layer_id,
                                            0u,
                                            key,
                                            per_position,
                                            value,
                                            per_position) != ATT1_OK,
            "kv_ooo: duplicate append rejected over socket");

    REQUIRE(att1_aimu_conformance_kv_destroy_session(endpoint, session_id) == ATT1_OK,
            "kv_ooo: destroy session over socket");

    att1_aimu_conformance_endpoint_destroy(endpoint);
    waitpid(pid, &status, 0);
    unlink(g_socket_path);
    PASS("out_of_order_kv_append_over_socket");
    return 0;
}

/*
 * DMA descriptor rejection over the socket transport: malformed descriptors
 * (unaligned host address, unsupported flags, zero-length transfer) must be
 * rejected with ATT1_ERR_INVALID_ARG over the wire, and the daemon must
 * remain usable for a subsequent valid transfer.
 */
static int test_dma_descriptor_rejection_over_socket(void)
{
    att1_aimu_conformance_endpoint *endpoint = NULL;
    pid_t pid;
    int status = 0;
    att1_aimu_conformance_config cfg;
    att1_aimu_dma_desc desc;

    unlink(g_socket_path);
    pid = spawn_daemon_argv((char *const[]){
        (char *)"att1-aimu-endpoint", (char *)"--socket", (char *)g_socket_path,
        (char *)"--once", NULL
    });
    REQUIRE(pid > 0, "dma_reject: daemon spawn");
    REQUIRE(connect_with_retry(g_socket_path, &endpoint) == 0,
            "dma_reject: client connect");

    att1_aimu_conformance_default_config(&cfg);

    memset(&desc, 0, sizeof(desc));
    desc.host_addr = cfg.dma_host_base + 1u; /* unaligned */
    desc.device_addr = cfg.dma_device_base;
    desc.byte_length = 4096u;
    desc.direction = (uint8_t)ATT1_AIMU_DMA_HOST_TO_DEVICE;
    desc.dtype = ATT1_AIMU_DMA_DTYPE_F32;
    REQUIRE(att1_aimu_conformance_dma_validate(endpoint, &desc) == ATT1_ERR_INVALID_ARG,
            "dma_reject: unaligned host address rejected over socket");
    REQUIRE(att1_aimu_conformance_dma_submit(endpoint, &desc) == ATT1_ERR_INVALID_ARG,
            "dma_reject: unaligned host address rejected by submit over socket");

    memset(&desc, 0, sizeof(desc));
    desc.host_addr = cfg.dma_host_base;
    desc.device_addr = cfg.dma_device_base;
    desc.byte_length = 4096u;
    desc.direction = (uint8_t)ATT1_AIMU_DMA_HOST_TO_DEVICE;
    desc.dtype = ATT1_AIMU_DMA_DTYPE_F32;
    desc.flags = UINT16_C(0x8000); /* unsupported */
    REQUIRE(att1_aimu_conformance_dma_validate(endpoint, &desc) == ATT1_ERR_INVALID_ARG,
            "dma_reject: unsupported flags rejected over socket");

    memset(&desc, 0, sizeof(desc));
    desc.host_addr = cfg.dma_host_base;
    desc.device_addr = cfg.dma_device_base;
    desc.byte_length = 0u; /* zero-length */
    desc.direction = (uint8_t)ATT1_AIMU_DMA_HOST_TO_DEVICE;
    desc.dtype = ATT1_AIMU_DMA_DTYPE_F32;
    REQUIRE(att1_aimu_conformance_dma_validate(endpoint, &desc) == ATT1_ERR_INVALID_ARG,
            "dma_reject: zero-length transfer rejected over socket");

    /* Daemon must still accept a valid transfer after the hostile input. */
    memset(&desc, 0, sizeof(desc));
    desc.host_addr = cfg.dma_host_base;
    desc.device_addr = cfg.dma_device_base;
    desc.byte_length = 4096u;
    desc.direction = (uint8_t)ATT1_AIMU_DMA_HOST_TO_DEVICE;
    desc.dtype = ATT1_AIMU_DMA_DTYPE_F32;
    REQUIRE(att1_aimu_conformance_dma_validate(endpoint, &desc) == ATT1_OK,
            "dma_reject: valid descriptor still accepted after rejections");
    REQUIRE(att1_aimu_conformance_dma_submit(endpoint, &desc) == ATT1_OK,
            "dma_reject: valid descriptor still submits after rejections");

    att1_aimu_conformance_endpoint_destroy(endpoint);
    waitpid(pid, &status, 0);
    unlink(g_socket_path);
    PASS("dma_descriptor_rejection_over_socket");
    return 0;
}

/*
 * Endpoint crash mid-decode: kill the daemon process with SIGKILL after a
 * command has been submitted but before it is dispatched/polled, mimicking
 * a hard crash mid-decode. Every subsequent client call must map cleanly to
 * ATT1_ERR_IO (short read/connection loss in att1_aimu_endpoint_protocol.c)
 * instead of hanging or crashing the client process.
 */
static int test_endpoint_crash_mid_decode(void)
{
    att1_aimu_conformance_endpoint *endpoint = NULL;
    pid_t pid;
    int status = 0;
    att1_aimu_cmd cmd;
    att1_aimu_completion completion;

    unlink(g_socket_path);
    pid = spawn_daemon_argv((char *const[]){
        (char *)"att1-aimu-endpoint", (char *)"--socket", (char *)g_socket_path,
        NULL
    });
    REQUIRE(pid > 0, "crash_mid_decode: daemon spawn");
    REQUIRE(connect_with_retry(g_socket_path, &endpoint) == 0,
            "crash_mid_decode: client connect");

    make_cmd(&cmd, ATT1_AIMU_CMD_NOP, 0u);
    REQUIRE(att1_aimu_conformance_cmd_submit(endpoint, &cmd) == ATT1_OK,
            "crash_mid_decode: submit before crash succeeds");

    REQUIRE(kill(pid, SIGKILL) == 0, "crash_mid_decode: kill daemon");
    waitpid(pid, &status, 0);

    /* The daemon process is now gone; any further request must fail with
     * ATT1_ERR_IO, never hang or crash the client. */
    memset(&completion, 0, sizeof(completion));
    REQUIRE(att1_aimu_conformance_cmd_dispatch_all(endpoint) == ATT1_ERR_IO,
            "crash_mid_decode: dispatch_all after crash maps to ATT1_ERR_IO");
    REQUIRE(att1_aimu_conformance_cmd_poll_completion(endpoint, &completion) == ATT1_ERR_IO,
            "crash_mid_decode: poll_completion after crash maps to ATT1_ERR_IO");
    REQUIRE(att1_aimu_conformance_sync_mmio(endpoint) == ATT1_ERR_IO,
            "crash_mid_decode: sync_mmio after crash maps to ATT1_ERR_IO");

    /* Destroying the endpoint (which sends a best-effort SHUTDOWN over the
     * now-dead socket) must not crash either. */
    att1_aimu_conformance_endpoint_destroy(endpoint);
    unlink(g_socket_path);
    PASS("endpoint_crash_mid_decode");
    return 0;
}

/*
 * Hostile mock daemon: a minimal listener that speaks just enough of the
 * wire protocol to receive one request and then reply with deliberately
 * malformed field values (oversized kv_key_bytes/kv_value_bytes/
 * payload_bytes claims) that a buggy or compromised real daemon might send.
 * Proves the client's fixed-size struct handling clamps these values
 * instead of reading past its own local response buffer (M168
 * hostile-endpoint hardening of src/aimu_endpoint_client.c).
 */
static pid_t spawn_hostile_mock_daemon(const char *socket_path)
{
    pid_t pid;
    int listen_fd;
    struct sockaddr_un addr;

    unlink(socket_path);
    listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        return -1;
    }
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1u);
    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) != 0 ||
        listen(listen_fd, 1) != 0) {
        close(listen_fd);
        return -1;
    }

    pid = fork();
    if (pid < 0) {
        close(listen_fd);
        return -1;
    }
    if (pid == 0) {
        int client_fd = accept(listen_fd, NULL, NULL);
        att1_aimu_endpoint_request req;
        att1_aimu_endpoint_response resp;

        if (client_fd < 0) {
            _exit(1);
        }
        for (;;) {
            if (att1_aimu_endpoint_recv_request(client_fd, &req) != ATT1_OK) {
                break;
            }
            memset(&resp, 0, sizeof(resp));
            resp.status = ATT1_OK;
            /* Hostile: claim far more bytes than the fixed payload array
             * / caller buffers can legitimately hold, on every op. */
            resp.kv_key_bytes = 0xFFFFFFFFu;
            resp.kv_value_bytes = 0xFFFFFFFFu;
            resp.payload_bytes = 0xFFFFFFFFu;
            memset(resp.payload, 0x5A, sizeof(resp.payload));
            if (att1_aimu_endpoint_send_response(client_fd, &resp) != ATT1_OK) {
                break;
            }
        }
        close(client_fd);
        close(listen_fd);
        _exit(0);
    }
    close(listen_fd);
    return pid;
}

static int test_malformed_completion_from_hostile_daemon(void)
{
    att1_aimu_conformance_endpoint *endpoint = NULL;
    pid_t pid;
    int status = 0;
    const uint64_t session_id = 1u;
    const size_t layer_id = 0u;
    const size_t head_id = 0u;
    float out_key[16];
    float out_value[16];
    att1_fabric_packet packet;
    unsigned char out_payload[16];
    size_t out_bytes = 0u;

    unlink(g_socket_path);
    pid = spawn_hostile_mock_daemon(g_socket_path);
    REQUIRE(pid > 0, "malformed: hostile daemon spawn");
    REQUIRE(connect_with_retry(g_socket_path, &endpoint) == 0,
            "malformed: client connect");

    memset(out_key, 0, sizeof(out_key));
    memset(out_value, 0, sizeof(out_value));
    /* Must not crash (buffer overflow/OOB read) despite the hostile
     * kv_key_bytes/kv_value_bytes claims; status is ATT1_OK per the mock's
     * (also hostile) resp.status, but the copied bytes must be clamped to
     * the caller-provided buffer sizes. */
    REQUIRE(att1_aimu_conformance_kv_read(endpoint,
                                          session_id,
                                          layer_id,
                                          head_id,
                                          0u,
                                          out_key,
                                          16u,
                                          out_value,
                                          16u) == ATT1_OK,
            "malformed: kv_read survives hostile oversized-length response");

    memset(out_payload, 0, sizeof(out_payload));
    REQUIRE(att1_aimu_conformance_fabric_receive(endpoint,
                                                 0u,
                                                 &packet,
                                                 out_payload,
                                                 sizeof(out_payload),
                                                 &out_bytes) == ATT1_OK,
            "malformed: fabric_receive survives hostile oversized-length response");
    REQUIRE(out_bytes <= sizeof(out_payload),
            "malformed: fabric_receive clamps reported payload_bytes to caller capacity");

    att1_aimu_conformance_endpoint_destroy(endpoint);
    kill(pid, SIGKILL);
    waitpid(pid, &status, 0);
    unlink(g_socket_path);
    PASS("malformed_completion_from_hostile_daemon");
    return 0;
}

int main(void)
{
    int failed = 0;

    failed |= test_queue_full_over_socket();
    failed |= test_out_of_order_kv_append_over_socket();
    failed |= test_dma_descriptor_rejection_over_socket();
    failed |= test_endpoint_crash_mid_decode();
    failed |= test_malformed_completion_from_hostile_daemon();

    unlink(g_socket_path);

    if (failed) {
        printf("FAIL: aimu_endpoint_fault_injection suite\n");
        return 1;
    }
    printf("PASS: aimu_endpoint_fault_injection suite\n");
    return 0;
}
