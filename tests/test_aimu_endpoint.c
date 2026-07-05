/*
 * test_aimu_endpoint.c  —  M162 out-of-process endpoint smoke test.
 *
 * Spawns the `att1-aimu-endpoint` daemon as a separate process, connects to
 * it over a Unix domain socket via att1_aimu_conformance_socket_connect(),
 * and exercises the same register/command-queue/DMA/fabric semantics as
 * tests/test_aimu_conformance.c (M161) to confirm the out-of-process
 * transport (M93 §8.8, M162) preserves identical queue semantics and
 * counters relative to the in-process simulator.
 */

#define _POSIX_C_SOURCE 200112L

#include "att1_aimu_conformance.h"
#include "att1_aimu_endpoint_client.h"

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define PASS(name) do { printf("PASS: aimu_endpoint: %s\n", (name)); } while (0)
#define FAIL(name) do { printf("FAIL: aimu_endpoint: %s\n", (name)); return 1; } while (0)
#define REQUIRE(cond, name) do { if (!(cond)) { FAIL(name); } } while (0)

static const char *g_socket_path = "build/test_aimu_endpoint.sock";

static pid_t spawn_daemon(const char *socket_path)
{
    pid_t pid = fork();
    if (pid < 0) {
        return -1;
    }
    if (pid == 0) {
        execl("./build/att1-aimu-endpoint", "att1-aimu-endpoint",
              "--socket", socket_path, "--once", (char *)NULL);
        _exit(127);
    }
    return pid;
}

static int connect_with_retry(const char *socket_path, att1_aimu_conformance_endpoint **out)
{
    int attempt;
    for (attempt = 0; attempt < 100; ++attempt) {
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

static int test_endpoint_register_and_command_lifecycle(pid_t *out_pid)
{
    att1_aimu_conformance_endpoint *endpoint = NULL;
    pid_t pid;
    uint32_t value32 = 0u;
    att1_aimu_cmd cmd;
    att1_aimu_completion completion;
    att1_aimu_cmdq_counters cmd_counters;

    unlink(g_socket_path);
    pid = spawn_daemon(g_socket_path);
    REQUIRE(pid > 0, "endpoint: daemon spawn");

    REQUIRE(connect_with_retry(g_socket_path, &endpoint) == 0,
            "endpoint: client connect");

    REQUIRE(att1_aimu_conformance_sync_mmio(endpoint) == ATT1_OK,
            "endpoint: initial sync");
    REQUIRE(att1_aimu_conformance_mmio_read32(endpoint,
                                              ATT1_MMIO_DEVICE_ID,
                                              &value32) == ATT1_OK &&
            value32 == ATT1_MMIO_DEVICE_ID_DEFAULT,
            "endpoint: DEVICE_ID matches frozen default over socket");
    REQUIRE(att1_aimu_conformance_mmio_read32(endpoint,
                                              ATT1_MMIO_TILE_COUNT,
                                              &value32) == ATT1_OK &&
            value32 == 4u,
            "endpoint: TILE_COUNT reflects daemon default config");
    REQUIRE(att1_aimu_conformance_mmio_write32(endpoint,
                                               ATT1_MMIO_DEVICE_ID,
                                               UINT32_C(0xDEADBEEF)) == ATT1_ERR_UNSUPPORTED,
            "endpoint: RO register write rejected over socket");

    make_cmd(&cmd, ATT1_AIMU_CMD_NOP, 0u);
    REQUIRE(att1_aimu_conformance_cmd_submit(endpoint, &cmd) == ATT1_OK &&
            cmd.command_id == 1u,
            "endpoint: NOP submit assigns command_id over socket");
    REQUIRE(att1_aimu_conformance_cmd_dispatch_one(endpoint) == ATT1_OK,
            "endpoint: dispatch_one completes NOP over socket");
    memset(&completion, 0, sizeof(completion));
    REQUIRE(att1_aimu_conformance_cmd_poll_completion(endpoint, &completion) == ATT1_OK &&
            completion.command_id == 1u &&
            completion.result_code == ATT1_AIMU_OK,
            "endpoint: completion record matches NOP lifecycle over socket");
    REQUIRE(att1_aimu_conformance_cmd_get_counters(endpoint, &cmd_counters) == ATT1_OK &&
            cmd_counters.commands_submitted == 1u &&
            cmd_counters.commands_completed == 1u,
            "endpoint: cmdq counters reflect lifecycle over socket");

    att1_aimu_conformance_endpoint_destroy(endpoint);
    *out_pid = pid;
    PASS("register_and_command_lifecycle");
    return 0;
}

static int test_endpoint_dma_and_fabric(void)
{
    att1_aimu_conformance_endpoint *endpoint = NULL;
    pid_t pid;
    att1_aimu_conformance_config cfg;
    att1_aimu_dma_desc desc;
    att1_aimu_dma_counters dma_counters;
    att1_fabric_packet packet;
    att1_fabric_counters fabric_counters;
    unsigned char payload[3] = {1u, 2u, 3u};
    unsigned char out[8] = {0u};
    size_t out_bytes = 0u;
    int status = 0;

    unlink(g_socket_path);
    pid = spawn_daemon(g_socket_path);
    REQUIRE(pid > 0, "endpoint_dma_fabric: daemon spawn");
    REQUIRE(connect_with_retry(g_socket_path, &endpoint) == 0,
            "endpoint_dma_fabric: client connect");

    att1_aimu_conformance_default_config(&cfg);

    memset(&desc, 0, sizeof(desc));
    desc.host_addr = cfg.dma_host_base;
    desc.device_addr = cfg.dma_device_base;
    desc.byte_length = 4096u;
    desc.direction = (uint8_t)ATT1_AIMU_DMA_HOST_TO_DEVICE;
    desc.dtype = ATT1_AIMU_DMA_DTYPE_F32;

    REQUIRE(att1_aimu_conformance_dma_validate(endpoint, &desc) == ATT1_OK,
            "endpoint_dma_fabric: dma validate over socket");
    REQUIRE(att1_aimu_conformance_dma_submit(endpoint, &desc) == ATT1_OK,
            "endpoint_dma_fabric: dma submit over socket");
    REQUIRE(att1_aimu_conformance_dma_get_counters(endpoint, &dma_counters) == ATT1_OK &&
            dma_counters.dma_submitted == 1u &&
            dma_counters.dma_completed == 1u &&
            dma_counters.bytes_host_to_device == desc.byte_length,
            "endpoint_dma_fabric: dma counters match over socket");

    REQUIRE(att1_aimu_conformance_fabric_send(endpoint,
                                              0u,
                                              1u,
                                              ATT1_PACKET_CONTROL,
                                              payload,
                                              sizeof(payload),
                                              11u) == ATT1_OK,
            "endpoint_dma_fabric: fabric send over socket");
    REQUIRE(att1_aimu_conformance_fabric_receive(endpoint,
                                                 1u,
                                                 &packet,
                                                 out,
                                                 sizeof(out),
                                                 &out_bytes) == ATT1_OK &&
            packet.type == ATT1_PACKET_CONTROL &&
            packet.source_tile == 0u &&
            packet.target_tile == 1u &&
            packet.tag == 11u &&
            out_bytes == sizeof(payload) &&
            memcmp(out, payload, sizeof(payload)) == 0,
            "endpoint_dma_fabric: fabric receive preserves payload over socket");
    REQUIRE(att1_aimu_conformance_fabric_get_counters(endpoint, &fabric_counters) == ATT1_OK &&
            fabric_counters.packets_sent == 1u &&
            fabric_counters.packets_received == 1u &&
            fabric_counters.payload_bytes_sent == sizeof(payload),
            "endpoint_dma_fabric: fabric counters match over socket");

    {
        uint32_t participants[1] = {0u};
        int complete = 0;
        REQUIRE(att1_aimu_conformance_fabric_barrier_arrive(endpoint,
                                                            0u,
                                                            participants,
                                                            1u,
                                                            &complete) == ATT1_OK &&
                complete != 0,
                "endpoint_dma_fabric: single-participant barrier completes over socket");
    }

    att1_aimu_conformance_endpoint_destroy(endpoint);
    waitpid(pid, &status, 0);
    unlink(g_socket_path);
    PASS("dma_and_fabric");
    return 0;
}

int main(void)
{
    int failed = 0;
    pid_t pid = -1;
    int status = 0;

    failed |= test_endpoint_register_and_command_lifecycle(&pid);
    if (pid > 0) {
        waitpid(pid, &status, 0);
    }
    failed |= test_endpoint_dma_and_fabric();

    unlink(g_socket_path);

    if (failed) {
        printf("FAIL: aimu_endpoint suite\n");
        return 1;
    }
    printf("PASS: aimu_endpoint suite\n");
    return 0;
}
