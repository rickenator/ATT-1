/*
 * test_aimu_conformance.c  —  substrate-independent AIMU conformance checks.
 *
 * References:
 *   - docs/aimu_register_map.md §1.6 (register-map freeze)
 *   - docs/aimu_pcie_command_requirements.md §1.5 (command/completion freeze)
 *   - docs/aimu_register_map.md §15.7 / docs/schema_compatibility.md §12
 *     (DMA descriptor freeze)
 *   - docs/aimu_fabric_routing.md §16 (fabric/barrier/counter freeze)
 *   - docs/aimu_architecture.md §8.9 (KV-MMU / session memory requirements,
 *     M165: device-local KV-MMU in the endpoint)
 *
 * The suite talks only to att1_aimu_conformance_endpoint and therefore can be
 * reused unchanged against future socket-emulator or FPGA backends.
 */

#include "att1_aimu_conformance.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define PASS(name) do { printf("PASS: aimu_conformance: %s\n", (name)); } while (0)
#define FAIL(name) do { printf("FAIL: aimu_conformance: %s\n", (name)); return 1; } while (0)
#define REQUIRE(cond, name) do { if (!(cond)) { FAIL(name); } } while (0)

static int make_endpoint(const att1_aimu_conformance_config *config,
                         att1_aimu_conformance_endpoint **out)
{
    return att1_aimu_conformance_inproc_create(config, out) == ATT1_OK;
}

static void make_cmd(att1_aimu_cmd *cmd, att1_aimu_cmd_type type, uint8_t tile_id)
{
    memset(cmd, 0, sizeof(*cmd));
    cmd->command_type = (uint8_t)type;
    cmd->tile_id = tile_id;
}

static void make_h2d(att1_aimu_dma_desc *desc,
                     const att1_aimu_conformance_config *cfg)
{
    memset(desc, 0, sizeof(*desc));
    desc->host_addr = cfg->dma_host_base;
    desc->device_addr = cfg->dma_device_base;
    desc->byte_length = 4096u;
    desc->direction = (uint8_t)ATT1_AIMU_DMA_HOST_TO_DEVICE;
    desc->dtype = ATT1_AIMU_DMA_DTYPE_F32;
}

static int test_register_semantics(void)
{
    att1_aimu_conformance_endpoint *endpoint = NULL;
    uint32_t value32 = 0u;
    uint64_t value64 = 0u;

    REQUIRE(make_endpoint(NULL, &endpoint), "registers: endpoint create");
    REQUIRE(att1_aimu_conformance_sync_mmio(endpoint) == ATT1_OK,
            "registers: initial sync");

    REQUIRE(att1_aimu_conformance_mmio_read32(endpoint,
                                              ATT1_MMIO_DEVICE_ID,
                                              &value32) == ATT1_OK &&
            value32 == ATT1_MMIO_DEVICE_ID_DEFAULT,
            "registers: DEVICE_ID matches frozen default");
    REQUIRE(att1_aimu_conformance_mmio_read32(endpoint,
                                              ATT1_MMIO_REGISTER_MAP_VERSION,
                                              &value32) == ATT1_OK &&
            value32 == ATT1_AIMU_REGISTER_MAP_VERSION,
            "registers: REGISTER_MAP_VERSION is v1.0");
    REQUIRE(att1_aimu_conformance_mmio_read32(endpoint,
                                              ATT1_MMIO_TILE_COUNT,
                                              &value32) == ATT1_OK &&
            value32 == 4u,
            "registers: TILE_COUNT reflects endpoint config");

    REQUIRE(att1_aimu_conformance_mmio_write32(endpoint,
                                               ATT1_MMIO_DEVICE_ID,
                                               UINT32_C(0xDEADBEEF)) == ATT1_ERR_UNSUPPORTED,
            "registers: RO write rejected");

    REQUIRE(att1_aimu_conformance_mmio_write32(endpoint,
                                               ATT1_MMIO_GLOBAL_CONTROL,
                                               ATT1_MMIO_GCTRL_DISABLE_DEVICE) == ATT1_OK,
            "registers: GLOBAL_CONTROL disable write accepted");
    REQUIRE(att1_aimu_conformance_mmio_read32(endpoint,
                                              ATT1_MMIO_GLOBAL_STATUS,
                                              &value32) == ATT1_OK &&
            (value32 & ATT1_MMIO_GSTAT_DEVICE_READY) == 0u,
            "registers: DEVICE_READY clears on disable");
    REQUIRE(att1_aimu_conformance_mmio_write32(endpoint,
                                               ATT1_MMIO_GLOBAL_CONTROL,
                                               ATT1_MMIO_GCTRL_ENABLE_DEVICE) == ATT1_OK,
            "registers: GLOBAL_CONTROL enable write accepted");
    REQUIRE(att1_aimu_conformance_mmio_read32(endpoint,
                                              ATT1_MMIO_GLOBAL_STATUS,
                                              &value32) == ATT1_OK &&
            (value32 & ATT1_MMIO_GSTAT_DEVICE_READY) != 0u,
            "registers: DEVICE_READY sets on enable");

    REQUIRE(att1_aimu_conformance_mmio_write64(endpoint,
                                               ATT1_MMIO_CQ_BASE_ADDR_LOW,
                                               UINT64_C(0x1122334455667788)) == ATT1_OK,
            "registers: 64-bit RW pair write accepted");
    REQUIRE(att1_aimu_conformance_mmio_read64(endpoint,
                                              ATT1_MMIO_CQ_BASE_ADDR_LOW,
                                              &value64) == ATT1_OK &&
            value64 == UINT64_C(0x1122334455667788),
            "registers: 64-bit RW pair round-trips");

    REQUIRE(att1_aimu_conformance_mmio_read32(endpoint,
                                              UINT32_C(0x0040),
                                              &value32) == ATT1_OK &&
            value32 == ATT1_AIMU_MMIO_RESERVED_RD,
            "registers: reserved offset reads 0xDEADBEEF");
    REQUIRE(att1_aimu_conformance_mmio_write32(endpoint,
                                               UINT32_C(0x0040),
                                               UINT32_C(0x12345678)) == ATT1_OK,
            "registers: reserved offset write is discarded");
    REQUIRE(att1_aimu_conformance_mmio_read32(endpoint,
                                              UINT32_C(0x0040),
                                              &value32) == ATT1_OK &&
            value32 == ATT1_AIMU_MMIO_RESERVED_RD,
            "registers: reserved offset remains unchanged");

    REQUIRE(att1_aimu_conformance_mmio_read32(endpoint, 1u, &value32) == ATT1_ERR_INVALID_ARG,
            "registers: unaligned read32 rejected");
    REQUIRE(att1_aimu_conformance_mmio_write32(endpoint, 3u, 0u) == ATT1_ERR_INVALID_ARG,
            "registers: unaligned write32 rejected");
    REQUIRE(att1_aimu_conformance_mmio_read64(endpoint,
                                              ATT1_MMIO_CQ_BASE_ADDR_HIGH,
                                              &value64) == ATT1_ERR_INVALID_ARG,
            "registers: unaligned read64 rejected");

    att1_aimu_conformance_endpoint_destroy(endpoint);
    PASS("register_semantics");
    return 0;
}

static int test_command_lifecycle_and_snapshot(void)
{
    att1_aimu_conformance_endpoint *endpoint = NULL;
    att1_aimu_cmd cmd;
    att1_aimu_completion completion;
    att1_aimu_cmdq_counters counters;
    att1_aimu_trace_snapshot snapshot;
    uint32_t snap_control = UINT32_C(0xFFFFFFFF);
    uint64_t counter64 = 0u;

    REQUIRE(make_endpoint(NULL, &endpoint), "commands: endpoint create");

    make_cmd(&cmd, ATT1_AIMU_CMD_NOP, 0u);
    cmd.completion_fence_id = 7u;
    REQUIRE(att1_aimu_conformance_cmd_submit(endpoint, &cmd) == ATT1_OK,
            "commands: submit NOP");
    REQUIRE(att1_aimu_conformance_mmio_write32(endpoint,
                                               ATT1_MMIO_CQ_DOORBELL,
                                               1u) == ATT1_OK,
            "commands: doorbell write accepted");
    REQUIRE(att1_aimu_conformance_cmd_dispatch_one(endpoint) == ATT1_OK,
            "commands: dispatch_one completes NOP");
    memset(&completion, 0, sizeof(completion));
    REQUIRE(att1_aimu_conformance_cmd_poll_completion(endpoint, &completion) == ATT1_OK &&
            completion.command_id == 1u &&
            completion.result_code == ATT1_AIMU_OK &&
            completion.fence_value == 1u,
            "commands: completion record matches NOP lifecycle");

    make_cmd(&cmd, ATT1_AIMU_CMD_EXEC_MATMUL, 0u);
    REQUIRE(att1_aimu_conformance_cmd_submit(endpoint, &cmd) == ATT1_OK,
            "commands: submit EXEC_MATMUL");
    REQUIRE(att1_aimu_conformance_cmd_dispatch_one(endpoint) == ATT1_OK,
            "commands: dispatch_one completes EXEC_MATMUL");
    memset(&completion, 0, sizeof(completion));
    /* M166: the in-process endpoint now really executes EXEC_MATMUL; a
     * zeroed command (no input/output buffers, no resident tensor) is
     * rejected as invalid rather than unsupported. */
    REQUIRE(att1_aimu_conformance_cmd_poll_completion(endpoint, &completion) == ATT1_OK &&
            completion.result_code == ATT1_AIMU_ERR_INVALID_COMMAND &&
            att1_aimu_result_to_status((att1_aimu_result)completion.result_code) == ATT1_ERR_INVALID_ARG,
            "commands: invalid EXEC_MATMUL command rejected");

    REQUIRE(att1_aimu_conformance_cmd_get_counters(endpoint, &counters) == ATT1_OK &&
            counters.commands_submitted == 2u &&
            counters.commands_completed == 1u &&
            counters.commands_failed == 1u &&
            counters.unsupported_commands == 0u &&
            counters.fence_value == 1u,
            "commands: cmdq counters reflect lifecycle");

    REQUIRE(att1_aimu_conformance_snapshot_counters(endpoint) == ATT1_OK,
            "commands: snapshot_counters succeeds");
    REQUIRE(att1_aimu_conformance_mmio_read64(endpoint,
                                              ATT1_MMIO_CNT_CMD_ISSUED_LO,
                                              &counter64) == ATT1_OK &&
            counter64 == 2u,
            "commands: MMIO issued counter matches snapshot");
    REQUIRE(att1_aimu_conformance_mmio_read64(endpoint,
                                              ATT1_MMIO_CNT_CMD_COMPLETED_LO,
                                              &counter64) == ATT1_OK &&
            counter64 == 1u,
            "commands: MMIO completed counter matches snapshot");
    REQUIRE(att1_aimu_conformance_trace_get_snapshot(endpoint, &snapshot) == ATT1_OK &&
            snapshot.meta.snapshot_id == 1u &&
            snapshot.cmdq.commands_submitted == 2u,
            "commands: trace snapshot captures cmdq counters");

    REQUIRE(att1_aimu_conformance_mmio_write32(endpoint,
                                               ATT1_MMIO_COUNTER_SNAPSHOT_CONTROL,
                                               ATT1_MMIO_SNAP_NOW) == ATT1_OK,
            "commands: SNAP_NOW write accepted");
    REQUIRE(att1_aimu_conformance_mmio_read32(endpoint,
                                              ATT1_MMIO_COUNTER_SNAPSHOT_CONTROL,
                                              &snap_control) == ATT1_OK &&
            (snap_control & ATT1_MMIO_SNAP_NOW) == 0u,
            "commands: SNAP_NOW self-clears");

    att1_aimu_conformance_endpoint_destroy(endpoint);
    PASS("command_lifecycle_and_snapshot");
    return 0;
}

static int test_command_queue_full_and_hostile_submit(void)
{
    att1_aimu_conformance_config cfg;
    att1_aimu_conformance_endpoint *endpoint = NULL;
    att1_aimu_cmd cmd;

    att1_aimu_conformance_default_config(&cfg);
    cfg.cmd_ring_depth = 4u;
    cfg.comp_ring_depth = 4u;

    REQUIRE(make_endpoint(&cfg, &endpoint), "queue_full: endpoint create");

    make_cmd(&cmd, ATT1_AIMU_CMD_NOP, 0u);
    REQUIRE(att1_aimu_conformance_cmd_submit(endpoint, &cmd) == ATT1_OK,
            "queue_full: submit 1");
    REQUIRE(att1_aimu_conformance_cmd_submit(endpoint, &cmd) == ATT1_OK,
            "queue_full: submit 2");
    REQUIRE(att1_aimu_conformance_cmd_submit(endpoint, &cmd) == ATT1_OK,
            "queue_full: submit 3");
    REQUIRE(att1_aimu_conformance_cmd_submit(endpoint, &cmd) == ATT1_ERR_QUEUE_FULL,
            "queue_full: 4th submit rejected");

    make_cmd(&cmd, ATT1_AIMU_CMD_NOP, (uint8_t)cfg.tile_count);
    REQUIRE(att1_aimu_conformance_cmd_submit(endpoint, &cmd) == ATT1_ERR_INVALID_ARG,
            "queue_full: invalid tile rejected");

    att1_aimu_conformance_endpoint_destroy(endpoint);
    PASS("command_queue_full_and_hostile_submit");
    return 0;
}

static int test_dma_validation_and_counters(void)
{
    att1_aimu_conformance_config cfg;
    att1_aimu_conformance_endpoint *endpoint = NULL;
    att1_aimu_dma_desc desc;
    att1_aimu_dma_counters counters;
    uint64_t tensor_bytes = 0u;

    att1_aimu_conformance_default_config(&cfg);
    REQUIRE(make_endpoint(&cfg, &endpoint), "dma: endpoint create");

    make_h2d(&desc, &cfg);
    REQUIRE(att1_aimu_conformance_dma_validate(endpoint, &desc) == ATT1_OK,
            "dma: valid descriptor passes validate");
    REQUIRE(att1_aimu_conformance_dma_submit(endpoint, &desc) == ATT1_OK,
            "dma: valid descriptor passes submit");

    desc.host_addr = cfg.dma_host_base + 1u;
    REQUIRE(att1_aimu_conformance_dma_validate(endpoint, &desc) == ATT1_ERR_INVALID_ARG,
            "dma: unaligned host address rejected by validate");
    REQUIRE(att1_aimu_conformance_dma_submit(endpoint, &desc) == ATT1_ERR_INVALID_ARG,
            "dma: unaligned host address rejected by submit");

    make_h2d(&desc, &cfg);
    desc.flags = UINT16_C(0x8000);
    REQUIRE(att1_aimu_conformance_dma_validate(endpoint, &desc) == ATT1_ERR_INVALID_ARG,
            "dma: unsupported flags rejected by validate");
    REQUIRE(att1_aimu_conformance_dma_submit(endpoint, &desc) == ATT1_ERR_INVALID_ARG,
            "dma: unsupported flags rejected by submit");

    REQUIRE(att1_aimu_conformance_dma_get_counters(endpoint, &counters) == ATT1_OK &&
            counters.dma_submitted == 3u &&
            counters.dma_completed == 1u &&
            counters.dma_failed == 2u &&
            counters.alignment_failures == 1u &&
            counters.unsupported_flags == 1u &&
            counters.bytes_host_to_device == 4096u,
            "dma: counters match frozen validation accounting");

    REQUIRE(att1_aimu_conformance_snapshot_counters(endpoint) == ATT1_OK,
            "dma: snapshot_counters succeeds");
    REQUIRE(att1_aimu_conformance_mmio_read64(endpoint,
                                              ATT1_MMIO_CNT_TENSOR_BYTES_RD_LO,
                                              &tensor_bytes) == ATT1_OK &&
            tensor_bytes == 4096u,
            "dma: tensor-bytes-read counter reflects DMA snapshot");

    att1_aimu_conformance_endpoint_destroy(endpoint);
    PASS("dma_validation_and_counters");
    return 0;
}

static int test_fabric_send_receive_and_counters(void)
{
    att1_aimu_conformance_endpoint *endpoint = NULL;
    att1_fabric_packet packet;
    att1_fabric_counters counters;
    unsigned char payload[3] = {1u, 2u, 3u};
    unsigned char out[8] = {0u};
    size_t out_bytes = 0u;

    REQUIRE(make_endpoint(NULL, &endpoint), "fabric_io: endpoint create");
    REQUIRE(att1_aimu_conformance_fabric_send(endpoint,
                                              0u,
                                              1u,
                                              ATT1_PACKET_CONTROL,
                                              payload,
                                              sizeof(payload),
                                              11u) == ATT1_OK,
            "fabric_io: send succeeds");
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
            "fabric_io: receive preserves payload and metadata");
    REQUIRE(att1_aimu_conformance_fabric_get_counters(endpoint, &counters) == ATT1_OK &&
            counters.packets_sent == 1u &&
            counters.packets_received == 1u &&
            counters.payload_bytes_sent == sizeof(payload) &&
            counters.payload_bytes_received == sizeof(payload),
            "fabric_io: counters track unicast traffic");

    att1_aimu_conformance_endpoint_destroy(endpoint);
    PASS("fabric_send_receive_and_counters");
    return 0;
}

static int test_fabric_broadcast_and_barrier_semantics(void)
{
    att1_aimu_conformance_config cfg;
    att1_aimu_conformance_endpoint *endpoint = NULL;
    att1_fabric_counters counters;
    unsigned char payload[3] = {4u, 5u, 6u};
    unsigned char out[8] = {0u};
    att1_fabric_packet packet;
    size_t out_bytes = 0u;
    uint32_t group[2] = {1u, 2u};
    uint32_t barrier_group[3] = {0u, 1u, 2u};
    uint32_t barrier_bad[2] = {0u, 1u};
    int complete = 0;

    att1_aimu_conformance_default_config(&cfg);
    cfg.fabric_queue_capacity = 1u;
    REQUIRE(make_endpoint(&cfg, &endpoint), "fabric_barrier: endpoint create");

    REQUIRE(att1_aimu_conformance_fabric_send(endpoint,
                                              0u,
                                              1u,
                                              ATT1_PACKET_ACTIVATION,
                                              payload,
                                              sizeof(payload),
                                              21u) == ATT1_OK,
            "fabric_barrier: prefill target queue");
    REQUIRE(att1_aimu_conformance_fabric_broadcast(endpoint,
                                                   0u,
                                                   group,
                                                   2u,
                                                   ATT1_PACKET_TRACE,
                                                   payload,
                                                   sizeof(payload),
                                                   22u) == ATT1_ERR_QUEUE_FULL,
            "fabric_barrier: broadcast is all-or-nothing on queue-full");
    REQUIRE(att1_aimu_conformance_fabric_receive(endpoint,
                                                 2u,
                                                 &packet,
                                                 out,
                                                 sizeof(out),
                                                 &out_bytes) == ATT1_ERR_QUEUE_EMPTY,
            "fabric_barrier: failed broadcast did not enqueue to other targets");

    complete = 1;
    REQUIRE(att1_aimu_conformance_fabric_barrier_arrive(endpoint,
                                                        0u,
                                                        barrier_group,
                                                        3u,
                                                        &complete) == ATT1_OK &&
            complete == 0,
            "fabric_barrier: first arrival starts barrier without completion");
    REQUIRE(att1_aimu_conformance_fabric_barrier_arrive(endpoint,
                                                        1u,
                                                        barrier_bad,
                                                        2u,
                                                        &complete) == ATT1_ERR_INVALID_ARG,
            "fabric_barrier: mismatched participant set rejected");
    REQUIRE(att1_aimu_conformance_fabric_barrier_arrive(endpoint,
                                                        1u,
                                                        barrier_group,
                                                        3u,
                                                        &complete) == ATT1_OK &&
            complete == 0,
            "fabric_barrier: second valid arrival still incomplete");
    REQUIRE(att1_aimu_conformance_fabric_barrier_arrive(endpoint,
                                                        2u,
                                                        barrier_group,
                                                        3u,
                                                        &complete) == ATT1_OK &&
            complete == 1,
            "fabric_barrier: completing arrival is sole complete=1 event");

    REQUIRE(att1_aimu_conformance_fabric_get_counters(endpoint, &counters) == ATT1_OK &&
            counters.queue_full_errors == 1u &&
            counters.empty_receives == 1u &&
            counters.invalid_packets == 1u &&
            counters.barrier_arrivals == 3u &&
            counters.barrier_completions == 1u,
            "fabric_barrier: queue-full and barrier counters preserved");

    att1_aimu_conformance_endpoint_destroy(endpoint);
    PASS("fabric_broadcast_and_barrier_semantics");
    return 0;
}

static int test_kv_mmu_session_lifecycle_and_semantics(void)
{
    att1_aimu_conformance_endpoint *endpoint = NULL;
    att1_aimu_conformance_config cfg;
    const uint64_t session_id = 42u;
    const size_t layer_id = 1u;
    const size_t num_heads = 4u;
    const size_t head_dim = 16u;
    const size_t per_position = num_heads * head_dim;
    float key_pos0[64];
    float value_pos0[64];
    float key_pos1[64];
    float value_pos1[64];
    float out_key[16];
    float out_value[16];
    float range_keys[32];
    float range_values[32];
    att1_kv_mmu_counters counters;
    size_t i;

    att1_aimu_conformance_default_config(&cfg);

    for (i = 0u; i < per_position; ++i) {
        key_pos0[i] = (float)(100u + i);
        value_pos0[i] = (float)(200u + i);
        key_pos1[i] = (float)(300u + i);
        value_pos1[i] = (float)(400u + i);
    }

    REQUIRE(make_endpoint(&cfg, &endpoint), "kv_mmu: endpoint create");

    REQUIRE(att1_aimu_conformance_kv_read(endpoint,
                                          session_id,
                                          layer_id,
                                          0u,
                                          0u,
                                          out_key,
                                          head_dim,
                                          out_value,
                                          head_dim) != ATT1_OK,
            "kv_mmu: read before session created is rejected");

    REQUIRE(att1_aimu_conformance_kv_create_session(endpoint, session_id) == ATT1_OK,
            "kv_mmu: create session");
    REQUIRE(att1_aimu_conformance_kv_create_session(endpoint, session_id) != ATT1_OK,
            "kv_mmu: duplicate session id rejected");

    REQUIRE(att1_aimu_conformance_kv_append(endpoint,
                                            session_id,
                                            layer_id,
                                            1u,
                                            key_pos1,
                                            per_position,
                                            value_pos1,
                                            per_position) != ATT1_OK,
            "kv_mmu: out-of-order append (position 1 before 0) rejected");

    REQUIRE(att1_aimu_conformance_kv_append(endpoint,
                                            session_id,
                                            layer_id,
                                            0u,
                                            key_pos0,
                                            per_position,
                                            value_pos0,
                                            per_position) == ATT1_OK,
            "kv_mmu: append position 0");
    REQUIRE(att1_aimu_conformance_kv_append(endpoint,
                                            session_id,
                                            layer_id,
                                            0u,
                                            key_pos0,
                                            per_position,
                                            value_pos0,
                                            per_position) != ATT1_OK,
            "kv_mmu: duplicate append at same position rejected");
    REQUIRE(att1_aimu_conformance_kv_append(endpoint,
                                            session_id,
                                            layer_id,
                                            1u,
                                            key_pos1,
                                            per_position,
                                            value_pos1,
                                            per_position) == ATT1_OK,
            "kv_mmu: append position 1");

    memset(out_key, 0, sizeof(out_key));
    memset(out_value, 0, sizeof(out_value));
    REQUIRE(att1_aimu_conformance_kv_read(endpoint,
                                          session_id,
                                          layer_id,
                                          2u,
                                          0u,
                                          out_key,
                                          head_dim,
                                          out_value,
                                          head_dim) == ATT1_OK &&
            out_key[0] == key_pos0[(2u * head_dim)] &&
            out_value[0] == value_pos0[(2u * head_dim)],
            "kv_mmu: read returns head-2 data written at append time");

    memset(range_keys, 0, sizeof(range_keys));
    memset(range_values, 0, sizeof(range_values));
    REQUIRE(att1_aimu_conformance_kv_copy_range(endpoint,
                                                session_id,
                                                layer_id,
                                                1u,
                                                0u,
                                                2u,
                                                range_keys,
                                                2u * head_dim,
                                                range_values,
                                                2u * head_dim) == ATT1_OK &&
            range_keys[0] == key_pos0[head_dim] &&
            range_keys[head_dim] == key_pos1[head_dim] &&
            range_values[0] == value_pos0[head_dim] &&
            range_values[head_dim] == value_pos1[head_dim],
            "kv_mmu: copy_range preserves token order across two positions");

    REQUIRE(att1_aimu_conformance_kv_get_counters(endpoint, &counters) == ATT1_OK &&
            counters.append_ops == 2u &&
            counters.read_ops == 3u &&
            counters.range_copy_ops == 1u &&
            counters.errors == 4u,
            "kv_mmu: counters reflect append/read/range-copy/error activity");

    REQUIRE(att1_aimu_conformance_kv_destroy_session(endpoint, session_id) == ATT1_OK,
            "kv_mmu: destroy session");
    REQUIRE(att1_aimu_conformance_kv_read(endpoint,
                                          session_id,
                                          layer_id,
                                          0u,
                                          0u,
                                          out_key,
                                          head_dim,
                                          out_value,
                                          head_dim) != ATT1_OK,
            "kv_mmu: read after session destroyed is rejected");

    att1_aimu_conformance_endpoint_destroy(endpoint);
    PASS("kv_mmu_session_lifecycle_and_semantics");
    return 0;
}

int main(void)
{
    int failed = 0;

    failed |= test_register_semantics();
    failed |= test_command_lifecycle_and_snapshot();
    failed |= test_command_queue_full_and_hostile_submit();
    failed |= test_dma_validation_and_counters();
    failed |= test_fabric_send_receive_and_counters();
    failed |= test_fabric_broadcast_and_barrier_semantics();
    failed |= test_kv_mmu_session_lifecycle_and_semantics();

    if (failed != 0) {
        return 1;
    }
    printf("PASS: aimu_conformance: all checks passed\n");
    return 0;
}
