/*
 * test_backend_pcie.c  —  M163 `backend_pcie.c` host backend smoke test.
 *
 * Exercises att1_backend_pcie_create() against an in-process
 * att1_aimu_conformance_endpoint (M161) to confirm the backend-swap
 * contract (M93 §8.3): alloc/free/sync succeed like any other backend,
 * and every EXEC_* math op round-trips through submit -> dispatch_one ->
 * poll_completion, mapping the (currently unsupported) completion result
 * code to att1_status_t via att1_aimu_result_to_status(). This suite does
 * not (yet) validate compute correctness — the command-queue simulator
 * does not execute EXEC_* commands until a later milestone (M166).
 */

#include "att1_aimu_conformance.h"
#include "att1_aimu_dma.h"
#include "att1_backend.h"

#include <stdio.h>

#define PASS(name) do { printf("PASS: backend_pcie: %s\n", (name)); } while (0)
#define FAIL(name) do { printf("FAIL: backend_pcie: %s\n", (name)); return 1; } while (0)
#define REQUIRE(cond, name) do { if (!(cond)) { FAIL(name); } } while (0)

static int test_create_invalid_args(void)
{
    att1_backend *backend = NULL;
    att1_aimu_conformance_endpoint *endpoint = NULL;

    REQUIRE(att1_aimu_conformance_inproc_create(NULL, &endpoint) == ATT1_OK,
            "invalid_args: endpoint create");

    REQUIRE(att1_backend_pcie_create(NULL, 0u, &backend) == ATT1_ERR_INVALID_ARG,
            "invalid_args: NULL endpoint rejected");
    REQUIRE(att1_backend_pcie_create(endpoint, 0u, NULL) == ATT1_ERR_INVALID_ARG,
            "invalid_args: NULL out_backend rejected");

    att1_aimu_conformance_endpoint_destroy(endpoint);
    PASS("invalid_args");
    return 0;
}

static int test_alloc_free_sync(void)
{
    att1_aimu_conformance_endpoint *endpoint = NULL;
    att1_backend *backend = NULL;
    void *buf = NULL;

    REQUIRE(att1_aimu_conformance_inproc_create(NULL, &endpoint) == ATT1_OK,
            "alloc_free_sync: endpoint create");
    REQUIRE(att1_backend_pcie_create(endpoint, 0u, &backend) == ATT1_OK,
            "alloc_free_sync: backend create");

    REQUIRE(backend->ops != NULL && backend->ops->name != NULL &&
            (backend->ops->name[0] == 'p') && (backend->ops->name[1] == 'c') &&
            (backend->ops->name[2] == 'i') && (backend->ops->name[3] == 'e'),
            "alloc_free_sync: backend name is \"pcie\"");

    buf = backend->ops->alloc(backend, 256u);
    REQUIRE(buf != NULL, "alloc_free_sync: alloc succeeds");
    backend->ops->free(backend, buf);

    REQUIRE(backend->ops->sync(backend) == 0, "alloc_free_sync: sync succeeds");

    /* softmax_f32 has no dedicated frozen command type on this backend. */
    REQUIRE(backend->ops->softmax_f32 == NULL,
            "alloc_free_sync: softmax_f32 left unimplemented");

    att1_backend_destroy(backend);
    att1_aimu_conformance_endpoint_destroy(endpoint);
    PASS("alloc_free_sync");
    return 0;
}

static int test_exec_ops_transport_roundtrip(void)
{
    att1_aimu_conformance_endpoint *endpoint = NULL;
    att1_backend *backend = NULL;
    att1_aimu_cmdq_counters counters;
    float scratch[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    REQUIRE(att1_aimu_conformance_inproc_create(NULL, &endpoint) == ATT1_OK,
            "exec_ops: endpoint create");
    REQUIRE(att1_backend_pcie_create(endpoint, 0u, &backend) == ATT1_OK,
            "exec_ops: backend create");

    /* Every EXEC_* op currently maps to ATT1_AIMU_ERR_UNSUPPORTED_OP at the
     * in-process simulator (src/aimu_cmdq.c), so the ops report failure
     * (-1) via att1_aimu_result_to_status() while still completing the
     * submit -> dispatch -> poll transport round trip (validated below via
     * the cmdq counters). */
    REQUIRE(backend->ops->matmul_f32(backend, scratch, scratch, scratch, 1u, 1u, 1u) == -1,
            "exec_ops: matmul_f32 maps unsupported completion to failure");
    REQUIRE(backend->ops->rmsnorm_f32(backend, scratch, scratch, scratch, 1u, 1e-5f) == -1,
            "exec_ops: rmsnorm_f32 maps unsupported completion to failure");
    REQUIRE(backend->ops->rope_f32(backend, scratch, 1u, 0u, 10000.0f) == -1,
            "exec_ops: rope_f32 maps unsupported completion to failure");
    REQUIRE(backend->ops->ffn_swiglu_f32(backend, scratch, scratch, scratch, 1u) == -1,
            "exec_ops: ffn_swiglu_f32 maps unsupported completion to failure");
    REQUIRE(backend->ops->matmul_q8xf32(backend, scratch, scratch, 1u, 1u, NULL) == -1,
            "exec_ops: matmul_q8xf32 maps unsupported completion to failure");
    REQUIRE(backend->ops->matmul_q4xf32(backend, scratch, scratch, 1u, 1u, NULL) == -1,
            "exec_ops: matmul_q4xf32 maps unsupported completion to failure");

    REQUIRE(att1_aimu_conformance_cmd_get_counters(endpoint, &counters) == ATT1_OK &&
            counters.commands_submitted == 6u &&
            counters.commands_completed == 6u &&
            counters.unsupported_commands == 6u,
            "exec_ops: cmdq counters reflect six submitted/completed/unsupported commands");

    att1_backend_destroy(backend);
    att1_aimu_conformance_endpoint_destroy(endpoint);
    PASS("exec_ops_transport_roundtrip");
    return 0;
}

/*
 * M164: att1_backend_pcie_load_tensor() one-time shard transfer / tensor
 * residency tests.
 */
static int test_load_tensor_residency(void)
{
    att1_aimu_conformance_config cfg;
    att1_aimu_conformance_endpoint *endpoint = NULL;
    att1_backend *backend = NULL;
    att1_backend_pcie_residency_counters counters;
    att1_aimu_dma_counters dma_counters;

    att1_aimu_conformance_default_config(&cfg);
    REQUIRE(att1_aimu_conformance_inproc_create(&cfg, &endpoint) == ATT1_OK,
            "load_tensor: endpoint create");
    REQUIRE(att1_backend_pcie_create(endpoint, 0u, &backend) == ATT1_OK,
            "load_tensor: backend create");

    REQUIRE(att1_backend_pcie_tensor_is_resident(backend, 7u) == 0,
            "load_tensor: tensor not resident before load");

    REQUIRE(att1_backend_pcie_load_tensor(backend, 7u,
                                          cfg.dma_host_base,
                                          cfg.dma_device_base,
                                          4096u,
                                          ATT1_AIMU_DMA_DTYPE_F32,
                                          0u) == ATT1_OK,
            "load_tensor: first load succeeds");

    REQUIRE(att1_backend_pcie_tensor_is_resident(backend, 7u) == 1,
            "load_tensor: tensor resident after load");

    /* Enforcement: reloading the same tensor_id is rejected and must not
     * resubmit any DMA transfer (M93 §8.12 "never re-read"). */
    REQUIRE(att1_backend_pcie_load_tensor(backend, 7u,
                                          cfg.dma_host_base,
                                          cfg.dma_device_base,
                                          4096u,
                                          ATT1_AIMU_DMA_DTYPE_F32,
                                          0u) == ATT1_ERR_STATE,
            "load_tensor: duplicate load rejected");

    /* A different tensor_id transfers independently. */
    REQUIRE(att1_backend_pcie_load_tensor(backend, 8u,
                                          cfg.dma_host_base + 4096u,
                                          cfg.dma_device_base + 4096u,
                                          1024u,
                                          ATT1_AIMU_DMA_DTYPE_F32,
                                          0u) == ATT1_OK,
            "load_tensor: second distinct tensor succeeds");

    REQUIRE(att1_backend_pcie_get_residency_counters(backend, &counters) == ATT1_OK &&
            counters.tensors_resident == 2u &&
            counters.transfers_submitted == 2u &&
            counters.descriptors_submitted == 2u &&
            counters.bytes_transferred == (4096u + 1024u) &&
            counters.duplicate_transfer_rejections == 1u,
            "load_tensor: residency counters reflect two loads and one rejection");

    REQUIRE(att1_aimu_conformance_dma_get_counters(endpoint, &dma_counters) == ATT1_OK &&
            dma_counters.dma_submitted == 2u &&
            dma_counters.dma_completed == 2u &&
            dma_counters.bytes_host_to_device == (4096u + 1024u),
            "load_tensor: underlying DMA counters confirm only two transfers occurred");

    /* Invalid args. */
    REQUIRE(att1_backend_pcie_load_tensor(NULL, 1u, 0u, 0u, 16u, 0u, 0u) == ATT1_ERR_INVALID_ARG,
            "load_tensor: NULL backend rejected");
    REQUIRE(att1_backend_pcie_load_tensor(backend, 9u, cfg.dma_host_base,
                                          cfg.dma_device_base, 0u, 0u, 0u) ==
                    ATT1_ERR_INVALID_ARG,
            "load_tensor: zero byte_length rejected");
    REQUIRE(att1_backend_pcie_tensor_is_resident(NULL, 7u) == 0,
            "load_tensor: is_resident(NULL) is false");
    REQUIRE(att1_backend_pcie_get_residency_counters(backend, NULL) == ATT1_ERR_INVALID_ARG,
            "load_tensor: get_residency_counters NULL out rejected");

    att1_backend_destroy(backend);
    att1_aimu_conformance_endpoint_destroy(endpoint);
    PASS("load_tensor_residency");
    return 0;
}

int main(void)
{
    int failed = 0;

    failed |= test_create_invalid_args();
    failed |= test_alloc_free_sync();
    failed |= test_exec_ops_transport_roundtrip();
    failed |= test_load_tensor_residency();

    if (failed) {
        printf("backend_pcie: SOME TESTS FAILED\n");
        return 1;
    }

    printf("backend_pcie: ALL TESTS PASSED\n");
    return 0;
}
