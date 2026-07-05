/*
 * test_backend_pcie.c  —  M163 `backend_pcie.c` host backend smoke test;
 * M166 real-execution correctness tests.
 *
 * Exercises att1_backend_pcie_create() against an in-process
 * att1_aimu_conformance_endpoint (M161) to confirm the backend-swap
 * contract (M93 §8.3): alloc/free/sync succeed like any other backend,
 * and every EXEC_* math op round-trips through submit -> dispatch_one ->
 * poll_completion, mapping the completion result code to att1_status_t
 * via att1_aimu_result_to_status(). As of M166 the endpoint actually
 * executes EXEC_MATMUL/EXEC_RMSNORM/EXEC_ROPE/EXEC_FFN, so this suite also
 * validates single-op and end-to-end (attention + transformer block,
 * multi-step decode) compute correctness against the cpu-f32/cpu-q8/cpu-q4
 * backends.
 */

#include "att1_aimu_conformance.h"
#include "att1_aimu_dma.h"
#include "att1_backend.h"
#include "att1_attention.h"
#include "att1_kv_cache.h"
#include "att1_math.h"
#include "att1_quant.h"
#include "att1_transformer_block.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int near_f32(float lhs, float rhs)
{
    return fabsf(lhs - rhs) < 0.00001f;
}

static int within_quant_tolerance(float lhs, float rhs)
{
    return fabsf(lhs - rhs) < 0.05f;
}

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

    /* M166: softmax_f32 is now implemented (computed locally on the host,
     * since no dedicated frozen command type covers plain softmax). */
    REQUIRE(backend->ops->softmax_f32 != NULL,
            "alloc_free_sync: softmax_f32 implemented");

    att1_backend_destroy(backend);
    att1_aimu_conformance_endpoint_destroy(endpoint);
    PASS("alloc_free_sync");
    return 0;
}

static int test_exec_ops_correctness(void)
{
    att1_aimu_conformance_endpoint *endpoint = NULL;
    att1_backend *backend = NULL;
    att1_aimu_cmdq_counters counters;

    const float lhs3[3] = {1.0f, 2.0f, 3.0f};
    const float rhs3x2[6] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f
    };
    float matmul_out[2] = {0.0f, 0.0f};
    float matmul_expected[2] = {0.0f, 0.0f};

    const float rms_src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float rms_weight[4] = {1.0f, 0.5f, 2.0f, 1.5f};
    float rms_out[4] = {0.0f};
    float rms_expected[4] = {0.0f};

    float rope_values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float rope_expected[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    float softmax_values[3] = {1.0f, 2.0f, 3.0f};
    float softmax_expected[3] = {1.0f, 2.0f, 3.0f};

    const float swiglu_gate[3] = {-1.0f, 0.0f, 2.0f};
    const float swiglu_value[3] = {3.0f, 4.0f, -2.0f};
    float swiglu_out[3] = {0.0f};
    float swiglu_expected[3] = {0.0f};

    float q8_lhs[3] = {1.0f, -2.0f, 3.0f};
    att1_q8_matrix q8_weights;
    float q8_out[2] = {0.0f};
    float q8_expected[2] = {0.0f};

    float q4_lhs[16] = {
        1.0f, -2.0f, 3.0f, 0.5f, -1.0f, 2.0f, 0.0f, 1.5f,
        -0.5f, 1.0f, 2.5f, -1.5f, 0.25f, -0.25f, 1.25f, -1.25f
    };
    const float q4_source_weights[32] = {
        1.0f, -1.0f, 2.0f, 0.5f, -2.0f, 1.0f, 0.0f, 3.0f,
        0.5f, -0.5f, 1.5f, -1.0f, 2.0f, -2.0f, 1.0f, 0.5f,
        -2.0f, 1.0f, 0.0f, 3.0f, 1.0f, -1.0f, 2.0f, 0.5f,
        2.0f, -2.0f, 1.0f, 0.5f, 0.5f, -0.5f, 1.5f, -1.0f
    };
    att1_q4_matrix q4_weights;
    float q4_out[2] = {0.0f};
    float q4_expected[2] = {0.0f};

    REQUIRE(att1_aimu_conformance_inproc_create(NULL, &endpoint) == ATT1_OK,
            "exec_ops: endpoint create");
    REQUIRE(att1_backend_pcie_create(endpoint, 0u, &backend) == ATT1_OK,
            "exec_ops: backend create");

    /* EXEC_MATMUL (f32): weight tensor auto-registered by pointer identity
     * on first use (LOAD_TENSOR_TILE), then referenced by tensor_id. */
    REQUIRE(backend->ops->matmul_f32(backend, matmul_out, lhs3, rhs3x2, 1u, 2u, 3u) == 0,
            "exec_ops: matmul_f32 succeeds");
    REQUIRE(att1_matmul_f32(matmul_expected, lhs3, rhs3x2, 1u, 2u, 3u) == 0,
            "exec_ops: matmul_f32 reference computed");
    REQUIRE(near_f32(matmul_out[0], matmul_expected[0]) &&
            near_f32(matmul_out[1], matmul_expected[1]),
            "exec_ops: matmul_f32 matches CPU reference exactly");

    /* EXEC_RMSNORM */
    REQUIRE(backend->ops->rmsnorm_f32(backend, rms_out, rms_src, rms_weight, 4u, 1e-6f) == 0,
            "exec_ops: rmsnorm_f32 succeeds");
    REQUIRE(att1_rmsnorm_f32(rms_expected, rms_src, rms_weight, 4u, 1e-6f) == 0,
            "exec_ops: rmsnorm_f32 reference computed");
    REQUIRE(near_f32(rms_out[0], rms_expected[0]) && near_f32(rms_out[1], rms_expected[1]) &&
            near_f32(rms_out[2], rms_expected[2]) && near_f32(rms_out[3], rms_expected[3]),
            "exec_ops: rmsnorm_f32 matches CPU reference exactly");

    /* EXEC_ROPE (in-place) */
    REQUIRE(backend->ops->rope_f32(backend, rope_values, 4u, 1u, 10000.0f) == 0,
            "exec_ops: rope_f32 succeeds");
    REQUIRE(att1_rope_f32(rope_expected, 4u, 1u, 10000.0f) == 0,
            "exec_ops: rope_f32 reference computed");
    REQUIRE(near_f32(rope_values[0], rope_expected[0]) &&
            near_f32(rope_values[1], rope_expected[1]) &&
            near_f32(rope_values[2], rope_expected[2]) &&
            near_f32(rope_values[3], rope_expected[3]),
            "exec_ops: rope_f32 matches CPU reference exactly");

    /* softmax_f32: computed locally (no frozen command type). */
    REQUIRE(backend->ops->softmax_f32(backend, softmax_values, 3u) == 0,
            "exec_ops: softmax_f32 succeeds");
    REQUIRE(att1_softmax_f32(softmax_expected, 3u) == 0,
            "exec_ops: softmax_f32 reference computed");
    REQUIRE(near_f32(softmax_values[0], softmax_expected[0]) &&
            near_f32(softmax_values[1], softmax_expected[1]) &&
            near_f32(softmax_values[2], softmax_expected[2]),
            "exec_ops: softmax_f32 matches CPU reference exactly");

    /* EXEC_FFN (swiglu combine step) */
    REQUIRE(backend->ops->ffn_swiglu_f32(backend, swiglu_out, swiglu_gate, swiglu_value, 3u) == 0,
            "exec_ops: ffn_swiglu_f32 succeeds");
    REQUIRE(att1_swiglu_f32(swiglu_expected, swiglu_gate, swiglu_value, 3u) == 0,
            "exec_ops: ffn_swiglu_f32 reference computed");
    REQUIRE(near_f32(swiglu_out[0], swiglu_expected[0]) &&
            near_f32(swiglu_out[1], swiglu_expected[1]) &&
            near_f32(swiglu_out[2], swiglu_expected[2]),
            "exec_ops: ffn_swiglu_f32 matches CPU reference exactly");

    /* EXEC_MATMUL (q8) */
    REQUIRE(att1_q8_matrix_alloc(&q8_weights, 2u, 3u) == 0,
            "exec_ops: q8 matrix alloc");
    q8_weights.values[0] = 10; q8_weights.values[1] = -20; q8_weights.values[2] = 30;
    q8_weights.values[3] = 5;  q8_weights.values[4] = 15;  q8_weights.values[5] = -25;
    q8_weights.scales[0] = 0.1f;
    q8_weights.scales[1] = 0.2f;
    REQUIRE(backend->ops->matmul_q8xf32(backend, q8_out, q8_lhs, 1u, 3u, &q8_weights) == 0,
            "exec_ops: matmul_q8xf32 succeeds");
    REQUIRE(att1_matmul_q8xf32(q8_expected, q8_lhs, 1u, 3u, &q8_weights) == 0,
            "exec_ops: matmul_q8xf32 reference computed");
    REQUIRE(near_f32(q8_out[0], q8_expected[0]) && near_f32(q8_out[1], q8_expected[1]),
            "exec_ops: matmul_q8xf32 matches CPU reference exactly");
    att1_q8_matrix_free(&q8_weights);

    /* EXEC_MATMUL (q4) */
    REQUIRE(att1_quantize_q4_per_group(&q4_weights, q4_source_weights, 2u, 16u, 16u) == 0,
            "exec_ops: q4 matrix quantized");
    REQUIRE(backend->ops->matmul_q4xf32(backend, q4_out, q4_lhs, 1u, 16u, &q4_weights) == 0,
            "exec_ops: matmul_q4xf32 succeeds");
    REQUIRE(att1_matmul_q4xf32(q4_expected, q4_lhs, 1u, 16u, &q4_weights) == 0,
            "exec_ops: matmul_q4xf32 reference computed");
    REQUIRE(near_f32(q4_out[0], q4_expected[0]) && near_f32(q4_out[1], q4_expected[1]),
            "exec_ops: matmul_q4xf32 matches CPU reference exactly");
    att1_q4_matrix_free(&q4_weights);

    REQUIRE(att1_aimu_conformance_cmd_get_counters(endpoint, &counters) == ATT1_OK &&
            counters.commands_failed == 0u &&
            counters.unsupported_commands == 0u,
            "exec_ops: cmdq counters show no failed/unsupported commands");

    att1_backend_destroy(backend);
    att1_aimu_conformance_endpoint_destroy(endpoint);
    PASS("exec_ops_correctness");
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

/*
 * M166: single-tile emulated decode, end to end.
 *
 * Runs a full transformer decoder block (pre-attention RMSNorm, causal
 * self-attention with RoPE, residual add, pre-FFN RMSNorm, SwiGLU FFN,
 * residual add) for three sequential decode steps against a tiny fixture,
 * once through the pcie backend and once through the corresponding cpu
 * backend, and requires the outputs to match at every step (f32 exactly,
 * q8/q4 within tolerance) — proving compute correctness, not just
 * transport plumbing, and exercising multi-step KV-cache-carrying decode.
 */
#define M166_MODEL_DIM 4u
#define M166_NUM_HEADS 2u
#define M166_HEAD_DIM  2u
#define M166_FFN_DIM   4u
#define M166_STEPS     3u
/* q4 requires cols divisible by a group size >= ATT1_Q4_GROUP_SIZE_MIN (16),
 * so the q4 fixture uses larger (but still tiny) dimensions. */
#define M166_Q4_MODEL_DIM 16u
#define M166_Q4_NUM_HEADS 4u
#define M166_Q4_HEAD_DIM  4u
#define M166_Q4_FFN_DIM   16u
#define M166_Q4_GROUP     16u

static void m166_fill_pattern(float *arr, size_t n, float seed)
{
    size_t i;

    for (i = 0u; i < n; i++) {
        arr[i] = sinf(seed + ((float)i * 0.37f)) * 0.5f;
    }
}

static int test_transformer_block_decode_f32(void)
{
    att1_aimu_conformance_endpoint *endpoint = NULL;
    att1_backend *pcie_backend = NULL;
    att1_backend *cpu_backend = NULL;
    att1_kv_cache pcie_cache;
    att1_kv_cache cpu_cache;
    att1_transformer_block_config config;
    att1_transformer_block_weights weights;
    float attention_norm[M166_MODEL_DIM];
    float ffn_norm[M166_MODEL_DIM];
    float wq[M166_MODEL_DIM * M166_MODEL_DIM];
    float wk[M166_MODEL_DIM * M166_MODEL_DIM];
    float wv[M166_MODEL_DIM * M166_MODEL_DIM];
    float wo[M166_MODEL_DIM * M166_MODEL_DIM];
    float w_gate[M166_MODEL_DIM * M166_FFN_DIM];
    float w_up[M166_MODEL_DIM * M166_FFN_DIM];
    float w_down[M166_FFN_DIM * M166_MODEL_DIM];
    size_t step;

    memset(&pcie_cache, 0, sizeof(pcie_cache));
    memset(&cpu_cache, 0, sizeof(cpu_cache));

    m166_fill_pattern(attention_norm, M166_MODEL_DIM, 0.1f);
    m166_fill_pattern(ffn_norm, M166_MODEL_DIM, 0.2f);
    m166_fill_pattern(wq, M166_MODEL_DIM * M166_MODEL_DIM, 0.3f);
    m166_fill_pattern(wk, M166_MODEL_DIM * M166_MODEL_DIM, 0.4f);
    m166_fill_pattern(wv, M166_MODEL_DIM * M166_MODEL_DIM, 0.5f);
    m166_fill_pattern(wo, M166_MODEL_DIM * M166_MODEL_DIM, 0.6f);
    m166_fill_pattern(w_gate, M166_MODEL_DIM * M166_FFN_DIM, 0.7f);
    m166_fill_pattern(w_up, M166_MODEL_DIM * M166_FFN_DIM, 0.8f);
    m166_fill_pattern(w_down, M166_FFN_DIM * M166_MODEL_DIM, 0.9f);

    /* Bias attention_norm/ffn_norm away from zero so RMSNorm is well
     * defined for every input. */
    for (step = 0u; step < M166_MODEL_DIM; step++) {
        attention_norm[step] += 1.0f;
        ffn_norm[step] += 1.0f;
    }

    config.model_dim = M166_MODEL_DIM;
    config.num_heads = M166_NUM_HEADS;
    config.head_dim = M166_HEAD_DIM;
    config.ffn_dim = M166_FFN_DIM;
    config.rms_epsilon = 1e-6f;
    config.rope_theta = 10000.0f;

    weights.attention_norm = attention_norm;
    weights.ffn_norm = ffn_norm;
    weights.wq = wq;
    weights.wk = wk;
    weights.wv = wv;
    weights.wo = wo;
    weights.w_gate = w_gate;
    weights.w_up = w_up;
    weights.w_down = w_down;

    REQUIRE(att1_aimu_conformance_inproc_create(NULL, &endpoint) == ATT1_OK,
            "transformer_block_f32: endpoint create");
    REQUIRE(att1_backend_pcie_create(endpoint, 0u, &pcie_backend) == ATT1_OK,
            "transformer_block_f32: pcie backend create");
    REQUIRE(att1_backend_cpu_f32_create(&cpu_backend) == ATT1_OK,
            "transformer_block_f32: cpu-f32 backend create");
    REQUIRE(att1_kv_cache_init(&pcie_cache, M166_STEPS, M166_NUM_HEADS, M166_HEAD_DIM) == ATT1_OK,
            "transformer_block_f32: pcie kv cache init");
    REQUIRE(att1_kv_cache_init(&cpu_cache, M166_STEPS, M166_NUM_HEADS, M166_HEAD_DIM) == ATT1_OK,
            "transformer_block_f32: cpu kv cache init");

    for (step = 0u; step < M166_STEPS; step++) {
        float input[M166_MODEL_DIM];
        float pcie_output[M166_MODEL_DIM];
        float cpu_output[M166_MODEL_DIM];

        m166_fill_pattern(input, M166_MODEL_DIM, 1.0f + (float)step);

        REQUIRE(att1_transformer_block_forward_backend(pcie_output, &pcie_cache, input,
                                                        &weights, &config, step,
                                                        pcie_backend) == 0,
                "transformer_block_f32: pcie forward succeeds");
        REQUIRE(att1_transformer_block_forward_backend(cpu_output, &cpu_cache, input,
                                                        &weights, &config, step,
                                                        cpu_backend) == 0,
                "transformer_block_f32: cpu forward succeeds");

        REQUIRE(near_f32(pcie_output[0], cpu_output[0]) &&
                near_f32(pcie_output[1], cpu_output[1]) &&
                near_f32(pcie_output[2], cpu_output[2]) &&
                near_f32(pcie_output[3], cpu_output[3]),
                "transformer_block_f32: pcie output matches cpu-f32 exactly");
    }

    att1_kv_cache_free(&pcie_cache);
    att1_kv_cache_free(&cpu_cache);
    att1_backend_destroy(pcie_backend);
    att1_backend_destroy(cpu_backend);
    att1_aimu_conformance_endpoint_destroy(endpoint);
    PASS("transformer_block_decode_f32");
    return 0;
}

static int test_transformer_block_decode_q8(void)
{
    att1_aimu_conformance_endpoint *endpoint = NULL;
    att1_backend *pcie_backend = NULL;
    att1_backend *cpu_backend = NULL;
    att1_kv_cache pcie_cache;
    att1_kv_cache cpu_cache;
    att1_transformer_block_config config;
    att1_transformer_block_q8_weights weights;
    float attention_norm[M166_MODEL_DIM];
    float ffn_norm[M166_MODEL_DIM];
    float wq_src[M166_MODEL_DIM * M166_MODEL_DIM];
    float wk_src[M166_MODEL_DIM * M166_MODEL_DIM];
    float wv_src[M166_MODEL_DIM * M166_MODEL_DIM];
    float wo_src[M166_MODEL_DIM * M166_MODEL_DIM];
    float w_gate_src[M166_FFN_DIM * M166_MODEL_DIM];
    float w_up_src[M166_FFN_DIM * M166_MODEL_DIM];
    float w_down_src[M166_MODEL_DIM * M166_FFN_DIM];
    att1_q8_matrix wq, wk, wv, wo, w_gate, w_up, w_down;
    size_t step;

    memset(&pcie_cache, 0, sizeof(pcie_cache));
    memset(&cpu_cache, 0, sizeof(cpu_cache));

    m166_fill_pattern(attention_norm, M166_MODEL_DIM, 0.1f);
    m166_fill_pattern(ffn_norm, M166_MODEL_DIM, 0.2f);
    m166_fill_pattern(wq_src, M166_MODEL_DIM * M166_MODEL_DIM, 0.3f);
    m166_fill_pattern(wk_src, M166_MODEL_DIM * M166_MODEL_DIM, 0.4f);
    m166_fill_pattern(wv_src, M166_MODEL_DIM * M166_MODEL_DIM, 0.5f);
    m166_fill_pattern(wo_src, M166_MODEL_DIM * M166_MODEL_DIM, 0.6f);
    /* w_gate/w_up: rows=ffn_dim(out), cols=model_dim(in). w_down: rows=
     * model_dim(out), cols=ffn_dim(in) — matching src/transformer_block.c's
     * matmul_q8xf32 call shapes. */
    m166_fill_pattern(w_gate_src, M166_FFN_DIM * M166_MODEL_DIM, 0.7f);
    m166_fill_pattern(w_up_src, M166_FFN_DIM * M166_MODEL_DIM, 0.8f);
    m166_fill_pattern(w_down_src, M166_MODEL_DIM * M166_FFN_DIM, 0.9f);

    for (step = 0u; step < M166_MODEL_DIM; step++) {
        attention_norm[step] += 1.0f;
        ffn_norm[step] += 1.0f;
    }

    REQUIRE(att1_quantize_q8_per_row(&wq, wq_src, M166_MODEL_DIM, M166_MODEL_DIM) == 0,
            "transformer_block_q8: wq quantized");
    REQUIRE(att1_quantize_q8_per_row(&wk, wk_src, M166_MODEL_DIM, M166_MODEL_DIM) == 0,
            "transformer_block_q8: wk quantized");
    REQUIRE(att1_quantize_q8_per_row(&wv, wv_src, M166_MODEL_DIM, M166_MODEL_DIM) == 0,
            "transformer_block_q8: wv quantized");
    REQUIRE(att1_quantize_q8_per_row(&wo, wo_src, M166_MODEL_DIM, M166_MODEL_DIM) == 0,
            "transformer_block_q8: wo quantized");
    REQUIRE(att1_quantize_q8_per_row(&w_gate, w_gate_src, M166_FFN_DIM, M166_MODEL_DIM) == 0,
            "transformer_block_q8: w_gate quantized");
    REQUIRE(att1_quantize_q8_per_row(&w_up, w_up_src, M166_FFN_DIM, M166_MODEL_DIM) == 0,
            "transformer_block_q8: w_up quantized");
    REQUIRE(att1_quantize_q8_per_row(&w_down, w_down_src, M166_MODEL_DIM, M166_FFN_DIM) == 0,
            "transformer_block_q8: w_down quantized");

    config.model_dim = M166_MODEL_DIM;
    config.num_heads = M166_NUM_HEADS;
    config.head_dim = M166_HEAD_DIM;
    config.ffn_dim = M166_FFN_DIM;
    config.rms_epsilon = 1e-6f;
    config.rope_theta = 10000.0f;

    weights.attention_norm = attention_norm;
    weights.ffn_norm = ffn_norm;
    weights.wq = &wq;
    weights.wk = &wk;
    weights.wv = &wv;
    weights.wo = &wo;
    weights.w_gate = &w_gate;
    weights.w_up = &w_up;
    weights.w_down = &w_down;

    REQUIRE(att1_aimu_conformance_inproc_create(NULL, &endpoint) == ATT1_OK,
            "transformer_block_q8: endpoint create");
    REQUIRE(att1_backend_pcie_create(endpoint, 0u, &pcie_backend) == ATT1_OK,
            "transformer_block_q8: pcie backend create");
    REQUIRE(att1_backend_cpu_q8_create(&cpu_backend) == ATT1_OK,
            "transformer_block_q8: cpu-q8 backend create");
    REQUIRE(att1_kv_cache_init(&pcie_cache, M166_STEPS, M166_NUM_HEADS, M166_HEAD_DIM) == ATT1_OK,
            "transformer_block_q8: pcie kv cache init");
    REQUIRE(att1_kv_cache_init(&cpu_cache, M166_STEPS, M166_NUM_HEADS, M166_HEAD_DIM) == ATT1_OK,
            "transformer_block_q8: cpu kv cache init");

    for (step = 0u; step < M166_STEPS; step++) {
        float input[M166_MODEL_DIM];
        float pcie_output[M166_MODEL_DIM];
        float cpu_output[M166_MODEL_DIM];

        m166_fill_pattern(input, M166_MODEL_DIM, 1.0f + (float)step);

        REQUIRE(att1_transformer_block_forward_backend_q8(pcie_output, &pcie_cache, input,
                                                           &weights, &config, step,
                                                           pcie_backend) == 0,
                "transformer_block_q8: pcie forward succeeds");
        REQUIRE(att1_transformer_block_forward_backend_q8(cpu_output, &cpu_cache, input,
                                                           &weights, &config, step,
                                                           cpu_backend) == 0,
                "transformer_block_q8: cpu forward succeeds");

        REQUIRE(within_quant_tolerance(pcie_output[0], cpu_output[0]) &&
                within_quant_tolerance(pcie_output[1], cpu_output[1]) &&
                within_quant_tolerance(pcie_output[2], cpu_output[2]) &&
                within_quant_tolerance(pcie_output[3], cpu_output[3]),
                "transformer_block_q8: pcie output matches cpu-q8 within tolerance");
    }

    att1_kv_cache_free(&pcie_cache);
    att1_kv_cache_free(&cpu_cache);
    att1_q8_matrix_free(&wq);
    att1_q8_matrix_free(&wk);
    att1_q8_matrix_free(&wv);
    att1_q8_matrix_free(&wo);
    att1_q8_matrix_free(&w_gate);
    att1_q8_matrix_free(&w_up);
    att1_q8_matrix_free(&w_down);
    att1_backend_destroy(pcie_backend);
    att1_backend_destroy(cpu_backend);
    att1_aimu_conformance_endpoint_destroy(endpoint);
    PASS("transformer_block_decode_q8");
    return 0;
}

static int test_transformer_block_decode_q4(void)
{
    att1_aimu_conformance_endpoint *endpoint = NULL;
    att1_backend *pcie_backend = NULL;
    att1_backend *cpu_backend = NULL;
    att1_kv_cache pcie_cache;
    att1_kv_cache cpu_cache;
    att1_transformer_block_config config;
    att1_transformer_block_q4_weights weights;
    float attention_norm[M166_Q4_MODEL_DIM];
    float ffn_norm[M166_Q4_MODEL_DIM];
    float wq_src[M166_Q4_MODEL_DIM * M166_Q4_MODEL_DIM];
    float wk_src[M166_Q4_MODEL_DIM * M166_Q4_MODEL_DIM];
    float wv_src[M166_Q4_MODEL_DIM * M166_Q4_MODEL_DIM];
    float wo_src[M166_Q4_MODEL_DIM * M166_Q4_MODEL_DIM];
    float w_gate_src[M166_Q4_FFN_DIM * M166_Q4_MODEL_DIM];
    float w_up_src[M166_Q4_FFN_DIM * M166_Q4_MODEL_DIM];
    float w_down_src[M166_Q4_MODEL_DIM * M166_Q4_FFN_DIM];
    att1_q4_matrix wq, wk, wv, wo, w_gate, w_up, w_down;
    size_t step;

    memset(&pcie_cache, 0, sizeof(pcie_cache));
    memset(&cpu_cache, 0, sizeof(cpu_cache));

    m166_fill_pattern(attention_norm, M166_Q4_MODEL_DIM, 0.1f);
    m166_fill_pattern(ffn_norm, M166_Q4_MODEL_DIM, 0.2f);
    m166_fill_pattern(wq_src, M166_Q4_MODEL_DIM * M166_Q4_MODEL_DIM, 0.3f);
    m166_fill_pattern(wk_src, M166_Q4_MODEL_DIM * M166_Q4_MODEL_DIM, 0.4f);
    m166_fill_pattern(wv_src, M166_Q4_MODEL_DIM * M166_Q4_MODEL_DIM, 0.5f);
    m166_fill_pattern(wo_src, M166_Q4_MODEL_DIM * M166_Q4_MODEL_DIM, 0.6f);
    m166_fill_pattern(w_gate_src, M166_Q4_FFN_DIM * M166_Q4_MODEL_DIM, 0.7f);
    m166_fill_pattern(w_up_src, M166_Q4_FFN_DIM * M166_Q4_MODEL_DIM, 0.8f);
    m166_fill_pattern(w_down_src, M166_Q4_MODEL_DIM * M166_Q4_FFN_DIM, 0.9f);

    for (step = 0u; step < M166_Q4_MODEL_DIM; step++) {
        attention_norm[step] += 1.0f;
        ffn_norm[step] += 1.0f;
    }

    REQUIRE(att1_quantize_q4_per_group(&wq, wq_src, M166_Q4_MODEL_DIM, M166_Q4_MODEL_DIM, 16u) == 0,
            "transformer_block_q4: wq quantized");
    REQUIRE(att1_quantize_q4_per_group(&wk, wk_src, M166_Q4_MODEL_DIM, M166_Q4_MODEL_DIM, 16u) == 0,
            "transformer_block_q4: wk quantized");
    REQUIRE(att1_quantize_q4_per_group(&wv, wv_src, M166_Q4_MODEL_DIM, M166_Q4_MODEL_DIM, 16u) == 0,
            "transformer_block_q4: wv quantized");
    REQUIRE(att1_quantize_q4_per_group(&wo, wo_src, M166_Q4_MODEL_DIM, M166_Q4_MODEL_DIM, 16u) == 0,
            "transformer_block_q4: wo quantized");
    REQUIRE(att1_quantize_q4_per_group(&w_gate, w_gate_src, M166_Q4_FFN_DIM, M166_Q4_MODEL_DIM, 16u) == 0,
            "transformer_block_q4: w_gate quantized");
    REQUIRE(att1_quantize_q4_per_group(&w_up, w_up_src, M166_Q4_FFN_DIM, M166_Q4_MODEL_DIM, 16u) == 0,
            "transformer_block_q4: w_up quantized");
    REQUIRE(att1_quantize_q4_per_group(&w_down, w_down_src, M166_Q4_MODEL_DIM, M166_Q4_FFN_DIM, 16u) == 0,
            "transformer_block_q4: w_down quantized");

    config.model_dim = M166_Q4_MODEL_DIM;
    config.num_heads = M166_Q4_NUM_HEADS;
    config.head_dim = M166_Q4_HEAD_DIM;
    config.ffn_dim = M166_Q4_FFN_DIM;
    config.rms_epsilon = 1e-6f;
    config.rope_theta = 10000.0f;

    weights.attention_norm = attention_norm;
    weights.ffn_norm = ffn_norm;
    weights.wq = &wq;
    weights.wk = &wk;
    weights.wv = &wv;
    weights.wo = &wo;
    weights.w_gate = &w_gate;
    weights.w_up = &w_up;
    weights.w_down = &w_down;

    REQUIRE(att1_aimu_conformance_inproc_create(NULL, &endpoint) == ATT1_OK,
            "transformer_block_q4: endpoint create");
    REQUIRE(att1_backend_pcie_create(endpoint, 0u, &pcie_backend) == ATT1_OK,
            "transformer_block_q4: pcie backend create");
    REQUIRE(att1_backend_cpu_q4_create(&cpu_backend) == ATT1_OK,
            "transformer_block_q4: cpu-q4 backend create");
    REQUIRE(att1_kv_cache_init(&pcie_cache, M166_STEPS, M166_Q4_NUM_HEADS, M166_Q4_HEAD_DIM) == ATT1_OK,
            "transformer_block_q4: pcie kv cache init");
    REQUIRE(att1_kv_cache_init(&cpu_cache, M166_STEPS, M166_Q4_NUM_HEADS, M166_Q4_HEAD_DIM) == ATT1_OK,
            "transformer_block_q4: cpu kv cache init");

    for (step = 0u; step < M166_STEPS; step++) {
        float input[M166_Q4_MODEL_DIM];
        float pcie_output[M166_Q4_MODEL_DIM];
        float cpu_output[M166_Q4_MODEL_DIM];

        m166_fill_pattern(input, M166_Q4_MODEL_DIM, 1.0f + (float)step);

        REQUIRE(att1_transformer_block_forward_backend_q4(pcie_output, &pcie_cache, input,
                                                           &weights, &config, step,
                                                           pcie_backend) == 0,
                "transformer_block_q4: pcie forward succeeds");
        REQUIRE(att1_transformer_block_forward_backend_q4(cpu_output, &cpu_cache, input,
                                                           &weights, &config, step,
                                                           cpu_backend) == 0,
                "transformer_block_q4: cpu forward succeeds");

        REQUIRE(within_quant_tolerance(pcie_output[0], cpu_output[0]) &&
                within_quant_tolerance(pcie_output[1], cpu_output[1]) &&
                within_quant_tolerance(pcie_output[2], cpu_output[2]) &&
                within_quant_tolerance(pcie_output[3], cpu_output[3]),
                "transformer_block_q4: pcie output matches cpu-q4 within tolerance");
    }

    att1_kv_cache_free(&pcie_cache);
    att1_kv_cache_free(&cpu_cache);
    att1_q4_matrix_free(&wq);
    att1_q4_matrix_free(&wk);
    att1_q4_matrix_free(&wv);
    att1_q4_matrix_free(&wo);
    att1_q4_matrix_free(&w_gate);
    att1_q4_matrix_free(&w_up);
    att1_q4_matrix_free(&w_down);
    att1_backend_destroy(pcie_backend);
    att1_backend_destroy(cpu_backend);
    att1_aimu_conformance_endpoint_destroy(endpoint);
    PASS("transformer_block_decode_q4");
    return 0;
}

int main(void)
{
    int failed = 0;

    failed |= test_create_invalid_args();
    failed |= test_alloc_free_sync();
    failed |= test_exec_ops_correctness();
    failed |= test_load_tensor_residency();
    failed |= test_transformer_block_decode_f32();
    failed |= test_transformer_block_decode_q8();
    failed |= test_transformer_block_decode_q4();

    if (failed) {
        printf("backend_pcie: SOME TESTS FAILED\n");
        return 1;
    }

    printf("backend_pcie: ALL TESTS PASSED\n");
    return 0;
}
