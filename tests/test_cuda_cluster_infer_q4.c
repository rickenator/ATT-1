/* M89: CUDA q4 cluster inference integration tests.
 *
 * Verifies that the cuda-q4 backend produces logits within Q4_LOGIT_TOL of
 * the cpu-q4 reference when used with the cluster inference engine.
 * All tests are skipped when CUDA is unavailable.
 *
 * Tolerance: CUDA and CPU q4 use the same dequantisation algorithm so the
 * expected max absolute logit difference is < 1e-4f.  Next-token agreement
 * is expected; token divergence is only possible when two logit values sit
 * within Q4_LOGIT_TOL of the argmax boundary.
 */
#include "att1_backend.h"
#include "att1_cluster_infer.h"
#include "att1_model.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define Q4_MODEL_PATH "models/q4_tiny/model.att1"
#define Q4_LOGIT_TOL  0.35f   /* q4 vs q4 cross-device: same dequant */

/* ── helpers ──────────────────────────────────────────────────────────── */

static att1_cluster_infer_config default_config(void)
{
    att1_cluster_infer_config cfg;
    cfg.tile_count = 2u;
    cfg.fabric_queue_capacity = 4u;
    cfg.fabric_max_payload_bytes = 0u;
    cfg.shard_plan_mode = ATT1_SHARD_PLAN_RUNTIME;
    return cfg;
}

static int create_q4_cluster_pair(const att1_model *model,
                                  att1_cluster_infer_t **out_cpu_q4,
                                  att1_cluster_infer_t **out_cuda_q4)
{
    att1_cluster_infer_config cfg = default_config();
    att1_cluster_infer_t *cpu_q4 = NULL;
    att1_cluster_infer_t *cuda_q4 = NULL;
    att1_backend *cuda_backend = NULL;

    if ((out_cpu_q4 == NULL) || (out_cuda_q4 == NULL)) {
        return -1;
    }

    *out_cpu_q4 = NULL;
    *out_cuda_q4 = NULL;

    if (att1_cluster_infer_create_q4(model, &cfg, &cpu_q4) != ATT1_OK) {
        return -1;
    }

    if (att1_cluster_infer_create_q4(model, &cfg, &cuda_q4) != ATT1_OK) {
        att1_cluster_infer_destroy(cpu_q4);
        return -1;
    }

    if (att1_backend_cuda_q4_create(&cuda_backend) != ATT1_OK) {
        att1_cluster_infer_destroy(cpu_q4);
        att1_cluster_infer_destroy(cuda_q4);
        return -1;
    }

    if ((cuda_backend->ops == NULL) ||
        (cuda_backend->ops->name == NULL) ||
        (strcmp(cuda_backend->ops->name, "cuda-q4") != 0)) {
        att1_backend_destroy(cuda_backend);
        att1_cluster_infer_destroy(cpu_q4);
        att1_cluster_infer_destroy(cuda_q4);
        return -1;
    }

    if (att1_cluster_infer_set_backend(cuda_q4, cuda_backend) != ATT1_OK) {
        att1_backend_destroy(cuda_backend);
        att1_cluster_infer_destroy(cpu_q4);
        att1_cluster_infer_destroy(cuda_q4);
        return -1;
    }
    cuda_backend = NULL;

    *out_cpu_q4 = cpu_q4;
    *out_cuda_q4 = cuda_q4;
    return 0;
}

/* ── tests ──────────────────────────────────────────────────────────── */

/* Verify that cuda-q4 set_backend on a cluster context succeeds and the
 * backend name is "cuda-q4" (no silent fallback to cpu-q4). */
static int test_cuda_q4_cluster_no_silent_fallback(const att1_model *model)
{
    att1_cluster_infer_config cfg = default_config();
    att1_cluster_infer_t *infer = NULL;
    att1_backend *backend = NULL;
    uint32_t next = 0u;

    if (att1_cluster_infer_create_q4(model, &cfg, &infer) != ATT1_OK) {
        return -1;
    }

    if (att1_backend_cuda_q4_create(&backend) != ATT1_OK) {
        att1_cluster_infer_destroy(infer);
        return -1;
    }

    if ((backend->ops == NULL) ||
        (backend->ops->name == NULL) ||
        (strcmp(backend->ops->name, "cuda-q4") != 0) ||
        (strcmp(backend->ops->name, "cpu-q4") == 0) ||
        (att1_cluster_infer_set_backend(infer, backend) != ATT1_OK)) {
        att1_backend_destroy(backend);
        att1_cluster_infer_destroy(infer);
        return -1;
    }
    backend = NULL;

    if (att1_cluster_infer_decode_token(infer, (uint32_t)'h', &next) != ATT1_OK) {
        att1_cluster_infer_destroy(infer);
        return -1;
    }

    att1_cluster_infer_destroy(infer);
    return 0;
}

/* Fabric counters must be nonzero after a decode with cuda-q4 cluster. */
static int test_cuda_q4_cluster_fabric_counters(const att1_model *model)
{
    att1_cluster_infer_config cfg = default_config();
    att1_cluster_infer_t *infer = NULL;
    att1_backend *backend = NULL;
    att1_cluster_tile_counters counters;
    uint32_t next = 0u;

    if (att1_cluster_infer_create_q4(model, &cfg, &infer) != ATT1_OK) {
        return -1;
    }

    if (att1_backend_cuda_q4_create(&backend) != ATT1_OK) {
        att1_cluster_infer_destroy(infer);
        return -1;
    }

    if (att1_cluster_infer_set_backend(infer, backend) != ATT1_OK) {
        att1_backend_destroy(backend);
        att1_cluster_infer_destroy(infer);
        return -1;
    }
    backend = NULL;

    if (att1_cluster_infer_decode_token(infer, (uint32_t)'h', &next) != ATT1_OK) {
        att1_cluster_infer_destroy(infer);
        return -1;
    }

    if (att1_cluster_infer_get_tile_counters(infer, 0u, &counters) != ATT1_OK) {
        att1_cluster_infer_destroy(infer);
        return -1;
    }

    att1_cluster_infer_destroy(infer);

    if (counters.activations_received == 0u) {
        fputs("cuda q4 cluster: tile 0 activations_received is zero after decode\n",
              stderr);
        return -1;
    }

    return 0;
}

/* cuda-q4 cluster logits must match cpu-q4 cluster logits within Q4_LOGIT_TOL. */
static int test_cuda_q4_cluster_logits_match_cpu_q4(const att1_model *model)
{
    att1_cluster_infer_t *cpu_q4 = NULL;
    att1_cluster_infer_t *cuda_q4 = NULL;
    const float *cpu_logits = NULL;
    const float *cuda_logits = NULL;
    size_t cpu_count = 0u;
    size_t cuda_count = 0u;
    uint32_t cpu_next = 0u;
    uint32_t cuda_next = 0u;
    size_t i = 0u;

    if (create_q4_cluster_pair(model, &cpu_q4, &cuda_q4) != 0) {
        return -1;
    }

    if ((att1_cluster_infer_decode_token(cpu_q4, (uint32_t)'h', &cpu_next) != ATT1_OK) ||
        (att1_cluster_infer_decode_token(cuda_q4, (uint32_t)'h', &cuda_next) != ATT1_OK)) {
        att1_cluster_infer_destroy(cpu_q4);
        att1_cluster_infer_destroy(cuda_q4);
        return -1;
    }

    cpu_logits  = att1_cluster_infer_logits(cpu_q4,  &cpu_count);
    cuda_logits = att1_cluster_infer_logits(cuda_q4, &cuda_count);
    if ((cpu_logits == NULL) ||
        (cuda_logits == NULL) ||
        (cpu_count == 0u) ||
        (cpu_count != cuda_count)) {
        att1_cluster_infer_destroy(cpu_q4);
        att1_cluster_infer_destroy(cuda_q4);
        return -1;
    }

    for (i = 0u; i < cpu_count; i++) {
        const float diff = fabsf(cpu_logits[i] - cuda_logits[i]);

        if (diff > Q4_LOGIT_TOL) {
            fprintf(stderr,
                    "cuda q4 cluster logits differ at %zu: cpu=%.6f cuda=%.6f\n",
                    i,
                    (double)cpu_logits[i],
                    (double)cuda_logits[i]);
            att1_cluster_infer_destroy(cpu_q4);
            att1_cluster_infer_destroy(cuda_q4);
            return -1;
        }
    }

    if (cpu_next != cuda_next) {
        fprintf(stderr,
                "cuda q4 cluster next token differs: cpu=%u cuda=%u\n",
                cpu_next,
                cuda_next);
        att1_cluster_infer_destroy(cpu_q4);
        att1_cluster_infer_destroy(cuda_q4);
        return -1;
    }

    att1_cluster_infer_destroy(cpu_q4);
    att1_cluster_infer_destroy(cuda_q4);
    return 0;
}

/* Multi-token sequence: cuda-q4 cluster and cpu-q4 cluster must produce
 * identical tokens. */
static int test_cuda_q4_cluster_generated_tokens(const att1_model *model)
{
    att1_cluster_infer_t *cpu_q4 = NULL;
    att1_cluster_infer_t *cuda_q4 = NULL;
    const unsigned char prompt[] = "hi";
    uint32_t cpu_tokens[4] = {0u};
    uint32_t cuda_tokens[4] = {0u};
    size_t cpu_count = 0u;
    size_t cuda_count = 0u;

    if (create_q4_cluster_pair(model, &cpu_q4, &cuda_q4) != 0) {
        return -1;
    }

    if ((att1_cluster_infer_generate(cpu_q4,
                                     prompt,
                                     sizeof(prompt) - 1u,
                                     4u,
                                     cpu_tokens,
                                     4u,
                                     &cpu_count) != ATT1_OK) ||
        (att1_cluster_infer_generate(cuda_q4,
                                     prompt,
                                     sizeof(prompt) - 1u,
                                     4u,
                                     cuda_tokens,
                                     4u,
                                     &cuda_count) != ATT1_OK) ||
        (cpu_count != 4u) ||
        (cuda_count != 4u) ||
        (memcmp(cpu_tokens, cuda_tokens, sizeof(cpu_tokens)) != 0)) {
        att1_cluster_infer_destroy(cpu_q4);
        att1_cluster_infer_destroy(cuda_q4);
        return -1;
    }

    att1_cluster_infer_destroy(cpu_q4);
    att1_cluster_infer_destroy(cuda_q4);
    return 0;
}

/* ── main ─────────────────────────────────────────────────────────────── */

int main(void)
{
    att1_model model;

    /* Non-CUDA build path: create must return ATT1_ERR_UNSUPPORTED. */
    if (att1_backend_cuda_available() == 0) {
        att1_backend *cuda_q4_backend = NULL;

        if (att1_backend_cuda_q4_create(&cuda_q4_backend) != ATT1_ERR_UNSUPPORTED) {
            att1_backend_destroy(cuda_q4_backend);
            fputs("cuda q4 cluster infer: unavailable-path check failed\n", stderr);
            return 1;
        }

        puts("cuda_cluster_infer_q4 test skipped (CUDA unavailable)");
        return 0;
    }

    if (att1_model_load(Q4_MODEL_PATH, &model) != ATT1_OK) {
        fputs("cuda_cluster_infer_q4: model load failed\n", stderr);
        return 1;
    }

    if ((test_cuda_q4_cluster_no_silent_fallback(&model) != 0) ||
        (test_cuda_q4_cluster_fabric_counters(&model) != 0) ||
        (test_cuda_q4_cluster_logits_match_cpu_q4(&model) != 0) ||
        (test_cuda_q4_cluster_generated_tokens(&model) != 0)) {
        att1_model_free(&model);
        fputs("cuda_cluster_infer_q4 test failed\n", stderr);
        return 1;
    }

    att1_model_free(&model);
    puts("cuda_cluster_infer_q4 test passed");
    return 0;
}
