#include "att1_backend.h"
#include "att1_infer.h"
#include "att1_model.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define MODEL_PATH "models/dummy/model.att1"
#define CUDA_Q8_LOGIT_TOL 1e-3f

static int create_infer_pair(const att1_model *model,
                             att1_infer_t **out_cpu,
                             att1_infer_t **out_cuda)
{
    att1_infer_t *cpu = NULL;
    att1_infer_t *cuda = NULL;
    att1_backend *cpu_backend = NULL;
    att1_backend *cuda_backend = NULL;

    if ((out_cpu == NULL) || (out_cuda == NULL)) {
        return -1;
    }

    *out_cpu = NULL;
    *out_cuda = NULL;

    if ((att1_infer_create(model, &cpu) != ATT1_OK) ||
        (att1_infer_create(model, &cuda) != ATT1_OK)) {
        att1_infer_destroy(cpu);
        att1_infer_destroy(cuda);
        return -1;
    }

    if ((att1_backend_cpu_f32_create(&cpu_backend) != ATT1_OK) ||
        (att1_infer_set_backend(cpu, cpu_backend) != ATT1_OK)) {
        att1_backend_destroy(cpu_backend);
        att1_infer_destroy(cpu);
        att1_infer_destroy(cuda);
        return -1;
    }
    cpu_backend = NULL;

    if (att1_backend_cuda_create(&cuda_backend) != ATT1_OK) {
        att1_infer_destroy(cpu);
        att1_infer_destroy(cuda);
        return -1;
    }

    if ((cuda_backend->ops == NULL) ||
        (cuda_backend->ops->name == NULL) ||
        (strcmp(cuda_backend->ops->name, "cuda") != 0)) {
        att1_backend_destroy(cuda_backend);
        att1_infer_destroy(cpu);
        att1_infer_destroy(cuda);
        return -1;
    }

    if (att1_infer_set_backend(cuda, cuda_backend) != ATT1_OK) {
        att1_backend_destroy(cuda_backend);
        att1_infer_destroy(cpu);
        att1_infer_destroy(cuda);
        return -1;
    }
    cuda_backend = NULL;

    *out_cpu = cpu;
    *out_cuda = cuda;
    return 0;
}

static int create_q8_infer_pair(const att1_model *model,
                                att1_infer_t **out_cpu_q8,
                                att1_infer_t **out_cuda_q8)
{
    att1_infer_t *cpu_q8 = NULL;
    att1_infer_t *cuda_q8 = NULL;
    att1_backend *cpu_backend = NULL;
    att1_backend *cuda_backend = NULL;

    if ((out_cpu_q8 == NULL) || (out_cuda_q8 == NULL)) {
        return -1;
    }

    *out_cpu_q8 = NULL;
    *out_cuda_q8 = NULL;

    if ((att1_infer_create(model, &cpu_q8) != ATT1_OK) ||
        (att1_infer_create(model, &cuda_q8) != ATT1_OK)) {
        att1_infer_destroy(cpu_q8);
        att1_infer_destroy(cuda_q8);
        return -1;
    }

    if ((att1_backend_cpu_q8_create(&cpu_backend) != ATT1_OK) ||
        (att1_infer_set_backend(cpu_q8, cpu_backend) != ATT1_OK)) {
        att1_backend_destroy(cpu_backend);
        att1_infer_destroy(cpu_q8);
        att1_infer_destroy(cuda_q8);
        return -1;
    }
    cpu_backend = NULL;

    if (att1_backend_cuda_q8_create(&cuda_backend) != ATT1_OK) {
        att1_infer_destroy(cpu_q8);
        att1_infer_destroy(cuda_q8);
        return -1;
    }

    if ((cuda_backend->ops == NULL) ||
        (cuda_backend->ops->name == NULL) ||
        (strcmp(cuda_backend->ops->name, "cuda-q8") != 0)) {
        att1_backend_destroy(cuda_backend);
        att1_infer_destroy(cpu_q8);
        att1_infer_destroy(cuda_q8);
        return -1;
    }

    if (att1_infer_set_backend(cuda_q8, cuda_backend) != ATT1_OK) {
        att1_backend_destroy(cuda_backend);
        att1_infer_destroy(cpu_q8);
        att1_infer_destroy(cuda_q8);
        return -1;
    }
    cuda_backend = NULL;

    *out_cpu_q8 = cpu_q8;
    *out_cuda_q8 = cuda_q8;
    return 0;
}

static int test_next_token_equivalence(const att1_model *model)
{
    att1_infer_t *cpu = NULL;
    att1_infer_t *cuda = NULL;
    uint32_t cpu_next = 0u;
    uint32_t cuda_next = 0u;

    if (create_infer_pair(model, &cpu, &cuda) != 0) {
        return -1;
    }

    if ((att1_infer_decode_token(cpu, (uint32_t)'h', &cpu_next) != ATT1_OK) ||
        (att1_infer_decode_token(cuda, (uint32_t)'h', &cuda_next) != ATT1_OK) ||
        (cpu_next != cuda_next)) {
        att1_infer_destroy(cpu);
        att1_infer_destroy(cuda);
        return -1;
    }

    att1_infer_destroy(cpu);
    att1_infer_destroy(cuda);
    return 0;
}

static int test_sequence_equivalence_2_to_4_tokens(const att1_model *model)
{
    att1_infer_t *cpu = NULL;
    att1_infer_t *cuda = NULL;
    const unsigned char prompt[] = "hi";
    uint32_t cpu_tokens[4] = {0u};
    uint32_t cuda_tokens[4] = {0u};
    size_t cpu_count = 0u;
    size_t cuda_count = 0u;

    if (create_infer_pair(model, &cpu, &cuda) != 0) {
        return -1;
    }

    if ((att1_infer_generate(cpu,
                             prompt,
                             sizeof(prompt) - 1u,
                             2u,
                             cpu_tokens,
                             4u,
                             &cpu_count) != ATT1_OK) ||
        (att1_infer_generate(cuda,
                             prompt,
                             sizeof(prompt) - 1u,
                             2u,
                             cuda_tokens,
                             4u,
                             &cuda_count) != ATT1_OK) ||
        (cpu_count != 2u) ||
        (cuda_count != 2u) ||
        (memcmp(cpu_tokens, cuda_tokens, 2u * sizeof(cpu_tokens[0])) != 0)) {
        att1_infer_destroy(cpu);
        att1_infer_destroy(cuda);
        return -1;
    }

    att1_infer_destroy(cpu);
    att1_infer_destroy(cuda);

    if (create_infer_pair(model, &cpu, &cuda) != 0) {
        return -1;
    }

    if ((att1_infer_generate(cpu,
                             prompt,
                             sizeof(prompt) - 1u,
                             4u,
                             cpu_tokens,
                             4u,
                             &cpu_count) != ATT1_OK) ||
        (att1_infer_generate(cuda,
                             prompt,
                             sizeof(prompt) - 1u,
                             4u,
                             cuda_tokens,
                             4u,
                             &cuda_count) != ATT1_OK) ||
        (cpu_count != 4u) ||
        (cuda_count != 4u) ||
        (memcmp(cpu_tokens, cuda_tokens, sizeof(cpu_tokens)) != 0)) {
        att1_infer_destroy(cpu);
        att1_infer_destroy(cuda);
        return -1;
    }

    att1_infer_destroy(cpu);
    att1_infer_destroy(cuda);
    return 0;
}

static int test_prompt_prefill_position_and_kv(const att1_model *model)
{
    att1_infer_t *cpu = NULL;
    att1_infer_t *cuda = NULL;
    const unsigned char prompt[] = "abc";
    size_t i = 0u;
    uint32_t layer = 0u;

    if (create_infer_pair(model, &cpu, &cuda) != 0) {
        return -1;
    }

    for (i = 0u; i < (sizeof(prompt) - 1u); i++) {
        uint32_t cpu_next = 0u;
        uint32_t cuda_next = 0u;
        size_t cpu_pos = 0u;
        size_t cuda_pos = 0u;

        if ((att1_infer_decode_token(cpu, (uint32_t)prompt[i], &cpu_next) != ATT1_OK) ||
            (att1_infer_decode_token(cuda, (uint32_t)prompt[i], &cuda_next) != ATT1_OK) ||
            (cpu_next != cuda_next) ||
            (att1_infer_position(cpu, &cpu_pos) != ATT1_OK) ||
            (att1_infer_position(cuda, &cuda_pos) != ATT1_OK) ||
            (cpu_pos != cuda_pos) ||
            (cpu_pos != (i + 1u))) {
            att1_infer_destroy(cpu);
            att1_infer_destroy(cuda);
            return -1;
        }

        for (layer = 0u; layer < model->config.n_layers; layer++) {
            size_t cpu_len = 0u;
            size_t cuda_len = 0u;

            if ((att1_infer_layer_kv_length(cpu, layer, &cpu_len) != ATT1_OK) ||
                (att1_infer_layer_kv_length(cuda, layer, &cuda_len) != ATT1_OK) ||
                (cpu_len != cuda_len) ||
                (cpu_len != (i + 1u))) {
                att1_infer_destroy(cpu);
                att1_infer_destroy(cuda);
                return -1;
            }
        }
    }

    att1_infer_destroy(cpu);
    att1_infer_destroy(cuda);
    return 0;
}

static int test_no_silent_cpu_fallback(const att1_model *model)
{
    att1_infer_t *cuda = NULL;
    att1_backend *cuda_backend = NULL;
    uint32_t next = 0u;

    if (att1_infer_create(model, &cuda) != ATT1_OK) {
        return -1;
    }

    if (att1_backend_cuda_create(&cuda_backend) != ATT1_OK) {
        att1_infer_destroy(cuda);
        return -1;
    }

    if ((cuda_backend->ops == NULL) ||
        (cuda_backend->ops->name == NULL) ||
        (strcmp(cuda_backend->ops->name, "cuda") != 0) ||
        (att1_infer_set_backend(cuda, cuda_backend) != ATT1_OK)) {
        att1_backend_destroy(cuda_backend);
        att1_infer_destroy(cuda);
        return -1;
    }
    cuda_backend = NULL;

    if (att1_infer_decode_token(cuda, (uint32_t)'z', &next) != ATT1_OK) {
        att1_infer_destroy(cuda);
        return -1;
    }

    att1_infer_destroy(cuda);
    return 0;
}

static int test_cuda_q8_logits_match_cpu_q8(const att1_model *model)
{
    att1_infer_t *cpu_q8 = NULL;
    att1_infer_t *cuda_q8 = NULL;
    const float *cpu_logits = NULL;
    const float *cuda_logits = NULL;
    size_t cpu_count = 0u;
    size_t cuda_count = 0u;
    uint32_t cpu_next = 0u;
    uint32_t cuda_next = 0u;
    size_t i = 0u;

    if (create_q8_infer_pair(model, &cpu_q8, &cuda_q8) != 0) {
        return -1;
    }

    if ((att1_infer_decode_token(cpu_q8, (uint32_t)'h', &cpu_next) != ATT1_OK) ||
        (att1_infer_decode_token(cuda_q8, (uint32_t)'h', &cuda_next) != ATT1_OK)) {
        att1_infer_destroy(cpu_q8);
        att1_infer_destroy(cuda_q8);
        return -1;
    }

    cpu_logits = att1_infer_logits(cpu_q8, &cpu_count);
    cuda_logits = att1_infer_logits(cuda_q8, &cuda_count);
    if ((cpu_logits == NULL) ||
        (cuda_logits == NULL) ||
        (cpu_count != model->config.vocab_size) ||
        (cuda_count != model->config.vocab_size)) {
        att1_infer_destroy(cpu_q8);
        att1_infer_destroy(cuda_q8);
        return -1;
    }

    for (i = 0u; i < cpu_count; i++) {
        const float diff = fabsf(cpu_logits[i] - cuda_logits[i]);

        if (diff > CUDA_Q8_LOGIT_TOL) {
            fprintf(stderr,
                    "cuda q8 logits differ at %zu: cpu=%.6f cuda=%.6f\n",
                    i,
                    (double)cpu_logits[i],
                    (double)cuda_logits[i]);
            att1_infer_destroy(cpu_q8);
            att1_infer_destroy(cuda_q8);
            return -1;
        }
    }

    if (cpu_next != cuda_next) {
        fprintf(stderr,
                "cuda q8 next token differs: cpu=%u cuda=%u\n",
                cpu_next,
                cuda_next);
        att1_infer_destroy(cpu_q8);
        att1_infer_destroy(cuda_q8);
        return -1;
    }

    att1_infer_destroy(cpu_q8);
    att1_infer_destroy(cuda_q8);
    return 0;
}

static int test_cuda_q8_generated_tokens(const att1_model *model)
{
    att1_infer_t *cpu_q8 = NULL;
    att1_infer_t *cuda_q8 = NULL;
    const unsigned char prompt[] = "hello";
    uint32_t cpu_tokens[4] = {0u};
    uint32_t cuda_tokens[4] = {0u};
    size_t cpu_count = 0u;
    size_t cuda_count = 0u;

    if (create_q8_infer_pair(model, &cpu_q8, &cuda_q8) != 0) {
        return -1;
    }

    if ((att1_infer_generate(cpu_q8,
                             prompt,
                             sizeof(prompt) - 1u,
                             4u,
                             cpu_tokens,
                             4u,
                             &cpu_count) != ATT1_OK) ||
        (att1_infer_generate(cuda_q8,
                             prompt,
                             sizeof(prompt) - 1u,
                             4u,
                             cuda_tokens,
                             4u,
                             &cuda_count) != ATT1_OK) ||
        (cpu_count != 4u) ||
        (cuda_count != 4u) ||
        (memcmp(cpu_tokens, cuda_tokens, sizeof(cpu_tokens)) != 0)) {
        att1_infer_destroy(cpu_q8);
        att1_infer_destroy(cuda_q8);
        return -1;
    }

    att1_infer_destroy(cpu_q8);
    att1_infer_destroy(cuda_q8);
    return 0;
}

static int test_cuda_q8_no_silent_cpu_fallback(const att1_model *model)
{
    att1_infer_t *cuda_q8 = NULL;
    att1_backend *backend = NULL;
    uint32_t next = 0u;

    if (att1_infer_create(model, &cuda_q8) != ATT1_OK) {
        return -1;
    }

    if (att1_backend_cuda_q8_create(&backend) != ATT1_OK) {
        att1_infer_destroy(cuda_q8);
        return -1;
    }

    if ((backend->ops == NULL) ||
        (backend->ops->name == NULL) ||
        (strcmp(backend->ops->name, "cuda-q8") != 0) ||
        (strcmp(backend->ops->name, "cpu-q8") == 0) ||
        (strcmp(backend->ops->name, "cpu-f32") == 0) ||
        (att1_infer_set_backend(cuda_q8, backend) != ATT1_OK)) {
        att1_backend_destroy(backend);
        att1_infer_destroy(cuda_q8);
        return -1;
    }
    backend = NULL;

    if (att1_infer_decode_token(cuda_q8, (uint32_t)'z', &next) != ATT1_OK) {
        att1_infer_destroy(cuda_q8);
        return -1;
    }

    att1_infer_destroy(cuda_q8);
    return 0;
}

int main(void)
{
    att1_model model;

    if (att1_backend_cuda_available() == 0) {
        att1_backend *cuda_backend = NULL;
        att1_backend *cuda_q8_backend = NULL;
        if (att1_backend_cuda_create(&cuda_backend) != ATT1_ERR_UNSUPPORTED) {
            att1_backend_destroy(cuda_backend);
            fputs("cuda infer unavailable-path check failed\n", stderr);
            return 1;
        }
        if (att1_backend_cuda_q8_create(&cuda_q8_backend) !=
            ATT1_ERR_UNSUPPORTED) {
            att1_backend_destroy(cuda_q8_backend);
            fputs("cuda q8 infer unavailable-path check failed\n", stderr);
            return 1;
        }

        puts("cuda_infer test skipped (CUDA unavailable)");
        return 0;
    }

    if (att1_model_load(MODEL_PATH, &model) != ATT1_OK) {
        fputs("cuda_infer model load failed\n", stderr);
        return 1;
    }

    if ((test_next_token_equivalence(&model) != 0) ||
        (test_sequence_equivalence_2_to_4_tokens(&model) != 0) ||
        (test_prompt_prefill_position_and_kv(&model) != 0) ||
        (test_no_silent_cpu_fallback(&model) != 0) ||
        (test_cuda_q8_logits_match_cpu_q8(&model) != 0) ||
        (test_cuda_q8_generated_tokens(&model) != 0) ||
        (test_cuda_q8_no_silent_cpu_fallback(&model) != 0)) {
        att1_model_free(&model);
        fputs("cuda_infer test failed\n", stderr);
        return 1;
    }

    att1_model_free(&model);
    puts("cuda_infer test passed");
    return 0;
}
