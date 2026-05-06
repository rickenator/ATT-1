#include "att1_infer.h"
#include "att1_sampler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MODEL_PATH "models/dummy/model.att1"
#define BAD_MISSING_PATH "build/infer_missing_tensor.att1"
#define BAD_SHAPE_PATH "build/infer_wrong_shape.att1"

static uint64_t read_u64le(const unsigned char *p)
{
    return ((uint64_t)p[0]) |
           ((uint64_t)p[1] << 8u) |
           ((uint64_t)p[2] << 16u) |
           ((uint64_t)p[3] << 24u) |
           ((uint64_t)p[4] << 32u) |
           ((uint64_t)p[5] << 40u) |
           ((uint64_t)p[6] << 48u) |
           ((uint64_t)p[7] << 56u);
}

static void write_u64le(unsigned char *p, uint64_t value)
{
    p[0] = (unsigned char)(value & 0xffu);
    p[1] = (unsigned char)((value >> 8u) & 0xffu);
    p[2] = (unsigned char)((value >> 16u) & 0xffu);
    p[3] = (unsigned char)((value >> 24u) & 0xffu);
    p[4] = (unsigned char)((value >> 32u) & 0xffu);
    p[5] = (unsigned char)((value >> 40u) & 0xffu);
    p[6] = (unsigned char)((value >> 48u) & 0xffu);
    p[7] = (unsigned char)((value >> 56u) & 0xffu);
}

static int read_file(const char *path, unsigned char **out_data, size_t *out_size)
{
    FILE *fp = fopen(path, "rb");
    long size = 0;
    unsigned char *data = NULL;

    if (fp == NULL) {
        return -1;
    }

    if ((fseek(fp, 0L, SEEK_END) != 0) || ((size = ftell(fp)) < 0) ||
        (fseek(fp, 0L, SEEK_SET) != 0)) {
        fclose(fp);
        return -1;
    }

    data = malloc((size_t)size);
    if ((data == NULL) && (size != 0)) {
        fclose(fp);
        return -1;
    }

    if ((size > 0) && (fread(data, 1u, (size_t)size, fp) != (size_t)size)) {
        free(data);
        fclose(fp);
        return -1;
    }

    fclose(fp);
    *out_data = data;
    *out_size = (size_t)size;
    return 0;
}

static int write_file(const char *path, const unsigned char *data, size_t size)
{
    FILE *fp = fopen(path, "wb");

    if (fp == NULL) {
        return -1;
    }

    if ((size > 0u) && (fwrite(data, 1u, size, fp) != size)) {
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return 0;
}

static unsigned char *find_desc(unsigned char *data,
                                size_t size,
                                const char *name)
{
    const uint64_t desc_offset = read_u64le(&data[32]);
    const uint64_t tensor_count = read_u64le(&data[40]);
    uint64_t i = 0u;

    if ((size < ATT1_MODEL_HEADER_SIZE) ||
        (desc_offset > size) ||
        (tensor_count > ((size - desc_offset) / ATT1_MODEL_TENSOR_DESC_SIZE))) {
        return NULL;
    }

    for (i = 0u; i < tensor_count; i++) {
        unsigned char *desc = &data[desc_offset +
                                   (i * ATT1_MODEL_TENSOR_DESC_SIZE)];
        if (strncmp((const char *)desc, name, ATT1_MODEL_NAME_SIZE) == 0) {
            return desc;
        }
    }

    return NULL;
}

static int rename_tensor_in_copy(const char *source_path,
                                 const char *dest_path,
                                 const char *old_name,
                                 const char *new_name)
{
    unsigned char *data = NULL;
    unsigned char *desc = NULL;
    size_t size = 0u;
    int rc = -1;

    if (read_file(source_path, &data, &size) != 0) {
        return -1;
    }

    desc = find_desc(data, size, old_name);
    if (desc != NULL) {
        memset(desc, 0, ATT1_MODEL_NAME_SIZE);
        (void)snprintf((char *)desc, ATT1_MODEL_NAME_SIZE, "%s", new_name);
        rc = write_file(dest_path, data, size);
    }

    free(data);
    return rc;
}

static int patch_output_shape_in_copy(const char *source_path,
                                      const char *dest_path)
{
    unsigned char *data = NULL;
    unsigned char *desc = NULL;
    size_t size = 0u;
    int rc = -1;

    if (read_file(source_path, &data, &size) != 0) {
        return -1;
    }

    desc = find_desc(data, size, "output.weight");
    if (desc != NULL) {
        write_u64le(&desc[80], 255u);
        write_u64le(&desc[112], 4u * 255u * sizeof(float));
        rc = write_file(dest_path, data, size);
    }

    free(data);
    return rc;
}

static int check_required_tensors(const att1_model *model)
{
    const char *required[] = {
        "tok_embeddings.weight",
        "layers.0.attention_norm.weight",
        "layers.0.attention.wq.weight",
        "layers.0.attention.wk.weight",
        "layers.0.attention.wv.weight",
        "layers.0.attention.wo.weight",
        "layers.0.ffn_norm.weight",
        "layers.0.ffn.w_gate.weight",
        "layers.0.ffn.w_up.weight",
        "layers.0.ffn.w_down.weight",
        "layers.1.attention_norm.weight",
        "layers.1.attention.wq.weight",
        "layers.1.attention.wk.weight",
        "layers.1.attention.wv.weight",
        "layers.1.attention.wo.weight",
        "layers.1.ffn_norm.weight",
        "layers.1.ffn.w_gate.weight",
        "layers.1.ffn.w_up.weight",
        "layers.1.ffn.w_down.weight",
        "output_norm.weight",
        "output.weight"
    };
    const size_t count = sizeof(required) / sizeof(required[0]);
    size_t i = 0u;

    for (i = 0u; i < count; i++) {
        if (att1_model_find_tensor(model, required[i]) == NULL) {
            fprintf(stderr, "missing required tensor: %s\n", required[i]);
            return -1;
        }
    }

    return 0;
}

static int check_invalid_models(void)
{
    att1_model model;
    att1_infer infer;

    if (rename_tensor_in_copy(MODEL_PATH,
                              BAD_MISSING_PATH,
                              "tok_embeddings.weight",
                              "missing_embeddings.weight") != 0) {
        fputs("failed to create missing-tensor model fixture\n", stderr);
        return -1;
    }

    if (att1_model_load(BAD_MISSING_PATH, &model) != 0) {
        fputs("missing-tensor model should load as a valid container\n", stderr);
        return -1;
    }

    if (att1_infer_init(&infer, &model) == 0) {
        att1_infer_free(&infer);
        att1_model_free(&model);
        fputs("inference init accepted missing required tensor\n", stderr);
        return -1;
    }
    att1_model_free(&model);

    if (patch_output_shape_in_copy(MODEL_PATH, BAD_SHAPE_PATH) != 0) {
        fputs("failed to create wrong-shape model fixture\n", stderr);
        return -1;
    }

    if (att1_model_load(BAD_SHAPE_PATH, &model) != 0) {
        fputs("wrong-shape model should load as a valid container\n", stderr);
        return -1;
    }

    if (att1_infer_init(&infer, &model) == 0) {
        att1_infer_free(&infer);
        att1_model_free(&model);
        fputs("inference init accepted wrong output shape\n", stderr);
        return -1;
    }
    att1_model_free(&model);

    return 0;
}

static int check_decode_determinism(const att1_model *model)
{
    att1_infer lhs;
    att1_infer rhs;
    uint32_t lhs_token = 0u;
    uint32_t rhs_token = 0u;
    uint32_t sampled = 0u;
    uint32_t layer = 0u;

    if ((att1_infer_init(&lhs, model) != 0) ||
        (att1_infer_init(&rhs, model) != 0)) {
        fputs("inference init failed for valid model\n", stderr);
        return -1;
    }

    if ((att1_infer_decode_token(&lhs, (uint32_t)'A', &lhs_token) != 0) ||
        (att1_infer_decode_token(&rhs, (uint32_t)'A', &rhs_token) != 0)) {
        att1_infer_free(&lhs);
        att1_infer_free(&rhs);
        fputs("one-token decode failed\n", stderr);
        return -1;
    }

    if ((lhs_token != rhs_token) ||
        (memcmp(lhs.logits,
                rhs.logits,
                model->config.vocab_size * sizeof(float)) != 0)) {
        att1_infer_free(&lhs);
        att1_infer_free(&rhs);
        fputs("one-token decode was not deterministic\n", stderr);
        return -1;
    }

    for (layer = 0u; layer < model->config.n_layers; layer++) {
        if (lhs.layer_kv[layer].length != 1u) {
            att1_infer_free(&lhs);
            att1_infer_free(&rhs);
            fputs("decode did not update KV cache once per layer\n", stderr);
            return -1;
        }
    }

    if ((att1_sampler_greedy_f32(lhs.logits,
                                 model->config.vocab_size,
                                 &sampled) != 0) ||
        (sampled != lhs_token) ||
        (sampled >= model->config.vocab_size)) {
        att1_infer_free(&lhs);
        att1_infer_free(&rhs);
        fputs("logits shape or sampled token check failed\n", stderr);
        return -1;
    }

    att1_infer_free(&lhs);
    att1_infer_free(&rhs);
    return 0;
}

static int check_prefill(const att1_model *model)
{
    const unsigned char prompt[3] = {'A', 'T', 'T'};
    uint32_t out_tokens[1] = {0u};
    size_t out_count = 99u;
    att1_infer infer;

    if (att1_infer_init(&infer, model) != 0) {
        fputs("prefill inference init failed\n", stderr);
        return -1;
    }

    if ((att1_infer_generate(&infer,
                             prompt,
                             sizeof(prompt),
                             0u,
                             out_tokens,
                             0u,
                             &out_count) != 0) ||
        (infer.position != sizeof(prompt)) ||
        (out_count != 0u)) {
        att1_infer_free(&infer);
        fputs("prompt prefill position check failed\n", stderr);
        return -1;
    }
    att1_infer_free(&infer);

    if (att1_infer_init(&infer, model) != 0) {
        fputs("empty prompt inference init failed\n", stderr);
        return -1;
    }

    if (att1_infer_generate(&infer,
                            prompt,
                            0u,
                            1u,
                            out_tokens,
                            1u,
                            &out_count) == 0) {
        att1_infer_free(&infer);
        fputs("empty prompt should be rejected\n", stderr);
        return -1;
    }

    att1_infer_free(&infer);
    return 0;
}

int main(void)
{
    att1_model model;

    if (att1_model_load(MODEL_PATH, &model) != 0) {
        fputs("failed to load dummy model\n", stderr);
        return 1;
    }

    if ((model.config.vocab_size != 256u) ||
        (check_required_tensors(&model) != 0) ||
        (check_decode_determinism(&model) != 0) ||
        (check_prefill(&model) != 0)) {
        att1_model_free(&model);
        return 1;
    }

    att1_model_free(&model);

    if (check_invalid_models() != 0) {
        return 1;
    }

    puts("infer test passed");
    return 0;
}
