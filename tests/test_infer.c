#include "att1_backend.h"
#include "att1_infer.h"
#include "att1_math.h"
#include "att1_sampler.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MODEL_PATH "models/dummy/model.att1"
#define BAD_MISSING_PATH "build/infer_missing_tensor.att1"
#define BAD_SHAPE_PATH "build/infer_wrong_shape.att1"
#define BAD_LAYER_SHAPE_PATH "build/infer_wrong_layer_shape.att1"
#define CPU_Q8_LOGIT_TOL 0.15f

static unsigned g_fake_q8_matmul_calls = 0u;
static unsigned g_fake_f32_matmul_calls = 0u;

static void *fake_q8_alloc(att1_backend *backend, size_t bytes)
{
    (void)backend;
    return malloc(bytes);
}

static void fake_q8_free(att1_backend *backend, void *ptr)
{
    (void)backend;
    free(ptr);
}

static int fake_q8_sync(att1_backend *backend)
{
    (void)backend;
    return 0;
}

static int fake_q8_matmul_f32(att1_backend *backend,
                              float *dst,
                              const float *lhs,
                              const float *rhs,
                              size_t rows,
                              size_t cols,
                              size_t inner)
{
    (void)backend;
    (void)dst;
    (void)lhs;
    (void)rhs;
    (void)rows;
    (void)cols;
    (void)inner;
    g_fake_f32_matmul_calls++;
    return -1;
}

static int fake_q8_matmul_q8xf32(att1_backend *backend,
                                 float *dst,
                                 const float *lhs,
                                 size_t lhs_rows,
                                 size_t lhs_cols,
                                 const att1_q8_matrix *weights)
{
    (void)backend;
    g_fake_q8_matmul_calls++;
    return att1_matmul_q8xf32(dst, lhs, lhs_rows, lhs_cols, weights);
}

static int fake_q8_rmsnorm_f32(att1_backend *backend,
                               float *dst,
                               const float *src,
                               const float *weight,
                               size_t count,
                               float epsilon)
{
    (void)backend;
    return att1_rmsnorm_f32(dst, src, weight, count, epsilon);
}

static int fake_q8_softmax_f32(att1_backend *backend,
                               float *values,
                               size_t count)
{
    (void)backend;
    return att1_softmax_f32(values, count);
}

static int fake_q8_rope_f32(att1_backend *backend,
                            float *values,
                            size_t count,
                            size_t position,
                            float theta)
{
    (void)backend;
    return att1_rope_f32(values, count, position, theta);
}

static int fake_q8_ffn_swiglu_f32(att1_backend *backend,
                                  float *dst,
                                  const float *gate,
                                  const float *value,
                                  size_t count)
{
    (void)backend;
    return att1_swiglu_f32(dst, gate, value, count);
}

static const att1_backend_ops fake_cpu_q8_ops = {
    "cpu-q8",
    fake_q8_alloc,
    fake_q8_free,
    fake_q8_sync,
    fake_q8_matmul_f32,
    fake_q8_matmul_q8xf32,
    fake_q8_rmsnorm_f32,
    fake_q8_softmax_f32,
    fake_q8_rope_f32,
    fake_q8_ffn_swiglu_f32
};

static const att1_backend_ops fake_cpu_q8_unsupported_ops = {
    "cpu-q8",
    fake_q8_alloc,
    fake_q8_free,
    fake_q8_sync,
    fake_q8_matmul_f32,
    NULL,
    fake_q8_rmsnorm_f32,
    fake_q8_softmax_f32,
    fake_q8_rope_f32,
    fake_q8_ffn_swiglu_f32
};

static att1_backend *make_fake_backend(const att1_backend_ops *ops)
{
    att1_backend *backend = calloc(1u, sizeof(*backend));

    if (backend != NULL) {
        backend->ops = ops;
    }

    return backend;
}

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

static int patch_layer_shape_in_copy(const char *source_path,
                                     const char *dest_path)
{
    unsigned char *data = NULL;
    unsigned char *desc = NULL;
    size_t size = 0u;
    int rc = -1;

    if (read_file(source_path, &data, &size) != 0) {
        return -1;
    }

    desc = find_desc(data, size, "layers.0.attention.wq.weight");
    if (desc != NULL) {
        write_u64le(&desc[80], 3u);
        write_u64le(&desc[112], 4u * 3u * sizeof(float));
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
    att1_infer_t *infer = NULL;
    att1_status_t status = ATT1_OK;

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

    status = att1_infer_create(&model, &infer);
    if (status != ATT1_ERR_NOT_FOUND) {
        att1_infer_destroy(infer);
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

    status = att1_infer_create(&model, &infer);
    if (status != ATT1_ERR_SHAPE) {
        att1_infer_destroy(infer);
        att1_model_free(&model);
        fputs("inference init accepted wrong output shape\n", stderr);
        return -1;
    }
    att1_model_free(&model);

    if (patch_layer_shape_in_copy(MODEL_PATH, BAD_LAYER_SHAPE_PATH) != 0) {
        fputs("failed to create wrong-layer-shape model fixture\n", stderr);
        return -1;
    }

    if (att1_model_load(BAD_LAYER_SHAPE_PATH, &model) != ATT1_OK) {
        fputs("wrong-layer-shape model should load as a valid container\n", stderr);
        return -1;
    }

    status = att1_infer_create(&model, &infer);
    if (status != ATT1_OK) {
        att1_model_free(&model);
        fputs("wrong-layer-shape model should create before layer decode\n", stderr);
        return -1;
    }

    status = att1_infer_decode_token(infer, (uint32_t)'A', &(uint32_t){0u});
    if (status != ATT1_ERR_SHAPE) {
        att1_infer_destroy(infer);
        att1_model_free(&model);
        fputs("wrong layer tensor shape did not fail cleanly\n", stderr);
        return -1;
    }
    att1_infer_destroy(infer);
    att1_model_free(&model);

    return 0;
}

static int check_decode_determinism(const att1_model *model)
{
    att1_infer_t *lhs = NULL;
    att1_infer_t *rhs = NULL;
    const float *lhs_logits = NULL;
    const float *rhs_logits = NULL;
    size_t lhs_logits_count = 0u;
    size_t rhs_logits_count = 0u;
    uint32_t lhs_token = 0u;
    uint32_t rhs_token = 0u;
    uint32_t sampled = 0u;
    uint32_t layer = 0u;

    if ((att1_infer_create(model, &lhs) != ATT1_OK) ||
        (att1_infer_create(model, &rhs) != ATT1_OK)) {
        att1_infer_destroy(lhs);
        att1_infer_destroy(rhs);
        fputs("inference init failed for valid model\n", stderr);
        return -1;
    }

    if ((att1_infer_decode_token(lhs, (uint32_t)'A', &lhs_token) != ATT1_OK) ||
        (att1_infer_decode_token(rhs, (uint32_t)'A', &rhs_token) != ATT1_OK)) {
        att1_infer_destroy(lhs);
        att1_infer_destroy(rhs);
        fputs("one-token decode failed\n", stderr);
        return -1;
    }

    lhs_logits = att1_infer_logits(lhs, &lhs_logits_count);
    rhs_logits = att1_infer_logits(rhs, &rhs_logits_count);
    if ((lhs_token != rhs_token) ||
        (lhs_logits == NULL) ||
        (rhs_logits == NULL) ||
        (lhs_logits_count != model->config.vocab_size) ||
        (rhs_logits_count != model->config.vocab_size) ||
        (memcmp(lhs_logits,
                rhs_logits,
                model->config.vocab_size * sizeof(float)) != 0)) {
        att1_infer_destroy(lhs);
        att1_infer_destroy(rhs);
        fputs("one-token decode was not deterministic\n", stderr);
        return -1;
    }

    for (layer = 0u; layer < model->config.n_layers; layer++) {
        size_t length = 0u;
        if ((att1_infer_layer_kv_length(lhs, layer, &length) != ATT1_OK) ||
            (length != 1u)) {
            att1_infer_destroy(lhs);
            att1_infer_destroy(rhs);
            fputs("decode did not update KV cache once per layer\n", stderr);
            return -1;
        }
    }

    if ((att1_sampler_greedy_f32(lhs_logits,
                                 model->config.vocab_size,
                                 &sampled) != 0) ||
        (sampled != lhs_token) ||
        (sampled >= model->config.vocab_size)) {
        att1_infer_destroy(lhs);
        att1_infer_destroy(rhs);
        fputs("logits shape or sampled token check failed\n", stderr);
        return -1;
    }

    att1_infer_destroy(lhs);
    att1_infer_destroy(rhs);
    return 0;
}

static int check_prefill(const att1_model *model)
{
    const unsigned char prompt[3] = {'A', 'T', 'T'};
    uint32_t out_tokens[1] = {0u};
    size_t out_count = 99u;
    size_t position = 0u;
    att1_infer_t *infer = NULL;

    if (att1_infer_create(model, &infer) != ATT1_OK) {
        fputs("prefill inference init failed\n", stderr);
        return -1;
    }

    if ((att1_infer_generate(infer,
                             prompt,
                             sizeof(prompt),
                             0u,
                             out_tokens,
                             0u,
                             &out_count) != ATT1_OK) ||
        (att1_infer_position(infer, &position) != ATT1_OK) ||
        (position != sizeof(prompt)) ||
        (out_count != 0u)) {
        att1_infer_destroy(infer);
        fputs("prompt prefill position check failed\n", stderr);
        return -1;
    }
    att1_infer_destroy(infer);

    if (att1_infer_create(model, &infer) != ATT1_OK) {
        fputs("empty prompt inference init failed\n", stderr);
        return -1;
    }

    if (att1_infer_generate(infer,
                            prompt,
                            0u,
                            1u,
                            out_tokens,
                            1u,
                            &out_count) != ATT1_ERR_INVALID_ARG) {
        att1_infer_destroy(infer);
        fputs("empty prompt should be rejected\n", stderr);
        return -1;
    }

    att1_infer_destroy(infer);
    return 0;
}

static int set_cpu_q8_backend(att1_infer_t *infer)
{
    att1_backend *backend = NULL;

    if (att1_backend_cpu_q8_create(&backend) != ATT1_OK) {
        return -1;
    }

    if (att1_infer_set_backend(infer, backend) != ATT1_OK) {
        att1_backend_destroy(backend);
        return -1;
    }

    return 0;
}

static int check_cpu_q8_logits(const att1_model *model)
{
    att1_infer_t *f32 = NULL;
    att1_infer_t *q8 = NULL;
    const float *f32_logits = NULL;
    const float *q8_logits = NULL;
    size_t f32_count = 0u;
    size_t q8_count = 0u;
    uint32_t f32_token = 0u;
    uint32_t q8_token = 0u;
    size_t i = 0u;
    float max_diff = 0.0f;

    if ((att1_infer_create(model, &f32) != ATT1_OK) ||
        (att1_infer_create(model, &q8) != ATT1_OK) ||
        (set_cpu_q8_backend(q8) != 0)) {
        att1_infer_destroy(f32);
        att1_infer_destroy(q8);
        fputs("cpu q8 logits: inference init failed\n", stderr);
        return -1;
    }

    if ((att1_infer_decode_token(f32, (uint32_t)'A', &f32_token) != ATT1_OK) ||
        (att1_infer_decode_token(q8, (uint32_t)'A', &q8_token) != ATT1_OK)) {
        att1_infer_destroy(f32);
        att1_infer_destroy(q8);
        fputs("cpu q8 logits: decode failed\n", stderr);
        return -1;
    }

    f32_logits = att1_infer_logits(f32, &f32_count);
    q8_logits = att1_infer_logits(q8, &q8_count);
    if ((f32_logits == NULL) ||
        (q8_logits == NULL) ||
        (f32_count != model->config.vocab_size) ||
        (q8_count != model->config.vocab_size)) {
        att1_infer_destroy(f32);
        att1_infer_destroy(q8);
        fputs("cpu q8 logits: logits shape check failed\n", stderr);
        return -1;
    }

    for (i = 0u; i < f32_count; i++) {
        const float diff = fabsf(f32_logits[i] - q8_logits[i]);

        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    if (max_diff > CPU_Q8_LOGIT_TOL) {
        fprintf(stderr,
                "cpu q8 logits: max diff %.6f exceeded %.6f "
                "(f32 token=%u q8 token=%u)\n",
                (double)max_diff,
                (double)CPU_Q8_LOGIT_TOL,
                f32_token,
                q8_token);
        att1_infer_destroy(f32);
        att1_infer_destroy(q8);
        return -1;
    }

    att1_infer_destroy(f32);
    att1_infer_destroy(q8);
    return 0;
}

static int check_cpu_q8_generated_tokens(const att1_model *model)
{
    const unsigned char prompt[5] = {'h', 'e', 'l', 'l', 'o'};
    uint32_t f32_tokens[4] = {0u};
    uint32_t q8_tokens[4] = {0u};
    size_t f32_count = 0u;
    size_t q8_count = 0u;
    size_t i = 0u;
    att1_infer_t *f32 = NULL;
    att1_infer_t *q8 = NULL;

    if ((att1_infer_create(model, &f32) != ATT1_OK) ||
        (att1_infer_create(model, &q8) != ATT1_OK) ||
        (set_cpu_q8_backend(q8) != 0)) {
        att1_infer_destroy(f32);
        att1_infer_destroy(q8);
        fputs("cpu q8 tokens: inference init failed\n", stderr);
        return -1;
    }

    if ((att1_infer_generate(f32,
                             prompt,
                             sizeof(prompt),
                             4u,
                             f32_tokens,
                             4u,
                             &f32_count) != ATT1_OK) ||
        (att1_infer_generate(q8,
                             prompt,
                             sizeof(prompt),
                             4u,
                             q8_tokens,
                             4u,
                             &q8_count) != ATT1_OK)) {
        att1_infer_destroy(f32);
        att1_infer_destroy(q8);
        fputs("cpu q8 tokens: generate failed\n", stderr);
        return -1;
    }

    if ((f32_count != 4u) || (q8_count != 4u)) {
        att1_infer_destroy(f32);
        att1_infer_destroy(q8);
        fputs("cpu q8 tokens: generated count mismatch\n", stderr);
        return -1;
    }

    for (i = 0u; i < 4u; i++) {
        if ((f32_tokens[i] >= model->config.vocab_size) ||
            (q8_tokens[i] >= model->config.vocab_size)) {
            att1_infer_destroy(f32);
            att1_infer_destroy(q8);
            fputs("cpu q8 tokens: generated token out of range\n", stderr);
            return -1;
        }

        if (f32_tokens[i] != q8_tokens[i]) {
            fprintf(stderr,
                    "cpu q8 tokens: token %zu f32=%u q8=%u\n",
                    i,
                    f32_tokens[i],
                    q8_tokens[i]);
            att1_infer_destroy(f32);
            att1_infer_destroy(q8);
            return -1;
        }
    }

    att1_infer_destroy(f32);
    att1_infer_destroy(q8);
    return 0;
}

static int check_cpu_q8_no_silent_f32_fallback(const att1_model *model)
{
    att1_infer_t *infer = NULL;
    att1_backend *backend = NULL;
    uint32_t token = 0u;

    g_fake_q8_matmul_calls = 0u;
    g_fake_f32_matmul_calls = 0u;

    if (att1_infer_create(model, &infer) != ATT1_OK) {
        fputs("cpu q8 fallback: inference init failed\n", stderr);
        return -1;
    }

    backend = make_fake_backend(&fake_cpu_q8_ops);
    if (backend == NULL) {
        att1_infer_destroy(infer);
        return -1;
    }

    if (att1_infer_set_backend(infer, backend) != ATT1_OK) {
        att1_backend_destroy(backend);
        att1_infer_destroy(infer);
        fputs("cpu q8 fallback: set backend failed\n", stderr);
        return -1;
    }

    if (att1_infer_decode_token(infer, (uint32_t)'A', &token) != ATT1_OK) {
        att1_infer_destroy(infer);
        fputs("cpu q8 fallback: decode failed\n", stderr);
        return -1;
    }

    if ((g_fake_q8_matmul_calls == 0u) || (g_fake_f32_matmul_calls != 0u)) {
        att1_infer_destroy(infer);
        fputs("cpu q8 fallback: f32 matmul path was used\n", stderr);
        return -1;
    }

    att1_infer_destroy(infer);
    return 0;
}

static int check_cpu_q8_unsupported_path(const att1_model *model)
{
    att1_infer_t *infer = NULL;
    att1_backend *backend = NULL;
    att1_status_t status = ATT1_OK;

    if (att1_infer_create(model, &infer) != ATT1_OK) {
        fputs("cpu q8 unsupported: inference init failed\n", stderr);
        return -1;
    }

    backend = make_fake_backend(&fake_cpu_q8_unsupported_ops);
    if (backend == NULL) {
        att1_infer_destroy(infer);
        return -1;
    }

    status = att1_infer_set_backend(infer, backend);
    if (status != ATT1_ERR_UNSUPPORTED) {
        att1_backend_destroy(backend);
        att1_infer_destroy(infer);
        fputs("cpu q8 unsupported: unsupported q8 path accepted\n", stderr);
        return -1;
    }

    att1_backend_destroy(backend);
    att1_infer_destroy(infer);
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
        (check_prefill(&model) != 0) ||
        (check_cpu_q8_logits(&model) != 0) ||
        (check_cpu_q8_generated_tokens(&model) != 0) ||
        (check_cpu_q8_no_silent_f32_fallback(&model) != 0) ||
        (check_cpu_q8_unsupported_path(&model) != 0)) {
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
