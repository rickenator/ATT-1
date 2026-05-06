#include "att1_model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

typedef struct test_tensor_def {
    const char *name;
    uint32_t ndims;
    uint64_t shape[ATT1_MODEL_MAX_DIMS];
} test_tensor_def;

static void write_u32le(unsigned char *p, uint32_t value)
{
    p[0] = (unsigned char)(value & 0xffu);
    p[1] = (unsigned char)((value >> 8u) & 0xffu);
    p[2] = (unsigned char)((value >> 16u) & 0xffu);
    p[3] = (unsigned char)((value >> 24u) & 0xffu);
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

static void write_f32le(unsigned char *p, float value)
{
    union {
        float f;
        uint32_t u;
    } bits;

    bits.f = value;
    write_u32le(p, bits.u);
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

static uint64_t tensor_element_count(const test_tensor_def *tensor)
{
    uint64_t count = 1u;
    uint32_t i = 0u;

    for (i = 0u; i < tensor->ndims; i++) {
        count *= tensor->shape[i];
    }

    return count;
}

static size_t write_tensor_desc(unsigned char *dst,
                                const test_tensor_def *tensor,
                                uint64_t data_offset)
{
    uint32_t i = 0u;
    const uint64_t nbytes = tensor_element_count(tensor) * sizeof(float);

    memset(dst, 0, ATT1_MODEL_TENSOR_DESC_SIZE);
    (void)snprintf((char *)dst, ATT1_MODEL_NAME_SIZE, "%s", tensor->name);
    write_u32le(&dst[64], ATT1_MODEL_DTYPE_F32);
    write_u32le(&dst[68], tensor->ndims);
    for (i = 0u; i < ATT1_MODEL_MAX_DIMS; i++) {
        write_u64le(&dst[72u + ((size_t)i * 8u)],
                    i < tensor->ndims ? tensor->shape[i] : 1u);
    }
    write_u64le(&dst[104], data_offset);
    write_u64le(&dst[112], nbytes);
    write_u32le(&dst[120], 0u);
    write_u32le(&dst[124], 0u);
    return (size_t)nbytes;
}

static int write_dummy_model_c(const char *path)
{
    const test_tensor_def tensors[] = {
        {"tok_embeddings.weight", 2u, {256u, 4u, 1u, 1u}},
        {"layers.0.attention.wq.weight", 2u, {4u, 4u, 1u, 1u}},
        {"layers.0.ffn.w_gate.weight", 2u, {4u, 8u, 1u, 1u}},
        {"output.weight", 2u, {4u, 256u, 1u, 1u}}
    };
    const size_t tensor_count = sizeof(tensors) / sizeof(tensors[0]);
    const uint64_t config_offset = ATT1_MODEL_HEADER_SIZE;
    const uint64_t desc_offset = config_offset + ATT1_MODEL_CONFIG_SIZE;
    const uint64_t data_offset = desc_offset +
        ((uint64_t)tensor_count * ATT1_MODEL_TENSOR_DESC_SIZE);
    uint64_t data_size = 0u;
    uint64_t tensor_data_offset = 0u;
    size_t file_size = 0u;
    unsigned char *file = NULL;
    size_t i = 0u;
    uint64_t value_index = 0u;
    int rc = -1;

    for (i = 0u; i < tensor_count; i++) {
        data_size += tensor_element_count(&tensors[i]) * sizeof(float);
    }

    file_size = (size_t)(data_offset + data_size);
    file = calloc(file_size, 1u);
    if (file == NULL) {
        return -1;
    }

    memcpy(&file[0], ATT1_MODEL_MAGIC, ATT1_MODEL_MAGIC_SIZE);
    write_u32le(&file[8], ATT1_MODEL_VERSION);
    write_u32le(&file[12], ATT1_MODEL_HEADER_SIZE);
    write_u64le(&file[16], config_offset);
    write_u64le(&file[24], ATT1_MODEL_CONFIG_SIZE);
    write_u64le(&file[32], desc_offset);
    write_u64le(&file[40], tensor_count);
    write_u64le(&file[48], data_offset);
    write_u64le(&file[56], data_size);
    write_u64le(&file[64], 0u);
    write_u64le(&file[72], 0u);

    write_u32le(&file[config_offset + 0u], 256u);
    write_u32le(&file[config_offset + 4u], 2u);
    write_u32le(&file[config_offset + 8u], 2u);
    write_u32le(&file[config_offset + 12u], 4u);
    write_u32le(&file[config_offset + 16u], 8u);
    write_u32le(&file[config_offset + 20u], 8u);
    write_u32le(&file[config_offset + 24u], 2u);
    write_u32le(&file[config_offset + 28u], 1u);
    write_u32le(&file[config_offset + 32u], 0u);

    for (i = 0u; i < tensor_count; i++) {
        const uint64_t elements = tensor_element_count(&tensors[i]);
        uint64_t elem = 0u;

        (void)write_tensor_desc(&file[desc_offset +
                                      (i * ATT1_MODEL_TENSOR_DESC_SIZE)],
                                &tensors[i],
                                tensor_data_offset);
        for (elem = 0u; elem < elements; elem++) {
            const float value = 0.01f + ((float)value_index * 0.001f);
            write_f32le(&file[data_offset + tensor_data_offset +
                              (elem * sizeof(float))],
                        value);
            value_index++;
        }
        tensor_data_offset += elements * sizeof(float);
    }

    rc = write_file(path, file, file_size);
    free(file);
    return rc;
}

static int files_equal(const char *lhs, const char *rhs)
{
    unsigned char *lhs_data = NULL;
    unsigned char *rhs_data = NULL;
    size_t lhs_size = 0u;
    size_t rhs_size = 0u;
    int equal = 0;

    if ((read_file(lhs, &lhs_data, &lhs_size) != 0) ||
        (read_file(rhs, &rhs_data, &rhs_size) != 0)) {
        free(lhs_data);
        free(rhs_data);
        return 0;
    }

    equal = (lhs_size == rhs_size) &&
            ((lhs_size == 0u) || (memcmp(lhs_data, rhs_data, lhs_size) == 0));
    free(lhs_data);
    free(rhs_data);
    return equal;
}

static int make_corrupt(const char *valid_path,
                        const char *bad_path,
                        size_t offset,
                        const unsigned char *patch,
                        size_t patch_size)
{
    unsigned char *data = NULL;
    size_t size = 0u;
    int rc = -1;

    if (read_file(valid_path, &data, &size) != 0) {
        return -1;
    }

    if (offset <= size && patch_size <= (size - offset)) {
        memcpy(&data[offset], patch, patch_size);
        rc = write_file(bad_path, data, size);
    }

    free(data);
    return rc;
}

static int buffer_contains(const unsigned char *data,
                           size_t size,
                           const char *needle)
{
    const size_t needle_len = strlen(needle);
    size_t i = 0u;

    if (needle_len == 0u) {
        return 1;
    }

    if (needle_len > size) {
        return 0;
    }

    for (i = 0u; i <= size - needle_len; i++) {
        if (memcmp(&data[i], needle, needle_len) == 0) {
            return 1;
        }
    }

    return 0;
}

static int expect_load_fail(const char *path)
{
    att1_model model;

    if (att1_model_load(path, &model) == 0) {
        att1_model_free(&model);
        return 0;
    }

    return 1;
}

static int corrupt_u32(const char *valid_path,
                       const char *bad_path,
                       size_t offset,
                       uint32_t value)
{
    unsigned char patch[4];

    write_u32le(patch, value);
    return make_corrupt(valid_path, bad_path, offset, patch, sizeof(patch));
}

static int corrupt_u64(const char *valid_path,
                       const char *bad_path,
                       size_t offset,
                       uint64_t value)
{
    unsigned char patch[8];

    write_u64le(patch, value);
    return make_corrupt(valid_path, bad_path, offset, patch, sizeof(patch));
}

int main(void)
{
    const char *dir = "build/model_test";
    const char *model_a = "build/model_test/model_a.att1";
    const char *model_b = "build/model_test/model_b.att1";
    unsigned char *data = NULL;
    size_t size = 0u;
    uint64_t desc_offset = 0u;
    uint64_t data_offset = 0u;
    uint64_t data_size = 0u;
    const att1_model_tensor *tensor = NULL;
    att1_model model;

    (void)mkdir(dir, 0777);

    if ((write_dummy_model_c(model_a) != 0) ||
        (write_dummy_model_c(model_b) != 0)) {
        fputs("dummy model C fixture writer failed\n", stderr);
        return 1;
    }

    if (!files_equal(model_a, model_b)) {
        fputs("dummy model determinism check failed\n", stderr);
        return 1;
    }

    if (att1_model_load(model_a, &model) != 0) {
        fputs("valid model load failed\n", stderr);
        return 1;
    }

    if ((model.config.vocab_size != 256u) ||
        (model.config.n_layers != 2u) ||
        (model.config.n_heads != 2u) ||
        (model.config.d_model != 4u) ||
        (model.config.d_ff != 8u) ||
        (model.config.max_seq_len != 8u) ||
        (model.config.rope_dim != 2u) ||
        (model.config.n_tiles != 1u)) {
        fputs("model config check failed\n", stderr);
        att1_model_free(&model);
        return 1;
    }

    tensor = att1_model_find_tensor(&model, "layers.0.attention.wq.weight");
    if ((tensor == NULL) ||
        (tensor->dtype != ATT1_MODEL_DTYPE_F32) ||
        (tensor->ndims != 2u) ||
        (tensor->shape[0] != 4u) ||
        (tensor->shape[1] != 4u) ||
        (tensor->data == NULL)) {
        fputs("tensor lookup existing check failed\n", stderr);
        att1_model_free(&model);
        return 1;
    }

    if (att1_model_find_tensor(&model, "missing.tensor") != NULL) {
        fputs("tensor lookup missing check failed\n", stderr);
        att1_model_free(&model);
        return 1;
    }

    att1_model_free(&model);

    if (read_file(model_a, &data, &size) != 0) {
        fputs("read valid model for corruption failed\n", stderr);
        return 1;
    }
    desc_offset = read_u64le(&data[32]);
    data_offset = read_u64le(&data[48]);
    data_size = read_u64le(&data[56]);
    free(data);

    if ((make_corrupt(model_a,
                      "build/model_test/bad_magic.att1",
                      0u,
                      (const unsigned char *)"BADMAGIC",
                      8u) != 0) ||
        !expect_load_fail("build/model_test/bad_magic.att1")) {
        fputs("bad magic validation failed\n", stderr);
        return 1;
    }

    if ((corrupt_u32(model_a,
                     "build/model_test/bad_version.att1",
                     8u,
                     99u) != 0) ||
        !expect_load_fail("build/model_test/bad_version.att1")) {
        fputs("bad version validation failed\n", stderr);
        return 1;
    }

    if ((corrupt_u64(model_a,
                     "build/model_test/bad_desc_offset.att1",
                     32u,
                     (uint64_t)size + 1u) != 0) ||
        !expect_load_fail("build/model_test/bad_desc_offset.att1")) {
        fputs("descriptor bounds validation failed\n", stderr);
        return 1;
    }

    if ((corrupt_u64(model_a,
                     "build/model_test/bad_data_offset.att1",
                     48u,
                     (uint64_t)size + 1u) != 0) ||
        !expect_load_fail("build/model_test/bad_data_offset.att1")) {
        fputs("data offset bounds validation failed\n", stderr);
        return 1;
    }

    if ((corrupt_u64(model_a,
                     "build/model_test/bad_nbytes.att1",
                     (size_t)desc_offset + 112u,
                     data_size + 4u) != 0) ||
        !expect_load_fail("build/model_test/bad_nbytes.att1")) {
        fputs("tensor nbytes bounds validation failed\n", stderr);
        return 1;
    }

    if ((corrupt_u32(model_a,
                     "build/model_test/bad_dtype.att1",
                     (size_t)desc_offset + 64u,
                     999u) != 0) ||
        !expect_load_fail("build/model_test/bad_dtype.att1")) {
        fputs("dtype validation failed\n", stderr);
        return 1;
    }

    if ((corrupt_u32(model_a,
                     "build/model_test/bad_ndims.att1",
                     (size_t)desc_offset + 68u,
                     5u) != 0) ||
        !expect_load_fail("build/model_test/bad_ndims.att1")) {
        fputs("ndims validation failed\n", stderr);
        return 1;
    }

    if ((corrupt_u64(model_a,
                     "build/model_test/bad_shape.att1",
                     (size_t)desc_offset + 72u,
                     0u) != 0) ||
        !expect_load_fail("build/model_test/bad_shape.att1")) {
        fputs("zero shape validation failed\n", stderr);
        return 1;
    }

    if (system("build/att1-inspect build/model_test/model_a.att1 > build/model_test/inspect.txt") != 0) {
        fputs("inspect valid model failed\n", stderr);
        return 1;
    }

    if (system("build/att1-inspect build/model_test/bad_magic.att1 > build/model_test/inspect_bad.txt 2>&1") == 0) {
        fputs("inspect invalid model exit check failed\n", stderr);
        return 1;
    }

    if (read_file("build/model_test/inspect.txt", &data, &size) != 0) {
        fputs("inspect output read failed\n", stderr);
        return 1;
    }

    if ((size == 0u) ||
        !buffer_contains(data, size, "vocab_size=256") ||
        !buffer_contains(data,
                         size,
                         "tensor name=layers.0.attention.wq.weight")) {
        fputs("inspect output content failed\n", stderr);
        free(data);
        return 1;
    }

    free(data);
    (void)data_offset;
    puts("model_loader test passed");
    return 0;
}
