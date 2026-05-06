#include "att1_model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t att1_read_u32le(const unsigned char *p)
{
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8u) |
           ((uint32_t)p[2] << 16u) |
           ((uint32_t)p[3] << 24u);
}

static uint64_t att1_read_u64le(const unsigned char *p)
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

static int att1_range_valid(size_t file_size, uint64_t offset, uint64_t size)
{
    if (offset > (uint64_t)file_size) {
        return 0;
    }

    if (size > ((uint64_t)file_size - offset)) {
        return 0;
    }

    return 1;
}

static int att1_u64_mul(uint64_t lhs, uint64_t rhs, uint64_t *out)
{
    if ((lhs != 0u) && (rhs > (UINT64_MAX / lhs))) {
        return -1;
    }

    *out = lhs * rhs;
    return 0;
}

static int att1_tensor_nbytes_expected(const att1_model_tensor *tensor,
                                       uint64_t *out_nbytes)
{
    uint64_t elements = 1u;
    uint32_t i = 0u;

    if ((tensor->ndims == 0u) || (tensor->ndims > ATT1_MODEL_MAX_DIMS)) {
        return -1;
    }

    if (tensor->dtype != ATT1_MODEL_DTYPE_F32) {
        return -1;
    }

    for (i = 0u; i < tensor->ndims; i++) {
        if (tensor->shape[i] == 0u) {
            return -1;
        }

        if (att1_u64_mul(elements, tensor->shape[i], &elements) != 0) {
            return -1;
        }
    }

    return att1_u64_mul(elements, sizeof(float), out_nbytes);
}

static int att1_model_read_file(const char *path,
                                unsigned char **out_data,
                                size_t *out_size)
{
    FILE *fp = NULL;
    long size = 0;
    unsigned char *data = NULL;

    fp = fopen(path, "rb");
    if (fp == NULL) {
        return -1;
    }

    if (fseek(fp, 0L, SEEK_END) != 0) {
        fclose(fp);
        return -1;
    }

    size = ftell(fp);
    if (size < 0) {
        fclose(fp);
        return -1;
    }

    if (fseek(fp, 0L, SEEK_SET) != 0) {
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

static void att1_model_parse_config(const unsigned char *p,
                                    att1_model_config *config)
{
    config->vocab_size = att1_read_u32le(&p[0]);
    config->n_layers = att1_read_u32le(&p[4]);
    config->n_heads = att1_read_u32le(&p[8]);
    config->d_model = att1_read_u32le(&p[12]);
    config->d_ff = att1_read_u32le(&p[16]);
    config->max_seq_len = att1_read_u32le(&p[20]);
    config->rope_dim = att1_read_u32le(&p[24]);
    config->n_tiles = att1_read_u32le(&p[28]);
    config->shard_count = att1_read_u32le(&p[32]);
}

static void att1_model_parse_tensor(const unsigned char *p,
                                    att1_model_tensor *tensor)
{
    uint32_t i = 0u;

    memcpy(tensor->name, p, ATT1_MODEL_NAME_SIZE);
    tensor->name[ATT1_MODEL_NAME_SIZE - 1u] = '\0';
    tensor->dtype = att1_read_u32le(&p[64]);
    tensor->ndims = att1_read_u32le(&p[68]);
    for (i = 0u; i < ATT1_MODEL_MAX_DIMS; i++) {
        tensor->shape[i] = att1_read_u64le(&p[72u + ((size_t)i * 8u)]);
    }
    tensor->offset = att1_read_u64le(&p[104]);
    tensor->nbytes = att1_read_u64le(&p[112]);
    tensor->shard_id = att1_read_u32le(&p[120]);
    tensor->flags = att1_read_u32le(&p[124]);
    tensor->data = NULL;
}

att1_status_t att1_model_load(const char *path, att1_model *model)
{
    unsigned char *data = NULL;
    size_t file_size = 0u;
    uint32_t version = 0u;
    uint32_t header_size = 0u;
    uint64_t config_offset = 0u;
    uint64_t config_size = 0u;
    uint64_t desc_offset = 0u;
    uint64_t tensor_count = 0u;
    uint64_t data_offset = 0u;
    uint64_t data_size = 0u;
    uint64_t shard_offset = 0u;
    uint64_t shard_size = 0u;
    uint64_t desc_bytes = 0u;
    uint64_t i = 0u;

    if ((path == NULL) || (model == NULL)) {
        return ATT1_ERR_INVALID_ARG;
    }

    memset(model, 0, sizeof(*model));

    if (att1_model_read_file(path, &data, &file_size) != 0) {
        return ATT1_ERR_IO;
    }

    if (!att1_range_valid(file_size, 0u, ATT1_MODEL_HEADER_SIZE)) {
        free(data);
        return ATT1_ERR_BAD_FORMAT;
    }

    if (memcmp(data, ATT1_MODEL_MAGIC, ATT1_MODEL_MAGIC_SIZE) != 0) {
        free(data);
        return ATT1_ERR_BAD_FORMAT;
    }

    version = att1_read_u32le(&data[8]);
    header_size = att1_read_u32le(&data[12]);
    if ((version != ATT1_MODEL_VERSION) ||
        (header_size != ATT1_MODEL_HEADER_SIZE)) {
        free(data);
        return ATT1_ERR_UNSUPPORTED;
    }

    config_offset = att1_read_u64le(&data[16]);
    config_size = att1_read_u64le(&data[24]);
    desc_offset = att1_read_u64le(&data[32]);
    tensor_count = att1_read_u64le(&data[40]);
    data_offset = att1_read_u64le(&data[48]);
    data_size = att1_read_u64le(&data[56]);
    shard_offset = att1_read_u64le(&data[64]);
    shard_size = att1_read_u64le(&data[72]);

    if (config_size != ATT1_MODEL_CONFIG_SIZE) {
        free(data);
        return ATT1_ERR_BAD_FORMAT;
    }

    if (!att1_range_valid(file_size, config_offset, config_size) ||
        !att1_range_valid(file_size, data_offset, data_size)) {
        free(data);
        return ATT1_ERR_BAD_FORMAT;
    }

    if (((shard_offset == 0u) != (shard_size == 0u)) ||
        ((shard_size != 0u) && !att1_range_valid(file_size, shard_offset, shard_size))) {
        free(data);
        return ATT1_ERR_BAD_FORMAT;
    }

    if (att1_u64_mul(tensor_count, ATT1_MODEL_TENSOR_DESC_SIZE, &desc_bytes) != 0) {
        free(data);
        return ATT1_ERR_BAD_FORMAT;
    }

    if (!att1_range_valid(file_size, desc_offset, desc_bytes)) {
        free(data);
        return ATT1_ERR_BAD_FORMAT;
    }

    if (tensor_count > (uint64_t)SIZE_MAX / sizeof(*model->tensors)) {
        free(data);
        return ATT1_ERR_OOM;
    }

    model->tensors = calloc((size_t)tensor_count, sizeof(*model->tensors));
    if ((model->tensors == NULL) && (tensor_count != 0u)) {
        free(data);
        return ATT1_ERR_OOM;
    }

    att1_model_parse_config(&data[config_offset], &model->config);
    model->tensor_count = tensor_count;
    model->file_data = data;
    model->file_size = file_size;

    for (i = 0u; i < tensor_count; i++) {
        const uint64_t tensor_desc_offset = desc_offset +
            (i * ATT1_MODEL_TENSOR_DESC_SIZE);
        uint64_t expected_nbytes = 0u;

        att1_model_parse_tensor(&data[tensor_desc_offset],
                                &model->tensors[i]);

        if (att1_tensor_nbytes_expected(&model->tensors[i], &expected_nbytes) != 0) {
            att1_model_free(model);
            return ATT1_ERR_SHAPE;
        }

        if (model->tensors[i].nbytes != expected_nbytes) {
            att1_model_free(model);
            return ATT1_ERR_SHAPE;
        }

        if (!att1_range_valid((size_t)data_size,
                              model->tensors[i].offset,
                              model->tensors[i].nbytes)) {
            att1_model_free(model);
            return ATT1_ERR_BAD_FORMAT;
        }

        model->tensors[i].data = &data[data_offset + model->tensors[i].offset];
    }

    return ATT1_OK;
}

void att1_model_free(att1_model *model)
{
    if (model == NULL) {
        return;
    }

    free(model->tensors);
    free(model->file_data);
    memset(model, 0, sizeof(*model));
}

const att1_model_tensor *att1_model_find_tensor(const att1_model *model,
                                                const char *name)
{
    uint64_t i = 0u;

    if ((model == NULL) || (name == NULL)) {
        return NULL;
    }

    for (i = 0u; i < model->tensor_count; i++) {
        if (strncmp(model->tensors[i].name, name, ATT1_MODEL_NAME_SIZE) == 0) {
            return &model->tensors[i];
        }
    }

    return NULL;
}
