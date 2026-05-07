#ifndef ATT1_MODEL_H
#define ATT1_MODEL_H

#include "att1_status.h"
#include "att1_shard_meta.h"

#include <stddef.h>
#include <stdint.h>

#define ATT1_MODEL_MAGIC "ATT1MODL"
#define ATT1_MODEL_MAGIC_SIZE 8u
#define ATT1_MODEL_VERSION 1u
#define ATT1_MODEL_NAME_SIZE 64u
#define ATT1_MODEL_MAX_DIMS 4u
#define ATT1_MODEL_HEADER_SIZE 80u
#define ATT1_MODEL_CONFIG_SIZE 36u
#define ATT1_MODEL_TENSOR_DESC_SIZE 128u

typedef struct att1_model_info {
    const char *name;
    uint32_t layer_count;
    size_t parameter_bytes;
} att1_model_info;

typedef enum att1_model_dtype {
    ATT1_MODEL_DTYPE_F32 = 1,
    ATT1_MODEL_DTYPE_Q8 = 2
} att1_model_dtype;

typedef struct att1_model_config {
    uint32_t vocab_size;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t d_model;
    uint32_t d_ff;
    uint32_t max_seq_len;
    uint32_t rope_dim;
    uint32_t n_tiles;
    uint32_t shard_count;
} att1_model_config;

typedef struct att1_model_tensor {
    char name[ATT1_MODEL_NAME_SIZE];
    uint32_t dtype;
    uint32_t ndims;
    uint64_t shape[ATT1_MODEL_MAX_DIMS];
    uint64_t offset;
    uint64_t nbytes;
    uint32_t shard_id;
    uint32_t flags;
    const void *data;
} att1_model_tensor;

typedef struct att1_model {
    att1_model_config config;
    uint64_t tensor_count;
    att1_model_tensor *tensors;
    unsigned char *file_data;
    size_t file_size;
    att1_shard_meta shard_meta;
} att1_model;

/*
 * Load and validate an ATT-1 little-endian binary model file.
 *
 * The loader owns the file mapping copy and tensor descriptors until
 * att1_model_free is called. Supported tensor dtypes are validated by the
 * loader before tensor views are exposed.
 */
att1_status_t att1_model_load(const char *path, att1_model *model);

void att1_model_free(att1_model *model);

const att1_model_tensor *att1_model_find_tensor(const att1_model *model,
                                                const char *name);

#endif
