#ifndef ATT1_ATTENTION_H
#define ATT1_ATTENTION_H

#include "att1_backend.h"
#include "att1_kv_cache.h"
#include "att1_quant.h"

#include <stddef.h>

typedef struct att1_attention_config {
    size_t model_dim;
    size_t num_heads;
    size_t head_dim;
    float rope_theta;
} att1_attention_config;

typedef struct att1_attention_weights {
    const float *wq;
    const float *wk;
    const float *wv;
    const float *wo;
} att1_attention_weights;

typedef struct att1_attention_q8_weights {
    const att1_q8_matrix *wq;
    const att1_q8_matrix *wk;
    const att1_q8_matrix *wv;
    const att1_q8_matrix *wo;
} att1_attention_q8_weights;

typedef struct att1_attention_q4_weights {
    const att1_q4_matrix *wq;
    const att1_q4_matrix *wk;
    const att1_q4_matrix *wv;
    const att1_q4_matrix *wo;
} att1_attention_q4_weights;

/*
 * Run one batch-1 causal self-attention decode step in float32.
 *
 * input and output contain model_dim values. Projection weights use the
 * vector-matrix convention from att1_matmul_f32:
 * input[1][in_dim] * W[in_dim][out_dim]. For self-attention, Wq/Wk/Wv/Wo are
 * row-major [model_dim][model_dim]. The function projects Q/K/V, applies RoPE
 * to Q and K per head, appends K/V to the local cache, attends over populated
 * cache positions 0..position, and applies the output projection.
 */
int att1_attention_forward_f32(float *output,
                               att1_kv_cache *cache,
                               const float *input,
                               const att1_attention_weights *weights,
                               const att1_attention_config *config,
                               size_t position);

int att1_attention_forward_backend(float *output,
                                   att1_kv_cache *cache,
                                   const float *input,
                                   const att1_attention_weights *weights,
                                   const att1_attention_config *config,
                                   size_t position,
                                   att1_backend *backend);

int att1_attention_forward_backend_q8(
    float *output,
    att1_kv_cache *cache,
    const float *input,
    const att1_attention_q8_weights *weights,
    const att1_attention_config *config,
    size_t position,
    att1_backend *backend);

int att1_attention_forward_backend_q4(
    float *output,
    att1_kv_cache *cache,
    const float *input,
    const att1_attention_q4_weights *weights,
    const att1_attention_config *config,
    size_t position,
    att1_backend *backend);

#endif
