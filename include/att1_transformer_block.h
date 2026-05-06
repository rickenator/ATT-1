#ifndef ATT1_TRANSFORMER_BLOCK_H
#define ATT1_TRANSFORMER_BLOCK_H

#include "att1_backend.h"
#include "att1_kv_cache.h"
#include "att1_quant.h"

#include <stddef.h>

typedef struct att1_transformer_block_config {
    size_t model_dim;
    size_t num_heads;
    size_t head_dim;
    size_t ffn_dim;
    float rms_epsilon;
    float rope_theta;
} att1_transformer_block_config;

typedef struct att1_transformer_block_weights {
    const float *attention_norm;
    const float *ffn_norm;
    const float *wq;
    const float *wk;
    const float *wv;
    const float *wo;
    const float *w_gate;
    const float *w_up;
    const float *w_down;
} att1_transformer_block_weights;

typedef struct att1_transformer_block_q8_weights {
    const float *attention_norm;
    const float *ffn_norm;
    const att1_q8_matrix *wq;
    const att1_q8_matrix *wk;
    const att1_q8_matrix *wv;
    const att1_q8_matrix *wo;
    const att1_q8_matrix *w_gate;
    const att1_q8_matrix *w_up;
    const att1_q8_matrix *w_down;
} att1_transformer_block_q8_weights;

/*
 * Run one local LLaMA-style decoder block for a single decoded token.
 *
 * input and output contain model_dim float32 values. Attention projection
 * weights follow input[1][in_dim] * W[in_dim][out_dim]. Wq/Wk/Wv/Wo are
 * row-major [model_dim][model_dim]. FFN gate/up weights are row-major
 * [model_dim][ffn_dim], and down weights are row-major
 * [ffn_dim][model_dim]. The block performs pre-attention RMSNorm, causal
 * attention over cache positions 0..position, residual add, pre-FFN RMSNorm,
 * SwiGLU FFN, and final residual add.
 */
int att1_transformer_block_forward_f32(
    float *output,
    att1_kv_cache *cache,
    const float *input,
    const att1_transformer_block_weights *weights,
    const att1_transformer_block_config *config,
    size_t position);

int att1_transformer_block_forward_backend(
    float *output,
    att1_kv_cache *cache,
    const float *input,
    const att1_transformer_block_weights *weights,
    const att1_transformer_block_config *config,
    size_t position,
    att1_backend *backend);

int att1_transformer_block_forward_backend_q8(
    float *output,
    att1_kv_cache *cache,
    const float *input,
    const att1_transformer_block_q8_weights *weights,
    const att1_transformer_block_config *config,
    size_t position,
    att1_backend *backend);

#endif
