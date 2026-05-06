#ifndef ATT1_INFER_H
#define ATT1_INFER_H

#include "att1_kv_cache.h"
#include "att1_model.h"

#include <stddef.h>
#include <stdint.h>

typedef struct att1_infer {
    const att1_model *model;
    att1_kv_cache *layer_kv;
    float *hidden;
    float *next_hidden;
    float *norm;
    float *logits;
    size_t position;
} att1_infer;

/*
 * Initialize a single-tile, batch-1 inference context for a loaded model.
 *
 * This path uses the Milestone 2 local transformer block and simple KV cache.
 * It does not use fabric, tile runtime, quantization, tokenizer files, or
 * multi-tile sharding.
 */
int att1_infer_init(att1_infer *infer, const att1_model *model);

void att1_infer_free(att1_infer *infer);

/*
 * Decode one input byte token and return the greedy next token.
 *
 * token_id must be in 0..255. out_token receives the argmax over output logits.
 */
int att1_infer_decode_token(att1_infer *infer,
                            uint32_t token_id,
                            uint32_t *out_token);

/*
 * Encode prompt bytes, feed them one token at a time, then continue greedy
 * decoding for generated_token_count additional tokens.
 */
int att1_infer_generate(att1_infer *infer,
                        const unsigned char *prompt,
                        size_t prompt_bytes,
                        size_t generated_token_count,
                        uint32_t *out_tokens,
                        size_t out_token_capacity,
                        size_t *out_token_count);

#endif
