#ifndef ATT1_INFER_H
#define ATT1_INFER_H

#include "att1_model.h"
#include "att1_status.h"

#include <stddef.h>
#include <stdint.h>

typedef struct att1_infer att1_infer_t;

/*
 * Create/destroy a single-tile, batch-1 inference context for a loaded model.
 *
 * The returned context owns all local buffers and KV caches. Callers must not
 * copy the opaque handle and must release it with att1_infer_destroy.
 */
att1_status_t att1_infer_create(const att1_model *model,
                                att1_infer_t **out_infer);

void att1_infer_destroy(att1_infer_t *infer);

/*
 * Decode one byte token and return the greedy next token.
 *
 * token_id must be in 0..255. out_token receives the argmax over output logits.
 */
att1_status_t att1_infer_decode_token(att1_infer_t *infer,
                                      uint32_t token_id,
                                      uint32_t *out_token);

/*
 * Feed prompt bytes one token at a time, then continue greedy decoding for
 * generated_token_count additional tokens.
 */
att1_status_t att1_infer_generate(att1_infer_t *infer,
                                  const unsigned char *prompt,
                                  size_t prompt_bytes,
                                  size_t generated_token_count,
                                  uint32_t *out_tokens,
                                  size_t out_token_capacity,
                                  size_t *out_token_count);

const float *att1_infer_logits(const att1_infer_t *infer,
                               size_t *out_count);

att1_status_t att1_infer_position(const att1_infer_t *infer,
                                  size_t *out_position);

att1_status_t att1_infer_layer_kv_length(const att1_infer_t *infer,
                                         uint32_t layer_id,
                                         size_t *out_length);

#endif
