#ifndef ATT1_MODEL_VIEW_H
#define ATT1_MODEL_VIEW_H

#include "att1_model.h"
#include "att1_status.h"
#include "att1_transformer_block.h"

#include <stdint.h>

/*
 * Shared validated view over Milestone 7/8 decoder model tensors.
 *
 * Returned float pointers are borrowed from the att1_model and remain valid
 * until att1_model_free is called.
 */
att1_status_t att1_model_view_validate_decoder(const att1_model *model);

att1_status_t att1_model_view_validate_decoder_q4(const att1_model *model);

att1_status_t att1_model_view_tensor_f32(const att1_model *model,
                                         const char *name,
                                         uint32_t ndims,
                                         uint64_t dim0,
                                         uint64_t dim1,
                                         const float **out_data);

att1_status_t att1_model_view_tensor_q8(const att1_model *model,
                                        const char *name,
                                        uint64_t rows,
                                        uint64_t cols,
                                        att1_q8_matrix *out_matrix);

att1_status_t att1_model_view_tensor_q4(const att1_model *model,
                                        const char *name,
                                        uint64_t rows,
                                        uint64_t cols,
                                        att1_q4_matrix *out_matrix);

att1_status_t att1_model_view_token_embedding(const att1_model *model,
                                              const float **out_data);

att1_status_t att1_model_view_output_norm(const att1_model *model,
                                          const float **out_data);

att1_status_t att1_model_view_output_weight(const att1_model *model,
                                            const float **out_data);

att1_status_t att1_model_view_load_layer_weights(
    const att1_model *model,
    uint32_t layer,
    att1_transformer_block_weights *out_weights);

#endif
