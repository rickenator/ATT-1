#include "att1_cluster_infer.h"
#include "att1_model.h"
#include "att1_tokenizer.h"

#include <stdio.h>

int main(void)
{
    const unsigned char prompt[] = "ATT1";
    const att1_cluster_infer_config config = {2u, 4u, 0u};
    uint32_t tokens[4];
    size_t token_count = 0u;
    size_t i = 0u;
    att1_model model;
    att1_cluster_infer infer;

    if (att1_model_load("models/dummy/model.att1", &model) != 0) {
        fputs("failed to load models/dummy/model.att1\n", stderr);
        return 1;
    }

    if (att1_cluster_infer_init(&infer, &model, &config) != 0) {
        fputs("failed to initialize cluster inference context\n", stderr);
        att1_model_free(&model);
        return 1;
    }

    if (att1_cluster_infer_generate(&infer,
                                    prompt,
                                    sizeof(prompt) - 1u,
                                    4u,
                                    tokens,
                                    4u,
                                    &token_count) != 0) {
        fputs("cluster generation failed\n", stderr);
        att1_cluster_infer_free(&infer);
        att1_model_free(&model);
        return 1;
    }

    printf("prompt: %s\n", prompt);
    printf("cluster tokens:");
    for (i = 0u; i < token_count; i++) {
        unsigned char byte = 0u;
        if (att1_tokenizer_decode_byte(tokens[i], &byte) != 0) {
            byte = '?';
        }
        printf(" %u", tokens[i]);
    }
    putchar('\n');

    att1_cluster_infer_free(&infer);
    att1_model_free(&model);
    return 0;
}
