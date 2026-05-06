#include "att1_model.h"

#include <inttypes.h>
#include <stdio.h>

static void print_shape(const att1_model_tensor *tensor)
{
    uint32_t i = 0u;

    putchar('[');
    for (i = 0u; i < tensor->ndims; i++) {
        if (i != 0u) {
            putchar(',');
        }
        printf("%" PRIu64, tensor->shape[i]);
    }
    putchar(']');
}

int main(int argc, char **argv)
{
    att1_model model;
    uint64_t i = 0u;

    if (argc != 2) {
        fputs("usage: att1-inspect <model.att1>\n", stderr);
        return 2;
    }

    if (att1_model_load(argv[1], &model) != 0) {
        fprintf(stderr, "failed to load model: %s\n", argv[1]);
        return 1;
    }

    printf("ATT-1 model\n");
    printf("vocab_size=%u\n", model.config.vocab_size);
    printf("n_layers=%u\n", model.config.n_layers);
    printf("n_heads=%u\n", model.config.n_heads);
    printf("d_model=%u\n", model.config.d_model);
    printf("d_ff=%u\n", model.config.d_ff);
    printf("max_seq_len=%u\n", model.config.max_seq_len);
    printf("rope_dim=%u\n", model.config.rope_dim);
    printf("n_tiles=%u\n", model.config.n_tiles);
    printf("shard_count=%u\n", model.config.shard_count);
    printf("tensor_count=%" PRIu64 "\n", model.tensor_count);

    for (i = 0u; i < model.tensor_count; i++) {
        const att1_model_tensor *tensor = &model.tensors[i];

        printf("tensor name=%s dtype=%u shape=", tensor->name, tensor->dtype);
        print_shape(tensor);
        printf(" offset=%" PRIu64 " nbytes=%" PRIu64 " shard=%u flags=%u\n",
               tensor->offset,
               tensor->nbytes,
               tensor->shard_id,
               tensor->flags);
    }

    att1_model_free(&model);
    return 0;
}
