#include "att1_model.h"
#include "att1_shard_meta.h"

#include <inttypes.h>
#include <stdio.h>

static const char *shard_dtype_name(uint32_t dtype)
{
    switch (dtype) {
    case ATT1_SHARD_DTYPE_F32: return "f32";
    case ATT1_SHARD_DTYPE_Q8:  return "q8";
    default:                   return "unknown";
    }
}

static const char *shard_repl_name(uint32_t repl)
{
    switch (repl) {
    case ATT1_SHARD_REPL_NONE:        return "none";
    case ATT1_SHARD_REPL_READ:        return "read";
    case ATT1_SHARD_REPL_WRITE_BCAST: return "write-bcast";
    default:                          return "unknown";
    }
}

static const char *shard_reduce_name(uint32_t reduce)
{
    switch (reduce) {
    case ATT1_SHARD_REDUCE_NONE:   return "none";
    case ATT1_SHARD_REDUCE_SUM:    return "sum";
    case ATT1_SHARD_REDUCE_MAX:    return "max";
    case ATT1_SHARD_REDUCE_CONCAT: return "concat";
    default:                       return "unknown";
    }
}

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

    if (model.shard_meta.count > 0u) {
        printf("shard_meta: %" PRIu64 " records\n", model.shard_meta.count);

        for (i = 0u; i < model.shard_meta.count; i++) {
            const att1_shard_meta_record *rec = &model.shard_meta.records[i];
            const char *name = (rec->tensor_id < model.tensor_count)
                               ? model.tensors[rec->tensor_id].name
                               : "(out-of-range)";

            printf("  shard[%" PRIu64 "] tile=%u aimu=%u"
                   " dtype=%s repl=%s reduce=%s  %s\n",
                   i,
                   rec->tile_id,
                   rec->owner_aimu,
                   shard_dtype_name(rec->dtype),
                   shard_repl_name(rec->replication_policy),
                   shard_reduce_name(rec->reduction_behavior),
                   name);
        }
    }

    att1_model_free(&model);
    return 0;
}
