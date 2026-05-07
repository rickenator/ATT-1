#include "att1_model.h"
#include "att1_quant.h"
#include "att1_shard.h"
#include "att1_shard_meta.h"
#include "att1_tok_meta.h"

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

static const char *model_dtype_name(uint32_t dtype)
{
    switch (dtype) {
    case ATT1_MODEL_DTYPE_F32: return "f32";
    case ATT1_MODEL_DTYPE_Q8:  return "q8";
    case ATT1_MODEL_DTYPE_Q4:  return "q4";
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
        printf(" dtype_name=%s", model_dtype_name(tensor->dtype));
        if (tensor->dtype == ATT1_MODEL_DTYPE_Q8) {
            const uint64_t rows = tensor->shape[0];
            const uint64_t cols = tensor->shape[1];
            printf(" quant=per-row-q8 q8_values=%" PRIu64 " q8_scales=%" PRIu64,
                   rows * cols,
                   rows);
        } else if (tensor->dtype == ATT1_MODEL_DTYPE_Q4) {
            const uint64_t rows = tensor->shape[0];
            const uint64_t cols = tensor->shape[1];
            const uint32_t group_size_raw = tensor->flags & ATT1_Q4_FLAGS_GROUP_MASK;
            const uint32_t group_size = (group_size_raw == 0u)
                                        ? ATT1_Q4_GROUP_SIZE_DEFAULT
                                        : group_size_raw;
            const uint64_t n_groups = rows * (cols / (uint64_t)group_size);
            const uint64_t packed_bytes = rows * cols / 2u;
            const uint64_t scale_bytes  = n_groups * sizeof(float);
            printf(" quant=grouped-q4-g%u"
                   " q4_groups=%" PRIu64
                   " q4_packed_bytes=%" PRIu64
                   " q4_scale_bytes=%" PRIu64,
                   group_size, n_groups, packed_bytes, scale_bytes);
        } else {
            printf(" quant=none");
        }
        printf(" offset=%" PRIu64 " nbytes=%" PRIu64 " shard=%u flags=%u\n",
               tensor->offset,
               tensor->nbytes,
               tensor->shard_id,
               tensor->flags);
    }

    if (model.shard_meta.count > 0u) {
        att1_shard_meta_summary    summary;
        att1_shard_meta_validation validation;

        att1_shard_meta_summarize(&model.shard_meta, &summary);

        printf("shard_meta: %" PRIu64 " records\n", model.shard_meta.count);
        printf("shard_meta_tiles=%" PRIu32 "\n", summary.unique_tiles);
        printf("shard_meta_aimus=%" PRIu32 "\n", summary.unique_aimus);
        printf("shard_meta_assigned=%" PRIu64 "\n", summary.assigned);
        printf("shard_meta_unassigned=%" PRIu64 "\n", summary.unassigned);
        printf("shard_meta_dtype_f32=%" PRIu64 "\n", summary.dtype_f32);
        printf("shard_meta_dtype_q8=%" PRIu64 "\n", summary.dtype_q8);

        if (att1_shard_meta_validate(&model.shard_meta,
                                     &model.config,
                                     model.tensors,
                                     model.tensor_count,
                                     &validation) == ATT1_OK) {
            if (validation.count > 0u) {
                printf("shard_meta_violations: %" PRIu64 "\n", validation.count);
                for (i = 0u; i < validation.count; i++) {
                    printf("  violation[%" PRIu64 "] tensor_id=%u field=%s: %s\n",
                           i,
                           validation.violations[i].tensor_id,
                           validation.violations[i].field,
                           validation.violations[i].description);
                }
            }
            att1_shard_meta_validation_free(&validation);
        }

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

        /* Proposed plan derived from metadata vs runtime plan */
        {
            att1_meta_plan      proposed;
            att1_shard_plan     runtime;
            size_t              tile_count;

            tile_count = (model.config.n_tiles > 0u)
                         ? (size_t)model.config.n_tiles
                         : 1u;

            if (att1_meta_plan_build(&model, &proposed) == ATT1_OK) {
                printf("shard_meta_plan_entries=%" PRIu32 "\n",
                       proposed.count);
                printf("shard_meta_plan_extra=%" PRIu32 "\n",
                       proposed.extra);
                printf("shard_meta_plan_conflict=%" PRIu32 "\n",
                       proposed.conflict);

                if (att1_shard_plan_build(&runtime, &model,
                                         tile_count) == ATT1_OK) {
                    att1_meta_plan_diff diff;

                    if (att1_meta_plan_compare(&proposed, &runtime,
                                              &diff) == ATT1_OK) {
                        printf("shard_meta_plan_matching=%" PRIu32 "\n",
                               diff.matching);
                        printf("shard_meta_plan_mismatch=%" PRIu32 "\n",
                               diff.mismatch);
                        printf("shard_meta_plan_missing=%" PRIu32 "\n",
                               diff.missing);
                    }
                    att1_shard_plan_free(&runtime);
                }
                att1_meta_plan_free(&proposed);
            }
        }
    }

    /* Tokenizer metadata section (M54, version-2 models only). */
    if (model.tok_meta.present) {
        const att1_tok_meta *tm = &model.tok_meta;
        uint32_t k = 0u;

        printf("tok_meta: present\n");
        printf("tok_meta_schema_version=%u\n",  tm->schema_version);
        printf("tok_meta_type=%s\n",             att1_tok_type_name(tm->tokenizer_type));
        printf("tok_meta_vocab_size=%u\n",       tm->vocab_size);
        printf("tok_meta_bos=%d\n",              tm->bos_token_id);
        printf("tok_meta_eos=%d\n",              tm->eos_token_id);
        printf("tok_meta_pad=%d\n",              tm->pad_token_id);
        printf("tok_meta_unk=%d\n",              tm->unk_token_id);
        printf("tok_meta_byte_fallback=%u\n",    tm->byte_fallback);
        printf("tok_meta_normalization=%s\n",    att1_tok_norm_name(tm->normalization_policy));
        printf("tok_meta_pretokenizer=%s\n",     att1_tok_pretok_name(tm->pretokenizer_policy));
        if (tm->asset_hash_kind == ATT1_TOK_HASH_SHA256) {
            printf("tok_meta_asset_hash=");
            for (k = 0u; k < 32u; k++) {
                printf("%02x", tm->asset_hash[k]);
            }
            printf("\n");
        } else {
            printf("tok_meta_asset_hash=none\n");
        }
    } else {
        printf("tok_meta: absent\n");
    }

    att1_model_free(&model);
    return 0;
}
