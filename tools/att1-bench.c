#include "att1_backend.h"
#include "att1_cluster_infer.h"
#include "att1_infer.h"
#include "att1_model.h"
#include "att1_shard_meta.h"
#include "att1_tok_ext.h"
#include "att1_trace.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *argv0)
{
    printf("usage: %s --model PATH --tokens N --mode single|cluster"
           " [--prompt TEXT]"
           " [--tiles N] [--backend cpu-f32|cpu-q8|cpu-q4|cuda|cuda-q8]"
           " [--shard-plan runtime|metadata]"
           " [--tokenizer byte|metadata|external]"
           " [--input-token-ids \"1,2,3\"]"
           " [--tokens-file PATH]\n"
           "\n"
           "byte mode (default): --prompt TEXT is required\n"
           "external mode:       --input-token-ids or --tokens-file is required;\n"
           "                     --prompt is not used\n",
           argv0);
}

/*
 * check_tokenizer_mode() validates the requested tokenizer mode against the
 * loaded model.  "byte" (default) and "external" always succeed here.
 * "metadata" performs full M57 runtime compatibility checks then fails with
 * "not implemented yet".  Returns 0 on success, 1 on any error.
 *
 * Note: external mode token-ID parsing and range validation are handled
 * separately in main() after this call.
 */
static int check_tokenizer_mode(const att1_model *model, const char *mode)
{
    if ((mode == NULL) || (strcmp(mode, "byte") == 0)) {
        return 0;
    }

    if (strcmp(mode, "metadata") == 0) {
        att1_status_t rc = ATT1_OK;

        if (!model->tok_meta.present) {
            fputs("error: tokenizer metadata absent\n", stderr);
            return 1;
        }

        rc = att1_tok_meta_check_runtime(&model->tok_meta,
                                         model->config.vocab_size);
        if (rc == ATT1_ERR_UNSUPPORTED) {
            fprintf(stderr, "error: tokenizer type unsupported: %s\n",
                    att1_tok_type_name(model->tok_meta.tokenizer_type));
            return 1;
        }
        if (rc != ATT1_OK) {
            fputs("error: tokenizer metadata incompatible with model\n",
                  stderr);
            return 1;
        }

        fputs("error: metadata tokenizer runtime not implemented yet\n",
              stderr);
        return 1;
    }

    if (strcmp(mode, "external") == 0) {
        /* Token-ID source is validated in main(); no error here. */
        return 0;
    }

    /* unreachable: invalid mode is caught at parse time */
    fputs("error: unknown tokenizer mode\n", stderr);
    return 1;
}

static int parse_size(const char *text, size_t *out)
{
    char *end = NULL;
    unsigned long value = 0u;

    if ((text == NULL) || (out == NULL)) {
        return -1;
    }

    value = strtoul(text, &end, 10);
    if ((end == text) || (*end != '\0')) {
        return -1;
    }

    *out = (size_t)value;
    return 0;
}

static size_t benchmark_token_count(const att1_model *model,
                                    size_t prompt_bytes,
                                    size_t requested_tokens)
{
    size_t available = 0u;

    if ((model == NULL) || (prompt_bytes == 0u) ||
        (prompt_bytes > model->config.max_seq_len)) {
        return requested_tokens;
    }

    available = (size_t)model->config.max_seq_len - prompt_bytes + 1u;
    return requested_tokens < available ? requested_tokens : available;
}

static att1_status_t create_backend(const char *name, att1_backend **out_backend)
{
    if (out_backend == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }
    *out_backend = NULL;

    if ((name == NULL) || (strcmp(name, "cpu-f32") == 0)) {
        return att1_backend_cpu_f32_create(out_backend);
    }
    if (strcmp(name, "cpu-q8") == 0) {
        return att1_backend_cpu_q8_create(out_backend);
    }
    if (strcmp(name, "cpu-q4") == 0) {
        return att1_backend_cpu_q4_create(out_backend);
    }
    if (strcmp(name, "cuda") == 0) {
        return att1_backend_cuda_create(out_backend);
    }
    if (strcmp(name, "cuda-q8") == 0) {
        return att1_backend_cuda_q8_create(out_backend);
    }
    if (strcmp(name, "cuda-q4") == 0) {
        fputs("error: cuda-q4 backend is not supported\n", stderr);
        return ATT1_ERR_UNSUPPORTED;
    }

    return ATT1_ERR_INVALID_ARG;
}

static void print_shard_meta_summary(const att1_shard_meta *meta)
{
    if (meta->count == 0u) {
        printf("shard_meta=absent\n");
        return;
    }

    {
        att1_shard_meta_summary summary;

        att1_shard_meta_summarize(meta, &summary);
        printf("shard_meta=present\n");
        printf("shard_meta_count=%" PRIu64 "\n", summary.count);
        printf("shard_meta_assigned=%" PRIu64 "\n", summary.assigned);
        printf("shard_meta_unassigned=%" PRIu64 "\n", summary.unassigned);
        printf("shard_meta_tiles=%" PRIu32 "\n", summary.unique_tiles);
        printf("shard_meta_aimus=%" PRIu32 "\n", summary.unique_aimus);
        printf("shard_meta_dtype_f32=%" PRIu64 "\n", summary.dtype_f32);
        printf("shard_meta_dtype_q8=%" PRIu64 "\n", summary.dtype_q8);
    }
}

static void print_counters(const att1_trace_t *trace)
{
    att1_trace_counters counters;
    size_t i = 0u;

    if (att1_trace_snapshot(trace, &counters) != ATT1_OK) {
        return;
    }

    printf("tokens_decoded=%llu\n",
           (unsigned long long)counters.tokens_decoded);
    printf("token_time_us_total=%llu\n",
           (unsigned long long)counters.token_time_us_total);
    printf("token_time_us_max=%llu\n",
           (unsigned long long)counters.token_time_us_max);
    printf("layer_time_us_total=%llu\n",
           (unsigned long long)counters.layer_time_us_total);
    printf("activation_bytes_sent=%llu\n",
           (unsigned long long)counters.activation_bytes_sent);
    printf("logits_bytes_produced=%llu\n",
           (unsigned long long)counters.logits_bytes_produced);
    printf("fabric_packets_sent=%llu\n",
           (unsigned long long)counters.fabric_packets_sent);
    printf("fabric_packets_received=%llu\n",
           (unsigned long long)counters.fabric_packets_received);
    printf("fabric_payload_bytes_sent=%llu\n",
           (unsigned long long)counters.fabric_payload_bytes_sent);
    printf("fabric_payload_bytes_received=%llu\n",
           (unsigned long long)counters.fabric_payload_bytes_received);
    printf("kv_appends=%llu\n",
           (unsigned long long)counters.kv_appends);
    printf("kv_key_reads=%llu\n",
           (unsigned long long)counters.kv_key_reads);
    printf("kv_value_reads=%llu\n",
           (unsigned long long)counters.kv_value_reads);
    printf("tile_layer_executions=%llu\n",
           (unsigned long long)counters.tile_layer_executions);

    for (i = 0u; i < att1_trace_layer_count(trace); i++) {
        att1_trace_layer layer;
        if (att1_trace_layer_snapshot(trace, i, &layer) == ATT1_OK) {
            printf("layer[%zu].executions=%llu time_us=%llu kv_appends=%llu\n",
                   i,
                   (unsigned long long)layer.executions,
                   (unsigned long long)layer.time_us_total,
                   (unsigned long long)layer.kv_appends);
        }
    }

    for (i = 0u; i < att1_trace_tile_count(trace); i++) {
        att1_trace_tile tile;
        if (att1_trace_tile_snapshot(trace, i, &tile) == ATT1_OK) {
            printf("tile[%zu].layers=%llu activation_bytes_sent=%llu logits_bytes=%llu\n",
                   i,
                   (unsigned long long)tile.layer_executions,
                   (unsigned long long)tile.activation_bytes_sent,
                   (unsigned long long)tile.logits_bytes_produced);
        }
    }
}

/*
 * run_single_external() runs single-tile inference using pre-tokenized input.
 *
 * ext_ids[0..ext_count-1] are fed via att1_infer_decode_token() as the
 * "prompt" phase.  max_tokens additional tokens are then generated and
 * collected in the output.  Reports tokenizer=external in bench output.
 */
static int run_single_external(const att1_model *model,
                                const uint32_t  *ext_ids,
                                size_t           ext_count,
                                size_t           max_tokens,
                                const char      *backend_name)
{
    att1_infer_t *infer = NULL;
    att1_trace_t *trace = NULL;
    att1_backend *backend = NULL;
    uint32_t *tokens = NULL;
    uint32_t token = 0u;
    size_t i = 0u;
    size_t produced = 0u;
    size_t run_tokens = 0u;
    size_t capacity = 0u;
    int rc = 1;
    int is_q4 = (backend_name != NULL) &&
                (strcmp(backend_name, "cpu-q4") == 0);

    if (ext_count >= (size_t)model->config.max_seq_len) {
        fputs("error: external token count fills or exceeds model max_seq_len\n",
              stderr);
        return 1;
    }

    run_tokens = benchmark_token_count(model, ext_count, max_tokens);
    capacity   = run_tokens == 0u ? 1u : run_tokens;

    tokens = calloc(capacity, sizeof(*tokens));
    if (tokens == NULL) {
        return 1;
    }

    if ((att1_trace_create(model->config.n_layers, 1u, &trace) != ATT1_OK)) {
        goto cleanup;
    }

    if (is_q4) {
        if (att1_infer_create_q4(model, &infer) != ATT1_OK) {
            fputs("error: cpu-q4 single inference failed (model may not contain q4 weights)\n",
                  stderr);
            goto cleanup;
        }
    } else {
        if (att1_infer_create(model, &infer) != ATT1_OK) {
            goto cleanup;
        }
        if (backend_name != NULL) {
            if (create_backend(backend_name, &backend) != ATT1_OK) {
                fprintf(stderr, "backend unsupported or unavailable: %s\n",
                        backend_name);
                goto cleanup;
            }
            if (att1_infer_set_backend(infer, backend) != ATT1_OK) {
                att1_backend_destroy(backend);
                backend = NULL;
                goto cleanup;
            }
            backend = NULL;
        }
    }

    if (att1_infer_set_trace(infer, trace) != ATT1_OK) {
        goto cleanup;
    }

    /* feed external prompt IDs one by one */
    for (i = 0u; i < ext_count; i++) {
        uint32_t next = 0u;
        if (att1_infer_decode_token(infer, ext_ids[i], &next) != ATT1_OK) {
            goto cleanup;
        }
        token = next;
    }

    /* generate run_tokens output tokens */
    for (produced = 0u; produced < run_tokens; produced++) {
        uint32_t next = 0u;
        tokens[produced] = token;
        if ((produced + 1u) < run_tokens) {
            if (att1_infer_decode_token(infer, token, &next) != ATT1_OK) {
                goto cleanup;
            }
            token = next;
        }
    }

    printf("mode=single\n");
    printf("shard_plan=runtime\n");
    printf("backend=%s\n", backend_name != NULL ? backend_name : "cpu-f32");
    printf("tokenizer=external\n");
    printf("prompt_tokens=%zu\n", ext_count);
    printf("requested_tokens=%zu\n", max_tokens);
    printf("benchmark_tokens=%zu\n", run_tokens);
    printf("generated_tokens=%zu\n", produced);
    if (produced > 0u) {
        printf("last_token=%u\n", tokens[produced - 1u]);
    }
    print_counters(trace);
    rc = 0;

cleanup:
    att1_backend_destroy(backend);
    att1_infer_destroy(infer);
    att1_trace_destroy(trace);
    free(tokens);
    return rc;
}

/*
 * run_cluster_external() runs cluster inference using pre-tokenized input.
 *
 * Same semantics as run_single_external() but uses the multi-tile cluster
 * inference context.
 */
static int run_cluster_external(const att1_model      *model,
                                 const uint32_t        *ext_ids,
                                 size_t                 ext_count,
                                 size_t                 max_tokens,
                                 size_t                 tile_count,
                                 const char            *backend_name,
                                 att1_shard_plan_mode   shard_plan_mode)
{
    att1_cluster_infer_config config;
    att1_cluster_infer_t *infer = NULL;
    att1_trace_t *trace = NULL;
    att1_backend *backend = NULL;
    uint32_t *tokens = NULL;
    uint32_t token = 0u;
    size_t i = 0u;
    size_t produced = 0u;
    size_t run_tokens = 0u;
    size_t capacity = 0u;
    int rc = 1;
    int is_q4 = (backend_name != NULL) &&
                (strcmp(backend_name, "cpu-q4") == 0);

    if (ext_count >= (size_t)model->config.max_seq_len) {
        fputs("error: external token count fills or exceeds model max_seq_len\n",
              stderr);
        return 1;
    }

    run_tokens = benchmark_token_count(model, ext_count, max_tokens);
    capacity   = run_tokens == 0u ? 1u : run_tokens;

    if (!is_q4 && (backend_name != NULL)) {
        if (create_backend(backend_name, &backend) != ATT1_OK) {
            fprintf(stderr, "backend unsupported or unavailable: %s\n",
                    backend_name);
            return 1;
        }
    }

    tokens = calloc(capacity, sizeof(*tokens));
    if (tokens == NULL) {
        goto cleanup;
    }

    config.tile_count              = tile_count;
    config.fabric_queue_capacity   = 4u;
    config.fabric_max_payload_bytes = 0u;
    config.shard_plan_mode         = shard_plan_mode;

    if (att1_trace_create(model->config.n_layers, tile_count, &trace) != ATT1_OK) {
        goto cleanup;
    }

    if (is_q4) {
        if (att1_cluster_infer_create_q4(model, &config, &infer) != ATT1_OK) {
            if (shard_plan_mode == ATT1_SHARD_PLAN_METADATA) {
                fputs("error: metadata shard plan not supported with cpu-q4\n",
                      stderr);
            } else {
                fputs("error: cpu-q4 cluster create failed\n", stderr);
            }
            goto cleanup;
        }
    } else {
        if (att1_cluster_infer_create(model, &config, &infer) != ATT1_OK) {
            if (shard_plan_mode == ATT1_SHARD_PLAN_METADATA) {
                fputs("error: metadata shard plan invalid, incomplete, or absent\n",
                      stderr);
            }
            goto cleanup;
        }
        if (backend != NULL) {
            if (att1_cluster_infer_set_backend(infer, backend) != ATT1_OK) {
                goto cleanup;
            }
            backend = NULL;
        }
    }

    if (att1_cluster_infer_set_trace(infer, trace) != ATT1_OK) {
        goto cleanup;
    }

    /* feed external prompt IDs one by one */
    for (i = 0u; i < ext_count; i++) {
        uint32_t next = 0u;
        if (att1_cluster_infer_decode_token(infer, ext_ids[i], &next) != ATT1_OK) {
            goto cleanup;
        }
        token = next;
    }

    /* generate run_tokens output tokens */
    for (produced = 0u; produced < run_tokens; produced++) {
        uint32_t next = 0u;
        tokens[produced] = token;
        if ((produced + 1u) < run_tokens) {
            if (att1_cluster_infer_decode_token(infer, token, &next) != ATT1_OK) {
                goto cleanup;
            }
            token = next;
        }
    }

    printf("mode=cluster\n");
    printf("shard_plan=%s\n",
           shard_plan_mode == ATT1_SHARD_PLAN_METADATA ? "metadata" : "runtime");
    printf("backend=%s\n", backend_name != NULL ? backend_name : "cpu-f32");
    printf("tokenizer=external\n");
    printf("tiles=%zu\n", tile_count);
    printf("prompt_tokens=%zu\n", ext_count);
    printf("requested_tokens=%zu\n", max_tokens);
    printf("benchmark_tokens=%zu\n", run_tokens);
    printf("generated_tokens=%zu\n", produced);
    if (produced > 0u) {
        printf("last_token=%u\n", tokens[produced - 1u]);
    }
    print_counters(trace);
    rc = 0;

cleanup:
    att1_backend_destroy(backend);
    att1_cluster_infer_destroy(infer);
    att1_trace_destroy(trace);
    free(tokens);
    return rc;
}

static int run_single(const att1_model *model,
                      const unsigned char *prompt,
                      size_t prompt_bytes,
                      size_t max_tokens,
                      const char *backend_name,
                      const char *tokenizer_name)
{
    att1_infer_t *infer = NULL;
    att1_trace_t *trace = NULL;
    att1_backend *backend = NULL;
    uint32_t *tokens = NULL;
    size_t out_count = 0u;
    size_t run_tokens = benchmark_token_count(model, prompt_bytes, max_tokens);
    size_t capacity = run_tokens == 0u ? 1u : run_tokens;
    int rc = 1;
    int is_q4 = (backend_name != NULL) &&
                (strcmp(backend_name, "cpu-q4") == 0);

    tokens = calloc(capacity, sizeof(*tokens));
    if (tokens == NULL) {
        return 1;
    }

    if (att1_trace_create(model->config.n_layers, 1u, &trace) != ATT1_OK) {
        goto cleanup;
    }

    if (is_q4) {
        if (att1_infer_create_q4(model, &infer) != ATT1_OK) {
            fputs("error: cpu-q4 single inference failed (model may not contain q4 weights)\n",
                  stderr);
            goto cleanup;
        }
    } else {
        if (att1_infer_create(model, &infer) != ATT1_OK) {
            goto cleanup;
        }
        if (backend_name != NULL) {
            if (create_backend(backend_name, &backend) != ATT1_OK) {
                fprintf(stderr, "backend unsupported or unavailable: %s\n",
                        backend_name);
                goto cleanup;
            }
            if (att1_infer_set_backend(infer, backend) != ATT1_OK) {
                att1_backend_destroy(backend);
                goto cleanup;
            }
            backend = NULL;
        }
    }

    if ((att1_infer_set_trace(infer, trace) != ATT1_OK) ||
        (att1_infer_generate(infer,
                             prompt,
                             prompt_bytes,
                             run_tokens,
                             tokens,
                             capacity,
                             &out_count) != ATT1_OK)) {
        goto cleanup;
    }

    printf("mode=single\n");
    printf("shard_plan=runtime\n");
    printf("backend=%s\n", backend_name != NULL ? backend_name : "cpu-f32");
    printf("tokenizer=%s\n", tokenizer_name != NULL ? tokenizer_name : "byte");
    printf("requested_tokens=%zu\n", max_tokens);
    printf("benchmark_tokens=%zu\n", run_tokens);
    printf("generated_tokens=%zu\n", out_count);
    if (out_count > 0u) {
        printf("last_token=%u\n", tokens[out_count - 1u]);
    }
    print_counters(trace);
    rc = 0;

cleanup:
    att1_backend_destroy(backend);
    att1_infer_destroy(infer);
    att1_trace_destroy(trace);
    free(tokens);
    return rc;
}

static int run_cluster(const att1_model *model,
                       const unsigned char *prompt,
                       size_t prompt_bytes,
                       size_t max_tokens,
                       size_t tile_count,
                       const char *backend_name,
                       att1_shard_plan_mode shard_plan_mode,
                       const char *tokenizer_name)
{
    att1_cluster_infer_config config;
    att1_cluster_infer_t *infer = NULL;
    att1_trace_t *trace = NULL;
    att1_backend *backend = NULL;
    uint32_t *tokens = NULL;
    size_t out_count = 0u;
    size_t run_tokens = benchmark_token_count(model, prompt_bytes, max_tokens);
    size_t capacity = run_tokens == 0u ? 1u : run_tokens;
    int rc = 1;
    int is_q4 = (backend_name != NULL) &&
                (strcmp(backend_name, "cpu-q4") == 0);

    if (!is_q4 && (backend_name != NULL)) {
        if (create_backend(backend_name, &backend) != ATT1_OK) {
            fprintf(stderr, "backend unsupported or unavailable: %s\n", backend_name);
            return 1;
        }
    }

    tokens = calloc(capacity, sizeof(*tokens));
    if (tokens == NULL) {
        goto cleanup;
    }

    config.tile_count = tile_count;
    config.fabric_queue_capacity = 4u;
    config.fabric_max_payload_bytes = 0u;
    config.shard_plan_mode = shard_plan_mode;

    if (att1_trace_create(model->config.n_layers, tile_count, &trace) != ATT1_OK) {
        goto cleanup;
    }

    if (is_q4) {
        if (att1_cluster_infer_create_q4(model, &config, &infer) != ATT1_OK) {
            if (shard_plan_mode == ATT1_SHARD_PLAN_METADATA) {
                fputs("error: metadata shard plan not supported with cpu-q4\n",
                      stderr);
            } else {
                fputs("error: cpu-q4 cluster create failed\n", stderr);
            }
            goto cleanup;
        }
    } else {
        if (att1_cluster_infer_create(model, &config, &infer) != ATT1_OK) {
            if (shard_plan_mode == ATT1_SHARD_PLAN_METADATA) {
                fputs("error: metadata shard plan invalid, incomplete, or absent\n",
                      stderr);
            }
            goto cleanup;
        }
        if (backend != NULL) {
            if (att1_cluster_infer_set_backend(infer, backend) != ATT1_OK) {
                goto cleanup;
            }
            backend = NULL;
        }
    }

    if ((att1_cluster_infer_set_trace(infer, trace) != ATT1_OK) ||
        (att1_cluster_infer_generate(infer,
                                     prompt,
                                     prompt_bytes,
                                     run_tokens,
                                     tokens,
                                     capacity,
                                     &out_count) != ATT1_OK)) {
        goto cleanup;
    }

    printf("mode=cluster\n");
    printf("shard_plan=%s\n",
           shard_plan_mode == ATT1_SHARD_PLAN_METADATA ? "metadata" : "runtime");
    printf("backend=%s\n", backend_name != NULL ? backend_name : "cpu-f32");
    printf("tokenizer=%s\n", tokenizer_name != NULL ? tokenizer_name : "byte");
    printf("tiles=%zu\n", tile_count);
    printf("requested_tokens=%zu\n", max_tokens);
    printf("benchmark_tokens=%zu\n", run_tokens);
    printf("generated_tokens=%zu\n", out_count);
    if (out_count > 0u) {
        printf("last_token=%u\n", tokens[out_count - 1u]);
    }
    print_counters(trace);
    rc = 0;

cleanup:
    att1_backend_destroy(backend);
    att1_cluster_infer_destroy(infer);
    att1_trace_destroy(trace);
    free(tokens);
    return rc;
}

int main(int argc, char **argv)
{
    const char *model_path = NULL;
    const char *prompt = NULL;
    const char *mode = NULL;
    const char *backend_name = NULL;
    const char *shard_plan_name = NULL;
    const char *tokenizer_name = "byte";
    const char *ext_ids_str = NULL;
    const char *ext_ids_file = NULL;
    size_t max_tokens = 0u;
    size_t tile_count = 2u;
    att1_shard_plan_mode shard_plan_mode = ATT1_SHARD_PLAN_RUNTIME;
    att1_model model;
    uint32_t *ext_ids = NULL;
    size_t ext_count = 0u;
    int i = 0;
    int rc = 1;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else if ((strcmp(argv[i], "--model") == 0) && ((i + 1) < argc)) {
            model_path = argv[++i];
        } else if ((strcmp(argv[i], "--prompt") == 0) && ((i + 1) < argc)) {
            prompt = argv[++i];
        } else if ((strcmp(argv[i], "--tokens") == 0) && ((i + 1) < argc)) {
            if (parse_size(argv[++i], &max_tokens) != 0) {
                usage(argv[0]);
                return 1;
            }
        } else if ((strcmp(argv[i], "--mode") == 0) && ((i + 1) < argc)) {
            mode = argv[++i];
        } else if ((strcmp(argv[i], "--backend") == 0) && ((i + 1) < argc)) {
            backend_name = argv[++i];
            if ((strcmp(backend_name, "cpu-f32") != 0) &&
                (strcmp(backend_name, "cpu-q8") != 0) &&
                (strcmp(backend_name, "cpu-q4") != 0) &&
                (strcmp(backend_name, "cuda") != 0) &&
                (strcmp(backend_name, "cuda-q8") != 0) &&
                (strcmp(backend_name, "cuda-q4") != 0)) {
                usage(argv[0]);
                return 1;
            }
            if (strcmp(backend_name, "cuda-q4") == 0) {
                fputs("error: cuda-q4 backend is not supported\n", stderr);
                return 1;
            }
        } else if ((strcmp(argv[i], "--tiles") == 0) && ((i + 1) < argc)) {
            if ((parse_size(argv[++i], &tile_count) != 0) ||
                (tile_count == 0u)) {
                usage(argv[0]);
                return 1;
            }
        } else if ((strcmp(argv[i], "--shard-plan") == 0) && ((i + 1) < argc)) {
            shard_plan_name = argv[++i];
            if (strcmp(shard_plan_name, "runtime") == 0) {
                shard_plan_mode = ATT1_SHARD_PLAN_RUNTIME;
            } else if (strcmp(shard_plan_name, "metadata") == 0) {
                shard_plan_mode = ATT1_SHARD_PLAN_METADATA;
            } else {
                usage(argv[0]);
                return 1;
            }
        } else if ((strcmp(argv[i], "--tokenizer") == 0) && ((i + 1) < argc)) {
            tokenizer_name = argv[++i];
            if ((strcmp(tokenizer_name, "byte")     != 0) &&
                (strcmp(tokenizer_name, "metadata") != 0) &&
                (strcmp(tokenizer_name, "external") != 0)) {
                usage(argv[0]);
                return 1;
            }
        } else if ((strcmp(argv[i], "--input-token-ids") == 0) &&
                   ((i + 1) < argc)) {
            ext_ids_str = argv[++i];
        } else if ((strcmp(argv[i], "--tokens-file") == 0) &&
                   ((i + 1) < argc)) {
            ext_ids_file = argv[++i];
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if ((model_path == NULL) || (mode == NULL)) {
        usage(argv[0]);
        return 1;
    }

    /* Validate mode-specific required arguments. */
    if (strcmp(tokenizer_name, "external") == 0) {
        /* External mode: require --input-token-ids or --tokens-file, not both.
         * --prompt is not used. */
        if ((ext_ids_str == NULL) && (ext_ids_file == NULL)) {
            fputs("error: --tokenizer external requires "
                  "--input-token-ids or --tokens-file\n", stderr);
            return 1;
        }
        if ((ext_ids_str != NULL) && (ext_ids_file != NULL)) {
            fputs("error: --input-token-ids and --tokens-file "
                  "are mutually exclusive\n", stderr);
            return 1;
        }
    } else {
        /* Byte / metadata modes: --prompt is required. */
        if ((prompt == NULL) || (prompt[0] == '\0')) {
            usage(argv[0]);
            return 1;
        }
    }

    if (att1_model_load(model_path, &model) != ATT1_OK) {
        fprintf(stderr, "failed to load model: %s\n", model_path);
        return 1;
    }

    if (check_tokenizer_mode(&model, tokenizer_name) != 0) {
        att1_model_free(&model);
        return 1;
    }

    /* Parse and validate external token IDs (after model load, so we have
     * vocab_size for range checking). */
    if (strcmp(tokenizer_name, "external") == 0) {
        att1_status_t parse_rc = ATT1_OK;

        if (ext_ids_str != NULL) {
            parse_rc = att1_tok_ext_parse_ids_str(ext_ids_str,
                                                   model.config.vocab_size,
                                                   &ext_ids, &ext_count);
        } else {
            parse_rc = att1_tok_ext_parse_ids_file(ext_ids_file,
                                                    model.config.vocab_size,
                                                    &ext_ids, &ext_count);
        }

        if (parse_rc != ATT1_OK) {
            att1_model_free(&model);
            return 1;
        }
    }

    print_shard_meta_summary(&model.shard_meta);

    if (strcmp(tokenizer_name, "external") == 0) {
        /* Route to external inference. */
        if (strcmp(mode, "single") == 0) {
            rc = run_single_external(&model, ext_ids, ext_count,
                                      max_tokens, backend_name);
        } else if (strcmp(mode, "cluster") == 0) {
            rc = run_cluster_external(&model, ext_ids, ext_count,
                                       max_tokens, tile_count, backend_name,
                                       shard_plan_mode);
        } else {
            usage(argv[0]);
            rc = 1;
        }
    } else if (strcmp(mode, "single") == 0) {
        rc = run_single(&model,
                        (const unsigned char *)prompt,
                        strlen(prompt),
                        max_tokens,
                        backend_name,
                        tokenizer_name);
    } else if (strcmp(mode, "cluster") == 0) {
        rc = run_cluster(&model,
                         (const unsigned char *)prompt,
                         strlen(prompt),
                         max_tokens,
                         tile_count,
                         backend_name,
                         shard_plan_mode,
                         tokenizer_name);
    } else {
        usage(argv[0]);
        rc = 1;
    }

    free(ext_ids);
    att1_model_free(&model);
    return rc;
}
