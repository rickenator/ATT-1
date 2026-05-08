/*
 * att1-size — ATT-1 model size and placement estimator.
 *
 * Usage:
 *   att1-size --preset tiny-dummy|gpt-oss-120b-shape [--tiles N] [--context N] [--dtype f32|f16|q8|q4]
 *   att1-size --config PATH [--tiles N] [--context N] [--dtype f32|f16|q8|q4] [--json]
 *   att1-size --layers N --d-model N --heads N --d-ff N --vocab-size N [--context N] [--tiles N] [--dtype f32|f16|q8|q4] [--json]
 *
 * --config reads a JSON-subset config.json (LLaMA style).
 * --json prints a machine-readable JSON report.
 * --preset is non-JSON only (legacy output unchanged).
 * All estimates are architectural projections; --preset shapes marked
 * synthetic are not executable.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Shape descriptor
 * ---------------------------------------------------------------------- */

typedef struct model_shape {
    char     name[128];
    char     note[256];
    uint64_t vocab_size;
    uint64_t n_layers;
    uint64_t n_heads;
    uint64_t head_dim;       /* derived: d_model / n_heads */
    uint64_t d_model;
    uint64_t d_ff;
    uint64_t max_seq_len;
    uint64_t n_tiles;
    /* computed after load */
    uint64_t total_params;
    uint64_t active_params_per_token;
    /* source tag */
    int      is_preset;
    int      is_synthetic;   /* 1 = mark non-executable in output */
} model_shape;

/* Legacy alias used by existing smoke test */
typedef model_shape preset_shape;

/* M96: capacity planning options */
typedef struct capacity_opts {
    uint64_t tile_memory_mib;   /* per-tile SRAM budget in MiB; 0 = not specified */
    uint64_t sessions;          /* concurrent KV sessions; 0 = not specified (treated as 1) */
    uint64_t target_tps;        /* target decode tokens/sec; 0 = not specified */
    double   fabric_gib_sec;    /* fabric bandwidth GiB/sec; 0.0 = not specified */
} capacity_opts;

/* -------------------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------------- */

static uint64_t div_round_up(uint64_t value, uint64_t divisor)
{
    return (value + divisor - 1u) / divisor;
}

static void print_bytes_line(const char *label, uint64_t bytes)
{
    printf("%s=%llu bytes (%.3f GiB)\n",
           label,
           (unsigned long long)bytes,
           (double)bytes / (1024.0 * 1024.0 * 1024.0));
}

static void usage(const char *argv0)
{
    printf("usage:\n"
           "  %s --preset tiny-dummy|gpt-oss-120b-shape [--tiles N] [--context N] [--dtype f32|f16|q8|q4]\n"
           "  %s --config PATH [--tiles N] [--context N] [--dtype f32|f16|q8|q4] [--json]\n"
           "  %s --layers N --d-model N --heads N --d-ff N --vocab-size N [--context N] [--tiles N] [--dtype f32|f16|q8|q4] [--json]\n"
           "\nCapacity planning options (any mode):\n"
           "  --tile-memory-mib N      Per-tile SRAM budget in MiB\n"
           "  --tile-memory-gib N      Per-tile SRAM budget in GiB (sets mib = N*1024)\n"
           "  --sessions N             Concurrent KV sessions for pressure estimate\n"
           "  --target-tokens-per-sec N  Target decode rate (tokens/sec)\n"
           "  --fabric-gib-sec N       Fabric bandwidth (GiB/sec) for PASS/WARN/FAIL check\n"
           "\nM100 placement report:\n"
           "  --placement-report-json PATH  Write M98-schema tensor placement report JSON to PATH\n",
           argv0, argv0, argv0);
}

static int parse_u64(const char *text, uint64_t *out)
{
    char *end = NULL;
    unsigned long long value = 0u;

    if ((text == NULL) || (out == NULL)) {
        return -1;
    }

    value = strtoull(text, &end, 10);
    if ((end == text) || (*end != '\0')) {
        return -1;
    }

    *out = (uint64_t)value;
    return 0;
}

static int dtype_supported(const char *dtype)
{
    return (dtype == NULL) ||
           (strcmp(dtype, "f32") == 0) ||
           (strcmp(dtype, "f16") == 0) ||
           (strcmp(dtype, "q8") == 0) ||
           (strcmp(dtype, "q4") == 0);
}

static int parse_double(const char *text, double *out)
{
    char  *end = NULL;
    double v   = 0.0;

    if ((text == NULL) || (out == NULL)) {
        return -1;
    }
    v = strtod(text, &end);
    if ((end == text) || (*end != '\0')) {
        return -1;
    }
    *out = v;
    return 0;
}

/* Returns PASS/WARN/FAIL/UNKNOWN for model+KV combined bytes vs tile memory. */
static const char *tile_capacity_status(uint64_t combined_bytes, uint64_t tile_mib)
{
    uint64_t tile_bytes;

    if (tile_mib == 0u) {
        return "UNKNOWN";
    }
    tile_bytes = tile_mib * 1024u * 1024u;
    if (combined_bytes <= (tile_bytes * 8u / 10u)) {
        return "PASS";
    }
    if (combined_bytes <= tile_bytes) {
        return "WARN";
    }
    return "FAIL";
}

/* Returns PASS/WARN/FAIL/UNKNOWN for required vs available fabric bandwidth. */
static const char *fabric_bandwidth_status(double required_gib_sec,
                                           double available_gib_sec)
{
    if (available_gib_sec <= 0.0) {
        return "UNKNOWN";
    }
    if (required_gib_sec <= (available_gib_sec * 0.8)) {
        return "PASS";
    }
    if (required_gib_sec <= available_gib_sec) {
        return "WARN";
    }
    return "FAIL";
}

/* -------------------------------------------------------------------------
 * Parameter count formula for LLaMA-style models
 * ---------------------------------------------------------------------- */

static uint64_t llama_params(uint64_t vocab, uint64_t layers, uint64_t d_model, uint64_t d_ff)
{
    /* embeddings */
    const uint64_t tok_embed = vocab * d_model;
    /* per-layer: attention_norm(d_model) + wq+wk+wv+wo(4*d_model^2)
     *           + ffn_norm(d_model) + w_gate+w_up(2*d_model*d_ff) + w_down(d_ff*d_model) */
    const uint64_t per_layer =
        d_model +
        (4u * d_model * d_model) +
        d_model +
        (2u * d_model * d_ff) +
        (d_ff * d_model);
    /* output: output_norm(d_model) + output.weight(d_model*vocab) */
    const uint64_t output = d_model + (d_model * vocab);

    return tok_embed + (layers * per_layer) + output;
}

static uint64_t tiny_params(void)
{
    return llama_params(256u, 2u, 4u, 8u);
}

/* Per-category parameter counts (for storage breakdown) */
typedef struct param_categories {
    uint64_t embeddings;       /* tok_embeddings.weight */
    uint64_t attn_projections; /* wq + wk + wv + wo per layer (all layers) */
    uint64_t ffn_projections;  /* w_gate + w_up + w_down per layer (all layers) */
    uint64_t norms;            /* all rmsnorm weight vectors */
    uint64_t lm_head;          /* output.weight */
    uint64_t total;
} param_categories;

/* -------------------------------------------------------------------------
 * Category breakdown
 * ---------------------------------------------------------------------- */

static param_categories compute_categories(const model_shape *s)
{
    param_categories c;
    c.embeddings       = s->vocab_size * s->d_model;
    c.attn_projections = s->n_layers * 4u * s->d_model * s->d_model;
    c.ffn_projections  = s->n_layers * (2u * s->d_model * s->d_ff + s->d_ff * s->d_model);
    c.norms            = s->n_layers * 2u * s->d_model + s->d_model; /* attn_norm+ffn_norm per layer + output_norm */
    c.lm_head          = s->d_model * s->vocab_size;
    c.total            = c.embeddings + c.attn_projections + c.ffn_projections + c.norms + c.lm_head;
    return c;
}

/* -------------------------------------------------------------------------
 * KV cache estimate (f32 bytes)
 * ---------------------------------------------------------------------- */

static uint64_t kv_bytes_f32(const model_shape *s, uint64_t ctx)
{
    /* 2 (K+V) * layers * ctx * n_heads * head_dim * 4 bytes */
    return 2u * s->n_layers * ctx * s->n_heads * s->head_dim * 4u;
}

/* -------------------------------------------------------------------------
 * Preset loader (preserves existing name/note strings)
 * ---------------------------------------------------------------------- */

static int load_preset(const char *name, model_shape *out)
{
    if ((name == NULL) || (out == NULL)) {
        return -1;
    }

    memset(out, 0, sizeof(*out));
    out->is_preset = 1;
    out->n_tiles   = 1u;

    if (strcmp(name, "tiny-dummy") == 0) {
        strncpy(out->name, "tiny-dummy", sizeof(out->name) - 1u);
        strncpy(out->note, "Executable dummy ATT-1 model shape.", sizeof(out->note) - 1u);
        out->is_synthetic  = 0;
        out->vocab_size    = 256u;
        out->n_layers      = 2u;
        out->n_heads       = 2u;
        out->d_model       = 4u;
        out->d_ff          = 8u;
        out->head_dim      = out->d_model / out->n_heads;
        out->max_seq_len   = 8u;
        out->total_params  = tiny_params();
        out->active_params_per_token = tiny_params();
        return 0;
    }

    if (strcmp(name, "gpt-oss-120b-shape") == 0) {
        strncpy(out->name, "gpt-oss-120b-shape", sizeof(out->name) - 1u);
        strncpy(out->note, "synthetic/non-executable architecture estimate only; no real gpt-oss-120b inference is attempted.", sizeof(out->note) - 1u);
        out->is_synthetic  = 1;
        out->vocab_size    = 100352u;
        out->n_layers      = 96u;
        out->n_heads       = 96u;
        out->head_dim      = 128u;
        out->d_model       = out->n_heads * out->head_dim;
        out->d_ff          = out->d_model * 4u;
        out->max_seq_len   = 131072u;
        out->n_tiles       = 16u;
        out->total_params  = 120000000000ull;
        out->active_params_per_token = 5100000000ull;
        return 0;
    }

    return -1;
}

/* -------------------------------------------------------------------------
 * Minimal JSON config reader — handles LLaMA config.json fields only.
 * Supports both HF and compact field names.  No external deps.
 * ---------------------------------------------------------------------- */

/*
 * Find a JSON string or integer value for a given key.
 * Returns a pointer to the value's text start and sets *vlen to the
 * printable length of the value token (without surrounding quotes for
 * strings).  Returns NULL if not found.
 */
static const char *json_find_value(const char *json, const char *key,
                                    size_t *vlen)
{
    const char *p   = json;
    size_t      klen = strlen(key);

    while (*p != '\0') {
        /* Find the key (quoted) */
        const char *q = strstr(p, key);
        if (q == NULL) {
            return NULL;
        }
        /* Ensure it is preceded by a quote */
        if ((q == json) || (q[-1] != '"')) {
            p = q + 1;
            continue;
        }
        /* Ensure it is followed by a quote then colon (possibly with spaces) */
        q += klen;
        if (*q != '"') {
            p = q + 1;
            continue;
        }
        q++;
        while ((*q == ' ') || (*q == '\t') || (*q == '\n') || (*q == '\r')) { q++; }
        if (*q != ':') {
            p = q + 1;
            continue;
        }
        q++;
        /* Skip whitespace after colon */
        while ((*q == ' ') || (*q == '\t') || (*q == '\n') || (*q == '\r')) { q++; }
        /* Return pointer to value start */
        if (*q == '"') {
            /* String value — skip open quote */
            const char *vs = q + 1;
            const char *ve = strchr(vs, '"');
            if (ve == NULL) {
                return NULL;
            }
            *vlen = (size_t)(ve - vs);
            return vs;
        } else {
            /* Numeric or keyword value */
            const char *vs = q;
            const char *ve = vs;
            while ((*ve != '\0') && (*ve != ',') && (*ve != '}') &&
                   (*ve != '\n') && (*ve != '\r')) {
                ve++;
            }
            *vlen = (size_t)(ve - vs);
            return vs;
        }
    }
    return NULL;
}

static int json_get_u64(const char *json, const char *key, uint64_t *out)
{
    size_t      vlen = 0u;
    const char *vs   = json_find_value(json, key, &vlen);
    char        buf[32];
    char       *end  = NULL;

    if ((vs == NULL) || (vlen == 0u) || (vlen >= sizeof(buf))) {
        return -1;
    }
    memcpy(buf, vs, vlen);
    buf[vlen] = '\0';
    *out = (uint64_t)strtoull(buf, &end, 10);
    return (end == buf) ? -1 : 0;
}

/*
 * Read a config.json (LLaMA-style subset) and populate a model_shape.
 * Returns 0 on success, -1 on error (with message to stderr).
 */
static int load_config(const char *path, model_shape *out)
{
    FILE       *fp    = NULL;
    char       *json  = NULL;
    long        fsize = 0;
    size_t      nread = 0u;
    uint64_t    v     = 0u;

    if ((path == NULL) || (out == NULL)) {
        return -1;
    }

    fp = fopen(path, "rb");
    if (fp == NULL) {
        fprintf(stderr, "att1-size: cannot open config: %s\n", path);
        return -1;
    }

    if ((fseek(fp, 0, SEEK_END) != 0) || ((fsize = ftell(fp)) < 0)) {
        fclose(fp);
        fprintf(stderr, "att1-size: cannot stat config: %s\n", path);
        return -1;
    }
    rewind(fp);

    if (fsize > 1048576L) {
        fclose(fp);
        fprintf(stderr, "att1-size: config file too large: %s\n", path);
        return -1;
    }

    json = (char *)malloc((size_t)fsize + 1u);
    if (json == NULL) {
        fclose(fp);
        return -1;
    }

    nread = fread(json, 1u, (size_t)fsize, fp);
    fclose(fp);
    json[nread] = '\0';

    memset(out, 0, sizeof(*out));
    out->is_preset    = 0;
    out->is_synthetic = 0;
    out->n_tiles      = 1u;
    strncpy(out->name, path, sizeof(out->name) - 1u);
    strncpy(out->note, "from config.json", sizeof(out->note) - 1u);

    /* vocab_size */
    if (json_get_u64(json, "vocab_size", &v) != 0) {
        fprintf(stderr, "att1-size: missing vocab_size in %s\n", path);
        free(json);
        return -1;
    }
    out->vocab_size = v;

    /* n_layers */
    if ((json_get_u64(json, "num_hidden_layers", &v) != 0) &&
        (json_get_u64(json, "n_layers", &v) != 0)) {
        fprintf(stderr, "att1-size: missing n_layers/num_hidden_layers in %s\n", path);
        free(json);
        return -1;
    }
    out->n_layers = v;

    /* n_heads */
    if ((json_get_u64(json, "num_attention_heads", &v) != 0) &&
        (json_get_u64(json, "n_heads", &v) != 0)) {
        fprintf(stderr, "att1-size: missing n_heads/num_attention_heads in %s\n", path);
        free(json);
        return -1;
    }
    out->n_heads = v;

    /* d_model / hidden_size */
    if ((json_get_u64(json, "hidden_size", &v) != 0) &&
        (json_get_u64(json, "d_model", &v) != 0)) {
        fprintf(stderr, "att1-size: missing hidden_size/d_model in %s\n", path);
        free(json);
        return -1;
    }
    out->d_model = v;

    /* d_ff / intermediate_size */
    if ((json_get_u64(json, "intermediate_size", &v) != 0) &&
        (json_get_u64(json, "d_ff", &v) != 0)) {
        fprintf(stderr, "att1-size: missing intermediate_size/d_ff in %s\n", path);
        free(json);
        return -1;
    }
    out->d_ff = v;

    /* max_position_embeddings (optional; defaults to 4096) */
    if (json_get_u64(json, "max_position_embeddings", &v) == 0) {
        out->max_seq_len = v;
    } else {
        out->max_seq_len = 4096u;
    }

    if (out->n_heads == 0u) {
        fprintf(stderr, "att1-size: n_heads must be > 0\n");
        free(json);
        return -1;
    }
    out->head_dim = out->d_model / out->n_heads;

    out->total_params = llama_params(out->vocab_size, out->n_layers, out->d_model, out->d_ff);
    out->active_params_per_token = out->total_params;

    free(json);
    return 0;
}

/* -------------------------------------------------------------------------
 * Report helpers
 * ---------------------------------------------------------------------- */

static void print_gib(uint64_t bytes)
{
    printf("%.3f GiB", (double)bytes / (1024.0 * 1024.0 * 1024.0));
}

static const char *feasible(uint64_t bytes, uint64_t threshold_bytes)
{
    return (bytes <= threshold_bytes) ? "likely_feasible" : "likely_unfeasible";
}

/* Backend thresholds (conservative, for planning only) */
#define CPU_RAM_THRESHOLD_BYTES  (64ull * 1024ull * 1024ull * 1024ull)  /* 64 GiB */
#define CUDA_VRAM_THRESHOLD_BYTES (24ull * 1024ull * 1024ull * 1024ull) /* 24 GiB (consumer 3090) */

static void print_report_preset(const model_shape *s, const char *dtype)
{
    const uint64_t f32_bytes = s->total_params * 4u;
    const uint64_t f16_bytes = s->total_params * 2u;
    const uint64_t q8_bytes  = s->total_params;
    const uint64_t q4_bytes  = div_round_up(s->total_params, 2u);
    const uint64_t kv_f16    = s->n_layers * s->max_seq_len * s->n_heads * s->head_dim * 2u * 2u;
    const uint64_t act_f32   = s->d_model * 4u;

    printf("preset=%s\n", s->name);
    if (dtype != NULL) {
        printf("dtype=%s\n", dtype);
    }
    printf("note=%s\n", s->note);
    printf("total_params=%llu\n",          (unsigned long long)s->total_params);
    printf("active_params_per_token=%llu\n",(unsigned long long)s->active_params_per_token);
    printf("layers=%llu heads=%llu head_dim=%llu d_model=%llu max_seq_len=%llu tiles=%llu\n",
           (unsigned long long)s->n_layers, (unsigned long long)s->n_heads,
           (unsigned long long)s->head_dim, (unsigned long long)s->d_model,
           (unsigned long long)s->max_seq_len, (unsigned long long)s->n_tiles);
    print_bytes_line("model_bytes_f32",  f32_bytes);
    print_bytes_line("model_bytes_f16",  f16_bytes);
    print_bytes_line("model_bytes_q8",   q8_bytes);
    print_bytes_line("model_bytes_q4",   q4_bytes);
    print_bytes_line("estimated_bytes_per_tile_f16",
                     div_round_up(f16_bytes, s->n_tiles));
    print_bytes_line("kv_bytes_per_session_f16", kv_f16);
    print_bytes_line("activation_bytes_per_token_f32", act_f32);
}

/* Full scaling report (config/manual mode) */
static void print_report_full(const model_shape *s, const char *dtype, int json_out,
                               const capacity_opts *opts)
{
    static const uint64_t ctx_list[] = { 512u, 2048u, 8192u, 32768u };
    static const size_t   n_ctx = sizeof(ctx_list) / sizeof(ctx_list[0]);
    param_categories cat;
    uint64_t tile;
    size_t   ci;
    uint64_t i;

    /* Storage */
    const uint64_t f32_bytes  = s->total_params * 4u;
    const uint64_t f16_bytes  = s->total_params * 2u;
    const uint64_t q8_bytes   = s->total_params;
    const uint64_t q4_est     = div_round_up(s->total_params, 2u);

    /* KV cache */
    const uint64_t kv_f32_default = kv_bytes_f32(s, s->max_seq_len);

    /* Activation / logits / fabric per generated token */
    const uint64_t act_bytes      = s->d_model * 4u; /* f32 activation vector */
    const uint64_t logits_bytes   = s->vocab_size * 4u; /* f32 logits */
    /* fabric = activations sent to every non-first tile + logits sent back */
    const uint64_t fabric_act     = (s->n_tiles > 1u)
                                    ? (s->n_tiles - 1u) * act_bytes
                                    : 0u;
    const uint64_t fabric_total   = fabric_act + (s->n_tiles > 1u ? logits_bytes : 0u);

    cat = compute_categories(s);

    if (!json_out) {
        printf("--- ATT-1 Model Scaling Report ---\n");
        if (s->is_synthetic) {
            printf("NOTE: synthetic estimate only; not executable in ATT-1 runtime.\n");
        }
        printf("\n");

        /* Shape summary */
        printf("[shape]\n");
        printf("name           = %s\n", s->name);
        printf("source         = %s\n", s->is_preset ? "preset" : "config");
        printf("vocab_size     = %llu\n", (unsigned long long)s->vocab_size);
        printf("n_layers       = %llu\n", (unsigned long long)s->n_layers);
        printf("d_model        = %llu\n", (unsigned long long)s->d_model);
        printf("n_heads        = %llu\n", (unsigned long long)s->n_heads);
        printf("head_dim       = %llu\n", (unsigned long long)s->head_dim);
        printf("d_ff           = %llu\n", (unsigned long long)s->d_ff);
        printf("max_seq_len    = %llu\n", (unsigned long long)s->max_seq_len);
        printf("n_tiles        = %llu\n", (unsigned long long)s->n_tiles);
        printf("total_params   = %llu\n", (unsigned long long)s->total_params);
        printf("\n");

        /* Storage estimates */
        printf("[storage]\n");
        printf("  %-30s  %12llu bytes  (", "f32_total", (unsigned long long)f32_bytes);
        print_gib(f32_bytes); printf(")\n");
        printf("  %-30s  %12llu bytes  (", "f16_total", (unsigned long long)f16_bytes);
        print_gib(f16_bytes); printf(")\n");
        printf("  %-30s  %12llu bytes  (", "q8_total", (unsigned long long)q8_bytes);
        print_gib(q8_bytes); printf(")\n");
        printf("  %-30s  %12llu bytes  (", "q4_estimate_only", (unsigned long long)q4_est);
        print_gib(q4_est); printf(") [not implemented]\n");
        printf("\n");

        /* Category breakdown */
        printf("[storage_by_category_f32]\n");
        printf("  %-30s  %12llu bytes\n", "embeddings",         (unsigned long long)(cat.embeddings * 4u));
        printf("  %-30s  %12llu bytes\n", "attention_proj",     (unsigned long long)(cat.attn_projections * 4u));
        printf("  %-30s  %12llu bytes\n", "ffn_proj",           (unsigned long long)(cat.ffn_projections * 4u));
        printf("  %-30s  %12llu bytes\n", "norms",              (unsigned long long)(cat.norms * 4u));
        printf("  %-30s  %12llu bytes\n", "lm_head",            (unsigned long long)(cat.lm_head * 4u));
        printf("\n");

        /* KV cache */
        printf("[kv_cache_f32_bytes_by_context]\n");
        for (ci = 0; ci < n_ctx; ci++) {
            uint64_t ctx = ctx_list[ci];
            uint64_t kb  = kv_bytes_f32(s, ctx);
            if (ctx > s->max_seq_len * 4u) { continue; }
            printf("  ctx=%-6llu  %12llu bytes  (", (unsigned long long)ctx, (unsigned long long)kb);
            print_gib(kb); printf(")\n");
        }
        printf("  ctx=%-6llu  %12llu bytes  (", (unsigned long long)s->max_seq_len, (unsigned long long)kv_f32_default);
        print_gib(kv_f32_default); printf(") [model max]\n");
        printf("\n");

        /* Cluster / AIMU placement */
        printf("[cluster_placement (n_tiles=%llu)]\n", (unsigned long long)s->n_tiles);
        tile = (s->n_tiles > 0u) ? s->n_tiles : 1u;
        printf("  layers_per_tile    = %llu  (round-up)\n",
               (unsigned long long)div_round_up(s->n_layers, tile));
        printf("  model_bytes_per_tile_f32 = %llu bytes  (",
               (unsigned long long)div_round_up(f32_bytes, tile));
        print_gib(div_round_up(f32_bytes, tile)); printf(")\n");
        printf("  model_bytes_per_tile_q8  = %llu bytes  (",
               (unsigned long long)div_round_up(q8_bytes, tile));
        print_gib(div_round_up(q8_bytes, tile)); printf(")\n");
        printf("  kv_bytes_per_tile_f32    = %llu bytes  (",
               (unsigned long long)div_round_up(kv_f32_default, tile));
        print_gib(div_round_up(kv_f32_default, tile)); printf(")\n");
        printf("  activation_bytes_per_token_f32 = %llu bytes\n",
               (unsigned long long)act_bytes);
        printf("  logits_bytes_per_token_f32     = %llu bytes\n",
               (unsigned long long)logits_bytes);
        printf("  est_fabric_bytes_per_token     = %llu bytes\n",
               (unsigned long long)fabric_total);
        printf("\n");

        /* AIMU per-tile table */
        printf("[aimu_tile_plan]\n");
        printf("  %-8s  %-16s  %-14s  %-14s  %-14s\n",
               "tile_id", "layer_range", "model_f32_GiB",
               "kv_f32_GiB", "act_traffic_B");
        for (i = 0; i < tile; i++) {
            uint64_t first_layer = i * div_round_up(s->n_layers, tile);
            uint64_t last_layer  = first_layer + div_round_up(s->n_layers, tile) - 1u;
            uint64_t tile_layers;
            uint64_t tile_model;
            uint64_t tile_kv;

            if (first_layer >= s->n_layers) { break; }
            if (last_layer >= s->n_layers) { last_layer = s->n_layers - 1u; }
            tile_layers = last_layer - first_layer + 1u;

            /* Model bytes for these layers only (excludes embeddings / lm_head on tile 0) */
            {
                const uint64_t attn_per = 4u * s->d_model * s->d_model;
                const uint64_t ffn_per  = 2u * s->d_model * s->d_ff + s->d_ff * s->d_model;
                const uint64_t norm_per = 2u * s->d_model;
                const uint64_t layer_bytes = (attn_per + ffn_per + norm_per) * 4u; /* f32 */
                tile_model = tile_layers * layer_bytes;
                if (i == 0u) {
                    tile_model += (cat.embeddings + cat.norms - s->n_layers * 2u * s->d_model) * 4u;
                }
                if (i == (tile - 1u)) {
                    tile_model += cat.lm_head * 4u;
                }
            }
            tile_kv = tile_layers * s->max_seq_len * s->n_heads * s->head_dim * 2u * 4u;
            printf("  %-8llu  %5llu..%5llu      %14.3f  %14.3f  %14llu\n",
                   (unsigned long long)i,
                   (unsigned long long)first_layer,
                   (unsigned long long)last_layer,
                   (double)tile_model / (1024.0 * 1024.0 * 1024.0),
                   (double)tile_kv    / (1024.0 * 1024.0 * 1024.0),
                   (unsigned long long)act_bytes);
        }
        printf("\n");

        /* Backend feasibility */
        printf("[backend_feasibility] (conservative; RAM=64GiB, VRAM=24GiB)\n");
        printf("  cpu_f32  = %-20s  (model=", feasible(f32_bytes, CPU_RAM_THRESHOLD_BYTES));
        print_gib(f32_bytes); printf(")\n");
        printf("  cpu_q8   = %-20s  (model=", feasible(q8_bytes,  CPU_RAM_THRESHOLD_BYTES));
        print_gib(q8_bytes); printf(")\n");
        printf("  cuda_f32 = %-20s  (model=", feasible(f32_bytes, CUDA_VRAM_THRESHOLD_BYTES));
        print_gib(f32_bytes); printf(")\n");
        printf("  cuda_q8  = %-20s  (model=", feasible(q8_bytes,  CUDA_VRAM_THRESHOLD_BYTES));
        print_gib(q8_bytes); printf(")\n");
        printf("  NOTE: estimates are architectural projections, not measured throughput.\n");

        if (dtype != NULL) {
            printf("\ndtype_selected=%s\n", dtype);
        }

        /* M96: tile capacity estimate section */
        if ((opts != NULL) && (opts->tile_memory_mib != 0u)) {
            const uint64_t tiles_eff    = (s->n_tiles > 0u) ? s->n_tiles : 1u;
            const uint64_t per_tile_f32 = div_round_up(f32_bytes,      tiles_eff);
            const uint64_t per_tile_q8  = div_round_up(q8_bytes,       tiles_eff);
            const uint64_t kv_per_tile  = div_round_up(kv_f32_default, tiles_eff);
            const uint64_t sessions     = (opts->sessions > 0u) ? opts->sessions : 1u;
            const uint64_t combined     = per_tile_f32 + kv_per_tile * sessions;
            const uint64_t tile_bytes   = opts->tile_memory_mib * 1024u * 1024u;
            const double   util_pct     = (tile_bytes > 0u)
                                          ? 100.0 * (double)combined / (double)tile_bytes
                                          : 0.0;
            printf("\n[tile_capacity_estimate] (estimate only)\n");
            printf("  tile_memory_mib            = %llu\n",
                   (unsigned long long)opts->tile_memory_mib);
            printf("  sessions                   = %llu\n", (unsigned long long)sessions);
            printf("  model_bytes_per_tile_f32   = %llu bytes  (",
                   (unsigned long long)per_tile_f32);
            print_gib(per_tile_f32); printf(")\n");
            printf("  model_bytes_per_tile_q8    = %llu bytes  (",
                   (unsigned long long)per_tile_q8);
            print_gib(per_tile_q8); printf(")\n");
            printf("  kv_bytes_per_tile_f32      = %llu bytes  (",
                   (unsigned long long)kv_per_tile);
            print_gib(kv_per_tile); printf(") [model max ctx, per session]\n");
            printf("  combined_bytes_per_tile_f32= %llu bytes  (",
                   (unsigned long long)combined);
            print_gib(combined); printf(") [model_f32 + kv*sessions]\n");
            printf("  combined_utilization_pct   = %.1f%%\n", util_pct);
            printf("  tile_capacity_status       = %s\n",
                   tile_capacity_status(combined, opts->tile_memory_mib));
        }

        /* M96: fabric bandwidth estimate section */
        if ((opts != NULL) &&
            ((opts->target_tps != 0u) || (opts->fabric_gib_sec > 0.0))) {
            const uint64_t tiles_eff   = (s->n_tiles > 0u) ? s->n_tiles : 1u;
            const uint64_t act_b       = s->d_model * 4u;
            const uint64_t logits_b    = s->vocab_size * 4u;
            const uint64_t fabric_b    = (tiles_eff > 1u)
                                         ? (tiles_eff - 1u) * act_b + logits_b
                                         : 0u;
            const double   req_gib_sec = (opts->target_tps != 0u)
                ? (double)fabric_b * (double)opts->target_tps / (1024.0 * 1024.0 * 1024.0)
                : 0.0;
            printf("\n[fabric_bandwidth_estimate] (estimate only)\n");
            printf("  fabric_bytes_per_token     = %llu bytes\n",
                   (unsigned long long)fabric_b);
            if (opts->target_tps != 0u) {
                printf("  target_tokens_per_sec      = %llu\n",
                       (unsigned long long)opts->target_tps);
                printf("  required_gib_sec           = %.4f\n", req_gib_sec);
            }
            if (opts->fabric_gib_sec > 0.0) {
                printf("  fabric_available_gib_sec   = %.4f\n", opts->fabric_gib_sec);
            }
            printf("  fabric_bandwidth_status    = %s\n",
                   fabric_bandwidth_status(req_gib_sec, opts->fabric_gib_sec));
        }

    } else {
        /* JSON output */
        printf("{\n");
        printf("  \"name\": \"%s\",\n", s->name);
        printf("  \"source\": \"%s\",\n", s->is_preset ? "preset" : "config");
        printf("  \"note\": \"%s\",\n", s->note);
        printf("  \"synthetic\": %s,\n", s->is_synthetic ? "true" : "false");
        printf("  \"shape\": {\n");
        printf("    \"vocab_size\": %llu,\n", (unsigned long long)s->vocab_size);
        printf("    \"n_layers\": %llu,\n",   (unsigned long long)s->n_layers);
        printf("    \"d_model\": %llu,\n",    (unsigned long long)s->d_model);
        printf("    \"n_heads\": %llu,\n",    (unsigned long long)s->n_heads);
        printf("    \"head_dim\": %llu,\n",   (unsigned long long)s->head_dim);
        printf("    \"d_ff\": %llu,\n",       (unsigned long long)s->d_ff);
        printf("    \"max_seq_len\": %llu,\n",(unsigned long long)s->max_seq_len);
        printf("    \"n_tiles\": %llu,\n",    (unsigned long long)s->n_tiles);
        printf("    \"total_params\": %llu\n",(unsigned long long)s->total_params);
        printf("  },\n");
        printf("  \"storage\": {\n");
        printf("    \"f32_bytes\": %llu,\n",    (unsigned long long)f32_bytes);
        printf("    \"f16_bytes\": %llu,\n",    (unsigned long long)f16_bytes);
        printf("    \"q8_bytes\": %llu,\n",     (unsigned long long)q8_bytes);
        printf("    \"q4_est_bytes\": %llu,\n", (unsigned long long)q4_est);
        printf("    \"by_category_f32\": {\n");
        printf("      \"embeddings\": %llu,\n",     (unsigned long long)(cat.embeddings * 4u));
        printf("      \"attn_proj\": %llu,\n",      (unsigned long long)(cat.attn_projections * 4u));
        printf("      \"ffn_proj\": %llu,\n",       (unsigned long long)(cat.ffn_projections * 4u));
        printf("      \"norms\": %llu,\n",          (unsigned long long)(cat.norms * 4u));
        printf("      \"lm_head\": %llu\n",         (unsigned long long)(cat.lm_head * 4u));
        printf("    }\n");
        printf("  },\n");
        printf("  \"kv_cache_f32\": {\n");
        for (ci = 0; ci < n_ctx; ci++) {
            uint64_t ctx = ctx_list[ci];
            uint64_t kb  = kv_bytes_f32(s, ctx);
            printf("    \"ctx_%llu\": %llu,\n",
                   (unsigned long long)ctx, (unsigned long long)kb);
        }
        printf("    \"ctx_%llu_model_max\": %llu\n",
               (unsigned long long)s->max_seq_len,
               (unsigned long long)kv_f32_default);
        printf("  },\n");
        printf("  \"cluster\": {\n");
        tile = (s->n_tiles > 0u) ? s->n_tiles : 1u;
        printf("    \"layers_per_tile\": %llu,\n",
               (unsigned long long)div_round_up(s->n_layers, tile));
        printf("    \"model_bytes_per_tile_f32\": %llu,\n",
               (unsigned long long)div_round_up(f32_bytes, tile));
        printf("    \"model_bytes_per_tile_q8\": %llu,\n",
               (unsigned long long)div_round_up(q8_bytes, tile));
        printf("    \"kv_bytes_per_tile_f32\": %llu,\n",
               (unsigned long long)div_round_up(kv_f32_default, tile));
        printf("    \"act_bytes_per_token_f32\": %llu,\n",
               (unsigned long long)act_bytes);
        printf("    \"logits_bytes_per_token_f32\": %llu,\n",
               (unsigned long long)logits_bytes);
        printf("    \"est_fabric_bytes_per_token\": %llu\n",
               (unsigned long long)fabric_total);
        printf("  },\n");
        printf("  \"backend_feasibility\": {\n");
        printf("    \"cpu_f32\": \"%s\",\n",  feasible(f32_bytes, CPU_RAM_THRESHOLD_BYTES));
        printf("    \"cpu_q8\": \"%s\",\n",   feasible(q8_bytes,  CPU_RAM_THRESHOLD_BYTES));
        printf("    \"cuda_f32\": \"%s\",\n", feasible(f32_bytes, CUDA_VRAM_THRESHOLD_BYTES));
        printf("    \"cuda_q8\": \"%s\"\n",   feasible(q8_bytes,  CUDA_VRAM_THRESHOLD_BYTES));
        printf("  }");
        if (dtype != NULL) {
            printf(",\n  \"dtype_selected\": \"%s\"", dtype);
        }
        /* M96: capacity/bandwidth JSON fields */
        if ((opts != NULL) &&
            ((opts->tile_memory_mib != 0u) || (opts->target_tps != 0u) ||
             (opts->fabric_gib_sec > 0.0))) {
            const uint64_t tiles_eff    = (s->n_tiles > 0u) ? s->n_tiles : 1u;
            const uint64_t per_tile_f32 = div_round_up(f32_bytes,      tiles_eff);
            const uint64_t per_tile_q8  = div_round_up(q8_bytes,       tiles_eff);
            const uint64_t kv_per_tile  = div_round_up(kv_f32_default, tiles_eff);
            const uint64_t sessions     = (opts->sessions > 0u) ? opts->sessions : 1u;
            const uint64_t combined     = per_tile_f32 + kv_per_tile * sessions;
            const uint64_t tile_bytes   = opts->tile_memory_mib * 1024u * 1024u;
            const double   util_pct     = (tile_bytes > 0u)
                                          ? 100.0 * (double)combined / (double)tile_bytes
                                          : 0.0;
            const uint64_t act_b        = s->d_model * 4u;
            const uint64_t logits_b     = s->vocab_size * 4u;
            const uint64_t fabric_b     = (tiles_eff > 1u)
                                          ? (tiles_eff - 1u) * act_b + logits_b
                                          : 0u;
            const double   req_gib_sec  = (opts->target_tps != 0u)
                ? (double)fabric_b * (double)opts->target_tps / (1024.0 * 1024.0 * 1024.0)
                : 0.0;
            printf(",\n  \"tile_capacity_estimate\": {\n");
            printf("    \"tile_memory_mib\": %llu,\n",
                   (unsigned long long)opts->tile_memory_mib);
            printf("    \"sessions\": %llu,\n", (unsigned long long)sessions);
            printf("    \"model_bytes_per_tile_f32\": %llu,\n",
                   (unsigned long long)per_tile_f32);
            printf("    \"model_bytes_per_tile_q8\": %llu,\n",
                   (unsigned long long)per_tile_q8);
            printf("    \"kv_bytes_per_tile_f32\": %llu,\n",
                   (unsigned long long)kv_per_tile);
            printf("    \"combined_bytes_per_tile_f32\": %llu,\n",
                   (unsigned long long)combined);
            printf("    \"combined_utilization_pct\": %.1f,\n", util_pct);
            printf("    \"tile_capacity_status\": \"%s\"\n",
                   tile_capacity_status(combined, opts->tile_memory_mib));
            printf("  }");
            printf(",\n  \"fabric_bandwidth_estimate\": {\n");
            printf("    \"fabric_bytes_per_token\": %llu,\n",
                   (unsigned long long)fabric_b);
            printf("    \"target_tokens_per_sec\": %llu,\n",
                   (unsigned long long)opts->target_tps);
            printf("    \"required_gib_sec\": %.4f,\n", req_gib_sec);
            printf("    \"fabric_available_gib_sec\": %.4f,\n", opts->fabric_gib_sec);
            printf("    \"fabric_bandwidth_status\": \"%s\"\n",
                   fabric_bandwidth_status(req_gib_sec, opts->fabric_gib_sec));
            printf("  }");
        }
        printf("\n}\n");
    }
}

/* =========================================================================
 * M100: Tensor-level placement report JSON emission
 * ===================================================================== */

/* Q4 default group size for placement report alignment checks. */
#define M100_Q4_GS  32u

/* Max per-tile issues tracked for warnings/failures arrays. */
#define M100_MAX_ISSUES 512u

/*
 * Map user-supplied dtype to a validator-accepted dtype string.
 * f16 is mapped to f32 because the M99 validator does not accept f16.
 */
static const char *m100_dtype(const char *dtype)
{
    if ((dtype == NULL) ||
        (strcmp(dtype, "f32") == 0) ||
        (strcmp(dtype, "f16") == 0)) {
        return "f32";
    }
    if (strcmp(dtype, "q8")   == 0) { return "q8";   }
    if (strcmp(dtype, "q4")   == 0) { return "q4";   }
    if (strcmp(dtype, "bf16") == 0) { return "bf16"; }
    return "f32";
}

/* Quantization family string for the given report dtype. */
static const char *m100_qfamily(const char *dr)
{
    if (strcmp(dr, "q8") == 0) { return "per_row_q8";   }
    if (strcmp(dr, "q4") == 0) { return "per_group_q4"; }
    return "none";
}

/* Packed bytes for a matrix (rows * cols) under dr. */
static uint64_t m100_packed(uint64_t rows, uint64_t cols, const char *dr)
{
    if (strcmp(dr, "q8") == 0) { return rows * cols; }
    if (strcmp(dr, "q4") == 0) { return (rows * cols + 1u) / 2u; }
    return rows * cols * 4u;
}

/* Q4 scale bytes: rows * ceil(cols / group_size) * sizeof(f16). */
static uint64_t m100_scale(uint64_t rows, uint64_t cols)
{
    return rows * ((cols + M100_Q4_GS - 1u) / M100_Q4_GS) * 2u;
}

/* Packed bytes for a 1D vector (always treated as f32). */
static uint64_t m100_vec_bytes(uint64_t dim)
{
    return dim * 4u;
}

/*
 * Compute model storage bytes for tile_model_params under dr.
 * Norms are always f32 (norm_bytes passed separately).
 */
static uint64_t m100_tile_model_bytes(uint64_t mat_params, uint64_t norm_bytes,
                                      const char *dr)
{
    uint64_t mat_bytes;
    if (strcmp(dr, "q8") == 0) { mat_bytes = mat_params; }
    else if (strcmp(dr, "q4") == 0) { mat_bytes = (mat_params + 1u) / 2u; }
    else { mat_bytes = mat_params * 4u; }
    return mat_bytes + norm_bytes;
}

/*
 * Emit one tensor placement record to fp.
 *
 * sep   — pointer to separator flag; set to 1 after first record.
 * dim1  — 0 for 1D tensors (norms); > 0 for 2D matrices.
 * 1D tensors are always emitted as dtype=f32 / quantization=none.
 * All tensors use slice_axis=0, slice_start=0, slice_end=dim0 (no
 * cross-tile splitting in layer-wise placement).
 */
static void m100_emit_tensor(
        FILE *fp, int *sep,
        uint64_t tid, const char *name, const char *cat, uint64_t layer,
        uint64_t dim0, uint64_t dim1,   /* dim1=0 → 1D */
        const char *dr, uint64_t owner_tile)
{
    const int   is_1d    = (dim1 == 0u);
    const char *eff_dr   = is_1d ? "f32" : dr;
    uint64_t    packed   = 0u;
    uint64_t    scale    = 0u;

    if (*sep) { fputs(",\n", fp); }
    *sep = 1;

    if (is_1d) {
        packed = m100_vec_bytes(dim0);
    } else {
        packed = m100_packed(dim0, dim1, dr);
        if (strcmp(dr, "q4") == 0) {
            scale = m100_scale(dim0, dim1);
        }
    }

    fprintf(fp,
            "    {\n"
            "      \"tensor_id\": %llu,\n"
            "      \"tensor_name\": \"%s\",\n"
            "      \"tensor_category\": \"%s\",\n"
            "      \"layer\": %llu,\n",
            (unsigned long long)tid, name, cat,
            (unsigned long long)layer);

    if (is_1d) {
        fprintf(fp,
                "      \"source_shape\": [%llu],\n"
                "      \"placed_shape\": [%llu],\n",
                (unsigned long long)dim0, (unsigned long long)dim0);
    } else {
        fprintf(fp,
                "      \"source_shape\": [%llu, %llu],\n"
                "      \"placed_shape\": [%llu, %llu],\n",
                (unsigned long long)dim0, (unsigned long long)dim1,
                (unsigned long long)dim0, (unsigned long long)dim1);
    }

    fprintf(fp, "      \"dtype\": \"%s\",\n", eff_dr);

    if (strcmp(eff_dr, "q4") == 0) {
        fprintf(fp,
                "      \"quantization\": \"per_group_q4\",\n"
                "      \"quantization_group_size\": %u,\n"
                "      \"scale_bytes\": %llu,\n"
                "      \"packed_bytes\": %llu,\n",
                (unsigned)M100_Q4_GS,
                (unsigned long long)scale,
                (unsigned long long)packed);
    } else if (strcmp(eff_dr, "q8") == 0) {
        fprintf(fp,
                "      \"quantization\": \"per_row_q8\",\n"
                "      \"quantization_group_size\": 0,\n"
                "      \"scale_bytes\": 0,\n"
                "      \"packed_bytes\": %llu,\n",
                (unsigned long long)packed);
    } else {
        fprintf(fp,
                "      \"quantization\": \"none\",\n"
                "      \"quantization_group_size\": 0,\n"
                "      \"scale_bytes\": 0,\n"
                "      \"packed_bytes\": %llu,\n",
                (unsigned long long)packed);
    }

    fprintf(fp,
            "      \"owner_tile\": %llu,\n"
            "      \"owner_aimu\": %llu,\n"
            "      \"slice_axis\": 0,\n"
            "      \"slice_start\": 0,\n"
            "      \"slice_end\": %llu,\n"
            "      \"replication_policy\": \"unique\",\n"
            "      \"routing_requirements\": \"none\",\n"
            "      \"reduction_behavior\": \"none\",\n"
            "      \"checksum\": \"0x0000000000000000\",\n"
            "      \"placement_status\": \"placed\"\n"
            "    }",
            (unsigned long long)owner_tile,
            (unsigned long long)owner_tile,
            (unsigned long long)dim0);
}

/*
 * Issue record used to accumulate per-tile capacity/bandwidth issues
 * while writing the tiles section so they can be re-emitted in the
 * warnings/failures arrays.
 */
typedef struct m100_issue {
    int    is_error;      /* 1 = failures[], 0 = warnings[] */
    char   msg[320];
} m100_issue;

/*
 * write_placement_report_json() — M100
 *
 * Emit a tensor-level placement report JSON file conforming to the M98
 * schema.  The report is generated from the estimated model shape and
 * capacity options; it does not execute any inference.
 *
 * Returns 0 on success, -1 on error (message already printed to stderr).
 */
static int write_placement_report_json(
        const char        *path,
        const model_shape *s,
        const char        *dtype_arg,
        const capacity_opts *opts)
{
    FILE       *fp         = NULL;
    const char *dr         = m100_dtype(dtype_arg);
    const char *qf         = m100_qfamily(dr);
    uint64_t    tile_count = (s->n_tiles > 0u) ? s->n_tiles : 1u;
    uint64_t    ctx        = s->max_seq_len;
    uint64_t    sessions   = ((opts != NULL) && (opts->sessions > 0u))
                             ? opts->sessions : 1u;
    uint64_t    lpt;    /* layers per tile (ceiling) */
    uint64_t    t;
    uint64_t    l;
    uint64_t    tid;    /* running tensor_id counter */
    int         tsep;   /* tensor-array separator flag */

    /* Issue accumulator (capacity/bandwidth problems found during tile pass) */
    m100_issue  issues[M100_MAX_ISSUES];
    uint64_t    n_issues  = 0u;
    int         n_warn    = 0;
    int         n_fail    = 0;
    int         first_sym = 1; /* synthetic-warning emitted flag */

    (void)first_sym;

    fp = fopen(path, "w");
    if (fp == NULL) {
        fprintf(stderr, "att1-size: cannot open --placement-report-json: %s\n",
                path);
        return -1;
    }

    lpt = div_round_up(s->n_layers, tile_count);

    /* ================================================================
     * Report header
     * ============================================================= */
    fprintf(fp, "{\n");
    fprintf(fp, "  \"report_version\": 1,\n");
    fprintf(fp, "  \"header\": {\n");
    fprintf(fp, "    \"model_name\": \"%s\",\n",    s->name);
    fprintf(fp, "    \"artifact_path\": \"\",\n");
    fprintf(fp, "    \"config_source\": \"%s\",\n",
            s->is_preset ? "preset" : "config");
    fprintf(fp, "    \"dtype\": \"%s\",\n",              dr);
    fprintf(fp, "    \"quantization_family\": \"%s\",\n", qf);
    fprintf(fp, "    \"n_layers\": %llu,\n",   (unsigned long long)s->n_layers);
    fprintf(fp, "    \"d_model\": %llu,\n",    (unsigned long long)s->d_model);
    fprintf(fp, "    \"n_heads\": %llu,\n",    (unsigned long long)s->n_heads);
    fprintf(fp, "    \"n_kv_heads\": %llu,\n", (unsigned long long)s->n_heads);
    fprintf(fp, "    \"ffn_hidden\": %llu,\n", (unsigned long long)s->d_ff);
    fprintf(fp, "    \"vocab_size\": %llu,\n", (unsigned long long)s->vocab_size);
    fprintf(fp, "    \"tile_count\": %llu,\n", (unsigned long long)tile_count);

    if ((opts != NULL) && (opts->tile_memory_mib > 0u)) {
        fprintf(fp, "    \"tile_memory_mib\": %llu,\n",
                (unsigned long long)opts->tile_memory_mib);
    } else {
        fprintf(fp, "    \"tile_memory_mib\": null,\n");
    }
    fprintf(fp, "    \"target_context_length\": %llu,\n", (unsigned long long)ctx);
    fprintf(fp, "    \"target_sessions\": %llu,\n",       (unsigned long long)sessions);

    if ((opts != NULL) && (opts->target_tps > 0u)) {
        fprintf(fp, "    \"target_tokens_per_sec\": %llu,\n",
                (unsigned long long)opts->target_tps);
    } else {
        fprintf(fp, "    \"target_tokens_per_sec\": null,\n");
    }
    if ((opts != NULL) && (opts->fabric_gib_sec > 0.0)) {
        fprintf(fp, "    \"fabric_gib_sec\": %.4f,\n", opts->fabric_gib_sec);
    } else {
        fprintf(fp, "    \"fabric_gib_sec\": null,\n");
    }
    fprintf(fp, "    \"placement_policy\": \"layer_wise\",\n");
    fprintf(fp, "    \"report_timestamp\": null\n");
    fprintf(fp, "  },\n");

    /* ================================================================
     * Tile summary records
     * ============================================================= */
    fprintf(fp, "  \"tiles\": [\n");

    for (t = 0u; t < tile_count; t++) {
        const uint64_t first_layer = t * lpt;
        uint64_t       last_layer;
        uint64_t       tile_layers;
        uint64_t       mat_params;
        uint64_t       norm_bytes_tile;
        uint64_t       tile_model_bytes;
        uint64_t       kv_bytes_per_sess;
        uint64_t       act_bytes;
        uint64_t       logits_bytes;
        uint64_t       fabric_bytes;
        uint64_t       fabric_pkts;
        uint64_t       combined;
        uint64_t       tile_mem_bytes;
        double         util_pct;
        const char    *cap_stat;
        double         req_gib;
        const char    *bw_stat;
        uint64_t       tcount;

        if (first_layer >= s->n_layers) {
            /* Empty tile (more tiles than layers). */
            last_layer  = (s->n_layers > 0u) ? (s->n_layers - 1u) : 0u;
            tile_layers = 0u;
        } else {
            const uint64_t end0 = first_layer + lpt - 1u;
            last_layer  = (end0 < s->n_layers) ? end0 : (s->n_layers - 1u);
            tile_layers = last_layer - first_layer + 1u;
        }

        /* Weight matrix parameter count for this tile. */
        {
            const uint64_t attn_per  = 4u * s->d_model * s->d_model;
            const uint64_t ffn_per   = 2u * s->d_model * s->d_ff
                                       + s->d_ff * s->d_model;
            mat_params = tile_layers * (attn_per + ffn_per);
            if (t == 0u) {
                mat_params += s->vocab_size * s->d_model; /* tok_embeddings */
            }
            if (t == (tile_count - 1u)) {
                mat_params += s->d_model * s->vocab_size; /* lm_head */
            }
        }
        /* Norm vectors are always f32 regardless of model dtype. */
        norm_bytes_tile = tile_layers * 2u * s->d_model * 4u; /* attn+ffn norm per layer */
        if (t == (tile_count - 1u)) {
            norm_bytes_tile += s->d_model * 4u; /* output_norm */
        }
        tile_model_bytes = m100_tile_model_bytes(mat_params, norm_bytes_tile, dr);

        /* KV cache: f32, K+V, per session. */
        kv_bytes_per_sess = tile_layers * ctx * s->n_heads * s->head_dim * 2u * 4u;

        /* Per-token activation/logit traffic. */
        act_bytes    = s->d_model * 4u;
        logits_bytes = (t == (tile_count - 1u)) ? (s->vocab_size * 4u) : 0u;
        fabric_bytes = (t < (tile_count - 1u))  ? act_bytes             : 0u;
        fabric_pkts  = (t < (tile_count - 1u))  ? 1u                    : 0u;

        /* Capacity. */
        combined      = tile_model_bytes + kv_bytes_per_sess * sessions;
        tile_mem_bytes = ((opts != NULL) && (opts->tile_memory_mib > 0u))
                         ? opts->tile_memory_mib * 1024u * 1024u : 0u;
        util_pct = (tile_mem_bytes > 0u)
                   ? 100.0 * (double)combined / (double)tile_mem_bytes
                   : 0.0;
        cap_stat = tile_capacity_status(combined,
                       (opts != NULL) ? opts->tile_memory_mib : 0u);

        /* Bandwidth. */
        req_gib = ((opts != NULL) && (opts->target_tps > 0u))
                  ? (double)fabric_bytes * (double)opts->target_tps
                    / (1024.0 * 1024.0 * 1024.0)
                  : 0.0;
        bw_stat = fabric_bandwidth_status(
                      req_gib,
                      (opts != NULL) ? opts->fabric_gib_sec : 0.0);

        /* Collect issues. */
        if ((strcmp(cap_stat, "FAIL") == 0) && (n_issues < M100_MAX_ISSUES)) {
            issues[n_issues].is_error = 1;
            snprintf(issues[n_issues].msg, sizeof(issues[n_issues].msg),
                     "tile %llu capacity overflow: combined=%.0f bytes, "
                     "limit=%llu bytes (%.1f%%)",
                     (unsigned long long)t,
                     (double)combined,
                     (unsigned long long)tile_mem_bytes,
                     util_pct);
            n_issues++;
            n_fail++;
        } else if ((strcmp(cap_stat, "WARN") == 0) &&
                   (n_issues < M100_MAX_ISSUES)) {
            issues[n_issues].is_error = 0;
            snprintf(issues[n_issues].msg, sizeof(issues[n_issues].msg),
                     "tile %llu capacity warning: utilization %.1f%%",
                     (unsigned long long)t, util_pct);
            n_issues++;
            n_warn++;
        }
        if ((strcmp(bw_stat, "FAIL") == 0) && (n_issues < M100_MAX_ISSUES)) {
            issues[n_issues].is_error = 1;
            snprintf(issues[n_issues].msg, sizeof(issues[n_issues].msg),
                     "tile %llu bandwidth overflow: required %.4f GiB/s, "
                     "available %.4f GiB/s",
                     (unsigned long long)t,
                     req_gib,
                     (opts != NULL) ? opts->fabric_gib_sec : 0.0);
            n_issues++;
            n_fail++;
        } else if ((strcmp(bw_stat, "WARN") == 0) &&
                   (n_issues < M100_MAX_ISSUES)) {
            issues[n_issues].is_error = 0;
            snprintf(issues[n_issues].msg, sizeof(issues[n_issues].msg),
                     "tile %llu bandwidth warning: required %.4f GiB/s",
                     (unsigned long long)t, req_gib);
            n_issues++;
            n_warn++;
        }

        /* Tensor count estimate for this tile. */
        tcount = tile_layers * 9u; /* 4 attn + 3 ffn + 2 norms per layer */
        if (t == 0u)               { tcount++;  } /* tok_embeddings */
        if (t == (tile_count - 1u)) { tcount += 2u; } /* output_norm + lm_head */

        if (t > 0u) { fputs(",\n", fp); }
        fprintf(fp,
                "    {\n"
                "      \"tile_id\": %llu,\n"
                "      \"aimu_id\": %llu,\n"
                "      \"layer_range_start\": %llu,\n"
                "      \"layer_range_end\": %llu,\n"
                "      \"assigned_tensor_count\": %llu,\n"
                "      \"assigned_tensor_slice_count\": %llu,\n"
                "      \"model_bytes\": %llu,\n"
                "      \"kv_bytes\": %llu,\n"
                "      \"activation_bytes_per_token\": %llu,\n"
                "      \"logits_bytes_per_token\": %llu,\n"
                "      \"fabric_payload_bytes_per_token\": %llu,\n"
                "      \"fabric_packets_per_token\": %llu,\n"
                "      \"memory_utilization_percent\": %.2f,\n"
                "      \"capacity_status\": \"%s\",\n"
                "      \"bandwidth_status\": \"%s\"\n"
                "    }",
                (unsigned long long)t,
                (unsigned long long)t,
                (unsigned long long)(first_layer < s->n_layers ? first_layer : 0u),
                (unsigned long long)last_layer,
                (unsigned long long)tcount,
                (unsigned long long)tcount,
                (unsigned long long)tile_model_bytes,
                (unsigned long long)kv_bytes_per_sess,
                (unsigned long long)act_bytes,
                (unsigned long long)logits_bytes,
                (unsigned long long)fabric_bytes,
                (unsigned long long)fabric_pkts,
                util_pct,
                cap_stat,
                bw_stat);
    }
    fprintf(fp, "\n  ],\n");

    /* ================================================================
     * Tensor placement records
     * ============================================================= */
    fprintf(fp, "  \"tensors\": [\n");
    tsep = 0;
    tid  = 0u;

    for (t = 0u; t < tile_count; t++) {
        const uint64_t first_layer = t * lpt;
        uint64_t       last_layer;
        char           nbuf[128];

        if (first_layer >= s->n_layers) {
            last_layer = (s->n_layers > 0u) ? (s->n_layers - 1u) : 0u;
        } else {
            const uint64_t end0 = first_layer + lpt - 1u;
            last_layer = (end0 < s->n_layers) ? end0 : (s->n_layers - 1u);
        }

        /* tok_embeddings lives on tile 0. */
        if (t == 0u) {
            m100_emit_tensor(fp, &tsep,
                    tid++, "tok_embeddings.weight", "embedding",
                    0u, s->vocab_size, s->d_model, dr, 0u);
        }

        /* Per-layer weights. */
        for (l = first_layer;
             (l <= last_layer) && (l < s->n_layers);
             l++) {
            snprintf(nbuf, sizeof(nbuf),
                     "layers.%llu.attention.wq.weight",
                     (unsigned long long)l);
            m100_emit_tensor(fp, &tsep, tid++, nbuf, "attention_q",
                    l, s->d_model, s->d_model, dr, t);

            snprintf(nbuf, sizeof(nbuf),
                     "layers.%llu.attention.wk.weight",
                     (unsigned long long)l);
            m100_emit_tensor(fp, &tsep, tid++, nbuf, "attention_k",
                    l, s->d_model, s->d_model, dr, t);

            snprintf(nbuf, sizeof(nbuf),
                     "layers.%llu.attention.wv.weight",
                     (unsigned long long)l);
            m100_emit_tensor(fp, &tsep, tid++, nbuf, "attention_v",
                    l, s->d_model, s->d_model, dr, t);

            snprintf(nbuf, sizeof(nbuf),
                     "layers.%llu.attention.wo.weight",
                     (unsigned long long)l);
            m100_emit_tensor(fp, &tsep, tid++, nbuf, "attention_o",
                    l, s->d_model, s->d_model, dr, t);

            snprintf(nbuf, sizeof(nbuf),
                     "layers.%llu.feed_forward.w_gate.weight",
                     (unsigned long long)l);
            m100_emit_tensor(fp, &tsep, tid++, nbuf, "ffn_gate",
                    l, s->d_ff, s->d_model, dr, t);

            snprintf(nbuf, sizeof(nbuf),
                     "layers.%llu.feed_forward.w_up.weight",
                     (unsigned long long)l);
            m100_emit_tensor(fp, &tsep, tid++, nbuf, "ffn_up",
                    l, s->d_ff, s->d_model, dr, t);

            snprintf(nbuf, sizeof(nbuf),
                     "layers.%llu.feed_forward.w_down.weight",
                     (unsigned long long)l);
            m100_emit_tensor(fp, &tsep, tid++, nbuf, "ffn_down",
                    l, s->d_model, s->d_ff, dr, t);

            snprintf(nbuf, sizeof(nbuf),
                     "layers.%llu.attention_norm.weight",
                     (unsigned long long)l);
            m100_emit_tensor(fp, &tsep, tid++, nbuf, "norm",
                    l, s->d_model, 0u, dr, t); /* 1D */

            snprintf(nbuf, sizeof(nbuf),
                     "layers.%llu.ffn_norm.weight",
                     (unsigned long long)l);
            m100_emit_tensor(fp, &tsep, tid++, nbuf, "norm",
                    l, s->d_model, 0u, dr, t); /* 1D */
        }

        /* output_norm and lm_head live on the last tile. */
        if (t == (tile_count - 1u)) {
            const uint64_t final_l = (s->n_layers > 0u)
                                     ? (s->n_layers - 1u) : 0u;
            m100_emit_tensor(fp, &tsep,
                    tid++, "output.norm", "norm",
                    final_l, s->d_model, 0u, dr, t); /* 1D */
            m100_emit_tensor(fp, &tsep,
                    tid++, "output.weight", "lm_head",
                    final_l, s->vocab_size, s->d_model, dr, t);
        }
    }
    fprintf(fp, "\n  ],\n");

    /* ================================================================
     * Validation metadata (informational; filled by M99 validator)
     * ============================================================= */
    fprintf(fp, "  \"validation\": {},\n");

    /* ================================================================
     * Warnings array
     * ============================================================= */
    fprintf(fp, "  \"warnings\": [");
    {
        int    first = 1;
        uint64_t wi;

        /* Synthetic / non-executable model warning. */
        if (s->is_synthetic) {
            fprintf(fp,
                    "\n    {\"rule\": 0, \"severity\": \"warning\", "
                    "\"message\": "
                    "\"synthetic/non-executable preset '%s': "
                    "estimates are architectural projections only\"}",
                    s->name);
            first = 0;
            n_warn++;
        }
        for (wi = 0u; wi < n_issues; wi++) {
            if (!issues[wi].is_error) {
                if (!first) { fputs(",", fp); }
                first = 0;
                /* Escape any double-quotes in the message (unlikely). */
                fprintf(fp,
                        "\n    {\"rule\": 0, \"severity\": \"warning\", "
                        "\"message\": \"%s\"}",
                        issues[wi].msg);
            }
        }
        (void)first;
    }
    fprintf(fp, "\n  ],\n");

    /* ================================================================
     * Failures array
     * ============================================================= */
    fprintf(fp, "  \"failures\": [");
    {
        int    first = 1;
        uint64_t wi;
        for (wi = 0u; wi < n_issues; wi++) {
            if (issues[wi].is_error) {
                if (!first) { fputs(",", fp); }
                first = 0;
                fprintf(fp,
                        "\n    {\"rule\": 0, \"severity\": \"error\", "
                        "\"message\": \"%s\"}",
                        issues[wi].msg);
            }
        }
        (void)first;
    }
    fprintf(fp, "\n  ],\n");

    /* ================================================================
     * Remediation suggestions
     * ============================================================= */
    fprintf(fp, "  \"remediation\": [");
    if (n_fail > 0) {
        fprintf(fp,
                "\n    {\"message\": "
                "\"reduce --tile-memory-mib, add more --tiles, or use a "
                "lower-bandwidth dtype such as q4\"}");
    }
    fprintf(fp, "\n  ]\n");
    fprintf(fp, "}\n");

    fclose(fp);
    return 0;
}

/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */

int main(int argc, char **argv)
{
    model_shape   shape;
    capacity_opts cap              = {0u, 0u, 0u, 0.0};
    const char   *preset           = NULL;
    const char   *config           = NULL;
    const char   *dtype            = NULL;
    const char   *placement_report = NULL;  /* --placement-report-json PATH */
    uint64_t      tiles            = 0u;
    uint64_t      context          = 0u;
    int           json_out         = 0;
    int           use_manual       = 0;
    uint64_t      m_layers         = 0u;
    uint64_t      m_dmodel         = 0u;
    uint64_t      m_heads          = 0u;
    uint64_t      m_dff            = 0u;
    uint64_t      m_vocab          = 0u;
    int           i                = 0;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else if ((strcmp(argv[i], "--preset") == 0) && ((i + 1) < argc)) {
            preset = argv[++i];
        } else if ((strcmp(argv[i], "--config") == 0) && ((i + 1) < argc)) {
            config = argv[++i];
        } else if ((strcmp(argv[i], "--json") == 0)) {
            json_out = 1;
        } else if ((strcmp(argv[i], "--tiles") == 0) && ((i + 1) < argc)) {
            if ((parse_u64(argv[++i], &tiles) != 0) || (tiles == 0u)) {
                usage(argv[0]);
                return 1;
            }
        } else if ((strcmp(argv[i], "--context") == 0) && ((i + 1) < argc)) {
            if ((parse_u64(argv[++i], &context) != 0) || (context == 0u)) {
                usage(argv[0]);
                return 1;
            }
        } else if ((strcmp(argv[i], "--dtype") == 0) && ((i + 1) < argc)) {
            dtype = argv[++i];
            if (!dtype_supported(dtype)) {
                usage(argv[0]);
                return 1;
            }
        } else if ((strcmp(argv[i], "--layers") == 0) && ((i + 1) < argc)) {
            if (parse_u64(argv[++i], &m_layers) != 0) { usage(argv[0]); return 1; }
            use_manual = 1;
        } else if ((strcmp(argv[i], "--d-model") == 0) && ((i + 1) < argc)) {
            if (parse_u64(argv[++i], &m_dmodel) != 0) { usage(argv[0]); return 1; }
            use_manual = 1;
        } else if ((strcmp(argv[i], "--heads") == 0) && ((i + 1) < argc)) {
            if (parse_u64(argv[++i], &m_heads) != 0) { usage(argv[0]); return 1; }
            use_manual = 1;
        } else if ((strcmp(argv[i], "--d-ff") == 0) && ((i + 1) < argc)) {
            if (parse_u64(argv[++i], &m_dff) != 0) { usage(argv[0]); return 1; }
            use_manual = 1;
        } else if ((strcmp(argv[i], "--vocab-size") == 0) && ((i + 1) < argc)) {
            if (parse_u64(argv[++i], &m_vocab) != 0) { usage(argv[0]); return 1; }
            use_manual = 1;
        } else if ((strcmp(argv[i], "--tile-memory-mib") == 0) && ((i + 1) < argc)) {
            if ((parse_u64(argv[++i], &cap.tile_memory_mib) != 0) ||
                (cap.tile_memory_mib == 0u)) {
                usage(argv[0]);
                return 1;
            }
        } else if ((strcmp(argv[i], "--tile-memory-gib") == 0) && ((i + 1) < argc)) {
            uint64_t gib = 0u;
            if ((parse_u64(argv[++i], &gib) != 0) || (gib == 0u)) {
                usage(argv[0]);
                return 1;
            }
            cap.tile_memory_mib = gib * 1024u;
        } else if ((strcmp(argv[i], "--sessions") == 0) && ((i + 1) < argc)) {
            if ((parse_u64(argv[++i], &cap.sessions) != 0) || (cap.sessions == 0u)) {
                usage(argv[0]);
                return 1;
            }
        } else if ((strcmp(argv[i], "--target-tokens-per-sec") == 0) && ((i + 1) < argc)) {
            if ((parse_u64(argv[++i], &cap.target_tps) != 0) || (cap.target_tps == 0u)) {
                usage(argv[0]);
                return 1;
            }
        } else if ((strcmp(argv[i], "--fabric-gib-sec") == 0) && ((i + 1) < argc)) {
            if ((parse_double(argv[++i], &cap.fabric_gib_sec) != 0) ||
                (cap.fabric_gib_sec <= 0.0)) {
                usage(argv[0]);
                return 1;
            }
        } else if ((strcmp(argv[i], "--placement-report-json") == 0) && ((i + 1) < argc)) {
            placement_report = argv[++i];
        } else {
            fprintf(stderr, "att1-size: unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    /* Validate: exactly one input mode */
    {
        int modes = (preset != NULL ? 1 : 0) + (config != NULL ? 1 : 0) + (use_manual ? 1 : 0);
        if (modes != 1) {
            if (modes == 0) {
                fprintf(stderr, "att1-size: specify --preset, --config, or manual shape arguments\n");
            } else {
                fprintf(stderr, "att1-size: --preset, --config, and manual shape args are mutually exclusive\n");
            }
            usage(argv[0]);
            return 1;
        }
    }

    memset(&shape, 0, sizeof(shape));

    if (preset != NULL) {
        if (load_preset(preset, &shape) != 0) {
            fprintf(stderr, "att1-size: unknown preset: %s\n", preset);
            return 1;
        }
        if (tiles != 0u)   { shape.n_tiles    = tiles; }
        if (context != 0u) { shape.max_seq_len = context; }
        print_report_preset(&shape, dtype);

        /* M96: capacity/bandwidth summary lines for preset mode */
        if (cap.tile_memory_mib != 0u) {
            const uint64_t tiles_eff = (shape.n_tiles > 0u) ? shape.n_tiles : 1u;
            const uint64_t per_tile  = div_round_up(shape.total_params * 4u, tiles_eff);
            const uint64_t kv_def    = kv_bytes_f32(&shape, shape.max_seq_len);
            const uint64_t kv_tile   = div_round_up(kv_def, tiles_eff);
            const uint64_t sessions  = (cap.sessions > 0u) ? cap.sessions : 1u;
            const uint64_t combined  = per_tile + kv_tile * sessions;
            printf("tile_memory_mib=%llu\n", (unsigned long long)cap.tile_memory_mib);
            print_bytes_line("model_bytes_per_tile", per_tile);
            printf("tile_capacity_status=%s\n",
                   tile_capacity_status(combined, cap.tile_memory_mib));
        }
        if ((cap.target_tps != 0u) || (cap.fabric_gib_sec > 0.0)) {
            const uint64_t tiles_eff = (shape.n_tiles > 0u) ? shape.n_tiles : 1u;
            const uint64_t act_b     = shape.d_model * 4u;
            const uint64_t logits_b  = shape.vocab_size * 4u;
            const uint64_t fabric_b  = (tiles_eff > 1u)
                                       ? (tiles_eff - 1u) * act_b + logits_b
                                       : 0u;
            const double   req_gib   = (cap.target_tps != 0u)
                ? (double)fabric_b * (double)cap.target_tps / (1024.0 * 1024.0 * 1024.0)
                : 0.0;
            printf("fabric_bandwidth_status=%s\n",
                   fabric_bandwidth_status(req_gib, cap.fabric_gib_sec));
        }
        if ((placement_report != NULL) &&
            (write_placement_report_json(placement_report, &shape, dtype, &cap) != 0)) {
            return 1;
        }
        return 0;
    }

    if (config != NULL) {
        if (load_config(config, &shape) != 0) {
            return 1;
        }
    } else {
        /* Manual shape */
        if ((m_layers == 0u) || (m_dmodel == 0u) || (m_heads == 0u) ||
            (m_dff == 0u) || (m_vocab == 0u)) {
            fprintf(stderr, "att1-size: manual mode requires --layers, --d-model, --heads, --d-ff, --vocab-size\n");
            return 1;
        }
        if (m_dmodel % m_heads != 0u) {
            fprintf(stderr, "att1-size: d_model (%llu) must be divisible by heads (%llu)\n",
                    (unsigned long long)m_dmodel, (unsigned long long)m_heads);
            return 1;
        }
        memset(&shape, 0, sizeof(shape));
        strncpy(shape.name, "manual", sizeof(shape.name) - 1u);
        strncpy(shape.note, "manual shape estimate", sizeof(shape.note) - 1u);
        shape.is_preset    = 0;
        shape.is_synthetic = 1;
        shape.vocab_size   = m_vocab;
        shape.n_layers     = m_layers;
        shape.n_heads      = m_heads;
        shape.d_model      = m_dmodel;
        shape.d_ff         = m_dff;
        shape.head_dim     = m_dmodel / m_heads;
        shape.max_seq_len  = (context != 0u) ? context : 4096u;
        shape.n_tiles      = 1u;
        shape.total_params = llama_params(m_vocab, m_layers, m_dmodel, m_dff);
        shape.active_params_per_token = shape.total_params;
    }

    if (tiles != 0u)   { shape.n_tiles    = tiles; }
    if (context != 0u) { shape.max_seq_len = context; }

    print_report_full(&shape, dtype, json_out, &cap);
    if ((placement_report != NULL) &&
        (write_placement_report_json(placement_report, &shape, dtype, &cap) != 0)) {
        return 1;
    }
    return 0;
}
