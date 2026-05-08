/*
 * test_converter_validation.c
 *
 * M44: Validation of converter-generated .att1 artifacts with shard metadata.
 * M48: Validation of checked-in tiny f32 safetensors conversion artifact.
 * M49: Validation of checked-in tiny q8 safetensors conversion artifact.
 * M60: Validation of real tiny f32/q8 artifacts with pretokenized external input.
 * M81: Validation of real_tiny_q4 artifact (from m63 fixture, cpu-q4 paths).
 *
 * Uses checked-in .att1 fixtures only; no Python is required at test time.
 *
 * Checks:
 *   1. att1-inspect produces expected tensor/shard_meta fields.
 *   2. att1-bench with --shard-plan runtime produces correct output.
 *   3. att1-bench with --shard-plan metadata produces identical last_token,
 *      logits_bytes_produced, and fabric_packets_sent.
 *   4. (M60) real_tiny_f32 / real_tiny_q8 accept --tokenizer external with
 *      fixed pretokenized IDs; all CPU and CUDA paths exercised.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MODEL_PATH "models/converted_stub_meta/model.att1"
#define REAL_TINY_MODEL_PATH "models/real_tiny_f32/model.att1"
#define REAL_TINY_Q8_MODEL_PATH "models/real_tiny_q8/model.att1"
#define REAL_TINY_Q4_MODEL_PATH "models/real_tiny_q4/model.att1"
#define REAL_TINY_PROMPT "\001"
#define TOKENS     "8"
#define TILES      "2"

/* ------------------------------------------------------------------ helpers */

static int run_command(const char *cmd)
{
    return system(cmd) == 0 ? 0 : -1;
}

static int read_file(const char *path, char *buf, size_t cap)
{
    FILE  *fp = NULL;
    size_t n  = 0u;

    if ((path == NULL) || (buf == NULL) || (cap == 0u)) {
        return -1;
    }

    fp = fopen(path, "rb");
    if (fp == NULL) {
        return -1;
    }

    n = fread(buf, 1u, cap - 1u, fp);
    if (ferror(fp) != 0) {
        fclose(fp);
        return -1;
    }

    buf[n] = '\0';
    fclose(fp);
    return 0;
}

static int parse_u64_line(const char *text, const char *key, uint64_t *out)
{
    const char        *p   = NULL;
    char              *end = NULL;
    unsigned long long v   = 0u;

    if ((text == NULL) || (key == NULL) || (out == NULL)) {
        return -1;
    }

    p = strstr(text, key);
    if (p == NULL) {
        return -1;
    }

    p += strlen(key);
    if (*p != '=') {
        return -1;
    }
    p++;

    v = strtoull(p, &end, 10);
    if (end == p) {
        return -1;
    }

    *out = (uint64_t)v;
    return 0;
}

static int output_is_cuda_unavailable(const char *out)
{
    return (out != NULL) &&
           ((strstr(out, "unsupported or unavailable") != NULL) ||
            (strstr(out, "CUDA unavailable") != NULL) ||
            (strstr(out, "cuda backend unsupported") != NULL));
}

/* ----------------------------------------------------------------- inspect  */

static int check_inspect(void)
{
    char     out[8192];
    uint64_t v = 0u;

    if (run_command("./build/att1-inspect " MODEL_PATH
                    " > build/m44_inspect.txt 2>&1") != 0) {
        return -1;
    }

    if (read_file("build/m44_inspect.txt", out, sizeof(out)) != 0) {
        return -1;
    }

    /* config fields */
    if ((strstr(out, "n_tiles=2")          == NULL) ||
        (strstr(out, "tensor_count=21")    == NULL)) {
        return -1;
    }

    /* shard_meta summary */
    if ((strstr(out, "shard_meta: 21 records") == NULL) ||
        (strstr(out, "shard_meta_tiles=2")     == NULL) ||
        (strstr(out, "shard_meta_dtype_f32=21") == NULL) ||
        (strstr(out, "shard_meta_dtype_q8=0")  == NULL)) {
        return -1;
    }

    if ((parse_u64_line(out, "shard_meta_assigned",   &v) != 0) || (v != 21u)) {
        return -1;
    }
    if ((parse_u64_line(out, "shard_meta_unassigned",  &v) != 0) || (v != 0u)) {
        return -1;
    }

    /* both tile groups must appear in the shard record listing */
    if ((strstr(out, "tile=0 aimu=0") == NULL) ||
        (strstr(out, "tile=1 aimu=1") == NULL)) {
        return -1;
    }

    return 0;
}

/* --------------------------------------------------------------- bench/shard */

static int check_bench_consistency(void)
{
    char     out_rt[4096];
    char     out_md[4096];
    uint64_t last_rt   = 0u;
    uint64_t last_md   = 0u;
    uint64_t logits_rt = 0u;
    uint64_t logits_md = 0u;
    uint64_t pkts_rt   = 0u;
    uint64_t pkts_md   = 0u;

    /* --- runtime plan --- */
    if (run_command("./build/att1-bench --model " MODEL_PATH
                    " --prompt hello --tokens " TOKENS
                    " --mode cluster --tiles " TILES
                    " --shard-plan runtime --backend cpu-f32"
                    " > build/m44_bench_runtime.txt 2>&1") != 0) {
        return -1;
    }
    if (read_file("build/m44_bench_runtime.txt", out_rt, sizeof(out_rt)) != 0) {
        return -1;
    }

    /* --- metadata plan --- */
    if (run_command("./build/att1-bench --model " MODEL_PATH
                    " --prompt hello --tokens " TOKENS
                    " --mode cluster --tiles " TILES
                    " --shard-plan metadata --backend cpu-f32"
                    " > build/m44_bench_metadata.txt 2>&1") != 0) {
        return -1;
    }
    if (read_file("build/m44_bench_metadata.txt", out_md, sizeof(out_md)) != 0) {
        return -1;
    }

    /* plan label check */
    if (strstr(out_rt, "shard_plan=runtime")  == NULL) { return -1; }
    if (strstr(out_md, "shard_plan=metadata") == NULL) { return -1; }

    /* shard meta reported present for both */
    if (strstr(out_rt, "shard_meta=present") == NULL) { return -1; }
    if (strstr(out_md, "shard_meta=present") == NULL) { return -1; }

    /* parse deterministic output fields */
    if (parse_u64_line(out_rt, "last_token",            &last_rt)   != 0) { return -1; }
    if (parse_u64_line(out_md, "last_token",            &last_md)   != 0) { return -1; }
    if (parse_u64_line(out_rt, "logits_bytes_produced", &logits_rt) != 0) { return -1; }
    if (parse_u64_line(out_md, "logits_bytes_produced", &logits_md) != 0) { return -1; }
    if (parse_u64_line(out_rt, "fabric_packets_sent",   &pkts_rt)   != 0) { return -1; }
    if (parse_u64_line(out_md, "fabric_packets_sent",   &pkts_md)   != 0) { return -1; }

    /* runtime and metadata must produce identical deterministic results */
    if (last_rt   != last_md)   { return -1; }
    if (logits_rt != logits_md) { return -1; }
    if (pkts_rt   != pkts_md)   { return -1; }

    return 0;
}

/* ------------------------------------------------------------ real tiny f32 */

static int check_real_tiny_f32(void)
{
    char     out[8192];
    uint64_t pkts = 0u;

    if (run_command("./build/att1-inspect " REAL_TINY_MODEL_PATH
                    " > build/m48_real_tiny_inspect.txt 2>&1") != 0) {
        return -1;
    }
    if (read_file("build/m48_real_tiny_inspect.txt", out, sizeof(out)) != 0) {
        return -1;
    }

    if ((strstr(out, "vocab_size=16") == NULL) ||
        (strstr(out, "n_layers=2") == NULL) ||
        (strstr(out, "d_model=8") == NULL) ||
        (strstr(out, "d_ff=16") == NULL) ||
        (strstr(out, "tensor_count=21") == NULL)) {
        return -1;
    }

    if ((strstr(out, "tensor name=tok_embeddings.weight dtype=1 shape=[16,8]") == NULL) ||
        (strstr(out, "tensor name=layers.0.attention.wq.weight dtype=1 shape=[8,8]") == NULL) ||
        (strstr(out, "tensor name=layers.0.ffn.w_gate.weight dtype=1 shape=[8,16]") == NULL) ||
        (strstr(out, "tensor name=layers.0.ffn.w_down.weight dtype=1 shape=[16,8]") == NULL) ||
        (strstr(out, "tensor name=output.weight dtype=1 shape=[8,16]") == NULL)) {
        return -1;
    }

    if (run_command("./build/att1-bench --model " REAL_TINY_MODEL_PATH
                    " --prompt " REAL_TINY_PROMPT
                    " --tokens 2 --mode single --backend cpu-f32"
                    " > build/m48_real_tiny_single.txt 2>&1") != 0) {
        return -1;
    }
    if (read_file("build/m48_real_tiny_single.txt", out, sizeof(out)) != 0) {
        return -1;
    }
    if ((strstr(out, "mode=single") == NULL) ||
        (strstr(out, "backend=cpu-f32") == NULL) ||
        (strstr(out, "logits_bytes_produced=128") == NULL)) {
        return -1;
    }

    if (run_command("./build/att1-bench --model " REAL_TINY_MODEL_PATH
                    " --prompt " REAL_TINY_PROMPT
                    " --tokens 2 --mode cluster --tiles " TILES
                    " --backend cpu-f32"
                    " > build/m48_real_tiny_cluster.txt 2>&1") != 0) {
        return -1;
    }
    if (read_file("build/m48_real_tiny_cluster.txt", out, sizeof(out)) != 0) {
        return -1;
    }
    if ((strstr(out, "mode=cluster") == NULL) ||
        (strstr(out, "backend=cpu-f32") == NULL) ||
        (strstr(out, "logits_bytes_produced=128") == NULL) ||
        (parse_u64_line(out, "fabric_packets_sent", &pkts) != 0) ||
        (pkts == 0u)) {
        return -1;
    }

    return 0;
}

/* ------------------------------------------------------------- real tiny q8 */

static int check_real_tiny_q8(void)
{
    char     out[8192];
    char     f32_out[4096];
    char     q8_out[4096];
    uint64_t f32_last = 0u;
    uint64_t q8_last = 0u;
    uint64_t pkts = 0u;

    if (run_command("./build/att1-inspect " REAL_TINY_Q8_MODEL_PATH
                    " > build/m49_real_tiny_q8_inspect.txt 2>&1") != 0) {
        return -1;
    }
    if (read_file("build/m49_real_tiny_q8_inspect.txt", out, sizeof(out)) != 0) {
        return -1;
    }

    if ((strstr(out, "vocab_size=16") == NULL) ||
        (strstr(out, "tensor_count=21") == NULL) ||
        (strstr(out, "tensor name=tok_embeddings.weight dtype=1 shape=[16,8]") == NULL) ||
        (strstr(out, "tensor name=layers.0.attention.wq.weight dtype=2 shape=[8,8]") == NULL) ||
        (strstr(out, "tensor name=layers.0.ffn.w_gate.weight dtype=2 shape=[16,8]") == NULL) ||
        (strstr(out, "tensor name=layers.0.ffn.w_down.weight dtype=2 shape=[8,16]") == NULL) ||
        (strstr(out, "tensor name=output.weight dtype=2 shape=[16,8]") == NULL) ||
        (strstr(out, "quant=per-row-q8") == NULL) ||
        (strstr(out, "q8_scales=") == NULL)) {
        return -1;
    }

    if (run_command("./build/att1-bench --model " REAL_TINY_MODEL_PATH
                    " --prompt " REAL_TINY_PROMPT
                    " --tokens 2 --mode single --backend cpu-q8"
                    " > build/m49_real_tiny_f32_cpu_q8_single.txt 2>&1") != 0) {
        return -1;
    }
    if (run_command("./build/att1-bench --model " REAL_TINY_Q8_MODEL_PATH
                    " --prompt " REAL_TINY_PROMPT
                    " --tokens 2 --mode single --backend cpu-q8"
                    " > build/m49_real_tiny_q8_cpu_q8_single.txt 2>&1") != 0) {
        return -1;
    }
    if ((read_file("build/m49_real_tiny_f32_cpu_q8_single.txt", f32_out, sizeof(f32_out)) != 0) ||
        (read_file("build/m49_real_tiny_q8_cpu_q8_single.txt", q8_out, sizeof(q8_out)) != 0)) {
        return -1;
    }
    if ((strstr(q8_out, "mode=single") == NULL) ||
        (strstr(q8_out, "backend=cpu-q8") == NULL) ||
        (strstr(q8_out, "logits_bytes_produced=128") == NULL) ||
        (parse_u64_line(f32_out, "last_token", &f32_last) != 0) ||
        (parse_u64_line(q8_out, "last_token", &q8_last) != 0) ||
        (f32_last != q8_last)) {
        return -1;
    }

    if (run_command("./build/att1-bench --model " REAL_TINY_Q8_MODEL_PATH
                    " --prompt " REAL_TINY_PROMPT
                    " --tokens 2 --mode cluster --tiles " TILES
                    " --backend cpu-q8"
                    " > build/m49_real_tiny_q8_cpu_q8_cluster.txt 2>&1") != 0) {
        return -1;
    }
    if (read_file("build/m49_real_tiny_q8_cpu_q8_cluster.txt", out, sizeof(out)) != 0) {
        return -1;
    }
    if ((strstr(out, "mode=cluster") == NULL) ||
        (strstr(out, "backend=cpu-q8") == NULL) ||
        (strstr(out, "logits_bytes_produced=128") == NULL) ||
        (parse_u64_line(out, "fabric_packets_sent", &pkts) != 0) ||
        (pkts == 0u)) {
        return -1;
    }

    if (run_command("./build/att1-bench --model " REAL_TINY_Q8_MODEL_PATH
                    " --prompt " REAL_TINY_PROMPT
                    " --tokens 2 --mode single --backend cuda-q8"
                    " > build/m49_real_tiny_q8_cuda_q8_single.txt 2>&1") != 0) {
        if ((read_file("build/m49_real_tiny_q8_cuda_q8_single.txt", out, sizeof(out)) != 0) ||
            !output_is_cuda_unavailable(out)) {
            return -1;
        }
    } else {
        if ((read_file("build/m49_real_tiny_q8_cuda_q8_single.txt", out, sizeof(out)) != 0) ||
            (strstr(out, "mode=single") == NULL) ||
            (strstr(out, "backend=cuda-q8") == NULL)) {
            return -1;
        }
    }

    if (run_command("./build/att1-bench --model " REAL_TINY_Q8_MODEL_PATH
                    " --prompt " REAL_TINY_PROMPT
                    " --tokens 2 --mode cluster --tiles " TILES
                    " --backend cuda-q8"
                    " > build/m49_real_tiny_q8_cuda_q8_cluster.txt 2>&1") != 0) {
        if ((read_file("build/m49_real_tiny_q8_cuda_q8_cluster.txt", out, sizeof(out)) != 0) ||
            !output_is_cuda_unavailable(out)) {
            return -1;
        }
    } else {
        if ((read_file("build/m49_real_tiny_q8_cuda_q8_cluster.txt", out, sizeof(out)) != 0) ||
            (strstr(out, "mode=cluster") == NULL) ||
            (strstr(out, "backend=cuda-q8") == NULL)) {
            return -1;
        }
    }

    return 0;
}

/* --------------------------------------------------------- M60 pretokenized */

/*
 * check_real_tiny_pretokenized() — M60
 *
 * Validates that real_tiny_f32 and real_tiny_q8 model artifacts accept
 * externally pretokenized token IDs via att1-bench --tokenizer external.
 * Token IDs are synthetic fixtures — they exercise the token-ID plumbing
 * path, not meaningful language generation.
 *
 * Token IDs used: 1,3,5 — all within the tiny model's vocab_size=16.
 * Mapping (tiny tokenizer fixture): BOS=1, 'a'=3, 'c'=5.
 *
 * This function writes a one-ID-per-line fixture file and runs all CPU paths.
 * CUDA paths are attempted and skipped if CUDA is unavailable.
 * No Python is required.
 */
static int check_real_tiny_pretokenized(void)
{
    char     out[4096];
    uint64_t f32_last = 0u;
    uint64_t q8_last  = 0u;
    uint64_t pkts     = 0u;

    /* Write a fixture token IDs file (one ID per line). */
    {
        FILE *fp = fopen("build/m60_ids.txt", "w");
        if (fp == NULL) {
            fputs("m60: could not create token IDs fixture file\n", stderr);
            return -1;
        }
        fputs("1\n3\n5\n", fp);
        fclose(fp);
    }

    /* ------ f32 single ------ */
    if (run_command("./build/att1-bench --model " REAL_TINY_MODEL_PATH
                    " --tokens 2 --mode single --backend cpu-f32"
                    " --tokenizer external --tokens-file build/m60_ids.txt"
                    " > build/m60_f32_single.txt 2>&1") != 0) {
        fputs("m60: f32 single failed\n", stderr);
        return -1;
    }
    if (read_file("build/m60_f32_single.txt", out, sizeof(out)) != 0) {
        return -1;
    }
    if ((strstr(out, "mode=single")           == NULL) ||
        (strstr(out, "backend=cpu-f32")        == NULL) ||
        (strstr(out, "tokenizer=external")     == NULL) ||
        (strstr(out, "prompt_tokens=3")        == NULL) ||
        (strstr(out, "generated_tokens=2")     == NULL)) {
        fputs("m60: f32 single output check failed\n", stderr);
        return -1;
    }
    if (parse_u64_line(out, "last_token", &f32_last) != 0) {
        fputs("m60: f32 single: could not parse last_token\n", stderr);
        return -1;
    }

    /* ------ f32 cluster ------ */
    if (run_command("./build/att1-bench --model " REAL_TINY_MODEL_PATH
                    " --tokens 2 --mode cluster --tiles " TILES
                    " --backend cpu-f32"
                    " --tokenizer external --tokens-file build/m60_ids.txt"
                    " > build/m60_f32_cluster.txt 2>&1") != 0) {
        fputs("m60: f32 cluster failed\n", stderr);
        return -1;
    }
    if (read_file("build/m60_f32_cluster.txt", out, sizeof(out)) != 0) {
        return -1;
    }
    if ((strstr(out, "mode=cluster")           == NULL) ||
        (strstr(out, "backend=cpu-f32")         == NULL) ||
        (strstr(out, "tokenizer=external")      == NULL) ||
        (strstr(out, "tiles=2")                 == NULL) ||
        (strstr(out, "prompt_tokens=3")         == NULL) ||
        (strstr(out, "generated_tokens=2")      == NULL) ||
        (parse_u64_line(out, "fabric_packets_sent", &pkts) != 0) ||
        (pkts == 0u)) {
        fputs("m60: f32 cluster output check failed\n", stderr);
        return -1;
    }

    /* ------ q8 single ------ */
    if (run_command("./build/att1-bench --model " REAL_TINY_Q8_MODEL_PATH
                    " --tokens 2 --mode single --backend cpu-q8"
                    " --tokenizer external --tokens-file build/m60_ids.txt"
                    " > build/m60_q8_single.txt 2>&1") != 0) {
        fputs("m60: q8 single failed\n", stderr);
        return -1;
    }
    if (read_file("build/m60_q8_single.txt", out, sizeof(out)) != 0) {
        return -1;
    }
    if ((strstr(out, "mode=single")           == NULL) ||
        (strstr(out, "backend=cpu-q8")         == NULL) ||
        (strstr(out, "tokenizer=external")     == NULL) ||
        (strstr(out, "prompt_tokens=3")        == NULL) ||
        (strstr(out, "generated_tokens=2")     == NULL)) {
        fputs("m60: q8 single output check failed\n", stderr);
        return -1;
    }
    if (parse_u64_line(out, "last_token", &q8_last) != 0) {
        fputs("m60: q8 single: could not parse last_token\n", stderr);
        return -1;
    }

    /* f32 and q8 last_token may differ (quantization rounding); just assert
     * they are in range.  Both must be < vocab_size=16. */
    if ((f32_last >= 16u) || (q8_last >= 16u)) {
        fputs("m60: last_token out of range for vocab_size=16\n", stderr);
        return -1;
    }

    /* ------ q8 cluster ------ */
    if (run_command("./build/att1-bench --model " REAL_TINY_Q8_MODEL_PATH
                    " --tokens 2 --mode cluster --tiles " TILES
                    " --backend cpu-q8"
                    " --tokenizer external --tokens-file build/m60_ids.txt"
                    " > build/m60_q8_cluster.txt 2>&1") != 0) {
        fputs("m60: q8 cluster failed\n", stderr);
        return -1;
    }
    if (read_file("build/m60_q8_cluster.txt", out, sizeof(out)) != 0) {
        return -1;
    }
    if ((strstr(out, "mode=cluster")           == NULL) ||
        (strstr(out, "backend=cpu-q8")         == NULL) ||
        (strstr(out, "tokenizer=external")     == NULL) ||
        (strstr(out, "tiles=2")                 == NULL) ||
        (strstr(out, "prompt_tokens=3")         == NULL) ||
        (strstr(out, "generated_tokens=2")      == NULL) ||
        (parse_u64_line(out, "fabric_packets_sent", &pkts) != 0) ||
        (pkts == 0u)) {
        fputs("m60: q8 cluster output check failed\n", stderr);
        return -1;
    }

    /* ------ also verify with --input-token-ids (inline form) ------ */
    if (run_command("./build/att1-bench --model " REAL_TINY_MODEL_PATH
                    " --tokens 2 --mode single --backend cpu-f32"
                    " --tokenizer external --input-token-ids \"1,3,5\""
                    " > build/m60_f32_inline.txt 2>&1") != 0) {
        fputs("m60: f32 single --input-token-ids failed\n", stderr);
        return -1;
    }
    if (read_file("build/m60_f32_inline.txt", out, sizeof(out)) != 0) {
        return -1;
    }
    if ((strstr(out, "tokenizer=external") == NULL) ||
        (strstr(out, "prompt_tokens=3") == NULL)) {
        fputs("m60: f32 --input-token-ids output check failed\n", stderr);
        return -1;
    }

    /* ------ CUDA f32 single (skip if unavailable) ------ */
    if (run_command("./build/att1-bench --model " REAL_TINY_MODEL_PATH
                    " --tokens 2 --mode single --backend cuda"
                    " --tokenizer external --tokens-file build/m60_ids.txt"
                    " > build/m60_f32_cuda_single.txt 2>&1") != 0) {
        if ((read_file("build/m60_f32_cuda_single.txt", out, sizeof(out)) != 0) ||
            !output_is_cuda_unavailable(out)) {
            fputs("m60: cuda f32 single failed unexpectedly\n", stderr);
            return -1;
        }
        /* CUDA unavailable — skip remaining CUDA tests */
        return 0;
    }
    if ((read_file("build/m60_f32_cuda_single.txt", out, sizeof(out)) != 0) ||
        (strstr(out, "mode=single")       == NULL) ||
        (strstr(out, "backend=cuda")      == NULL) ||
        (strstr(out, "tokenizer=external") == NULL)) {
        fputs("m60: cuda f32 single output check failed\n", stderr);
        return -1;
    }

    /* ------ CUDA f32 cluster ------ */
    if (run_command("./build/att1-bench --model " REAL_TINY_MODEL_PATH
                    " --tokens 2 --mode cluster --tiles " TILES
                    " --backend cuda"
                    " --tokenizer external --tokens-file build/m60_ids.txt"
                    " > build/m60_f32_cuda_cluster.txt 2>&1") != 0) {
        if ((read_file("build/m60_f32_cuda_cluster.txt", out, sizeof(out)) != 0) ||
            !output_is_cuda_unavailable(out)) {
            fputs("m60: cuda f32 cluster failed unexpectedly\n", stderr);
            return -1;
        }
        return 0;
    }
    if ((read_file("build/m60_f32_cuda_cluster.txt", out, sizeof(out)) != 0) ||
        (strstr(out, "mode=cluster")      == NULL) ||
        (strstr(out, "backend=cuda")      == NULL) ||
        (strstr(out, "tokenizer=external") == NULL)) {
        fputs("m60: cuda f32 cluster output check failed\n", stderr);
        return -1;
    }

    /* ------ CUDA q8 single ------ */
    if (run_command("./build/att1-bench --model " REAL_TINY_Q8_MODEL_PATH
                    " --tokens 2 --mode single --backend cuda-q8"
                    " --tokenizer external --tokens-file build/m60_ids.txt"
                    " > build/m60_q8_cuda_single.txt 2>&1") != 0) {
        if ((read_file("build/m60_q8_cuda_single.txt", out, sizeof(out)) != 0) ||
            !output_is_cuda_unavailable(out)) {
            fputs("m60: cuda q8 single failed unexpectedly\n", stderr);
            return -1;
        }
        return 0;
    }
    if ((read_file("build/m60_q8_cuda_single.txt", out, sizeof(out)) != 0) ||
        (strstr(out, "mode=single")       == NULL) ||
        (strstr(out, "backend=cuda-q8")   == NULL) ||
        (strstr(out, "tokenizer=external") == NULL)) {
        fputs("m60: cuda q8 single output check failed\n", stderr);
        return -1;
    }

    /* ------ CUDA q8 cluster ------ */
    if (run_command("./build/att1-bench --model " REAL_TINY_Q8_MODEL_PATH
                    " --tokens 2 --mode cluster --tiles " TILES
                    " --backend cuda-q8"
                    " --tokenizer external --tokens-file build/m60_ids.txt"
                    " > build/m60_q8_cuda_cluster.txt 2>&1") != 0) {
        if ((read_file("build/m60_q8_cuda_cluster.txt", out, sizeof(out)) != 0) ||
            !output_is_cuda_unavailable(out)) {
            fputs("m60: cuda q8 cluster failed unexpectedly\n", stderr);
            return -1;
        }
        return 0;
    }
    if ((read_file("build/m60_q8_cuda_cluster.txt", out, sizeof(out)) != 0) ||
        (strstr(out, "mode=cluster")      == NULL) ||
        (strstr(out, "backend=cuda-q8")   == NULL) ||
        (strstr(out, "tokenizer=external") == NULL)) {
        fputs("m60: cuda q8 cluster output check failed\n", stderr);
        return -1;
    }

    return 0;
}

/* --------------------------------------------------------------- main ------- */

/*
 * check_real_tiny_q4() — M81
 *
 * Validates the real_tiny_q4 artifact (converted from m63 safetensors fixture:
 * vocab=64, d_model=32, d_ff=64, n_layers=2) via cpu-q4 single and cluster
 * inference paths using externally pretokenized IDs.
 *
 * Token IDs: 1, 3, 5 — all within vocab_size=64.
 * No Python is required at test time; the artifact is checked in.
 */
static int check_real_tiny_q4(void)
{
    char     out[4096];
    uint64_t pkts = 0u;

    /* Write a fixture token IDs file (one ID per line). */
    {
        FILE *fp = fopen("build/m81_ids.txt", "w");
        if (fp == NULL) {
            fputs("m81: could not create token IDs fixture file\n", stderr);
            return -1;
        }
        fputs("1\n3\n5\n", fp);
        fclose(fp);
    }

    /* --- inspect: verify q4 dtype fields and m63 config --- */
    if (run_command("./build/att1-inspect " REAL_TINY_Q4_MODEL_PATH
                    " > build/m81_inspect.txt 2>&1") != 0) {
        fputs("m81: att1-inspect failed\n", stderr);
        return -1;
    }
    if (read_file("build/m81_inspect.txt", out, sizeof(out)) != 0) {
        return -1;
    }
    if ((strstr(out, "vocab_size=64")  == NULL) ||
        (strstr(out, "d_model=32")     == NULL) ||
        (strstr(out, "n_layers=2")     == NULL) ||
        (strstr(out, "tensor_count=21") == NULL)) {
        fputs("m81: inspect config fields check failed\n", stderr);
        return -1;
    }
    if ((strstr(out, "tensor name=layers.0.attention.wq.weight dtype=3 shape=[32,32]") == NULL) ||
        (strstr(out, "tensor name=layers.0.ffn.w_gate.weight dtype=3 shape=[64,32]")  == NULL) ||
        (strstr(out, "tensor name=layers.0.ffn.w_down.weight dtype=3 shape=[32,64]")  == NULL) ||
        (strstr(out, "tensor name=output.weight dtype=3 shape=[64,32]")               == NULL) ||
        (strstr(out, "quant=grouped-q4-g32") == NULL)) {
        fputs("m81: inspect q4 tensor fields check failed\n", stderr);
        return -1;
    }

    /* --- cpu-q4 single-tile bench --- */
    if (run_command("./build/att1-bench --model " REAL_TINY_Q4_MODEL_PATH
                    " --tokens 2 --mode single --backend cpu-q4"
                    " --tokenizer external --tokens-file build/m81_ids.txt"
                    " > build/m81_q4_single.txt 2>&1") != 0) {
        fputs("m81: cpu-q4 single failed\n", stderr);
        return -1;
    }
    if (read_file("build/m81_q4_single.txt", out, sizeof(out)) != 0) {
        return -1;
    }
    if ((strstr(out, "mode=single")       == NULL) ||
        (strstr(out, "backend=cpu-q4")    == NULL) ||
        (strstr(out, "tokenizer=external") == NULL) ||
        (strstr(out, "prompt_tokens=3")   == NULL) ||
        (strstr(out, "generated_tokens=2") == NULL)) {
        fputs("m81: cpu-q4 single output check failed\n", stderr);
        return -1;
    }

    /* --- cpu-q4 cluster bench --- */
    if (run_command("./build/att1-bench --model " REAL_TINY_Q4_MODEL_PATH
                    " --tokens 2 --mode cluster --tiles " TILES
                    " --backend cpu-q4"
                    " --tokenizer external --tokens-file build/m81_ids.txt"
                    " > build/m81_q4_cluster.txt 2>&1") != 0) {
        fputs("m81: cpu-q4 cluster failed\n", stderr);
        return -1;
    }
    if (read_file("build/m81_q4_cluster.txt", out, sizeof(out)) != 0) {
        return -1;
    }
    if ((strstr(out, "mode=cluster")        == NULL) ||
        (strstr(out, "backend=cpu-q4")      == NULL) ||
        (strstr(out, "tokenizer=external")  == NULL) ||
        (strstr(out, "tiles=2")             == NULL) ||
        (strstr(out, "prompt_tokens=3")     == NULL) ||
        (strstr(out, "generated_tokens=2")  == NULL) ||
        (parse_u64_line(out, "fabric_packets_sent", &pkts) != 0) ||
        (pkts == 0u)) {
        fputs("m81: cpu-q4 cluster output check failed\n", stderr);
        return -1;
    }

    return 0;
}

/* --------------------------------------------------------------- main ------- */

int main(void)
{
    if ((check_inspect()                  != 0) ||
        (check_bench_consistency()        != 0) ||
        (check_real_tiny_f32()            != 0) ||
        (check_real_tiny_q8()             != 0) ||
        (check_real_tiny_pretokenized()   != 0) ||
        (check_real_tiny_q4()             != 0)) {
        fputs("converter validation test failed\n", stderr);
        return 1;
    }

    puts("converter validation test passed");
    return 0;
}
