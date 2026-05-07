/*
 * test_converter_validation.c
 *
 * M44: Validation of converter-generated .att1 artifacts with shard metadata.
 * M48: Validation of checked-in tiny f32 safetensors conversion artifact.
 * M49: Validation of checked-in tiny q8 safetensors conversion artifact.
 *
 * Uses checked-in .att1 fixtures only; no Python is required at test time.
 *
 * Checks:
 *   1. att1-inspect produces expected tensor/shard_meta fields.
 *   2. att1-bench with --shard-plan runtime produces correct output.
 *   3. att1-bench with --shard-plan metadata produces identical last_token,
 *      logits_bytes_produced, and fabric_packets_sent.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MODEL_PATH "models/converted_stub_meta/model.att1"
#define REAL_TINY_MODEL_PATH "models/real_tiny_f32/model.att1"
#define REAL_TINY_Q8_MODEL_PATH "models/real_tiny_q8/model.att1"
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

/* --------------------------------------------------------------- main ------- */

int main(void)
{
    if ((check_inspect()           != 0) ||
        (check_bench_consistency() != 0) ||
        (check_real_tiny_f32()     != 0) ||
        (check_real_tiny_q8()      != 0)) {
        fputs("converter validation test failed\n", stderr);
        return 1;
    }

    puts("converter validation test passed");
    return 0;
}
