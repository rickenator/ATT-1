/* q8 benchmark and trace integration tests (M26) */
#include "att1_backend.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

static int run_command(const char *command)
{
    const int rc = system(command);

    return rc == 0 ? 0 : -1;
}

static int run_command_with_exit(const char *command, int *out_rc)
{
    if (out_rc == NULL) {
        return -1;
    }
    *out_rc = system(command);
    return 0;
}

static int read_file(const char *path, char *buffer, size_t capacity)
{
    FILE *fp = NULL;
    size_t nread = 0u;

    if ((path == NULL) || (buffer == NULL) || (capacity == 0u)) {
        return -1;
    }

    fp = fopen(path, "rb");
    if (fp == NULL) {
        return -1;
    }

    nread = fread(buffer, 1u, capacity - 1u, fp);
    if (ferror(fp) != 0) {
        fclose(fp);
        return -1;
    }

    buffer[nread] = '\0';
    fclose(fp);
    return 0;
}

static int parse_u32_line(const char *text, const char *key, uint32_t *out)
{
    const char *line = NULL;
    char *end = NULL;
    unsigned long value = 0u;

    if ((text == NULL) || (key == NULL) || (out == NULL)) {
        return -1;
    }

    line = strstr(text, key);
    if (line == NULL) {
        return -1;
    }

    line += strlen(key);
    if (*line != '=') {
        return -1;
    }
    line++;

    value = strtoul(line, &end, 10);
    if (end == line) {
        return -1;
    }

    *out = (uint32_t)value;
    return 0;
}

/* Test 1: att1-bench --backend cpu-q8 --mode single exits zero with
 * required output fields */
static int test_cpu_q8_bench_single_mode(void)
{
    char output[4096];

    if (run_command("./build/att1-bench --model models/dummy/model.att1 "
                    "--prompt hello --tokens 8 --mode single --backend cpu-q8 "
                    "> build/q8bench_cpu_single.txt 2>&1") != 0) {
        fputs("cpu q8 single mode exited non-zero\n", stderr);
        return -1;
    }

    if (read_file("build/q8bench_cpu_single.txt", output, sizeof(output)) !=
        0) {
        fputs("failed to read cpu q8 bench output\n", stderr);
        return -1;
    }

    if (strstr(output, "mode=single") == NULL) {
        fputs("cpu q8 bench output missing 'mode=single'\n", stderr);
        return -1;
    }

    if (strstr(output, "backend=cpu-q8") == NULL) {
        fputs("cpu q8 bench output missing 'backend=cpu-q8'\n", stderr);
        return -1;
    }

    if (strstr(output, "tokens_decoded=") == NULL) {
        fputs("cpu q8 bench output missing trace counters\n", stderr);
        return -1;
    }

    if (strstr(output, "generated_tokens=") == NULL) {
        fputs("cpu q8 bench output missing generated_tokens\n", stderr);
        return -1;
    }

    if (strstr(output, "logits_bytes_produced=") == NULL) {
        fputs("cpu q8 bench output missing logits_bytes_produced\n", stderr);
        return -1;
    }

    if (strstr(output, "kv_appends=") == NULL) {
        fputs("cpu q8 bench output missing kv_appends\n", stderr);
        return -1;
    }

    /* Must not silently report a different backend */
    if ((strstr(output, "backend=cpu-f32") != NULL) ||
        (strstr(output, "backend=cuda") != NULL)) {
        fputs("cpu q8 bench silently reported a different backend\n", stderr);
        return -1;
    }

    fputs("PASS: cpu q8 bench single mode\n", stderr);
    return 0;
}

/* Test 2: CPU f32 and CPU q8 generate same last_token for dummy model */
static int test_cpu_q8_bench_deterministic(void)
{
    char f32_output[4096];
    char q8_output[4096];
    uint32_t f32_token = 0u;
    uint32_t q8_token = 0u;

    if (run_command("./build/att1-bench --model models/dummy/model.att1 "
                    "--prompt hello --tokens 8 --mode single --backend cpu-f32 "
                    "> build/q8bench_f32_det.txt") != 0) {
        fputs("cpu f32 bench failed\n", stderr);
        return -1;
    }

    if (run_command("./build/att1-bench --model models/dummy/model.att1 "
                    "--prompt hello --tokens 8 --mode single --backend cpu-q8 "
                    "> build/q8bench_q8_det.txt") != 0) {
        fputs("cpu q8 bench failed\n", stderr);
        return -1;
    }

    if ((read_file("build/q8bench_f32_det.txt",
                   f32_output,
                   sizeof(f32_output)) != 0) ||
        (read_file("build/q8bench_q8_det.txt",
                   q8_output,
                   sizeof(q8_output)) != 0)) {
        fputs("failed to read bench outputs\n", stderr);
        return -1;
    }

    if ((parse_u32_line(f32_output, "last_token", &f32_token) != 0) ||
        (parse_u32_line(q8_output, "last_token", &q8_token) != 0)) {
        fputs("failed to parse last_token\n", stderr);
        return -1;
    }

    if (f32_token != q8_token) {
        fprintf(stderr,
                "determinism check failed: f32_token=%u q8_token=%u\n",
                f32_token, q8_token);
        return -1;
    }

    fputs("PASS: cpu f32 and cpu q8 generate identical last token\n", stderr);
    return 0;
}

/* Test 3: CPU q8 and CUDA q8 generate same last_token (CUDA-only) */
static int test_cuda_q8_bench_deterministic(void)
{
    char q8_output[4096];
    char cuda_q8_output[4096];
    uint32_t q8_token = 0u;
    uint32_t cuda_q8_token = 0u;

    if (!att1_backend_cuda_available()) {
        fputs("SKIP: CUDA not available\n", stderr);
        return 0;
    }

    if (run_command("./build/att1-bench --model models/dummy/model.att1 "
                    "--prompt hello --tokens 8 --mode single --backend cpu-q8 "
                    "> build/q8bench_cpuq8_cmp.txt") != 0) {
        fputs("cpu q8 bench failed\n", stderr);
        return -1;
    }

    if (run_command("./build/att1-bench --model models/dummy/model.att1 "
                    "--prompt hello --tokens 8 --mode single --backend cuda-q8 "
                    "> build/q8bench_cudaq8_cmp.txt") != 0) {
        fputs("cuda q8 bench failed\n", stderr);
        return -1;
    }

    if ((read_file("build/q8bench_cpuq8_cmp.txt",
                   q8_output,
                   sizeof(q8_output)) != 0) ||
        (read_file("build/q8bench_cudaq8_cmp.txt",
                   cuda_q8_output,
                   sizeof(cuda_q8_output)) != 0)) {
        fputs("failed to read bench outputs\n", stderr);
        return -1;
    }

    if ((parse_u32_line(q8_output, "last_token", &q8_token) != 0) ||
        (parse_u32_line(cuda_q8_output, "last_token", &cuda_q8_token) != 0)) {
        fputs("failed to parse last_token\n", stderr);
        return -1;
    }

    if (q8_token != cuda_q8_token) {
        fprintf(stderr,
                "determinism check failed: q8_token=%u cuda_q8_token=%u\n",
                q8_token, cuda_q8_token);
        return -1;
    }

    fputs("PASS: cpu q8 and cuda q8 generate identical last token\n", stderr);
    return 0;
}

/* Test 4: Cluster mode with q8 backends fails clearly */
static int test_q8_bench_cluster_mode(void)
{
    char output[4096];
    int rc = 0;

    /* cuda-q8 cluster must fail explicitly */
    if (run_command_with_exit(
            "./build/att1-bench --model models/dummy/model.att1 "
            "--prompt hello --tokens 4 --mode cluster --tiles 2 "
            "--backend cuda-q8 > build/q8bench_cudaq8_cluster.txt 2>&1",
            &rc) != 0) {
        return -1;
    }

    if (rc == 0) {
        fputs("cuda q8 cluster mode unexpectedly succeeded\n", stderr);
        return -1;
    }

    if (read_file("build/q8bench_cudaq8_cluster.txt",
                  output,
                  sizeof(output)) != 0) {
        fputs("failed to read cuda q8 cluster output\n", stderr);
        return -1;
    }

    if (strstr(output, "backend=cpu-f32") != NULL) {
        fputs("cuda q8 cluster silently fell back to cpu-f32\n", stderr);
        return -1;
    }

    if (strstr(output, "unsupported") == NULL) {
        fputs("cuda q8 cluster mode missing 'unsupported' message\n", stderr);
        return -1;
    }

    fputs("PASS: q8 cluster mode fails clearly\n", stderr);
    return 0;
}

/* Test 5: Non-CUDA build reports cuda-q8 unsupported cleanly */
static int test_cuda_q8_bench_unsupported(void)
{
    char output[4096];
    int rc = 0;

    if (att1_backend_cuda_available()) {
        fputs("SKIP: CUDA is available (test requires CPU-only build)\n",
              stderr);
        return 0;
    }

    if (run_command_with_exit(
            "./build/att1-bench --model models/dummy/model.att1 "
            "--prompt hello --tokens 4 --mode single --backend cuda-q8 "
            "> build/q8bench_cudaq8_unsupported.txt 2>&1",
            &rc) != 0) {
        return -1;
    }

    if (rc == 0) {
        fputs("expected non-zero exit for cuda-q8 on CPU-only build\n",
              stderr);
        return -1;
    }

    if (read_file("build/q8bench_cudaq8_unsupported.txt",
                  output,
                  sizeof(output)) != 0) {
        fputs("failed to read cuda q8 unsupported output\n", stderr);
        return -1;
    }

    if (strstr(output, "unsupported") == NULL) {
        fputs("missing 'unsupported' message for cuda-q8 on CPU-only build\n",
              stderr);
        return -1;
    }

    fputs("PASS: cuda q8 unsupported on CPU-only build\n", stderr);
    return 0;
}

int main(void)
{
    int failures = 0;

    if (test_cpu_q8_bench_single_mode() != 0) {
        failures++;
    }
    if (test_cpu_q8_bench_deterministic() != 0) {
        failures++;
    }
    if (test_cuda_q8_bench_deterministic() != 0) {
        failures++;
    }
    if (test_q8_bench_cluster_mode() != 0) {
        failures++;
    }
    if (test_cuda_q8_bench_unsupported() != 0) {
        failures++;
    }

    return failures == 0 ? 0 : 1;
}
