/* CUDA benchmark and trace integration tests (M21) */
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

static int command_exit_code(int rc)
{
    if (WIFEXITED(rc)) {
        return WEXITSTATUS(rc);
    }
    return -1;
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

static int parse_u64_line(const char *text, const char *key, uint64_t *out)
{
    const char *line = NULL;
    char *end = NULL;
    unsigned long long value = 0u;

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

    value = strtoull(line, &end, 10);
    if (end == line) {
        return -1;
    }

    *out = (uint64_t)value;
    return 0;
}

/* Test 1: att1-bench --backend cuda --mode single exits zero (CUDA-only) */
static int test_cuda_bench_single_mode(void)
{
    char output[4096];
    int rc = 0;

    if (!att1_backend_cuda_available()) {
        fputs("SKIP: CUDA not available\n", stderr);
        return 0;
    }

    if (run_command_with_exit(
            "./build/att1-bench --model models/dummy/model.att1 "
            "--prompt hello --tokens 8 --mode single --backend cuda "
            "> build/bench_cuda_single.txt 2>&1",
            &rc) != 0) {
        return -1;
    }

    if (rc != 0) {
        fputs("cuda single mode exited non-zero\n", stderr);
        return -1;
    }

    if (read_file("build/bench_cuda_single.txt", output, sizeof(output)) != 0) {
        fputs("failed to read cuda bench output\n", stderr);
        return -1;
    }

    if (strstr(output, "backend=cuda") == NULL) {
        fputs("cuda bench output missing 'backend=cuda'\n", stderr);
        return -1;
    }

    if (strstr(output, "mode=single") == NULL) {
        fputs("cuda bench output missing 'mode=single'\n", stderr);
        return -1;
    }

    if (strstr(output, "tokens_decoded=") == NULL) {
        fputs("cuda bench output missing trace counters\n", stderr);
        return -1;
    }

    if (strstr(output, "generated_tokens=") == NULL) {
        fputs("cuda bench output missing generated_tokens\n", stderr);
        return -1;
    }

    fputs("PASS: cuda bench single mode\n", stderr);
    return 0;
}

/* Test 2: att1-bench --backend cuda --mode cluster follows availability */
static int test_cuda_bench_cluster_mode(void)
{
    char output[4096];
    int rc = 0;
    int exit_code = 0;
    uint64_t fabric_packets = 0u;
    uint64_t activation_bytes = 0u;

    if (run_command_with_exit(
            "./build/att1-bench --model models/dummy/model.att1 "
            "--prompt hello --tokens 8 --mode cluster --tiles 2 --backend cuda "
            "> build/bench_cuda_cluster.txt 2>&1",
            &rc) != 0) {
        return -1;
    }

    if (WIFSIGNALED(rc)) {
        fprintf(stderr,
                "cuda cluster mode crashed with signal %d\n",
                WTERMSIG(rc));
        return -1;
    }

    exit_code = command_exit_code(rc);
    if ((exit_code == 139) || (exit_code == 134)) {
        fprintf(stderr,
                "cuda cluster mode crashed with exit status %d\n",
                exit_code);
        return -1;
    }

    if (read_file("build/bench_cuda_cluster.txt", output, sizeof(output)) != 0) {
        fputs("failed to read cuda cluster bench output\n", stderr);
        return -1;
    }

    if (att1_backend_cuda_available()) {
        if (rc != 0) {
            fputs("cuda cluster mode exited non-zero\n", stderr);
            return -1;
        }
        if ((strstr(output, "mode=cluster") == NULL) ||
            (strstr(output, "backend=cuda") == NULL) ||
            (strstr(output, "tokens_decoded=") == NULL) ||
            (parse_u64_line(output,
                            "fabric_packets_sent",
                            &fabric_packets) != 0) ||
            (parse_u64_line(output,
                            "activation_bytes_sent",
                            &activation_bytes) != 0) ||
            (fabric_packets == 0u) ||
            (activation_bytes == 0u)) {
            fputs("cuda cluster bench output missing required counters\n",
                  stderr);
            return -1;
        }
    } else {
        if (rc == 0) {
            fputs("cuda cluster mode unexpectedly succeeded\n", stderr);
            return -1;
        }
        if (strstr(output, "backend unsupported or unavailable: cuda") ==
            NULL) {
            fputs("cuda cluster mode missing unsupported message\n", stderr);
            return -1;
        }
    }

    if (strstr(output, "backend=cpu-f32") != NULL) {
        fputs("cuda cluster mode silently fell back to cpu-f32\n", stderr);
        return -1;
    }

    fputs("PASS: cuda cluster mode availability behavior\n", stderr);
    return 0;
}

/* Test 3: CPU f32 and CUDA single-tile generate identical tokens for dummy
 * model */
static int test_cuda_bench_deterministic_output(void)
{
    char cpu_output[4096];
    char cuda_output[4096];
    uint32_t cpu_token = 0u;
    uint32_t cuda_token = 0u;

    if (!att1_backend_cuda_available()) {
        fputs("SKIP: CUDA not available\n", stderr);
        return 0;
    }

    /* Run CPU benchmark */
    if (run_command("./build/att1-bench --model models/dummy/model.att1 "
                    "--prompt hello --tokens 8 --mode single --backend cpu-f32 "
                    "> build/bench_cpu_det.txt") != 0) {
        fputs("cpu f32 bench failed\n", stderr);
        return -1;
    }

    /* Run CUDA benchmark */
    if (run_command("./build/att1-bench --model models/dummy/model.att1 "
                    "--prompt hello --tokens 8 --mode single --backend cuda "
                    "> build/bench_cuda_det.txt") != 0) {
        fputs("cuda bench failed\n", stderr);
        return -1;
    }

    if ((read_file("build/bench_cpu_det.txt", cpu_output, sizeof(cpu_output)) !=
         0) ||
        (read_file("build/bench_cuda_det.txt", cuda_output,
                   sizeof(cuda_output)) != 0)) {
        fputs("failed to read bench outputs\n", stderr);
        return -1;
    }

    /* Extract last token from each */
    if ((parse_u32_line(cpu_output, "last_token", &cpu_token) != 0) ||
        (parse_u32_line(cuda_output, "last_token", &cuda_token) != 0)) {
        fputs("failed to parse last_token from bench outputs\n", stderr);
        return -1;
    }

    if (cpu_token != cuda_token) {
        fprintf(stderr,
                "determinism check failed: cpu_token=%u cuda_token=%u\n",
                cpu_token, cuda_token);
        return -1;
    }

    fputs("PASS: cpu and cuda generate identical tokens\n", stderr);
    return 0;
}

/* Test 4: Benchmark output explicitly includes backend=cuda */
static int test_cuda_bench_backend_label(void)
{
    char output[4096];

    if (!att1_backend_cuda_available()) {
        fputs("SKIP: CUDA not available\n", stderr);
        return 0;
    }

    if (run_command("./build/att1-bench --model models/dummy/model.att1 "
                    "--prompt hello --tokens 4 --mode single --backend cuda "
                    "> build/bench_cuda_label.txt") != 0) {
        fputs("cuda bench failed\n", stderr);
        return -1;
    }

    if (read_file("build/bench_cuda_label.txt", output, sizeof(output)) != 0) {
        fputs("failed to read bench output\n", stderr);
        return -1;
    }

    if (strstr(output, "backend=cuda") == NULL) {
        fputs("bench output missing backend=cuda label\n", stderr);
        return -1;
    }

    fputs("PASS: backend=cuda explicitly reported\n", stderr);
    return 0;
}

/* Test 5: CUDA q8 single mode follows explicit backend selection */
static int test_cuda_q8_bench_single_mode(void)
{
    char output[4096];
    int rc = 0;

    if (!att1_backend_cuda_available()) {
        fputs("SKIP: CUDA not available\n", stderr);
        return 0;
    }

    if (run_command_with_exit(
            "./build/att1-bench --model models/dummy/model.att1 "
            "--prompt hello --tokens 4 --mode single --backend cuda-q8 "
            "> build/bench_cuda_q8_single.txt 2>&1",
            &rc) != 0) {
        return -1;
    }

    if (rc != 0) {
        fputs("cuda q8 single mode exited non-zero\n", stderr);
        return -1;
    }

    if (read_file("build/bench_cuda_q8_single.txt",
                  output,
                  sizeof(output)) != 0) {
        fputs("failed to read cuda q8 bench output\n", stderr);
        return -1;
    }

    if ((strstr(output, "mode=single") == NULL) ||
        (strstr(output, "backend=cuda-q8") == NULL) ||
        (strstr(output, "tokens_decoded=") == NULL) ||
        (strstr(output, "generated_tokens=") == NULL)) {
        fputs("cuda q8 bench output missing required fields\n", stderr);
        return -1;
    }

    if ((strstr(output, "backend=cpu-f32") != NULL) ||
        (strstr(output, "backend=cpu-q8") != NULL)) {
        fputs("cuda q8 bench silently reported a CPU backend\n", stderr);
        return -1;
    }

    fputs("PASS: cuda q8 bench single mode\n", stderr);
    return 0;
}

/* Test 6: Non-CUDA build reports CUDA unsupported cleanly */
static int test_cuda_bench_unsupported_message(void)
{
    char output[4096];
    char q8_output[4096];
    int rc = 0;
    int q8_rc = 0;

    if (att1_backend_cuda_available()) {
        fputs("SKIP: CUDA is available (test requires CPU-only build)\n", stderr);
        return 0;
    }

    if (run_command_with_exit(
            "./build/att1-bench --model models/dummy/model.att1 "
            "--prompt hello --tokens 4 --mode single --backend cuda "
            "> build/bench_cuda_unsupported.txt 2>&1",
            &rc) != 0) {
        return -1;
    }

    /* Non-CUDA build should reject CUDA backend */
    if (rc == 0) {
        fputs("expected non-zero exit for CUDA on CPU-only build\n", stderr);
        return -1;
    }

    if (read_file("build/bench_cuda_unsupported.txt", output,
                  sizeof(output)) != 0) {
        fputs("failed to read bench output\n", stderr);
        return -1;
    }

    if (strstr(output, "unsupported") == NULL) {
        fputs("missing 'unsupported' error message\n", stderr);
        return -1;
    }

    if (run_command_with_exit(
            "./build/att1-bench --model models/dummy/model.att1 "
            "--prompt hello --tokens 4 --mode single --backend cuda-q8 "
            "> build/bench_cuda_q8_unsupported.txt 2>&1",
            &q8_rc) != 0) {
        return -1;
    }

    if (q8_rc == 0) {
        fputs("expected non-zero exit for CUDA q8 on CPU-only build\n", stderr);
        return -1;
    }

    if (read_file("build/bench_cuda_q8_unsupported.txt",
                  q8_output,
                  sizeof(q8_output)) != 0) {
        fputs("failed to read cuda q8 bench output\n", stderr);
        return -1;
    }

    if (strstr(q8_output, "unsupported") == NULL) {
        fputs("missing cuda q8 'unsupported' error message\n", stderr);
        return -1;
    }

    fputs("PASS: CUDA unsupported message on CPU-only build\n", stderr);
    return 0;
}

int main(void)
{
    int failures = 0;

    if (test_cuda_bench_single_mode() != 0) {
        failures++;
    }
    if (test_cuda_bench_cluster_mode() != 0) {
        failures++;
    }
    if (test_cuda_bench_deterministic_output() != 0) {
        failures++;
    }
    if (test_cuda_bench_backend_label() != 0) {
        failures++;
    }
    if (test_cuda_q8_bench_single_mode() != 0) {
        failures++;
    }
    if (test_cuda_bench_unsupported_message() != 0) {
        failures++;
    }

    return failures == 0 ? 0 : 1;
}
