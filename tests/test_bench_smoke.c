#include "att1_backend.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int run_command(const char *command)
{
    const int rc = system(command);

    return rc == 0 ? 0 : -1;
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

static int check_bench_tools(void)
{
    char output[4096];

    if (run_command("./build/att1-bench --model models/dummy/model.att1 "
                    "--prompt hello --tokens 8 --mode single "
                    "> build/bench_single.txt") != 0) {
        return -1;
    }

    if (run_command("./build/att1-bench --model models/dummy/model.att1 "
                    "--prompt hello --tokens 8 --mode cluster --tiles 2 "
                    "> build/bench_cluster.txt") != 0) {
        return -1;
    }

    if ((read_file("build/bench_single.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "mode=single") == NULL) ||
        (strstr(output, "backend=cpu-f32") == NULL) ||
        (strstr(output, "tokens_decoded=") == NULL)) {
        return -1;
    }

    if ((read_file("build/bench_cluster.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "mode=cluster") == NULL) ||
        (strstr(output, "tiles=2") == NULL) ||
        (strstr(output, "fabric_packets_sent=") == NULL)) {
        return -1;
    }

    if (run_command("./build/att1-bench --model models/dummy/model.att1 "
                    "--prompt hello --tokens 8 --mode single --backend cpu-q8 "
                    "> build/bench_single_cpu_q8.txt") != 0) {
        return -1;
    }

    if ((read_file("build/bench_single_cpu_q8.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "mode=single") == NULL) ||
        (strstr(output, "backend=cpu-q8") == NULL) ||
        (strstr(output, "tokens_decoded=") == NULL)) {
        return -1;
    }

    if (run_command("./build/att1-q8-bench --iterations 8 "
                    "> build/bench_q8.txt") != 0) {
        return -1;
    }

    if ((read_file("build/bench_q8.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "mode=q8-matmul") == NULL) ||
        (strstr(output, "backend=cpu-q8") == NULL) ||
        (strstr(output, "max_abs_error=") == NULL)) {
        return -1;
    }

    if (att1_backend_cuda_available()) {
        /* CUDA is present: --backend cuda must not silently fall back to
           cpu-f32.  Commands will exit non-zero while CUDA kernels are not
           yet implemented, but no cpu-f32 output should appear. */
        (void)run_command(
            "./build/att1-bench --model models/dummy/model.att1 "
            "--prompt hello --tokens 8 --mode single --backend cuda "
            "> build/bench_cuda.txt 2>&1");

        if ((read_file("build/bench_cuda.txt", output, sizeof(output)) != 0) ||
            (strstr(output, "backend=cpu-f32") != NULL)) {
            fputs("cuda bench silently fell back to cpu-f32\n", stderr);
            return -1;
        }

        (void)run_command(
            "./build/att1-q8-bench --iterations 8 --backend cuda "
            "> build/bench_q8_cuda.txt 2>&1");

        if ((read_file("build/bench_q8_cuda.txt",
                       output,
                       sizeof(output)) != 0) ||
            (strstr(output, "backend=cpu-f32") != NULL)) {
            fputs("cuda q8-bench silently fell back to cpu-f32\n", stderr);
            return -1;
        }
    } else {
        /* CUDA is unavailable or not compiled in: must fail with clear
           unsupported/unavailable messaging. */
        if (run_command(
                "./build/att1-bench --model models/dummy/model.att1 "
                "--prompt hello --tokens 8 --mode single --backend cuda "
                "> build/bench_cuda_unsupported.txt 2>&1") == 0) {
            return -1;
        }

        if ((read_file("build/bench_cuda_unsupported.txt",
                       output,
                       sizeof(output)) != 0) ||
            (strstr(output, "backend unsupported or unavailable: cuda") ==
             NULL)) {
            return -1;
        }

        if (run_command(
                "./build/att1-q8-bench --iterations 8 --backend cuda "
                "> build/bench_q8_cuda_unsupported.txt 2>&1") == 0) {
            return -1;
        }

        if ((read_file("build/bench_q8_cuda_unsupported.txt",
                       output,
                       sizeof(output)) != 0) ||
            (strstr(output, "cuda backend unsupported or unavailable") ==
             NULL)) {
            return -1;
        }
    }

    return 0;
}

static int check_size_tools(void)
{
    char output[4096];
    uint64_t f32_bytes = 0u;
    uint64_t f16_bytes = 0u;
    uint64_t q8_bytes = 0u;
    uint64_t q4_bytes = 0u;

    if (run_command("./build/att1-size --preset tiny-dummy "
                    "> build/size_tiny.txt") != 0) {
        return -1;
    }

    if (run_command("./build/att1-size --preset gpt-oss-120b-shape "
                    "--tiles 8 --context 8192 --dtype q4 "
                    "> build/size_gptoss.txt") != 0) {
        return -1;
    }

    if (read_file("build/size_gptoss.txt", output, sizeof(output)) != 0) {
        return -1;
    }

    if ((strstr(output, "synthetic/non-executable") == NULL) ||
        (strstr(output, "no real gpt-oss-120b inference is attempted") == NULL) ||
        (strstr(output, "dtype=q4") == NULL) ||
        (strstr(output, "max_seq_len=8192") == NULL) ||
        (strstr(output, "tiles=8") == NULL)) {
        return -1;
    }

    if (read_file("build/size_tiny.txt", output, sizeof(output)) != 0) {
        return -1;
    }

    if ((parse_u64_line(output, "model_bytes_f32", &f32_bytes) != 0) ||
        (parse_u64_line(output, "model_bytes_f16", &f16_bytes) != 0) ||
        (parse_u64_line(output, "model_bytes_q8", &q8_bytes) != 0) ||
        (parse_u64_line(output, "model_bytes_q4", &q4_bytes) != 0)) {
        return -1;
    }

    if ((f32_bytes != (f16_bytes * 2u)) ||
        (f16_bytes != (q8_bytes * 2u)) ||
        (q4_bytes != ((q8_bytes + 1u) / 2u))) {
        return -1;
    }

    return 0;
}

int main(void)
{
    if ((check_bench_tools() != 0) || (check_size_tools() != 0)) {
        fputs("bench smoke test failed\n", stderr);
        return 1;
    }

    puts("bench smoke test passed");
    return 0;
}
