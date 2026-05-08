/* q4 benchmark and trace integration tests (M80) */
#include "att1_backend.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define Q4_MODEL  "models/q4_tiny/model.att1"
#define F32_MODEL "models/dummy/model.att1"

/* Q4 fixture: vocab=256, d_model=32, max_seq_len=8.
 * With --prompt A (1 byte) and --tokens 4:
 *   benchmark_token_count caps run_tokens = min(4, 8-1+1) = 4.
 *   Total decode calls = 1 prompt + 3 generation = 4; all within max_seq_len. */
#define Q4_PROMPT "A"
#define Q4_TOKENS "4"

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

/* Test 1: att1-bench --backend cpu-q4 --mode single exits zero with
 * required output fields. */
static int test_q4_bench_single_exits_zero(void)
{
    char output[4096];

    if (run_command("./build/att1-bench"
                    " --model " Q4_MODEL
                    " --prompt " Q4_PROMPT
                    " --tokens " Q4_TOKENS
                    " --mode single"
                    " --backend cpu-q4"
                    " > build/q4bench_single.txt 2>&1") != 0) {
        fputs("FAIL: q4 bench single mode exited non-zero\n", stderr);
        return -1;
    }

    if (read_file("build/q4bench_single.txt", output, sizeof(output)) != 0) {
        fputs("FAIL: q4 bench single — failed to read output\n", stderr);
        return -1;
    }

    if (strstr(output, "mode=single") == NULL) {
        fputs("FAIL: q4 bench single — output missing 'mode=single'\n", stderr);
        return -1;
    }

    if (strstr(output, "backend=cpu-q4") == NULL) {
        fputs("FAIL: q4 bench single — output missing 'backend=cpu-q4'\n", stderr);
        return -1;
    }

    if (strstr(output, "tokens_decoded=") == NULL) {
        fputs("FAIL: q4 bench single — output missing 'tokens_decoded'\n", stderr);
        return -1;
    }

    if (strstr(output, "logits_bytes_produced=") == NULL) {
        fputs("FAIL: q4 bench single — output missing 'logits_bytes_produced'\n",
              stderr);
        return -1;
    }

    if (strstr(output, "kv_appends=") == NULL) {
        fputs("FAIL: q4 bench single — output missing 'kv_appends'\n", stderr);
        return -1;
    }

    if (strstr(output, "generated_tokens=") == NULL) {
        fputs("FAIL: q4 bench single — output missing 'generated_tokens'\n", stderr);
        return -1;
    }

    /* Must not silently report a different backend. */
    if ((strstr(output, "backend=cpu-f32") != NULL) ||
        (strstr(output, "backend=cpu-q8") != NULL)) {
        fputs("FAIL: q4 bench single silently reported a different backend\n",
              stderr);
        return -1;
    }

    fputs("PASS: q4 bench single mode exits zero\n", stderr);
    return 0;
}

/* Test 2: att1-bench --backend cpu-q4 --mode cluster exits zero and reports
 * fabric_packets_sent > 0. */
static int test_q4_bench_cluster_exits_zero(void)
{
    char output[4096];
    uint64_t fabric_packets = 0u;

    if (run_command("./build/att1-bench"
                    " --model " Q4_MODEL
                    " --prompt " Q4_PROMPT
                    " --tokens " Q4_TOKENS
                    " --mode cluster --tiles 2"
                    " --backend cpu-q4"
                    " > build/q4bench_cluster.txt 2>&1") != 0) {
        fputs("FAIL: q4 bench cluster mode exited non-zero\n", stderr);
        return -1;
    }

    if (read_file("build/q4bench_cluster.txt", output, sizeof(output)) != 0) {
        fputs("FAIL: q4 bench cluster — failed to read output\n", stderr);
        return -1;
    }

    if (strstr(output, "mode=cluster") == NULL) {
        fputs("FAIL: q4 bench cluster — output missing 'mode=cluster'\n", stderr);
        return -1;
    }

    if (strstr(output, "backend=cpu-q4") == NULL) {
        fputs("FAIL: q4 bench cluster — output missing 'backend=cpu-q4'\n", stderr);
        return -1;
    }

    if (strstr(output, "shard_plan=runtime") == NULL) {
        fputs("FAIL: q4 bench cluster — output missing 'shard_plan=runtime'\n",
              stderr);
        return -1;
    }

    if (parse_u64_line(output, "fabric_packets_sent", &fabric_packets) != 0) {
        fputs("FAIL: q4 bench cluster — failed to parse fabric_packets_sent\n",
              stderr);
        return -1;
    }

    if (fabric_packets == 0u) {
        fputs("FAIL: q4 bench cluster — fabric_packets_sent is zero\n", stderr);
        return -1;
    }

    fputs("PASS: q4 bench cluster mode exits zero\n", stderr);
    return 0;
}

/* Test 3: --backend cuda-q4 single mode.
 * M88: succeeds when CUDA is available; fails clearly when CUDA is unavailable. */
static int test_q4_bench_cuda_q4_status(void)
{
    int exit_code = 0;
    char output[4096];

    exit_code = system("./build/att1-bench"
                       " --model " Q4_MODEL
                       " --prompt " Q4_PROMPT
                       " --tokens " Q4_TOKENS
                       " --mode single"
                       " --backend cuda-q4"
                       " > build/q4bench_cudaq4.txt 2>&1");

    if (att1_backend_cuda_available()) {
        /* CUDA present: single-tile cuda-q4 must succeed. */
        if (exit_code != 0) {
            if (read_file("build/q4bench_cudaq4.txt", output, sizeof(output)) == 0) {
                fputs(output, stderr);
            }
            fputs("FAIL: cuda-q4 bench should have exited zero with CUDA available\n",
                  stderr);
            return -1;
        }
        if (read_file("build/q4bench_cudaq4.txt", output, sizeof(output)) == 0) {
            if (strstr(output, "backend=cuda-q4") == NULL) {
                fputs("FAIL: cuda-q4 bench output missing 'backend=cuda-q4'\n", stderr);
                return -1;
            }
        }
        fputs("PASS: cuda-q4 backend bench single exits zero\n", stderr);
    } else {
        /* No CUDA: must exit non-zero with a clear error message. */
        if (exit_code == 0) {
            fputs("FAIL: cuda-q4 bench should have exited non-zero without CUDA\n",
                  stderr);
            return -1;
        }
        if (read_file("build/q4bench_cudaq4.txt", output, sizeof(output)) == 0) {
            if (strstr(output, "not supported") == NULL &&
                strstr(output, "unsupported") == NULL &&
                strstr(output, "error") == NULL &&
                strstr(output, "unavailable") == NULL) {
                fputs("FAIL: cuda-q4 bench output has no error indication\n", stderr);
                return -1;
            }
        }
        fputs("PASS: cuda-q4 backend fails clearly as unsupported\n", stderr);
    }
    return 0;
}

/* Test 4: cpu-f32 on dummy model still works (regression guard). */
static int test_q4_bench_f32_path_unchanged(void)
{
    if (run_command("./build/att1-bench"
                    " --model " F32_MODEL
                    " --prompt hello"
                    " --tokens 4"
                    " --mode single"
                    " --backend cpu-f32"
                    " > build/q4bench_f32_reg.txt 2>&1") != 0) {
        fputs("FAIL: f32 bench regression — single mode exited non-zero\n", stderr);
        return -1;
    }

    fputs("PASS: f32 bench path unchanged (regression)\n", stderr);
    return 0;
}

int main(void)
{
    int failures = 0;

    failures += (test_q4_bench_single_exits_zero()   != 0) ? 1 : 0;
    failures += (test_q4_bench_cluster_exits_zero()  != 0) ? 1 : 0;
    failures += (test_q4_bench_cuda_q4_status()      != 0) ? 1 : 0;
    failures += (test_q4_bench_f32_path_unchanged()  != 0) ? 1 : 0;

    return (failures > 0) ? 1 : 0;
}
