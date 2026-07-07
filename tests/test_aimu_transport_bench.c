#define _POSIX_C_SOURCE 200112L

/*
 * test_aimu_transport_bench.c  —  smoke tests for M170 transport
 * characterization. The tool itself spawns the M162 endpoint daemon and
 * measures the socket-backed conformance path.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define PASS(name) do { printf("PASS: aimu_transport_bench: %s\n", (name)); } while (0)
#define FAIL(name) do { printf("FAIL: aimu_transport_bench: %s\n", (name)); return 1; } while (0)
#define REQUIRE(cond, name) do { if (!(cond)) { FAIL(name); } } while (0)

static int file_contains(const char *path, const char *needle)
{
    FILE *fp = fopen(path, "r");
    char buf[4096];
    size_t n;
    int found = 0;

    if (fp == NULL) {
        return 0;
    }
    n = fread(buf, 1u, sizeof(buf) - 1u, fp);
    buf[n] = '\0';
    if (strstr(buf, needle) != NULL) {
        found = 1;
    }
    fclose(fp);
    return found;
}

static int run_tool(const char *json_path)
{
    pid_t pid;
    int status = 0;

    pid = fork();
    if (pid < 0) {
        return -1;
    }
    if (pid == 0) {
        execl("./build/att1-aimu-transport-bench",
              "att1-aimu-transport-bench",
              "--iterations",
              "4",
              "--payload-bytes",
              "256",
              "--report-json",
              json_path,
              (char *)NULL);
        _exit(127);
    }
    if (waitpid(pid, &status, 0) < 0) {
        return -1;
    }
    if (!WIFEXITED(status)) {
        return -1;
    }
    return WEXITSTATUS(status);
}

static int test_transport_bench_json(void)
{
    const char *json_path = "build/test_aimu_transport_bench.json";

    unlink(json_path);
    REQUIRE(run_tool(json_path) == 0, "json: tool exits zero");
    REQUIRE(file_contains(json_path, "\"transport_bench_report_version\": 1"),
            "json: report version present");
    REQUIRE(file_contains(json_path, "\"m118_calibration\""),
            "json: calibration block present");
    REQUIRE(file_contains(json_path, "\"fabric_send_receive\""),
            "json: fabric sample present");
    unlink(json_path);
    PASS("transport_bench_json");
    return 0;
}

int main(void)
{
    if (test_transport_bench_json() != 0) {
        return 1;
    }
    return 0;
}
