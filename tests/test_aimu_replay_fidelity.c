#define _POSIX_C_SOURCE 200112L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define PASS(name) do { printf("PASS: aimu_replay_fidelity: %s\n", (name)); } while (0)
#define FAIL(name) do { printf("FAIL: aimu_replay_fidelity: %s\n", (name)); return 1; } while (0)
#define REQUIRE(cond, name) do { if (!(cond)) { FAIL(name); } } while (0)

static int write_text_file(const char *path, const char *text)
{
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    fputs(text, fp);
    fclose(fp);
    return 0;
}

static int run_tool(const char *plan_path, const char *route_path)
{
    pid_t pid;
    int status = 0;

    pid = fork();
    if (pid < 0) {
        return -1;
    }
    if (pid == 0) {
        if (route_path != NULL) {
            execl("./build/att1-aimu-replay-fidelity",
                  "att1-aimu-replay-fidelity",
                  "--plan",
                  plan_path,
                  "--routes",
                  route_path,
                  (char *)NULL);
        } else {
            execl("./build/att1-aimu-replay-fidelity",
                  "att1-aimu-replay-fidelity",
                  "--plan",
                  plan_path,
                  (char *)NULL);
        }
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

static int test_plan_and_routes_pass(void)
{
    static const char *plan_json =
        "{\n"
        "  \"command_plan_version\": 1,\n"
        "  \"header\": {\n"
        "    \"command_plan_version\": 1,\n"
        "    \"model_id\": \"m169_tiny\",\n"
        "    \"session_id\": \"session_0\",\n"
        "    \"tile_count\": 2,\n"
        "    \"command_count\": 5,\n"
        "    \"status\": \"ok\"\n"
        "  },\n"
        "  \"commands\": [\n"
        "    {\n"
        "      \"command_id\": 1,\n"
        "      \"command_type\": \"LOAD_TENSOR_TILE\",\n"
        "      \"tile_id\": 0,\n"
        "      \"tensor_id\": 1,\n"
        "      \"tensor_name\": \"tiny.weight\",\n"
        "      \"dtype\": \"f32\",\n"
        "      \"packed_bytes\": 64,\n"
        "      \"total_bytes\": 64,\n"
        "      \"fence_id\": 1,\n"
        "      \"expected_status\": \"ATT1_AIMU_ERR_OK\"\n"
        "    },\n"
        "    {\n"
        "      \"command_id\": 2,\n"
        "      \"command_type\": \"VALIDATE_TENSOR\",\n"
        "      \"tile_id\": 0,\n"
        "      \"tensor_id\": 1,\n"
        "      \"tensor_name\": \"tiny.weight\",\n"
        "      \"dtype\": \"f32\",\n"
        "      \"packed_bytes\": 64,\n"
        "      \"total_bytes\": 64,\n"
        "      \"fence_id\": 2,\n"
        "      \"expected_status\": \"ATT1_AIMU_ERR_OK\"\n"
        "    },\n"
        "    {\n"
        "      \"command_id\": 3,\n"
        "      \"command_type\": \"TILE_BARRIER\",\n"
        "      \"tile_id\": 0,\n"
        "      \"fence_id\": 3,\n"
        "      \"expected_status\": \"ATT1_AIMU_ERR_OK\"\n"
        "    },\n"
        "    {\n"
        "      \"command_id\": 4,\n"
        "      \"command_type\": \"QUERY_COUNTERS\",\n"
        "      \"tile_id\": 0,\n"
        "      \"fence_id\": 4,\n"
        "      \"expected_status\": \"ATT1_AIMU_ERR_OK\"\n"
        "    },\n"
        "    {\n"
        "      \"command_id\": 5,\n"
        "      \"command_type\": \"TRACE_SNAPSHOT\",\n"
        "      \"tile_id\": 0,\n"
        "      \"fence_id\": 5,\n"
        "      \"expected_status\": \"ATT1_AIMU_ERR_OK\"\n"
        "    }\n"
        "  ]\n"
        "}\n";
    static const char *routes_json =
        "{\n"
        "  \"header\": {\n"
        "    \"route_report_version\": 1,\n"
        "    \"tile_count\": 2,\n"
        "    \"route_count\": 2\n"
        "  },\n"
        "  \"routes\": [\n"
        "    {\n"
        "      \"route_id\": 1,\n"
        "      \"route_type\": \"ACTIVATION_SEND\",\n"
        "      \"source_tile\": 0,\n"
        "      \"destination_tiles\": [1],\n"
        "      \"payload_bytes\": 32,\n"
        "      \"dependency_fence\": 0,\n"
        "      \"reduction_id\": 0,\n"
        "      \"reduction_behavior\": \"none\"\n"
        "    },\n"
        "    {\n"
        "      \"route_id\": 2,\n"
        "      \"route_type\": \"TILE_BARRIER\",\n"
        "      \"source_tile\": 0,\n"
        "      \"destination_tiles\": [0, 1],\n"
        "      \"payload_bytes\": 0,\n"
        "      \"dependency_fence\": 0,\n"
        "      \"reduction_id\": 0,\n"
        "      \"reduction_behavior\": \"none\"\n"
        "    }\n"
        "  ]\n"
        "}\n";
    const char *plan_path = "build/test_aimu_replay_fidelity_plan.json";
    const char *route_path = "build/test_aimu_replay_fidelity_routes.json";

    REQUIRE(write_text_file(plan_path, plan_json) == 0, "plan_routes_pass: write plan");
    REQUIRE(write_text_file(route_path, routes_json) == 0, "plan_routes_pass: write routes");
    REQUIRE(run_tool(plan_path, route_path) == 0, "plan_routes_pass: tool exit zero");
    unlink(plan_path);
    unlink(route_path);
    PASS("plan_and_routes_pass");
    return 0;
}

static int test_unsupported_command_matches(void)
{
    static const char *plan_json =
        "{\n"
        "  \"command_plan_version\": 1,\n"
        "  \"header\": {\n"
        "    \"command_plan_version\": 1,\n"
        "    \"model_id\": \"m169_unsupported\",\n"
        "    \"session_id\": \"session_0\",\n"
        "    \"tile_count\": 1,\n"
        "    \"command_count\": 1,\n"
        "    \"status\": \"ok\"\n"
        "  },\n"
        "  \"commands\": [\n"
        "    {\n"
        "      \"command_id\": 1,\n"
        "      \"command_type\": \"EXEC_ATTENTION\",\n"
        "      \"tile_id\": 0,\n"
        "      \"dtype\": \"f32\",\n"
        "      \"packed_bytes\": 0,\n"
        "      \"total_bytes\": 0,\n"
        "      \"fence_id\": 1,\n"
        "      \"expected_status\": \"ATT1_AIMU_ERR_UNSUPPORTED_OP\"\n"
        "    }\n"
        "  ]\n"
        "}\n";
    const char *plan_path = "build/test_aimu_replay_fidelity_unsupported.json";

    REQUIRE(write_text_file(plan_path, plan_json) == 0, "unsupported_matches: write plan");
    REQUIRE(run_tool(plan_path, NULL) == 0, "unsupported_matches: tool exit zero");
    unlink(plan_path);
    PASS("unsupported_command_matches");
    return 0;
}

int main(void)
{
    if (test_plan_and_routes_pass() != 0) {
        return 1;
    }
    if (test_unsupported_command_matches() != 0) {
        return 1;
    }
    return 0;
}
