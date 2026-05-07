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
        (strstr(output, "shard_plan=runtime") == NULL) ||
        (strstr(output, "tokens_decoded=") == NULL)) {
        return -1;
    }

    if ((read_file("build/bench_cluster.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "mode=cluster") == NULL) ||
        (strstr(output, "tiles=2") == NULL) ||
        (strstr(output, "shard_plan=runtime") == NULL) ||
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

static int check_converter_report(void)
{
    /* Skip gracefully when Python 3 is not available — converter tooling is
     * Python-only and not required by the default make test path. */
    if (run_command("python3 --version > /dev/null 2>&1") != 0) {
        return 0; /* skip — Python absent */
    }

    /* --- text report --- */
    if (run_command(
            "python3 compiler/convert_llama_to_att1.py"
            " --config compiler/fixtures/tiny_llama_config.json"
            " --tiles 2 --shard-meta --report"
            " > build/converter_report.txt 2>&1") != 0) {
        return -1;
    }

    {
        char rpt[8192];
        if (read_file("build/converter_report.txt", rpt, sizeof(rpt)) != 0) {
            return -1;
        }
        if ((strstr(rpt, "ATT-1 shard metadata plan report") == NULL) ||
            (strstr(rpt, "source_arch     : llama")           == NULL) ||
            (strstr(rpt, "tensor_count    : 21")              == NULL) ||
            (strstr(rpt, "present")                           == NULL) ||
            (strstr(rpt, "tile_count      : 2")               == NULL) ||
            (strstr(rpt, "aimu_count      : 2")               == NULL) ||
            (strstr(rpt, "dtype_f32       : 21")              == NULL) ||
            (strstr(rpt, "tile[0]")                           == NULL) ||
            (strstr(rpt, "tile[1]")                           == NULL) ||
            (strstr(rpt, "layer[0]")                          == NULL) ||
            (strstr(rpt, "layer[1]")                          == NULL) ||
            (strstr(rpt, "status          : ok")              == NULL)) {
            return -1;
        }
    }

    /* --- JSON report --- */
    if (run_command(
            "python3 compiler/convert_llama_to_att1.py"
            " --config compiler/fixtures/tiny_llama_config.json"
            " --tiles 2 --shard-meta --report-json build/converter_report.json"
            " > /dev/null 2>&1") != 0) {
        return -1;
    }

    {
        char rpt[8192];
        if (read_file("build/converter_report.json", rpt, sizeof(rpt)) != 0) {
            return -1;
        }
        if ((strstr(rpt, "\"schema_version\"") == NULL) ||
            (strstr(rpt, "\"source_arch\"")    == NULL) ||
            (strstr(rpt, "\"llama\"")           == NULL) ||
            (strstr(rpt, "\"tensor_count\"")   == NULL) ||
            (strstr(rpt, ": 21")               == NULL) ||
            (strstr(rpt, "\"shard_meta\"")     == NULL) ||
            (strstr(rpt, "\"present\"")        == NULL) ||
            (strstr(rpt, "\"tile_count\"")     == NULL) ||
            (strstr(rpt, "\"tiles\"")           == NULL) ||
            (strstr(rpt, "\"layers\"")          == NULL) ||
            (strstr(rpt, "\"status\"")          == NULL) ||
            (strstr(rpt, "\"ok\"")              == NULL)) {
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

static int check_tensor_reader(void)
{
    /* Skip gracefully when Python 3 is not available. */
    if (run_command("python3 --version > /dev/null 2>&1") != 0) {
        return 0; /* skip — Python absent */
    }

    /* --- token embedding tensor --- */
    if (run_command(
            "python3 compiler/load_safetensors.py"
            " compiler/fixtures/tiny_llama_2l.safetensors"
            " --tensor model.embed_tokens.weight"
            " > build/load_embed.txt 2>&1") != 0) {
        return -1;
    }
    {
        char rpt[4096];
        if (read_file("build/load_embed.txt", rpt, sizeof(rpt)) != 0) {
            return -1;
        }
        if ((strstr(rpt, "load: ok")      == NULL) ||
            (strstr(rpt, "dtype: F32")    == NULL) ||
            (strstr(rpt, "shape: [16,8]") == NULL) ||
            (strstr(rpt, "elements: 128") == NULL)) {
            return -1;
        }
    }

    /* --- attention projection tensor --- */
    if (run_command(
            "python3 compiler/load_safetensors.py"
            " compiler/fixtures/tiny_llama_2l.safetensors"
            " --tensor model.layers.0.self_attn.q_proj.weight"
            " > build/load_qproj.txt 2>&1") != 0) {
        return -1;
    }
    {
        char rpt[4096];
        if (read_file("build/load_qproj.txt", rpt, sizeof(rpt)) != 0) {
            return -1;
        }
        if ((strstr(rpt, "load: ok")     == NULL) ||
            (strstr(rpt, "elements: 64") == NULL)) {
            return -1;
        }
    }

    /* --- FFN tensor --- */
    if (run_command(
            "python3 compiler/load_safetensors.py"
            " compiler/fixtures/tiny_llama_2l.safetensors"
            " --tensor model.layers.0.mlp.gate_proj.weight"
            " > build/load_gate.txt 2>&1") != 0) {
        return -1;
    }
    {
        char rpt[4096];
        if (read_file("build/load_gate.txt", rpt, sizeof(rpt)) != 0) {
            return -1;
        }
        if ((strstr(rpt, "load: ok")      == NULL) ||
            (strstr(rpt, "elements: 128") == NULL)) {
            return -1;
        }
    }

    /* --- final norm tensor --- */
    if (run_command(
            "python3 compiler/load_safetensors.py"
            " compiler/fixtures/tiny_llama_2l.safetensors"
            " --tensor model.norm.weight"
            " > build/load_norm.txt 2>&1") != 0) {
        return -1;
    }
    {
        char rpt[4096];
        if (read_file("build/load_norm.txt", rpt, sizeof(rpt)) != 0) {
            return -1;
        }
        if ((strstr(rpt, "load: ok")    == NULL) ||
            (strstr(rpt, "elements: 8") == NULL)) {
            return -1;
        }
    }

    /* --- check-values: all 21 tensors finite --- */
    if (run_command(
            "python3 compiler/load_safetensors.py"
            " compiler/fixtures/tiny_llama_2l.safetensors"
            " --check-values"
            " > build/load_chkval.txt 2>&1") != 0) {
        return -1;
    }
    {
        char rpt[4096];
        if (read_file("build/load_chkval.txt", rpt, sizeof(rpt)) != 0) {
            return -1;
        }
        if ((strstr(rpt, "values_ok=21")     == NULL) ||
            (strstr(rpt, "check_values: ok") == NULL)) {
            return -1;
        }
    }

    /* --- error-case self-test --- */
    if (run_command(
            "python3 compiler/test_load_safetensors.py"
            " > build/load_selftest.txt 2>&1") != 0) {
        return -1;
    }
    {
        char rpt[4096];
        if (read_file("build/load_selftest.txt", rpt, sizeof(rpt)) != 0) {
            return -1;
        }
        if (strstr(rpt, "self_test: ok") == NULL) {
            return -1;
        }
    }

    return 0;
}

static int check_scanner(void)
{
    /* Skip gracefully when Python 3 is not available. */
    if (run_command("python3 --version > /dev/null 2>&1") != 0) {
        return 0; /* skip — Python absent */
    }

    /* --- basic scan of checked-in tiny fixture --- */
    if (run_command(
            "python3 compiler/scan_safetensors.py"
            " compiler/fixtures/tiny_llama_2l.safetensors"
            " > build/scan_output.txt 2>&1") != 0) {
        return -1;
    }

    {
        char rpt[8192];
        if (read_file("build/scan_output.txt", rpt, sizeof(rpt)) != 0) {
            return -1;
        }
        if ((strstr(rpt, "tensor_count=21")          == NULL) ||
            (strstr(rpt, "scan_errors=0")            == NULL) ||
            (strstr(rpt, "scan: ok")                 == NULL) ||
            (strstr(rpt, "model.embed_tokens.weight") == NULL) ||
            (strstr(rpt, "lm_head.weight")            == NULL) ||
            (strstr(rpt, "F32")                       == NULL)) {
            return -1;
        }
    }

    /* --- LLaMA tensor presence check --- */
    if (run_command(
            "python3 compiler/scan_safetensors.py"
            " compiler/fixtures/tiny_llama_2l.safetensors"
            " --check-llama --n-layers 2"
            " > build/scan_check.txt 2>&1") != 0) {
        return -1;
    }

    {
        char rpt[8192];
        if (read_file("build/scan_check.txt", rpt, sizeof(rpt)) != 0) {
            return -1;
        }
        if ((strstr(rpt, "llama_check: ok") == NULL) ||
            (strstr(rpt, "llama_missing: 0") == NULL)) {
            return -1;
        }
    }

    return 0;
}

static int check_tokenizer_scanner(void)
{
    /* Skip gracefully when Python 3 is not available. */
    if (run_command("python3 --version > /dev/null 2>&1") != 0) {
        return 0; /* skip — Python absent */
    }

    /* --- basic scan of tiny tokenizer fixture --- */
    if (run_command(
            "python3 compiler/scan_tokenizer.py"
            " compiler/fixtures/tiny_tokenizer"
            " > build/tok_scan.txt 2>&1") != 0) {
        return -1;
    }
    {
        char rpt[4096];
        if (read_file("build/tok_scan.txt", rpt, sizeof(rpt)) != 0) {
            return -1;
        }
        if ((strstr(rpt, "tokenizer_type=bpe_json") == NULL) ||
            (strstr(rpt, "vocab_size=16")           == NULL) ||
            (strstr(rpt, "bos_id=1")                == NULL) ||
            (strstr(rpt, "eos_id=2")                == NULL) ||
            (strstr(rpt, "scan: ok")                == NULL)) {
            return -1;
        }
    }

    /* --- cross-check vocab_size against model config --- */
    if (run_command(
            "python3 compiler/scan_tokenizer.py"
            " compiler/fixtures/tiny_tokenizer"
            " --config compiler/fixtures/tiny_llama/config.json"
            " > build/tok_config_check.txt 2>&1") != 0) {
        return -1;
    }
    {
        char rpt[4096];
        if (read_file("build/tok_config_check.txt", rpt, sizeof(rpt)) != 0) {
            return -1;
        }
        if ((strstr(rpt, "vocab_size_match=yes") == NULL) ||
            (strstr(rpt, "scan: ok")             == NULL)) {
            return -1;
        }
    }

    return 0;
}

static int check_tokenizer_import_report(void)
{
    char rpt[8192];

    /* Skip gracefully when Python 3 is not available. */
    if (run_command("python3 --version > /dev/null 2>&1") != 0) {
        return 0; /* skip — Python absent */
    }

    /* --- human-readable import report --- */
    if (run_command(
            "python3 compiler/scan_tokenizer.py"
            " compiler/fixtures/tiny_tokenizer"
            " --report"
            " --model-config compiler/fixtures/tiny_llama/config.json"
            " > build/tok_import_report.txt 2>&1") != 0) {
        return -1;
    }
    if (read_file("build/tok_import_report.txt", rpt, sizeof(rpt)) != 0) {
        return -1;
    }
    if ((strstr(rpt, "tokenizer_type=bpe_json")    == NULL) ||
        (strstr(rpt, "vocab_size=16")               == NULL) ||
        (strstr(rpt, "bos_id=1")                    == NULL) ||
        (strstr(rpt, "eos_id=2")                    == NULL) ||
        (strstr(rpt, "vocab_size_match=yes")         == NULL) ||
        (strstr(rpt, "import_ready=yes")             == NULL) ||
        (strstr(rpt, "canonical_hash=")              == NULL) ||
        (strstr(rpt, "unsupported_fields=none")      == NULL) ||
        (strstr(rpt, "report: ok")                   == NULL)) {
        return -1;
    }

    /* --- JSON import report written to file --- */
    if (run_command(
            "python3 compiler/scan_tokenizer.py"
            " compiler/fixtures/tiny_tokenizer"
            " --report-json build/tok_import_report.json"
            " --model-config compiler/fixtures/tiny_llama/config.json"
            " > /dev/null 2>&1") != 0) {
        return -1;
    }
    if (read_file("build/tok_import_report.json", rpt, sizeof(rpt)) != 0) {
        return -1;
    }
    if ((strstr(rpt, "\"tokenizer_type\"")   == NULL) ||
        (strstr(rpt, "\"bpe_json\"")         == NULL) ||
        (strstr(rpt, "\"vocab_size\"")       == NULL) ||
        (strstr(rpt, "\"import_ready\"")     == NULL) ||
        (strstr(rpt, "\"canonical_hash\"")   == NULL) ||
        (strstr(rpt, "\"unsupported_fields\"") == NULL)) {
        return -1;
    }

    /* --- mismatch: tokenizer vocab_size=16, config says 32 --- */
    if (run_command(
            "printf '{\"vocab_size\":32}'"
            " > build/tok_mismatch_config.json") != 0) {
        return -1;
    }
    /* exit code 2 expected on vocab_size mismatch; ignore it */
    run_command(
        "python3 compiler/scan_tokenizer.py"
        " compiler/fixtures/tiny_tokenizer"
        " --report"
        " --model-config build/tok_mismatch_config.json"
        " > build/tok_mismatch.txt 2>&1");
    if (read_file("build/tok_mismatch.txt", rpt, sizeof(rpt)) != 0) {
        return -1;
    }
    if ((strstr(rpt, "vocab_size_match=no") == NULL) ||
        (strstr(rpt, "import_ready=no")     == NULL)) {
        return -1;
    }

    return 0;
}

int main(void)
{
    if ((check_bench_tools()              != 0) ||
        (check_size_tools()               != 0) ||
        (check_converter_report()         != 0) ||
        (check_scanner()                  != 0) ||
        (check_tensor_reader()            != 0) ||
        (check_tokenizer_scanner()        != 0) ||
        (check_tokenizer_import_report()  != 0)) {
        fputs("bench smoke test failed\n", stderr);
        return 1;
    }

    puts("bench smoke test passed");
    return 0;
}
