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
        (strstr(output, "mode=single")       == NULL) ||
        (strstr(output, "backend=cpu-f32")   == NULL) ||
        (strstr(output, "shard_plan=runtime") == NULL) ||
        (strstr(output, "tokenizer=byte")     == NULL) ||
        (strstr(output, "tokens_decoded=")   == NULL)) {
        return -1;
    }

    if ((read_file("build/bench_cluster.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "mode=cluster")       == NULL) ||
        (strstr(output, "tiles=2")            == NULL) ||
        (strstr(output, "shard_plan=runtime") == NULL) ||
        (strstr(output, "tokenizer=byte")     == NULL) ||
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

static int check_tokenizer_selection(void)
{
    char output[4096];

    /* --- default mode is byte (no --tokenizer flag) --- */
    if (run_command("./build/att1-bench --model models/dummy/model.att1 "
                    "--prompt hello --tokens 4 --mode single "
                    "> build/tok_sel_default.txt 2>&1") != 0) {
        fputs("tok_sel: default run failed\n", stderr);
        return -1;
    }
    if ((read_file("build/tok_sel_default.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "tokenizer=byte") == NULL)) {
        fputs("tok_sel: default tokenizer=byte missing from output\n", stderr);
        return -1;
    }

    /* --- explicit --tokenizer byte works --- */
    if (run_command("./build/att1-bench --model models/dummy/model.att1 "
                    "--prompt hello --tokens 4 --mode single "
                    "--tokenizer byte "
                    "> build/tok_sel_byte.txt 2>&1") != 0) {
        fputs("tok_sel: explicit byte run failed\n", stderr);
        return -1;
    }
    if ((read_file("build/tok_sel_byte.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "tokenizer=byte") == NULL)) {
        fputs("tok_sel: explicit byte tokenizer=byte missing from output\n", stderr);
        return -1;
    }

    /* --- --tokenizer metadata with tok_meta absent fails clearly --- */
    /* models/dummy/model.att1 is a v1 model with no tok_meta section */
    if (run_command("./build/att1-bench --model models/dummy/model.att1 "
                    "--prompt hello --tokens 4 --mode single "
                    "--tokenizer metadata "
                    "> build/tok_sel_meta_absent.txt 2>&1") == 0) {
        fputs("tok_sel: metadata absent should have failed\n", stderr);
        return -1;
    }
    if ((read_file("build/tok_sel_meta_absent.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "tokenizer metadata absent") == NULL)) {
        fputs("tok_sel: expected \"tokenizer metadata absent\" message\n", stderr);
        return -1;
    }

    /* --- M57: --tokenizer metadata on v2 fixture passes check_runtime
     *   then fails with "not implemented yet" --- */
    if (run_command("./build/att1-bench --model models/tok_meta/model.att1 "
                    "--prompt hello --tokens 4 --mode single "
                    "--tokenizer metadata "
                    "> build/tok_sel_meta_notimpl.txt 2>&1") == 0) {
        fputs("tok_sel: metadata not-impl should have failed\n", stderr);
        return -1;
    }
    if ((read_file("build/tok_sel_meta_notimpl.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "not implemented yet") == NULL)) {
        fputs("tok_sel: expected \"not implemented yet\" message\n", stderr);
        return -1;
    }

    /* --- M58: --tokenizer external without ID source fails with clear message
     *   (external mode is now wired; this tests the missing-source error path)
     *   The full positive-path tests live in check_external_tokenizer(). --- */
    if (run_command("./build/att1-bench --model models/dummy/model.att1 "
                    "--tokens 4 --mode single "
                    "--tokenizer external "
                    "> build/tok_sel_external.txt 2>&1") == 0) {
        fputs("tok_sel: external without IDs should have failed\n", stderr);
        return -1;
    }
    if ((read_file("build/tok_sel_external.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "--input-token-ids or --tokens-file") == NULL)) {
        fputs("tok_sel: expected missing-source error for external\n", stderr);
        return -1;
    }

    /* --- invalid tokenizer mode fails clearly (exit nonzero) --- */
    if (run_command("./build/att1-bench --model models/dummy/model.att1 "
                    "--prompt hello --tokens 4 --mode single "
                    "--tokenizer bogus "
                    "> /dev/null 2>&1") == 0) {
        fputs("tok_sel: invalid mode should have failed\n", stderr);
        return -1;
    }

    return 0;
}

static int check_external_tokenizer(void)
{
    char output[4096];

    /* ── helper: write a token IDs file ─────────────────────────────── */
    {
        FILE *fp = fopen("build/ext_ids.txt", "w");
        if (fp == NULL) {
            fputs("ext_tok: could not create token IDs file\n", stderr);
            return -1;
        }
        fputs("1\n2\n3\n", fp);
        fclose(fp);
    }

    /* --- external single: --input-token-ids accepted; tokenizer=external --- */
    if (run_command("./build/att1-bench --model models/dummy/model.att1 "
                    "--tokens 1 --mode single "
                    "--tokenizer external --input-token-ids \"1,2,3\" "
                    "> build/ext_single.txt 2>&1") != 0) {
        fputs("ext_tok: input-token-ids single failed\n", stderr);
        return -1;
    }
    if ((read_file("build/ext_single.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "tokenizer=external") == NULL)) {
        fputs("ext_tok: tokenizer=external missing from single output\n",
              stderr);
        return -1;
    }
    if (strstr(output, "last_token=") == NULL) {
        fputs("ext_tok: last_token missing from single output\n", stderr);
        return -1;
    }

    /* --- external cluster: --input-token-ids accepted --- */
    if (run_command("./build/att1-bench --model models/dummy/model.att1 "
                    "--tokens 1 --mode cluster --tiles 2 "
                    "--tokenizer external --input-token-ids \"1,2,3\" "
                    "> build/ext_cluster.txt 2>&1") != 0) {
        fputs("ext_tok: input-token-ids cluster failed\n", stderr);
        return -1;
    }
    if ((read_file("build/ext_cluster.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "tokenizer=external") == NULL)) {
        fputs("ext_tok: tokenizer=external missing from cluster output\n",
              stderr);
        return -1;
    }

    /* --- external single: --tokens-file accepted --- */
    if (run_command("./build/att1-bench --model models/dummy/model.att1 "
                    "--tokens 1 --mode single "
                    "--tokenizer external --tokens-file build/ext_ids.txt "
                    "> build/ext_file.txt 2>&1") != 0) {
        fputs("ext_tok: tokens-file single failed\n", stderr);
        return -1;
    }
    if ((read_file("build/ext_file.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "tokenizer=external") == NULL)) {
        fputs("ext_tok: tokenizer=external missing from tokens-file output\n",
              stderr);
        return -1;
    }

    /* --- out-of-range token ID fails clearly --- */
    /* dummy model vocab_size=256; ID 256 is out of range */
    if (run_command("./build/att1-bench --model models/dummy/model.att1 "
                    "--tokens 1 --mode single "
                    "--tokenizer external --input-token-ids \"1,256,3\" "
                    "> build/ext_oor.txt 2>&1") == 0) {
        fputs("ext_tok: out-of-range should have failed\n", stderr);
        return -1;
    }
    if ((read_file("build/ext_oor.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "out of range") == NULL)) {
        fputs("ext_tok: expected \"out of range\" in error output\n", stderr);
        return -1;
    }

    /* --- malformed token ID fails clearly --- */
    if (run_command("./build/att1-bench --model models/dummy/model.att1 "
                    "--tokens 1 --mode single "
                    "--tokenizer external --input-token-ids \"1,foo,3\" "
                    "> build/ext_bad.txt 2>&1") == 0) {
        fputs("ext_tok: malformed should have failed\n", stderr);
        return -1;
    }
    if ((read_file("build/ext_bad.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "malformed") == NULL)) {
        fputs("ext_tok: expected \"malformed\" in error output\n", stderr);
        return -1;
    }

    /* --- empty --input-token-ids fails clearly --- */
    if (run_command("./build/att1-bench --model models/dummy/model.att1 "
                    "--tokens 1 --mode single "
                    "--tokenizer external --input-token-ids \"\" "
                    "> build/ext_empty.txt 2>&1") == 0) {
        fputs("ext_tok: empty IDs should have failed\n", stderr);
        return -1;
    }

    /* --- external without ID source fails clearly --- */
    if (run_command("./build/att1-bench --model models/dummy/model.att1 "
                    "--tokens 1 --mode single "
                    "--tokenizer external "
                    "> build/ext_nosrc.txt 2>&1") == 0) {
        fputs("ext_tok: missing ID source should have failed\n", stderr);
        return -1;
    }
    if ((read_file("build/ext_nosrc.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "--input-token-ids or --tokens-file") == NULL)) {
        fputs("ext_tok: expected missing-source error message\n", stderr);
        return -1;
    }

    /* --- byte mode unchanged by M58 --- */
    if (run_command("./build/att1-bench --model models/dummy/model.att1 "
                    "--prompt hello --tokens 2 --mode single "
                    "--tokenizer byte "
                    "> build/ext_byte_check.txt 2>&1") != 0) {
        fputs("ext_tok: byte mode broken\n", stderr);
        return -1;
    }
    if ((read_file("build/ext_byte_check.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "tokenizer=byte") == NULL)) {
        fputs("ext_tok: byte mode tokenizer=byte missing\n", stderr);
        return -1;
    }

    /* --- metadata mode still validation-only / not-impl --- */
    if (run_command("./build/att1-bench --model models/tok_meta/model.att1 "
                    "--prompt hello --tokens 1 --mode single "
                    "--tokenizer metadata "
                    "> build/ext_meta_check.txt 2>&1") == 0) {
        fputs("ext_tok: metadata should still fail (not-impl)\n", stderr);
        return -1;
    }
    if ((read_file("build/ext_meta_check.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "not implemented yet") == NULL)) {
        fputs("ext_tok: metadata not-impl message missing\n", stderr);
        return -1;
    }

    return 0;
}

static int check_hf_tokenizer(void)
{
    char output[4096];
    int  has_tokenizers;
    int  has_transformers;

    /* Skip gracefully when Python 3 is not available. */
    if (run_command("python3 --version > /dev/null 2>&1") != 0) {
        return 0; /* skip — Python absent */
    }

    /* --- missing tokenizer directory fails clearly (always runs) ---
     * Path validation happens before any package import, so this test
     * exercises the early-exit path regardless of package availability. */
    if (run_command("python3 compiler/tokenize_hf.py "
                    "--tokenizer /nonexistent/tok/dir "
                    "--text hello "
                    "> build/hf_notfound.txt 2>&1") == 0) {
        fputs("hf_tok: missing path should have failed\n", stderr);
        return -1;
    }
    if ((read_file("build/hf_notfound.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "not found") == NULL)) {
        fputs("hf_tok: expected \"not found\" in missing-path error\n", stderr);
        return -1;
    }

    /* --- skip positive-path tests when HF packages absent --- */
    has_tokenizers   = (run_command(
        "python3 -c 'import tokenizers' > /dev/null 2>&1") == 0);
    has_transformers = (run_command(
        "python3 -c 'import transformers' > /dev/null 2>&1") == 0);
    if (!has_tokenizers && !has_transformers) {
        return 0; /* skip — neither tokenizers nor transformers installed */
    }

    /* ---- positive-path tests (require tokenizers or transformers) ---- */

    /* --- basic tokenize: CSV token IDs printed to stdout --- */
    if (run_command("python3 compiler/tokenize_hf.py "
                    "--tokenizer compiler/fixtures/tiny_tokenizer "
                    "--text hello "
                    "--add-special-tokens false "
                    "> build/hf_basic.txt 2>&1") != 0) {
        fputs("hf_tok: basic tokenize failed\n", stderr);
        return -1;
    }
    if (read_file("build/hf_basic.txt", output, sizeof(output)) != 0) {
        fputs("hf_tok: could not read basic output\n", stderr);
        return -1;
    }
    /* Output must contain at least one digit (a token ID). */
    {
        int has_digit = 0;
        int i;
        for (i = 0; output[i] != '\0'; i++) {
            if ((output[i] >= '0') && (output[i] <= '9')) {
                has_digit = 1;
                break;
            }
        }
        if (!has_digit) {
            fputs("hf_tok: basic output has no token IDs\n", stderr);
            return -1;
        }
    }

    /* --- JSON output contains expected fields --- */
    if (run_command("python3 compiler/tokenize_hf.py "
                    "--tokenizer compiler/fixtures/tiny_tokenizer "
                    "--text hello "
                    "--add-special-tokens false "
                    "--json-out build/hf_basic.json "
                    "> /dev/null 2>&1") != 0) {
        fputs("hf_tok: --json-out failed\n", stderr);
        return -1;
    }
    if (read_file("build/hf_basic.json", output, sizeof(output)) != 0) {
        fputs("hf_tok: could not read JSON output\n", stderr);
        return -1;
    }
    if ((strstr(output, "\"token_ids\"")         == NULL) ||
        (strstr(output, "\"token_count\"")        == NULL) ||
        (strstr(output, "\"tokenizer_path\"")     == NULL) ||
        (strstr(output, "\"add_special_tokens\"") == NULL)) {
        fputs("hf_tok: JSON output missing expected fields\n", stderr);
        return -1;
    }

    /* --- --out writes one-ID-per-line file for --tokens-file --- */
    if (run_command("python3 compiler/tokenize_hf.py "
                    "--tokenizer compiler/fixtures/tiny_tokenizer "
                    "--text hello "
                    "--add-special-tokens false "
                    "--out build/hf_ids.txt "
                    "> /dev/null 2>&1") != 0) {
        fputs("hf_tok: --out failed\n", stderr);
        return -1;
    }
    if (read_file("build/hf_ids.txt", output, sizeof(output)) != 0) {
        fputs("hf_tok: could not read IDs file\n", stderr);
        return -1;
    }
    {
        int has_digit = 0;
        int i;
        for (i = 0; output[i] != '\0'; i++) {
            if ((output[i] >= '0') && (output[i] <= '9')) {
                has_digit = 1;
                break;
            }
        }
        if (!has_digit) {
            fputs("hf_tok: IDs file has no token IDs\n", stderr);
            return -1;
        }
    }

    /* --- pipeline: tokenize → --tokens-file → att1-bench --tokenizer external --- */
    if (run_command("python3 compiler/tokenize_hf.py "
                    "--tokenizer compiler/fixtures/tiny_tokenizer "
                    "--text hello "
                    "--add-special-tokens false "
                    "--out build/hf_pipe_ids.txt "
                    "> /dev/null 2>&1") != 0) {
        fputs("hf_tok: pipeline tokenize step failed\n", stderr);
        return -1;
    }
    if (run_command("./build/att1-bench "
                    "--model models/dummy/model.att1 "
                    "--tokens 1 --mode single "
                    "--tokenizer external "
                    "--tokens-file build/hf_pipe_ids.txt "
                    "> build/hf_pipe_bench.txt 2>&1") != 0) {
        fputs("hf_tok: att1-bench with tokenized IDs failed\n", stderr);
        return -1;
    }
    if ((read_file("build/hf_pipe_bench.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "tokenizer=external") == NULL)) {
        fputs("hf_tok: tokenizer=external missing in bench output\n", stderr);
        return -1;
    }

    /* --- --timing flag does not cause an error --- */
    if (run_command("python3 compiler/tokenize_hf.py "
                    "--tokenizer compiler/fixtures/tiny_tokenizer "
                    "--text hi "
                    "--add-special-tokens false "
                    "--timing "
                    "> /dev/null 2>&1") != 0) {
        fputs("hf_tok: --timing flag failed\n", stderr);
        return -1;
    }

    return 0;
}

/*
 * check_pretokenized_pipeline() — M60
 *
 * Python-skippable.  When tokenizers or transformers is available, validates
 * the full M59+M60 pipeline: tokenize_hf.py writes token IDs → att1-bench
 * accepts them via --tokenizer external on real_tiny_f32 / real_tiny_q8.
 *
 * This exercises the end-to-end path; tokenizer output is not semantically
 * meaningful because the tiny tokenizer fixture is synthetic.
 */
static int check_pretokenized_pipeline(void)
{
    char output[4096];
    int  has_tokenizers;
    int  has_transformers;

    /* Skip gracefully when Python 3 is not available. */
    if (run_command("python3 --version > /dev/null 2>&1") != 0) {
        return 0; /* skip — Python absent */
    }

    has_tokenizers   = (run_command(
        "python3 -c 'import tokenizers' > /dev/null 2>&1") == 0);
    has_transformers = (run_command(
        "python3 -c 'import transformers' > /dev/null 2>&1") == 0);
    if (!has_tokenizers && !has_transformers) {
        return 0; /* skip — neither tokenizers nor transformers installed */
    }

    /* Tokenize "abc" with the tiny fixture, skip BOS/EOS. */
    if (run_command("python3 compiler/tokenize_hf.py "
                    "--tokenizer compiler/fixtures/tiny_tokenizer "
                    "--text abc "
                    "--add-special-tokens false "
                    "--out build/m60_hf_ids.txt "
                    "> /dev/null 2>&1") != 0) {
        fputs("m60_pipe: tokenize_hf failed\n", stderr);
        return -1;
    }

    /* --- real_tiny_f32 single via --tokens-file --- */
    if (run_command("./build/att1-bench "
                    "--model models/real_tiny_f32/model.att1 "
                    "--tokens 2 --mode single --backend cpu-f32 "
                    "--tokenizer external "
                    "--tokens-file build/m60_hf_ids.txt "
                    "> build/m60_hf_f32_single.txt 2>&1") != 0) {
        fputs("m60_pipe: f32 single failed\n", stderr);
        return -1;
    }
    if ((read_file("build/m60_hf_f32_single.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "tokenizer=external") == NULL) ||
        (strstr(output, "mode=single")        == NULL) ||
        (strstr(output, "backend=cpu-f32")    == NULL)) {
        fputs("m60_pipe: f32 single output check failed\n", stderr);
        return -1;
    }

    /* --- real_tiny_f32 cluster via --tokens-file --- */
    if (run_command("./build/att1-bench "
                    "--model models/real_tiny_f32/model.att1 "
                    "--tokens 2 --mode cluster --tiles 2 --backend cpu-f32 "
                    "--tokenizer external "
                    "--tokens-file build/m60_hf_ids.txt "
                    "> build/m60_hf_f32_cluster.txt 2>&1") != 0) {
        fputs("m60_pipe: f32 cluster failed\n", stderr);
        return -1;
    }
    if ((read_file("build/m60_hf_f32_cluster.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "tokenizer=external") == NULL) ||
        (strstr(output, "mode=cluster")       == NULL)) {
        fputs("m60_pipe: f32 cluster output check failed\n", stderr);
        return -1;
    }

    /* --- real_tiny_q8 single via --tokens-file --- */
    if (run_command("./build/att1-bench "
                    "--model models/real_tiny_q8/model.att1 "
                    "--tokens 2 --mode single --backend cpu-q8 "
                    "--tokenizer external "
                    "--tokens-file build/m60_hf_ids.txt "
                    "> build/m60_hf_q8_single.txt 2>&1") != 0) {
        fputs("m60_pipe: q8 single failed\n", stderr);
        return -1;
    }
    if ((read_file("build/m60_hf_q8_single.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "tokenizer=external") == NULL) ||
        (strstr(output, "backend=cpu-q8")     == NULL)) {
        fputs("m60_pipe: q8 single output check failed\n", stderr);
        return -1;
    }

    /* --- real_tiny_q8 cluster via --tokens-file --- */
    if (run_command("./build/att1-bench "
                    "--model models/real_tiny_q8/model.att1 "
                    "--tokens 2 --mode cluster --tiles 2 --backend cpu-q8 "
                    "--tokenizer external "
                    "--tokens-file build/m60_hf_ids.txt "
                    "> build/m60_hf_q8_cluster.txt 2>&1") != 0) {
        fputs("m60_pipe: q8 cluster failed\n", stderr);
        return -1;
    }
    if ((read_file("build/m60_hf_q8_cluster.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "tokenizer=external") == NULL) ||
        (strstr(output, "mode=cluster")       == NULL) ||
        (strstr(output, "backend=cpu-q8")     == NULL)) {
        fputs("m60_pipe: q8 cluster output check failed\n", stderr);
        return -1;
    }

    return 0;
}

/*
 * check_source_comparison() — M61
 *
 * Python-skippable.  Requires numpy.  Tests the M62 report harness against
 * the m61 f32/q8 fixtures:
 *   1. Default mode (key=value): checks result/forward_match lines.
 *   2. --report mode: checks report: ok line.
 *   3. --report-json: checks JSON file contains expected keys.
 *   4. Bad --report-json path: checks that the harness exits non-zero.
 */

/*
 * command_fails() — runs a command and returns 0 iff the command exits
 * non-zero.  Inverse of run_command(); used for negative tests.
 */
static int command_fails(const char *command)
{
    const int rc = system(command);

    return (rc != 0) ? 0 : -1;
}

static int check_source_comparison(void)
{
    char output[16384];

    /* Skip gracefully when Python 3 is not available. */
    if (run_command("python3 --version > /dev/null 2>&1") != 0) {
        return 0; /* skip — Python absent */
    }

    /* numpy is required for the forward-pass comparison. */
    if (run_command("python3 -c 'import numpy' > /dev/null 2>&1") != 0) {
        return 0; /* skip — numpy not installed */
    }

    /* 1. Default (key=value) mode: harness must exit 0 and report pass. */
    if (run_command("python3 compiler/compare_att1_to_source.py "
                    "--report-json build/m61_comparison.json "
                    "> build/m61_comparison.txt 2>&1") != 0) {
        fputs("m61_compare: harness exited with non-zero status\n", stderr);
        return -1;
    }

    if (read_file("build/m61_comparison.txt", output, sizeof(output)) != 0) {
        fputs("m61_compare: cannot read harness output\n", stderr);
        return -1;
    }

    /* Check overall result. */
    if (strstr(output, "result:") != NULL) {
        if (strstr(output, "result:              pass") == NULL) {
            fputs("m61_compare: result is not pass\n", stderr);
            return -1;
        }
    }

    /* Check forward-pass match. */
    if (strstr(output, "forward_match:") != NULL) {
        if (strstr(output, "forward_match:       yes") == NULL) {
            fputs("m61_compare: forward_match is not yes\n", stderr);
            return -1;
        }
    }

    /* 2. --report mode: must produce report: ok */
    if (run_command("python3 compiler/compare_att1_to_source.py "
                    "--report "
                    "> build/m61_report.txt 2>&1") != 0) {
        fputs("m62_report: --report run failed\n", stderr);
        return -1;
    }

    if (read_file("build/m61_report.txt", output, sizeof(output)) != 0) {
        fputs("m62_report: cannot read --report output\n", stderr);
        return -1;
    }

    if (strstr(output, "report:              ok") == NULL) {
        fputs("m62_report: report: ok not found\n", stderr);
        return -1;
    }

    if (strstr(output, "result:              pass") == NULL) {
        fputs("m62_report: result: pass not found in --report output\n", stderr);
        return -1;
    }

    /* 3. --report-json: JSON file must contain expected top-level keys. */
    if (run_command("python3 compiler/compare_att1_to_source.py "
                    "--report-json build/m61_rpt.json "
                    "> /dev/null 2>&1") != 0) {
        fputs("m62_report: --report-json run failed\n", stderr);
        return -1;
    }

    if (read_file("build/m61_rpt.json", output, sizeof(output)) != 0) {
        fputs("m62_report: cannot read JSON report\n", stderr);
        return -1;
    }

    if ((strstr(output, "\"result\"")     == NULL) ||
        (strstr(output, "\"f32_static\"") == NULL) ||
        (strstr(output, "\"q8_static\"")  == NULL) ||
        (strstr(output, "\"forward\"")    == NULL) ||
        (strstr(output, "\"config\"")     == NULL)) {
        fputs("m62_report: JSON missing expected keys\n", stderr);
        return -1;
    }

    /* 4. M69 local model-dir path: public-model workflow uses local source
     *    directories and explicitly supplied external artifacts. */
    if (run_command("python3 compiler/compare_att1_to_source.py "
                    "--model-dir compiler/fixtures/tiny_llama "
                    "--safetensors compiler/fixtures/tiny_llama_2l.safetensors "
                    "--att1-f32 models/real_tiny_f32/model.att1 "
                    "--att1-q8 models/real_tiny_q8/model.att1 "
                    "--prompt-ids 1,2,3 "
                    "--report "
                    "> build/m69_public_model_report.txt 2>&1") != 0) {
        fputs("m69_report: local model-dir report run failed\n", stderr);
        return -1;
    }

    if (read_file("build/m69_public_model_report.txt", output, sizeof(output)) != 0) {
        fputs("m69_report: cannot read local model-dir report\n", stderr);
        return -1;
    }

    if ((strstr(output, "source_model_path:") == NULL) ||
        (strstr(output, "reference:           source_safetensors") == NULL) ||
        (strstr(output, "next_token_result:   pass") == NULL) ||
        (strstr(output, "q8_backend:          cpu-q8") == NULL) ||
        (strstr(output, "result:              pass") == NULL)) {
        fputs("m69_report: report missing public-model fields\n", stderr);
        return -1;
    }

    /* 5. Bad --report-json path: harness must exit non-zero. */
    if (command_fails(
            "python3 compiler/compare_att1_to_source.py "
            "--report-json /nonexistent/path/m62.json "
            "> /dev/null 2>&1") != 0) {
        fputs("m62_report: bad report-json path did not fail\n", stderr);
        return -1;
    }

    return 0;
}

/*
 * check_m63_validation() — M63
 *
 * Python-skippable.  Requires numpy.  Validates the M63 larger tiny fixture
 * (vocab=64, d_model=32, n_heads=4, d_ff=64, n_layers=2) through:
 *   1. att1-bench cpu-f32 single and cluster modes
 *   2. att1-bench cpu-q8 single and cluster modes
 *   3. source comparison harness (--report and --report-json)
 */
static int check_m63_validation(void)
{
    char output[16384];

    /* Skip gracefully when Python 3 is not available. */
    if (run_command("python3 --version > /dev/null 2>&1") != 0) {
        return 0; /* skip — Python absent */
    }
    if (run_command("python3 -c 'import numpy' > /dev/null 2>&1") != 0) {
        return 0; /* skip — numpy not installed */
    }

    /* Write token IDs file for external tokenizer */
    {
        FILE *fp = fopen("build/m63_ids.txt", "w");
        if (fp == NULL) {
            fputs("m63: could not create token IDs file\n", stderr);
            return -1;
        }
        fputs("5\n20\n40\n", fp);
        fclose(fp);
    }

    /* 1. f32 single */
    if (run_command("./build/att1-bench "
                    "--model models/m63_f32/model.att1 "
                    "--tokenizer external --tokens-file build/m63_ids.txt "
                    "--tokens 1 --mode single --backend cpu-f32 "
                    "> build/m63_f32_single.txt 2>&1") != 0) {
        fputs("m63: f32 single failed\n", stderr);
        return -1;
    }
    if ((read_file("build/m63_f32_single.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "mode=single")     == NULL) ||
        (strstr(output, "backend=cpu-f32") == NULL) ||
        (strstr(output, "last_token=")     == NULL)) {
        fputs("m63: f32 single output check failed\n", stderr);
        return -1;
    }

    /* 2. f32 cluster */
    if (run_command("./build/att1-bench "
                    "--model models/m63_f32/model.att1 "
                    "--tokenizer external --tokens-file build/m63_ids.txt "
                    "--tokens 1 --mode cluster --tiles 2 --backend cpu-f32 "
                    "> build/m63_f32_cluster.txt 2>&1") != 0) {
        fputs("m63: f32 cluster failed\n", stderr);
        return -1;
    }
    if ((read_file("build/m63_f32_cluster.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "mode=cluster")         == NULL) ||
        (strstr(output, "fabric_packets_sent=") == NULL)) {
        fputs("m63: f32 cluster output check failed\n", stderr);
        return -1;
    }

    /* 3. q8 single */
    if (run_command("./build/att1-bench "
                    "--model models/m63_q8/model.att1 "
                    "--tokenizer external --tokens-file build/m63_ids.txt "
                    "--tokens 1 --mode single --backend cpu-q8 "
                    "> build/m63_q8_single.txt 2>&1") != 0) {
        fputs("m63: q8 single failed\n", stderr);
        return -1;
    }
    if ((read_file("build/m63_q8_single.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "mode=single")    == NULL) ||
        (strstr(output, "backend=cpu-q8") == NULL) ||
        (strstr(output, "last_token=")    == NULL)) {
        fputs("m63: q8 single output check failed\n", stderr);
        return -1;
    }

    /* 4. q8 cluster */
    if (run_command("./build/att1-bench "
                    "--model models/m63_q8/model.att1 "
                    "--tokenizer external --tokens-file build/m63_ids.txt "
                    "--tokens 1 --mode cluster --tiles 2 --backend cpu-q8 "
                    "> build/m63_q8_cluster.txt 2>&1") != 0) {
        fputs("m63: q8 cluster failed\n", stderr);
        return -1;
    }
    if ((read_file("build/m63_q8_cluster.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "mode=cluster")         == NULL) ||
        (strstr(output, "backend=cpu-q8")       == NULL) ||
        (strstr(output, "fabric_packets_sent=") == NULL)) {
        fputs("m63: q8 cluster output check failed\n", stderr);
        return -1;
    }

    /* 5. source comparison --report */
    if (run_command(
            "python3 compiler/compare_att1_to_source.py "
            "--safetensors compiler/fixtures/m63_llama_2l.safetensors "
            "--config compiler/fixtures/m63_llama_config.json "
            "--att1-f32 models/m63_f32/model.att1 "
            "--att1-q8 models/m63_q8/model.att1 "
            "--prompt-ids 5,20,40 "
            "--report "
            "> build/m63_report.txt 2>&1") != 0) {
        fputs("m63: source comparison --report failed\n", stderr);
        return -1;
    }
    if (read_file("build/m63_report.txt", output, sizeof(output)) != 0) {
        fputs("m63: cannot read --report output\n", stderr);
        return -1;
    }
    if (strstr(output, "result:              pass") == NULL) {
        fputs("m63: result not pass\n", stderr);
        return -1;
    }
    if (strstr(output, "report:              ok") == NULL) {
        fputs("m63: report: ok not found\n", stderr);
        return -1;
    }

    /* 6. source comparison --report-json */
    if (run_command(
            "python3 compiler/compare_att1_to_source.py "
            "--safetensors compiler/fixtures/m63_llama_2l.safetensors "
            "--config compiler/fixtures/m63_llama_config.json "
            "--att1-f32 models/m63_f32/model.att1 "
            "--att1-q8 models/m63_q8/model.att1 "
            "--prompt-ids 5,20,40 "
            "--report-json build/m63_rpt.json "
            "> /dev/null 2>&1") != 0) {
        fputs("m63: --report-json failed\n", stderr);
        return -1;
    }
    if (read_file("build/m63_rpt.json", output, sizeof(output)) != 0) {
        fputs("m63: cannot read JSON report\n", stderr);
        return -1;
    }
    if ((strstr(output, "\"result\"")     == NULL) ||
        (strstr(output, "\"f32_static\"") == NULL) ||
        (strstr(output, "\"forward\"")    == NULL)) {
        fputs("m63: JSON missing expected keys\n", stderr);
        return -1;
    }

    return 0;
}

/*
 * check_compat_scanner() — M66
 *
 * Python-skippable.  Validates compiler/check_llama_compat.py against the
 * checked-in m66_compat_fixture (vocab=16, d_model=8, n_heads=2, n_layers=2)
 * and tiny_llama_2l.safetensors (F32, 21 tensors):
 *   1. Human-readable pass report
 *   2. JSON pass report
 *   3. Missing model-dir fails with exit 1 and "compat: error"
 *   4. Missing safetensors (no --no-tensors) → compat: fail with error message
 */
static int check_compat_scanner(void)
{
    char output[16384];

    /* Skip gracefully when Python 3 is not available. */
    if (run_command("python3 --version > /dev/null 2>&1") != 0) {
        return 0; /* skip — Python absent */
    }

    /* 1. Human-readable pass report against tiny fixture. */
    if (run_command(
            "python3 compiler/check_llama_compat.py"
            " --model-dir compiler/fixtures/m66_compat_fixture"
            " --safetensors compiler/fixtures/tiny_llama_2l.safetensors"
            " > build/compat_pass.txt 2>&1") != 0) {
        fputs("compat: pass run failed\n", stderr);
        return -1;
    }
    if (read_file("build/compat_pass.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if ((strstr(output, "compat: pass")          == NULL) ||
        (strstr(output, "model_type")             == NULL) ||
        (strstr(output, "llama")                  == NULL) ||
        (strstr(output, "tensor_count")           == NULL) ||
        (strstr(output, "21")                     == NULL) ||
        (strstr(output, "source_dtype")           == NULL) ||
        (strstr(output, "F32")                    == NULL) ||
        (strstr(output, "llama_check")            == NULL) ||
        (strstr(output, "ok")                     == NULL) ||
        (strstr(output, "vocab_size_match")       == NULL) ||
        (strstr(output, "yes")                    == NULL) ||
        (strstr(output, "required_changes")       == NULL) ||
        (strstr(output, "none")                   == NULL)) {
        fputs("compat: pass output missing expected fields\n", stderr);
        return -1;
    }

    /* 2. JSON pass report. */
    if (run_command(
            "python3 compiler/check_llama_compat.py"
            " --model-dir compiler/fixtures/m66_compat_fixture"
            " --safetensors compiler/fixtures/tiny_llama_2l.safetensors"
            " --json"
            " > build/compat_pass.json 2>&1") != 0) {
        fputs("compat: --json run failed\n", stderr);
        return -1;
    }
    if (read_file("build/compat_pass.json", output, sizeof(output)) != 0) {
        return -1;
    }
    if ((strstr(output, "\"compat\"")           == NULL) ||
        (strstr(output, "\"pass\"")             == NULL) ||
        (strstr(output, "\"arch\"")             == NULL) ||
        (strstr(output, "\"llama\"")            == NULL) ||
        (strstr(output, "\"required_changes\"") == NULL) ||
        (strstr(output, "\"safetensors\"")      == NULL) ||
        (strstr(output, "\"tokenizer\"")        == NULL) ||
        (strstr(output, "\"estimates\"")        == NULL)) {
        fputs("compat: --json output missing expected keys\n", stderr);
        return -1;
    }

    /* 3. Missing model-dir must exit non-zero and print "compat: error". */
    if (run_command(
            "python3 compiler/check_llama_compat.py"
            " --model-dir /nonexistent_att1_m66_test_dir"
            " > build/compat_nodir.txt 2>&1") == 0) {
        fputs("compat: missing dir should have failed\n", stderr);
        return -1;
    }
    if ((read_file("build/compat_nodir.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "compat: error") == NULL)) {
        fputs("compat: expected \"compat: error\" for missing dir\n", stderr);
        return -1;
    }

    /* 4. Missing safetensors (default path, no --no-tensors) must exit
     *    non-zero and report "safetensors file not found". */
    if (run_command(
            "python3 compiler/check_llama_compat.py"
            " --model-dir compiler/fixtures/m66_compat_fixture"
            " > build/compat_no_st.txt 2>&1") == 0) {
        fputs("compat: missing safetensors should have failed\n", stderr);
        return -1;
    }
    if ((read_file("build/compat_no_st.txt", output, sizeof(output)) != 0) ||
        (strstr(output, "safetensors file not found") == NULL)) {
        fputs("compat: expected \"safetensors file not found\" message\n", stderr);
        return -1;
    }

    return 0;
}

/*
 * check_bf16_coercion() — M67
 *
 * Python-skippable.  Validates BF16→F32 coercion in load_safetensors.py and
 * the updated compat scanner verdict using the checked-in BF16 fixture
 * (compiler/fixtures/m67_bf16_llama_2l.safetensors, 21 tensors, BF16, seed=67):
 *   1. Load a single BF16 tensor: dtype shows "BF16->F32", load: ok.
 *   2. --check-values: all 21 tensors loadable (values_ok=21, skipped=0).
 *   3. Convert BF16 fixture to .att1 via convert_llama_to_att1.py: exit 0.
 *   4. Compat scanner: compat: pass, no required_change: line for BF16.
 */
static int check_bf16_coercion(void)
{
    char output[8192];

    /* Skip gracefully when Python 3 is not available. */
    if (run_command("python3 --version > /dev/null 2>&1") != 0) {
        return 0; /* skip — Python absent */
    }

    /* 1. Load single BF16 tensor — coercion display "BF16->F32". */
    if (run_command(
            "python3 compiler/load_safetensors.py"
            " compiler/fixtures/m67_bf16_llama_2l.safetensors"
            " --tensor model.embed_tokens.weight"
            " > build/bf16_load_embed.txt 2>&1") != 0) {
        fputs("bf16_coercion: single tensor load failed\n", stderr);
        return -1;
    }
    if (read_file("build/bf16_load_embed.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if ((strstr(output, "dtype: BF16->F32") == NULL) ||
        (strstr(output, "shape: [16,8]")    == NULL) ||
        (strstr(output, "elements: 128")    == NULL) ||
        (strstr(output, "load: ok")         == NULL)) {
        fputs("bf16_coercion: single tensor report missing expected fields\n", stderr);
        return -1;
    }

    /* 2. --check-values: all 21 BF16 tensors must be loadable. */
    if (run_command(
            "python3 compiler/load_safetensors.py"
            " compiler/fixtures/m67_bf16_llama_2l.safetensors"
            " --check-values"
            " > build/bf16_chkval.txt 2>&1") != 0) {
        fputs("bf16_coercion: --check-values failed\n", stderr);
        return -1;
    }
    if (read_file("build/bf16_chkval.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if ((strstr(output, "values_ok=21")     == NULL) ||
        (strstr(output, "values_skipped=0") == NULL) ||
        (strstr(output, "check_values: ok") == NULL)) {
        fputs("bf16_coercion: --check-values output unexpected\n", stderr);
        return -1;
    }

    /* 3. Convert BF16 fixture to .att1: must succeed (exit 0). */
    if (run_command(
            "python3 compiler/convert_llama_to_att1.py"
            " --safetensors compiler/fixtures/m67_bf16_llama_2l.safetensors"
            " --config compiler/fixtures/tiny_llama_config.json"
            " --out build/m67_f32/model.att1"
            " > build/bf16_convert.txt 2>&1") != 0) {
        fputs("bf16_coercion: conversion failed\n", stderr);
        return -1;
    }

    /* 4. Compat scanner: BF16 is now a warning, not a required change.
     *    Expect compat: pass and no "required_change:" line. */
    if (run_command(
            "python3 compiler/check_llama_compat.py"
            " --model-dir compiler/fixtures/m66_compat_fixture"
            " --safetensors compiler/fixtures/m67_bf16_llama_2l.safetensors"
            " > build/bf16_compat.txt 2>&1") != 0) {
        fputs("bf16_coercion: compat scan unexpectedly failed\n", stderr);
        return -1;
    }
    if (read_file("build/bf16_compat.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if ((strstr(output, "compat: pass")     == NULL) ||
        (strstr(output, "required_change:") != NULL)) {
        fputs("bf16_coercion: compat output unexpected\n", stderr);
        return -1;
    }

    return 0;
}

/*
 * check_q8_conversion() — M68
 *
 * Python-skippable.  Validates the q8 conversion path using the checked-in
 * BF16 fixture (compiler/fixtures/m67_bf16_llama_2l.safetensors):
 *   1. Convert BF16 fixture → q8 .att1 (--weight-format q8).
 *   2. att1-inspect: verify dtype_name=q8 appears in output.
 *   3. att1-bench --backend cpu-q8 --mode single: exits 0, backend label.
 *   4. att1-bench --backend cpu-q8 --mode cluster --tiles 2: exits 0,
 *      fabric_packets_sent nonzero.
 *   5. (numpy-skippable) compare_att1_to_source.py: static q8 error within
 *      tolerance and forward-pass last_token matches f32.
 */
static int check_q8_conversion(void)
{
    char output[8192];

    /* Skip gracefully when Python 3 is not available. */
    if (run_command("python3 --version > /dev/null 2>&1") != 0) {
        return 0; /* skip — Python absent */
    }

    /* Write token IDs file (vocab=16 in tiny fixture; 1,2,3 are all valid). */
    {
        FILE *fp = fopen("build/m68_ids.txt", "w");
        if (fp == NULL) {
            fputs("q8_conversion: could not create token IDs file\n", stderr);
            return -1;
        }
        fputs("1\n2\n3\n", fp);
        fclose(fp);
    }

    /* 1. Convert BF16 fixture to q8 .att1. */
    if (run_command(
            "python3 compiler/convert_llama_to_att1.py"
            " --safetensors compiler/fixtures/m67_bf16_llama_2l.safetensors"
            " --config compiler/fixtures/tiny_llama_config.json"
            " --weight-format q8"
            " --out build/m68_q8/model.att1"
            " > build/m68_convert.txt 2>&1") != 0) {
        fputs("q8_conversion: q8 conversion failed\n", stderr);
        return -1;
    }

    /* 2. att1-inspect: q8 tensors must appear. */
    if (run_command(
            "./build/att1-inspect build/m68_q8/model.att1"
            " > build/m68_inspect.txt 2>&1") != 0) {
        fputs("q8_conversion: att1-inspect failed\n", stderr);
        return -1;
    }
    if (read_file("build/m68_inspect.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if ((strstr(output, "dtype_name=q8")   == NULL) ||
        (strstr(output, "quant=per-row-q8") == NULL) ||
        (strstr(output, "tensor_count=21")  == NULL)) {
        fputs("q8_conversion: inspect output missing q8 fields\n", stderr);
        return -1;
    }

    /* 3. cpu-q8 single-tile inference. */
    if (run_command(
            "./build/att1-bench"
            " --model build/m68_q8/model.att1"
            " --tokenizer external"
            " --tokens-file build/m68_ids.txt"
            " --tokens 1 --mode single --backend cpu-q8"
            " > build/m68_q8_single.txt 2>&1") != 0) {
        fputs("q8_conversion: cpu-q8 single failed\n", stderr);
        return -1;
    }
    if (read_file("build/m68_q8_single.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if ((strstr(output, "mode=single")    == NULL) ||
        (strstr(output, "backend=cpu-q8") == NULL) ||
        (strstr(output, "last_token=")    == NULL)) {
        fputs("q8_conversion: cpu-q8 single output unexpected\n", stderr);
        return -1;
    }

    /* 4. cpu-q8 cluster inference. */
    if (run_command(
            "./build/att1-bench"
            " --model build/m68_q8/model.att1"
            " --tokenizer external"
            " --tokens-file build/m68_ids.txt"
            " --tokens 1 --mode cluster --tiles 2 --backend cpu-q8"
            " > build/m68_q8_cluster.txt 2>&1") != 0) {
        fputs("q8_conversion: cpu-q8 cluster failed\n", stderr);
        return -1;
    }
    if (read_file("build/m68_q8_cluster.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if ((strstr(output, "mode=cluster")         == NULL) ||
        (strstr(output, "backend=cpu-q8")       == NULL) ||
        (strstr(output, "fabric_packets_sent=") == NULL)) {
        fputs("q8_conversion: cpu-q8 cluster output unexpected\n", stderr);
        return -1;
    }
    /* Cluster mode must have sent at least one fabric packet. */
    if (strstr(output, "fabric_packets_sent=0") != NULL) {
        fputs("q8_conversion: cpu-q8 cluster sent zero fabric packets\n", stderr);
        return -1;
    }

    /* 5. Source comparison (numpy-skippable): static q8 error within
     *    tolerance and forward-pass token matches. */
    if (run_command("python3 -c 'import numpy' > /dev/null 2>&1") != 0) {
        return 0; /* skip forward-pass comparison — numpy absent */
    }
    if (run_command(
            "python3 compiler/compare_att1_to_source.py"
            " --safetensors compiler/fixtures/m67_bf16_llama_2l.safetensors"
            " --config compiler/fixtures/tiny_llama_config.json"
            " --att1-f32 build/m67_f32/model.att1"
            " --att1-q8 build/m68_q8/model.att1"
            " --prompt-ids 1,2,3"
            " --report"
            " > build/m68_cmp.txt 2>&1") != 0) {
        fputs("q8_conversion: source comparison failed\n", stderr);
        return -1;
    }
    if (read_file("build/m68_cmp.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if ((strstr(output, "q8_status:           pass") == NULL) ||
        (strstr(output, "result:              pass") == NULL)) {
        fputs("q8_conversion: comparison result not pass\n", stderr);
        return -1;
    }

    return 0;
}

/*
 * check_public_backend_smoke() — M70
 *
 * Python-skippable. Validates the manual public-model backend smoke driver
 * against checked-in tiny converted artifacts. CUDA rows may report
 * unsupported on CPU-only hosts; that is expected and must not fail the repo.
 */
static int check_public_backend_smoke(void)
{
    char output[16384];

    /* Skip gracefully when Python 3 is not available. */
    if (run_command("python3 --version > /dev/null 2>&1") != 0) {
        return 0; /* skip — Python absent */
    }

    {
        FILE *fp = fopen("build/m70_ids.txt", "w");
        if (fp == NULL) {
            fputs("public_backend_smoke: could not create token IDs file\n", stderr);
            return -1;
        }
        fputs("1\n2\n3\n", fp);
        fclose(fp);
    }

    if (run_command(
            "python3 compiler/validate_public_backends.py"
            " --model-dir compiler/fixtures/tiny_llama"
            " --att1-f32 models/real_tiny_f32/model.att1"
            " --att1-q8 models/real_tiny_q8/model.att1"
            " --tokens-file build/m70_ids.txt"
            " --tokens 1 --tiles 2"
            " --report-json build/m70_backend_report.json"
            " > build/m70_backend_report.txt 2>&1") != 0) {
        fputs("public_backend_smoke: validation script failed\n", stderr);
        return -1;
    }

    if (read_file("build/m70_backend_report.txt", output, sizeof(output)) != 0) {
        fputs("public_backend_smoke: cannot read report\n", stderr);
        return -1;
    }

    if ((strstr(output, "backend=cpu-f32 mode=single") == NULL) ||
        (strstr(output, "backend=cpu-f32 mode=cluster") == NULL) ||
        (strstr(output, "backend=cpu-q8 mode=single") == NULL) ||
        (strstr(output, "backend=cpu-q8 mode=cluster") == NULL) ||
        (strstr(output, "generated_tokens=1") == NULL) ||
        (strstr(output, "last_token=") == NULL) ||
        (strstr(output, "token_time_us_total=") == NULL) ||
        (strstr(output, "status=pass") == NULL) ||
        (strstr(output, "result: pass") == NULL) ||
        (strstr(output, "report: ok") == NULL)) {
        fputs("public_backend_smoke: report missing expected fields\n", stderr);
        return -1;
    }

    if (read_file("build/m70_backend_report.json", output, sizeof(output)) != 0) {
        fputs("public_backend_smoke: cannot read JSON report\n", stderr);
        return -1;
    }
    if ((strstr(output, "\"runs\"") == NULL) ||
        (strstr(output, "\"result\"") == NULL) ||
        (strstr(output, "\"pass\"") == NULL)) {
        fputs("public_backend_smoke: JSON report missing expected fields\n", stderr);
        return -1;
    }

    return 0;
}

/*
 * check_public_tokenized_validation() - M71
 *
 * Python-skippable.  When tokenizers or transformers is available, validates
 * the full public-model manual path on tiny fixtures:
 * tokenize_hf.py -> token IDs file -> pretokenized f32/q8 backend smoke.
 */
static int check_public_tokenized_validation(void)
{
    char output[16384];
    int  has_tokenizers;
    int  has_transformers;

    /* Skip gracefully when Python 3 is not available. */
    if (run_command("python3 --version > /dev/null 2>&1") != 0) {
        return 0; /* skip - Python absent */
    }

    has_tokenizers   = (run_command(
        "python3 -c 'import tokenizers' > /dev/null 2>&1") == 0);
    has_transformers = (run_command(
        "python3 -c 'import transformers' > /dev/null 2>&1") == 0);
    if (!has_tokenizers && !has_transformers) {
        return 0; /* skip - neither tokenizers nor transformers installed */
    }

    if (run_command(
            "python3 compiler/validate_public_tokenized.py"
            " --model-dir compiler/fixtures/tiny_llama"
            " --tokenizer-dir compiler/fixtures/tiny_tokenizer"
            " --prompt-text abc"
            " --att1-f32 models/real_tiny_f32/model.att1"
            " --att1-q8 models/real_tiny_q8/model.att1"
            " --tokens-file build/m71_ids.txt"
            " --tokenizer-json build/m71_tokenizer.json"
            " --tokens 1 --tiles 2"
            " --report-json build/m71_tokenized_report.json"
            " > build/m71_tokenized_report.txt 2>&1") != 0) {
        fputs("public_tokenized: validation script failed\n", stderr);
        return -1;
    }

    if (read_file("build/m71_tokenized_report.txt", output, sizeof(output)) != 0) {
        fputs("public_tokenized: cannot read report\n", stderr);
        return -1;
    }

    if ((strstr(output, "prompt_text: abc") == NULL) ||
        (strstr(output, "token_ids:") == NULL) ||
        (strstr(output, "token_count:") == NULL) ||
        (strstr(output, "backend=cpu-f32 mode=single") == NULL) ||
        (strstr(output, "backend=cpu-f32 mode=cluster") == NULL) ||
        (strstr(output, "backend=cpu-q8 mode=single") == NULL) ||
        (strstr(output, "backend=cpu-q8 mode=cluster") == NULL) ||
        (strstr(output, "generated_tokens=1") == NULL) ||
        (strstr(output, "last_token=") == NULL) ||
        (strstr(output, "token_time_us_total=") == NULL) ||
        (strstr(output, "status=pass") == NULL) ||
        (strstr(output, "result: pass") == NULL) ||
        (strstr(output, "report: ok") == NULL)) {
        fputs("public_tokenized: report missing expected fields\n", stderr);
        return -1;
    }

    if (read_file("build/m71_tokenized_report.json", output, sizeof(output)) != 0) {
        fputs("public_tokenized: cannot read JSON report\n", stderr);
        return -1;
    }
    if ((strstr(output, "\"prompt_text\"") == NULL) ||
        (strstr(output, "\"token_ids\"") == NULL) ||
        (strstr(output, "\"token_count\"") == NULL) ||
        (strstr(output, "\"runs\"") == NULL) ||
        (strstr(output, "\"result\"") == NULL)) {
        fputs("public_tokenized: JSON report missing expected fields\n", stderr);
        return -1;
    }

    return 0;
}

/*
 * check_scaling_report() — M72
 *
 * Validates the extended att1-size scaling report for three input modes:
 *   1. Existing preset modes still work (regression).
 *   2. --config PATH mode with tiny LLaMA fixture.
 *   3. --config --json mode.
 *   4. Manual shape mode (--layers, --d-model, --heads, --d-ff, --vocab-size).
 *   5. Bad config path fails with exit code 1.
 *   6. Invalid d_model/heads combination fails with exit code 1.
 */
static int check_scaling_report(void)
{
    char output[8192];

    /* 1a. Existing tiny-dummy preset still works. */
    if (run_command("./build/att1-size --preset tiny-dummy"
                    " > build/m72_size_tiny.txt 2>&1") != 0) {
        fputs("scaling_report: tiny-dummy preset failed\n", stderr);
        return -1;
    }

    /* 1b. Existing gpt-oss-120b-shape preset still works. */
    if (run_command("./build/att1-size --preset gpt-oss-120b-shape"
                    " --tiles 8 --context 8192 --dtype q4"
                    " > build/m72_size_gptoss.txt 2>&1") != 0) {
        fputs("scaling_report: gpt-oss preset failed\n", stderr);
        return -1;
    }
    if (read_file("build/m72_size_gptoss.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if ((strstr(output, "synthetic/non-executable") == NULL) ||
        (strstr(output, "dtype=q4")                 == NULL) ||
        (strstr(output, "max_seq_len=8192")          == NULL) ||
        (strstr(output, "tiles=8")                   == NULL)) {
        fputs("scaling_report: gpt-oss preset output unexpected\n", stderr);
        return -1;
    }

    /* 2. --config mode with tiny LLaMA fixture. */
    if (run_command("./build/att1-size"
                    " --config compiler/fixtures/tiny_llama_config.json"
                    " --tiles 2"
                    " > build/m72_size_config.txt 2>&1") != 0) {
        fputs("scaling_report: --config mode failed\n", stderr);
        return -1;
    }
    if (read_file("build/m72_size_config.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if ((strstr(output, "vocab_size")            == NULL) ||
        (strstr(output, "n_layers")              == NULL) ||
        (strstr(output, "storage")               == NULL) ||
        (strstr(output, "kv_cache")              == NULL) ||
        (strstr(output, "cluster_placement")     == NULL) ||
        (strstr(output, "aimu_tile_plan")        == NULL) ||
        (strstr(output, "backend_feasibility")   == NULL)) {
        fputs("scaling_report: --config report missing sections\n", stderr);
        return -1;
    }

    /* 3. --config --json mode. */
    if (run_command("./build/att1-size"
                    " --config compiler/fixtures/tiny_llama_config.json"
                    " --json"
                    " > build/m72_size_json.txt 2>&1") != 0) {
        fputs("scaling_report: --config --json failed\n", stderr);
        return -1;
    }
    if (read_file("build/m72_size_json.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if ((strstr(output, "\"shape\"")               == NULL) ||
        (strstr(output, "\"storage\"")             == NULL) ||
        (strstr(output, "\"kv_cache_f32\"")        == NULL) ||
        (strstr(output, "\"cluster\"")             == NULL) ||
        (strstr(output, "\"backend_feasibility\"") == NULL)) {
        fputs("scaling_report: JSON report missing fields\n", stderr);
        return -1;
    }

    /* 4. Manual shape mode. */
    if (run_command("./build/att1-size"
                    " --layers 4 --d-model 256 --heads 4 --d-ff 512"
                    " --vocab-size 1000 --tiles 2 --context 512"
                    " > build/m72_size_manual.txt 2>&1") != 0) {
        fputs("scaling_report: manual mode failed\n", stderr);
        return -1;
    }
    if (read_file("build/m72_size_manual.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if ((strstr(output, "vocab_size     = 1000") == NULL) ||
        (strstr(output, "n_layers       = 4")    == NULL) ||
        (strstr(output, "backend_feasibility")   == NULL)) {
        fputs("scaling_report: manual mode output unexpected\n", stderr);
        return -1;
    }

    /* 5. Bad config path must fail. */
    if (run_command("./build/att1-size"
                    " --config /nonexistent/config.json"
                    " > build/m72_size_bad.txt 2>&1") == 0) {
        fputs("scaling_report: bad config path should fail\n", stderr);
        return -1;
    }

    /* 6. Invalid d_model/heads must fail. */
    if (run_command("./build/att1-size"
                    " --layers 4 --d-model 100 --heads 7 --d-ff 256"
                    " --vocab-size 1000"
                    " > build/m72_size_invalid.txt 2>&1") == 0) {
        fputs("scaling_report: invalid d_model/heads should fail\n", stderr);
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
        (check_tokenizer_import_report()  != 0) ||
        (check_tokenizer_selection()      != 0) ||
        (check_external_tokenizer()       != 0) ||
        (check_hf_tokenizer()             != 0) ||
        (check_pretokenized_pipeline()    != 0) ||
        (check_source_comparison()        != 0) ||
        (check_m63_validation()           != 0) ||
        (check_compat_scanner()           != 0) ||
        (check_bf16_coercion()            != 0) ||
        (check_q8_conversion()            != 0) ||
        (check_public_backend_smoke()     != 0) ||
        (check_public_tokenized_validation() != 0) ||
        (check_scaling_report()           != 0)) {
        fputs("bench smoke test failed\n", stderr);
        return 1;
    }

    puts("bench smoke test passed");
    return 0;
}
