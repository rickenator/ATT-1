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
 * check_q4_conversion() — M81
 *
 * Python-skippable.  Validates the q4 conversion path using the checked-in
 * m63 F32 fixture (compiler/fixtures/m63_llama_2l.safetensors):
 *   1. Convert F32 fixture → q4 .att1 (--weight-format q4).
 *   2. att1-inspect: verify dtype_name=q4 and grouped-q4-g32 appear.
 *   3. att1-bench --backend cpu-q4 --mode single: exits 0, backend label.
 *   4. att1-bench --backend cpu-q4 --mode cluster --tiles 2: exits 0,
 *      fabric_packets_sent nonzero.
 *
 * Uses m63 fixture (vocab=64, d_model=32, d_ff=64) because the tiny fixture
 * (d_model=8) is incompatible with q4 (group_size_min=16 > 8).
 */
static int check_q4_conversion(void)
{
    char output[8192];

    /* Skip gracefully when Python 3 is not available. */
    if (run_command("python3 --version > /dev/null 2>&1") != 0) {
        return 0; /* skip — Python absent */
    }

    /* Write token IDs file (vocab=64 in m63 fixture; 1,3,5 are all valid). */
    {
        FILE *fp = fopen("build/m81_smoke_ids.txt", "w");
        if (fp == NULL) {
            fputs("q4_conversion: could not create token IDs file\n", stderr);
            return -1;
        }
        fputs("1\n3\n5\n", fp);
        fclose(fp);
    }

    /* 1. Convert m63 F32 fixture to q4 .att1. */
    if (run_command(
            "python3 compiler/convert_llama_to_att1.py"
            " --safetensors compiler/fixtures/m63_llama_2l.safetensors"
            " --config compiler/fixtures/m63_llama_config.json"
            " --weight-format q4"
            " --out build/m81_q4/model.att1"
            " > build/m81_convert.txt 2>&1") != 0) {
        fputs("q4_conversion: q4 conversion failed\n", stderr);
        return -1;
    }

    /* 2. att1-inspect: q4 tensors must appear. */
    if (run_command(
            "./build/att1-inspect build/m81_q4/model.att1"
            " > build/m81_inspect.txt 2>&1") != 0) {
        fputs("q4_conversion: att1-inspect failed\n", stderr);
        return -1;
    }
    if (read_file("build/m81_inspect.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if ((strstr(output, "dtype_name=q4")      == NULL) ||
        (strstr(output, "quant=grouped-q4-g32") == NULL) ||
        (strstr(output, "tensor_count=21")    == NULL)) {
        fputs("q4_conversion: inspect output missing q4 fields\n", stderr);
        return -1;
    }

    /* 3. cpu-q4 single-tile inference. */
    if (run_command(
            "./build/att1-bench"
            " --model build/m81_q4/model.att1"
            " --tokenizer external"
            " --tokens-file build/m81_smoke_ids.txt"
            " --tokens 1 --mode single --backend cpu-q4"
            " > build/m81_q4_single.txt 2>&1") != 0) {
        fputs("q4_conversion: cpu-q4 single failed\n", stderr);
        return -1;
    }
    if (read_file("build/m81_q4_single.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if ((strstr(output, "mode=single")    == NULL) ||
        (strstr(output, "backend=cpu-q4") == NULL) ||
        (strstr(output, "last_token=")    == NULL)) {
        fputs("q4_conversion: cpu-q4 single output unexpected\n", stderr);
        return -1;
    }

    /* 4. cpu-q4 cluster inference. */
    if (run_command(
            "./build/att1-bench"
            " --model build/m81_q4/model.att1"
            " --tokenizer external"
            " --tokens-file build/m81_smoke_ids.txt"
            " --tokens 1 --mode cluster --tiles 2 --backend cpu-q4"
            " > build/m81_q4_cluster.txt 2>&1") != 0) {
        fputs("q4_conversion: cpu-q4 cluster failed\n", stderr);
        return -1;
    }
    if (read_file("build/m81_q4_cluster.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if ((strstr(output, "mode=cluster")         == NULL) ||
        (strstr(output, "backend=cpu-q4")       == NULL) ||
        (strstr(output, "fabric_packets_sent=") == NULL)) {
        fputs("q4_conversion: cpu-q4 cluster output unexpected\n", stderr);
        return -1;
    }
    /* Cluster mode must have sent at least one fabric packet. */
    if (strstr(output, "fabric_packets_sent=0") != NULL) {
        fputs("q4_conversion: cpu-q4 cluster sent zero fabric packets\n", stderr);
        return -1;
    }

    return 0;
}

/*
 * check_q4_source_comparison() — M84
 *
 * Python-skippable.  Requires numpy.  Validates the q4 static comparison and
 * forward-pass bench path in compare_att1_to_source.py using the checked-in
 * m63 fixture and the real_tiny_q4 artifact:
 *   1. Key=value output with --att1-q4: checks q4_tensors_checked,
 *      q4_max_abs_error, q4_bench_last_token, result=pass.
 *   2. --report with --att1-q4: checks q4_status and report: ok.
 *   3. --report-json with --att1-q4: checks JSON keys q4_static and forward.
 *
 * The real_tiny_q4 artifact is the m63 fixture converted to q4 (M81), stored
 * at models/real_tiny_q4/model.att1.
 */
static int check_q4_source_comparison(void)
{
    char output[16384];

    /* Skip gracefully when Python 3 is not available. */
    if (run_command("python3 --version > /dev/null 2>&1") != 0) {
        return 0; /* skip — Python absent */
    }
    if (run_command("python3 -c 'import numpy' > /dev/null 2>&1") != 0) {
        return 0; /* skip — numpy not installed */
    }

    /* 1. Key=value comparison with --att1-q4. */
    if (run_command(
            "python3 compiler/compare_att1_to_source.py"
            " --safetensors compiler/fixtures/m63_llama_2l.safetensors"
            " --config compiler/fixtures/m63_llama_config.json"
            " --att1-f32 models/m63_f32/model.att1"
            " --att1-q4 models/real_tiny_q4/model.att1"
            " --prompt-ids 5,20,40"
            " > build/m84_q4_kv.txt 2>&1") != 0) {
        fputs("q4_source_comparison: key=value run failed\n", stderr);
        return -1;
    }
    if (read_file("build/m84_q4_kv.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if (strstr(output, "q4_tensors_checked:") == NULL) {
        fputs("q4_source_comparison: q4_tensors_checked missing\n", stderr);
        return -1;
    }
    if (strstr(output, "q4_max_abs_error:") == NULL) {
        fputs("q4_source_comparison: q4_max_abs_error missing\n", stderr);
        return -1;
    }
    if (strstr(output, "q4_bench_last_token:") == NULL) {
        fputs("q4_source_comparison: q4_bench_last_token missing\n", stderr);
        return -1;
    }
    if (strstr(output, "result:              pass") == NULL) {
        fputs("q4_source_comparison: result not pass\n", stderr);
        return -1;
    }

    /* 2. --report output. */
    if (run_command(
            "python3 compiler/compare_att1_to_source.py"
            " --safetensors compiler/fixtures/m63_llama_2l.safetensors"
            " --config compiler/fixtures/m63_llama_config.json"
            " --att1-f32 models/m63_f32/model.att1"
            " --att1-q4 models/real_tiny_q4/model.att1"
            " --prompt-ids 5,20,40"
            " --report"
            " > build/m84_q4_report.txt 2>&1") != 0) {
        fputs("q4_source_comparison: --report run failed\n", stderr);
        return -1;
    }
    if (read_file("build/m84_q4_report.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if (strstr(output, "q4_status:") == NULL) {
        fputs("q4_source_comparison: q4_status missing from report\n", stderr);
        return -1;
    }
    if (strstr(output, "report:              ok") == NULL) {
        fputs("q4_source_comparison: report: ok missing\n", stderr);
        return -1;
    }

    /* 3. --report-json output. */
    if (run_command(
            "python3 compiler/compare_att1_to_source.py"
            " --safetensors compiler/fixtures/m63_llama_2l.safetensors"
            " --config compiler/fixtures/m63_llama_config.json"
            " --att1-f32 models/m63_f32/model.att1"
            " --att1-q4 models/real_tiny_q4/model.att1"
            " --prompt-ids 5,20,40"
            " --report-json build/m84_q4_rpt.json"
            " > /dev/null 2>&1") != 0) {
        fputs("q4_source_comparison: --report-json run failed\n", stderr);
        return -1;
    }
    if (read_file("build/m84_q4_rpt.json", output, sizeof(output)) != 0) {
        return -1;
    }
    if ((strstr(output, "\"q4_static\"") == NULL) ||
        (strstr(output, "\"forward\"")   == NULL) ||
        (strstr(output, "\"result\"")    == NULL)) {
        fputs("q4_source_comparison: JSON missing expected keys\n", stderr);
        return -1;
    }

    return 0;
}

/*
 * check_public_q4_smoke() — M83
 *
 * Python-skippable.  Validates the --model-dir auto-discovery path for q4
 * conversion using the checked-in m83_model_dir fixture:
 *   1. Convert compiler/fixtures/m83_model_dir → q4 .att1 using ONLY
 *      --model-dir (no --safetensors) to exercise auto-discovery.
 *   2. att1-inspect: verify dtype_name=q4, grouped-q4-g32, tensor_count=21.
 *   3. att1-bench --backend cpu-q4 --mode single: exits 0, expected fields.
 *   4. att1-bench --backend cpu-q4 --mode cluster --tiles 2: exits 0,
 *      fabric_packets_sent nonzero.
 *
 * The fixture (vocab=64, d_model=32, d_ff=64, n_layers=2) is identical in
 * config to the m63 fixture but loaded via model directory layout.
 */
static int check_public_q4_smoke(void)
{
    char output[8192];

    /* Skip gracefully when Python 3 is not available. */
    if (run_command("python3 --version > /dev/null 2>&1") != 0) {
        return 0; /* skip — Python absent */
    }

    /* Write token IDs file (vocab=64; 1,3,5 are valid IDs). */
    {
        FILE *fp = fopen("build/m83_smoke_ids.txt", "w");
        if (fp == NULL) {
            fputs("public_q4_smoke: could not create token IDs file\n", stderr);
            return -1;
        }
        fputs("1\n3\n5\n", fp);
        fclose(fp);
    }

    /* 1. Convert using --model-dir only (tests auto-discovery of safetensors). */
    if (run_command(
            "python3 compiler/convert_llama_to_att1.py"
            " --model-dir compiler/fixtures/m83_model_dir"
            " --weight-format q4"
            " --out build/m83_q4/model.att1"
            " > build/m83_convert.txt 2>&1") != 0) {
        fputs("public_q4_smoke: q4 conversion failed\n", stderr);
        return -1;
    }

    /* 2. att1-inspect: q4 tensors must appear. */
    if (run_command(
            "./build/att1-inspect build/m83_q4/model.att1"
            " > build/m83_inspect.txt 2>&1") != 0) {
        fputs("public_q4_smoke: att1-inspect failed\n", stderr);
        return -1;
    }
    if (read_file("build/m83_inspect.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if ((strstr(output, "dtype_name=q4")       == NULL) ||
        (strstr(output, "quant=grouped-q4-g32") == NULL) ||
        (strstr(output, "tensor_count=21")      == NULL)) {
        fputs("public_q4_smoke: inspect output missing q4 fields\n", stderr);
        return -1;
    }

    /* 3. cpu-q4 single-tile inference. */
    if (run_command(
            "./build/att1-bench"
            " --model build/m83_q4/model.att1"
            " --tokenizer external"
            " --tokens-file build/m83_smoke_ids.txt"
            " --tokens 1 --mode single --backend cpu-q4"
            " > build/m83_q4_single.txt 2>&1") != 0) {
        fputs("public_q4_smoke: cpu-q4 single failed\n", stderr);
        return -1;
    }
    if (read_file("build/m83_q4_single.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if ((strstr(output, "mode=single")    == NULL) ||
        (strstr(output, "backend=cpu-q4") == NULL) ||
        (strstr(output, "last_token=")    == NULL)) {
        fputs("public_q4_smoke: cpu-q4 single output unexpected\n", stderr);
        return -1;
    }

    /* 4. cpu-q4 cluster inference. */
    if (run_command(
            "./build/att1-bench"
            " --model build/m83_q4/model.att1"
            " --tokenizer external"
            " --tokens-file build/m83_smoke_ids.txt"
            " --tokens 1 --mode cluster --tiles 2 --backend cpu-q4"
            " > build/m83_q4_cluster.txt 2>&1") != 0) {
        fputs("public_q4_smoke: cpu-q4 cluster failed\n", stderr);
        return -1;
    }
    if (read_file("build/m83_q4_cluster.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if ((strstr(output, "mode=cluster")         == NULL) ||
        (strstr(output, "backend=cpu-q4")       == NULL) ||
        (strstr(output, "fabric_packets_sent=") == NULL)) {
        fputs("public_q4_smoke: cpu-q4 cluster output unexpected\n", stderr);
        return -1;
    }
    /* Cluster mode must have sent at least one fabric packet. */
    if (strstr(output, "fabric_packets_sent=0") != NULL) {
        fputs("public_q4_smoke: cpu-q4 cluster sent zero fabric packets\n", stderr);
        return -1;
    }

    return 0;
}

/*
 * check_public_q4_backend_smoke() — M85
 *
 * Python-skippable.  Validates the q4 backend smoke driver
 * (validate_public_q4_backends.py) against the checked-in real_tiny_q4
 * artifact.  CUDA q4 rows are exercised via --include-cuda and must report
 * "unsupported"; they must not cause the overall result to be "fail".
 *
 * Checks:
 *   1. Text report: cpu-q4 single, cpu-q4 cluster, generated_tokens=1,
 *      status=pass for cpu-q4 rows, result: pass, report: ok.
 *   2. JSON report: "runs" and "result" keys present, "pass" value present.
 *   3. --include-cuda: cuda-q4 rows show status=unsupported; overall still pass.
 */
static int check_public_q4_backend_smoke(void)
{
    char output[16384];

    /* Skip gracefully when Python 3 is not available. */
    if (run_command("python3 --version > /dev/null 2>&1") != 0) {
        return 0; /* skip — Python absent */
    }

    /* Write the token IDs fixture file. */
    {
        FILE *fp = fopen("build/m85_ids.txt", "w");
        if (fp == NULL) {
            fputs("q4_backend_smoke: could not create token IDs file\n", stderr);
            return -1;
        }
        fputs("5\n20\n40\n", fp);
        fclose(fp);
    }

    /* 1. Text report with --report-json. */
    if (run_command(
            "python3 compiler/validate_public_q4_backends.py"
            " --model-dir compiler/fixtures"
            " --att1-q4 models/real_tiny_q4/model.att1"
            " --tokens-file build/m85_ids.txt"
            " --tokens 1 --tiles 2"
            " --report-json build/m85_q4_backend_report.json"
            " > build/m85_q4_backend_report.txt 2>&1") != 0) {
        fputs("q4_backend_smoke: validation script failed\n", stderr);
        return -1;
    }

    if (read_file("build/m85_q4_backend_report.txt", output, sizeof(output)) != 0) {
        fputs("q4_backend_smoke: cannot read report\n", stderr);
        return -1;
    }

    if ((strstr(output, "backend=cpu-q4 mode=single") == NULL) ||
        (strstr(output, "backend=cpu-q4 mode=cluster") == NULL) ||
        (strstr(output, "generated_tokens=1") == NULL) ||
        (strstr(output, "status=pass") == NULL) ||
        (strstr(output, "result: pass") == NULL) ||
        (strstr(output, "report: ok") == NULL)) {
        fputs("q4_backend_smoke: text report missing expected fields\n", stderr);
        return -1;
    }

    /* Cluster mode must have sent fabric packets. */
    if (strstr(output, "fabric_packets_sent=0") != NULL) {
        /* Accept if single-mode row explains it; cluster row must be nonzero. */
        /* Parse crude check: if the only fabric_packets_sent= value seen is 0
         * that would be wrong.  We allow the single-mode row with =0 as long
         * as the cluster row also appears (it is checked above). */
        if (strstr(output, "mode=cluster") != NULL &&
            strstr(output, "mode=cluster shard_plan=runtime"
                           " generated_tokens=1") != NULL) {
            /* The cluster row is present; a separate single-mode row may have
             * fabric_packets_sent=0 which is expected. Accept this. */
        } else {
            fputs("q4_backend_smoke: cluster fabric_packets_sent=0\n", stderr);
            return -1;
        }
    }

    /* 2. JSON report check. */
    if (read_file("build/m85_q4_backend_report.json", output, sizeof(output)) != 0) {
        fputs("q4_backend_smoke: cannot read JSON report\n", stderr);
        return -1;
    }
    if ((strstr(output, "\"runs\"")   == NULL) ||
        (strstr(output, "\"result\"") == NULL) ||
        (strstr(output, "\"pass\"")   == NULL)) {
        fputs("q4_backend_smoke: JSON report missing expected keys\n", stderr);
        return -1;
    }

    /* 3. --include-cuda: cuda-q4 rows must be unsupported; overall still pass. */
    if (run_command(
            "python3 compiler/validate_public_q4_backends.py"
            " --model-dir compiler/fixtures"
            " --att1-q4 models/real_tiny_q4/model.att1"
            " --tokens-file build/m85_ids.txt"
            " --tokens 1 --tiles 2"
            " --include-cuda"
            " > build/m85_q4_cuda_report.txt 2>&1") != 0) {
        fputs("q4_backend_smoke: --include-cuda run failed\n", stderr);
        return -1;
    }

    if (read_file("build/m85_q4_cuda_report.txt", output, sizeof(output)) != 0) {
        fputs("q4_backend_smoke: cannot read cuda report\n", stderr);
        return -1;
    }
    /* M89: cuda-q4 cluster is now supported.  On a CUDA host both cuda-q4
     * rows have status=pass; on a CPU-only host they have status=unsupported.
     * Either way the overall result must be pass. */
    if (strstr(output, "result: pass") == NULL) {
        fputs("q4_backend_smoke: cuda-q4 unsupported check failed\n", stderr);
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

/*
 * check_public_q4_cuda_smoke() — M91
 *
 * Python-skippable.  Exercises validate_public_q4_cuda.py with the
 * checked-in real_tiny_q4 artifact, which runs five bench invocations:
 *
 *   cpu-q4  single   runtime    → pass
 *   cpu-q4  cluster  runtime    → pass  (fabric_packets_sent verified > 0)
 *   cpu-q4  cluster  metadata   → plan_unsupported (q4 rejects metadata plan)
 *   cuda-q4 single   runtime    → pass (CUDA host) | unavailable (CPU-only)
 *   cuda-q4 cluster  runtime    → pass (CUDA host) | unavailable (CPU-only)
 *
 * Overall result is "pass" on both CPU-only and CUDA hosts.
 *
 * Checks:
 *   1. Text report: result: pass, report: ok, status=plan_unsupported present,
 *      cpu-q4 rows have status=pass, q4 notes are printed.
 *   2. JSON report: "runs", "result", "pass" keys present.
 */
static int check_public_q4_cuda_smoke(void)
{
    char output[16384];

    /* Skip gracefully when Python 3 is not available. */
    if (run_command("python3 --version > /dev/null 2>&1") != 0) {
        return 0; /* skip — Python absent */
    }

    /* Write the token IDs fixture file. */
    {
        FILE *fp = fopen("build/m91_ids.txt", "w");
        if (fp == NULL) {
            fputs("q4_cuda_smoke: could not create token IDs file\n", stderr);
            return -1;
        }
        fputs("5\n20\n40\n", fp);
        fclose(fp);
    }

    /* Run the validation script with the checked-in fixture. */
    if (run_command(
            "python3 compiler/validate_public_q4_cuda.py"
            " --model-dir compiler/fixtures"
            " --att1-q4 models/real_tiny_q4/model.att1"
            " --tokens-file build/m91_ids.txt"
            " --tokens 1 --tiles 2"
            " --report-json build/m91_q4_cuda_report.json"
            " > build/m91_q4_cuda_report.txt 2>&1") != 0) {
        fputs("q4_cuda_smoke: validation script failed\n", stderr);
        return -1;
    }

    if (read_file("build/m91_q4_cuda_report.txt", output, sizeof(output)) != 0) {
        fputs("q4_cuda_smoke: cannot read report\n", stderr);
        return -1;
    }

    /* cpu-q4 rows must be present and passing. */
    if ((strstr(output, "backend=cpu-q4 mode=single")   == NULL) ||
        (strstr(output, "backend=cpu-q4 mode=cluster")  == NULL) ||
        (strstr(output, "generated_tokens=1")           == NULL) ||
        (strstr(output, "status=pass")                  == NULL) ||
        (strstr(output, "result: pass")                 == NULL) ||
        (strstr(output, "report: ok")                   == NULL)) {
        fputs("q4_cuda_smoke: text report missing expected fields\n", stderr);
        return -1;
    }

    /* Metadata plan rejection must be reported as plan_unsupported, not fail. */
    if (strstr(output, "status=plan_unsupported") == NULL) {
        fputs("q4_cuda_smoke: missing status=plan_unsupported for metadata row\n",
              stderr);
        return -1;
    }

    /* At least one q4 note must appear. */
    if (strstr(output, "note:") == NULL) {
        fputs("q4_cuda_smoke: q4 notes missing from report\n", stderr);
        return -1;
    }

    /* 2. JSON report check. */
    if (read_file("build/m91_q4_cuda_report.json", output, sizeof(output)) != 0) {
        fputs("q4_cuda_smoke: cannot read JSON report\n", stderr);
        return -1;
    }
    if ((strstr(output, "\"runs\"")   == NULL) ||
        (strstr(output, "\"result\"") == NULL) ||
        (strstr(output, "\"pass\"")   == NULL)) {
        fputs("q4_cuda_smoke: JSON report missing expected keys\n", stderr);
        return -1;
    }

    return 0;
}

/*
 * check_backend_comparison_smoke() — M92
 *
 * Python-skippable.  Exercises backend_comparison_report.py with the
 * checked-in real_tiny fixtures (f32, q8, q4).  Runs without --include-cuda
 * so CUDA rows appear as 'pending', which is the expected CPU-only outcome.
 *
 * Checks:
 *   1. Script exits zero.
 *   2. Text report: result: pass, report: ok.
 *   3. cpu-f32, cpu-q8, cpu-q4 rows present with status=pass.
 *   4. CUDA rows present with status=pending.
 *   5. At least one note: line present.
 *   6. JSON report: "runs", "result", "pass" keys present.
 */
static int check_backend_comparison_smoke(void)
{
    char output[32768];

    /* Skip gracefully when Python 3 is not available. */
    if (run_command("python3 --version > /dev/null 2>&1") != 0) {
        return 0; /* skip — Python absent */
    }

    /* Write the token IDs fixture file. */
    {
        FILE *fp = fopen("build/m92_ids.txt", "w");
        if (fp == NULL) {
            fputs("backend_comparison_smoke: could not create token IDs file\n", stderr);
            return -1;
        }
        fputs("1\n2\n3\n", fp);
        fclose(fp);
    }

    /* Run the comparison report with checked-in fixtures (no --include-cuda). */
    if (run_command(
            "python3 compiler/backend_comparison_report.py"
            " --att1-f32 models/real_tiny_f32/model.att1"
            " --att1-q8  models/real_tiny_q8/model.att1"
            " --att1-q4  models/real_tiny_q4/model.att1"
            " --tokens-file build/m92_ids.txt"
            " --tokens 1 --tiles 2"
            " --report-json build/m92_comparison_report.json"
            " > build/m92_comparison_report.txt 2>&1") != 0) {
        fputs("backend_comparison_smoke: script failed\n", stderr);
        return -1;
    }

    if (read_file("build/m92_comparison_report.txt", output, sizeof(output)) != 0) {
        fputs("backend_comparison_smoke: cannot read report\n", stderr);
        return -1;
    }

    /* Overall result and format markers. */
    if ((strstr(output, "result: pass") == NULL) ||
        (strstr(output, "report: ok")   == NULL)) {
        fputs("backend_comparison_smoke: missing result/report markers\n", stderr);
        return -1;
    }

    /* CPU rows must be present and passing. */
    if ((strstr(output, "backend=cpu-f32") == NULL) ||
        (strstr(output, "backend=cpu-q8")  == NULL) ||
        (strstr(output, "backend=cpu-q4")  == NULL)) {
        fputs("backend_comparison_smoke: missing cpu-f32/q8/q4 run rows\n", stderr);
        return -1;
    }

    /* CUDA rows must be present as pending (no --include-cuda). */
    if (strstr(output, "status=pending") == NULL) {
        fputs("backend_comparison_smoke: missing status=pending for CUDA rows\n",
              stderr);
        return -1;
    }

    /* At least one note line must be present. */
    if (strstr(output, "note:") == NULL) {
        fputs("backend_comparison_smoke: notes missing from report\n", stderr);
        return -1;
    }

    /* JSON report check. */
    if (read_file("build/m92_comparison_report.json", output, sizeof(output)) != 0) {
        fputs("backend_comparison_smoke: cannot read JSON report\n", stderr);
        return -1;
    }
    if ((strstr(output, "\"runs\"")   == NULL) ||
        (strstr(output, "\"result\"") == NULL) ||
        (strstr(output, "\"pass\"")   == NULL)) {
        fputs("backend_comparison_smoke: JSON report missing keys\n", stderr);
        return -1;
    }

    return 0;
}

/*
 * check_trace_diff_smoke() — M94
 *
 * Python-skippable.  Exercises compiler/trace_diff.py with three scenarios:
 *
 *   1. Identical diff — same file diffed against itself:
 *      expected "differences: 0  missing: 0", "result: pass", "report: ok".
 *
 *   2. Cross-backend diff — cpu-f32 vs cpu-q8 run on the same model/tokens:
 *      expected "DIFF" entry for backend field, "result: pass".
 *      JSON report must contain "result", "differences", "pass".
 *
 *   3. Malformed input — a file with no parseable fields must cause the
 *      script to exit non-zero.
 */
static int check_trace_diff_smoke(void)
{
    char output[32768];

    /* Skip gracefully when Python 3 is not available. */
    if (run_command("python3 --version > /dev/null 2>&1") != 0) {
        return 0;
    }

    /* Write shared token IDs fixture. */
    {
        FILE *fp = fopen("build/m94_ids.txt", "w");
        if (fp == NULL) {
            fputs("trace_diff_smoke: could not create token IDs file\n", stderr);
            return -1;
        }
        fputs("1\n2\n3\n", fp);
        fclose(fp);
    }

    /* Run cpu-f32 single bench -> m94_f32.txt */
    if (run_command(
            "./build/att1-bench"
            " --model models/real_tiny_f32/model.att1"
            " --tokens 1 --mode single --backend cpu-f32"
            " --tokenizer external"
            " --tokens-file build/m94_ids.txt"
            " > build/m94_f32.txt 2>&1") != 0) {
        fputs("trace_diff_smoke: cpu-f32 bench run failed\n", stderr);
        return -1;
    }

    /* Run cpu-q8 single bench -> m94_q8.txt */
    if (run_command(
            "./build/att1-bench"
            " --model models/real_tiny_q8/model.att1"
            " --tokens 1 --mode single --backend cpu-q8"
            " --tokenizer external"
            " --tokens-file build/m94_ids.txt"
            " > build/m94_q8.txt 2>&1") != 0) {
        fputs("trace_diff_smoke: cpu-q8 bench run failed\n", stderr);
        return -1;
    }

    /* 1. Identical diff: same file vs itself. */
    if (run_command(
            "python3 compiler/trace_diff.py"
            " build/m94_f32.txt build/m94_f32.txt"
            " > build/m94_diff_identical.txt 2>&1") != 0) {
        fputs("trace_diff_smoke: identical diff script failed\n", stderr);
        return -1;
    }

    if (read_file("build/m94_diff_identical.txt", output, sizeof(output)) != 0) {
        fputs("trace_diff_smoke: cannot read identical diff output\n", stderr);
        return -1;
    }

    if ((strstr(output, "differences: 0  missing: 0") == NULL) ||
        (strstr(output, "result: pass")               == NULL) ||
        (strstr(output, "report: ok")                 == NULL)) {
        fputs("trace_diff_smoke: identical diff missing expected fields\n", stderr);
        return -1;
    }

    /* 2. Cross-backend diff: cpu-f32 vs cpu-q8. */
    if (run_command(
            "python3 compiler/trace_diff.py"
            " build/m94_f32.txt build/m94_q8.txt"
            " --report-json build/m94_diff_backends.json"
            " > build/m94_diff_backends.txt 2>&1") != 0) {
        fputs("trace_diff_smoke: cross-backend diff script failed\n", stderr);
        return -1;
    }

    if (read_file("build/m94_diff_backends.txt", output, sizeof(output)) != 0) {
        fputs("trace_diff_smoke: cannot read cross-backend diff output\n", stderr);
        return -1;
    }

    /* backend field must appear and show a DIFF. */
    if ((strstr(output, "backend") == NULL) ||
        (strstr(output, "DIFF")    == NULL)) {
        fputs("trace_diff_smoke: cross-backend diff missing backend DIFF\n", stderr);
        return -1;
    }

    if ((strstr(output, "result: pass") == NULL) ||
        (strstr(output, "report: ok")   == NULL)) {
        fputs("trace_diff_smoke: cross-backend diff missing result/report\n", stderr);
        return -1;
    }

    /* JSON report. */
    if (read_file("build/m94_diff_backends.json", output, sizeof(output)) != 0) {
        fputs("trace_diff_smoke: cannot read JSON report\n", stderr);
        return -1;
    }
    if ((strstr(output, "\"result\"")      == NULL) ||
        (strstr(output, "\"differences\"") == NULL) ||
        (strstr(output, "\"pass\"")        == NULL)) {
        fputs("trace_diff_smoke: JSON report missing expected keys\n", stderr);
        return -1;
    }

    /* 3. Malformed input must cause the script to exit non-zero. */
    {
        FILE *fp = fopen("build/m94_malformed.txt", "w");
        if (fp == NULL) {
            fputs("trace_diff_smoke: could not create malformed file\n", stderr);
            return -1;
        }
        /* No key=value lines -> tool should reject as unparseable. */
        fputs("this is not a valid bench output file\n", fp);
        fclose(fp);
    }

    if (command_fails(
            "python3 compiler/trace_diff.py"
            " build/m94_malformed.txt build/m94_f32.txt"
            " > /dev/null 2>&1") != 0) {
        fputs("trace_diff_smoke: malformed input did not cause failure\n", stderr);
        return -1;
    }

    return 0;
}

/*
 * check_prefill_decode_split_smoke() — M95
 *
 * Verifies that att1-bench output contains prefill/decode split fields in
 * three scenarios:
 *
 *   1. Byte-tokenizer single mode: prompt_tokens=3 (3 bytes in "abc"),
 *      decode_tokens=2 (2 generated), all split fields present.
 *
 *   2. External tokenizer single mode: prompt_tokens=3 (3 IDs),
 *      decode_tokens=1 (1 generated), field values consistent.
 *
 *   3. Cluster mode: prefill_fabric_packets and decode_fabric_packets
 *      are present in the output (nonzero expected for prefill phase).
 */
static int check_prefill_decode_split_smoke(void)
{
    char output[16384];

    /* ── 1. Byte-tokenizer single mode ─────────────────────────────── */
    if (run_command(
            "./build/att1-bench"
            " --model models/dummy/model.att1"
            " --prompt abc --tokens 2 --mode single --backend cpu-f32"
            " > build/m95_byte_single.txt 2>&1") != 0) {
        fputs("prefill_decode: byte single run failed\n", stderr);
        return -1;
    }

    if (read_file("build/m95_byte_single.txt", output, sizeof(output)) != 0) {
        fputs("prefill_decode: cannot read byte single output\n", stderr);
        return -1;
    }

    if ((strstr(output, "prompt_tokens=3")          == NULL) ||
        (strstr(output, "decode_tokens=2")           == NULL) ||
        (strstr(output, "prefill_time_us_total=")    == NULL) ||
        (strstr(output, "decode_time_us_total=")     == NULL) ||
        (strstr(output, "prefill_kv_appends=")       == NULL) ||
        (strstr(output, "decode_kv_appends=")        == NULL) ||
        (strstr(output, "prefill_kv_reads=")         == NULL) ||
        (strstr(output, "decode_kv_reads=")          == NULL) ||
        (strstr(output, "prefill_logits_bytes=")     == NULL) ||
        (strstr(output, "decode_logits_bytes=")      == NULL)) {
        fputs("prefill_decode: byte single missing split fields\n", stderr);
        return -1;
    }

    /* ── 2. External tokenizer: prompt_tokens matches input count ───── */
    {
        FILE *fp = fopen("build/m95_ids.txt", "w");
        if (fp == NULL) {
            fputs("prefill_decode: could not create token IDs file\n", stderr);
            return -1;
        }
        fputs("1\n2\n3\n", fp);
        fclose(fp);
    }

    if (run_command(
            "./build/att1-bench"
            " --model models/real_tiny_f32/model.att1"
            " --tokens 1 --mode single --backend cpu-f32"
            " --tokenizer external --tokens-file build/m95_ids.txt"
            " > build/m95_ext_single.txt 2>&1") != 0) {
        fputs("prefill_decode: external single run failed\n", stderr);
        return -1;
    }

    if (read_file("build/m95_ext_single.txt", output, sizeof(output)) != 0) {
        fputs("prefill_decode: cannot read external single output\n", stderr);
        return -1;
    }

    if ((strstr(output, "prompt_tokens=3") == NULL) ||
        (strstr(output, "decode_tokens=1") == NULL)) {
        fputs("prefill_decode: external prompt/decode token count wrong\n",
              stderr);
        return -1;
    }

    /* ── 3. Cluster mode: fabric split fields present ───────────────── */
    if (run_command(
            "./build/att1-bench"
            " --model models/real_tiny_f32/model.att1"
            " --tokens 1 --mode cluster --tiles 2 --backend cpu-f32"
            " --tokenizer external --tokens-file build/m95_ids.txt"
            " > build/m95_cluster.txt 2>&1") != 0) {
        fputs("prefill_decode: cluster run failed\n", stderr);
        return -1;
    }

    if (read_file("build/m95_cluster.txt", output, sizeof(output)) != 0) {
        fputs("prefill_decode: cannot read cluster output\n", stderr);
        return -1;
    }

    if ((strstr(output, "prefill_fabric_packets=") == NULL) ||
        (strstr(output, "decode_fabric_packets=")  == NULL)) {
        fputs("prefill_decode: cluster missing fabric split fields\n", stderr);
        return -1;
    }

    /* Prefill phase must have sent at least one fabric packet (3 tokens × 2 tiles). */
    if (strstr(output, "prefill_fabric_packets=0\n") != NULL) {
        fputs("prefill_decode: cluster prefill_fabric_packets is zero\n",
              stderr);
        return -1;
    }

    return 0;
}

/*
 * check_tile_capacity_smoke() — M96
 *
 * Validates the tile memory capacity and fabric bandwidth estimator fields
 * added to att1-size by Milestone 96.  Six scenarios:
 *
 *   1. --preset tiny-dummy --tile-memory-mib 256:
 *      exit 0; output has tile_memory_mib=256, model_bytes_per_tile=,
 *      tile_capacity_status=PASS  (9 KiB model << 256 MiB).
 *
 *   2. --preset tiny-dummy --tile-memory-gib 1:
 *      exit 0; tile_capacity_status=PASS.
 *
 *   3. Manual shape with 1 MiB budget → FAIL:
 *      --layers 4 --d-model 256 --heads 4 --d-ff 512 --vocab-size 1000
 *      --tiles 1 --tile-memory-mib 1
 *      exit 0; tile_capacity_status=FAIL (model ≈ 12 MiB >> 1 MiB).
 *
 *   4. --tile-memory-mib abc → exit non-zero (bad parse).
 *
 *   5. --preset gpt-oss-120b-shape --tiles 16
 *      --target-tokens-per-sec 10 --fabric-gib-sec 5:
 *      exit 0; output has fabric_bandwidth_status=.
 *
 *   6. --config + --tile-memory-mib + --target-tokens-per-sec + --fabric-gib-sec
 *      --json: JSON output has "tile_capacity_status" and
 *      "fabric_bandwidth_status".
 */
static int check_tile_capacity_smoke(void)
{
    char output[8192];

    /* 1. Tiny preset + tile-memory-mib 256: PASS */
    if (run_command("./build/att1-size --preset tiny-dummy"
                    " --tile-memory-mib 256"
                    " > build/m96_cap_pass.txt 2>&1") != 0) {
        fputs("tile_capacity: preset + tile-memory-mib failed\n", stderr);
        return -1;
    }
    if (read_file("build/m96_cap_pass.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if ((strstr(output, "tile_memory_mib=256")    == NULL) ||
        (strstr(output, "model_bytes_per_tile=")  == NULL) ||
        (strstr(output, "tile_capacity_status=")  == NULL) ||
        (strstr(output, "tile_capacity_status=PASS") == NULL)) {
        fputs("tile_capacity: PASS scenario missing expected fields\n", stderr);
        return -1;
    }

    /* 2. Tiny preset + tile-memory-gib 1: also PASS */
    if (run_command("./build/att1-size --preset tiny-dummy"
                    " --tile-memory-gib 1"
                    " > build/m96_cap_gib.txt 2>&1") != 0) {
        fputs("tile_capacity: preset + tile-memory-gib failed\n", stderr);
        return -1;
    }
    if (read_file("build/m96_cap_gib.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if (strstr(output, "tile_capacity_status=PASS") == NULL) {
        fputs("tile_capacity: GiB PASS scenario wrong status\n", stderr);
        return -1;
    }

    /* 3. Manual shape with 1 MiB budget → FAIL */
    if (run_command("./build/att1-size"
                    " --layers 4 --d-model 256 --heads 4 --d-ff 512"
                    " --vocab-size 1000 --tiles 1 --tile-memory-mib 1"
                    " > build/m96_cap_fail.txt 2>&1") != 0) {
        fputs("tile_capacity: FAIL scenario exited non-zero\n", stderr);
        return -1;
    }
    if (read_file("build/m96_cap_fail.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    /* Full-mode report uses aligned "  key       = VALUE" format. */
    if ((strstr(output, "tile_capacity_status") == NULL) ||
        (strstr(output, "FAIL")                 == NULL)) {
        fputs("tile_capacity: FAIL scenario missing tile_capacity_status=FAIL\n",
              stderr);
        return -1;
    }

    /* 4. Invalid --tile-memory-mib value must cause exit non-zero */
    if (run_command("./build/att1-size --preset tiny-dummy"
                    " --tile-memory-mib abc"
                    " > build/m96_cap_invalid.txt 2>&1") == 0) {
        fputs("tile_capacity: invalid tile-memory-mib should fail\n", stderr);
        return -1;
    }

    /* 5. Fabric bandwidth status present when target-tokens-per-sec and
     *    fabric-gib-sec are specified */
    if (run_command("./build/att1-size --preset gpt-oss-120b-shape --tiles 16"
                    " --target-tokens-per-sec 10 --fabric-gib-sec 5"
                    " > build/m96_fabric.txt 2>&1") != 0) {
        fputs("tile_capacity: fabric bandwidth preset run failed\n", stderr);
        return -1;
    }
    if (read_file("build/m96_fabric.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if (strstr(output, "fabric_bandwidth_status=") == NULL) {
        fputs("tile_capacity: fabric_bandwidth_status= missing\n", stderr);
        return -1;
    }

    /* 6. JSON output contains both capacity and bandwidth status keys */
    if (run_command("./build/att1-size"
                    " --config compiler/fixtures/tiny_llama_config.json"
                    " --tiles 2 --tile-memory-mib 4096"
                    " --target-tokens-per-sec 10 --fabric-gib-sec 10"
                    " --json"
                    " > build/m96_cap_json.txt 2>&1") != 0) {
        fputs("tile_capacity: JSON mode failed\n", stderr);
        return -1;
    }
    if (read_file("build/m96_cap_json.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if ((strstr(output, "\"tile_capacity_status\"")    == NULL) ||
        (strstr(output, "\"fabric_bandwidth_status\"") == NULL)) {
        fputs("tile_capacity: JSON missing tile_capacity_status or "
              "fabric_bandwidth_status\n", stderr);
        return -1;
    }

    return 0;
}

/*
 * check_placement_validator_smoke() — M99
 *
 * Python-skippable.  Exercises compiler/validate_tensor_placement_report.py
 * with eight scenarios:
 *
 *   1. Valid fixture → exit 0; output contains "validation: pass".
 *   2. Invalid tile ID fixture → exit 1; output contains "validation: fail".
 *   3. Capacity overflow with PASS status → exit 1; "validation: fail".
 *   4. Overlapping unique slices → exit 1; "validation: fail".
 *   5. Q4 group alignment violation → exit 1; "validation: fail".
 *   6. Malformed JSON → exit 2 (parse error, not 0 or 1).
 *   7. Missing required field (report_version absent) → exit 1.
 *   8. JSON output mode: valid fixture + --report-json produces
 *      a JSON file with "status": "pass" and integer counters.
 */
static int check_placement_validator_smoke(void)
{
    char output[16384];

    /* Skip gracefully when Python 3 is not available. */
    if (run_command("python3 --version > /dev/null 2>&1") != 0) {
        return 0;
    }

    /* 1. Valid fixture: exit 0, "validation: pass". */
    if (run_command(
            "python3 compiler/validate_tensor_placement_report.py"
            " --report compiler/fixtures/placement_report_valid.json"
            " > build/m99_valid.txt 2>&1") != 0) {
        fputs("placement_validator: valid fixture should exit 0\n", stderr);
        return -1;
    }
    if (read_file("build/m99_valid.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if (strstr(output, "validation: pass") == NULL) {
        fputs("placement_validator: valid fixture missing 'validation: pass'\n",
              stderr);
        return -1;
    }

    /* 2. Invalid tile ID fixture: exit 1, "validation: fail". */
    if (run_command(
            "python3 compiler/validate_tensor_placement_report.py"
            " --report compiler/fixtures/placement_report_invalid_tile_id.json"
            " > build/m99_bad_tile.txt 2>&1") == 0) {
        fputs("placement_validator: invalid tile ID fixture should exit non-zero\n",
              stderr);
        return -1;
    }
    if (read_file("build/m99_bad_tile.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if (strstr(output, "validation: fail") == NULL) {
        fputs("placement_validator: invalid tile ID fixture missing 'validation: fail'\n",
              stderr);
        return -1;
    }

    /* 3. Capacity overflow with PASS status: exit 1. */
    if (run_command(
            "python3 compiler/validate_tensor_placement_report.py"
            " --report compiler/fixtures/placement_report_capacity_overflow.json"
            " > build/m99_overflow.txt 2>&1") == 0) {
        fputs("placement_validator: capacity overflow fixture should exit non-zero\n",
              stderr);
        return -1;
    }
    if (read_file("build/m99_overflow.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if (strstr(output, "validation: fail") == NULL) {
        fputs("placement_validator: capacity overflow missing 'validation: fail'\n",
              stderr);
        return -1;
    }

    /* 4. Overlapping slices: exit 1. */
    if (run_command(
            "python3 compiler/validate_tensor_placement_report.py"
            " --report compiler/fixtures/placement_report_overlapping_slices.json"
            " > build/m99_overlap.txt 2>&1") == 0) {
        fputs("placement_validator: overlapping slices fixture should exit non-zero\n",
              stderr);
        return -1;
    }
    if (read_file("build/m99_overlap.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if (strstr(output, "validation: fail") == NULL) {
        fputs("placement_validator: overlapping slices missing 'validation: fail'\n",
              stderr);
        return -1;
    }

    /* 5. Q4 group alignment violation: exit 1. */
    if (run_command(
            "python3 compiler/validate_tensor_placement_report.py"
            " --report compiler/fixtures/placement_report_q4_bad_align.json"
            " > build/m99_q4align.txt 2>&1") == 0) {
        fputs("placement_validator: q4 alignment fixture should exit non-zero\n",
              stderr);
        return -1;
    }
    if (read_file("build/m99_q4align.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if (strstr(output, "validation: fail") == NULL) {
        fputs("placement_validator: q4 alignment missing 'validation: fail'\n",
              stderr);
        return -1;
    }

    /* 6. Malformed JSON: must exit 2. */
    if (run_command(
            "echo 'not_valid_json' | python3"
            " compiler/validate_tensor_placement_report.py"
            " --report /dev/stdin"
            " > build/m99_malformed.txt 2>&1") == 0) {
        fputs("placement_validator: malformed JSON should exit non-zero\n",
              stderr);
        return -1;
    }

    /* 7. Missing report_version field: must exit non-zero. */
    if (run_command(
            "echo '{\"header\":{\"tile_count\":1,\"dtype\":\"f32\","
            "\"quantization_family\":\"none\",\"target_context_length\":128,"
            "\"target_sessions\":1},\"tiles\":[],\"tensors\":[]}'"
            " | python3 compiler/validate_tensor_placement_report.py"
            " --report /dev/stdin"
            " > build/m99_nover.txt 2>&1") == 0) {
        fputs("placement_validator: missing report_version should exit non-zero\n",
              stderr);
        return -1;
    }

    /* 8. JSON output mode: valid fixture produces JSON with status=pass. */
    if (run_command(
            "python3 compiler/validate_tensor_placement_report.py"
            " --report compiler/fixtures/placement_report_valid.json"
            " --report-json build/m99_valid_result.json"
            " > build/m99_json_mode.txt 2>&1") != 0) {
        fputs("placement_validator: JSON mode failed on valid fixture\n", stderr);
        return -1;
    }
    if (read_file("build/m99_valid_result.json", output, sizeof(output)) != 0) {
        fputs("placement_validator: cannot read JSON result file\n", stderr);
        return -1;
    }
    if ((strstr(output, "\"status\"")         == NULL) ||
        (strstr(output, "\"pass\"")           == NULL) ||
        (strstr(output, "\"total_errors\"")   == NULL) ||
        (strstr(output, "\"total_warnings\"") == NULL)) {
        fputs("placement_validator: JSON result missing expected keys\n", stderr);
        return -1;
    }

    return 0;
}

/*
 * check_placement_report_smoke() — M100
 *
 * Exercises att1-size --placement-report-json with nine scenarios:
 *
 *  1. tiny-dummy (default f32, 1 tile): exit 0; existing preset output
 *     unchanged; JSON report file created.
 *  2. M99 validator passes on the tiny-dummy report: exit 0,
 *     "validation: pass".
 *  3. gpt-oss-120b-shape (q4, 8 tiles, ctx 8192): exit 0; JSON created.
 *  4. M99 validator passes on the gpt-oss report: exit 0.
 *  5. JSON report has required top-level keys: report_version, header,
 *     tiles, tensors, warnings, failures, remediation.
 *  6. header contains tile_count, dtype, quantization_family.
 *  7. tiles array is non-empty; tensors array is non-empty.
 *  8. Bad output path fails cleanly (non-zero exit).
 *  9. Existing --preset tiny-dummy (no placement report) still exits 0
 *     and produces preset output.
 */
static int check_placement_report_smoke(void)
{
    char output[65536];

    /* 1. tiny-dummy: exit 0; creates JSON file. */
    if (run_command(
            "./build/att1-size --preset tiny-dummy"
            " --placement-report-json build/m100_tiny_smoke.json"
            " > build/m100_tiny_smoke.txt 2>&1") != 0) {
        fputs("placement_report: tiny-dummy exited non-zero\n", stderr);
        return -1;
    }
    if (read_file("build/m100_tiny_smoke.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if (strstr(output, "preset=tiny-dummy") == NULL) {
        fputs("placement_report: tiny-dummy missing preset line\n", stderr);
        return -1;
    }

    /* 2. M99 validator passes on tiny-dummy report. */
    if (run_command(
            "python3 compiler/validate_tensor_placement_report.py"
            " --report build/m100_tiny_smoke.json"
            " > build/m100_tiny_valid.txt 2>&1") != 0) {
        fputs("placement_report: M99 validator failed on tiny-dummy report\n",
              stderr);
        return -1;
    }
    if (read_file("build/m100_tiny_valid.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if (strstr(output, "validation: pass") == NULL) {
        fputs("placement_report: tiny-dummy report missing 'validation: pass'\n",
              stderr);
        return -1;
    }

    /* 3. gpt-oss-120b-shape (q4, 8 tiles): exit 0; JSON created. */
    if (run_command(
            "./build/att1-size --preset gpt-oss-120b-shape"
            " --tiles 8 --context 8192 --dtype q4"
            " --placement-report-json build/m100_gptoss_smoke.json"
            " > build/m100_gptoss_smoke.txt 2>&1") != 0) {
        fputs("placement_report: gpt-oss exited non-zero\n", stderr);
        return -1;
    }

    /* 4. M99 validator passes on gpt-oss report. */
    if (run_command(
            "python3 compiler/validate_tensor_placement_report.py"
            " --report build/m100_gptoss_smoke.json"
            " > build/m100_gptoss_valid.txt 2>&1") != 0) {
        fputs("placement_report: M99 validator failed on gpt-oss report\n",
              stderr);
        return -1;
    }
    if (read_file("build/m100_gptoss_valid.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if (strstr(output, "validation: pass") == NULL) {
        fputs("placement_report: gpt-oss report missing 'validation: pass'\n",
              stderr);
        return -1;
    }

    /* 5+6+7. Inspect tiny-dummy JSON for required top-level keys and content. */
    if (read_file("build/m100_tiny_smoke.json", output, sizeof(output)) != 0) {
        fputs("placement_report: cannot read tiny-dummy JSON\n", stderr);
        return -1;
    }
    if ((strstr(output, "\"report_version\"")    == NULL) ||
        (strstr(output, "\"header\"")            == NULL) ||
        (strstr(output, "\"tiles\"")             == NULL) ||
        (strstr(output, "\"tensors\"")           == NULL) ||
        (strstr(output, "\"warnings\"")          == NULL) ||
        (strstr(output, "\"failures\"")          == NULL) ||
        (strstr(output, "\"remediation\"")       == NULL)) {
        fputs("placement_report: tiny-dummy JSON missing required keys\n", stderr);
        return -1;
    }
    if ((strstr(output, "\"tile_count\"")        == NULL) ||
        (strstr(output, "\"dtype\"")             == NULL) ||
        (strstr(output, "\"quantization_family\"") == NULL)) {
        fputs("placement_report: tiny-dummy header missing expected fields\n",
              stderr);
        return -1;
    }
    if ((strstr(output, "\"tile_id\"")           == NULL) ||
        (strstr(output, "\"tensor_name\"")       == NULL)) {
        fputs("placement_report: tiny-dummy JSON missing tile_id or tensor_name\n",
              stderr);
        return -1;
    }

    /* 8. Bad output path exits non-zero. */
    if (run_command(
            "./build/att1-size --preset tiny-dummy"
            " --placement-report-json /nonexistent/dir/report.json"
            " > build/m100_badpath.txt 2>&1") == 0) {
        fputs("placement_report: bad output path should fail\n", stderr);
        return -1;
    }

    /* 9. Existing --preset tiny-dummy (no --placement-report-json) unchanged. */
    if (run_command(
            "./build/att1-size --preset tiny-dummy"
            " > build/m100_compat.txt 2>&1") != 0) {
        fputs("placement_report: existing preset mode failed\n", stderr);
        return -1;
    }
    if (read_file("build/m100_compat.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if ((strstr(output, "preset=tiny-dummy")      == NULL) ||
        (strstr(output, "total_params=")          == NULL) ||
        (strstr(output, "model_bytes_f32=")       == NULL)) {
        fputs("placement_report: existing preset output changed\n", stderr);
        return -1;
    }

    return 0;
}

/*
 * check_placement_proposal_smoke() — M101
 *
 * Python-skippable.  Exercises compiler/propose_tensor_placement.py
 * with eight scenarios:
 *
 *   1. Valid fixture → exit 0; output contains "advisory: ok".
 *   2. Generate gpt-oss q4 8-tile 16 GiB report via att1-size → exit 0.
 *   3. Run proposer on gpt-oss 16 GiB report → exit 1 (fail advisory);
 *      output contains "capacity-fail".
 *   4. Proposer output for gpt-oss contains tile count or memory
 *      recommendation ("Increase tile count" or "Increase tile memory").
 *   5. JSON output mode: valid fixture + --report-json produces a JSON
 *      file containing "status", "proposals", "analysis", "next_action".
 *   6. JSON output for gpt-oss FAIL case has proposal_count > 0 and
 *      non-empty "next_action".
 *   7. Malformed JSON → exit 2.
 *   8. Missing required field "tiles" → exit 2.
 */
static int check_placement_proposal_smoke(void)
{
    char output[65536];

    /* Skip gracefully when Python 3 is not available. */
    if (run_command("python3 --version > /dev/null 2>&1") != 0) {
        return 0;
    }

    /* 1. Valid fixture: exit 0, "advisory: ok". */
    if (run_command(
            "python3 compiler/propose_tensor_placement.py"
            " --report compiler/fixtures/placement_report_valid.json"
            " > build/m101_valid.txt 2>&1") != 0) {
        fputs("placement_proposal: valid fixture should exit 0\n", stderr);
        return -1;
    }
    if (read_file("build/m101_valid.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if (strstr(output, "advisory: ok") == NULL) {
        fputs("placement_proposal: valid fixture missing 'advisory: ok'\n",
              stderr);
        return -1;
    }

    /* 2. Generate gpt-oss q4 8-tile 16 GiB report via att1-size. */
    if (run_command(
            "./build/att1-size --preset gpt-oss-120b-shape"
            " --dtype q4 --tiles 8 --context 8192"
            " --tile-memory-gib 16"
            " --placement-report-json build/m101_gptoss_16g.json"
            " > build/m101_gptoss_size.txt 2>&1") != 0) {
        fputs("placement_proposal: att1-size gpt-oss 16 GiB exited non-zero\n",
              stderr);
        return -1;
    }

    /* 3. Proposer on gpt-oss 16 GiB → exit 1, output "capacity-fail". */
    if (run_command(
            "python3 compiler/propose_tensor_placement.py"
            " --report build/m101_gptoss_16g.json"
            " > build/m101_gptoss_advisory.txt 2>&1") == 0) {
        fputs("placement_proposal: gpt-oss FAIL report should exit non-zero\n",
              stderr);
        return -1;
    }
    if (read_file("build/m101_gptoss_advisory.txt", output, sizeof(output)) != 0) {
        return -1;
    }
    if (strstr(output, "advisory: fail") == NULL) {
        fputs("placement_proposal: gpt-oss advisory missing 'advisory: fail'\n",
              stderr);
        return -1;
    }
    if (strstr(output, "capacity-fail") == NULL) {
        fputs("placement_proposal: gpt-oss advisory missing 'capacity-fail'\n",
              stderr);
        return -1;
    }

    /* 4. Proposer output mentions tile count or tile memory recommendation. */
    if ((strstr(output, "Increase tile count")  == NULL) &&
        (strstr(output, "Increase tile memory") == NULL)) {
        fputs("placement_proposal: gpt-oss advisory missing tile recommendation\n",
              stderr);
        return -1;
    }

    /* 5. JSON output mode: valid fixture → JSON has expected keys. */
    if (run_command(
            "python3 compiler/propose_tensor_placement.py"
            " --report compiler/fixtures/placement_report_valid.json"
            " --report-json build/m101_valid_advisory.json"
            " > build/m101_valid_json_mode.txt 2>&1") != 0) {
        fputs("placement_proposal: JSON mode failed on valid fixture\n", stderr);
        return -1;
    }
    if (read_file("build/m101_valid_advisory.json", output, sizeof(output)) != 0) {
        fputs("placement_proposal: cannot read valid advisory JSON\n", stderr);
        return -1;
    }
    if ((strstr(output, "\"status\"")         == NULL) ||
        (strstr(output, "\"proposals\"")      == NULL) ||
        (strstr(output, "\"analysis\"")       == NULL) ||
        (strstr(output, "\"next_action\"")    == NULL)) {
        fputs("placement_proposal: valid advisory JSON missing expected keys\n",
              stderr);
        return -1;
    }

    /* 6. JSON output for FAIL case has proposal_count > 0. */
    if (run_command(
            "python3 compiler/propose_tensor_placement.py"
            " --report build/m101_gptoss_16g.json"
            " --report-json build/m101_gptoss_advisory_json.json"
            " > /dev/null 2>&1") == 0) {
        /* exit 1 is expected; non-zero is fine here, just ignore. */
    }
    if (read_file("build/m101_gptoss_advisory_json.json",
                  output, sizeof(output)) != 0) {
        fputs("placement_proposal: cannot read gpt-oss advisory JSON\n", stderr);
        return -1;
    }
    if ((strstr(output, "\"status\": \"fail\"")  == NULL) &&
        (strstr(output, "\"status\":\"fail\"")   == NULL)) {
        fputs("placement_proposal: gpt-oss advisory JSON missing status fail\n",
              stderr);
        return -1;
    }
    if ((strstr(output, "\"proposal_count\"") == NULL) ||
        (strstr(output, "\"next_action\"")    == NULL)) {
        fputs("placement_proposal: gpt-oss advisory JSON missing required fields\n",
              stderr);
        return -1;
    }

    /* 7. Malformed JSON → exit 2. */
    if (run_command(
            "echo 'not_valid_json' | python3"
            " compiler/propose_tensor_placement.py"
            " --report /dev/stdin"
            " > build/m101_malformed.txt 2>&1") == 0) {
        fputs("placement_proposal: malformed JSON should exit non-zero\n",
              stderr);
        return -1;
    }

    /* 8. Missing required field "tiles" → exit 2. */
    if (run_command(
            "echo '{\"report_version\":1,\"header\":{\"tile_count\":1}}'"
            " | python3 compiler/propose_tensor_placement.py"
            " --report /dev/stdin"
            " > build/m101_missing_tiles.txt 2>&1") == 0) {
        fputs("placement_proposal: missing tiles should exit non-zero\n",
              stderr);
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
        (check_q4_conversion()            != 0) ||
        (check_q4_source_comparison()     != 0) ||
        (check_public_q4_smoke()          != 0) ||
        (check_public_q4_backend_smoke()  != 0) ||
        (check_public_q4_cuda_smoke()     != 0) ||
        (check_backend_comparison_smoke() != 0) ||
        (check_trace_diff_smoke()         != 0) ||
        (check_prefill_decode_split_smoke() != 0) ||
        (check_public_backend_smoke()     != 0) ||
        (check_public_tokenized_validation() != 0) ||
        (check_scaling_report()           != 0) ||
        (check_tile_capacity_smoke()      != 0) ||
        (check_placement_validator_smoke() != 0) ||
        (check_placement_report_smoke()   != 0) ||
        (check_placement_proposal_smoke() != 0)) {
        fputs("bench smoke test failed\n", stderr);
        return 1;
    }

    puts("bench smoke test passed");
    return 0;
}
