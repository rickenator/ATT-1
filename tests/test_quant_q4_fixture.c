/*
 * test_quant_q4_fixture.c  -  M77: q4 fixture generation and validation tests
 *
 * Validates the checked-in models/q4_tiny/model.att1 fixture produced by
 * compiler/make_q4_fixture.py.
 *
 * Fixture model configuration:
 *   vocab_size=256, n_layers=2, n_heads=2, d_model=32, d_ff=64,
 *   max_seq_len=8, rope_dim=8, n_tiles=1, group_size=32
 *
 * Tensor layout (21 tensors total):
 *   tok_embeddings.weight          [256, 32]  f32
 *   layers.N.attention_norm.weight [32]       f32
 *   layers.N.attention.wq.weight   [32, 32]   q4 g32
 *   layers.N.attention.wk.weight   [32, 32]   q4 g32
 *   layers.N.attention.wv.weight   [32, 32]   q4 g32
 *   layers.N.attention.wo.weight   [32, 32]   q4 g32
 *   layers.N.ffn_norm.weight       [32]       f32
 *   layers.N.ffn.w_gate.weight     [64, 32]   q4 g32   (rows=d_ff, cols=d_model)
 *   layers.N.ffn.w_up.weight       [64, 32]   q4 g32   (rows=d_ff, cols=d_model)
 *   layers.N.ffn.w_down.weight     [32, 64]   q4 g32   (rows=d_model, cols=d_ff)
 *   output_norm.weight             [32]       f32
 *   output.weight                  [256, 32]  q4 g32   (rows=vocab_size, cols=d_model)
 *
 * Expected q4 nbytes (nbytes are shape-product-invariant — same either orientation):
 *   [32, 32] g32 → packed=512,  scales=128,  total=640
 *   [64, 32] g32 → packed=1024, scales=256,  total=1280
 *   [32, 64] g32 → packed=1024, scales=256,  total=1280
 *   [256,32] g32 → packed=4096, scales=1024, total=5120
 *
 * Tests:
 *   1. test_fixture_loads               – ATT1_OK, tensor_count=21
 *   2. test_fixture_q4_tensor_dtypes    – projection tensors are q4
 *   3. test_fixture_f32_tensor_dtypes   – norms and embeddings are f32
 *   4. test_fixture_nbytes              – exact nbytes for known tensors
 *   5. test_fixture_scales_finite       – scale values are finite and > 0
 *   6. test_fixture_packed_nonzero      – packed bytes are not all zero
 *   7. test_fixture_inference_rejected  – validate_decoder → ERR_UNSUPPORTED
 *   8. test_fixture_inspect_output      – inspect prints expected q4 fields
 */

#include "att1_model.h"
#include "att1_model_view.h"
#include "att1_quant.h"
#include "att1_status.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FIXTURE_PATH "models/q4_tiny/model.att1"

/* ── small utilities ─────────────────────────────────────────────────────── */

static int read_file_alloc(const char *path,
                           unsigned char **out_data,
                           size_t *out_size)
{
    FILE *fp = fopen(path, "rb");
    long sz = 0;
    unsigned char *buf = NULL;

    if (fp == NULL) { return -1; }
    if ((fseek(fp, 0L, SEEK_END) != 0) ||
        ((sz = ftell(fp)) < 0) ||
        (fseek(fp, 0L, SEEK_SET) != 0)) {
        fclose(fp); return -1;
    }
    buf = malloc((size_t)sz);
    if ((buf == NULL) && (sz != 0)) { fclose(fp); return -1; }
    if ((sz > 0) && (fread(buf, 1u, (size_t)sz, fp) != (size_t)sz)) {
        free(buf); fclose(fp); return -1;
    }
    fclose(fp);
    *out_data = buf;
    *out_size = (size_t)sz;
    return 0;
}

static int buf_contains(const unsigned char *data, size_t size, const char *needle)
{
    const size_t nlen = strlen(needle);
    size_t i = 0u;
    if (nlen > size) { return 0; }
    for (i = 0u; i <= size - nlen; i++) {
        if (memcmp(&data[i], needle, nlen) == 0) { return 1; }
    }
    return 0;
}

/* ── test 1: fixture loads cleanly ──────────────────────────────────────── */

static int test_fixture_loads(void)
{
    att1_model model;

    if (att1_model_load(FIXTURE_PATH, &model) != ATT1_OK) {
        fputs("fixture_loads: load failed\n", stderr);
        return -1;
    }

    if (model.tensor_count != 21u) {
        fprintf(stderr, "fixture_loads: expected 21 tensors, got %" PRIu64 "\n",
                (uint64_t)model.tensor_count);
        att1_model_free(&model);
        return -1;
    }

    att1_model_free(&model);
    return 0;
}

/* ── test 2: projection tensors are dtype Q4 ────────────────────────────── */

static int test_fixture_q4_tensor_dtypes(void)
{
    att1_model model;
    const char *q4_names[] = {
        "layers.0.attention.wq.weight",
        "layers.0.attention.wk.weight",
        "layers.0.attention.wv.weight",
        "layers.0.attention.wo.weight",
        "layers.0.ffn.w_gate.weight",
        "layers.0.ffn.w_up.weight",
        "layers.0.ffn.w_down.weight",
        "layers.1.attention.wq.weight",
        "layers.1.attention.wk.weight",
        "layers.1.attention.wv.weight",
        "layers.1.attention.wo.weight",
        "layers.1.ffn.w_gate.weight",
        "layers.1.ffn.w_up.weight",
        "layers.1.ffn.w_down.weight",
        "output.weight",
    };
    const size_t n = sizeof(q4_names) / sizeof(q4_names[0]);
    size_t i = 0u;

    if (att1_model_load(FIXTURE_PATH, &model) != ATT1_OK) {
        fputs("q4_tensor_dtypes: load failed\n", stderr);
        return -1;
    }

    for (i = 0u; i < n; i++) {
        const att1_model_tensor *t = att1_model_find_tensor(&model, q4_names[i]);
        if (t == NULL) {
            fprintf(stderr, "q4_tensor_dtypes: tensor %s not found\n", q4_names[i]);
            att1_model_free(&model);
            return -1;
        }
        if (t->dtype != ATT1_MODEL_DTYPE_Q4) {
            fprintf(stderr, "q4_tensor_dtypes: %s dtype=%u expected=%u\n",
                    q4_names[i], t->dtype, ATT1_MODEL_DTYPE_Q4);
            att1_model_free(&model);
            return -1;
        }
        /* group_size=32 stored in flags[7:0] */
        if ((t->flags & ATT1_Q4_FLAGS_GROUP_MASK) != 32u) {
            fprintf(stderr, "q4_tensor_dtypes: %s group_size=%u expected=32\n",
                    q4_names[i],
                    t->flags & ATT1_Q4_FLAGS_GROUP_MASK);
            att1_model_free(&model);
            return -1;
        }
    }

    att1_model_free(&model);
    return 0;
}

/* ── test 3: norm and embedding tensors are f32 ──────────────────────────── */

static int test_fixture_f32_tensor_dtypes(void)
{
    att1_model model;
    const char *f32_names[] = {
        "tok_embeddings.weight",
        "layers.0.attention_norm.weight",
        "layers.0.ffn_norm.weight",
        "layers.1.attention_norm.weight",
        "layers.1.ffn_norm.weight",
        "output_norm.weight",
    };
    const size_t n = sizeof(f32_names) / sizeof(f32_names[0]);
    size_t i = 0u;

    if (att1_model_load(FIXTURE_PATH, &model) != ATT1_OK) {
        fputs("f32_tensor_dtypes: load failed\n", stderr);
        return -1;
    }

    for (i = 0u; i < n; i++) {
        const att1_model_tensor *t = att1_model_find_tensor(&model, f32_names[i]);
        if (t == NULL) {
            fprintf(stderr, "f32_tensor_dtypes: tensor %s not found\n", f32_names[i]);
            att1_model_free(&model);
            return -1;
        }
        if (t->dtype != ATT1_MODEL_DTYPE_F32) {
            fprintf(stderr, "f32_tensor_dtypes: %s dtype=%u expected=%u\n",
                    f32_names[i], t->dtype, ATT1_MODEL_DTYPE_F32);
            att1_model_free(&model);
            return -1;
        }
    }

    att1_model_free(&model);
    return 0;
}

/* ── test 4: exact nbytes for known tensor shapes ───────────────────────── */

static int test_fixture_nbytes(void)
{
    att1_model model;
    const struct {
        const char *name;
        uint64_t    expected_nbytes;
    } cases[] = {
        { "layers.0.attention.wq.weight", 640u   }, /* [32,32]  g32: 512+128 */
        { "layers.0.ffn.w_gate.weight",   1280u  }, /* [32,64]  g32: 1024+256 */
        { "layers.0.ffn.w_down.weight",   1280u  }, /* [64,32]  g32: 1024+256 */
        { "output.weight",                5120u  }, /* [32,256] g32: 4096+1024 */
        { "tok_embeddings.weight",        32768u }, /* [256,32] f32: 256*32*4 */
        { "output_norm.weight",           128u   }, /* [32]     f32: 32*4 */
    };
    const size_t n = sizeof(cases) / sizeof(cases[0]);
    size_t i = 0u;

    if (att1_model_load(FIXTURE_PATH, &model) != ATT1_OK) {
        fputs("fixture_nbytes: load failed\n", stderr);
        return -1;
    }

    for (i = 0u; i < n; i++) {
        const att1_model_tensor *t = att1_model_find_tensor(&model, cases[i].name);
        if (t == NULL) {
            fprintf(stderr, "fixture_nbytes: tensor %s not found\n", cases[i].name);
            att1_model_free(&model);
            return -1;
        }
        if (t->nbytes != cases[i].expected_nbytes) {
            fprintf(stderr,
                    "fixture_nbytes: %s nbytes=%" PRIu64 " expected=%" PRIu64 "\n",
                    cases[i].name,
                    (uint64_t)t->nbytes,
                    (uint64_t)cases[i].expected_nbytes);
            att1_model_free(&model);
            return -1;
        }
    }

    att1_model_free(&model);
    return 0;
}

/* ── test 5: scale values are finite and positive ───────────────────────── */

/*
 * For layers.0.attention.wq.weight [32, 32] g32:
 *   packed_bytes = 32 * 32 / 2 = 512
 *   n_scales     = 32 * (32 / 32) = 32  (one per row)
 *   scales start at data + 512
 */
static int test_fixture_scales_finite(void)
{
    att1_model model;
    const att1_model_tensor *t = NULL;
    const float *scales = NULL;
    const size_t packed_bytes = 512u;
    const size_t n_scales     = 32u;
    size_t i = 0u;

    if (att1_model_load(FIXTURE_PATH, &model) != ATT1_OK) {
        fputs("scales_finite: load failed\n", stderr);
        return -1;
    }

    t = att1_model_find_tensor(&model, "layers.0.attention.wq.weight");
    if (t == NULL) {
        fputs("scales_finite: tensor not found\n", stderr);
        att1_model_free(&model);
        return -1;
    }

    if (t->nbytes < packed_bytes + n_scales * sizeof(float)) {
        fputs("scales_finite: tensor too small\n", stderr);
        att1_model_free(&model);
        return -1;
    }

    /* NOLINTNEXTLINE(clang-analyzer-core.NullDereference) */
    scales = (const float *)((const unsigned char *)t->data + packed_bytes);

    for (i = 0u; i < n_scales; i++) {
        if (!isfinite(scales[i])) {
            fprintf(stderr, "scales_finite: scales[%zu] is not finite\n", i);
            att1_model_free(&model);
            return -1;
        }
        if (scales[i] <= 0.0f) {
            fprintf(stderr, "scales_finite: scales[%zu]=%f is not positive\n",
                    i, (double)scales[i]);
            att1_model_free(&model);
            return -1;
        }
    }

    att1_model_free(&model);
    return 0;
}

/* ── test 6: packed bytes are not all zero ───────────────────────────────── */

/*
 * Deterministic sine-wave weights mean the packed nibbles must contain
 * nonzero int4 values.  At least one byte in the 512-byte packed block
 * for layers.0.attention.wq.weight should be nonzero.
 */
static int test_fixture_packed_nonzero(void)
{
    att1_model model;
    const att1_model_tensor *t = NULL;
    const size_t packed_bytes = 512u;
    size_t i = 0u;
    int found_nonzero = 0;

    if (att1_model_load(FIXTURE_PATH, &model) != ATT1_OK) {
        fputs("packed_nonzero: load failed\n", stderr);
        return -1;
    }

    t = att1_model_find_tensor(&model, "layers.0.attention.wq.weight");
    if (t == NULL) {
        fputs("packed_nonzero: tensor not found\n", stderr);
        att1_model_free(&model);
        return -1;
    }

    {
        const unsigned char *packed = (const unsigned char *)t->data;
        for (i = 0u; i < packed_bytes; i++) {
            if (packed[i] != 0u) {
                found_nonzero = 1;
                break;
            }
        }
    }

    att1_model_free(&model);

    if (!found_nonzero) {
        fputs("packed_nonzero: all packed bytes are zero\n", stderr);
        return -1;
    }
    return 0;
}

/* ── test 7: inference is rejected ──────────────────────────────────────── */

static int test_fixture_inference_rejected(void)
{
    att1_model model;
    att1_status_t rc = ATT1_OK;

    if (att1_model_load(FIXTURE_PATH, &model) != ATT1_OK) {
        fputs("inference_rejected: load failed\n", stderr);
        return -1;
    }

    rc = att1_model_view_validate_decoder(&model);
    att1_model_free(&model);

    if (rc != ATT1_ERR_UNSUPPORTED) {
        fprintf(stderr,
                "inference_rejected: validate_decoder returned %d "
                "(expected ATT1_ERR_UNSUPPORTED=%d)\n",
                (int)rc, (int)ATT1_ERR_UNSUPPORTED);
        return -1;
    }
    return 0;
}

/* ── test 8: inspect output has expected fields ──────────────────────────── */

static int test_fixture_inspect_output(void)
{
    const char *output_path = "build/test_q4_fixture/inspect_out.txt";
    char cmd[512];
    unsigned char *data = NULL;
    size_t size = 0u;
    int rc = 0;
    int ret = 0;

    (void)system("mkdir -p build/test_q4_fixture");

    (void)snprintf(cmd, sizeof(cmd),
                   "build/att1-inspect %s > %s 2>&1",
                   FIXTURE_PATH, output_path);
    if (system(cmd) != 0) {
        fputs("inspect_output: att1-inspect exited nonzero\n", stderr);
        return -1;
    }

    if (read_file_alloc(output_path, &data, &size) != 0) {
        fputs("inspect_output: cannot read inspect output\n", stderr);
        return -1;
    }

    /* Per-tensor q4 fields */
    if (!buf_contains(data, size, "dtype_name=q4")) {
        fputs("inspect_output: missing dtype_name=q4\n", stderr);
        rc = -1;
    }
    if (!buf_contains(data, size, "quant=grouped-q4-g32")) {
        fputs("inspect_output: missing quant=grouped-q4-g32\n", stderr);
        rc = -1;
    }
    /* wq [32,32] g32: n_groups=32, packed=512, scale=128 */
    if (!buf_contains(data, size, "q4_groups=32")) {
        fputs("inspect_output: missing q4_groups=32\n", stderr);
        rc = -1;
    }
    if (!buf_contains(data, size, "q4_packed_bytes=512")) {
        fputs("inspect_output: missing q4_packed_bytes=512\n", stderr);
        rc = -1;
    }
    if (!buf_contains(data, size, "q4_scale_bytes=128")) {
        fputs("inspect_output: missing q4_scale_bytes=128\n", stderr);
        rc = -1;
    }
    /* Model-level inference status note */
    if (!buf_contains(data, size, "inference_status=q4_unsupported")) {
        fputs("inspect_output: missing inference_status=q4_unsupported\n", stderr);
        rc = -1;
    }

    ret = rc;
    free(data);
    return ret;
}

/* ── main ────────────────────────────────────────────────────────────────── */

int main(void)
{
    if (test_fixture_loads() != 0) {
        fputs("FAIL: fixture_loads\n", stderr);
        return 1;
    }
    if (test_fixture_q4_tensor_dtypes() != 0) {
        fputs("FAIL: q4_tensor_dtypes\n", stderr);
        return 1;
    }
    if (test_fixture_f32_tensor_dtypes() != 0) {
        fputs("FAIL: f32_tensor_dtypes\n", stderr);
        return 1;
    }
    if (test_fixture_nbytes() != 0) {
        fputs("FAIL: fixture_nbytes\n", stderr);
        return 1;
    }
    if (test_fixture_scales_finite() != 0) {
        fputs("FAIL: scales_finite\n", stderr);
        return 1;
    }
    if (test_fixture_packed_nonzero() != 0) {
        fputs("FAIL: packed_nonzero\n", stderr);
        return 1;
    }
    if (test_fixture_inference_rejected() != 0) {
        fputs("FAIL: inference_rejected\n", stderr);
        return 1;
    }
    if (test_fixture_inspect_output() != 0) {
        fputs("FAIL: inspect_output\n", stderr);
        return 1;
    }

    puts("quant_q4_fixture test passed");
    return 0;
}
