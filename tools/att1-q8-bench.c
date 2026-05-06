#include "att1_math.h"
#include "att1_quant.h"
#include "att1_trace.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *argv0)
{
    printf("usage: %s [--iterations N]\n", argv0);
}

static int parse_size(const char *text, size_t *out)
{
    char *end = NULL;
    unsigned long value = 0u;

    if ((text == NULL) || (out == NULL)) {
        return -1;
    }

    value = strtoul(text, &end, 10);
    if ((end == text) || (*end != '\0')) {
        return -1;
    }

    *out = (size_t)value;
    return 0;
}

int main(int argc, char **argv)
{
    const size_t lhs_rows = 3u;
    const size_t inner = 8u;
    const size_t outputs = 4u;
    const float lhs[24] = {
        0.25f, -0.50f, 1.00f, 2.00f, -1.50f, 0.75f, 3.00f, -2.00f,
        1.25f, 0.50f, -0.25f, 0.75f, 2.50f, -3.00f, 1.50f, 0.00f,
        -1.00f, 2.00f, 0.50f, -0.75f, 1.25f, 2.25f, -2.50f, 0.50f
    };
    const float weights_out_in[32] = {
        0.10f, -0.20f, 0.30f, -0.40f, 0.50f, -0.60f, 0.70f, -0.80f,
        1.00f, 0.75f, -0.50f, -0.25f, 0.25f, 0.50f, -0.75f, -1.00f,
        -1.20f, 0.00f, 0.40f, 0.80f, -0.40f, -0.80f, 1.20f, 0.20f,
        2.00f, -1.50f, 1.00f, -0.50f, 0.25f, -0.25f, 0.50f, -1.00f
    };
    float weights_in_out[32];
    float f32_out[12];
    float q8_out[12];
    att1_q8_matrix q8;
    uint64_t f32_start = 0u;
    uint64_t q8_start = 0u;
    uint64_t f32_time = 0u;
    uint64_t q8_time = 0u;
    float max_abs_error = 0.0f;
    size_t iterations = 1000u;
    size_t i = 0u;
    size_t row = 0u;
    size_t col = 0u;

    memset(&q8, 0, sizeof(q8));

    for (i = 1u; i < (size_t)argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else if ((strcmp(argv[i], "--iterations") == 0) &&
                   ((i + 1u) < (size_t)argc)) {
            if ((parse_size(argv[++i], &iterations) != 0) ||
                (iterations == 0u)) {
                usage(argv[0]);
                return 1;
            }
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    for (row = 0u; row < inner; row++) {
        for (col = 0u; col < outputs; col++) {
            weights_in_out[(row * outputs) + col] =
                weights_out_in[(col * inner) + row];
        }
    }

    if (att1_quantize_q8_per_row(&q8, weights_out_in, outputs, inner) != 0) {
        fputs("q8 quantization failed\n", stderr);
        return 1;
    }

    f32_start = att1_trace_now_us();
    for (i = 0u; i < iterations; i++) {
        if (att1_matmul_f32(f32_out,
                            lhs,
                            weights_in_out,
                            lhs_rows,
                            outputs,
                            inner) != 0) {
            att1_q8_matrix_free(&q8);
            return 1;
        }
    }
    f32_time = att1_trace_now_us() - f32_start;

    q8_start = att1_trace_now_us();
    for (i = 0u; i < iterations; i++) {
        if (att1_matmul_q8xf32(q8_out, lhs, lhs_rows, inner, &q8) != 0) {
            att1_q8_matrix_free(&q8);
            return 1;
        }
    }
    q8_time = att1_trace_now_us() - q8_start;

    for (i = 0u; i < lhs_rows * outputs; i++) {
        const float error = fabsf(f32_out[i] - q8_out[i]);
        if (error > max_abs_error) {
            max_abs_error = error;
        }
    }

    printf("mode=q8-matmul\n");
    printf("iterations=%zu\n", iterations);
    printf("lhs_rows=%zu\n", lhs_rows);
    printf("inner=%zu\n", inner);
    printf("outputs=%zu\n", outputs);
    printf("f32_time_us_total=%llu\n", (unsigned long long)f32_time);
    printf("q8_time_us_total=%llu\n", (unsigned long long)q8_time);
    printf("max_abs_error=%.8f\n", max_abs_error);
    printf("q8_weight_bytes=%zu\n", q8.rows * q8.cols);
    printf("q8_scale_bytes=%zu\n", q8.rows * sizeof(*q8.scales));

    att1_q8_matrix_free(&q8);
    return 0;
}
