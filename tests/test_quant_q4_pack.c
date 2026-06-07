/*
 * test_quant_q4_pack.c  -  M75: CPU q4 packing/unpacking primitive tests
 *
 * Covers:
 *   - Two q4 values pack into one byte with the documented nibble order
 *     (low nibble = even index, high nibble = odd index).
 *   - Unpack reverses pack exactly for all int4 values in [-7, 7].
 *   - Signed/unsigned policy: values -7..7 round-trip; -8 is excluded.
 *   - Zero row quantizes/dequantizes safely (scale defaults to 1.0).
 *   - Saturation clamps to [-7, 7] before packing.
 *   - Deterministic medium vector round-trips within documented tolerance.
 *   - Invalid group size fails clearly.
 *   - Null/invalid args fail clearly.
 *   - Existing f32/q8 tests still pass (tested via test_quant, not here).
 *   - Selecting q4 for inference still fails as ATT1_ERR_UNSUPPORTED
 *     (tested via test_quant_q4, not here).
 */

#include "att1_quant.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

/* ── helpers ─────────────────────────────────────────────────────────────── */

static int near_f32(float a, float b, float tol)
{
    return fabsf(a - b) <= tol;
}

/* ── nibble order ─────────────────────────────────────────────────────────── */

/*
 * Pack {lo=-3, hi=+5} into one byte and verify:
 *   packed byte low nibble  = lo & 0x0F = 0xFD & 0x0F = 0x0D
 *   packed byte high nibble = hi & 0x0F = 0x05 << 4   = 0x50
 *   packed = 0x5D
 * Then unpack and check we recover the originals.
 */
static int test_nibble_order(void)
{
    const int8_t src[2]   = { -3, 5 };
    uint8_t packed[1]     = { 0 };
    int8_t  unpacked[2]   = { 0 };
    uint8_t expected_byte = 0u;

    if (att1_q4_pack_group(src, ATT1_Q4_GROUP_SIZE_MIN, NULL) == 0) {
        fputs("nibble_order: null dst should fail\n", stderr);
        return -1;
    }

    /* Build a 16-element source: only first two matter for byte 0 */
    {
        int8_t src16[ATT1_Q4_GROUP_SIZE_MIN];
        uint8_t packed8[ATT1_Q4_GROUP_SIZE_MIN / 2];
        int8_t unpacked16[ATT1_Q4_GROUP_SIZE_MIN];

        memset(src16, 0, sizeof(src16));
        src16[0] = -3;
        src16[1] =  5;

        if (att1_q4_pack_group(src16, ATT1_Q4_GROUP_SIZE_MIN, packed8) != 0) {
            fputs("nibble_order: pack failed\n", stderr);
            return -1;
        }

        /* low nibble of byte 0: -3 in 4-bit two's-complement = 0x0D */
        /* high nibble: 5 = 0x05 -> 0x50 */
        expected_byte = (uint8_t)(((uint8_t)(-3) & 0x0Fu) |
                                  (((uint8_t)(5) & 0x0Fu) << 4u));
        if (packed8[0] != expected_byte) {
            fprintf(stderr, "nibble_order: packed byte 0 = 0x%02X, expected 0x%02X\n",
                    packed8[0], expected_byte);
            return -1;
        }

        if (att1_q4_unpack_group(packed8, ATT1_Q4_GROUP_SIZE_MIN, unpacked16) != 0) {
            fputs("nibble_order: unpack failed\n", stderr);
            return -1;
        }

        if ((unpacked16[0] != -3) || (unpacked16[1] != 5)) {
            fprintf(stderr, "nibble_order: unpacked[0]=%d unpacked[1]=%d expected -3 5\n",
                    (int)unpacked16[0], (int)unpacked16[1]);
            return -1;
        }
    }

    (void)src;
    (void)packed;
    (void)unpacked;
    return 0;
}

/* ── full round-trip for all int4 values ─────────────────────────────────── */

static int test_pack_unpack_all_values(void)
{
    /* One group of 16: values 0..6, -1..-7, repeated to fill 16 */
    int8_t src[ATT1_Q4_GROUP_SIZE_MIN];
    uint8_t packed[ATT1_Q4_GROUP_SIZE_MIN / 2];
    int8_t unpacked[ATT1_Q4_GROUP_SIZE_MIN];
    uint32_t i = 0u;
    /* Populate: alternating positive/negative in range [-7,7] */
    const int8_t vals[16] = {
        0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7, 0
    };

    memcpy(src, vals, sizeof(src));

    if (att1_q4_pack_group(src, ATT1_Q4_GROUP_SIZE_MIN, packed) != 0) {
        fputs("pack_unpack_all: pack failed\n", stderr);
        return -1;
    }

    if (att1_q4_unpack_group(packed, ATT1_Q4_GROUP_SIZE_MIN, unpacked) != 0) {
        fputs("pack_unpack_all: unpack failed\n", stderr);
        return -1;
    }

    for (i = 0u; i < ATT1_Q4_GROUP_SIZE_MIN; i++) {
        if (unpacked[i] != src[i]) {
            fprintf(stderr,
                    "pack_unpack_all: mismatch at %u: src=%d unpacked=%d\n",
                    i, (int)src[i], (int)unpacked[i]);
            return -1;
        }
    }
    return 0;
}

/* ── saturation clamps to [-7, 7] ────────────────────────────────────────── */

static int test_saturation(void)
{
    int8_t src[ATT1_Q4_GROUP_SIZE_MIN];
    uint8_t packed[ATT1_Q4_GROUP_SIZE_MIN / 2];
    int8_t unpacked[ATT1_Q4_GROUP_SIZE_MIN];
    uint32_t i = 0u;

    memset(src, 0, sizeof(src));
    src[0] =  127; /* should clamp to  7 */
    src[1] = -127; /* should clamp to -7 */

    if (att1_q4_pack_group(src, ATT1_Q4_GROUP_SIZE_MIN, packed) != 0) {
        fputs("saturation: pack failed\n", stderr);
        return -1;
    }

    if (att1_q4_unpack_group(packed, ATT1_Q4_GROUP_SIZE_MIN, unpacked) != 0) {
        fputs("saturation: unpack failed\n", stderr);
        return -1;
    }

    if ((unpacked[0] != 7) || (unpacked[1] != -7)) {
        fprintf(stderr, "saturation: expected +7/-7, got %d/%d\n",
                (int)unpacked[0], (int)unpacked[1]);
        return -1;
    }

    /* Remaining elements were 0 */
    for (i = 2u; i < ATT1_Q4_GROUP_SIZE_MIN; i++) {
        if (unpacked[i] != 0) {
            fprintf(stderr, "saturation: element %u should be 0, got %d\n",
                    i, (int)unpacked[i]);
            return -1;
        }
    }
    return 0;
}

/* ── zero row ────────────────────────────────────────────────────────────── */

static int test_zero_row(void)
{
    float src[ATT1_Q4_GROUP_SIZE_DEFAULT];
    uint8_t packed[ATT1_Q4_GROUP_SIZE_DEFAULT / 2];
    float dst[ATT1_Q4_GROUP_SIZE_DEFAULT];
    float scale = -99.0f;
    uint32_t i = 0u;

    memset(src, 0, sizeof(src));

    if (att1_q4_quantize_group(src, ATT1_Q4_GROUP_SIZE_DEFAULT,
                               packed, &scale) != 0) {
        fputs("zero_row: quantize_group failed\n", stderr);
        return -1;
    }

    /* Scale must be 1.0 (default for zero row), not zero or negative */
    if (!near_f32(scale, 1.0f, 0.000001f)) {
        fprintf(stderr, "zero_row: scale=%f, expected 1.0\n", (double)scale);
        return -1;
    }

    if (att1_q4_dequantize_group(packed, ATT1_Q4_GROUP_SIZE_DEFAULT,
                                 scale, dst) != 0) {
        fputs("zero_row: dequantize_group failed\n", stderr);
        return -1;
    }

    for (i = 0u; i < ATT1_Q4_GROUP_SIZE_DEFAULT; i++) {
        if (!near_f32(dst[i], 0.0f, 0.000001f)) {
            fprintf(stderr, "zero_row: dst[%u]=%f, expected 0\n",
                    i, (double)dst[i]);
            return -1;
        }
    }
    return 0;
}

/* ── deterministic medium vector round-trip ──────────────────────────────── */

/*
 * Use a deterministic 32-element vector covering the range [-1, 1].
 * Expected max_abs_error < 1/7 * scale  (one quantization step).
 */
static int test_round_trip_tolerance(void)
{
    float src[ATT1_Q4_GROUP_SIZE_DEFAULT];
    uint8_t packed[ATT1_Q4_GROUP_SIZE_DEFAULT / 2];
    float dst[ATT1_Q4_GROUP_SIZE_DEFAULT];
    float scale = 0.0f;
    float max_err = 0.0f;
    float tolerance = 0.0f;
    uint32_t i = 0u;

    /* Deterministic source: evenly spaced in [-1, 1] with step 2/31 */
    for (i = 0u; i < ATT1_Q4_GROUP_SIZE_DEFAULT; i++) {
        src[i] = -1.0f + (float)i * (2.0f / (float)(ATT1_Q4_GROUP_SIZE_DEFAULT - 1u));
    }

    if (att1_q4_quantize_group(src, ATT1_Q4_GROUP_SIZE_DEFAULT,
                               packed, &scale) != 0) {
        fputs("round_trip_tolerance: quantize_group failed\n", stderr);
        return -1;
    }

    if (att1_q4_dequantize_group(packed, ATT1_Q4_GROUP_SIZE_DEFAULT,
                                 scale, dst) != 0) {
        fputs("round_trip_tolerance: dequantize_group failed\n", stderr);
        return -1;
    }

    tolerance = scale; /* one quantization step = scale * 1 */

    for (i = 0u; i < ATT1_Q4_GROUP_SIZE_DEFAULT; i++) {
        const float err = fabsf(src[i] - dst[i]);
        if (err > max_err) { max_err = err; }
    }

    if (max_err > tolerance) {
        fprintf(stderr,
                "round_trip_tolerance: max_err=%f > tolerance=%f\n",
                (double)max_err, (double)tolerance);
        return -1;
    }
    return 0;
}

/* ── invalid group size ──────────────────────────────────────────────────── */

static int test_invalid_group_size(void)
{
    int8_t src[ATT1_Q4_GROUP_SIZE_MIN];
    uint8_t packed[ATT1_Q4_GROUP_SIZE_MIN / 2];
    int8_t unpacked[ATT1_Q4_GROUP_SIZE_MIN];
    float fsrc[ATT1_Q4_GROUP_SIZE_MIN];
    float fdst[ATT1_Q4_GROUP_SIZE_MIN];
    float scale = 0.0f;
    const uint32_t bad_sizes[] = { 0u, 1u, 3u, 15u, 48u, 256u };
    const size_t n = sizeof(bad_sizes) / sizeof(bad_sizes[0]);
    size_t k = 0u;

    memset(src, 0, sizeof(src));
    memset(fsrc, 0, sizeof(fsrc));

    for (k = 0u; k < n; k++) {
        const uint32_t gs = bad_sizes[k];

        if (att1_q4_pack_group(src, gs, packed) == 0) {
            fprintf(stderr, "invalid_group_size: pack succeeded for gs=%u\n", gs);
            return -1;
        }
        if (att1_q4_unpack_group(packed, gs, unpacked) == 0) {
            fprintf(stderr, "invalid_group_size: unpack succeeded for gs=%u\n", gs);
            return -1;
        }
        if (att1_q4_group_scale(fsrc, gs, &scale) == 0) {
            fprintf(stderr, "invalid_group_size: group_scale succeeded for gs=%u\n", gs);
            return -1;
        }
        if (att1_q4_quantize_group(fsrc, gs, packed, &scale) == 0) {
            fprintf(stderr, "invalid_group_size: quantize_group succeeded for gs=%u\n", gs);
            return -1;
        }
        if (att1_q4_dequantize_group(packed, gs, 1.0f, fdst) == 0) {
            fprintf(stderr, "invalid_group_size: dequantize_group succeeded for gs=%u\n", gs);
            return -1;
        }
    }
    return 0;
}

/* ── null / invalid arg checks ───────────────────────────────────────────── */

static int test_null_args(void)
{
    const uint32_t gs = ATT1_Q4_GROUP_SIZE_DEFAULT;
    int8_t int4_buf[ATT1_Q4_GROUP_SIZE_DEFAULT];
    uint8_t packed_buf[ATT1_Q4_GROUP_SIZE_DEFAULT / 2];
    float float_buf[ATT1_Q4_GROUP_SIZE_DEFAULT];
    float scale = 1.0f;

    memset(int4_buf, 0, sizeof(int4_buf));
    memset(packed_buf, 0, sizeof(packed_buf));
    memset(float_buf, 0, sizeof(float_buf));

    if (att1_q4_group_scale(NULL, gs, &scale) == 0) {
        fputs("null_args: group_scale with null src should fail\n", stderr);
        return -1;
    }
    if (att1_q4_group_scale(float_buf, gs, NULL) == 0) {
        fputs("null_args: group_scale with null out_scale should fail\n", stderr);
        return -1;
    }
    if (att1_q4_pack_group(NULL, gs, packed_buf) == 0) {
        fputs("null_args: pack with null src should fail\n", stderr);
        return -1;
    }
    if (att1_q4_pack_group(int4_buf, gs, NULL) == 0) {
        fputs("null_args: pack with null dst should fail\n", stderr);
        return -1;
    }
    if (att1_q4_unpack_group(NULL, gs, int4_buf) == 0) {
        fputs("null_args: unpack with null src should fail\n", stderr);
        return -1;
    }
    if (att1_q4_unpack_group(packed_buf, gs, NULL) == 0) {
        fputs("null_args: unpack with null dst should fail\n", stderr);
        return -1;
    }
    if (att1_q4_quantize_group(NULL, gs, packed_buf, &scale) == 0) {
        fputs("null_args: quantize_group with null src should fail\n", stderr);
        return -1;
    }
    if (att1_q4_quantize_group(float_buf, gs, NULL, &scale) == 0) {
        fputs("null_args: quantize_group with null dst should fail\n", stderr);
        return -1;
    }
    if (att1_q4_quantize_group(float_buf, gs, packed_buf, NULL) == 0) {
        fputs("null_args: quantize_group with null out_scale should fail\n", stderr);
        return -1;
    }
    if (att1_q4_dequantize_group(NULL, gs, scale, float_buf) == 0) {
        fputs("null_args: dequantize_group with null src should fail\n", stderr);
        return -1;
    }
    if (att1_q4_dequantize_group(packed_buf, gs, scale, NULL) == 0) {
        fputs("null_args: dequantize_group with null dst should fail\n", stderr);
        return -1;
    }
    return 0;
}

/* ── non-finite input ────────────────────────────────────────────────────── */

static int test_nonfinite_input(void)
{
    float src[ATT1_Q4_GROUP_SIZE_DEFAULT];
    uint8_t packed[ATT1_Q4_GROUP_SIZE_DEFAULT / 2];
    float scale = 0.0f;

    memset(src, 0, sizeof(src));
    src[0] = 1.0f / 0.0f; /* +inf */

    if (att1_q4_group_scale(src, ATT1_Q4_GROUP_SIZE_DEFAULT, &scale) == 0) {
        fputs("nonfinite: group_scale with inf should fail\n", stderr);
        return -1;
    }

    src[0] = 0.0f / 0.0f; /* nan */
    if (att1_q4_quantize_group(src, ATT1_Q4_GROUP_SIZE_DEFAULT,
                               packed, &scale) == 0) {
        fputs("nonfinite: quantize_group with nan should fail\n", stderr);
        return -1;
    }
    return 0;
}

/* ── valid group sizes 16/32/64/128 ──────────────────────────────────────── */

static int test_all_valid_group_sizes(void)
{
    const uint32_t valid[] = {
        ATT1_Q4_GROUP_SIZE_MIN,          /* 16  */
        ATT1_Q4_GROUP_SIZE_DEFAULT,       /* 32  */
        ATT1_Q4_GROUP_SIZE_DEFAULT * 2u,  /* 64  */
        ATT1_Q4_GROUP_SIZE_MAX            /* 128 */
    };
    const size_t n = sizeof(valid) / sizeof(valid[0]);
    size_t k = 0u;

    for (k = 0u; k < n; k++) {
        const uint32_t gs = valid[k];
        float src[ATT1_Q4_GROUP_SIZE_MAX];
        uint8_t packed[ATT1_Q4_GROUP_SIZE_MAX / 2];
        float dst[ATT1_Q4_GROUP_SIZE_MAX];
        float scale = 0.0f;
        uint32_t i = 0u;

        for (i = 0u; i < gs; i++) {
            src[i] = -0.5f + (float)i * (1.0f / (float)gs);
        }

        if (att1_q4_quantize_group(src, gs, packed, &scale) != 0) {
            fprintf(stderr, "all_valid_gs: quantize failed for gs=%u\n", gs);
            return -1;
        }
        if (att1_q4_dequantize_group(packed, gs, scale, dst) != 0) {
            fprintf(stderr, "all_valid_gs: dequantize failed for gs=%u\n", gs);
            return -1;
        }
    }
    return 0;
}

/* ── main ────────────────────────────────────────────────────────────────── */

int main(void)
{
    if (test_nibble_order() != 0) {
        fputs("FAIL: nibble_order\n", stderr);
        return 1;
    }
    if (test_pack_unpack_all_values() != 0) {
        fputs("FAIL: pack_unpack_all_values\n", stderr);
        return 1;
    }
    if (test_saturation() != 0) {
        fputs("FAIL: saturation\n", stderr);
        return 1;
    }
    if (test_zero_row() != 0) {
        fputs("FAIL: zero_row\n", stderr);
        return 1;
    }
    if (test_round_trip_tolerance() != 0) {
        fputs("FAIL: round_trip_tolerance\n", stderr);
        return 1;
    }
    if (test_invalid_group_size() != 0) {
        fputs("FAIL: invalid_group_size\n", stderr);
        return 1;
    }
    if (test_null_args() != 0) {
        fputs("FAIL: null_args\n", stderr);
        return 1;
    }
    if (test_nonfinite_input() != 0) {
        fputs("FAIL: nonfinite_input\n", stderr);
        return 1;
    }
    if (test_all_valid_group_sizes() != 0) {
        fputs("FAIL: all_valid_group_sizes\n", stderr);
        return 1;
    }

    puts("quant_q4_pack test passed");
    return 0;
}
