#include "att1_kv_mmu.h"

#include <stdio.h>

static int exact_vec(const float *actual, const float *expected, size_t count)
{
    size_t i = 0u;

    for (i = 0u; i < count; i++) {
        if (actual[i] != expected[i]) {
            return 0;
        }
    }

    return 1;
}

static int read_head(att1_kv_mmu *mmu,
                     uint64_t session_id,
                     size_t layer_id,
                     size_t head_id,
                     size_t position,
                     const float *expected_key,
                     const float *expected_value)
{
    float out_key[2] = {0.0f, 0.0f};
    float out_value[2] = {0.0f, 0.0f};

    if (att1_kv_mmu_read(mmu,
                         session_id,
                         layer_id,
                         head_id,
                         position,
                         out_key,
                         out_value) != 0) {
        return 0;
    }

    return exact_vec(out_key, expected_key, 2u) &&
           exact_vec(out_value, expected_value, 2u);
}

static int expect_page(att1_kv_mmu *mmu,
                       uint64_t session_id,
                       size_t layer_id,
                       size_t position,
                       size_t logical_page)
{
    att1_kv_mmu_page_ref page_ref;

    if (att1_kv_mmu_lookup_page(mmu,
                                session_id,
                                layer_id,
                                position,
                                &page_ref) != 0) {
        return 0;
    }

    return (page_ref.session_id == session_id) &&
           (page_ref.layer_id == layer_id) &&
           (page_ref.logical_page == logical_page) &&
           (page_ref.page_tokens == 2u);
}

static int test_isolation_and_boundaries(void)
{
    const att1_kv_mmu_config config = {
        .max_sessions = 4u,
        .max_pages = 16u,
        .num_layers = 3u,
        .num_heads = 2u,
        .head_dim = 2u,
        .page_tokens = 2u,
        .max_positions = 6u
    };
    const float key_a0[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float val_a0[4] = {101.0f, 102.0f, 103.0f, 104.0f};
    const float key_a1[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    const float val_a1[4] = {105.0f, 106.0f, 107.0f, 108.0f};
    const float key_a2[4] = {9.0f, 10.0f, 11.0f, 12.0f};
    const float val_a2[4] = {109.0f, 110.0f, 111.0f, 112.0f};
    const float key_a3[4] = {13.0f, 14.0f, 15.0f, 16.0f};
    const float val_a3[4] = {113.0f, 114.0f, 115.0f, 116.0f};
    const float key_b0[4] = {21.0f, 22.0f, 23.0f, 24.0f};
    const float val_b0[4] = {201.0f, 202.0f, 203.0f, 204.0f};
    const float key_l1[4] = {31.0f, 32.0f, 33.0f, 34.0f};
    const float val_l1[4] = {301.0f, 302.0f, 303.0f, 304.0f};
    const float expected_a_head0[2] = {1.0f, 2.0f};
    const float expected_a_head1[2] = {3.0f, 4.0f};
    const float expected_a_val0[2] = {101.0f, 102.0f};
    const float expected_a_val1[2] = {103.0f, 104.0f};
    const float expected_b_head0[2] = {21.0f, 22.0f};
    const float expected_b_val0[2] = {201.0f, 202.0f};
    const float expected_l1_head0[2] = {31.0f, 32.0f};
    const float expected_l1_val0[2] = {301.0f, 302.0f};
    const float expected_range_keys[6] = {
        7.0f, 8.0f,
        11.0f, 12.0f,
        15.0f, 16.0f
    };
    const float expected_range_values[6] = {
        107.0f, 108.0f,
        111.0f, 112.0f,
        115.0f, 116.0f
    };
    float range_keys[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float range_values[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float out_key[2] = {0.0f, 0.0f};
    float out_value[2] = {0.0f, 0.0f};
    att1_kv_mmu mmu;

    if (att1_kv_mmu_init(&mmu, &config) != 0) {
        fputs("kv mmu init failed\n", stderr);
        return 0;
    }

    if ((att1_kv_mmu_create_session(&mmu, 100u) != 0) ||
        (att1_kv_mmu_create_session(&mmu, 200u) != 0) ||
        (att1_kv_mmu_create_session(&mmu, 300u) != 0)) {
        fputs("kv mmu session setup failed\n", stderr);
        att1_kv_mmu_free(&mmu);
        return 0;
    }

    if ((att1_kv_mmu_append(&mmu, 100u, 0u, 0u, key_a0, val_a0) != 0) ||
        (att1_kv_mmu_append(&mmu, 200u, 0u, 0u, key_b0, val_b0) != 0) ||
        (att1_kv_mmu_append(&mmu, 100u, 1u, 0u, key_l1, val_l1) != 0)) {
        fputs("kv mmu isolation append failed\n", stderr);
        att1_kv_mmu_free(&mmu);
        return 0;
    }

    if (!read_head(&mmu, 100u, 0u, 0u, 0u, expected_a_head0, expected_a_val0) ||
        !read_head(&mmu, 100u, 0u, 1u, 0u, expected_a_head1, expected_a_val1) ||
        !read_head(&mmu, 200u, 0u, 0u, 0u, expected_b_head0, expected_b_val0) ||
        !read_head(&mmu, 100u, 1u, 0u, 0u, expected_l1_head0, expected_l1_val0)) {
        fputs("kv mmu isolation read failed\n", stderr);
        att1_kv_mmu_free(&mmu);
        return 0;
    }

    if (att1_kv_mmu_read(&mmu,
                         100u,
                         0u,
                         0u,
                         0u,
                         out_key,
                         out_value) != 0) {
        fputs("kv mmu session A read failed\n", stderr);
        att1_kv_mmu_free(&mmu);
        return 0;
    }

    if (exact_vec(out_key, expected_b_head0, 2u) ||
        exact_vec(out_value, expected_b_val0, 2u)) {
        fputs("kv mmu session isolation returned session B data\n", stderr);
        att1_kv_mmu_free(&mmu);
        return 0;
    }

    if ((att1_kv_mmu_append(&mmu, 100u, 0u, 1u, key_a1, val_a1) != 0) ||
        (att1_kv_mmu_append(&mmu, 100u, 0u, 2u, key_a2, val_a2) != 0) ||
        (att1_kv_mmu_append(&mmu, 100u, 0u, 3u, key_a3, val_a3) != 0)) {
        fputs("kv mmu page boundary append failed\n", stderr);
        att1_kv_mmu_free(&mmu);
        return 0;
    }

    if (!expect_page(&mmu, 100u, 0u, 0u, 0u) ||
        !expect_page(&mmu, 100u, 0u, 1u, 0u) ||
        !expect_page(&mmu, 100u, 0u, 2u, 1u) ||
        !expect_page(&mmu, 100u, 0u, 3u, 1u)) {
        fputs("kv mmu page boundary lookup failed\n", stderr);
        att1_kv_mmu_free(&mmu);
        return 0;
    }

    if (att1_kv_mmu_copy_range(&mmu,
                               100u,
                               0u,
                               1u,
                               1u,
                               3u,
                               range_keys,
                               range_values) != 0) {
        fputs("kv mmu page-crossing range copy failed\n", stderr);
        att1_kv_mmu_free(&mmu);
        return 0;
    }

    if (!exact_vec(range_keys, expected_range_keys, 6u) ||
        !exact_vec(range_values, expected_range_values, 6u)) {
        fputs("kv mmu page-crossing range order failed\n", stderr);
        att1_kv_mmu_free(&mmu);
        return 0;
    }

    if (att1_kv_mmu_append(&mmu, 300u, 0u, 5u, key_a0, val_a0) == 0) {
        fputs("kv mmu gap append did not fail\n", stderr);
        att1_kv_mmu_free(&mmu);
        return 0;
    }

    if (att1_kv_mmu_append(&mmu, 300u, 0u, 0u, key_a0, val_a0) != 0) {
        fputs("kv mmu sequential append after gap rejection failed\n", stderr);
        att1_kv_mmu_free(&mmu);
        return 0;
    }

    if ((att1_kv_mmu_read(&mmu, 300u, 0u, 0u, 1u, out_key, out_value) == 0) ||
        (att1_kv_mmu_copy_range(&mmu,
                                300u,
                                0u,
                                0u,
                                0u,
                                2u,
                                range_keys,
                                range_values) == 0)) {
        fputs("kv mmu missing sequential position check failed\n", stderr);
        att1_kv_mmu_free(&mmu);
        return 0;
    }

    if (att1_kv_mmu_append(&mmu, 100u, 0u, 1u, key_a1, val_a1) == 0) {
        fputs("kv mmu duplicate append did not fail\n", stderr);
        att1_kv_mmu_free(&mmu);
        return 0;
    }

    if ((att1_kv_mmu_read(&mmu, 999u, 0u, 0u, 0u, out_key, out_value) == 0) ||
        (att1_kv_mmu_read(&mmu, 100u, 3u, 0u, 0u, out_key, out_value) == 0) ||
        (att1_kv_mmu_read(&mmu, 100u, 0u, 2u, 0u, out_key, out_value) == 0) ||
        (att1_kv_mmu_read(&mmu, 100u, 0u, 0u, 6u, out_key, out_value) == 0) ||
        (att1_kv_mmu_copy_range(&mmu,
                                100u,
                                0u,
                                0u,
                                5u,
                                2u,
                                range_keys,
                                range_values) == 0) ||
        (att1_kv_mmu_copy_range(&mmu,
                                100u,
                                0u,
                                0u,
                                3u,
                                3u,
                                range_keys,
                                range_values) == 0)) {
        fputs("kv mmu invalid access checks failed\n", stderr);
        att1_kv_mmu_free(&mmu);
        return 0;
    }

    att1_kv_mmu_free(&mmu);
    return 1;
}

static int test_trace_counters(void)
{
    const att1_kv_mmu_config config = {
        .max_sessions = 1u,
        .max_pages = 2u,
        .num_layers = 1u,
        .num_heads = 1u,
        .head_dim = 2u,
        .page_tokens = 2u,
        .max_positions = 4u
    };
    const float key[2] = {1.0f, 2.0f};
    const float value[2] = {3.0f, 4.0f};
    att1_kv_mmu_page_ref page_ref;
    att1_kv_mmu_counters counters;
    att1_kv_mmu mmu;

    if (att1_kv_mmu_init(&mmu, &config) != 0) {
        fputs("kv mmu counter init failed\n", stderr);
        return 0;
    }

    if (att1_kv_mmu_create_session(&mmu, 1u) != 0) {
        fputs("kv mmu counter session failed\n", stderr);
        att1_kv_mmu_free(&mmu);
        return 0;
    }

    att1_kv_mmu_reset_counters(&mmu);

    if (att1_kv_mmu_lookup_page(&mmu, 1u, 0u, 0u, &page_ref) == 0) {
        fputs("kv mmu missing lookup unexpectedly hit\n", stderr);
        att1_kv_mmu_free(&mmu);
        return 0;
    }

    if (att1_kv_mmu_append(&mmu, 1u, 0u, 0u, key, value) != 0) {
        fputs("kv mmu counter append failed\n", stderr);
        att1_kv_mmu_free(&mmu);
        return 0;
    }

    if (att1_kv_mmu_lookup_page(&mmu, 1u, 0u, 0u, &page_ref) != 0) {
        fputs("kv mmu hit lookup failed\n", stderr);
        att1_kv_mmu_free(&mmu);
        return 0;
    }

    att1_kv_mmu_get_counters(&mmu, &counters);
    if ((counters.page_allocations != 1u) ||
        (counters.page_hits != 1u) ||
        (counters.page_misses != 2u) ||
        (counters.append_ops != 1u)) {
        fputs("kv mmu counter values failed\n", stderr);
        att1_kv_mmu_free(&mmu);
        return 0;
    }

    att1_kv_mmu_free(&mmu);
    return 1;
}

int main(void)
{
    if (!test_isolation_and_boundaries()) {
        return 1;
    }

    if (!test_trace_counters()) {
        return 1;
    }

    puts("kv_mmu test passed");
    return 0;
}
