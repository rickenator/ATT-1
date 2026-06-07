#include "att1_kv_mmu.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct att1_kv_mmu_session {
    uint64_t session_id;
    int active;
} att1_kv_mmu_session;

typedef struct att1_kv_mmu_page {
    uint64_t physical_page;
    uint64_t session_id;
    size_t layer_id;
    size_t logical_page;
    int active;
    unsigned char *token_present;
    float *keys;
    float *values;
} att1_kv_mmu_page;

struct att1_kv_mmu {
    att1_kv_mmu_config config;
    att1_kv_mmu_counters counters;
    att1_kv_mmu_session *sessions;
    att1_kv_mmu_page *pages;
};

static int att1_mul_size(size_t lhs, size_t rhs, size_t *out)
{
    if ((lhs != 0u) && (rhs > (SIZE_MAX / lhs))) {
        return -1;
    }

    *out = lhs * rhs;
    return 0;
}

static void att1_kv_mmu_error(att1_kv_mmu *mmu)
{
    if (mmu != NULL) {
        mmu->counters.errors++;
    }
}

static int att1_kv_mmu_config_valid(const att1_kv_mmu_config *config)
{
    size_t heads_dims = 0u;
    size_t page_elements = 0u;
    size_t capacity_positions = 0u;

    if (config == NULL) {
        return 0;
    }

    if ((config->max_sessions == 0u) ||
        (config->max_pages == 0u) ||
        (config->num_layers == 0u) ||
        (config->num_heads == 0u) ||
        (config->head_dim == 0u) ||
        (config->page_tokens == 0u) ||
        (config->max_positions == 0u)) {
        return 0;
    }

    if (att1_mul_size(config->num_heads, config->head_dim, &heads_dims) != 0) {
        return 0;
    }

    if (att1_mul_size(config->page_tokens, heads_dims, &page_elements) != 0) {
        return 0;
    }

    if (att1_mul_size(config->max_pages,
                      config->page_tokens,
                      &capacity_positions) != 0) {
        return 0;
    }

    if (config->max_positions > capacity_positions) {
        return 0;
    }

    return page_elements > 0u;
}

static size_t att1_kv_mmu_page_elements(const att1_kv_mmu *mmu)
{
    return mmu->config.page_tokens *
           mmu->config.num_heads *
           mmu->config.head_dim;
}

static size_t att1_kv_mmu_position_offset(const att1_kv_mmu *mmu,
                                          size_t token_in_page,
                                          size_t head_id)
{
    return ((token_in_page * mmu->config.num_heads) + head_id) *
           mmu->config.head_dim;
}

static att1_kv_mmu_session *att1_kv_mmu_find_session(att1_kv_mmu *mmu,
                                                     uint64_t session_id)
{
    size_t i = 0u;

    if (mmu == NULL) {
        return NULL;
    }

    for (i = 0u; i < mmu->config.max_sessions; i++) {
        if ((mmu->sessions[i].active != 0) &&
            (mmu->sessions[i].session_id == session_id)) {
            return &mmu->sessions[i];
        }
    }

    return NULL;
}

static att1_kv_mmu_page *att1_kv_mmu_find_page(att1_kv_mmu *mmu,
                                               uint64_t session_id,
                                               size_t layer_id,
                                               size_t logical_page)
{
    size_t i = 0u;

    if (mmu == NULL) {
        return NULL;
    }

    for (i = 0u; i < mmu->config.max_pages; i++) {
        att1_kv_mmu_page *page = &mmu->pages[i];

        if ((page->active != 0) &&
            (page->session_id == session_id) &&
            (page->layer_id == layer_id) &&
            (page->logical_page == logical_page)) {
            return page;
        }
    }

    return NULL;
}

static void att1_kv_mmu_release_page(att1_kv_mmu_page *page)
{
    const uint64_t physical_page = page->physical_page;

    free(page->token_present);
    free(page->keys);
    free(page->values);
    memset(page, 0, sizeof(*page));
    page->physical_page = physical_page;
}

static att1_kv_mmu_page *att1_kv_mmu_alloc_page(att1_kv_mmu *mmu,
                                                uint64_t session_id,
                                                size_t layer_id,
                                                size_t logical_page)
{
    size_t i = 0u;
    size_t page_elements = 0u;

    for (i = 0u; i < mmu->config.max_pages; i++) {
        att1_kv_mmu_page *page = &mmu->pages[i];

        if (page->active == 0) {
            page_elements = att1_kv_mmu_page_elements(mmu);
            page->token_present = calloc(mmu->config.page_tokens,
                                         sizeof(unsigned char));
            page->keys = calloc(page_elements, sizeof(float));
            page->values = calloc(page_elements, sizeof(float));

            if ((page->token_present == NULL) ||
                (page->keys == NULL) ||
                (page->values == NULL)) {
                att1_kv_mmu_release_page(page);
                return NULL;
            }

            page->session_id = session_id;
            page->layer_id = layer_id;
            page->logical_page = logical_page;
            page->active = 1;
            mmu->counters.page_allocations++;
            return page;
        }
    }

    return NULL;
}

static att1_kv_mmu_page *att1_kv_mmu_lookup_existing(att1_kv_mmu *mmu,
                                                     uint64_t session_id,
                                                     size_t layer_id,
                                                     size_t position)
{
    const size_t logical_page = position / mmu->config.page_tokens;
    att1_kv_mmu_page *page = att1_kv_mmu_find_page(mmu,
                                                   session_id,
                                                   layer_id,
                                                   logical_page);

    if (page == NULL) {
        mmu->counters.page_misses++;
        return NULL;
    }

    mmu->counters.page_hits++;
    return page;
}

static int att1_kv_mmu_token_present(att1_kv_mmu *mmu,
                                     uint64_t session_id,
                                     size_t layer_id,
                                     size_t position)
{
    const size_t logical_page = position / mmu->config.page_tokens;
    const size_t token_in_page = position % mmu->config.page_tokens;
    att1_kv_mmu_page *page = att1_kv_mmu_find_page(mmu,
                                                   session_id,
                                                   layer_id,
                                                   logical_page);

    if (page == NULL) {
        return 0;
    }

    return page->token_present[token_in_page] != 0u;
}

static size_t att1_kv_mmu_next_position(att1_kv_mmu *mmu,
                                        uint64_t session_id,
                                        size_t layer_id)
{
    size_t position = 0u;

    while (position < mmu->config.max_positions) {
        if (!att1_kv_mmu_token_present(mmu, session_id, layer_id, position)) {
            break;
        }

        position++;
    }

    return position;
}

static att1_status_t att1_kv_mmu_validate_address(att1_kv_mmu *mmu,
                                                  uint64_t session_id,
                                                  size_t layer_id)
{
    if (mmu == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }

    if ((mmu->sessions == NULL) || (mmu->pages == NULL)) {
        att1_kv_mmu_error(mmu);
        return ATT1_ERR_INVALID_ARG;
    }

    if (att1_kv_mmu_find_session(mmu, session_id) == NULL) {
        att1_kv_mmu_error(mmu);
        return ATT1_ERR_NOT_FOUND;
    }

    if (layer_id >= mmu->config.num_layers) {
        att1_kv_mmu_error(mmu);
        return ATT1_ERR_INVALID_ARG;
    }

    return ATT1_OK;
}

static att1_status_t att1_kv_mmu_validate_position(att1_kv_mmu *mmu,
                                                   size_t position)
{
    if (mmu == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }

    if (position >= mmu->config.max_positions) {
        att1_kv_mmu_error(mmu);
        return ATT1_ERR_INVALID_ARG;
    }

    return ATT1_OK;
}

static att1_status_t att1_kv_mmu_validate_range(att1_kv_mmu *mmu,
                                                size_t start_position,
                                                size_t position_count)
{
    if (mmu == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }

    if (position_count == 0u) {
        att1_kv_mmu_error(mmu);
        return ATT1_ERR_INVALID_ARG;
    }

    if (start_position > (SIZE_MAX - (position_count - 1u))) {
        att1_kv_mmu_error(mmu);
        return ATT1_ERR_INVALID_ARG;
    }

    if ((start_position + position_count) > mmu->config.max_positions) {
        att1_kv_mmu_error(mmu);
        return ATT1_ERR_INVALID_ARG;
    }

    return ATT1_OK;
}

att1_status_t att1_kv_mmu_create(const att1_kv_mmu_config *config,
                                 att1_kv_mmu **out_mmu)
{
    att1_kv_mmu *mmu = NULL;
    size_t i = 0u;

    if (out_mmu == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }

    *out_mmu = NULL;

    if (!att1_kv_mmu_config_valid(config)) {
        return ATT1_ERR_INVALID_ARG;
    }

    mmu = calloc(1u, sizeof(*mmu));
    if (mmu == NULL) {
        return ATT1_ERR_OOM;
    }

    mmu->config = *config;
    mmu->sessions = calloc(config->max_sessions, sizeof(*mmu->sessions));
    mmu->pages = calloc(config->max_pages, sizeof(*mmu->pages));

    if ((mmu->sessions == NULL) || (mmu->pages == NULL)) {
        att1_kv_mmu_destroy(mmu);
        return ATT1_ERR_OOM;
    }

    for (i = 0u; i < config->max_pages; i++) {
        mmu->pages[i].physical_page = (uint64_t)i;
    }

    *out_mmu = mmu;
    return ATT1_OK;
}

void att1_kv_mmu_destroy(att1_kv_mmu *mmu)
{
    size_t i = 0u;

    if (mmu == NULL) {
        return;
    }

    if (mmu->pages != NULL) {
        for (i = 0u; i < mmu->config.max_pages; i++) {
            att1_kv_mmu_release_page(&mmu->pages[i]);
        }
    }

    free(mmu->sessions);
    free(mmu->pages);
    free(mmu);
}

att1_status_t att1_kv_mmu_create_session(att1_kv_mmu *mmu,
                                         uint64_t session_id)
{
    size_t i = 0u;

    if ((mmu == NULL) || (mmu->sessions == NULL)) {
        att1_kv_mmu_error(mmu);
        return ATT1_ERR_INVALID_ARG;
    }

    if (att1_kv_mmu_find_session(mmu, session_id) != NULL) {
        att1_kv_mmu_error(mmu);
        return ATT1_ERR_STATE;
    }

    for (i = 0u; i < mmu->config.max_sessions; i++) {
        if (mmu->sessions[i].active == 0) {
            mmu->sessions[i].active = 1;
            mmu->sessions[i].session_id = session_id;
            return ATT1_OK;
        }
    }

    att1_kv_mmu_error(mmu);
    return ATT1_ERR_OOM;
}

att1_status_t att1_kv_mmu_destroy_session(att1_kv_mmu *mmu,
                                          uint64_t session_id)
{
    att1_kv_mmu_session *session = NULL;
    size_t i = 0u;

    if ((mmu == NULL) || (mmu->sessions == NULL) || (mmu->pages == NULL)) {
        att1_kv_mmu_error(mmu);
        return ATT1_ERR_INVALID_ARG;
    }

    session = att1_kv_mmu_find_session(mmu, session_id);
    if (session == NULL) {
        att1_kv_mmu_error(mmu);
        return ATT1_ERR_NOT_FOUND;
    }

    for (i = 0u; i < mmu->config.max_pages; i++) {
        if ((mmu->pages[i].active != 0) &&
            (mmu->pages[i].session_id == session_id)) {
            att1_kv_mmu_release_page(&mmu->pages[i]);
        }
    }

    memset(session, 0, sizeof(*session));
    return ATT1_OK;
}

att1_status_t att1_kv_mmu_append(att1_kv_mmu *mmu,
                                 uint64_t session_id,
                                 size_t layer_id,
                                 size_t position,
                                 const float *key,
                                 const float *value)
{
    att1_status_t status = ATT1_OK;
    size_t page_id = 0u;
    size_t token_in_page = 0u;
    size_t head_values = 0u;
    size_t position_values = 0u;
    att1_kv_mmu_page *page = NULL;

    if ((key == NULL) || (value == NULL)) {
        att1_kv_mmu_error(mmu);
        return ATT1_ERR_INVALID_ARG;
    }

    status = att1_kv_mmu_validate_address(mmu, session_id, layer_id);
    if (status != ATT1_OK) {
        return status;
    }

    status = att1_kv_mmu_validate_position(mmu, position);
    if (status != ATT1_OK) {
        return status;
    }

    if (position != att1_kv_mmu_next_position(mmu, session_id, layer_id)) {
        att1_kv_mmu_error(mmu);
        return ATT1_ERR_STATE;
    }

    page_id = position / mmu->config.page_tokens;
    token_in_page = position % mmu->config.page_tokens;
    page = att1_kv_mmu_lookup_existing(mmu, session_id, layer_id, position);

    if (page == NULL) {
        page = att1_kv_mmu_alloc_page(mmu, session_id, layer_id, page_id);
        if (page == NULL) {
            att1_kv_mmu_error(mmu);
            return ATT1_ERR_OOM;
        }
    }

    if (page->token_present[token_in_page] != 0u) {
        att1_kv_mmu_error(mmu);
        return ATT1_ERR_STATE;
    }

    if (att1_mul_size(mmu->config.num_heads,
                      mmu->config.head_dim,
                      &head_values) != 0) {
        att1_kv_mmu_error(mmu);
        return ATT1_ERR_INVALID_ARG;
    }

    position_values = token_in_page * head_values;
    memcpy(&page->keys[position_values], key, head_values * sizeof(float));
    memcpy(&page->values[position_values], value, head_values * sizeof(float));
    page->token_present[token_in_page] = 1u;
    mmu->counters.append_ops++;
    return ATT1_OK;
}

att1_status_t att1_kv_mmu_read(att1_kv_mmu *mmu,
                               uint64_t session_id,
                               size_t layer_id,
                               size_t head_id,
                               size_t position,
                               float *out_key,
                               float *out_value)
{
    att1_status_t status = ATT1_OK;
    size_t token_in_page = 0u;
    size_t offset = 0u;
    att1_kv_mmu_page *page = NULL;

    if ((out_key == NULL) && (out_value == NULL)) {
        att1_kv_mmu_error(mmu);
        return ATT1_ERR_INVALID_ARG;
    }

    status = att1_kv_mmu_validate_address(mmu, session_id, layer_id);
    if (status != ATT1_OK) {
        return status;
    }

    status = att1_kv_mmu_validate_position(mmu, position);
    if (status != ATT1_OK) {
        return status;
    }

    if (head_id >= mmu->config.num_heads) {
        att1_kv_mmu_error(mmu);
        return ATT1_ERR_INVALID_ARG;
    }

    token_in_page = position % mmu->config.page_tokens;
    page = att1_kv_mmu_lookup_existing(mmu, session_id, layer_id, position);
    if ((page == NULL) || (page->token_present[token_in_page] == 0u)) {
        att1_kv_mmu_error(mmu);
        return ATT1_ERR_NOT_FOUND;
    }

    offset = att1_kv_mmu_position_offset(mmu, token_in_page, head_id);
    if (out_key != NULL) {
        memcpy(out_key, &page->keys[offset], mmu->config.head_dim * sizeof(float));
    }

    if (out_value != NULL) {
        memcpy(out_value, &page->values[offset], mmu->config.head_dim * sizeof(float));
    }

    mmu->counters.read_ops++;
    return ATT1_OK;
}

att1_status_t att1_kv_mmu_copy_range(att1_kv_mmu *mmu,
                                     uint64_t session_id,
                                     size_t layer_id,
                                     size_t head_id,
                                     size_t start_position,
                                     size_t position_count,
                                     float *out_keys,
                                     float *out_values)
{
    att1_status_t status = ATT1_OK;
    size_t pos = 0u;

    if (mmu == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }

    if ((out_keys == NULL) && (out_values == NULL)) {
        att1_kv_mmu_error(mmu);
        return ATT1_ERR_INVALID_ARG;
    }

    status = att1_kv_mmu_validate_range(mmu,
                                        start_position,
                                        position_count);
    if (status != ATT1_OK) {
        return status;
    }

    if (head_id >= mmu->config.num_heads) {
        att1_kv_mmu_error(mmu);
        return ATT1_ERR_INVALID_ARG;
    }

    for (pos = 0u; pos < position_count; pos++) {
        const size_t output_offset = pos * mmu->config.head_dim;
        float *key_dst = out_keys != NULL ? &out_keys[output_offset] : NULL;
        float *value_dst = out_values != NULL ? &out_values[output_offset] : NULL;

        status = att1_kv_mmu_read(mmu,
                                  session_id,
                                  layer_id,
                                  head_id,
                                  start_position + pos,
                                  key_dst,
                                  value_dst);
        if (status != ATT1_OK) {
            return status;
        }
    }

    mmu->counters.range_copy_ops++;
    return ATT1_OK;
}

att1_status_t att1_kv_mmu_lookup_page(att1_kv_mmu *mmu,
                                      uint64_t session_id,
                                      size_t layer_id,
                                      size_t position,
                                      att1_kv_mmu_page_ref *out_page)
{
    att1_status_t status = ATT1_OK;
    size_t logical_page = 0u;
    att1_kv_mmu_page *page = NULL;

    if (out_page == NULL) {
        att1_kv_mmu_error(mmu);
        return ATT1_ERR_INVALID_ARG;
    }

    memset(out_page, 0, sizeof(*out_page));

    status = att1_kv_mmu_validate_address(mmu, session_id, layer_id);
    if (status != ATT1_OK) {
        return status;
    }

    status = att1_kv_mmu_validate_position(mmu, position);
    if (status != ATT1_OK) {
        return status;
    }

    logical_page = position / mmu->config.page_tokens;
    page = att1_kv_mmu_lookup_existing(mmu, session_id, layer_id, position);
    if (page == NULL) {
        return ATT1_ERR_NOT_FOUND;
    }

    out_page->physical_page = page->physical_page;
    out_page->session_id = page->session_id;
    out_page->layer_id = page->layer_id;
    out_page->logical_page = logical_page;
    out_page->page_tokens = mmu->config.page_tokens;
    return ATT1_OK;
}

void att1_kv_mmu_get_counters(const att1_kv_mmu *mmu,
                              att1_kv_mmu_counters *out_counters)
{
    if ((mmu == NULL) || (out_counters == NULL)) {
        return;
    }

    *out_counters = mmu->counters;
}

void att1_kv_mmu_reset_counters(att1_kv_mmu *mmu)
{
    if (mmu == NULL) {
        return;
    }

    memset(&mmu->counters, 0, sizeof(mmu->counters));
}
