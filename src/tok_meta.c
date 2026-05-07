#include "att1_tok_meta.h"

#include <string.h>

static uint32_t tok_read_u32le(const unsigned char *p)
{
    return ((uint32_t)p[0])       |
           ((uint32_t)p[1] << 8u) |
           ((uint32_t)p[2] << 16u)|
           ((uint32_t)p[3] << 24u);
}

static uint64_t tok_read_u64le(const unsigned char *p)
{
    return ((uint64_t)p[0])        |
           ((uint64_t)p[1] << 8u)  |
           ((uint64_t)p[2] << 16u) |
           ((uint64_t)p[3] << 24u) |
           ((uint64_t)p[4] << 32u) |
           ((uint64_t)p[5] << 40u) |
           ((uint64_t)p[6] << 48u) |
           ((uint64_t)p[7] << 56u);
}

static int tok_type_valid(uint32_t v)
{
    return (v <= ATT1_TOK_TYPE_SENTPIECE);
}

static int tok_norm_valid(uint32_t v)
{
    return (v <= ATT1_TOK_NORM_CUSTOM);
}

static int tok_pretok_valid(uint32_t v)
{
    return (v <= ATT1_TOK_PRETOK_CUSTOM);
}

static int tok_hash_valid(uint32_t v)
{
    return (v <= ATT1_TOK_HASH_SHA256);
}

att1_status_t att1_tok_meta_parse(const unsigned char *section_data,
                                  uint64_t             section_size,
                                  uint32_t             model_vocab,
                                  att1_tok_meta       *out_meta)
{
    uint32_t schema_version = 0u;
    uint32_t tokenizer_type = 0u;
    uint32_t vocab_size     = 0u;
    int32_t  bos_id         = 0;
    int32_t  eos_id         = 0;
    int32_t  pad_id         = 0;
    int32_t  unk_id         = 0;
    uint32_t byte_fallback  = 0u;
    uint32_t norm_policy    = 0u;
    uint32_t pretok_policy  = 0u;
    uint32_t hash_kind      = 0u;
    uint32_t flags          = 0u;
    uint64_t asset_offset   = 0u;
    uint64_t asset_size     = 0u;

    if ((section_data == NULL) || (out_meta == NULL)) {
        return ATT1_ERR_INVALID_ARG;
    }

    memset(out_meta, 0, sizeof(*out_meta));

    /* Section must be exactly ATT1_TOK_META_SIZE bytes. */
    if (section_size != ATT1_TOK_META_SIZE) {
        return ATT1_ERR_BAD_FORMAT;
    }

    schema_version = tok_read_u32le(&section_data[0]);
    tokenizer_type = tok_read_u32le(&section_data[4]);
    vocab_size     = tok_read_u32le(&section_data[8]);

    /* Read signed IDs as unsigned first, then reinterpret. */
    {
        uint32_t tmp = 0u;
        tmp = tok_read_u32le(&section_data[12]);
        memcpy(&bos_id, &tmp, sizeof(bos_id));
        tmp = tok_read_u32le(&section_data[16]);
        memcpy(&eos_id, &tmp, sizeof(eos_id));
        tmp = tok_read_u32le(&section_data[20]);
        memcpy(&pad_id, &tmp, sizeof(pad_id));
        tmp = tok_read_u32le(&section_data[24]);
        memcpy(&unk_id, &tmp, sizeof(unk_id));
    }

    byte_fallback = tok_read_u32le(&section_data[28]);
    norm_policy   = tok_read_u32le(&section_data[32]);
    pretok_policy = tok_read_u32le(&section_data[36]);
    hash_kind     = tok_read_u32le(&section_data[40]);
    flags         = tok_read_u32le(&section_data[44]);
    /* asset_hash at [48..79] — 32 bytes */
    asset_offset  = tok_read_u64le(&section_data[80]);
    asset_size    = tok_read_u64le(&section_data[88]);

    /* --- validate --- */

    if (schema_version != ATT1_TOK_META_SCHEMA_VERSION) {
        return ATT1_ERR_BAD_FORMAT;
    }

    if (!tok_type_valid(tokenizer_type)) {
        return ATT1_ERR_BAD_FORMAT;
    }

    if (vocab_size == 0u) {
        return ATT1_ERR_BAD_FORMAT;
    }

    /* Cross-check vocab_size against model config. */
    if (vocab_size != model_vocab) {
        return ATT1_ERR_BAD_FORMAT;
    }

    /* Special token IDs must be -1 (absent) or [0, vocab_size). */
    {
        const int32_t ids[4] = { bos_id, eos_id, pad_id, unk_id };
        uint32_t k = 0u;
        for (k = 0u; k < 4u; k++) {
            if (ids[k] != ATT1_TOK_ID_ABSENT) {
                if ((ids[k] < 0) || ((uint32_t)ids[k] >= vocab_size)) {
                    return ATT1_ERR_BAD_FORMAT;
                }
            }
        }
    }

    if (byte_fallback > 1u) {
        return ATT1_ERR_BAD_FORMAT;
    }

    if (!tok_norm_valid(norm_policy)) {
        return ATT1_ERR_BAD_FORMAT;
    }

    if (!tok_pretok_valid(pretok_policy)) {
        return ATT1_ERR_BAD_FORMAT;
    }

    if (!tok_hash_valid(hash_kind)) {
        return ATT1_ERR_BAD_FORMAT;
    }

    /* Reserved flags must be zero. */
    if (flags != 0u) {
        return ATT1_ERR_BAD_FORMAT;
    }

    /* asset_offset and asset_size must both be zero or both nonzero. */
    if ((asset_offset == 0u) != (asset_size == 0u)) {
        return ATT1_ERR_BAD_FORMAT;
    }

    /* --- populate output --- */
    out_meta->present              = 1;
    out_meta->schema_version       = schema_version;
    out_meta->tokenizer_type       = tokenizer_type;
    out_meta->vocab_size           = vocab_size;
    out_meta->bos_token_id         = bos_id;
    out_meta->eos_token_id         = eos_id;
    out_meta->pad_token_id         = pad_id;
    out_meta->unk_token_id         = unk_id;
    out_meta->byte_fallback        = byte_fallback;
    out_meta->normalization_policy = norm_policy;
    out_meta->pretokenizer_policy  = pretok_policy;
    out_meta->asset_hash_kind      = hash_kind;
    out_meta->flags                = flags;
    memcpy(out_meta->asset_hash, &section_data[48], 32u);
    out_meta->asset_offset         = asset_offset;
    out_meta->asset_size           = asset_size;

    return ATT1_OK;
}

const char *att1_tok_type_name(uint32_t tokenizer_type)
{
    switch (tokenizer_type) {
    case ATT1_TOK_TYPE_UNKNOWN:   return "unknown";
    case ATT1_TOK_TYPE_BYTE:      return "byte";
    case ATT1_TOK_TYPE_BPE_JSON:  return "bpe_json";
    case ATT1_TOK_TYPE_SENTPIECE: return "sentencepiece";
    default:                      return "invalid";
    }
}

const char *att1_tok_norm_name(uint32_t policy)
{
    switch (policy) {
    case ATT1_TOK_NORM_UNKNOWN: return "unknown";
    case ATT1_TOK_NORM_NONE:    return "none";
    case ATT1_TOK_NORM_UTF8:    return "utf8";
    case ATT1_TOK_NORM_NFKC:    return "nfkc";
    case ATT1_TOK_NORM_CUSTOM:  return "custom";
    default:                    return "invalid";
    }
}

const char *att1_tok_pretok_name(uint32_t policy)
{
    switch (policy) {
    case ATT1_TOK_PRETOK_UNKNOWN:    return "unknown";
    case ATT1_TOK_PRETOK_NONE:       return "none";
    case ATT1_TOK_PRETOK_BYTE_LEVEL: return "byte_level";
    case ATT1_TOK_PRETOK_WHITESPACE: return "whitespace";
    case ATT1_TOK_PRETOK_SENTPIECE:  return "sentencepiece";
    case ATT1_TOK_PRETOK_CUSTOM:     return "custom";
    default:                         return "invalid";
    }
}
