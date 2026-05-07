#ifndef ATT1_TOK_META_H
#define ATT1_TOK_META_H

#include "att1_status.h"

#include <stdint.h>

/*
 * ATT-1 optional tokenizer metadata section (Milestone 54).
 *
 * A fixed-size 96-byte section that summarises tokenizer provenance for
 * converter-side validation and inspection.  The section is NEVER used for
 * runtime tokenization; the byte-level tokenizer remains the default.
 *
 * Present only in version-2 (.att1 header_size == 96) model files.
 * Absent section (tok_meta_size == 0) is valid; .present is 0.
 *
 * Wire layout (96 bytes, all little-endian):
 *   @  0  schema_version       u32   must be ATT1_TOK_META_SCHEMA_VERSION
 *   @  4  tokenizer_type       u32   ATT1_TOK_TYPE_*
 *   @  8  vocab_size           u32   > 0; must match model config vocab_size
 *   @ 12  bos_token_id         i32   -1 = absent; else [0, vocab_size)
 *   @ 16  eos_token_id         i32
 *   @ 20  pad_token_id         i32
 *   @ 24  unk_token_id         i32
 *   @ 28  byte_fallback        u32   0 or 1
 *   @ 32  normalization_policy u32   ATT1_TOK_NORM_*
 *   @ 36  pretokenizer_policy  u32   ATT1_TOK_PRETOK_*
 *   @ 40  asset_hash_kind      u32   ATT1_TOK_HASH_*
 *   @ 44  flags                u32   reserved; must be 0
 *   @ 48  asset_hash[32]       u8    SHA-256 digest (or zeros when absent)
 *   @ 80  asset_offset         u64   0 when no embedded asset
 *   @ 88  asset_size           u64   0 when no embedded asset
 *   ────────────────────────────────
 *         Total                96 bytes
 */

/* Fixed wire size of the tokenizer metadata section. */
#define ATT1_TOK_META_SIZE 96u

/* schema_version value for M54. */
#define ATT1_TOK_META_SCHEMA_VERSION 1u

/* Sentinel for absent special token IDs (stored as int32). */
#define ATT1_TOK_ID_ABSENT (-1)

/* tokenizer_type codes */
#define ATT1_TOK_TYPE_UNKNOWN   0u
#define ATT1_TOK_TYPE_BYTE      1u
#define ATT1_TOK_TYPE_BPE_JSON  2u
#define ATT1_TOK_TYPE_SENTPIECE 3u

/* normalization_policy codes */
#define ATT1_TOK_NORM_UNKNOWN 0u
#define ATT1_TOK_NORM_NONE    1u
#define ATT1_TOK_NORM_UTF8    2u
#define ATT1_TOK_NORM_NFKC    3u
#define ATT1_TOK_NORM_CUSTOM  4u

/* pretokenizer_policy codes */
#define ATT1_TOK_PRETOK_UNKNOWN    0u
#define ATT1_TOK_PRETOK_NONE       1u
#define ATT1_TOK_PRETOK_BYTE_LEVEL 2u
#define ATT1_TOK_PRETOK_WHITESPACE 3u
#define ATT1_TOK_PRETOK_SENTPIECE  4u
#define ATT1_TOK_PRETOK_CUSTOM     5u

/* asset_hash_kind codes */
#define ATT1_TOK_HASH_NONE   0u
#define ATT1_TOK_HASH_SHA256 1u

/*
 * Parsed tokenizer metadata.
 * .present == 0 when the section is absent (tok_meta_size == 0 in header).
 */
typedef struct att1_tok_meta {
    int      present;              /* 0 = absent; 1 = parsed and valid */
    uint32_t schema_version;
    uint32_t tokenizer_type;
    uint32_t vocab_size;
    int32_t  bos_token_id;        /* ATT1_TOK_ID_ABSENT (-1) when not set */
    int32_t  eos_token_id;
    int32_t  pad_token_id;
    int32_t  unk_token_id;
    uint32_t byte_fallback;       /* 0 or 1 */
    uint32_t normalization_policy;
    uint32_t pretokenizer_policy;
    uint32_t asset_hash_kind;
    uint32_t flags;               /* reserved, must be 0 */
    uint8_t  asset_hash[32];
    uint64_t asset_offset;        /* 0 = no embedded asset */
    uint64_t asset_size;          /* 0 = no embedded asset */
} att1_tok_meta;

/*
 * Parse and validate a tokenizer metadata section.
 *
 * Parameters:
 *   section_data  pointer to the first byte of the section
 *   section_size  byte count (must equal ATT1_TOK_META_SIZE)
 *   model_vocab   model config vocab_size for cross-check
 *   out_meta      populated on success; .present set to 1
 *
 * Returns:
 *   ATT1_OK             valid section parsed
 *   ATT1_ERR_INVALID_ARG  NULL pointers
 *   ATT1_ERR_BAD_FORMAT   wrong size, bad version, bad enum, wrong vocab_size,
 *                         bad special-token ID, nonzero reserved flags,
 *                         mismatched asset_offset/asset_size
 */
att1_status_t att1_tok_meta_parse(const unsigned char *section_data,
                                  uint64_t             section_size,
                                  uint32_t             model_vocab,
                                  att1_tok_meta       *out_meta);

/*
 * Return a human-readable name for a tokenizer_type code.
 * Never returns NULL.
 */
const char *att1_tok_type_name(uint32_t tokenizer_type);

/*
 * Return a human-readable name for a normalization_policy code.
 * Never returns NULL.
 */
const char *att1_tok_norm_name(uint32_t policy);

/*
 * Return a human-readable name for a pretokenizer_policy code.
 * Never returns NULL.
 */
const char *att1_tok_pretok_name(uint32_t policy);

/*
 * Validate tokenizer metadata for runtime selection (Milestone 57).
 *
 * Performs explicit selection-time compatibility checks on an already-parsed
 * att1_tok_meta struct.  Called by the runtime tokenizer selection path after
 * the model is loaded; provides defense-in-depth beyond the loader checks in
 * att1_tok_meta_parse().
 *
 * Parameters:
 *   meta         pointer to parsed tokenizer metadata (meta->present must be 1)
 *   model_vocab  model config vocab_size to cross-check against
 *
 * Returns:
 *   ATT1_OK                meta is structurally valid and safe to select
 *   ATT1_ERR_INVALID_ARG   meta is NULL or meta->present == 0
 *   ATT1_ERR_UNSUPPORTED   tokenizer_type is ATT1_TOK_TYPE_UNKNOWN
 *   ATT1_ERR_BAD_FORMAT    any field fails validation (version, vocab mismatch,
 *                          special token ID out of range, bad enum, etc.)
 *
 * Note: ATT1_OK does NOT mean the tokenizer is implemented.  The caller must
 * still check whether the declared tokenizer_type has a runtime implementation.
 */
att1_status_t att1_tok_meta_check_runtime(const att1_tok_meta *meta,
                                          uint32_t             model_vocab);

#endif /* ATT1_TOK_META_H */
