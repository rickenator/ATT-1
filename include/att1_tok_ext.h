#ifndef ATT1_TOK_EXT_H
#define ATT1_TOK_EXT_H

#include "att1_status.h"

#include <stddef.h>
#include <stdint.h>

/*
 * Parse a comma-separated string of decimal token IDs for external tokenizer
 * mode (supplied via --input-token-ids).
 *
 * Accepted format: "1,2,3,4".  Each token ID must be a decimal integer in
 * [0, vocab_size).  No leading/trailing whitespace is accepted around IDs.
 * str must be non-NULL and non-empty; at least one token ID is required.
 *
 * On success allocates *out_ids (caller must free()) and stores the element
 * count in *out_count.  On failure writes a human-readable message to stderr.
 *
 * Returns:
 *   ATT1_OK              parsed OK; *out_ids allocated, *out_count set
 *   ATT1_ERR_INVALID_ARG str is NULL/empty, or out_ids/out_count are NULL
 *   ATT1_ERR_BAD_FORMAT  a segment is not a valid non-negative decimal integer,
 *                        or a token ID is >= vocab_size
 *   ATT1_ERR_OOM         allocation failed
 */
att1_status_t att1_tok_ext_parse_ids_str(const char  *str,
                                         uint32_t     vocab_size,
                                         uint32_t   **out_ids,
                                         size_t      *out_count);

/*
 * Read token IDs from a plain-text file for external tokenizer mode
 * (supplied via --tokens-file).
 *
 * Format: one decimal integer per line.  Lines that are blank or whose first
 * non-NUL character is '#' are ignored.  Lines must not exceed 255 bytes.
 * At least one token ID must be present.  Same range rules apply.
 *
 * On success allocates *out_ids (caller must free()) and stores the element
 * count in *out_count.  On failure writes a human-readable message to stderr.
 *
 * Returns:
 *   ATT1_OK              all data lines parsed; *out_ids allocated, count set
 *   ATT1_ERR_INVALID_ARG path/out_ids/out_count is NULL; file yields no IDs
 *   ATT1_ERR_NOT_FOUND   file cannot be opened
 *   ATT1_ERR_BAD_FORMAT  a line is not a valid non-negative decimal integer,
 *                        or a token ID is >= vocab_size
 *   ATT1_ERR_OOM         allocation failed
 */
att1_status_t att1_tok_ext_parse_ids_file(const char  *path,
                                          uint32_t     vocab_size,
                                          uint32_t   **out_ids,
                                          size_t      *out_count);

#endif /* ATT1_TOK_EXT_H */
