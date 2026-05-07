#include "att1_tok_ext.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── internal helpers ─────────────────────────────────────────────────────── */

/*
 * Parse a single decimal token ID from the character range [seg, seg+len).
 *
 * Rejects any segment that does not start with an ASCII digit (no sign, no
 * whitespace).  Rejects trailing non-numeric characters.  Rejects values that
 * exceed ULONG_MAX (strtoul overflow) or UINT32_MAX.  Rejects values that are
 * >= vocab_size.
 *
 * On failure prints a message to stderr before returning the error code.
 *
 * Returns:
 *   ATT1_OK             *out_id set to the parsed value
 *   ATT1_ERR_BAD_FORMAT segment is malformed or value >= vocab_size
 */
static att1_status_t parse_one_id(const char *seg,
                                  size_t      seg_len,
                                  uint32_t    vocab_size,
                                  size_t      position,
                                  uint32_t   *out_id)
{
    char buf[64];
    size_t copy_len = 0u;
    char *endp = NULL;
    unsigned long val = 0u;

    /* copy for display and for NUL-termination required by strtoul */
    copy_len = seg_len < 63u ? seg_len : 63u;
    memcpy(buf, seg, copy_len);
    buf[copy_len] = '\0';

    if ((seg_len == 0u) || ((buf[0] < '0') || (buf[0] > '9'))) {
        fprintf(stderr, "error: malformed token ID '%s' at position %zu\n",
                buf, position);
        return ATT1_ERR_BAD_FORMAT;
    }

    errno = 0;
    val = strtoul(buf, &endp, 10);

    if ((*endp != '\0') || (errno != 0) || (val > (unsigned long)UINT32_MAX)) {
        fprintf(stderr, "error: malformed token ID '%s' at position %zu\n",
                buf, position);
        return ATT1_ERR_BAD_FORMAT;
    }

    if ((uint32_t)val >= vocab_size) {
        fprintf(stderr,
                "error: token ID %lu out of range at position %zu"
                " (vocab_size=%u)\n",
                val, position, vocab_size);
        return ATT1_ERR_BAD_FORMAT;
    }

    *out_id = (uint32_t)val;
    return ATT1_OK;
}

/* ── public functions ─────────────────────────────────────────────────────── */

att1_status_t att1_tok_ext_parse_ids_str(const char  *str,
                                         uint32_t     vocab_size,
                                         uint32_t   **out_ids,
                                         size_t      *out_count)
{
    size_t cap = 0u;
    uint32_t *ids = NULL;
    const char *p = NULL;
    size_t n = 0u;
    att1_status_t rc = ATT1_OK;

    if ((str == NULL) || (str[0] == '\0') ||
        (out_ids == NULL) || (out_count == NULL)) {
        fputs("error: external token IDs string is NULL or empty\n", stderr);
        return ATT1_ERR_INVALID_ARG;
    }

    /* count commas to pre-allocate (number of commas + 1 = number of tokens) */
    {
        const char *s = str;
        cap = 1u;
        while (*s != '\0') {
            if (*s == ',') {
                cap++;
            }
            s++;
        }
    }

    ids = malloc(cap * sizeof(*ids));
    if (ids == NULL) {
        fputs("error: out of memory allocating token ID array\n", stderr);
        return ATT1_ERR_OOM;
    }

    /* scan each comma-delimited segment */
    p = str;
    while (1) {
        const char *seg_start = p;
        const char *seg_end = p;
        uint32_t id = 0u;

        while ((*seg_end != '\0') && (*seg_end != ',')) {
            seg_end++;
        }

        rc = parse_one_id(seg_start,
                          (size_t)(seg_end - seg_start),
                          vocab_size,
                          n + 1u,
                          &id);
        if (rc != ATT1_OK) {
            free(ids);
            return rc;
        }

        ids[n++] = id;

        if (*seg_end == '\0') {
            break;
        }
        p = seg_end + 1u; /* advance past the comma */
    }

    *out_ids = ids;
    *out_count = n;
    return ATT1_OK;
}

att1_status_t att1_tok_ext_parse_ids_file(const char  *path,
                                          uint32_t     vocab_size,
                                          uint32_t   **out_ids,
                                          size_t      *out_count)
{
    FILE *fp = NULL;
    uint32_t *ids = NULL;
    size_t cap = 16u;
    size_t n = 0u;
    size_t lineno = 0u;
    char line[256];
    att1_status_t rc = ATT1_OK;

    if ((path == NULL) || (out_ids == NULL) || (out_count == NULL)) {
        fputs("error: NULL argument passed to att1_tok_ext_parse_ids_file\n",
              stderr);
        return ATT1_ERR_INVALID_ARG;
    }

    fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "error: cannot open token IDs file: %s\n", path);
        return ATT1_ERR_NOT_FOUND;
    }

    ids = malloc(cap * sizeof(*ids));
    if (ids == NULL) {
        fclose(fp);
        fputs("error: out of memory allocating token ID array\n", stderr);
        return ATT1_ERR_OOM;
    }

    while (fgets(line, (int)sizeof(line), fp) != NULL) {
        char *nl = NULL;
        uint32_t id = 0u;
        size_t len = 0u;

        lineno++;

        /* strip newline */
        nl = strchr(line, '\n');
        if (nl != NULL) {
            *nl = '\0';
        }

        /* skip blank lines and comment lines */
        if ((line[0] == '\0') || (line[0] == '#')) {
            continue;
        }

        len = strlen(line);

        rc = parse_one_id(line, len, vocab_size, lineno, &id);
        if (rc != ATT1_OK) {
            free(ids);
            fclose(fp);
            return rc;
        }

        /* grow dynamic array if needed */
        if (n >= cap) {
            uint32_t *new_ids = NULL;
            cap *= 2u;
            new_ids = realloc(ids, cap * sizeof(*new_ids));
            if (new_ids == NULL) {
                free(ids);
                fclose(fp);
                fputs("error: out of memory growing token ID array\n", stderr);
                return ATT1_ERR_OOM;
            }
            ids = new_ids;
        }

        ids[n++] = id;
    }

    fclose(fp);

    if (n == 0u) {
        fprintf(stderr, "error: token IDs file contains no valid IDs: %s\n",
                path);
        free(ids);
        return ATT1_ERR_INVALID_ARG;
    }

    *out_ids = ids;
    *out_count = n;
    return ATT1_OK;
}
