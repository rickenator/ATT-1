#include "att1_tokenizer.h"

int att1_tokenizer_encode_bytes(const unsigned char *bytes,
                                size_t byte_count,
                                uint32_t *tokens,
                                size_t token_capacity,
                                size_t *out_token_count)
{
    size_t i = 0u;

    if ((bytes == NULL) || (tokens == NULL) || (out_token_count == NULL)) {
        return -1;
    }

    if (token_capacity < byte_count) {
        return -1;
    }

    for (i = 0u; i < byte_count; i++) {
        tokens[i] = (uint32_t)bytes[i];
    }

    *out_token_count = byte_count;
    return 0;
}

int att1_tokenizer_decode_byte(uint32_t token, unsigned char *out_byte)
{
    if (out_byte == NULL) {
        return -1;
    }

    if (token > 255u) {
        return -1;
    }

    *out_byte = (unsigned char)token;
    return 0;
}
