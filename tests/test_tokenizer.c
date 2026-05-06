#include "att1_tokenizer.h"

#include <stdio.h>

int main(void)
{
    unsigned char bytes[256];
    uint32_t tokens[256];
    size_t token_count = 0u;
    size_t i = 0u;

    for (i = 0u; i < 256u; i++) {
        bytes[i] = (unsigned char)i;
    }

    if (att1_tokenizer_encode_bytes(bytes,
                                    256u,
                                    tokens,
                                    256u,
                                    &token_count) != 0) {
        fputs("tokenizer encode failed\n", stderr);
        return 1;
    }

    if (token_count != 256u) {
        fputs("tokenizer count failed\n", stderr);
        return 1;
    }

    for (i = 0u; i < 256u; i++) {
        unsigned char byte = 0u;

        if (tokens[i] != (uint32_t)i) {
            fputs("tokenizer token id check failed\n", stderr);
            return 1;
        }

        if ((att1_tokenizer_decode_byte(tokens[i], &byte) != 0) ||
            (byte != bytes[i])) {
            fputs("tokenizer decode check failed\n", stderr);
            return 1;
        }
    }

    if (att1_tokenizer_decode_byte(256u, bytes) == 0) {
        fputs("tokenizer invalid token check failed\n", stderr);
        return 1;
    }

    puts("tokenizer test passed");
    return 0;
}
