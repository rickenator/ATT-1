#ifndef ATT1_TOKENIZER_H
#define ATT1_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

/*
 * Byte-level tokenizer.
 *
 * Token ids 0..255 map directly to byte values 0x00..0xff. No tokenizer model,
 * merges, unicode normalization, or special tokens are used in Milestone 7.
 */
int att1_tokenizer_encode_bytes(const unsigned char *bytes,
                                size_t byte_count,
                                uint32_t *tokens,
                                size_t token_capacity,
                                size_t *out_token_count);

int att1_tokenizer_decode_byte(uint32_t token, unsigned char *out_byte);

#endif
