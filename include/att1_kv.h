#ifndef ATT1_KV_H
#define ATT1_KV_H

#include <stddef.h>
#include <stdint.h>

typedef struct att1_kv_page {
    uint64_t logical_page;
    uint64_t physical_page;
    size_t bytes;
} att1_kv_page;

#endif
