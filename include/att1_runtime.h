#ifndef ATT1_RUNTIME_H
#define ATT1_RUNTIME_H

#include <stdint.h>

typedef enum att1_request_state {
    ATT1_REQUEST_PENDING = 0,
    ATT1_REQUEST_RUNNING,
    ATT1_REQUEST_DONE,
    ATT1_REQUEST_FAILED
} att1_request_state;

typedef struct att1_request {
    uint64_t id;
    att1_request_state state;
} att1_request;

#endif
