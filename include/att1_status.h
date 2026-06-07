#ifndef ATT1_STATUS_H
#define ATT1_STATUS_H

typedef enum att1_status {
    ATT1_OK = 0,
    ATT1_ERR_INVALID_ARG = -1,
    ATT1_ERR_OOM = -2,
    ATT1_ERR_IO = -3,
    ATT1_ERR_BAD_FORMAT = -4,
    ATT1_ERR_NOT_FOUND = -5,
    ATT1_ERR_SHAPE = -6,
    ATT1_ERR_QUEUE_FULL = -7,
    ATT1_ERR_STATE = -8,
    ATT1_ERR_UNSUPPORTED = -9,
    ATT1_ERR_QUEUE_EMPTY = -10,
    ATT1_ERR_TIMEOUT = -11,
    ATT1_ERR_ALREADY_STARTED = -12
} att1_status_t;

#endif
