#ifndef ATT1_LOG_H
#define ATT1_LOG_H

#include <stdarg.h>

typedef enum att1_log_level {
    ATT1_LOG_DEBUG = 0,
    ATT1_LOG_INFO,
    ATT1_LOG_WARN,
    ATT1_LOG_ERROR
} att1_log_level;

void att1_log_set_level(att1_log_level level);
att1_log_level att1_log_get_level(void);
void att1_log(att1_log_level level, const char *fmt, ...);
void att1_vlog(att1_log_level level, const char *fmt, va_list args);

#endif
