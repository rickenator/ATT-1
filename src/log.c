#include "att1_log.h"

#include <stdio.h>

static att1_log_level g_att1_log_level = ATT1_LOG_INFO;

static const char *att1_log_level_name(att1_log_level level)
{
    switch (level) {
    case ATT1_LOG_DEBUG:
        return "debug";
    case ATT1_LOG_INFO:
        return "info";
    case ATT1_LOG_WARN:
        return "warn";
    case ATT1_LOG_ERROR:
        return "error";
    default:
        return "unknown";
    }
}

void att1_log_set_level(att1_log_level level)
{
    g_att1_log_level = level;
}

att1_log_level att1_log_get_level(void)
{
    return g_att1_log_level;
}

void att1_vlog(att1_log_level level, const char *fmt, va_list args)
{
    if (level < g_att1_log_level) {
        return;
    }

    fprintf(stderr, "att1:%s: ", att1_log_level_name(level));
    vfprintf(stderr, fmt, args);
    fputc('\n', stderr);
}

void att1_log(att1_log_level level, const char *fmt, ...)
{
    va_list args;

    va_start(args, fmt);
    att1_vlog(level, fmt, args);
    va_end(args);
}
