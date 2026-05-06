#include "att1.h"

#include <stdio.h>

int main(void)
{
    att1_log_set_level(ATT1_LOG_ERROR);

    if (att1_log_get_level() != ATT1_LOG_ERROR) {
        fputs("log level smoke check failed\n", stderr);
        return 1;
    }

    puts("smoke test passed");
    return 0;
}
