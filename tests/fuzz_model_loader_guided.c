/*
 * ATT-1 M152: coverage-guided model-loader fuzz harness.
 *
 * This file is intentionally separate from fuzz_model_loader.c. The M143
 * harness is deterministic smoke coverage; this harness is for optional
 * libFuzzer/AFL-style runs.
 */

#define _POSIX_C_SOURCE 200809L

#include "att1_model.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int att1_fuzz_write_tmp(const uint8_t *data, size_t size, char *out_path)
{
    int fd = -1;
    ssize_t nw = 0;

    memcpy(out_path, "/tmp/att1_guided_XXXXXX", 24u);
    fd = mkstemp(out_path);
    if (fd < 0) {
        return -1;
    }

    if (size > 0u) {
        nw = write(fd, data, size);
        close(fd);
        if (nw != (ssize_t)size) {
            (void)unlink(out_path);
            return -1;
        }
    } else {
        close(fd);
    }

    return 0;
}

static void att1_fuzz_one_input(const uint8_t *data, size_t size)
{
    char path[32];
    att1_model model;

    if (data == NULL) {
        return;
    }

    if (att1_fuzz_write_tmp(data, size, path) != 0) {
        return;
    }

    memset(&model, 0, sizeof(model));
    (void)att1_model_load(path, &model);
    att1_model_free(&model);
    (void)unlink(path);
}

#ifdef ATT1_LIBFUZZER
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    att1_fuzz_one_input(data, size);
    return 0;
}
#else
int main(int argc, char **argv)
{
    FILE *fh = stdin;
    uint8_t *buf = NULL;
    size_t cap = 0u;
    size_t len = 0u;
    int ch = 0;

    if (argc > 1) {
        fh = fopen(argv[1], "rb");
        if (fh == NULL) {
            return 2;
        }
    }

    while ((ch = fgetc(fh)) != EOF) {
        if (len == cap) {
            size_t next = cap == 0u ? 4096u : cap * 2u;
            uint8_t *tmp = realloc(buf, next);
            if (tmp == NULL) {
                free(buf);
                if (fh != stdin) {
                    fclose(fh);
                }
                return 2;
            }
            buf = tmp;
            cap = next;
        }
        buf[len++] = (uint8_t)ch;
    }

    if (fh != stdin) {
        fclose(fh);
    }

    att1_fuzz_one_input(buf, len);
    free(buf);
    return 0;
}
#endif
