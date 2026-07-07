#define _POSIX_C_SOURCE 200112L

/*
 * att1-aimu-transport-bench.c  —  M170 endpoint transport characterization.
 *
 * Measures the M162 Unix-domain-socket endpoint transport through the public
 * M161 conformance API. The result is an empirical anchor for the existing
 * M118 fabric bandwidth/latency simulator. This is not a PCIe benchmark and
 * does not claim hardware latency.
 */

#include "att1_aimu_conformance.h"
#include "att1_aimu_endpoint_client.h"
#include "att1_aimu_endpoint_protocol.h"
#include "att1_aimu_mmio.h"

#include <errno.h>
#include <inttypes.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define REPORT_VERSION 1
#define DEFAULT_ITERATIONS 64u
#define DEFAULT_PAYLOAD_BYTES 4096u
#define MAX_PAYLOAD_BYTES ATT1_AIMU_ENDPOINT_MAX_PAYLOAD

typedef struct bench_args {
    unsigned iterations;
    size_t payload_bytes;
    const char *socket_path;
    const char *report_json;
    int verbose;
} bench_args;

typedef struct sample_stats {
    const char *name;
    uint64_t iterations;
    uint64_t payload_bytes_per_iter;
    double min_us;
    double max_us;
    double avg_us;
    double effective_mib_sec;
} sample_stats;

typedef struct bench_report {
    sample_stats mmio_read32;
    sample_stats command_nop;
    sample_stats fabric_roundtrip;
    sample_stats kv_append_read;
    double m118_base_latency_ns;
    double m118_per_hop_latency_ns;
    double m118_fabric_gib_sec;
} bench_report;

static void usage(FILE *fp)
{
    fprintf(fp,
            "Usage: att1-aimu-transport-bench [--iterations N] [--payload-bytes N]\n"
            "                                [--socket PATH] [--report-json PATH]\n"
            "                                [--verbose]\n");
}

static int parse_uint(const char *s, unsigned *out)
{
    char *end = NULL;
    unsigned long v;

    if ((s == NULL) || (out == NULL) || (s[0] == '\0')) {
        return -1;
    }
    errno = 0;
    v = strtoul(s, &end, 10);
    if ((errno != 0) || (end == s) || (*end != '\0') || (v == 0ul) || (v > 1000000ul)) {
        return -1;
    }
    *out = (unsigned)v;
    return 0;
}

static int parse_size(const char *s, size_t *out)
{
    char *end = NULL;
    unsigned long v;

    if ((s == NULL) || (out == NULL) || (s[0] == '\0')) {
        return -1;
    }
    errno = 0;
    v = strtoul(s, &end, 10);
    if ((errno != 0) || (end == s) || (*end != '\0') ||
        (v == 0ul) || (v > (unsigned long)MAX_PAYLOAD_BYTES)) {
        return -1;
    }
    *out = (size_t)v;
    return 0;
}

static int parse_args(int argc, char **argv, bench_args *args)
{
    int i;

    args->iterations = DEFAULT_ITERATIONS;
    args->payload_bytes = DEFAULT_PAYLOAD_BYTES;
    args->socket_path = NULL;
    args->report_json = NULL;
    args->verbose = 0;

    for (i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "--iterations") == 0) && (i + 1 < argc)) {
            if (parse_uint(argv[++i], &args->iterations) != 0) {
                fprintf(stderr, "transport-bench: invalid --iterations\n");
                return -1;
            }
        } else if ((strcmp(argv[i], "--payload-bytes") == 0) && (i + 1 < argc)) {
            if (parse_size(argv[++i], &args->payload_bytes) != 0) {
                fprintf(stderr, "transport-bench: invalid --payload-bytes\n");
                return -1;
            }
        } else if ((strcmp(argv[i], "--socket") == 0) && (i + 1 < argc)) {
            args->socket_path = argv[++i];
        } else if ((strcmp(argv[i], "--report-json") == 0) && (i + 1 < argc)) {
            args->report_json = argv[++i];
        } else if (strcmp(argv[i], "--verbose") == 0) {
            args->verbose = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            usage(stdout);
            exit(0);
        } else {
            fprintf(stderr, "transport-bench: unknown argument: %s\n", argv[i]);
            return -1;
        }
    }
    return 0;
}

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((double)ts.tv_sec * 1000000.0) + ((double)ts.tv_nsec / 1000.0);
}

static void fill_payload(unsigned char *buf, size_t n)
{
    size_t i;
    for (i = 0u; i < n; ++i) {
        buf[i] = (unsigned char)((i * 31u + 7u) & 0xffu);
    }
}

static pid_t spawn_daemon(const char *socket_path)
{
    pid_t pid = fork();
    if (pid < 0) {
        return -1;
    }
    if (pid == 0) {
        execl("./build/att1-aimu-endpoint",
              "att1-aimu-endpoint",
              "--socket",
              socket_path,
              "--once",
              (char *)NULL);
        _exit(127);
    }
    return pid;
}

static int connect_with_retry(const char *socket_path, att1_aimu_conformance_endpoint **out)
{
    int attempt;
    for (attempt = 0; attempt < 200; ++attempt) {
        if (att1_aimu_conformance_socket_connect(socket_path, out) == ATT1_OK) {
            return 0;
        }
        {
            struct timespec ts;
            ts.tv_sec = 0;
            ts.tv_nsec = 10 * 1000 * 1000L;
            nanosleep(&ts, NULL);
        }
    }
    return -1;
}

static void init_stats(sample_stats *s,
                       const char *name,
                       uint64_t iterations,
                       uint64_t payload_bytes_per_iter)
{
    memset(s, 0, sizeof(*s));
    s->name = name;
    s->iterations = iterations;
    s->payload_bytes_per_iter = payload_bytes_per_iter;
    s->min_us = 1.0e100;
    s->max_us = 0.0;
}

static void add_sample(sample_stats *s, double elapsed_us)
{
    if (elapsed_us < s->min_us) {
        s->min_us = elapsed_us;
    }
    if (elapsed_us > s->max_us) {
        s->max_us = elapsed_us;
    }
    s->avg_us += elapsed_us;
}

static void finish_stats(sample_stats *s)
{
    double seconds;
    double mib;

    if (s->iterations == 0u) {
        s->min_us = 0.0;
        return;
    }
    s->avg_us /= (double)s->iterations;
    seconds = (s->avg_us * (double)s->iterations) / 1000000.0;
    mib = ((double)s->payload_bytes_per_iter * (double)s->iterations) / (1024.0 * 1024.0);
    if (seconds > 0.0) {
        s->effective_mib_sec = mib / seconds;
    }
}

static int bench_mmio(att1_aimu_conformance_endpoint *ep, unsigned iterations, sample_stats *out)
{
    unsigned i;
    uint32_t v = 0u;

    init_stats(out, "mmio_read32", iterations, 0u);
    for (i = 0u; i < iterations; ++i) {
        double t0 = now_us();
        if (att1_aimu_conformance_mmio_read32(ep, ATT1_MMIO_DEVICE_ID, &v) != ATT1_OK) {
            return -1;
        }
        add_sample(out, now_us() - t0);
    }
    finish_stats(out);
    return 0;
}

static int bench_command(att1_aimu_conformance_endpoint *ep, unsigned iterations, sample_stats *out)
{
    unsigned i;

    init_stats(out, "command_nop_submit_dispatch_poll", iterations, 0u);
    for (i = 0u; i < iterations; ++i) {
        att1_aimu_cmd cmd;
        att1_aimu_completion comp;
        double t0;

        memset(&cmd, 0, sizeof(cmd));
        cmd.command_type = ATT1_AIMU_CMD_NOP;
        cmd.tile_id = 0u;
        t0 = now_us();
        if ((att1_aimu_conformance_cmd_submit(ep, &cmd) != ATT1_OK) ||
            (att1_aimu_conformance_cmd_dispatch_one(ep) != ATT1_OK) ||
            (att1_aimu_conformance_cmd_poll_completion(ep, &comp) != ATT1_OK) ||
            (comp.result_code != ATT1_AIMU_OK)) {
            return -1;
        }
        add_sample(out, now_us() - t0);
    }
    finish_stats(out);
    return 0;
}

static int bench_fabric(att1_aimu_conformance_endpoint *ep,
                        unsigned iterations,
                        unsigned char *payload,
                        unsigned char *out_payload,
                        size_t payload_bytes,
                        sample_stats *out)
{
    unsigned i;

    init_stats(out, "fabric_send_receive", iterations, (uint64_t)payload_bytes * 2u);
    for (i = 0u; i < iterations; ++i) {
        att1_fabric_packet packet;
        size_t out_bytes = 0u;
        double t0 = now_us();
        if ((att1_aimu_conformance_fabric_send(ep,
                                               0u,
                                               1u,
                                               ATT1_PACKET_ACTIVATION,
                                               payload,
                                               payload_bytes,
                                               (uint64_t)i) != ATT1_OK) ||
            (att1_aimu_conformance_fabric_receive(ep,
                                                  1u,
                                                  &packet,
                                                  out_payload,
                                                  payload_bytes,
                                                  &out_bytes) != ATT1_OK) ||
            (out_bytes != payload_bytes) ||
            (memcmp(payload, out_payload, payload_bytes) != 0)) {
            return -1;
        }
        add_sample(out, now_us() - t0);
    }
    finish_stats(out);
    return 0;
}

static int bench_kv(att1_aimu_conformance_endpoint *ep, unsigned iterations, sample_stats *out)
{
    unsigned i;
    float key[64];
    float value[64];
    float out_key[16];
    float out_value[16];
    size_t j;
    const uint64_t session_id = 170u;
    const size_t per_position = 64u;
    const size_t head_dim = 16u;

    for (j = 0u; j < per_position; ++j) {
        key[j] = (float)(j + 1u);
        value[j] = (float)(j + 100u);
    }
    if (att1_aimu_conformance_kv_create_session(ep, session_id) != ATT1_OK) {
        return -1;
    }

    init_stats(out, "kv_append_read", iterations, (uint64_t)((per_position + head_dim) * 2u * sizeof(float)));
    for (i = 0u; i < iterations; ++i) {
        double t0 = now_us();
        if ((att1_aimu_conformance_kv_append(ep,
                                             session_id,
                                             0u,
                                             (size_t)i,
                                             key,
                                             per_position,
                                             value,
                                             per_position) != ATT1_OK) ||
            (att1_aimu_conformance_kv_read(ep,
                                           session_id,
                                           0u,
                                           0u,
                                           (size_t)i,
                                           out_key,
                                           head_dim,
                                           out_value,
                                           head_dim) != ATT1_OK)) {
            return -1;
        }
        add_sample(out, now_us() - t0);
    }
    finish_stats(out);
    (void)att1_aimu_conformance_kv_destroy_session(ep, session_id);
    return 0;
}

static void compute_calibration(bench_report *r)
{
    double fabric_one_way_ns = (r->fabric_roundtrip.avg_us * 1000.0) / 2.0;

    r->m118_base_latency_ns = r->mmio_read32.avg_us * 1000.0;
    if (fabric_one_way_ns > r->m118_base_latency_ns) {
        r->m118_per_hop_latency_ns = fabric_one_way_ns - r->m118_base_latency_ns;
    } else {
        r->m118_per_hop_latency_ns = fabric_one_way_ns;
    }
    r->m118_fabric_gib_sec = r->fabric_roundtrip.effective_mib_sec / 1024.0;
}

static void print_stat(const sample_stats *s)
{
    printf("  %-28s iter=%" PRIu64 " avg_us=%.3f min_us=%.3f max_us=%.3f",
           s->name,
           s->iterations,
           s->avg_us,
           s->min_us,
           s->max_us);
    if (s->payload_bytes_per_iter > 0u) {
        printf(" effective_mib_sec=%.3f", s->effective_mib_sec);
    }
    printf("\n");
}

static void print_report(const bench_args *args, const bench_report *r)
{
    printf("att1-aimu-transport-bench report\n");
    printf("  iterations          : %u\n", args->iterations);
    printf("  payload_bytes       : %zu\n", args->payload_bytes);
    print_stat(&r->mmio_read32);
    print_stat(&r->command_nop);
    print_stat(&r->fabric_roundtrip);
    print_stat(&r->kv_append_read);
    printf("  m118_base_latency_ns: %.3f\n", r->m118_base_latency_ns);
    printf("  m118_per_hop_latency_ns: %.3f\n", r->m118_per_hop_latency_ns);
    printf("  m118_fabric_gib_sec : %.6f\n", r->m118_fabric_gib_sec);
}

static void json_stat(FILE *fp, const char *name, const sample_stats *s, int comma)
{
    fprintf(fp,
            "    \"%s\": {\"iterations\": %" PRIu64 ", \"payload_bytes_per_iter\": %" PRIu64
            ", \"avg_us\": %.6f, \"min_us\": %.6f, \"max_us\": %.6f"
            ", \"effective_mib_sec\": %.6f}%s\n",
            name,
            s->iterations,
            s->payload_bytes_per_iter,
            s->avg_us,
            s->min_us,
            s->max_us,
            s->effective_mib_sec,
            comma ? "," : "");
}

static int write_json_report(const char *path, const bench_args *args, const bench_report *r)
{
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    fprintf(fp, "{\n");
    fprintf(fp, "  \"transport_bench_report_version\": %d,\n", REPORT_VERSION);
    fprintf(fp, "  \"transport\": \"unix_domain_socket\",\n");
    fprintf(fp, "  \"endpoint_binary\": \"build/att1-aimu-endpoint\",\n");
    fprintf(fp, "  \"iterations\": %u,\n", args->iterations);
    fprintf(fp, "  \"payload_bytes\": %zu,\n", args->payload_bytes);
    fprintf(fp, "  \"samples\": {\n");
    json_stat(fp, "mmio_read32", &r->mmio_read32, 1);
    json_stat(fp, "command_nop_submit_dispatch_poll", &r->command_nop, 1);
    json_stat(fp, "fabric_send_receive", &r->fabric_roundtrip, 1);
    json_stat(fp, "kv_append_read", &r->kv_append_read, 0);
    fprintf(fp, "  },\n");
    fprintf(fp, "  \"m118_calibration\": {\n");
    fprintf(fp, "    \"base_latency_ns\": %.6f,\n", r->m118_base_latency_ns);
    fprintf(fp, "    \"per_hop_latency_ns\": %.6f,\n", r->m118_per_hop_latency_ns);
    fprintf(fp, "    \"fabric_gib_sec\": %.9f\n", r->m118_fabric_gib_sec);
    fprintf(fp, "  }\n");
    fprintf(fp, "}\n");
    fclose(fp);
    return 0;
}

int main(int argc, char **argv)
{
    bench_args args;
    bench_report report;
    char default_socket[256];
    const char *socket_path;
    pid_t pid;
    int status = 0;
    int exit_code = 1;
    att1_aimu_conformance_endpoint *ep = NULL;
    unsigned char *payload = NULL;
    unsigned char *out_payload = NULL;

    if (parse_args(argc, argv, &args) != 0) {
        usage(stderr);
        return 2;
    }
    snprintf(default_socket,
             sizeof(default_socket),
             "build/att1-aimu-transport-bench-%ld.sock",
             (long)getpid());
    socket_path = (args.socket_path != NULL) ? args.socket_path : default_socket;

    payload = (unsigned char *)malloc(args.payload_bytes);
    out_payload = (unsigned char *)malloc(args.payload_bytes);
    if ((payload == NULL) || (out_payload == NULL)) {
        fprintf(stderr, "transport-bench: allocation failed\n");
        goto cleanup;
    }
    fill_payload(payload, args.payload_bytes);
    memset(out_payload, 0, args.payload_bytes);

    unlink(socket_path);
    pid = spawn_daemon(socket_path);
    if (pid <= 0) {
        fprintf(stderr, "transport-bench: failed to spawn endpoint daemon\n");
        goto cleanup;
    }
    if (connect_with_retry(socket_path, &ep) != 0) {
        fprintf(stderr, "transport-bench: failed to connect to endpoint daemon\n");
        goto cleanup_daemon;
    }

    memset(&report, 0, sizeof(report));
    if ((bench_mmio(ep, args.iterations, &report.mmio_read32) != 0) ||
        (bench_command(ep, args.iterations, &report.command_nop) != 0) ||
        (bench_fabric(ep, args.iterations, payload, out_payload, args.payload_bytes,
                      &report.fabric_roundtrip) != 0) ||
        (bench_kv(ep, args.iterations, &report.kv_append_read) != 0)) {
        fprintf(stderr, "transport-bench: benchmark operation failed\n");
        goto cleanup_daemon;
    }
    compute_calibration(&report);
    print_report(&args, &report);
    if (args.report_json != NULL) {
        if (write_json_report(args.report_json, &args, &report) != 0) {
            fprintf(stderr, "transport-bench: failed to write report JSON: %s\n",
                    args.report_json);
            goto cleanup_daemon;
        }
    }
    if (args.verbose) {
        fprintf(stderr, "transport-bench: socket=%s pid=%ld\n", socket_path, (long)pid);
    }
    exit_code = 0;

cleanup_daemon:
    if (ep != NULL) {
        att1_aimu_conformance_endpoint_destroy(ep);
        ep = NULL;
    }
    if (pid > 0) {
        if (waitpid(pid, &status, WNOHANG) == 0) {
            kill(pid, SIGTERM);
            waitpid(pid, &status, 0);
        }
    }
    unlink(socket_path);
cleanup:
    free(payload);
    free(out_payload);
    return exit_code;
}
