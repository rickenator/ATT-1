/*
 * test_aimu_cluster_decode.c  —  M167 two-tile emulated cluster decode.
 *
 * Exercises the M167 "Two-tile emulated cluster decode" milestone in two
 * complementary ways:
 *
 *   1. `test_two_tile_decode_inprocess`: two independent *in-process*
 *      att1_aimu_conformance_endpoint instances (M161), each backing its
 *      own att1_backend_pcie (M163/M166), really execute a tiny two-layer
 *      transformer's math end to end (LOAD_TENSOR_TILE/EXEC_MATMUL/
 *      EXEC_RMSNORM/EXEC_ROPE/EXEC_FFN, M166) split one layer per tile,
 *      cooperating via att1_aimu_cluster_bridge (M167) for activation
 *      routing (M93 §8.2-2), a two-participant fabric barrier
 *      (M93 §8.2-4), and a row-parallel partial-logit reduction over the
 *      output projection's contraction dimension. The combined logits are
 *      compared against a direct cpu-f32 reference and must match exactly.
 *
 *   2. `test_two_tile_decode_socket`: the same activation-routing/barrier/
 *      reduction protocol, driven by att1_aimu_cluster_bridge, over two
 *      real `att1-aimu-endpoint` daemon *processes* connected via separate
 *      Unix domain sockets (M162), proving the transport-agnostic fabric
 *      protocol genuinely crosses OS process boundaries (M93 §8.8). Real
 *      EXEC_* tensor-math execution is intentionally *not* exercised on
 *      these socket-backed endpoints: M166's exec hook resolves tensor
 *      operands via raw host pointers embedded in the command packet
 *      (`input_buf_addr`/`output_buf_addr`/the resident-tensor registry's
 *      `host_addr`), which are only valid within the process that issued
 *      them; forwarding those same pointer values verbatim to a *different*
 *      OS process (the daemon) and dereferencing them there is undefined
 *      behavior. Giving the socket-backed endpoint real device-local
 *      tensor memory (so tensor payloads are transferred and dereferenced
 *      only within the daemon's own address space) is future work, not
 *      M167 scope (M167 is scoped to fabric/barrier/reduction protocol
 *      correctness, M93 §8.2-2/8.2-4); this test therefore reuses the
 *      cpu-f32-computed reference values as the "per-tile activations"
 *      being routed, so only genuine socket traffic is exercised for the
 *      fabric/barrier calls, and asserts the daemons' own fabric counters
 *      (queried back over the same sockets) reflect that real traffic.
 */

#define _POSIX_C_SOURCE 200112L

#include "att1_aimu_cluster_bridge.h"
#include "att1_aimu_conformance.h"
#include "att1_aimu_endpoint_client.h"
#include "att1_backend.h"
#include "att1_kv_cache.h"
#include "att1_transformer_block.h"

#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define PASS(name) do { printf("PASS: aimu_cluster_decode: %s\n", (name)); } while (0)
#define FAIL(name) do { printf("FAIL: aimu_cluster_decode: %s\n", (name)); return 1; } while (0)
#define REQUIRE(cond, name) do { if (!(cond)) { FAIL(name); } } while (0)

#define M167_MODEL_DIM  4u
#define M167_NUM_HEADS  2u
#define M167_HEAD_DIM   2u
#define M167_FFN_DIM    4u
#define M167_VOCAB      4u
#define M167_HALF_DIM   (M167_MODEL_DIM / 2u)

static int near_f32(float lhs, float rhs)
{
    return fabsf(lhs - rhs) < 0.00001f;
}

static void m167_fill_pattern(float *arr, size_t n, float seed)
{
    size_t i;
    for (i = 0u; i < n; i++) {
        arr[i] = seed + ((float)i * 0.05f);
    }
}

typedef struct m167_layer {
    float attention_norm[M167_MODEL_DIM];
    float ffn_norm[M167_MODEL_DIM];
    float wq[M167_MODEL_DIM * M167_MODEL_DIM];
    float wk[M167_MODEL_DIM * M167_MODEL_DIM];
    float wv[M167_MODEL_DIM * M167_MODEL_DIM];
    float wo[M167_MODEL_DIM * M167_MODEL_DIM];
    float w_gate[M167_MODEL_DIM * M167_FFN_DIM];
    float w_up[M167_MODEL_DIM * M167_FFN_DIM];
    float w_down[M167_FFN_DIM * M167_MODEL_DIM];
} m167_layer;

static void m167_build_layer(m167_layer *layer, float seed)
{
    size_t i;

    m167_fill_pattern(layer->attention_norm, M167_MODEL_DIM, seed + 0.1f);
    m167_fill_pattern(layer->ffn_norm, M167_MODEL_DIM, seed + 0.2f);
    m167_fill_pattern(layer->wq, M167_MODEL_DIM * M167_MODEL_DIM, seed + 0.3f);
    m167_fill_pattern(layer->wk, M167_MODEL_DIM * M167_MODEL_DIM, seed + 0.4f);
    m167_fill_pattern(layer->wv, M167_MODEL_DIM * M167_MODEL_DIM, seed + 0.5f);
    m167_fill_pattern(layer->wo, M167_MODEL_DIM * M167_MODEL_DIM, seed + 0.6f);
    m167_fill_pattern(layer->w_gate, M167_MODEL_DIM * M167_FFN_DIM, seed + 0.7f);
    m167_fill_pattern(layer->w_up, M167_MODEL_DIM * M167_FFN_DIM, seed + 0.8f);
    m167_fill_pattern(layer->w_down, M167_FFN_DIM * M167_MODEL_DIM, seed + 0.9f);

    for (i = 0u; i < M167_MODEL_DIM; i++) {
        layer->attention_norm[i] += 1.0f;
        layer->ffn_norm[i] += 1.0f;
    }
}

static void m167_weights(const m167_layer *layer, att1_transformer_block_weights *w)
{
    w->attention_norm = layer->attention_norm;
    w->ffn_norm = layer->ffn_norm;
    w->wq = layer->wq;
    w->wk = layer->wk;
    w->wv = layer->wv;
    w->wo = layer->wo;
    w->w_gate = layer->w_gate;
    w->w_up = layer->w_up;
    w->w_down = layer->w_down;
}

static void m167_block_config(att1_transformer_block_config *cfg)
{
    cfg->model_dim = M167_MODEL_DIM;
    cfg->num_heads = M167_NUM_HEADS;
    cfg->head_dim = M167_HEAD_DIM;
    cfg->ffn_dim = M167_FFN_DIM;
    cfg->rms_epsilon = 1e-6f;
    cfg->rope_theta = 10000.0f;
}

/*
 * Fixed two-layer tiny transformer fixture shared by both tests, plus the
 * cpu-f32 reference values every variant is checked against.
 */
typedef struct m167_fixture {
    m167_layer layer0;
    m167_layer layer1;
    float output_norm[M167_MODEL_DIM];
    float output_weight[M167_VOCAB * M167_MODEL_DIM]; /* [model_dim x vocab], k-major */
    float input[M167_MODEL_DIM];

    /* cpu-f32 reference (single source of truth for both tests). */
    float ref_hidden_after_l0[M167_MODEL_DIM];
    float ref_hidden_after_l1[M167_MODEL_DIM];
    float ref_normed[M167_MODEL_DIM];
    float ref_logits[M167_VOCAB];
} m167_fixture;

static int m167_build_fixture(m167_fixture *fx)
{
    att1_backend *cpu = NULL;
    att1_kv_cache cpu_kv0, cpu_kv1;
    att1_transformer_block_config cfg;
    att1_transformer_block_weights w0, w1;
    size_t i;

    memset(fx, 0, sizeof(*fx));
    memset(&cpu_kv0, 0, sizeof(cpu_kv0));
    memset(&cpu_kv1, 0, sizeof(cpu_kv1));

    m167_build_layer(&fx->layer0, 0.0f);
    m167_build_layer(&fx->layer1, 1.0f);
    m167_weights(&fx->layer0, &w0);
    m167_weights(&fx->layer1, &w1);
    m167_block_config(&cfg);

    m167_fill_pattern(fx->output_norm, M167_MODEL_DIM, 0.15f);
    for (i = 0u; i < M167_MODEL_DIM; i++) {
        fx->output_norm[i] += 1.0f;
    }
    m167_fill_pattern(fx->output_weight, M167_VOCAB * M167_MODEL_DIM, 0.25f);
    m167_fill_pattern(fx->input, M167_MODEL_DIM, 1.0f);

    if (att1_backend_cpu_f32_create(&cpu) != ATT1_OK) {
        return -1;
    }
    if ((att1_kv_cache_init(&cpu_kv0, 1u, M167_NUM_HEADS, M167_HEAD_DIM) != ATT1_OK) ||
        (att1_kv_cache_init(&cpu_kv1, 1u, M167_NUM_HEADS, M167_HEAD_DIM) != ATT1_OK)) {
        att1_backend_destroy(cpu);
        return -1;
    }

    if (att1_transformer_block_forward_backend(fx->ref_hidden_after_l0, &cpu_kv0,
                                               fx->input, &w0, &cfg, 0u, cpu) != 0) {
        return -1;
    }
    if (att1_transformer_block_forward_backend(fx->ref_hidden_after_l1, &cpu_kv1,
                                               fx->ref_hidden_after_l0, &w1, &cfg, 0u,
                                               cpu) != 0) {
        return -1;
    }
    if (cpu->ops->rmsnorm_f32(cpu, fx->ref_normed, fx->ref_hidden_after_l1,
                              fx->output_norm, M167_MODEL_DIM, 1e-6f) != 0) {
        return -1;
    }
    if (cpu->ops->matmul_f32(cpu, fx->ref_logits, fx->ref_normed, fx->output_weight,
                             1u, M167_VOCAB, M167_MODEL_DIM) != 0) {
        return -1;
    }

    att1_kv_cache_free(&cpu_kv0);
    att1_kv_cache_free(&cpu_kv1);
    att1_backend_destroy(cpu);
    return 0;
}

/*
 * Drives the M167 activation-routing / barrier / partial-logit-reduction
 * protocol across two already-connected endpoints (ep0 hosting "tile 0",
 * ep1 hosting "tile 1"), using att1_aimu_cluster_bridge for every
 * cross-tile step. When `pcie0`/`pcie1` are non-NULL, each tile's layer
 * forward pass and partial-logit matmul are *really* executed on that
 * tile's backend (M166 EXEC_*); when NULL, the fixture's precomputed
 * cpu-f32 reference values stand in for that tile's local computation, so
 * only the fabric traffic itself (send/receive/barrier) is real. Returns 0
 * on success (via REQUIRE/FAIL macros), 1 on any failure.
 */
static int m167_run_protocol(const m167_fixture *fx,
                             att1_aimu_conformance_endpoint *ep0,
                             att1_aimu_conformance_endpoint *ep1,
                             att1_backend *pcie0,
                             att1_backend *pcie1,
                             float *out_combined_logits)
{
    att1_aimu_cluster_bridge bridge;
    att1_transformer_block_config cfg;
    att1_transformer_block_weights w0, w1;
    att1_kv_cache pcie_kv0, pcie_kv1;
    float hidden_after_l0[M167_MODEL_DIM];
    float hidden_after_l1[M167_MODEL_DIM];
    float normed_tile0[M167_MODEL_DIM];
    float normed_tile1[M167_MODEL_DIM];
    float partial_logits_0[M167_VOCAB];
    float partial_logits_1[M167_VOCAB];
    const float *w_half0 = &fx->output_weight[0];
    const float *w_half1 = &fx->output_weight[M167_HALF_DIM * M167_VOCAB];
    att1_fabric_packet packet;
    size_t recv_bytes = 0u;
    size_t i;
    int barrier_complete = 0;

    memset(&pcie_kv0, 0, sizeof(pcie_kv0));
    memset(&pcie_kv1, 0, sizeof(pcie_kv1));
    m167_block_config(&cfg);

    REQUIRE(att1_aimu_cluster_bridge_init(&bridge, ep0, ep1) == ATT1_OK,
            "protocol: bridge init");

    /* --- tile 0: layer 0 --- */
    if (pcie0 != NULL) {
        m167_weights(&fx->layer0, &w0);
        REQUIRE(att1_kv_cache_init(&pcie_kv0, 1u, M167_NUM_HEADS, M167_HEAD_DIM) == ATT1_OK,
                "protocol: tile0 kv cache init");
        REQUIRE(att1_transformer_block_forward_backend(hidden_after_l0, &pcie_kv0,
                                                        fx->input, &w0, &cfg, 0u,
                                                        pcie0) == 0,
                "protocol: tile0 layer0 forward");
        REQUIRE(near_f32(hidden_after_l0[0], fx->ref_hidden_after_l0[0]),
                "protocol: tile0 layer0 forward matches cpu-f32 reference");
    } else {
        memcpy(hidden_after_l0, fx->ref_hidden_after_l0, sizeof(hidden_after_l0));
    }

    /* Activation routing: tile0 -> tile1 (M93 §8.2-2). */
    REQUIRE(att1_aimu_cluster_bridge_send(&bridge, 1, ATT1_PACKET_ACTIVATION,
                                         hidden_after_l0, sizeof(hidden_after_l0),
                                         0u) == ATT1_OK,
            "protocol: activation send tile0->tile1");
    memset(&packet, 0, sizeof(packet));
    REQUIRE(att1_aimu_conformance_fabric_receive(ep1, ATT1_CLUSTER_BRIDGE_LOCAL_TILE,
                                                 &packet, hidden_after_l1,
                                                 sizeof(hidden_after_l1),
                                                 &recv_bytes) == ATT1_OK &&
            recv_bytes == sizeof(hidden_after_l0) &&
            packet.type == ATT1_PACKET_ACTIVATION,
            "protocol: tile1 receives activation from tile0");

    /* --- tile 1: layer 1 + final rmsnorm --- */
    if (pcie1 != NULL) {
        m167_weights(&fx->layer1, &w1);
        REQUIRE(att1_kv_cache_init(&pcie_kv1, 1u, M167_NUM_HEADS, M167_HEAD_DIM) == ATT1_OK,
                "protocol: tile1 kv cache init");
        REQUIRE(att1_transformer_block_forward_backend(hidden_after_l1, &pcie_kv1,
                                                        hidden_after_l1, &w1, &cfg, 0u,
                                                        pcie1) == 0,
                "protocol: tile1 layer1 forward");
        REQUIRE(pcie1->ops->rmsnorm_f32(pcie1, normed_tile1, hidden_after_l1,
                                        fx->output_norm, M167_MODEL_DIM, 1e-6f) == 0,
                "protocol: tile1 final rmsnorm");
    } else {
        memcpy(normed_tile1, fx->ref_normed, sizeof(normed_tile1));
    }

    /* Route the normed hidden state back to tile0 so both tiles hold an
     * identical input to the (split) output projection. */
    REQUIRE(att1_aimu_cluster_bridge_send(&bridge, 0, ATT1_PACKET_ACTIVATION,
                                         normed_tile1, sizeof(normed_tile1),
                                         1u) == ATT1_OK,
            "protocol: normed activation send tile1->tile0");
    memset(&packet, 0, sizeof(packet));
    REQUIRE(att1_aimu_conformance_fabric_receive(ep0, ATT1_CLUSTER_BRIDGE_LOCAL_TILE,
                                                 &packet, normed_tile0,
                                                 sizeof(normed_tile0),
                                                 &recv_bytes) == ATT1_OK &&
            recv_bytes == sizeof(normed_tile1) &&
            packet.type == ATT1_PACKET_ACTIVATION,
            "protocol: tile0 receives normed activation from tile1");

    /* --- barrier rendezvous before combining (M93 §8.2-4) --- */
    REQUIRE(att1_aimu_cluster_bridge_barrier(&bridge, 1, &barrier_complete) == ATT1_OK,
            "protocol: tile0 barrier arrive");
    REQUIRE(barrier_complete == 0, "protocol: barrier not complete after one side");
    REQUIRE(att1_aimu_cluster_bridge_barrier(&bridge, 0, &barrier_complete) == ATT1_OK,
            "protocol: tile1 barrier arrive");
    REQUIRE(barrier_complete == 1, "protocol: barrier complete after both sides");
    REQUIRE(bridge.barriers_completed == 1u, "protocol: barrier counter incremented");

    /* --- partial-logit reduction (row-parallel split of the contraction
     * dimension; matmul_f32's rhs is k-major, see src/matmul.c, so each
     * tile's half is a contiguous slice of output_weight) --- */
    if (pcie0 != NULL) {
        REQUIRE(pcie0->ops->matmul_f32(pcie0, partial_logits_0, normed_tile0, w_half0,
                                       1u, M167_VOCAB, M167_HALF_DIM) == 0,
                "protocol: tile0 partial logits");
    } else {
        for (i = 0u; i < M167_VOCAB; i++) {
            size_t k;
            partial_logits_0[i] = 0.0f;
            for (k = 0u; k < M167_HALF_DIM; k++) {
                partial_logits_0[i] += normed_tile0[k] * w_half0[(k * M167_VOCAB) + i];
            }
        }
    }
    if (pcie1 != NULL) {
        /* &normed_tile1[M167_HALF_DIM] is in-bounds: normed_tile1 has
         * M167_MODEL_DIM (4) elements, M167_HALF_DIM is 2, and matmul_f32
         * only reads `inner` (== M167_HALF_DIM == 2) elements starting at
         * that offset (indices 2 and 3), i.e. the second half of the
         * contraction dimension. */
        REQUIRE(pcie1->ops->matmul_f32(pcie1, partial_logits_1, &normed_tile1[M167_HALF_DIM],
                                       w_half1, 1u, M167_VOCAB, M167_HALF_DIM) == 0,
                "protocol: tile1 partial logits");
    } else {
        for (i = 0u; i < M167_VOCAB; i++) {
            size_t k;
            partial_logits_1[i] = 0.0f;
            for (k = 0u; k < M167_HALF_DIM; k++) {
                partial_logits_1[i] += normed_tile1[M167_HALF_DIM + k] *
                                      w_half1[(k * M167_VOCAB) + i];
            }
        }
    }

    REQUIRE(att1_aimu_cluster_bridge_send(&bridge, 0, ATT1_PACKET_LOGITS,
                                         partial_logits_1, sizeof(partial_logits_1),
                                         2u) == ATT1_OK,
            "protocol: partial logits send tile1->tile0");
    memset(&packet, 0, sizeof(packet));
    {
        float relayed_partial[M167_VOCAB];
        REQUIRE(att1_aimu_conformance_fabric_receive(ep0, ATT1_CLUSTER_BRIDGE_LOCAL_TILE,
                                                     &packet, relayed_partial,
                                                     sizeof(relayed_partial),
                                                     &recv_bytes) == ATT1_OK &&
                recv_bytes == sizeof(partial_logits_1) &&
                packet.type == ATT1_PACKET_LOGITS,
                "protocol: tile0 receives partial logits from tile1");
        for (i = 0u; i < M167_VOCAB; i++) {
            out_combined_logits[i] = partial_logits_0[i] + relayed_partial[i];
        }
    }

    if (pcie0 != NULL) {
        att1_kv_cache_free(&pcie_kv0);
    }
    if (pcie1 != NULL) {
        att1_kv_cache_free(&pcie_kv1);
    }
    return 0;
}

static int test_two_tile_decode_inprocess(void)
{
    m167_fixture fx;
    att1_aimu_conformance_endpoint *ep0 = NULL;
    att1_aimu_conformance_endpoint *ep1 = NULL;
    att1_aimu_conformance_config config;
    att1_backend *pcie0 = NULL;
    att1_backend *pcie1 = NULL;
    float combined_logits[M167_VOCAB];
    size_t i;
    int rc;

    REQUIRE(m167_build_fixture(&fx) == 0, "two_tile_inprocess: fixture build");

    att1_aimu_conformance_default_config(&config);
    config.tile_count = ATT1_CLUSTER_BRIDGE_TILE_COUNT;

    REQUIRE(att1_aimu_conformance_inproc_create(&config, &ep0) == ATT1_OK,
            "two_tile_inprocess: endpoint0 create");
    REQUIRE(att1_aimu_conformance_inproc_create(&config, &ep1) == ATT1_OK,
            "two_tile_inprocess: endpoint1 create");
    REQUIRE(att1_backend_pcie_create(ep0, 0u, &pcie0) == ATT1_OK,
            "two_tile_inprocess: pcie backend0 create");
    REQUIRE(att1_backend_pcie_create(ep1, 0u, &pcie1) == ATT1_OK,
            "two_tile_inprocess: pcie backend1 create");

    rc = m167_run_protocol(&fx, ep0, ep1, pcie0, pcie1, combined_logits);

    att1_backend_destroy(pcie0);
    att1_backend_destroy(pcie1);
    att1_aimu_conformance_endpoint_destroy(ep0);
    att1_aimu_conformance_endpoint_destroy(ep1);

    if (rc != 0) {
        return rc;
    }

    for (i = 0u; i < M167_VOCAB; i++) {
        REQUIRE(near_f32(combined_logits[i], fx.ref_logits[i]),
                "two_tile_inprocess: combined pcie logits match cpu-f32 reference");
    }

    PASS("two_tile_decode_inprocess");
    return 0;
}

static const char *g_socket_path_0 = "build/test_aimu_cluster_decode_0.sock";
static const char *g_socket_path_1 = "build/test_aimu_cluster_decode_1.sock";

static pid_t spawn_daemon(const char *socket_path)
{
    pid_t pid = fork();
    if (pid < 0) {
        return -1;
    }
    if (pid == 0) {
        execl("./build/att1-aimu-endpoint", "att1-aimu-endpoint",
              "--socket", socket_path, "--tiles", "2", "--once", (char *)NULL);
        _exit(127);
    }
    return pid;
}

static int connect_with_retry(const char *socket_path, att1_aimu_conformance_endpoint **out)
{
    int attempt;
    for (attempt = 0; attempt < 100; ++attempt) {
        if (att1_aimu_conformance_socket_connect(socket_path, out) == ATT1_OK) {
            return 0;
        }
        struct timespec ts = {0, 10 * 1000 * 1000L}; /* 10ms */
        nanosleep(&ts, NULL);
    }
    return -1;
}

static int test_two_tile_decode_socket(void)
{
    m167_fixture fx;
    att1_aimu_conformance_endpoint *ep0 = NULL;
    att1_aimu_conformance_endpoint *ep1 = NULL;
    att1_fabric_counters fc0, fc1;
    pid_t pid0, pid1;
    int status0 = 0, status1 = 0;
    float combined_logits[M167_VOCAB];
    size_t i;
    int rc;

    REQUIRE(m167_build_fixture(&fx) == 0, "two_tile_socket: fixture build");

    unlink(g_socket_path_0);
    unlink(g_socket_path_1);

    pid0 = spawn_daemon(g_socket_path_0);
    REQUIRE(pid0 > 0, "two_tile_socket: daemon0 spawn");
    pid1 = spawn_daemon(g_socket_path_1);
    REQUIRE(pid1 > 0, "two_tile_socket: daemon1 spawn");

    REQUIRE(connect_with_retry(g_socket_path_0, &ep0) == 0,
            "two_tile_socket: client0 connect");
    REQUIRE(connect_with_retry(g_socket_path_1, &ep1) == 0,
            "two_tile_socket: client1 connect");

    /* pcie0/pcie1 == NULL: only the fabric traffic (send/receive/barrier)
     * crosses the real socket transport; see the file header comment for
     * why real EXEC_* tensor math is out of scope for the socket-backed
     * endpoint until it has real device-local tensor memory. */
    rc = m167_run_protocol(&fx, ep0, ep1, NULL, NULL, combined_logits);

    if (rc == 0) {
        for (i = 0u; i < M167_VOCAB; i++) {
            if (!near_f32(combined_logits[i], fx.ref_logits[i])) {
                rc = 1;
            }
        }
        if (rc != 0) {
            FAIL("two_tile_socket: combined logits match cpu-f32 reference");
        }
    }

    if (rc == 0) {
        /* m167_run_protocol() performs exactly one barrier rendezvous: each
         * endpoint receives one real-tile arrival (tile 0) and one
         * bridge-completed proxy-tile arrival (tile 1), so each endpoint's
         * barrier_arrivals is exactly 2 and the two-endpoint total is
         * exactly 4. */
        REQUIRE(att1_aimu_conformance_fabric_get_counters(ep0, &fc0) == ATT1_OK &&
                att1_aimu_conformance_fabric_get_counters(ep1, &fc1) == ATT1_OK &&
                (fc0.packets_sent + fc0.packets_received) > 0u &&
                (fc1.packets_sent + fc1.packets_received) > 0u &&
                (fc0.barrier_arrivals + fc1.barrier_arrivals) == 4u,
                "two_tile_socket: daemon fabric counters reflect real socket traffic");
    }

    att1_aimu_conformance_endpoint_destroy(ep0);
    att1_aimu_conformance_endpoint_destroy(ep1);
    waitpid(pid0, &status0, 0);
    waitpid(pid1, &status1, 0);
    unlink(g_socket_path_0);
    unlink(g_socket_path_1);

    if (rc != 0) {
        return rc;
    }
    PASS("two_tile_decode_socket");
    return 0;
}

int main(void)
{
    int failed = 0;

    failed |= test_two_tile_decode_inprocess();
    failed |= test_two_tile_decode_socket();

    if (failed) {
        printf("FAIL: aimu_cluster_decode suite\n");
        return 1;
    }
    printf("PASS: aimu_cluster_decode suite\n");
    return 0;
}
