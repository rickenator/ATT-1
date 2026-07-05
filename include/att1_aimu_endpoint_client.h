#ifndef ATT1_AIMU_ENDPOINT_CLIENT_H
#define ATT1_AIMU_ENDPOINT_CLIENT_H

/*
 * att1_aimu_endpoint_client.h  —  M162 socket-backed conformance client.
 *
 * Connects to a running `att1-aimu-endpoint` daemon over a Unix domain
 * socket and exposes the connection as an att1_aimu_conformance_endpoint
 * (M161), so the same substrate-independent tests/tools that talk to the
 * in-process simulator can talk to a separate endpoint process instead.
 */

#include "att1_aimu_conformance.h"
#include "att1_status.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Connects to the Unix domain socket at `socket_path` (must already be
 * listening, e.g. an `att1-aimu-endpoint` daemon) and returns a conformance
 * endpoint backed by that connection. The caller owns *out_endpoint and
 * must destroy it with att1_aimu_conformance_endpoint_destroy(), which
 * closes the underlying socket.
 */
att1_status_t att1_aimu_conformance_socket_connect(
        const char *socket_path,
        att1_aimu_conformance_endpoint **out_endpoint);

#ifdef __cplusplus
}
#endif

#endif
