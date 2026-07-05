/*
 * aimu_endpoint_protocol.c  —  M162 endpoint process wire protocol I/O.
 *
 * Minimal, same-architecture, length-implicit (fixed struct size) message
 * transfer helpers shared by the `att1-aimu-endpoint` daemon and the
 * socket-backed conformance client. See att1_aimu_endpoint_protocol.h.
 */

#include "att1_aimu_endpoint_protocol.h"

#include <errno.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

static att1_status_t endpoint_write_all(int fd, const void *buf, size_t len)
{
    const unsigned char *p = (const unsigned char *)buf;
    size_t remaining = len;

    while (remaining > 0u) {
        /*
         * send(..., MSG_NOSIGNAL) instead of write(): if the peer has
         * already closed the connection (e.g. an endpoint daemon crash
         * mid-decode, M168), a plain write() to that socket raises SIGPIPE,
         * whose default disposition kills this process outright -- not a
         * clean att1_status_t error. MSG_NOSIGNAL suppresses that signal so
         * the call instead fails with errno == EPIPE, which is handled
         * below like any other write error.
         */
        ssize_t n = send(fd, p, remaining, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return ATT1_ERR_IO;
        }
        if (n == 0) {
            return ATT1_ERR_IO;
        }
        p += (size_t)n;
        remaining -= (size_t)n;
    }
    return ATT1_OK;
}

static att1_status_t endpoint_read_all(int fd, void *buf, size_t len)
{
    unsigned char *p = (unsigned char *)buf;
    size_t remaining = len;

    while (remaining > 0u) {
        ssize_t n = read(fd, p, remaining);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return ATT1_ERR_IO;
        }
        if (n == 0) {
            /* peer closed the connection */
            return ATT1_ERR_IO;
        }
        p += (size_t)n;
        remaining -= (size_t)n;
    }
    return ATT1_OK;
}

att1_status_t att1_aimu_endpoint_send_request(int fd, const att1_aimu_endpoint_request *req)
{
    if (req == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }
    return endpoint_write_all(fd, req, sizeof(*req));
}

att1_status_t att1_aimu_endpoint_recv_request(int fd, att1_aimu_endpoint_request *req)
{
    if (req == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }
    memset(req, 0, sizeof(*req));
    return endpoint_read_all(fd, req, sizeof(*req));
}

att1_status_t att1_aimu_endpoint_send_response(int fd, const att1_aimu_endpoint_response *resp)
{
    if (resp == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }
    return endpoint_write_all(fd, resp, sizeof(*resp));
}

att1_status_t att1_aimu_endpoint_recv_response(int fd, att1_aimu_endpoint_response *resp)
{
    if (resp == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }
    memset(resp, 0, sizeof(*resp));
    return endpoint_read_all(fd, resp, sizeof(*resp));
}
