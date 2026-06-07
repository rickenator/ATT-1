/*
 * aimu_cmdq.c  —  AIMU/PCIe command-queue simulator (M105)
 *
 * In-process ring-buffer implementation.  See include/att1_aimu_cmdq.h for
 * the public API and docs/aimu_pcie_command_requirements.md for the protocol
 * specification.
 */

#include "att1_aimu_cmdq.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Internal helpers
 * ---------------------------------------------------------------------- */

/* CRC-32/ISO-HDLC (same polynomial used by M103 §3.2 checksum field). */
static uint32_t crc32_update(uint32_t crc, const uint8_t *data, size_t len)
{
    static const uint32_t table[256] = {
        0x00000000u, 0x77073096u, 0xEE0E612Cu, 0x990951BAu,
        0x076DC419u, 0x706AF48Fu, 0xE963A535u, 0x9E6495A3u,
        0x0EDB8832u, 0x79DCB8A4u, 0xE0D5E91Bu, 0x97D2D988u,
        0x09B64C2Bu, 0x7EB17CBFu, 0xE7B82D09u, 0x90BF1CBCu,
        0x1DB71064u, 0x6AB020F2u, 0xF3B97148u, 0x84BE41DEu,
        0x1ADAD47Du, 0x6DDDE4EBu, 0xF4D4B551u, 0x83D385C7u,
        0x136C9856u, 0x646BA8C0u, 0xFD62F97Au, 0x8A65C9ECu,
        0x14015C4Fu, 0x63066CD9u, 0xFA0F3D63u, 0x8D080DF5u,
        0x3B6E20C8u, 0x4C69105Eu, 0xD56041E4u, 0xA2677172u,
        0x3C03E4D1u, 0x4B04D447u, 0xD20D85FDu, 0xA50AB56Bu,
        0x35B5A8FAu, 0x42B2986Cu, 0xDBBBC9D6u, 0xACBCF940u,
        0x32D86CE3u, 0x45DF5C75u, 0xDCD60DCFu, 0xABD13D59u,
        0x26D930ACu, 0x51DE003Au, 0xC8D75180u, 0xBFD06116u,
        0x21B4F6B5u, 0x56B3C423u, 0xCFBA9599u, 0xB8BDA50Fu,
        0x2802B89Eu, 0x5F058808u, 0xC60CD9B2u, 0xB10BE924u,
        0x2F6F7C87u, 0x58684C11u, 0xC1611DABu, 0xB6662D3Du,
        0x76DC4190u, 0x01DB7106u, 0x98D220BCu, 0xEFD5102Au,
        0x71B18589u, 0x06B6B51Fu, 0x9FBFE4A5u, 0xE8B8D433u,
        0x7807C9A2u, 0x0F00F934u, 0x9609A88Eu, 0xE10E9818u,
        0x7F6396BBu, 0x086D3D2Du, 0x91646C97u, 0xE6635C01u,
        0x6B6B51F4u, 0x1C6C6162u, 0x856530D8u, 0xF262004Eu,
        0x6C0695EDu, 0x1B01A57Bu, 0x8208F4C1u, 0xF50FC457u,
        0x65B0D9C6u, 0x12B7E950u, 0x8BBEB8EAu, 0xFCB9887Cu,
        0x62DD1DDFu, 0x15DA2D49u, 0x8CD37CF3u, 0xFBD44C65u,
        0x4DB26158u, 0x3AB551CEu, 0xA3BC0074u, 0xD4BB30E2u,
        0x4ADFA541u, 0x3DD895D7u, 0xA4D1C46Du, 0xD3D6F4FBu,
        0x4369E96Au, 0x346ED9FCu, 0xAD678846u, 0xDA60B8D0u,
        0x44042D73u, 0x33031DE5u, 0xAA0A4C5Fu, 0xDD0D7CC9u,
        0x5005713Cu, 0x270241AAu, 0xBE0B1010u, 0xC90C2086u,
        0x5768B525u, 0x206F85B3u, 0xB966D409u, 0xCE61E49Fu,
        0x5EDEF90Eu, 0x29D9C998u, 0xB0D09822u, 0xC7D7A8B4u,
        0x59B33D17u, 0x2EB40D81u, 0xB7BD5C3Bu, 0xC0BA6CADu,
        0xEDB88320u, 0x9ABFB3B6u, 0x03B6E20Cu, 0x74B1D29Au,
        0xEAD54739u, 0x9DD277AFu, 0x04DB2615u, 0x73DC1683u,
        0xE3630B12u, 0x94643B84u, 0x0D6D6A3Eu, 0x7A6A5AA8u,
        0xE40ECF0Bu, 0x9309FF9Du, 0x0A00AE27u, 0x7D079EB1u,
        0xF00F9344u, 0x8708A3D2u, 0x1E01F268u, 0x6906C2FEu,
        0xF762575Du, 0x806567CBu, 0x196C3671u, 0x6E6B06E7u,
        0xFED41B76u, 0x89D32BE0u, 0x10DA7A5Au, 0x67DD4ACCu,
        0xF9B9DF6Fu, 0x8EBEEFF9u, 0x17B7BE43u, 0x60B08ED5u,
        0xD6D6A3E8u, 0xA1D1937Eu, 0x38D8C2C4u, 0x4FDFF252u,
        0xD1BB67F1u, 0xA6BC5767u, 0x3FB506DDu, 0x48B2364Bu,
        0xD80D2BDAu, 0xAF0A1B4Cu, 0x36034AF6u, 0x41047A60u,
        0xDF60EFC3u, 0xA8670955u, 0x316658EFu, 0x46616879u,
        0xB40BBE37u, 0xC30C8EA1u, 0x5A05DF1Bu, 0x2D02EF8Du
    };
    uint32_t c = crc ^ 0xFFFFFFFFu;
    for (size_t i = 0u; i < len; i++) {
        c = table[(c ^ data[i]) & 0xFFu] ^ (c >> 8u);
    }
    return c ^ 0xFFFFFFFFu;
}

/* Compute a simple checksum over the first 56 bytes of a command packet,
 * matching M103 §3.1 (checksum covers bytes 0–55). */
static uint32_t cmd_checksum(const att1_aimu_cmd *cmd)
{
    return crc32_update(0u, (const uint8_t *)cmd,
                        offsetof(att1_aimu_cmd, checksum));
}

/* Power-of-two check. */
static int is_power_of_two(size_t n)
{
    return (n >= 1u) && ((n & (n - 1u)) == 0u);
}

/* Ring-buffer used-slot count (command ring). */
static size_t cmdq_used(const att1_aimu_cmdq *q)
{
    return (q->cmd_tail - q->cmd_head) & (q->config.cmd_ring_depth - 1u);
}

/* Ring-buffer used-slot count (completion ring). */
static size_t compq_used(const att1_aimu_cmdq *q)
{
    return (q->comp_tail - q->comp_head) & (q->config.comp_ring_depth - 1u);
}

/* -------------------------------------------------------------------------
 * Lifecycle
 * ---------------------------------------------------------------------- */

att1_status_t att1_aimu_cmdq_create(const att1_aimu_cmdq_config *config,
                                     att1_aimu_cmdq             **out)
{
    att1_aimu_cmdq_config   cfg;
    att1_aimu_cmdq         *q;

    if (out == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }
    *out = NULL;

    /* Apply defaults when config is NULL. */
    if (config == NULL) {
        cfg.tile_count      = 1u;
        cfg.cmd_ring_depth  = ATT1_AIMU_CMDQ_DEFAULT_DEPTH;
        cfg.comp_ring_depth = ATT1_AIMU_COMPQ_DEFAULT_DEPTH;
    } else {
        cfg = *config;
    }

    /* Validate. */
    if (cfg.tile_count == 0u || cfg.tile_count > ATT1_AIMU_CMDQ_MAX_TILES) {
        return ATT1_ERR_INVALID_ARG;
    }
    if (!is_power_of_two(cfg.cmd_ring_depth) ||
        cfg.cmd_ring_depth > ATT1_AIMU_CMDQ_MAX_DEPTH) {
        return ATT1_ERR_INVALID_ARG;
    }
    if (!is_power_of_two(cfg.comp_ring_depth) ||
        cfg.comp_ring_depth > ATT1_AIMU_CMDQ_MAX_DEPTH) {
        return ATT1_ERR_INVALID_ARG;
    }

    q = (att1_aimu_cmdq *)calloc(1u, sizeof(*q));
    if (q == NULL) {
        return ATT1_ERR_OOM;
    }

    q->cmd_ring = (att1_aimu_cmd *)calloc(cfg.cmd_ring_depth,
                                           sizeof(att1_aimu_cmd));
    if (q->cmd_ring == NULL) {
        free(q);
        return ATT1_ERR_OOM;
    }

    q->comp_ring = (att1_aimu_completion *)calloc(cfg.comp_ring_depth,
                                                   sizeof(att1_aimu_completion));
    if (q->comp_ring == NULL) {
        free(q->cmd_ring);
        free(q);
        return ATT1_ERR_OOM;
    }

    q->config           = cfg;
    q->magic            = ATT1_AIMU_CMDQ_MAGIC;
    q->next_command_id  = 1u;
    *out = q;
    return ATT1_OK;
}

void att1_aimu_cmdq_destroy(att1_aimu_cmdq *q)
{
    if (q == NULL) {
        return;
    }
    free(q->cmd_ring);
    free(q->comp_ring);
    memset(q, 0, sizeof(*q));
    free(q);
}

/* -------------------------------------------------------------------------
 * Host-side submission
 * ---------------------------------------------------------------------- */

att1_status_t att1_aimu_cmdq_submit(att1_aimu_cmdq *q,
                                     att1_aimu_cmd  *cmd)
{
    size_t used;
    size_t slot;

    if (q == NULL || cmd == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }
    if ((size_t)cmd->tile_id >= q->config.tile_count) {
        q->counters.commands_failed++;
        return ATT1_ERR_INVALID_ARG;
    }

    /* Check ring capacity: full when (tail+1)%depth == head. */
    used = cmdq_used(q);
    if (used + 1u >= q->config.cmd_ring_depth) {
        q->counters.queue_full_count++;
        return ATT1_ERR_QUEUE_FULL;
    }

    /* Assign command ID and checksum. */
    cmd->command_id = q->next_command_id++;
    cmd->status     = (uint8_t)ATT1_AIMU_PENDING;
    cmd->checksum   = cmd_checksum(cmd);

    slot = q->cmd_tail & (q->config.cmd_ring_depth - 1u);
    q->cmd_ring[slot] = *cmd;
    q->cmd_tail++;

    q->counters.commands_submitted++;
    return ATT1_OK;
}

/* -------------------------------------------------------------------------
 * Simulator dispatch
 * ---------------------------------------------------------------------- */

/* Write a completion record. Returns ATT1_ERR_QUEUE_FULL if comp ring full. */
static att1_status_t write_completion(att1_aimu_cmdq           *q,
                                       const att1_aimu_cmd      *cmd,
                                       att1_aimu_result          result)
{
    att1_aimu_completion comp;
    size_t comp_used;
    size_t slot;

    comp_used = compq_used(q);
    if (comp_used + 1u >= q->config.comp_ring_depth) {
        q->counters.compq_full_count++;
        return ATT1_ERR_QUEUE_FULL;
    }

    memset(&comp, 0, sizeof(comp));
    comp.command_id    = cmd->command_id;
    comp.tile_id       = cmd->tile_id;
    comp.session_id    = cmd->session_id;
    comp.result_code   = (uint8_t)result;
    comp.latency_us    = 0u;

    if (cmd->completion_fence_id != 0u) {
        q->counters.fence_value++;
        comp.fence_value = (uint32_t)q->counters.fence_value;
    }

    slot = q->comp_tail & (q->config.comp_ring_depth - 1u);
    q->comp_ring[slot] = comp;
    q->comp_tail++;

    return ATT1_OK;
}

att1_status_t att1_aimu_cmdq_dispatch_one(att1_aimu_cmdq *q)
{
    att1_aimu_cmd       *cmd;
    att1_aimu_result     result;
    att1_status_t        st;
    size_t               slot;
    att1_aimu_cmd_type   ctype;

    if (q == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }

    if (cmdq_used(q) == 0u) {
        return ATT1_ERR_QUEUE_EMPTY;
    }

    slot = q->cmd_head & (q->config.cmd_ring_depth - 1u);
    cmd  = &q->cmd_ring[slot];
    q->cmd_head++;

    ctype = (att1_aimu_cmd_type)cmd->command_type;

    switch (ctype) {
    case ATT1_AIMU_CMD_NOP:
    case ATT1_AIMU_CMD_TRACE_SNAPSHOT:
    case ATT1_AIMU_CMD_TILE_BARRIER:
    case ATT1_AIMU_CMD_QUERY_COUNTERS:
        result = ATT1_AIMU_OK;
        break;

    case ATT1_AIMU_CMD_RESET_TILE:
        q->counters.resets++;
        result = ATT1_AIMU_OK;
        break;

    case ATT1_AIMU_CMD_LOAD_TENSOR_TILE:
    case ATT1_AIMU_CMD_VALIDATE_TENSOR:
    case ATT1_AIMU_CMD_EXEC_MATMUL:
    case ATT1_AIMU_CMD_EXEC_RMSNORM:
    case ATT1_AIMU_CMD_EXEC_ROPE:
    case ATT1_AIMU_CMD_EXEC_ATTENTION:
    case ATT1_AIMU_CMD_EXEC_FFN:
    case ATT1_AIMU_CMD_KV_APPEND:
    case ATT1_AIMU_CMD_KV_READ:
    case ATT1_AIMU_CMD_FABRIC_SEND:
    case ATT1_AIMU_CMD_FABRIC_REDUCE:
        /* Execution commands not yet implemented in the simulator. */
        q->counters.unsupported_commands++;
        result = ATT1_AIMU_ERR_UNSUPPORTED_OP;
        break;

    default:
        q->counters.unsupported_commands++;
        q->counters.commands_failed++;
        result = ATT1_AIMU_ERR_INVALID_COMMAND;
        break;
    }

    /* Write completion — always produced, even for errors. */
    st = write_completion(q, cmd, result);
    if (st != ATT1_OK) {
        return st;
    }

    if (result == ATT1_AIMU_OK) {
        q->counters.commands_completed++;
    } else if (result != ATT1_AIMU_ERR_UNSUPPORTED_OP) {
        q->counters.commands_failed++;
    }
    /* Unsupported commands count in commands_completed so the host can drain
     * the completion ring; the unsupported_commands counter tracks them
     * separately. */
    if (result == ATT1_AIMU_ERR_UNSUPPORTED_OP) {
        q->counters.commands_completed++;
    }

    return ATT1_OK;
}

att1_status_t att1_aimu_cmdq_dispatch_all(att1_aimu_cmdq *q)
{
    att1_status_t st;

    if (q == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }

    for (;;) {
        st = att1_aimu_cmdq_dispatch_one(q);
        if (st == ATT1_ERR_QUEUE_EMPTY) {
            return ATT1_OK;
        }
        if (st != ATT1_OK) {
            return st;
        }
    }
}

/* -------------------------------------------------------------------------
 * Completion polling
 * ---------------------------------------------------------------------- */

att1_status_t att1_aimu_cmdq_poll_completion(att1_aimu_cmdq        *q,
                                              att1_aimu_completion  *out)
{
    size_t slot;

    if (q == NULL || out == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }

    if (compq_used(q) == 0u) {
        return ATT1_ERR_QUEUE_EMPTY;
    }

    slot = q->comp_head & (q->config.comp_ring_depth - 1u);
    *out = q->comp_ring[slot];
    q->comp_head++;
    return ATT1_OK;
}

/* -------------------------------------------------------------------------
 * Introspection
 * ---------------------------------------------------------------------- */

size_t att1_aimu_cmdq_pending(const att1_aimu_cmdq *q)
{
    if (q == NULL) {
        return 0u;
    }
    return cmdq_used(q);
}

size_t att1_aimu_cmdq_completions_available(const att1_aimu_cmdq *q)
{
    if (q == NULL) {
        return 0u;
    }
    return compq_used(q);
}

att1_status_t att1_aimu_cmdq_get_counters(const att1_aimu_cmdq       *q,
                                           att1_aimu_cmdq_counters    *out)
{
    if (q == NULL || out == NULL) {
        return ATT1_ERR_INVALID_ARG;
    }
    *out = q->counters;
    return ATT1_OK;
}

/* -------------------------------------------------------------------------
 * Name helpers
 * ---------------------------------------------------------------------- */

const char *att1_aimu_cmd_type_name(att1_aimu_cmd_type t)
{
    switch (t) {
    case ATT1_AIMU_CMD_NOP:             return "NOP";
    case ATT1_AIMU_CMD_LOAD_TENSOR_TILE:return "LOAD_TENSOR_TILE";
    case ATT1_AIMU_CMD_VALIDATE_TENSOR: return "VALIDATE_TENSOR_TILE";
    case ATT1_AIMU_CMD_EXEC_MATMUL:     return "EXEC_MATMUL";
    case ATT1_AIMU_CMD_EXEC_RMSNORM:    return "EXEC_RMSNORM";
    case ATT1_AIMU_CMD_EXEC_ROPE:       return "EXEC_ROPE";
    case ATT1_AIMU_CMD_EXEC_ATTENTION:  return "EXEC_ATTENTION";
    case ATT1_AIMU_CMD_EXEC_FFN:        return "EXEC_FFN";
    case ATT1_AIMU_CMD_KV_APPEND:       return "KV_APPEND";
    case ATT1_AIMU_CMD_KV_READ:         return "KV_READ";
    case ATT1_AIMU_CMD_FABRIC_SEND:     return "FABRIC_SEND";
    case ATT1_AIMU_CMD_FABRIC_REDUCE:   return "FABRIC_REDUCE";
    case ATT1_AIMU_CMD_TRACE_SNAPSHOT:  return "TRACE_SNAPSHOT";
    case ATT1_AIMU_CMD_TILE_BARRIER:    return "TILE_BARRIER";
    case ATT1_AIMU_CMD_RESET_TILE:      return "RESET_TILE";
    case ATT1_AIMU_CMD_QUERY_COUNTERS:  return "QUERY_COUNTERS";
    default:                             return "UNKNOWN";
    }
}

const char *att1_aimu_result_name(att1_aimu_result r)
{
    switch (r) {
    case ATT1_AIMU_OK:                      return "OK";
    case ATT1_AIMU_BUSY:                    return "BUSY";
    case ATT1_AIMU_PENDING:                 return "PENDING";
    case ATT1_AIMU_ERR_INVALID_COMMAND:     return "ERR_INVALID_COMMAND";
    case ATT1_AIMU_ERR_INVALID_TENSOR:      return "ERR_INVALID_TENSOR";
    case ATT1_AIMU_ERR_INVALID_DTYPE:       return "ERR_INVALID_DTYPE";
    case ATT1_AIMU_ERR_INVALID_SHAPE:       return "ERR_INVALID_SHAPE";
    case ATT1_AIMU_ERR_INVALID_SESSION:     return "ERR_INVALID_SESSION";
    case ATT1_AIMU_ERR_OUT_OF_MEMORY:       return "ERR_OUT_OF_MEMORY";
    case ATT1_AIMU_ERR_QUEUE_FULL:          return "ERR_QUEUE_FULL";
    case ATT1_AIMU_ERR_KV_OVERFLOW:         return "ERR_KV_OVERFLOW";
    case ATT1_AIMU_ERR_FABRIC:              return "ERR_FABRIC";
    case ATT1_AIMU_ERR_FABRIC_TIMEOUT:      return "ERR_FABRIC_TIMEOUT";
    case ATT1_AIMU_ERR_BARRIER_TIMEOUT:     return "ERR_BARRIER_TIMEOUT";
    case ATT1_AIMU_ERR_CHECKSUM:            return "ERR_CHECKSUM";
    case ATT1_AIMU_ERR_DMA_FAULT:           return "ERR_DMA_FAULT";
    case ATT1_AIMU_ERR_ALIGNMENT:           return "ERR_ALIGNMENT";
    case ATT1_AIMU_ERR_TIMEOUT:             return "ERR_TIMEOUT";
    case ATT1_AIMU_ERR_FENCE_DEADLOCK:      return "ERR_FENCE_DEADLOCK";
    case ATT1_AIMU_ERR_UNSUPPORTED_OP:      return "ERR_UNSUPPORTED_OP";
    case ATT1_AIMU_ERR_UNSUPPORTED_DTYPE:   return "ERR_UNSUPPORTED_DTYPE";
    case ATT1_AIMU_ERR_INTERNAL:            return "ERR_INTERNAL";
    case ATT1_AIMU_ERR_FATAL:               return "ERR_FATAL";
    default:                                 return "UNKNOWN";
    }
}
