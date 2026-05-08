/*
 * att1_aimu_cmdq.h  —  AIMU/PCIe command-queue simulator (M105)
 *
 * In-process ring-buffer simulator for the 64-byte command packet model
 * defined in M103 (docs/aimu_pcie_command_requirements.md) and the MMIO
 * register map defined in M104 (docs/aimu_register_map.md).
 *
 * This is NOT a real PCIe driver or MMIO accessor.  It models the host ↔
 * AIMU command exchange in a single process over a shared-memory ring buffer
 * so that the M103 command protocol can be validated against the ATT-1 C11
 * simulator before hardware is available.
 */

#ifndef ATT1_AIMU_CMDQ_H
#define ATT1_AIMU_CMDQ_H

#include "att1_status.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Constants
 * ---------------------------------------------------------------------- */

/* Maximum number of AIMU tiles per simulated device. */
#define ATT1_AIMU_CMDQ_MAX_TILES    16u

/* Default command ring capacity (slots; must be power of two). */
#define ATT1_AIMU_CMDQ_DEFAULT_DEPTH   256u

/* Maximum command ring capacity. */
#define ATT1_AIMU_CMDQ_MAX_DEPTH    4096u

/* Default completion ring capacity (slots; must be power of two). */
#define ATT1_AIMU_COMPQ_DEFAULT_DEPTH  256u

/* Magic number written by the simulator to verify ring integrity. */
#define ATT1_AIMU_CMDQ_MAGIC  UINT32_C(0xA771C051)

/* -------------------------------------------------------------------------
 * Command type enumeration  (M103 §4)
 * ---------------------------------------------------------------------- */

typedef enum att1_aimu_cmd_type {
    ATT1_AIMU_CMD_NOP               = 0x00,

    /* Tensor management */
    ATT1_AIMU_CMD_LOAD_TENSOR_TILE  = 0x01,
    ATT1_AIMU_CMD_VALIDATE_TENSOR   = 0x02,

    /* Local execution */
    ATT1_AIMU_CMD_EXEC_MATMUL       = 0x10,
    ATT1_AIMU_CMD_EXEC_RMSNORM      = 0x11,
    ATT1_AIMU_CMD_EXEC_ROPE         = 0x12,
    ATT1_AIMU_CMD_EXEC_ATTENTION    = 0x13,
    ATT1_AIMU_CMD_EXEC_FFN          = 0x14,

    /* KV cache */
    ATT1_AIMU_CMD_KV_APPEND         = 0x20,
    ATT1_AIMU_CMD_KV_READ           = 0x21,

    /* Fabric */
    ATT1_AIMU_CMD_FABRIC_SEND       = 0x30,
    ATT1_AIMU_CMD_FABRIC_REDUCE     = 0x31,

    /* Trace / control */
    ATT1_AIMU_CMD_TRACE_SNAPSHOT    = 0x40,
    ATT1_AIMU_CMD_TILE_BARRIER      = 0x41,
    ATT1_AIMU_CMD_RESET_TILE        = 0x50,
    ATT1_AIMU_CMD_QUERY_COUNTERS    = 0x51
} att1_aimu_cmd_type;

/* -------------------------------------------------------------------------
 * Result codes  (M104 §9 / M103 §2.8)
 * ---------------------------------------------------------------------- */

typedef enum att1_aimu_result {
    ATT1_AIMU_OK                    = 0x00,
    ATT1_AIMU_BUSY                  = 0x01,
    ATT1_AIMU_PENDING               = 0x02,

    ATT1_AIMU_ERR_INVALID_COMMAND   = 0x10,
    ATT1_AIMU_ERR_INVALID_TENSOR    = 0x11,
    ATT1_AIMU_ERR_INVALID_DTYPE     = 0x12,
    ATT1_AIMU_ERR_INVALID_SHAPE     = 0x13,
    ATT1_AIMU_ERR_INVALID_SESSION   = 0x14,

    ATT1_AIMU_ERR_OUT_OF_MEMORY     = 0x20,
    ATT1_AIMU_ERR_QUEUE_FULL        = 0x21,
    ATT1_AIMU_ERR_KV_OVERFLOW       = 0x22,

    ATT1_AIMU_ERR_FABRIC            = 0x30,
    ATT1_AIMU_ERR_FABRIC_TIMEOUT    = 0x31,
    ATT1_AIMU_ERR_BARRIER_TIMEOUT   = 0x32,

    ATT1_AIMU_ERR_CHECKSUM          = 0x40,
    ATT1_AIMU_ERR_DMA_FAULT         = 0x41,
    ATT1_AIMU_ERR_ALIGNMENT         = 0x42,

    ATT1_AIMU_ERR_TIMEOUT           = 0x50,
    ATT1_AIMU_ERR_FENCE_DEADLOCK    = 0x51,

    ATT1_AIMU_ERR_UNSUPPORTED_OP    = 0x60,
    ATT1_AIMU_ERR_UNSUPPORTED_DTYPE = 0x61,

    ATT1_AIMU_ERR_INTERNAL          = 0xF0,
    ATT1_AIMU_ERR_FATAL             = 0xFF
} att1_aimu_result;

/* -------------------------------------------------------------------------
 * Command descriptor  (M103 §3 — condensed to C-friendly layout)
 *
 * The on-wire M103 packet is 64 bytes.  This struct mirrors those fields in
 * a C-native layout; the simulator does not perform network byte-order
 * conversion (in-process only).
 * ---------------------------------------------------------------------- */

typedef struct att1_aimu_cmd {
    /* --- 8-byte fields (natural 8-byte alignment, no implicit pads) --- */

    /* Input buffer descriptor (host-physical or AIMU-local address) */
    uint64_t            input_buf_addr;         /* offset  0 */
    /* Output buffer descriptor */
    uint64_t            output_buf_addr;         /* offset  8 */

    /* --- 4-byte fields --- */
    uint32_t            command_id;              /* offset 16 — monotonic host-issued ID */
    uint32_t            input_buf_bytes;         /* offset 20 */
    uint32_t            output_buf_bytes;        /* offset 24 */
    uint32_t            kv_position;             /* offset 28 — KV_APPEND/KV_READ/ATTENTION */
    uint32_t            op_param_0;              /* offset 32 — operation-specific */
    uint32_t            op_param_1;              /* offset 36 */
    uint32_t            checksum;                /* offset 40 — CRC over bytes 0–39 */

    /* --- 2-byte fields --- */
    uint16_t            model_id;                /* offset 44 — logical model identifier */
    uint16_t            tensor_id;               /* offset 46 — tensor slot on the tile */
    uint16_t            fence_id;                /* offset 48 — wait on this fence */
    uint16_t            completion_fence_id;     /* offset 50 — signal when done */
    uint16_t            timeout_ms;              /* offset 52 — 0=no timeout */

    /* --- 1-byte fields --- */
    uint8_t             command_type;            /* offset 54 — att1_aimu_cmd_type */
    uint8_t             tile_id;                 /* offset 55 — 0-based tile index */
    uint8_t             session_id;              /* offset 56 — inference session slot */
    uint8_t             dtype;                   /* offset 57 — 0=f32, 1=q8, 2=q4 */
    uint8_t             trace_flags;             /* offset 58 — bit0=trace_on */
    uint8_t             priority;                /* offset 59 — 0=normal, 1=high */
    uint8_t             status;                  /* offset 60 — att1_aimu_result */
    uint8_t             _pad[3];                 /* offset 61–63 — explicit pad to 64 */
} att1_aimu_cmd;

/* Ensure the command descriptor is exactly 64 bytes. */
typedef char att1_aimu_cmd_size_check[
    (sizeof(att1_aimu_cmd) == 64u) ? 1 : -1];

/* -------------------------------------------------------------------------
 * Completion descriptor  (M103 §7.4 — 16 bytes on wire; extended here)
 * ---------------------------------------------------------------------- */

typedef struct att1_aimu_completion {
    uint32_t    command_id;
    uint8_t     tile_id;
    uint8_t     session_id;
    uint8_t     result_code;       /* att1_aimu_result */
    uint8_t     _pad0;

    uint32_t    latency_us;        /* execution latency (simulated: 0) */
    uint32_t    fence_value;       /* fence signaled by this completion */

    uint32_t    bytes_read;        /* tensor bytes read during EXEC */
    uint32_t    bytes_written;     /* bytes written (KV append / load) */
    uint32_t    packets_sent;      /* fabric packets sent */
    uint32_t    packets_received;  /* fabric packets received */
    uint32_t    trace_event_count; /* trace records emitted */

    uint32_t    _pad1;             /* pad to 40 bytes (multiple of 8) */
} att1_aimu_completion;

/* -------------------------------------------------------------------------
 * Per-device simulator counters
 * ---------------------------------------------------------------------- */

typedef struct att1_aimu_cmdq_counters {
    uint64_t    commands_submitted;
    uint64_t    commands_completed;
    uint64_t    commands_failed;
    uint64_t    queue_full_count;
    uint64_t    compq_full_count;
    uint64_t    resets;
    uint64_t    unsupported_commands;
    uint64_t    fence_value;       /* current completed fence counter */
} att1_aimu_cmdq_counters;

/* -------------------------------------------------------------------------
 * Simulator configuration
 * ---------------------------------------------------------------------- */

typedef struct att1_aimu_cmdq_config {
    size_t      tile_count;        /* number of simulated tiles (1–16) */
    size_t      cmd_ring_depth;    /* command ring slots (power of 2) */
    size_t      comp_ring_depth;   /* completion ring slots (power of 2) */
} att1_aimu_cmdq_config;

/* -------------------------------------------------------------------------
 * Simulator object
 * ---------------------------------------------------------------------- */

typedef struct att1_aimu_cmdq {
    uint32_t                magic;

    att1_aimu_cmdq_config   config;
    att1_aimu_cmdq_counters counters;

    /* Command ring */
    att1_aimu_cmd          *cmd_ring;
    size_t                  cmd_head;   /* next slot AIMU consumes */
    size_t                  cmd_tail;   /* next slot host produces */

    /* Completion ring */
    att1_aimu_completion   *comp_ring;
    size_t                  comp_head;  /* next slot host reads */
    size_t                  comp_tail;  /* next slot AIMU writes */

    /* Next command_id the host will assign */
    uint32_t                next_command_id;
} att1_aimu_cmdq;

/* -------------------------------------------------------------------------
 * Lifecycle
 * ---------------------------------------------------------------------- */

/*
 * att1_aimu_cmdq_create
 *
 * Allocate and initialise an AIMU command-queue simulator.
 * Pass NULL for config to use ATT1_AIMU_CMDQ_DEFAULT_DEPTH and 1 tile.
 *
 * Returns ATT1_OK and writes *out on success.
 * Returns ATT1_ERR_INVALID_ARG if out is NULL or config is invalid.
 * Returns ATT1_ERR_OOM if allocation fails.
 */
att1_status_t att1_aimu_cmdq_create(const att1_aimu_cmdq_config *config,
                                     att1_aimu_cmdq             **out);

/*
 * att1_aimu_cmdq_destroy
 *
 * Free all resources held by the simulator.  Safe to call with NULL.
 */
void att1_aimu_cmdq_destroy(att1_aimu_cmdq *q);

/* -------------------------------------------------------------------------
 * Host-side command submission
 * ---------------------------------------------------------------------- */

/*
 * att1_aimu_cmdq_submit
 *
 * Write a command descriptor to the command ring and advance the tail
 * pointer.  cmd->command_id is assigned automatically (host-monotonic).
 * cmd->checksum is computed and written before the descriptor is stored.
 *
 * Returns ATT1_ERR_INVALID_ARG if q or cmd is NULL.
 * Returns ATT1_ERR_INVALID_ARG if cmd->tile_id >= config.tile_count.
 * Returns ATT1_ERR_QUEUE_FULL if the command ring is full.
 */
att1_status_t att1_aimu_cmdq_submit(att1_aimu_cmdq *q,
                                     att1_aimu_cmd  *cmd);

/* -------------------------------------------------------------------------
 * Simulator-side dispatch (AIMU consumer)
 * ---------------------------------------------------------------------- */

/*
 * att1_aimu_cmdq_dispatch_one
 *
 * Consume the next pending command, simulate its execution, and write a
 * completion descriptor to the completion ring.
 *
 * Simulated behaviour:
 *   ATT1_AIMU_CMD_NOP             → OK
 *   ATT1_AIMU_CMD_RESET_TILE      → OK; counters.resets++
 *   ATT1_AIMU_CMD_QUERY_COUNTERS  → OK
 *   ATT1_AIMU_CMD_TRACE_SNAPSHOT  → OK
 *   ATT1_AIMU_CMD_TILE_BARRIER    → OK
 *   All EXEC_*, KV_*, FABRIC_*, LOAD_*, VALIDATE_*
 *                                 → ATT1_AIMU_ERR_UNSUPPORTED_OP
 *                                   (still produces a completion)
 *
 * Returns ATT1_ERR_QUEUE_EMPTY if no commands are pending.
 * Returns ATT1_ERR_QUEUE_FULL  if the completion ring is full.
 * Returns ATT1_ERR_INVALID_ARG if q is NULL.
 */
att1_status_t att1_aimu_cmdq_dispatch_one(att1_aimu_cmdq *q);

/*
 * att1_aimu_cmdq_dispatch_all
 *
 * Dispatch all pending commands in submission order.
 * Stops and returns the first non-ATT1_OK status (other than QUEUE_EMPTY).
 * Returns ATT1_OK when the command ring is empty.
 */
att1_status_t att1_aimu_cmdq_dispatch_all(att1_aimu_cmdq *q);

/* -------------------------------------------------------------------------
 * Host-side completion polling
 * ---------------------------------------------------------------------- */

/*
 * att1_aimu_cmdq_poll_completion
 *
 * Read and consume the next completion record from the completion ring.
 *
 * Returns ATT1_ERR_QUEUE_EMPTY if no completions are pending.
 * Returns ATT1_ERR_INVALID_ARG if q or out is NULL.
 */
att1_status_t att1_aimu_cmdq_poll_completion(att1_aimu_cmdq        *q,
                                              att1_aimu_completion  *out);

/* -------------------------------------------------------------------------
 * Introspection
 * ---------------------------------------------------------------------- */

/*
 * att1_aimu_cmdq_pending
 *
 * Return the number of command descriptors currently in the ring (submitted
 * but not yet dispatched).
 */
size_t att1_aimu_cmdq_pending(const att1_aimu_cmdq *q);

/*
 * att1_aimu_cmdq_completions_available
 *
 * Return the number of completion records available to the host.
 */
size_t att1_aimu_cmdq_completions_available(const att1_aimu_cmdq *q);

/*
 * att1_aimu_cmdq_counters
 *
 * Copy current device-level counters into *out.
 * Returns ATT1_ERR_INVALID_ARG if q or out is NULL.
 */
att1_status_t att1_aimu_cmdq_get_counters(const att1_aimu_cmdq       *q,
                                           att1_aimu_cmdq_counters    *out);

/* -------------------------------------------------------------------------
 * Name helpers
 * ---------------------------------------------------------------------- */

/*
 * att1_aimu_cmd_type_name
 *
 * Return a stable ASCII name for a command type, e.g. "EXEC_MATMUL".
 * Returns "UNKNOWN" for unrecognised values.
 */
const char *att1_aimu_cmd_type_name(att1_aimu_cmd_type t);

/*
 * att1_aimu_result_name
 *
 * Return a stable ASCII name for a result code, e.g. "ERR_UNSUPPORTED_OP".
 * Returns "UNKNOWN" for unrecognised values.
 */
const char *att1_aimu_result_name(att1_aimu_result r);

#ifdef __cplusplus
}
#endif

#endif /* ATT1_AIMU_CMDQ_H */
